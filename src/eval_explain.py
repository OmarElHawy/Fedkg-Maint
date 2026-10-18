"""
Stage 7: evaluation of LLM explanations (rule-based, no LLM judge).

Turns the raw explanations + the rule-based checks from src/explain.py into
per-version metrics (overall, by true-RUL bucket, quiet vs non-quiet), a
bootstrap 95% confidence interval for the hallucination rate, a paired McNemar
test (and sign test) comparing version A vs version B on the same engines, the
3 worst version-B explanations, and the human-evaluation sheet + answer key.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src import explain

BUCKETS = ["0-20", "20-50", "50-100", "100+"]

_CHECK_COLS = [
    "n_claims", "n_unsupported", "has_hallucination", "number_ok",
    "direction_ok", "n_direction_claims", "citation_valid_rate", "n_citations",
    "insufficient_evidence_handled", "quiet", "unsupported", "reasons",
]


def add_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rul_bucket"] = pd.cut(
        df["true_rul"], bins=[-1, 20, 50, 100, np.inf], labels=BUCKETS
    ).astype(object)
    return df


def merge_checks(expl_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the raw explanations with the rule-based checks and a RUL bucket."""
    checks = explain.check_dataframe(expl_df).reset_index(drop=True)
    merged = expl_df.reset_index(drop=True).copy()
    for c in _CHECK_COLS:
        merged[c] = checks[c].values
    return add_bucket(merged)


def bootstrap_ci(flags, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for a mean (e.g. a hallucination rate)."""
    flags = np.asarray(flags, dtype=float)
    n = len(flags)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([flags[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def _citation_metrics(sub: pd.DataFrame) -> Tuple[float, float]:
    cited = sub[sub["n_citations"] > 0]
    rate = float(cited["citation_valid_rate"].mean()) if len(cited) else float("nan")
    return rate, float((sub["n_citations"] > 0).mean())


def group_metrics(sub: pd.DataFrame) -> dict:
    """All metrics for one group of explanations (one version)."""
    n = len(sub)
    lo, hi = bootstrap_ci(sub["has_hallucination"].values)
    dir_rows = sub[sub["n_direction_claims"] > 0]
    cite_rate, share_cited = _citation_metrics(sub)
    quiet_rows = sub[sub["quiet"]]
    insuff = (quiet_rows["insufficient_evidence_handled"].mean()
              if len(quiet_rows) else float("nan"))
    return {
        "n": n,
        "hallucination_rate": float(sub["has_hallucination"].mean()) if n else float("nan"),
        "hall_ci_lo": lo,
        "hall_ci_hi": hi,
        "unsupported_per_expl": float(sub["n_unsupported"].mean()) if n else float("nan"),
        "number_accuracy": float(sub["number_ok"].mean()) if n else float("nan"),
        "direction_accuracy": float(dir_rows["direction_ok"].mean()) if len(dir_rows) else float("nan"),
        "direction_coverage": float(len(dir_rows) / n) if n else float("nan"),
        "citation_valid_rate": cite_rate,
        "share_with_citations": share_cited,
        "insufficient_evidence_handled_rate": float(insuff),
        "n_quiet": int(len(quiet_rows)),
    }


def summarise(merged: pd.DataFrame) -> pd.DataFrame:
    """Tidy metrics table: per version, overall, by RUL bucket, quiet/non-quiet."""
    rows = []
    for v in ["A", "B"]:
        sub = merged[merged["version"] == v]
        rows.append({"version": v, "group_type": "overall", "group": "all",
                     **group_metrics(sub)})
        for b in BUCKETS:
            s = sub[sub["rul_bucket"] == b]
            if len(s):
                rows.append({"version": v, "group_type": "rul_bucket", "group": b,
                             **group_metrics(s)})
        for q, lbl in [(True, "quiet"), (False, "non-quiet")]:
            s = sub[sub["quiet"] == q]
            if len(s):
                rows.append({"version": v, "group_type": "quietness", "group": lbl,
                             **group_metrics(s)})
    return pd.DataFrame(rows)


def mcnemar_and_sign(merged: pd.DataFrame) -> dict:
    """Paired comparison of A vs B on the same engines (McNemar + sign test)."""
    A = merged[merged["version"] == "A"].set_index("engine")
    B = merged[merged["version"] == "B"].set_index("engine")
    common = sorted(A.index.intersection(B.index))
    a = A.loc[common, "has_hallucination"].astype(bool).to_numpy()
    b = B.loc[common, "has_hallucination"].astype(bool).to_numpy()

    n_ab = int(np.sum(a & ~b))   # A hallucinates, B does not
    n_ba = int(np.sum(~a & b))   # B hallucinates, A does not
    discordant = n_ab + n_ba
    p_mcnemar = (float(binomtest(n_ab, discordant, 0.5).pvalue)
                 if discordant else float("nan"))

    diff = A.loc[common, "n_unsupported"].to_numpy() - B.loc[common, "n_unsupported"].to_numpy()
    nz = diff[diff != 0]
    pos, neg = int(np.sum(nz > 0)), int(np.sum(nz < 0))
    p_sign = (float(binomtest(min(pos, neg), pos + neg, 0.5).pvalue)
              if (pos + neg) else float("nan"))
    return {"n_pairs": len(common), "A_only_halluc": n_ab, "B_only_halluc": n_ba,
            "p_mcnemar": p_mcnemar, "sign_pos": pos, "sign_neg": neg,
            "p_sign": p_sign}


def worst_version_b(merged: pd.DataFrame, k: int = 3) -> List[dict]:
    """The k version-B explanations with the most unsupported claims."""
    b = merged[merged["version"] == "B"].copy()
    b["n_problems"] = b["n_unsupported"] + (~b["number_ok"]).astype(int) + (~b["direction_ok"]).astype(int)
    # most unsupported claims first; if (as here) B is clean, fall back to the
    # explanations with the most entity claims (the closest to risky).
    b = b.sort_values(["n_unsupported", "n_problems", "n_claims"],
                      ascending=False).head(k)
    out = []
    for _, r in b.iterrows():
        out.append({
            "engine": int(r["engine"]), "predicted_rul": float(r["predicted_rul"]),
            "true_rul": float(r["true_rul"]), "n_unsupported": int(r["n_unsupported"]),
            "n_claims": int(r["n_claims"]),
            "unsupported": r["unsupported"], "reasons": list(r["reasons"]),
            "explanation": r["explanation"],
        })
    return out


def make_human_eval(merged: pd.DataFrame, n_engines: int = 10,
                    seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    20 explanations (n_engines from A, n_engines from B) in random order with no
    version label, plus a separate answer key.
    """
    rng = np.random.default_rng(seed)
    engines = merged[merged["version"] == "A"]["engine"].unique()
    pick = rng.choice(engines, size=min(n_engines, len(engines)), replace=False)

    items = []
    for e in pick:
        for v in ["A", "B"]:
            r = merged[(merged["engine"] == e) & (merged["version"] == v)].iloc[0]
            items.append({
                "machine": f"Engine_{int(e)}",
                "predicted_rul": int(round(r["predicted_rul"])),
                "explanation": r["explanation"], "version": v,
                "factory": r["factory"], "engine": int(e),
                "test_index": int(r["index"]),
            })
    order = rng.permutation(len(items))

    sheet, key = [], []
    for i, idx in enumerate(order):
        it = items[idx]
        hid = f"H{i + 1:02d}"
        sheet.append({"id": hid, "machine": it["machine"],
                      "predicted RUL": it["predicted_rul"],
                      "explanation": it["explanation"],
                      "clarity": "", "correctness": "", "useful_action": ""})
        key.append({"id": hid, "version": it["version"], "factory": it["factory"],
                    "engine": it["engine"], "test_index": it["test_index"]})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def _self_tests() -> None:
    # tiny synthetic frame to make sure the pipeline runs end to end
    rows = []
    for eng, true_rul in [(1, 10), (2, 80), (3, 130)]:
        for v in ["A", "B"]:
            rows.append({
                "factory": "FD004", "split": "test", "index": eng, "engine": eng,
                "predicted_rul": 10.0, "risk_level": "HIGH", "true_rul": float(true_rul),
                "version": v,
                "explanation": "Predicted RUL is 10 cycles (risk: HIGH).",
                "ref_facts": ["[1] Predicted RUL is 10 cycles -> risk level HIGH",
                              "[2] No sensor deviates strongly from its healthy level"],
            })
    merged = merge_checks(pd.DataFrame(rows))
    tbl = summarise(merged)
    assert {"version", "group_type", "group", "hallucination_rate"} <= set(tbl.columns)
    assert mcnemar_and_sign(merged)["n_pairs"] == 3
    assert len(worst_version_b(merged, k=2)) == 2
    sheet, key = make_human_eval(merged, n_engines=3, seed=1)
    assert len(sheet) == 6 and len(key) == 6
    print("eval_explain.py: all self-tests passed")


if __name__ == "__main__":
    _self_tests()
