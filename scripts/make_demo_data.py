"""
Stage 10: build the precomputed demo table results/demo/fd004_demo_data.csv.

Reads ONLY saved artifacts:
  - results/explanations_repeat/seed43/sample.csv       (60 FD004 test rows)
  - results/explanations_repeat/seed43/explanations_A.csv / explanations_B.csv

and re-runs the deterministic, rule-based checker (src.explain.check_explanation)
on the saved texts so each row carries the supported/unsupported verdict that
results/final/explanation_stats.csv already reports in the aggregate.

The app (app/streamlit_app.py) needs GPU-free, model-free data: this script is
run ONCE. It intentionally writes results/demo/ and never touches any existing
result file.

An extra consistency gate recomputes the per-window FD004 test predictions from
results/checkpoints/FedProx_4b_seed0.pt (the model that produced the saved
sample) and compares them with sample.csv; a mismatch above tolerance prints a
warning instead of overwriting the saved values.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import explain  # noqa: E402

REPEAT = PROJECT_ROOT / "results" / "explanations_repeat" / "seed43"
OUT_DIR = PROJECT_ROOT / "results" / "demo"
OUT_CSV = OUT_DIR / "fd004_demo_data.csv"
TOLERANCE = 0.05


def _parse_facts(value) -> list:
    if isinstance(value, list):
        return value
    s = str(value).strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]


def _verify_predictions(sample: pd.DataFrame) -> None:
    preds = explain.load_predictions()
    key = preds.set_index(["engine", "index"])
    check = []
    for _, r in sample.iterrows():
        row = key.loc[(int(r["engine"]), int(r["index"]))]
        check.append(abs(float(row["predicted_rul"]) - float(r["predicted_rul"])))
    diff = float(max(check))
    status = "OK" if diff <= TOLERANCE else "WARNING"
    print(f"prediction gate {status}: max |recomputed - saved| = {diff:.6f} "
          f"(tolerance {TOLERANCE}); source checkpoint = "
          f"results/checkpoints/FedProx_4b_seed0.pt")
    if status == "WARNING":
        print("saved sample values are kept as-is (they are what the "
              "explanations were generated from).")


def main() -> None:
    sample = pd.read_csv(REPEAT / "sample.csv")
    a = pd.read_csv(REPEAT / "explanations_A.csv")
    b = pd.read_csv(REPEAT / "explanations_B.csv")
    a["_facts"] = a["ref_facts"].map(_parse_facts)
    b["_facts"] = b["ref_facts"].map(_parse_facts)

    rows = []
    for _, s in sample.iterrows():
        eng = int(s["engine"])
        ra = a[a["engine"] == eng].iloc[0]
        rb = b[b["engine"] == eng].iloc[0]

        ca = explain.check_explanation(ra["explanation"], ra["_facts"],
                                       float(ra["predicted_rul"]),
                                       ra["risk_level"], "A")
        cb = explain.check_explanation(rb["explanation"], rb["_facts"],
                                       float(rb["predicted_rul"]),
                                       rb["risk_level"], "B")

        def verdict(c: dict) -> str:
            return "supported" if not c["has_hallucination"] else "unsupported"

        rows.append({
            "engine": eng,
            "window_index": int(s["index"]),
            "predicted_rul": float(round(float(s["predicted_rul"]), 2)),
            "true_rul": float(s["true_rul"]),
            "risk_level": str(s["risk_level"]),
            "kg_facts": "\n".join(ra["_facts"]),
            "explanation_A": ra["explanation"],
            "explanation_B": rb["explanation"],
            "check_A": verdict(ca),
            "check_B": verdict(cb),
            "n_claims_A": int(ca["n_claims"]),
            "n_unsupported_claims_A": int(ca["n_unsupported"]),
            "n_claims_B": int(cb["n_claims"]),
            "n_unsupported_claims_B": int(cb["n_unsupported"]),
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(df)} rows)")

    _verify_predictions(sample)


if __name__ == "__main__":
    main()