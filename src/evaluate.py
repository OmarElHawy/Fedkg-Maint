"""
Stage 9 evaluation: read-only rebuild of final tables and figures.

Reads only existing results (baselines.csv, federated_final.csv,
privacy_results.csv, stage9_*.csv/jsonl, azure_results.csv,
azure_fl_histories_*.jsonl, azure_run_meta.json, azure_ablation_*.csv,
explanations/metrics.csv, explanations_repeat/*, checkpoints) and writes NEW
files under results/final/. It never modifies an existing result file and
never trains any model. The only inference it performs is scoring the held-out
FD004 test set with the SAVED checkpoints for the engine-level statistics.

Usage:
    python src/evaluate.py            # rebuild everything, print summary
    python src/evaluate.py --only cmapss
"""
# --------------------------------------------------------------------------- #
# imports / constants
# --------------------------------------------------------------------------- #

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "results"
FINAL = RESULTS / "final"
CHECKPOINTS = RESULTS / "checkpoints"
if not CHECKPOINTS.exists() and (PROJECT_ROOT.parent / "results" / "checkpoints").exists():
    CHECKPOINTS = PROJECT_ROOT.parent / "results" / "checkpoints"
FINAL.mkdir(parents=True, exist_ok=True)

CLIP_MAX = 125.0
CMAPSS_SETUPS = ["Local FD001", "Local FD002", "Local FD003", "Centralized",
                 "FedAvg", "FedProx", "DP 0.03", "DP 0.1"]
FACTORIES = ["FD001", "FD002", "FD003", "FD004"]
AZURE_SETUPS = ["LocalOnly", "Centralized", "FedAvg", "FedProx"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _mns(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce")
    if len(s) == 0 or s.isna().all():
        return "nan"
    return f"{s.mean():.3f}\u00b1{s.std(ddof=1):.3f}"


def _boot_ci(d, n=5000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _holm(pv):
    pv = np.asarray(pv, dtype=float)
    order = np.argsort(pv)
    m = len(pv)
    adj = np.ones_like(pv)
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, max(running, pv[i] * (m - rank)))
        adj[i] = v
        running = max(running, v)
    return adj


# --------------------------------------------------------------------------- #
# C-MAPSS trials frame
# --------------------------------------------------------------------------- #

def cmapss_trials() -> pd.DataFrame:
    """One row per (setup, seed, test_factory): rmse, nasa, final_val_rmse."""
    bas = pd.read_csv(RESULTS / "baselines.csv")
    new = pd.read_csv(RESULTS / "stage9_cmapss_runs.csv")
    fed = pd.read_csv(RESULTS / "federated_final.csv")
    priv = pd.read_csv(RESULTS / "privacy_results.csv")
    priv9 = pd.read_csv(RESULTS / "stage9_privacy_rows.csv")

    rows = []
    for _, r in bas.iterrows():
        rows.append({"setup": r["setup"], "seed": int(r["seed"]),
                     "test_factory": r["test_factory"], "rmse": r["rmse"],
                     "nasa_score": r["nasa_score"],
                     "final_val_rmse": np.nan, "source": "baselines"})
    for _, r in new[new["source"] == "new"].iterrows():
        setup = r["setup"]
        if setup.startswith("Local"):
            s = setup
        elif setup == "Centralized":
            s = "Centralized"
        elif setup == "FedAvg":
            s = "FedAvg"
        elif setup == "FedProx":
            if pd.isna(r["dp_sigma"]):
                s = "FedProx"
            else:
                continue  # DP seeds 3-4 are sourced from stage9_privacy_rows.csv
        else:
            continue
        rows.append({"setup": s, "seed": int(r["seed"]),
                     "test_factory": r["test_factory"], "rmse": r["rmse"],
                     "nasa_score": r["nasa_score"],
                     "final_val_rmse": r["final_val_rmse"], "source": "stage9-new"})
    for _, r in fed.iterrows():
        rows.append({"setup": r["setup"], "seed": int(r["seed"]),
                     "test_factory": r["test_factory"], "rmse": r["rmse"],
                     "nasa_score": r["nasa_score"],
                     "final_val_rmse": np.nan, "source": "stage5"})
    for _, r in pd.concat([priv, priv9]).iterrows():
        if r["label"] == "DP" and float(r["sigma"]) in (0.03, 0.1):
            s = "DP 0.03" if float(r["sigma"]) == 0.03 else "DP 0.1"
            rows.append({"setup": s, "seed": int(r["seed"]),
                         "test_factory": r["test_factory"], "rmse": r["rmse"],
                         "nasa_score": r["nasa_score"],
                         "final_val_rmse": r["final_val_rmse"], "source": "privacy"})
    return pd.DataFrame(rows)


def table_cmapss_main(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setup in CMAPSS_SETUPS:
        s = trials[(trials["setup"] == setup) & (trials["test_factory"] == "FD004")]
        if len(s) == 0:
            continue
        n = len(s)
        rows.append({
            "setup": setup, "n_seeds": n,
            "fd004_rmse_mean": s["rmse"].mean(), "fd004_rmse_std": s["rmse"].std(ddof=1),
            "fd004_nasa_mean": s["nasa_score"].mean(),
            "fd004_nasa_std": s["nasa_score"].std(ddof=1),
            "final_val_rmse_mean": s["final_val_rmse"].dropna().mean() if s["final_val_rmse"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def table_cmapss_all(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setup in CMAPSS_SETUPS:
        for f in FACTORIES:
            s = trials[(trials["setup"] == setup) & (trials["test_factory"] == f)]
            rows.append({"setup": setup, "test_factory": f,
                         "rmse_mean": s["rmse"].mean() if len(s) else np.nan,
                         "rmse_std": s["rmse"].std(ddof=1) if len(s) > 1 else np.nan,
                         "n_seeds": len(s)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# DP / privacy summary
# --------------------------------------------------------------------------- #

def table_dp(trials: pd.DataFrame) -> pd.DataFrame:
    from src.privacy import compute_epsilon
    rows = []
    for setup in ["FedProx", "DP 0.03", "DP 0.1"]:
        s = trials[(trials["setup"] == setup) & (trials["test_factory"] == "FD004")]
        sigma = 0.0 if setup == "FedProx" else (0.03 if setup == "DP 0.03" else 0.1)
        eps = float("inf") if sigma == 0 else compute_epsilon(sigma, 30, 1e-5)
        rows.append({"setup": setup, "sigma": sigma, "epsilon": eps,
                     "fd004_rmse_mean": s["rmse"].mean(),
                     "fd004_rmse_std": s["rmse"].std(ddof=1), "n_seeds": len(s)})
    return pd.DataFrame(rows)


def table_system_ablation() -> pd.DataFrame:
    """System ablation table: remove one component at a time from Full system.
    
    Columns: system, fd004_rmse_mean, fd004_rmse_std, fd004_nasa_mean, fd004_nasa_std,
             epsilon, hallucination_rate, note
    All values read from existing result files."""
    cmapss = pd.read_csv(FINAL / "cmapss_main.csv")
    dp = pd.read_csv(FINAL / "dp_overview.csv")
    expl = pd.read_csv(FINAL / "explanations_overall.csv")
    
    # Full system (FedProx + DP 0.03 + KG + LLM)
    full = cmapss[cmapss["setup"] == "DP 0.03"].iloc[0]
    full_eps = dp[dp["setup"] == "DP 0.03"]["epsilon"].iloc[0]
    hall_b = expl[expl["version"] == "B"]["hallucination_rate"].mean()
    
    # No DP (FedProx without DP)
    nodp = cmapss[cmapss["setup"] == "FedProx"].iloc[0]
    nodp_eps = float("inf")
    
    # No KG (same prediction as Full system, hallucination A)
    nolog = full.copy()
    hall_a = expl[expl["version"] == "A"]["hallucination_rate"].mean()
    
    # No LLM (same prediction as Full system, no explanation metrics)
    nollm = full.copy()
    
    # No FL (Centralized)
    cent = cmapss[cmapss["setup"] == "Centralized"].iloc[0]
    cent_eps = float("inf")
    
    # Local-only FD001, FD002, FD003
    loc1 = cmapss[cmapss["setup"] == "Local FD001"].iloc[0]
    loc2 = cmapss[cmapss["setup"] == "Local FD002"].iloc[0]
    loc3 = cmapss[cmapss["setup"] == "Local FD003"].iloc[0]
    loc_eps = "n/a"
    
    note_base = (
        "Explanation experiments used the FedProx seed-0 model without DP; "
        "hallucination rate is the same for Full system and No DP. "
        "Removing the KG or the LLM does not change the prediction metrics, only the explanations."
    )
    
    rows = [
        {
            "system": "Full system (FedProx + DP 0.03 + KG + LLM)",
            "fd004_rmse_mean": full["fd004_rmse_mean"],
            "fd004_rmse_std": full["fd004_rmse_std"],
            "fd004_nasa_mean": full["fd004_nasa_mean"],
            "fd004_nasa_std": full["fd004_nasa_std"],
            "epsilon": full_eps,
            "hallucination_rate": hall_b,
            "note": note_base,
        },
        {
            "system": "No DP (FedProx, no DP)",
            "fd004_rmse_mean": nodp["fd004_rmse_mean"],
            "fd004_rmse_std": nodp["fd004_rmse_std"],
            "fd004_nasa_mean": nodp["fd004_nasa_mean"],
            "fd004_nasa_std": nodp["fd004_nasa_std"],
            "epsilon": nodp_eps,
            "hallucination_rate": hall_b,
            "note": note_base,
        },
        {
            "system": "No KG (FedProx + DP 0.03, no knowledge graph)",
            "fd004_rmse_mean": full["fd004_rmse_mean"],
            "fd004_rmse_std": full["fd004_rmse_std"],
            "fd004_nasa_mean": full["fd004_nasa_mean"],
            "fd004_nasa_std": full["fd004_nasa_std"],
            "epsilon": full_eps,
            "hallucination_rate": hall_a,
            "note": note_base,
        },
        {
            "system": "No LLM (FedProx + DP 0.03 + KG, rule-based only)",
            "fd004_rmse_mean": full["fd004_rmse_mean"],
            "fd004_rmse_std": full["fd004_rmse_std"],
            "fd004_nasa_mean": full["fd004_nasa_mean"],
            "fd004_nasa_std": full["fd004_nasa_std"],
            "epsilon": full_eps,
            "hallucination_rate": float("nan"),
            "note": note_base,
        },
        {
            "system": "No FL (Centralized)",
            "fd004_rmse_mean": cent["fd004_rmse_mean"],
            "fd004_rmse_std": cent["fd004_rmse_std"],
            "fd004_nasa_mean": cent["fd004_nasa_mean"],
            "fd004_nasa_std": cent["fd004_nasa_std"],
            "epsilon": cent_eps,
            "hallucination_rate": float("nan"),
            "note": note_base,
        },
        {
            "system": "Local-only FD001",
            "fd004_rmse_mean": loc1["fd004_rmse_mean"],
            "fd004_rmse_std": loc1["fd004_rmse_std"],
            "fd004_nasa_mean": loc1["fd004_nasa_mean"],
            "fd004_nasa_std": loc1["fd004_nasa_std"],
            "epsilon": loc_eps,
            "hallucination_rate": float("nan"),
            "note": note_base,
        },
        {
            "system": "Local-only FD002",
            "fd004_rmse_mean": loc2["fd004_rmse_mean"],
            "fd004_rmse_std": loc2["fd004_rmse_std"],
            "fd004_nasa_mean": loc2["fd004_nasa_mean"],
            "fd004_nasa_std": loc2["fd004_nasa_std"],
            "epsilon": loc_eps,
            "hallucination_rate": float("nan"),
            "note": note_base,
        },
        {
            "system": "Local-only FD003",
            "fd004_rmse_mean": loc3["fd004_rmse_mean"],
            "fd004_rmse_std": loc3["fd004_rmse_std"],
            "fd004_nasa_mean": loc3["fd004_nasa_mean"],
            "fd004_nasa_std": loc3["fd004_nasa_std"],
            "epsilon": loc_eps,
            "hallucination_rate": float("nan"),
            "note": note_base,
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# convergence / rounds-within-5%
# --------------------------------------------------------------------------- #

def _curve_records() -> pd.DataFrame:
    recs = []
    for line in (RESULTS / "stage9_fl_curves.jsonl").open(encoding="utf-8"):
        recs.append(json.loads(line))
    df = pd.DataFrame(recs)
    df["dp_sigma"] = df["dp_sigma"].astype(float)
    df["dp_sigma"] = df["dp_sigma"].where(df["dp_sigma"].isin([0.0, 0.03, 0.1]), np.nan)
    df["mu"] = df["mu"].astype(float)
    return df


def _converged_round(vals: np.ndarray, factor: float = 1.05) -> float:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan
    final = vals[-1]
    idx = np.where(vals <= factor * final)[0]
    return float(idx[0]) if len(idx) else float(len(vals))


def table_rounds_within5() -> pd.DataFrame:
    curves = _curve_records()
    tune = pd.read_csv(RESULTS / "federated_tuning.csv")
    rows = []

    def add(meth, mu, sigma, seed, vals):
        rows.append({"method": meth, "mu": mu, "dp_sigma": sigma, "seed": seed,
                     "round_to_5pct": _converged_round(vals)})

    for seed in [0, 1, 2, 3, 4]:
        for sig in [0.03, 0.1]:
            sub = curves[(curves["method"] == "fedprox") & (curves["dp_sigma"] == sig)
                         & (curves["seed"] == seed)]
            for _, r in sub.iterrows():
                add("dp", 0.01, sig, seed, np.asarray(r["val_rmse"]))

    for seed in [3, 4]:
        for meth, mu in [("fedavg", 0.0), ("fedprox", 0.01)]:
            sub = curves[(curves["method"] == meth) & (curves["dp_sigma"].isna())
                         & (curves["seed"] == seed)]
            for _, r in sub.iterrows():
                add(meth, mu, np.nan, seed, np.asarray(r["val_rmse"]))

    for meth, cfg in [("fedavg", "FedAvg"), ("fedprox", "FedProx(mu=0.01)")]:
        t = tune[(tune["config"] == cfg) & (tune["seed"] == 0) & (tune["round"] <= 30)]
        vals = t.sort_values("round")["val_rmse"].to_numpy()
        add(meth, tune[tune["config"] == cfg]["mu"].iloc[0], np.nan, 0, vals)

    df = pd.DataFrame(rows)
    out = []
    for (meth, mu, sig), g in df.groupby(["method", "mu", "dp_sigma"], dropna=False):
        out.append({"method": meth, "mu": mu, "dp_sigma": sig,
                    "n_seeds": len(g), "round_to_5pct_mean": g["round_to_5pct"].mean(),
                    "round_to_5pct_std": g["round_to_5pct"].std(ddof=1)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# number-of-factories + heterogeneity
# --------------------------------------------------------------------------- #

def table_num_factories() -> pd.DataFrame:
    nf = pd.read_csv(RESULTS / "stage9_num_factories.csv")
    het = pd.read_csv(RESULTS / "stage9_heterogeneity.csv")
    het_map = dict(zip(het["set_name"], het["hetero"]))
    nf = nf.copy()
    nf["hetero"] = nf["set_name"].map(het_map)
    rows = []
    for (sn, meth), g in nf.groupby(["set_name", "method"]):
        rows.append({"set_name": sn, "factories": g["factories"].iloc[0],
                     "n_clients": g["n_clients"].iloc[0], "method": meth,
                     "hetero": g["hetero"].iloc[0],
                     "fd004_rmse_mean": g["fd004_rmse"].mean(),
                     "fd004_rmse_std": g["fd004_rmse"].std(ddof=1),
                     "n_seeds": len(g)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# azure
# --------------------------------------------------------------------------- #

def table_azure() -> pd.DataFrame:
    az = pd.read_csv(RESULTS / "azure_results.csv")
    rows = []
    for setting in ["next24h", "gap24_48h"]:
        for setup in AZURE_SETUPS:
            for ev in ["heldout_test", "heldout_all"]:
                s = az[(az["setting"] == setting) & (az["setup"] == setup)
                       & (az["eval_set"] == ev)]
                if len(s) == 0:
                    continue
                rows.append({"setting": setting, "setup": setup, "eval_set": ev,
                             "reps": len(s),
                             "pr_auc_mean": s["pr_auc"].mean(),
                             "pr_auc_std": s["pr_auc"].std(ddof=1),
                             "f1_mean": s["f1"].mean(), "f1_std": s["f1"].std(ddof=1),
                             "prevalence_mean": s["prevalence"].mean()})
    return pd.DataFrame(rows)


def table_azure_stats() -> pd.DataFrame:
    az = pd.read_csv(RESULTS / "azure_results.csv")
    methods = ["Centralized", "FedAvg", "FedProx"]
    settings = ["next24h", "gap24_48h"]
    comparisons = [
        ("FedAvg", "LocalOnly", "FedAvg vs LocalOnly"),
        ("FedProx", "LocalOnly", "FedProx vs LocalOnly"),
        ("Centralized", "LocalOnly", "Centralized vs LocalOnly"),
        ("FedProx", "FedAvg", "FedProx vs FedAvg"),
        ("FedProx", "Centralized", "FedProx vs Centralized"),
        ("FedAvg", "Centralized", "FedAvg vs Centralized"),
    ]
    rows = []
    for setting in settings:
        sub = az[(az["setting"] == setting) & (az["eval_set"] == "heldout_test")]
        pivot_fl = sub[sub["setup"].isin(methods)].pivot_table(
            index="seed", columns="setup", values="pr_auc", aggfunc="mean").reset_index()
        lo = sub[sub["setup"] == "LocalOnly"]["pr_auc"].values
        lo_mean = float(lo.mean())
        for m_a, m_b, label in comparisons:
            if m_a == "LocalOnly" or m_b == "LocalOnly":
                if m_b == "LocalOnly":
                    vals_a = pivot_fl[m_a].values if m_a in pivot_fl.columns else None
                    vals_b = np.array([lo_mean] * 5)
                else:
                    vals_a = np.array([lo_mean] * 5)
                    vals_b = pivot_fl[m_b].values if m_b in pivot_fl.columns else None
                if vals_a is None or vals_b is None:
                    continue
            else:
                if m_a not in pivot_fl.columns or m_b not in pivot_fl.columns:
                    continue
                vals_a = pivot_fl[m_a].values
                vals_b = pivot_fl[m_b].values

            diffs = vals_a - vals_b
            mean_diff = float(np.mean(diffs))
            t, p = stats.ttest_rel(vals_a, vals_b)
            pooled_std = float(np.std(diffs, ddof=1))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("nan")
            ci_lo, ci_hi = _boot_ci(diffs, n=2000, seed=42)
            rows.append({
                "setting": setting,
                "comparison": label,
                "mean_diff": round(mean_diff, 6),
                "t_stat": round(float(t), 4),
                "p_value": round(float(p), 6),
                "cohens_d": round(cohens_d, 4),
                "ci_95_lo": round(ci_lo, 6),
                "ci_95_hi": round(ci_hi, 6),
            })
    df = pd.DataFrame(rows)
    for setting in settings:
        m = df["setting"] == setting
        df.loc[m, "holm_p"] = np.round(_holm(df.loc[m, "p_value"].to_numpy()), 6)
    return df


def table_azure_kg_sizes_and_edges():
    kg_dir = RESULTS / "kg"
    sizes_rows = []
    edges_rows = []
    factories = ["model1", "model2", "model3", "model4"]
    for m in factories:
        pkl_path = kg_dir / f"azure_{m}.pkl"
        if not pkl_path.exists():
            continue
        import pickle
        with open(pkl_path, "rb") as fp:
            G = pickle.load(fp)
        ec_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "ErrorCode"]
        fm_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "FailureMode"]
        sym_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Symptom"]
        comp_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Component"]
        mach_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Machine"]
        prec_edges = [(u, v, d) for u, v, k, d in G.edges(data=True, keys=True) if d.get("relation") == "precedes"]

        sizes_rows.append({
            "factory": m,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "n_error_codes": len(ec_nodes),
            "n_failure_modes": len(fm_nodes),
            "n_symptoms": len(sym_nodes),
            "n_components": len(comp_nodes),
            "n_machines": len(mach_nodes),
            "n_precedes_edges": len(prec_edges),
        })

        for u, v, d in sorted(prec_edges, key=lambda x: (x[0], x[1])):
            edges_rows.append({
                "factory": m,
                "error_code": u,
                "failure_mode": v,
                "lift": round(float(d.get("lift", 0.0)), 2),
                "supporting_events": int(d.get("supporting_events", 0)),
                "total_failures": int(d.get("total_failures", 0)),
            })
    return pd.DataFrame(sizes_rows), pd.DataFrame(edges_rows)


def table_azure_kg_sanity() -> pd.DataFrame:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "azure"
    tel_path = raw_dir / "PdM_telemetry.csv"
    if not tel_path.exists():
        exist = FINAL / "azure_kg_sanity.csv"
        return pd.read_csv(exist) if exist.exists() else pd.DataFrame()

    import json, pickle
    from src.kg_azure import FACTORIES, SENSORS
    tel_all = pd.read_csv(tel_path, parse_dates=["datetime"]).sort_values(["machineID", "datetime"])
    fail_all = pd.read_csv(raw_dir / "PdM_failures.csv", parse_dates=["datetime"]).sort_values(["machineID", "datetime"])
    err_all = pd.read_csv(raw_dir / "PdM_errors.csv", parse_dates=["datetime"]).sort_values(["machineID", "datetime"])
    mach_all = pd.read_csv(raw_dir / "PdM_machines.csv")

    tel_all[["r_volt", "r_rotate", "r_pressure", "r_vibration"]] = (
        tel_all.groupby("machineID")[SENSORS].rolling(3, min_periods=1).mean().values
    )
    err_counts = err_all.groupby(["machineID", "datetime"]).size().rename("err_cnt").reset_index()
    tel_all = pd.merge(tel_all, err_counts, on=["machineID", "datetime"], how="left")
    tel_all["err_cnt"] = tel_all["err_cnt"].fillna(0)
    tel_all["err_24h"] = tel_all.groupby("machineID")["err_cnt"].rolling(25, min_periods=1).sum().values

    tel_test = tel_all[tel_all["datetime"] >= "2015-11-01"].sort_values("datetime")
    fail_sorted = fail_all.sort_values("datetime")

    fail_right = fail_sorted[["machineID", "datetime"]].copy()
    fail_right["next_fail_dt"] = fail_right["datetime"]

    m_next = pd.merge_asof(
        tel_test,
        fail_right[["machineID", "datetime", "next_fail_dt"]],
        on="datetime",
        by="machineID",
        direction="forward",
    )
    m_next["dt_to_next"] = (m_next["next_fail_dt"] - m_next["datetime"]).dt.total_seconds()

    is_prefail = (m_next["dt_to_next"] >= 0) & (m_next["dt_to_next"] <= 24 * 3600)
    is_healthy = m_next["dt_to_next"].isna() | (m_next["dt_to_next"] > 72 * 3600)
    m_next["category"] = np.where(is_prefail, "prefail", np.where(is_healthy, "healthy", "mid"))

    mach_map = dict(zip(mach_all["machineID"], mach_all["model"]))
    m_next["model"] = m_next["machineID"].map(mach_map)

    sanity_rows = []
    kg_dir = RESULTS / "kg"
    for fac in FACTORIES:
        pkl_path = kg_dir / f"azure_{fac}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as fp:
            G = pickle.load(fp)
        h_means = json.loads(G.graph["healthy_means"])
        h_stds = json.loads(G.graph["healthy_stds"])

        sub = m_next[m_next["model"] == fac].copy()
        s_active = pd.Series(False, index=sub.index)
        for s in SENSORS:
            z = (sub["r_" + s] - h_means[s]) / max(h_stds[s], 1e-6)
            s_active = s_active | (z.abs() >= 2.0)

        e_active = sub["err_24h"] > 0
        sub["has_active"] = s_active | e_active

        pf_share = float(sub[sub["category"] == "prefail"]["has_active"].mean())
        h_share = float(sub[sub["category"] == "healthy"]["has_active"].mean())
        n_pf = int((sub["category"] == "prefail").sum())
        n_h = int((sub["category"] == "healthy").sum())
        ratio = pf_share / h_share if h_share > 0 else float("nan")

        sanity_rows.append({
            "factory": fac,
            "prefail_active_share": round(pf_share, 4),
            "prefail_n": n_pf,
            "healthy_active_share": round(h_share, 4),
            "healthy_n": n_h,
            "ratio": round(ratio, 2),
        })
    return pd.DataFrame(sanity_rows)


def azure_curves():
    chosen = {}
    meta = json.loads((RESULTS / "azure_run_meta.json").read_text(encoding="utf-8"))
    for k, v in meta.items():
        if "chosen_mu" in v:
            chosen[k] = v["chosen_mu"]
    out = {}
    for setting in ["next24h", "gap24_48h"]:
        rows = [json.loads(l) for l in
                (RESULTS / f"azure_fl_histories_{setting}.jsonl").open(encoding="utf-8")]
        df = pd.DataFrame(rows)
        keep = []
        for _, r in df.iterrows():
            if r["setup"] == "FedAvg" and abs(float(r["mu"]) - 0.0) < 1e-12:
                keep.append(r)
            elif r["setup"] == "FedProx" and abs(float(r["mu"]) - chosen[setting]) < 1e-12:
                keep.append(r)
        cur = pd.DataFrame(keep)
        arr = {}
        for _, r in cur.iterrows():
            arr[(r["setup"], int(r["seed"]))] = (np.asarray(r["round"]),
                                                 np.asarray(r["weighted_val_pr_auc"]))
        med = {}
        for setup in ["FedAvg", "FedProx"]:
            series = {k: v[1] for k, v in arr.items() if k[0] == setup}
            if not series:
                continue
            lens = {len(v) for v in series.values()}
            n = min(lens) if lens else 0
            rr = np.arange(n)
            m = np.vstack([v[:n] for v in series.values()])
            med[setup] = (rr, np.median(m, axis=0), m.min(axis=0), m.max(axis=0))
        out[setting] = med
    return out


def table_azure_curves():
    med = azure_curves()
    rows = []
    for setting, d in med.items():
        for setup, (rr, md, lo, hi) in d.items():
            rows.append({"setting": setting, "setup": setup,
                         "n_seeds": 1 if len(d) == 0 else 5,
                         "round": int(rr[-1]),
                         "final_pr_auc_median": float(md[-1]),
                         "final_pr_auc_min": float(lo[-1]),
                         "final_pr_auc_max": float(hi[-1])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# explanations
# --------------------------------------------------------------------------- #

def table_explanations() -> pd.DataFrame:
    om = pd.read_csv(RESULTS / "explanations_repeat" / "overall_metrics.csv")
    rows = []
    for _, r in om.iterrows():
        if r["group_type"] != "overall":
            continue
        rows.append({"sample": r["sample"], "version": r["version"],
                     "n": r["n"], "hallucination_rate": r["hallucination_rate"],
                     "hall_ci_lo": r["hall_ci_lo"], "hall_ci_hi": r["hall_ci_hi"]})
    df = pd.DataFrame(rows)
    ex = []
    for seed in [43, 44]:
        m = json.loads((RESULTS / "explanations_repeat" / f"seed{seed}"
                        / "mcnemar.json").read_text(encoding="utf-8"))
        m["sample"] = seed
        ex.append(m)
    return df, pd.DataFrame(ex)


# --------------------------------------------------------------------------- #
# engine-level statistics (scoring saved checkpoints only)
# --------------------------------------------------------------------------- #

def _load_model(name: str) -> torch.nn.Module:
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models import LSTMRegressor
    obj = torch.load(CHECKPOINTS / name, map_location=DEVICE, weights_only=False)
    m = LSTMRegressor(input_dim=14, hidden_size=64, num_layers=2, dropout=0.2)
    m.load_state_dict(obj["model_state"])
    m.to(DEVICE).eval()
    return m


def _fd004_abs_errors(setup: str, seeds, X, y) -> np.ndarray:
    names = {"Centralized": lambda s: f"Centralized_seed{s}.pt",
             "FedAvg": lambda s: f"FedAvg_4b_seed{s}.pt",
             "FedProx": lambda s: f"FedProx_4b_seed{s}.pt",
             "DP 0.03": lambda s: f"DP_DP_sigma0.03_seed{s}.pt",
             "DP 0.1": lambda s: f"DP_DP_sigma0.1_seed{s}.pt"}
    out = []
    for s in seeds:
        m = _load_model(names[setup](s))
        with torch.no_grad():
            p = m(torch.tensor(X, device=DEVICE, dtype=torch.float32)).squeeze(-1).cpu().numpy()
        cycles = p * CLIP_MAX
        y_clipped = np.minimum(y * 1.0, CLIP_MAX)
        out.append(np.abs(cycles - y_clipped))
    return np.vstack(out)  # (n_seeds, 248)


def table_engine_stats(seed_trials: pd.DataFrame) -> pd.DataFrame:
    cand_paths = [
        PROJECT_ROOT / "data" / "processed" / "FD004" / "arrays.npz",
        PROJECT_ROOT.parent / "data" / "processed" / "FD004" / "arrays.npz",
        Path(os.environ.get("FEDKG_DATA_DIR", "")) / "processed" / "FD004" / "arrays.npz",
    ]
    npz_path = next((p for p in cand_paths if p.exists()), None)
    can_compute = (npz_path is not None and CHECKPOINTS.exists() and (CHECKPOINTS / "Centralized_seed0.pt").exists())
    if can_compute:
        fd = np.load(npz_path)
        X, y = fd["X_test"], fd["y_test"]
        setups = ["Centralized", "FedAvg", "FedProx", "DP 0.03", "DP 0.1"]
        abs_by = {}
        for st in setups:
            abs_by[st] = _fd004_abs_errors(st, [0, 1, 2, 3, 4], X, y)
        pairs = [("FedProx", "FedAvg"), ("FedProx", "Centralized"),
                 ("FedAvg", "Centralized"), ("DP 0.03", "FedProx"),
                 ("DP 0.1", "FedProx")]
        rows = []
        for a, b in pairs:
            ma = abs_by[a].mean(axis=0)
            mb = abs_by[b].mean(axis=0)
            d = ma - mb
            if np.all(d == 0):
                w, p = np.nan, np.nan
            else:
                w, p = stats.wilcoxon(ma, mb, zero_method="wilcox")
            lo, hi = _boot_ci(d)
            rows.append({"compare": f"{a} vs {b}", "units": "cycles (abs err/engine)",
                         "mean_a": ma.mean(), "mean_b": mb.mean(),
                         "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": p, "n_engines": len(ma)})
        return pd.DataFrame(rows)
    elif (FINAL / "stats_cmapss_engine.csv").exists():
        return pd.read_csv(FINAL / "stats_cmapss_engine.csv")
    elif (PROJECT_ROOT.parent / "results" / "final" / "stats_cmapss_engine.csv").exists():
        return pd.read_csv(PROJECT_ROOT.parent / "results" / "final" / "stats_cmapss_engine.csv")
    else:
        raise FileNotFoundError("arrays.npz or checkpoints not found")


# --------------------------------------------------------------------------- #
# paired stats over 5 seeds
# --------------------------------------------------------------------------- #

def table_seed_stats(trials: pd.DataFrame) -> pd.DataFrame:
    f = trials[(trials["test_factory"] == "FD004")].copy()
    f["seed"] = f["seed"].astype(int)
    piv = f.pivot_table(index="seed", columns="setup", values="rmse")
    pairs = [("FedProx", "FedAvg"), ("FedProx", "Centralized"),
             ("FedAvg", "Centralized"), ("DP 0.03", "FedProx"),
             ("DP 0.1", "FedProx")]
    rows = []
    for a, b in pairs:
        if a not in piv.columns or b not in piv.columns:
            continue
        av = piv[a].to_numpy()
        bv = piv[b].to_numpy()
        d = av - bv
        t, p = stats.ttest_rel(av, bv)
        dd = d.std(ddof=1)
        cohen = float(d.mean() / dd) if dd > 0 else float("nan")
        lo, hi = _boot_ci(d)
        rows.append({"compare": f"{a} - {b}", "mean_a": av.mean(), "mean_b": bv.mean(),
                     "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "t": t, "p_value": p, "cohens_d": cohen, "n_pairs": len(d)})
    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = _holm(df["p_value"].to_numpy())
    return df


# --------------------------------------------------------------------------- #
# comm/cost
# --------------------------------------------------------------------------- #

def table_comm() -> pd.DataFrame:
    cmapss_params = 53825
    cmapss_mb = cmapss_params * 4 * 2 / 1e6  # fp32 weights + running doubles
    azmeta = json.loads((RESULTS / "azure_run_meta.json").read_text(encoding="utf-8"))["comm_cost"]
    return pd.DataFrame([
        {"task": "C-MAPSS (LSTM)", "n_params": cmapss_params,
         "mb_per_client_round": cmapss_mb,
         "total_mb_30_rounds": cmapss_mb * 30},
        {"task": "Azure (MLP)", "n_params": azmeta["n_params"],
         "mb_per_client_round": azmeta["mb_per_round"],
         "total_mb_30_rounds": azmeta["total_mb"]},
    ])


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figure_cmapss(tbl: pd.DataFrame):
    setups = list(tbl["setup"])
    mean = tbl["fd004_rmse_mean"].to_numpy()
    sd = tbl["fd004_rmse_std"].fillna(0).to_numpy()
    colors = {"Local FD001": "#4C72B0", "Local FD002": "#4C72B0", "Local FD003": "#4C72B0",
              "Centralized": "#DD8452", "FedAvg": "#55A868", "FedProx": "#C44E52",
              "DP 0.03": "#8172B3", "DP 0.1": "#937860"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(setups))
    ax.bar(x, mean, yerr=sd, width=0.6, capsize=4,
           color=[colors.get(s_, "#999") for s_ in setups], alpha=0.85)
    ax.set_xticks(x, ["\n".join(s.split()) for s in setups], rotation=0, fontsize=9)
    ax.set_ylabel("FD004 test RMSE (cycles)")
    ax.set_title("C-MAPSS: FD004 test RMSE over 5 seeds (\u00b1 s.d.)")
    for xi, m_, s_ in zip(x, mean, sd):
        ax.text(xi, m_ + s_ + 0.3, f"{m_:.2f}", ha="center", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "cmapss_fd004_rmse.png", dpi=150)
    plt.close(fig)


def figure_dp(tbl: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    s = tbl.copy()
    s["eps"] = s["epsilon"].replace(float("inf"), 1e6)
    ax1.semilogx(s["eps"], s["fd004_rmse_mean"], "o-", color="#C44E52")
    ax1.errorbar(s["eps"], s["fd004_rmse_mean"], yerr=s["fd004_rmse_std"],
                 fmt="none", ecolor="#C44E52", capsize=3, alpha=0.6)
    ax1.set_ylabel("FD004 test RMSE")
    ax1.set_xlabel(r"$\epsilon$ (30 rounds, $\delta=10^{-5}$)")
    ax1.set_title("Privacy-utility trade-off")
    ax1.grid(alpha=0.3)
    names = list(tbl["setup"])
    ax2.bar(np.arange(len(names)), tbl["fd004_rmse_mean"],
            yerr=tbl["fd004_rmse_std"].fillna(0), capsize=4, alpha=0.8, color="#8172B3")
    ax2.set_xticks(np.arange(len(names)), names, rotation=15, fontsize=8)
    ax2.set_ylabel("FD004 test RMSE")
    ax2.set_title("NoDP vs DP FedProx")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "dp_epsilon_tradeoff.png", dpi=150)
    plt.close(fig)


def figure_num_factories(tbl: pd.DataFrame):
    order = [f"size1_{f}" for f in ["FD001", "FD002", "FD003"]] + \
            ["size2_FD001+FD002", "size2_FD001+FD003", "size2_FD002+FD003",
             "size3_FD001+FD002+FD003"]
    tbl = tbl.set_index("set_name").loc[[s for s in order if s in tbl.set_index("set_name").index]].reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.34
    for j, meth in enumerate(["FedAvg", "FedProx"]):
        sub = tbl[tbl["method"] == meth]
        xx = np.arange(len(sub)) + (j - 0.5) * w
        ax.bar(xx, sub["fd004_rmse_mean"], w, yerr=sub["fd004_rmse_std"].fillna(0),
               capsize=3, label=meth, alpha=0.85)
    locs = np.arange(len(sub))
    ax.set_xticks(locs, sub["set_name"].str.replace("size1_", "1:").str.replace("size2_", "2:")
                  .str.replace("size3_", "3:"), rotation=20, fontsize=8.5)
    ax.set_ylabel("FD004 test RMSE")
    ax.set_title("Effect of the number of factories and heterogeneity")
    ax.legend(title="method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "num_factories.png", dpi=150)
    plt.close(fig)


def figure_hetero(tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for meth in ["FedAvg", "FedProx"]:
        sub = tbl[tbl["method"] == meth]
        ax.plot(sub["hetero"], sub["fd004_rmse_mean"], "o-", label=meth)
        ax.fill_between(sub["hetero"],
                        sub["fd004_rmse_mean"] - sub["fd004_rmse_std"],
                        sub["fd004_rmse_mean"] + sub["fd004_rmse_std"], alpha=0.15)
    ax.set_xlabel("heterogeneity (mean pairwise 1-D Wasserstein, last step)")
    ax.set_ylabel("FD004 test RMSE")
    ax.set_title("FD004 RMSE vs client-set heterogeneity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "heterogeneity_vs_rmse.png", dpi=150)
    plt.close(fig)


def figure_azure(tbl: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, setting in zip(axes, ["next24h", "gap24_48h"]):
        sub = tbl[(tbl["setting"] == setting) & (tbl["eval_set"] == "heldout_test")]
        names = list(sub["setup"])
        ax.bar(np.arange(len(names)), sub["pr_auc_mean"],
               yerr=sub["pr_auc_std"].fillna(0), capsize=4, alpha=0.85, color="#55A868")
        ax.axhline(0.5, color="#888", ls="--", lw=1)
        ax.set_xticks(np.arange(len(names)), names, rotation=10, fontsize=8)
        ax.set_title(f"Azure {setting} (held-out test)")
        ax.set_ylabel("PR-AUC")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "azure_main.png", dpi=150)
    plt.close(fig)


def figure_azure_convergence():
    med = azure_curves()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, setting in zip(axes, ["next24h", "gap24_48h"]):
        c = med[setting]
        for color, setup in [("#55A868", "FedAvg"), ("#C44E52", "FedProx")]:
            if setup not in c:
                continue
            rr, md, lo, hi = c[setup]
            ax.plot(rr, md, color=color, label=setup)
            ax.fill_between(rr, lo, hi, color=color, alpha=0.18)
        ax.set_title(f"Azure {setting} (seeds 0-4)")
        ax.set_xlabel("round")
        ax.set_ylabel("weighted val PR-AUC")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FINAL / "azure_convergence.png", dpi=150)
    plt.close(fig)


def figure_explanations(tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tbl))
    for j, v in enumerate(["A", "B"]):
        sub = tbl[tbl["version"] == v]
        xx = sub["sample"].astype(str) + "·" + v
        idx = [tbl.index[i] for i in range(len(tbl)) if tbl["version"].iloc[i] == v]
        ax.bar(idx, sub["hallucination_rate"], 0.35,
               yerr=[sub["hallucination_rate"] - sub["hall_ci_lo"],
                     sub["hall_ci_hi"] - sub["hallucination_rate"]],
               capsize=4, label=f"version {v}", alpha=0.85)
    ax.set_xticks(np.arange(len(tbl)), [f"{r['sample']}\n{r['version']}" for _, r in tbl.iterrows()],
                  fontsize=8)
    ax.set_ylabel("hallucination rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"LLM explanation hallucination (seeds 42, 43, 44; n=60 each)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FINAL / "explanation_hallucination.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run(only=None):
    if only and only not in ("cmapss", "all"):
        raise SystemExit("--only must be cmapss or all (default all)")

    trials = cmapss_trials()

    summary = []
    summary.append("=" * 64)
    summary.append("Stage 9 final evaluation summary")
    summary.append("=" * 64)

    if only == "cmapss" or only is None:
        tb_main = table_cmapss_main(trials)
        tb_main.to_csv(FINAL / "cmapss_main.csv", index=False)
        tb_all = table_cmapss_all(trials)
        tb_all.to_csv(FINAL / "cmapss_all_factories.csv", index=False)
        tb_dp = table_dp(trials)
        tb_dp.to_csv(FINAL / "dp_overview.csv", index=False)
        tb_r5 = table_rounds_within5()
        tb_r5.to_csv(FINAL / "rounds_within_5pct.csv", index=False)
        tb_nf = table_num_factories()
        tb_nf.to_csv(FINAL / "num_factories_summary.csv", index=False)
        tb_sc = table_seed_stats(trials)
        tb_sc.to_csv(FINAL / "stats_cmapss.csv", index=False)
        tb_eg = table_engine_stats(trials)
        tb_eg.to_csv(FINAL / "stats_cmapss_engine.csv", index=False)

        figure_cmapss(tb_main)
        figure_dp(tb_dp)
        figure_num_factories(tb_nf)
        figure_hetero(tb_nf)

        summary.append("-- C-MAPSS FD004 RMSE (5 seeds) --")
        for _, r in tb_main.iterrows():
            summary.append(f"  {r['setup']:<13s} {r['fd004_rmse_mean']:.3f} ± "
                           f"{r['fd004_rmse_std']:.3f} (n={int(r['n_seeds'])})")
        summary.append("-- DP / epsilon --")
        for _, r in tb_dp.iterrows():
            eps = "inf" if r["epsilon"] == float("inf") else f"{r['epsilon']:.2f}"
            summary.append(f"  {r['setup']:<9s} eps={eps:<10s} "
                           f"{r['fd004_rmse_mean']:.3f} ± {r['fd004_rmse_std']:.3f}")
        summary.append("-- paired tests over seeds (Holm-corrected) --")
        for _, r in tb_sc.iterrows():
            summary.append(f"  {r['compare']:<28s} diff={r['mean_diff']:+.3f} "
                           f"p={r['p_value']:.4f} (holm {r['holm_p']:.4f}) d={r['cohens_d']:+.2f}")
        summary.append("-- engine-level (FD004, 248 engines) --")
        for _, r in tb_eg.iterrows():
            summary.append(f"  {r['compare']:<28s} diff={r['mean_diff']:+.3f} "
                           f"CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}] p={r['wilcoxon_p']:.4g}")

    if only is None:
        tb_az = table_azure()
        tb_az.to_csv(FINAL / "azure_main.csv", index=False)
        tb_az_stat = table_azure_stats()
        tb_az_stat.to_csv(FINAL / "stats_azure.csv", index=False)
        tb_az_sizes, tb_az_edges = table_azure_kg_sizes_and_edges()
        if len(tb_az_sizes):
            tb_az_sizes.to_csv(FINAL / "azure_kg_sizes.csv", index=False)
        if len(tb_az_edges):
            tb_az_edges.to_csv(FINAL / "azure_kg_edges.csv", index=False)
        tb_az_sanity = table_azure_kg_sanity()
        if len(tb_az_sanity):
            tb_az_sanity.to_csv(FINAL / "azure_kg_sanity.csv", index=False)
        tb_azc = table_azure_curves()
        tb_azc.to_csv(FINAL / "azure_convergence_summary.csv", index=False)
        tb_ex, ex_stat = table_explanations()
        tb_ex.to_csv(FINAL / "explanations_overall.csv", index=False)
        ex_stat.to_csv(FINAL / "explanation_stats.csv", index=False)
        tb_abl = table_system_ablation()
        tb_abl.to_csv(FINAL / "system_ablation.csv", index=False)
        tb_comm = table_comm()
        tb_comm.to_csv(FINAL / "comm_cost.csv", index=False)

        ab = {"next24h": pd.read_csv(RESULTS / "azure_ablation_next24h.csv"),
              "gap24_48h": pd.read_csv(RESULTS / "azure_ablation_gap24_48h.csv")}
        pd.concat(ab.values(), ignore_index=True).to_csv(
            FINAL / "azure_feature_ablation.csv", index=False)

        (FINAL / "regression.json").write_text(
            (RESULTS / "stage9_regression.json").read_text(encoding="utf-8"), encoding="utf-8")
        (FINAL / "task0.csv").write_text(
            (RESULTS / "stage9_task0.csv").read_text(encoding="utf-8"), encoding="utf-8")

        figure_azure(tb_az)
        figure_azure_convergence()
        figure_explanations(tb_ex)

        summary.append("-- Azure held-out test PR-AUC (seeds 0-4) --")
        for _, r in tb_az[tb_az["eval_set"] == "heldout_test"].iterrows():
            summary.append(f"  {r['setting']:<10s} {r['setup']:<11s} "
                           f"{r['pr_auc_mean']:.3f} ± {r['pr_auc_std']:.3f}")
        summary.append("-- explanations (A vs B hallucination) --")
        for _, r in tb_ex.iterrows():
            summary.append(f"  sample {int(r['sample'])} {r['version']}: "
                           f"{r['hallucination_rate']:.3f} "
                           f"[{r['hall_ci_lo']:.3f}, {r['hall_ci_hi']:.3f}]")
        summary.append("-- communication cost (30 rounds) --")
        for _, r in tb_comm.iterrows():
            summary.append(f"  {r['task']:<18s} {r['total_mb_30_rounds']:.4f} MB total "
                           f"({r['mb_per_client_round']:.4f} MB/client/round)")

    summary.append("=" * 64)
    txt = "\n".join(summary) + "\n"
    (FINAL / "run_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=["cmapss", "all"])
    args = ap.parse_args()
    run(args.only)