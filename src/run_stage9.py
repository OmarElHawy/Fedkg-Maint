"""
Stage 9: heavy runner for the remaining evaluation work.

Adds ONLY new runs and writes ONLY new output files (one append is allowed:
results/azure_results.csv gets seeds 3 and 4 appended; seed 0-2 rows are never
touched). No existing training code or existing results are modified.

Sections (can be run one at a time with --only):
  regression    - reproduce FedProx(mu=0.01) seed 0, round-30 weighted
                  client-validation RMSE == 16.5021 (gating check)
  task0         - independent raw-row count of model1 vs model4 train positives
  task1_cmapss  - C-MAPSS seeds 3,4: Local FD001-3, Centralized, FedAvg,
                  FedProx(=NoDP),  DP sigma 0.03/0.1   (+ save checkpoints)
  backfill      - evaluate existing Local/Centralized seeds 0-2 checkpoints on
                  the test factories they are missing (read-only eval)
  task2         - number-of-factories: 3 pairs x {FedAvg, FedProx} x seeds 0-2,
                  evaluated on FD004 only (size1/size3 reused from baselines)
  task3         - per-client-set heterogeneity (mean pairwise 1-D Wasserstein
                  over the 14 sensors, last time step)
  dp_curves     - per-round client-validation curves for DP sigma 0.03/0.1 seeds
                  0,1,2 (Stage 5 stored only the final validation value)
  azure         - Azure seeds 3,4: LocalOnly, Centralized, FedAvg, FedProx
                  (chosen mu per setting), appended to azure_results.csv
  explanations  - repeat of the Stage 7 FD004 explanation evaluation on fresh
                  60-engine samples (selection seeds 43, 44); skipped with a
                  note if Ollama is unreachable

Usage:
    python src/run_stage9.py               # run every section
    python src/run_stage9.py --only task1_cmapss
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated import run_federated, evaluate_on_factory
from src.privacy import compute_epsilon
from src.train_local import train_model
from src.utils import set_seed, check_gpu

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MAX = 125
BATCH_SIZE = 256
LR = 1e-3
LOCAL_EPOCHS = 2
ROUNDS = 30
MU = 0.01
CLIENTS = ["FD001", "FD002", "FD003"]
TEST_FACTORIES = CLIENTS + ["FD004"]
NEW_SEEDS = [3, 4]
DP_SIGMAS = [0.03, 0.1]

START = time.time()


def log(msg: str) -> None:
    print(f"[stage9 {time.time() - START:8.1f}s] {msg}", flush=True)


def load_factory_data(name: str) -> dict:
    data = np.load(PROJECT_ROOT / "data" / "processed" / name / "arrays.npz")
    return {k: data[k] for k in data.files}


def cmapss_data() -> dict:
    factories = {f: load_factory_data(f) for f in ["FD001", "FD002", "FD003", "FD004"]}
    client_data = {f: {"X_train": factories[f]["X_train"],
                       "y_train": factories[f]["y_train"] / CLIP_MAX} for f in CLIENTS}
    client_val_data = {f: {"X_val": factories[f]["X_val"],
                           "y_val": factories[f]["y_val"]} for f in CLIENTS}
    assert "FD004" not in client_data
    return factories, client_data, client_val_data


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_privacy_C() -> float:
    p = PROJECT_ROOT / "configs" / "privacy_config.yaml"
    if p.exists():
        return float(yaml.safe_load(p.read_text(encoding="utf-8"))["clip_norm_C"])
    return 0.32


# --------------------------------------------------------------------------- #
# regression check
# --------------------------------------------------------------------------- #

EXPECTED_VAL30 = 16.502083720307283


def run_regression() -> bool:
    out_json = RESULTS_DIR / "stage9_regression.json"
    factories, client_data, client_val_data = cmapss_data()
    t0 = time.time()
    _, hist, _ = run_federated(
        "fedprox", client_data, held_out_X=None,
        n_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, mu=MU, lr=LR,
        batch_size=BATCH_SIZE, seed=0, device=device(),
        client_val_data=client_val_data,
        eval_every=ROUNDS, log_every=ROUNDS, verbose=True)
    val30 = float(hist["val_rmse"][-1])
    dt = time.time() - t0
    ok = abs(val30 - EXPECTED_VAL30) < 1e-3
    rec = {"expected": EXPECTED_VAL30, "reproduced": val30,
           "abs_diff": abs(val30 - EXPECTED_VAL30), "passed": ok,
           "runtime_s": dt}
    out_json.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    log(f"regression: reproduced={val30!r} expected={EXPECTED_VAL30!r} "
        f"diff={abs(val30 - EXPECTED_VAL30):.2e} passed={ok} [{dt:.0f}s]")
    return ok


# --------------------------------------------------------------------------- #
# task1: C-MAPSS seeds 3,4
# --------------------------------------------------------------------------- #

CMAPSS_COLS = ["setup", "train_factory", "method", "mu", "dp_sigma", "seed",
               "test_factory", "rmse", "nasa_score", "rmse_unclipped",
               "final_val_rmse", "epochs_trained", "rounds", "runtime_s", "source"]
CMAPSS_CSV = RESULTS_DIR / "stage9_cmapss_runs.csv"
CURVES_JSONL = RESULTS_DIR / "stage9_fl_curves.jsonl"


def _read_cmapss_csv() -> pd.DataFrame:
    if CMAPSS_CSV.exists():
        return pd.read_csv(CMAPSS_CSV)
    return pd.DataFrame(columns=CMAPSS_COLS)


def _append_rows(rows: list) -> None:
    df = _read_cmapss_csv()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CMAPSS_CSV, index=False)


def _done_fl(method, mu, dp_sigma, seed) -> bool:
    df = _read_cmapss_csv()
    if df.empty:
        return False
    m = (df["seed"] == seed) & (df["source"] == "new")
    if dp_sigma is None:
        m &= df["mu"].apply(lambda x: float(x) == float(mu))
    else:
        m &= df["dp_sigma"].apply(lambda x: float(x) == float(dp_sigma))
    return m.sum() == len(TEST_FACTORIES)


def _eval_model_rows(model, setup: str, train_factory: str, method: str,
                     mu, dp_sigma, seed, factories, final_val_rmse,
                     epochs_trained, rounds, runtime_s, source="new") -> list:
    rows = []
    for tf in TEST_FACTORIES:
        mm = evaluate_on_factory(model, factories[tf]["X_test"],
                                 factories[tf]["y_test"], device(), CLIP_MAX)
        rows.append({"setup": setup, "train_factory": train_factory,
                     "method": method, "mu": mu, "dp_sigma": dp_sigma,
                     "seed": seed, "test_factory": tf, "rmse": mm["rmse"],
                     "nasa_score": mm["nasa_score"],
                     "rmse_unclipped": mm["rmse_unclipped"],
                     "final_val_rmse": final_val_rmse,
                     "epochs_trained": epochs_trained, "rounds": rounds,
                     "runtime_s": runtime_s, "source": source})
    return rows


def _save_curve(rec: dict) -> None:
    with open(CURVES_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_local_cmapss(factories, seed: int) -> None:
    dev = device()
    for f in CLIENTS:
        setup = f"Local {f}"
        df = _read_cmapss_csv()
        if not df.empty and ((df["setup"] == setup) & (df["seed"] == seed)
                             & (df["source"] == "new")).sum() == len(TEST_FACTORIES):
            log(f"skip Local {f} seed {seed} (already done)")
            continue
        set_seed(seed)
        tds = torch.utils.data.TensorDataset(
            torch.FloatTensor(factories[f]["X_train"]),
            torch.FloatTensor(factories[f]["y_train"] / CLIP_MAX))
        vds = torch.utils.data.TensorDataset(
            torch.FloatTensor(factories[f]["X_val"]),
            torch.FloatTensor(factories[f]["y_val"] / CLIP_MAX))
        train_loader = torch.utils.data.DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False)
        t0 = time.time()
        model, history = train_model(
            train_loader, val_loader, input_dim=14, hidden_size=64,
            num_layers=2, dropout=0.2, lr=LR, max_epochs=60, patience=8,
            seed=seed, device=dev, clip_max=CLIP_MAX,
            checkpoint_path=str(CHECKPOINTS_DIR / f"Local_{f}_seed{seed}.pt"))
        dt = time.time() - t0
        rows = _eval_model_rows(model, setup, f, "local", np.nan, np.nan,
                                seed, factories,
                                final_val_rmse=float(min(history["val_rmse"])),
                                epochs_trained=len(history["train_loss"]),
                                rounds=np.nan, runtime_s=dt)
        _append_rows(rows)
        fd = [r["rmse"] for r in rows if r["test_factory"] == "FD004"][0]
        log(f"Local {f} seed {seed} done  FD004 RMSE={fd:.3f} [{dt:.0f}s]")


def run_centralized_cmapss(factories, seed: int) -> None:
    setup = "Centralized"
    df = _read_cmapss_csv()
    if not df.empty and ((df["setup"] == setup) & (df["seed"] == seed)
                         & (df["source"] == "new")).sum() == len(TEST_FACTORIES):
        log(f"skip Centralized seed {seed} (already done)")
        return
    Xtr = np.concatenate([factories[f]["X_train"] for f in CLIENTS], axis=0)
    ytr = np.concatenate([factories[f]["y_train"] for f in CLIENTS], axis=0) / CLIP_MAX
    Xva = np.concatenate([factories[f]["X_val"] for f in CLIENTS], axis=0)
    yva = np.concatenate([factories[f]["y_val"] for f in CLIENTS], axis=0) / CLIP_MAX
    set_seed(seed)
    tds = torch.utils.data.TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    vds = torch.utils.data.TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva))
    train_loader = torch.utils.data.DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False)
    t0 = time.time()
    model, history = train_model(
        train_loader, val_loader, input_dim=14, hidden_size=64,
        num_layers=2, dropout=0.2, lr=LR, max_epochs=60, patience=8,
        seed=seed, device=device(), clip_max=CLIP_MAX,
        checkpoint_path=str(CHECKPOINTS_DIR / f"Centralized_seed{seed}.pt"))
    dt = time.time() - t0
    rows = _eval_model_rows(model, setup, "+".join(CLIENTS), "centralized",
                            np.nan, np.nan, seed, factories,
                            final_val_rmse=float(min(history["val_rmse"])),
                            epochs_trained=len(history["train_loss"]),
                            rounds=np.nan, runtime_s=dt)
    _append_rows(rows)
    fd = [r["rmse"] for r in rows if r["test_factory"] == "FD004"][0]
    log(f"Centralized seed {seed} done  FD004 RMSE={fd:.3f} [{dt:.0f}s]")


def run_fl_cmapss(client_data, client_val_data, factories, method, mu, dp_sigma,
                  seed: int) -> None:
    if dp_sigma is None:
        key_mu = mu
    else:
        key_mu = None
    if _done_fl(method, key_mu, dp_sigma, seed):
        log(f"skip {method} mu={mu:g} dp={dp_sigma} seed {seed} (already done)")
        return
    dp = None
    C = None
    if dp_sigma is not None:
        C = load_privacy_C()
        dp = {"C": C, "sigma": float(dp_sigma)}
    t0 = time.time()
    model, hist, _ = run_federated(
        method, client_data, n_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
        mu=mu, lr=LR, batch_size=BATCH_SIZE, seed=seed, device=device(),
        client_val_data=client_val_data, val_every=1, dp=dp, verbose=False)
    dt = time.time() - t0
    final_val = float(hist["val_rmse"][-1])

    if method == "fedprox":
        setup = "FedProx" if mu > 0 else "FedAvg"
    else:
        setup = "FedAvg"
    rows = _eval_model_rows(model, setup, "+".join(CLIENTS), method, mu,
                            dp_sigma, seed, factories, final_val, np.nan,
                            ROUNDS, dt)
    _append_rows(rows)

    ckpt = {"model_state": model.state_dict(), "history": hist,
            "mu": mu, "rounds": ROUNDS, "seed": seed}
    if dp_sigma is None:
        name = f"{setup}_4b_seed{seed}.pt"
    else:
        name = f"DP_DP_sigma{dp_sigma:g}_seed{seed}.pt"
    torch.save(ckpt, CHECKPOINTS_DIR / name)

    _save_curve({"method": method, "mu": mu, "dp_sigma": dp_sigma, "seed": seed,
                 "round": hist["round"], "val_rmse": hist["val_rmse"],
                 "val_round": hist["val_round"],
                 "comm_cost_bytes": hist["comm_cost_bytes"]})

    if dp_sigma is not None:
        eps = compute_epsilon(float(dp_sigma), ROUNDS, 1e-5)
        prows = []
        for tf in TEST_FACTORIES:
            mm = evaluate_on_factory(model, factories[tf]["X_test"],
                                     factories[tf]["y_test"], device(), CLIP_MAX)
            prows.append({"label": "DP", "sigma": float(dp_sigma), "epsilon": eps,
                          "seed": seed, "test_factory": tf, "rmse": mm["rmse"],
                          "nasa_score": mm["nasa_score"], "final_val_rmse": final_val,
                          "runtime_s": dt})
        pfile = RESULTS_DIR / "stage9_privacy_rows.csv"
        pold = pd.read_csv(pfile) if pfile.exists() else pd.DataFrame()
        pd.concat([pold, pd.DataFrame(prows)], ignore_index=True).to_csv(pfile, index=False)
    fd = [r["rmse"] for r in rows if r["test_factory"] == "FD004"][0]
    log(f"{setup} mu={mu:g} dp={dp_sigma} seed {seed} done  "
        f"FD004 RMSE={fd:.3f} val={final_val:.3f} [{dt:.0f}s]")


def task1_cmapss() -> None:
    factories, client_data, client_val_data = cmapss_data()
    for seed in NEW_SEEDS:
        run_local_cmapss(factories, seed)
    for seed in NEW_SEEDS:
        run_centralized_cmapss(factories, seed)
    for method, mu in [("fedavg", 0.0), ("fedprox", MU)]:
        for seed in NEW_SEEDS:
            run_fl_cmapss(client_data, client_val_data, factories, method, mu,
                          None, seed)
    for sigma in DP_SIGMAS:
        for seed in NEW_SEEDS:
            run_fl_cmapss(client_data, client_val_data, factories, "fedprox",
                          MU, sigma, seed)
    log("task1_cmapss complete")


# --------------------------------------------------------------------------- #
# Local/Centralized seeds 0-2 backfill on missing factories
# --------------------------------------------------------------------------- #

def task_backfill() -> None:
    factories, _, _ = cmapss_data()
    df_existing = pd.DataFrame(columns=["setup", "seed", "test_factory"])
    for p in [RESULTS_DIR / "baselines.csv", CMAPSS_CSV]:
        if p.exists():
            d = pd.read_csv(p)
            df_existing = pd.concat([df_existing, d[["setup", "seed", "test_factory"]]])
    keys = set(map(tuple, df_existing[["setup", "seed", "test_factory"]].values))

    setup_specs = [("Local FD001", "Local_FD001", "FD001"),
                   ("Local FD002", "Local_FD002", "FD002"),
                   ("Local FD003", "Local_FD003", "FD003"),
                   ("Centralized", "Centralized", "+".join(CLIENTS)),
                   ("FD004-local", "FD004-local", "FD004")]
    dev = device()
    for setup, ckname, train_f in setup_specs:
        for seed in [0, 1, 2]:
            ckpt = CHECKPOINTS_DIR / f"{ckname}_seed{seed}.pt"
            if not ckpt.exists():
                log(f"backfill: missing checkpoint {ckpt.name}")
                continue
            missing = [tf for tf in TEST_FACTORIES
                       if (setup, seed, tf) not in keys]
            if not missing:
                continue
            obj = torch.load(ckpt, map_location=dev, weights_only=False)
            from src.models import LSTMRegressor
            m = LSTMRegressor(input_dim=14, hidden_size=64, num_layers=2, dropout=0.2)
            m.load_state_dict(obj["model_state"])
            m.to(dev)
            hist = obj.get("history", {})
            rows = []
            for tf in missing:
                mm = evaluate_on_factory(m, factories[tf]["X_test"],
                                         factories[tf]["y_test"], dev, CLIP_MAX)
                rows.append({"setup": setup, "train_factory": train_f,
                             "method": "local" if setup != "Centralized" else "centralized",
                             "mu": np.nan, "dp_sigma": np.nan, "seed": seed,
                             "test_factory": tf, "rmse": mm["rmse"],
                             "nasa_score": mm["nasa_score"],
                             "rmse_unclipped": mm["rmse_unclipped"],
                             "final_val_rmse": float(min(hist["val_rmse"])) if hist.get("val_rmse") else np.nan,
                             "epochs_trained": len(hist.get("train_loss", [])),
                             "rounds": np.nan, "runtime_s": np.nan, "source": "backfill"})
            _append_rows(rows)
            for r in rows:
                keys.add((setup, seed, r["test_factory"]))
            log(f"backfill: {setup} seed {seed} -> {missing}")
    log("task_backfill complete")


# --------------------------------------------------------------------------- #
# task2: number of factories
# --------------------------------------------------------------------------- #

NUM_COLS = ["set_name", "factories", "n_clients", "method", "mu", "seed",
            "fd004_rmse", "fd004_nasa", "final_val_rmse", "rounds", "source"]
NUM_CSV = RESULTS_DIR / "stage9_num_factories.csv"
NUM_CURVES = RESULTS_DIR / "stage9_num_factories_curves.jsonl"
SIZE2_SETS = [("FD001", "FD002"), ("FD001", "FD003"), ("FD002", "FD003")]


def _read_num() -> pd.DataFrame:
    if NUM_CSV.exists():
        return pd.read_csv(NUM_CSV)
    return pd.DataFrame(columns=NUM_COLS)


def task2() -> None:
    factories, client_data, client_val_data = cmapss_data()

    df_new = _read_num()

    def size1_rows():
        if not pd.read_csv(RESULTS_DIR / "baselines.csv").empty:
            pass
        base = pd.read_csv(RESULTS_DIR / "baselines.csv")
        for f in CLIENTS:
            for seed in [0, 1, 2]:
                m = (base["setup"] == f"Local {f}") & (base["seed"] == seed) \
                    & (base["test_factory"] == "FD004")
                if m.sum() == 0:
                    continue
                r = base[m].iloc[0]
                df_n = _read_num()
                key = (f"size1_{f}", "Local", seed)
                if not df_n.empty and any(df_n.apply(
                        lambda x: (x["set_name"], x["method"], x["seed"]) == key, axis=1)):
                    continue
                yield {"set_name": f"size1_{f}", "factories": f, "n_clients": 1,
                       "method": "Local", "mu": np.nan, "seed": seed,
                       "fd004_rmse": r["rmse"], "fd004_nasa": r["nasa_score"],
                       "final_val_rmse": np.nan, "rounds": np.nan,
                       "source": "size1_local_baselines"}

    def size3_rows():
        fed = pd.read_csv(RESULTS_DIR / "federated_final.csv")
        for setup, method, mu in [("FedAvg", "FedAvg", 0.0), ("FedProx", "FedProx", MU)]:
            for seed in [0, 1, 2]:
                m = (fed["setup"] == setup) & (fed["seed"] == seed) \
                    & (fed["test_factory"] == "FD004")
                if m.sum() == 0:
                    continue
                r = fed[m].iloc[0]
                df_n = _read_num()
                key = ("size3_FD001+FD002+FD003", method, seed)
                if not df_n.empty and any(df_n.apply(
                        lambda x: (x["set_name"], x["method"], x["seed"]) == key, axis=1)):
                    continue
                yield {"set_name": "size3_FD001+FD002+FD003",
                       "factories": "+".join(CLIENTS), "n_clients": 3,
                       "method": method, "mu": mu, "seed": seed,
                       "fd004_rmse": r["rmse"], "fd004_nasa": r["nasa_score"],
                       "final_val_rmse": np.nan, "rounds": 30,
                       "source": "size3_final_reuse"}

    df_n = _read_num()
    df_n = pd.concat([df_n, pd.DataFrame(list(size1_rows()) + list(size3_rows()))],
                     ignore_index=True)
    df_n.to_csv(NUM_CSV, index=False)

    for pair in SIZE2_SETS:
        names = sorted(pair)
        sub = {f: client_data[f] for f in names}
        set_name = f"size2_" + "+".join(names)
        for method, mu in [("fedavg", 0.0), ("fedprox", MU)]:
            mname = "FedAvg" if method == "fedavg" else "FedProx"
            for seed in [0, 1, 2]:
                df_n = _read_num()
                if not df_n.empty and any(df_n.apply(
                        lambda x: (x["set_name"], x["method"], x["seed"])
                        == (set_name, mname, seed), axis=1)):
                    log(f"skip {set_name} {mname} seed {seed}")
                    continue
                t0 = time.time()
                model, hist, _ = run_federated(
                    method, sub, n_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                    mu=mu, lr=LR, batch_size=BATCH_SIZE, seed=seed, device=device(),
                    client_val_data={f: {"X_val": client_val_data[f]["X_val"],
                                         "y_val": client_val_data[f]["y_val"]} for f in names},
                    val_every=1, verbose=False)
                dt = time.time() - t0
                mm = evaluate_on_factory(model, factories["FD004"]["X_test"],
                                         factories["FD004"]["y_test"], device(), CLIP_MAX)
                final_val = float(hist["val_rmse"][-1])
                df_n = _read_num()
                df_n = pd.concat([df_n, pd.DataFrame([{
                    "set_name": set_name, "factories": "+".join(names),
                    "n_clients": 2, "method": mname, "mu": mu, "seed": seed,
                    "fd004_rmse": mm["rmse"], "fd004_nasa": mm["nasa_score"],
                    "final_val_rmse": final_val, "rounds": ROUNDS,
                    "source": "size2"}])], ignore_index=True)
                df_n.to_csv(NUM_CSV, index=False)
                with open(NUM_CURVES, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"set_name": set_name, "method": mname,
                                         "mu": mu, "seed": seed,
                                         "round": hist["round"],
                                         "val_rmse": hist["val_rmse"]}) + "\n")
                log(f"{set_name} {mname} seed {seed}  FD004 RMSE={mm['rmse']:.3f} "
                    f"[{dt:.0f}s]")
    log("task2 complete")


# --------------------------------------------------------------------------- #
# task3: heterogeneity per client set
# --------------------------------------------------------------------------- #

def task3() -> None:
    from scipy.stats import wasserstein_distance
    from src import kg as kg_mod
    factories, client_data, _ = cmapss_data()
    schema = kg_mod.load_schema()
    sets = {("FD001", "FD002"), ("FD001", "FD003"), ("FD002", "FD003"),
            ("FD001", "FD002", "FD003")}
    rows = []
    for s in sorted(sets, key=lambda x: (len(x), x)):
        names = sorted(s)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        pair_vals = []
        for a, b in pairs:
            per_sensor = []
            Xa, Xb = client_data[a]["X_train"][:, -1, :], client_data[b]["X_train"][:, -1, :]
            for col in range(Xa.shape[1]):
                per_sensor.append(wasserstein_distance(Xa[:, col], Xb[:, col]))
            pair_vals.append(float(np.mean(per_sensor)))
        hetero = float(np.mean(pair_vals))
        has_fan = any("Fan_degradation" in schema["factory_failure_modes"].get(f, [])
                      for f in names)
        rows.append({"set_name": "size%d_" % len(names) + "+".join(names),
                     "factories": "+".join(names), "n_clients": len(names),
                     "has_fan": has_fan, "hetero": hetero})
    out = RESULTS_DIR / "stage9_heterogeneity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    for r in rows:
        log(f"hetero {r['set_name']:28s} fan={r['has_fan']} hetero={r['hetero']:.5f}")
    log("task3 complete")


# --------------------------------------------------------------------------- #
# dp convergence curves (seeds 0-2, sigma 0.03 / 0.1)
# --------------------------------------------------------------------------- #

def task_dp_curves() -> None:
    factories, client_data, client_val_data = cmapss_data()
    for sigma in DP_SIGMAS:
        for seed in [0, 1, 2]:
            done = False
            if CURVES_JSONL.exists():
                for line in CURVES_JSONL.read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if (rec.get("dp_sigma") == sigma and rec.get("seed") == seed
                            and rec.get("method") == "fedprox"):
                        done = True
                        break
            if done:
                log(f"skip dp curves sigma={sigma:g} seed {seed}")
                continue
            labels = pd.read_csv(RESULTS_DIR / "privacy_results.csv")
            ref = labels[(labels["label"] == "DP")
                         & np.isclose(labels["sigma"], sigma)
                         & (labels["seed"] == seed)]["final_val_rmse"].iloc[0]
            t0 = time.time()
            _, hist, _ = run_federated(
                "fedprox", client_data, n_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                mu=MU, lr=LR, batch_size=BATCH_SIZE, seed=seed, device=device(),
                client_val_data=client_val_data, val_every=1,
                dp={"C": load_privacy_C(), "sigma": float(sigma)}, verbose=False)
            dt = time.time() - t0
            final_val = float(hist["val_rmse"][-1])
            with open(CURVES_JSONL, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "fedprox", "mu": MU,
                                     "dp_sigma": float(sigma), "seed": seed,
                                     "round": hist["round"],
                                     "val_rmse": hist["val_rmse"],
                                     "val_round": hist["val_round"],
                                     "comm_cost_bytes": hist["comm_cost_bytes"],
                                     "kind": "dp_backfill"}) + "\n")
            log(f"dp curve sigma={sigma:g} seed {seed} val={final_val:.4f} "
                f"(ref {ref:.4f}) diff={abs(final_val - ref):.2e} [{dt:.0f}s]")
    log("task_dp_curves complete")


# --------------------------------------------------------------------------- #
# azure seeds 3,4
# --------------------------------------------------------------------------- #

def azure_task() -> None:
    from src import azure_data as ad
    from src import azure_fl as azf
    from src import azure_models as am

    meta_path = RESULTS_DIR / "azure_run_meta.json"
    if not meta_path.exists():
        log("azure: azure_run_meta.json missing; using config defaults")
        chosen = {"next24h": 0.1, "gap24_48h": 0.001}
    else:
        chosen = {k: v["chosen_mu"] for k, v in json.loads(meta_path.read_text(encoding="utf-8")).items()
                  if "chosen_mu" in v}

    out_csv = RESULTS_DIR / "azure_results.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
    else:
        existing = pd.DataFrame()

    def azure_fl_seeds(setting, setup, mu, seeds, clients, ho_all, ho_test,
                       failures, dev):
        rows = []
        for seed in seeds:
            r, hist = azf.run_federated(clients, ho_all, ho_test, failures, dev,
                                        setting, setup=setup, mu=mu, seed=seed,
                                        verbose=True)
            azf._save_history(setting, setup, mu, seed, hist)
            rows += r
        return rows

    for setting in ["next24h", "gap24_48h"]:
        mu_chosen = chosen[setting]
        dev = azf.get_device()
        clients = azf.load_clients(ad.CLIENTS, setting=setting)
        ho_all, ho_test = azf._heldout_sets(setting)
        failures = ad.load_failures()
        new_rows = []

        for cname, d in clients.items():
            for seed in NEW_SEEDS:
                if not existing.empty and ((existing["setting"] == setting)
                                           & (existing["setup"] == "LocalOnly")
                                           & (existing["train_factory"] == cname)
                                           & (existing["seed"] == seed)).sum() >= 3:
                    continue
                model, hist = am.train_local_mlp(
                    azf.INPUT_DIM, d["train_X"], d["train_y"], d["val_X"], d["val_y"],
                    epochs=azf.MAX_EPOCHS, batch_size=azf.BATCH, lr=azf.LR,
                    dropout=azf.DROPOUT, hidden=azf.HIDDEN, patience=azf.PATIENCE,
                    seed=seed, device=dev, verbose=False)
                thr = am.pick_threshold(d["val_y"], am.predict_proba(model, d["val_X"], dev))
                own = {"X": d["test_X"], "y": d["test_y"], "time": d["test_time"],
                       "machine": d["test_machine"]}
                new_rows.append(azf._eval_row("LocalOnly", cname, seed, "own_test",
                                              d["test_y"], am.predict_proba(model, d["test_X"], dev),
                                              thr, own, failures, setting))
                new_rows.append(azf._eval_row("LocalOnly", cname, seed, "heldout_all",
                                              ho_all["y"], am.predict_proba(model, ho_all["X"], dev),
                                              thr, ho_all, failures, setting))
                new_rows.append(azf._eval_row("LocalOnly", cname, seed, "heldout_test",
                                              ho_test["y"], am.predict_proba(model, ho_test["X"], dev),
                                              thr, ho_test, failures, setting))
                log(f"azure {setting} LocalOnly {cname} seed {seed} done")

        Xtr, ytr = azf._pool(clients, "train", "X"), azf._pool(clients, "train", "y")
        Xva, yva = azf._pool(clients, "val", "X"), azf._pool(clients, "val", "y")
        for seed in NEW_SEEDS:
            if not existing.empty and ((existing["setting"] == setting)
                                       & (existing["setup"] == "Centralized")
                                       & (existing["seed"] == seed)).sum() >= 2:
                continue
            model, hist = am.train_local_mlp(
                azf.INPUT_DIM, Xtr, ytr, Xva, yva, epochs=azf.MAX_EPOCHS,
                batch_size=azf.BATCH, lr=azf.LR, dropout=azf.DROPOUT,
                hidden=azf.HIDDEN, patience=azf.PATIENCE, seed=seed,
                device=dev, verbose=False)
            thr = am.pick_threshold(yva, am.predict_proba(model, Xva, dev))
            tf = "+".join(clients.keys())
            new_rows.append(azf._eval_row("Centralized", tf, seed, "heldout_all",
                                          ho_all["y"], am.predict_proba(model, ho_all["X"], dev),
                                          thr, ho_all, failures, setting))
            new_rows.append(azf._eval_row("Centralized", tf, seed, "heldout_test",
                                          ho_test["y"], am.predict_proba(model, ho_test["X"], dev),
                                          thr, ho_test, failures, setting))
            log(f"azure {setting} Centralized seed {seed} done")

        tf = "+".join(clients.keys())
        for setup, mu in [("FedAvg", 0.0), ("FedProx", mu_chosen)]:
            for seed in NEW_SEEDS:
                if not existing.empty and ((existing["setting"] == setting)
                                           & (existing["setup"] == setup)
                                           & (existing["seed"] == seed)).sum() >= 2:
                    continue
                r, hist = azf.run_federated(clients, ho_all, ho_test, failures, dev,
                                            setting, setup=setup, mu=mu, seed=seed,
                                            verbose=True)
                azf._save_history(setting, setup, mu, seed, hist)
                new_rows += r
                log(f"azure {setting} {setup} mu={mu:g} seed {seed} done")

        if new_rows:
            out_csv = RESULTS_DIR / "azure_results.csv"
            existing = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["setting", "setup", "train_factory", "seed", "eval_set"],
                keep="first").sort_values(
                ["setting", "setup", "train_factory", "seed", "eval_set"])
            combined.to_csv(out_csv, index=False)
            log(f"azure {setting}: appended {len(new_rows)} rows "
                f"(total {len(combined)})")
    log("azure task complete")


# --------------------------------------------------------------------------- #
# explanation repeat (seeds 43, 44)
# --------------------------------------------------------------------------- #

def _ollama_reachable() -> bool:
    try:
        import requests as _req
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "llm_config.yaml")
                             .read_text(encoding="utf-8"))
        base = cfg["ollama"]["base_url"].rstrip("/")
        resp = _req.get(base + "/api/tags", timeout=5)
        return resp.status_code == 200 and bool(resp.json().get("models"))
    except Exception:
        return False


def _try_start_ollama() -> bool:
    import shutil as _shutil
    _which = _shutil.which("ollama")
    _candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"]
    if _which:
        _candidates.insert(0, Path(_which))
    for p in _candidates:
        if p.exists():
            try:
                subprocess.Popen([str(p), "serve"], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(8)
                return True
            except Exception as e:
                log(f"could not start ollama: {e}")
            return False
    return False


def _stratified_sample(df: pd.DataFrame, per_bucket: int = 15, seed: int = 0) -> pd.DataFrame:
    from src.eval_explain import BUCKETS, add_bucket
    framed = add_bucket(df)
    rng = np.random.default_rng(seed)
    picked = []
    for b in BUCKETS:
        sub = framed[framed["rul_bucket"] == b]
        if len(sub) < per_bucket:
            raise ValueError(f"bucket {b} has only {len(sub)} engines (need {per_bucket})")
        chosen = rng.choice(sub["engine"].unique(), size=per_bucket, replace=False)
        picked.append(sub[sub["engine"].isin(chosen)])
    return pd.concat(picked, ignore_index=True)


def explanation_repeat() -> None:
    import os as _os  # noqa: F401  (kept for parity with task0)

    from src import eval_explain
    from src import explain

    out_dir = RESULTS_DIR / "explanations_repeat"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _ollama_reachable():
        if _try_start_ollama() and _ollama_reachable():
            log("ollama started OK")
        else:
            (out_dir / "status.json").write_text(
                json.dumps({"skipped": True, "reason": "ollama unreachable",
                            "note": "only the explanation repeat was skipped"}),
                encoding="utf-8")
            log("explanations: Ollama unreachable - SKIPPED (documented in "
                "results/explanations_repeat/status.json)")
            return

    predictions = explain.load_predictions()
    combined = []
    for seed in [43, 44]:
        sample = _stratified_sample(predictions, per_bucket=15, seed=seed)
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(seed_dir / "sample.csv", index=False)
        merged = {}
        for v in ["A", "B"]:
            expl = explain.run_explanations(sample, v)
            m = eval_explain.merge_checks(expl)
            merged[v] = m
            expl.to_csv(seed_dir / f"explanations_{v}.csv", index=False)
        merged_all = pd.concat([merged["A"], merged["B"]], ignore_index=True)
        tbl_all = eval_explain.summarise(merged_all)
        for v in ["A", "B"]:
            tbl = tbl_all[tbl_all["version"] == v]
            tbl.to_csv(seed_dir / f"metrics_{v}.csv", index=False)
            overall = tbl[tbl["group_type"] == "overall"].iloc[0]
            combined.append({"sample": seed, "version": v,
                             **{k: overall[k] for k in tbl.columns if k not in ("sample", "version")}})
            g = eval_explain.group_metrics(merged[v])
            log(f"explanations seed {seed} version {v}: "
                f"halluc={g['hallucination_rate']:.3f} "
                f"[{g['hall_ci_lo']:.3f},{g['hall_ci_hi']:.3f}] n={g['n']}")
        mcn = eval_explain.mcnemar_and_sign(merged_all)
        (seed_dir / "mcnemar.json").write_text(json.dumps(mcn, indent=2),
                                               encoding="utf-8")
        (seed_dir / "worst_version_b.json").write_text(
            json.dumps(eval_explain.worst_version_b(merged["B"]), indent=2),
            encoding="utf-8")
        log(f"explanations seed {seed}: mcnemar={mcn['p_mcnemar']:.4f} "
            f"(A_only_halluc={mcn['A_only_halluc']}, B_only_halluc={mcn['B_only_halluc']})")

    ref = pd.read_csv(RESULTS_DIR / "explanations" / "metrics.csv")
    ref = ref[ref["group_type"] == "overall"].copy()
    ref["sample"] = 42
    cols = ["sample", "version"] + [c for c in ref.columns if c not in ("sample", "version")]
    out = pd.concat([ref[cols], pd.DataFrame(combined)], ignore_index=True)
    out.to_csv(out_dir / "overall_metrics.csv", index=False)
    (out_dir / "status.json").write_text(
        json.dumps({"skipped": False, "samples": [42, 43, 44]}), encoding="utf-8")
    log("explanations repeat complete")


# --------------------------------------------------------------------------- #
# task0: independent raw-row verification of model1/model4 positive counts
# --------------------------------------------------------------------------- #

def task0() -> None:
    """Independent raw-row count of train positives for model1 vs model4.

    Re-derives the label rule from the raw tables (PdM_failures.csv +
    PdM_telemetry.csv + PdM_machines.csv) with the EXACT keep/split rules of
    src/azure_data.build_features + split_factory:
      - per machine: drop the first 24h (rolling-window completeness)
      - drop samples after global_max - (gap+horizon)   (label-window completeness)
      - train part: t < train_end AND t + (gap+horizon) <= train_end
      - label: any failure in (t + gap, t + gap + horizon]
    and compares the count with stats.json. Also cross-checks against the
    library path (build_features + split_factory).
    """
    from src import azure_data as ad

    raw = ad.load_raw(verbose=False)
    fails, mach, tel = raw["failures"], raw["machines"], raw["telemetry"]
    mach_map = dict(zip(mach["machineID"].astype(int), mach["model"]))
    HOUR_NS = 3600 * 1_000_000_000
    TRAIN_END_NS = int(ad.TRAIN_END.value)
    GLOBAL_MAX_NS = int(tel["datetime"].max().value)

    fail_by_machine = {}
    for m, sub in fails.groupby("machineID"):
        fail_by_machine[int(m)] = np.sort(ad._ns(sub["datetime"]))
    tel_by_machine = {}
    for m, sub in tel.groupby("machineID"):
        tel_by_machine[int(m)] = np.sort(ad._ns(sub["datetime"]))

    out_rows = []
    for setting in ["next24h", "gap24_48h"]:
        gap_h, horizon_h = ad.SETTINGS[setting]["gap_h"], ad.SETTINGS[setting]["horizon_h"]
        gap_ns = gap_h * HOUR_NS
        win_ns = (gap_h + horizon_h) * HOUR_NS
        first24_ns = 24 * HOUR_NS

        feats_all = ad.build_features(raw, gap_h=gap_h, horizon_h=horizon_h)

        for model in ["model1", "model4"]:
            stats = ad.load_processed(model, setting=setting)["stats"]
            pos0 = int(stats["positives"]["train"])
            ms = sorted(m for m, mod in mach_map.items() if mod == model)

            pos_hand = 0
            n_hand = 0
            for m in ms:
                t = tel_by_machine.get(m)
                if t is None or len(t) == 0:
                    continue
                f = fail_by_machine.get(m)
                keep = (t >= t[0] + first24_ns) & (t <= GLOBAL_MAX_NS - win_ns)
                train_keep = keep & (t <= TRAIN_END_NS - win_ns)
                tt = t[train_keep]
                n_hand += len(tt)
                if f is None or len(f) == 0:
                    continue
                lo = tt + gap_ns
                hi = tt + win_ns
                pos_hand += int(((f[:, None] > lo[None, :]) & (f[:, None] <= hi[None, :])).sum())

            sub = feats_all[feats_all["model"] == model]
            splits = ad.split_factory(sub, gap_h=gap_h, horizon_h=horizon_h)
            pos_lib = int(splits["train"]["label"].sum())

            out_rows.append({
                "setting": setting, "model": model,
                "machines": len(ms),
                "stats_train_pos": pos0,
                "stats_train_n": int(stats["counts"]["train"]),
                "hand_labeled_pos": int(pos_hand),
                "hand_train_samples": int(n_hand),
                "lib_labeled_pos": int(pos_lib),
                "match_stats_hand": pos0 == int(pos_hand),
                "match_stats_lib": pos0 == int(pos_lib),
            })
            log(f"task0 {setting:10s} {model}: stats={pos0} "
                f"raw-hand={pos_hand} lib={pos_lib} "
                f"{'MATCH' if pos0 == pos_hand == pos_lib else 'MISMATCH'}")

    out = RESULTS_DIR / "stage9_task0.csv"
    pd.DataFrame(out_rows).to_csv(out, index=False)
    log(f"task0 complete -> {out.name}")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 9 heavy runner")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["regression", "task0", "task1_cmapss", "backfill",
                             "task2", "task3", "dp_curves", "azure", "explanations"])
    args = ap.parse_args()
    only = args.only

    check_gpu()
    log(f"device={device()}")

    sections = ["regression", "task0", "task1_cmapss", "backfill", "task2",
                "task3", "dp_curves", "azure", "explanations"]
    if only is None:
        only = sections

    for name in only:
        if name not in sections:
            continue
        log(f"=== section: {name} ===")
        try:
            if name == "regression":
                run_regression()
            elif name == "task0":
                task0()
            elif name == "task1_cmapss":
                task1_cmapss()
            elif name == "backfill":
                task_backfill()
            elif name == "task2":
                task2()
            elif name == "task3":
                task3()
            elif name == "dp_curves":
                task_dp_curves()
            elif name == "azure":
                azure_task()
            elif name == "explanations":
                explanation_repeat()
        except Exception as e:
            import traceback
            log(f"section {name} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    log(f"finished in {(time.time() - START) / 60:.1f} minutes")


if __name__ == "__main__":
    main()