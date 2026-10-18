"""
Stage 8: experiments for the Azure binary task, for both prediction settings.

Settings (label window relative to sample time t):
  - "next24h"   : any failure in (t, t + 24h]
  - "gap24_48h" : any failure in (t + 24h, t + 48h]

Setups (all evaluated on the held-out factory model4, plus each local model on
its own factory test set):
  - LocalOnly   : one MLP per client factory
  - Centralized : one MLP on the pooled client factories
  - FedAvg      : federated averaging over the three client factories
  - FedProx     : FedAvg + proximal term, mu chosen on client validation only
  - LogReg      : logistic regression reference on pooled client train
  - NegControl  : centralized MLP trained on per-machine shuffled labels (seed 0)

F1 decision thresholds come only from validation data of the training
factories. Nothing is tuned on the held-out factory. Held-out inference uses
manual slicing (no DataLoader RNG draws) so the training random stream is never
perturbed by evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src import azure_data as ad
from src import azure_models as am
from src.federated import fedavg, snapshot_global, compute_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "configs" / "azure_fl_config.yaml"

ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH = 1024
LR = 1e-3
DROPOUT = 0.2
HIDDEN = 64
MAX_EPOCHS = 40
PATIENCE = 6
SEEDS = [0, 1, 2]
MU_GRID = [0.001, 0.01, 0.1]
INPUT_DIM = len(ad.FEATURE_NAMES)

FEATURE_GROUPS = {
    "all_features": lambda C: None,
    "errors_only": lambda C: [i for i, n in enumerate(C) if n.startswith("err_")],
    "telemetry_only": lambda C: [i for i, n in enumerate(C) if "_mean_" in n or "_std_" in n],
    "errors+telemetry": lambda C: [i for i, n in enumerate(C)
                                   if n.startswith("err_") or "_mean_" in n or "_std_" in n],
    "errors+maint": lambda C: [i for i, n in enumerate(C)
                               if n.startswith("err_") or n.startswith("hours_since_maint")],
    "no_errors": lambda C: [i for i, n in enumerate(C) if not n.startswith("err_")],
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def load_clients(factories: List[str], setting: str) -> Dict[str, dict]:
    return {f: ad.load_processed(f, setting=setting) for f in factories}


def _heldout_sets(setting: str) -> Tuple[dict, dict]:
    d = ad.load_processed(ad.HELDOUT, setting=setting)
    heldout_all = {"X": d["all_X"], "y": d["all_y"], "time": d["all_time"], "machine": d["all_machine"]}
    heldout_test = {"X": d["test_X"], "y": d["test_y"], "time": d["test_time"], "machine": d["test_machine"]}
    return heldout_all, heldout_test


def n_failure_events(part: dict, failures: pd.DataFrame, setting: str) -> int:
    return ad.count_failure_events(np.unique(part["machine"]),
                                   int(part["time"].min()), int(part["time"].max()),
                                   failures=failures, setting=setting)


def _eval_row(setup: str, train_factory: str, seed: int, eval_set: str,
              y_true: np.ndarray, proba: np.ndarray, threshold: float,
              part: dict, failures: pd.DataFrame, setting: str) -> dict:
    m = am.evaluate_binary(y_true, proba, threshold)
    return {
        "setting": setting, "setup": setup, "train_factory": train_factory, "seed": seed,
        "eval_set": eval_set, "pr_auc": m["pr_auc"], "f1": m["f1"], "threshold": m["threshold"],
        "prevalence": m["prevalence"], "n_positive_samples": m["n_positive_samples"],
        "n_failure_events": n_failure_events(part, failures, setting),
    }


# --------------------------------------------------------------------------- #
# Local-only and centralized
# --------------------------------------------------------------------------- #

def run_local_only(clients: Dict[str, dict], heldout_all: dict, heldout_test: dict,
                   failures: pd.DataFrame, device, setting: str, verbose=True) -> List[dict]:
    rows = []
    for cname, d in clients.items():
        for seed in SEEDS:
            model, hist = am.train_local_mlp(
                INPUT_DIM, d["train_X"], d["train_y"], d["val_X"], d["val_y"],
                epochs=MAX_EPOCHS, batch_size=BATCH, lr=LR, dropout=DROPOUT, hidden=HIDDEN,
                patience=PATIENCE, seed=seed, device=device, verbose=False)
            thr = am.pick_threshold(d["val_y"], am.predict_proba(model, d["val_X"], device))
            own = {"X": d["test_X"], "y": d["test_y"], "time": d["test_time"], "machine": d["test_machine"]}
            rows.append(_eval_row("LocalOnly", cname, seed, "own_test", d["test_y"],
                                  am.predict_proba(model, d["test_X"], device), thr, own, failures, setting))
            rows.append(_eval_row("LocalOnly", cname, seed, "heldout_all", heldout_all["y"],
                                  am.predict_proba(model, heldout_all["X"], device), thr, heldout_all, failures, setting))
            rows.append(_eval_row("LocalOnly", cname, seed, "heldout_test", heldout_test["y"],
                                  am.predict_proba(model, heldout_test["X"], device), thr, heldout_test, failures, setting))
            if verbose:
                print(f"    LocalOnly {cname} seed{seed} val_auc={hist['best_val_pr_auc']:.4f} thr={thr:.4f}")
    return rows


def _pool(clients: Dict[str, dict], part: str, key: str) -> np.ndarray:
    return np.concatenate([clients[c][f"{part}_{key}"] for c in clients], axis=0)


def run_centralized(clients: Dict[str, dict], heldout_all: dict, heldout_test: dict,
                    failures: pd.DataFrame, device, setting: str, verbose=True) -> List[dict]:
    Xtr, ytr = _pool(clients, "train", "X"), _pool(clients, "train", "y")
    Xva, yva = _pool(clients, "val", "X"), _pool(clients, "val", "y")
    rows = []
    for seed in SEEDS:
        model, hist = am.train_local_mlp(
            INPUT_DIM, Xtr, ytr, Xva, yva, epochs=MAX_EPOCHS, batch_size=BATCH, lr=LR,
            dropout=DROPOUT, hidden=HIDDEN, patience=PATIENCE, seed=seed, device=device, verbose=False)
        thr = am.pick_threshold(yva, am.predict_proba(model, Xva, device))
        tf = "+".join(clients.keys())
        rows.append(_eval_row("Centralized", tf, seed, "heldout_all", heldout_all["y"],
                              am.predict_proba(model, heldout_all["X"], device), thr, heldout_all, failures, setting))
        rows.append(_eval_row("Centralized", tf, seed, "heldout_test", heldout_test["y"],
                              am.predict_proba(model, heldout_test["X"], device), thr, heldout_test, failures, setting))
        if verbose:
            print(f"    Centralized seed{seed} val_auc={hist['best_val_pr_auc']:.4f} thr={thr:.4f}")
    return rows


# --------------------------------------------------------------------------- #
# Federated (FedAvg / FedProx)
# --------------------------------------------------------------------------- #

def _local_train(model, X, y, criterion, device, global_state, mu, rng):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for bidx in am.manual_batches(len(X), BATCH, rng=rng, shuffle=True):
        xb = torch.as_tensor(X[bidx], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[bidx], dtype=torch.float32, device=device)
        opt.zero_grad()
        loss, _ = compute_loss(model, xb, yb, criterion, device, global_state, mu)
        loss.backward()
        opt.step()


def run_federated(clients: Dict[str, dict], heldout_all: dict, heldout_test: dict,
                  failures: pd.DataFrame, device, setting: str, setup="FedAvg", mu=0.0,
                  seed=0, verbose=False) -> Tuple[List[dict], dict]:
    """One federated run for a setting. Returns (rows, history)."""
    names = list(clients.keys())
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = am.AzureMLP(INPUT_DIM, hidden=HIDDEN, dropout=DROPOUT).to(device)
    histories = {"round": [], "weighted_val_pr_auc": []}
    sizes = [len(clients[c]["train_y"]) for c in names]

    def weighted_val_auc(model):
        num, den = 0.0, 0.0
        for c in names:
            p = am.predict_proba(model, clients[c]["val_X"], device)
            a = am.pr_auc(clients[c]["val_y"], p)
            if not np.isnan(a):
                num += len(clients[c]["val_y"]) * a
                den += len(clients[c]["val_y"])
        return num / den if den else float("nan")

    for rnd in range(ROUNDS):
        global_state = snapshot_global(global_model)
        states = []
        for c in names:
            d = clients[c]
            lm = am.AzureMLP(INPUT_DIM, hidden=HIDDEN, dropout=DROPOUT).to(device)
            lm.load_state_dict(global_state)
            crit = am.make_criterion(d["train_y"]).to(device)
            rng = np.random.default_rng(seed * 1000 + rnd)
            for _ in range(LOCAL_EPOCHS):
                _local_train(lm, d["train_X"], d["train_y"], crit, device, global_state, mu, rng)
            states.append(snapshot_global(lm))
        global_model.load_state_dict(fedavg(states, sizes))
        wauc = weighted_val_auc(global_model)
        histories["round"].append(rnd)
        histories["weighted_val_pr_auc"].append(wauc)
        if verbose and (rnd % 5 == 0 or rnd == ROUNDS - 1):
            print(f"    {setup} mu={mu} seed{seed} round {rnd:02d} weighted_val_pr_auc={wauc:.4f}")

    Xva = _pool(clients, "val", "X")
    yva = _pool(clients, "val", "y")
    thr = am.pick_threshold(yva, am.predict_proba(global_model, Xva, device))
    tf = "+".join(names)
    rows = [
        _eval_row(setup, tf, seed, "heldout_all", heldout_all["y"],
                  am.predict_proba(global_model, heldout_all["X"], device), thr, heldout_all, failures, setting),
        _eval_row(setup, tf, seed, "heldout_test", heldout_test["y"],
                  am.predict_proba(global_model, heldout_test["X"], device), thr, heldout_test, failures, setting),
    ]
    histories["final_val_pr_auc"] = weighted_val_auc(global_model)
    histories["best_val_pr_auc"] = float(np.nanmax(histories["weighted_val_pr_auc"]))
    return rows, histories


# --------------------------------------------------------------------------- #
# LogReg reference
# --------------------------------------------------------------------------- #

def run_logreg(clients: Dict[str, dict], heldout_all: dict, heldout_test: dict,
               failures: pd.DataFrame, setting: str, seed=0) -> List[dict]:
    Xtr, ytr = _pool(clients, "train", "X"), _pool(clients, "train", "y")
    Xva, yva = _pool(clients, "val", "X"), _pool(clients, "val", "y")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             solver="lbfgs", random_state=seed)
    clf.fit(Xtr, ytr)
    thr = am.pick_threshold(yva, clf.predict_proba(Xva)[:, 1])
    tf = "+".join(clients.keys())
    rows = [
        _eval_row("LogReg", tf, 0, "heldout_all", heldout_all["y"],
                  clf.predict_proba(heldout_all["X"])[:, 1], thr, heldout_all, failures, setting),
        _eval_row("LogReg", tf, 0, "heldout_test", heldout_test["y"],
                  clf.predict_proba(heldout_test["X"])[:, 1], thr, heldout_test, failures, setting),
    ]
    return rows


# --------------------------------------------------------------------------- #
# Negative control (per-machine circular shift of labels, timestamp-aligned)
# --------------------------------------------------------------------------- #

def _shifted_labels(part: dict) -> np.ndarray:
    """Give each machine the label series of the next machine in sorted order,
    aligned by timestamp (all machines of a factory share the same hour grid)."""
    time, machine, y = part["time"], part["machine"], part["y"]
    s = pd.Series(y, index=pd.MultiIndex.from_arrays([time, machine]))
    ms = sorted(np.unique(machine))
    shift = {m: ms[(i + 1) % len(ms)] for i, m in enumerate(ms)}
    new_m = np.array([shift[m] for m in machine])
    out = s.reindex(pd.MultiIndex.from_arrays([time, new_m])).to_numpy()
    if np.isnan(out).any():
        raise ValueError("_shifted_labels: machines do not share an identical time grid")
    return out


def run_negative_control(clients: Dict[str, dict], heldout_all: dict, heldout_test: dict,
                         failures: pd.DataFrame, device, setting: str,
                         verbose=True) -> List[dict]:
    """Centralized MLP (seed 0) trained on timestamp-aligned shuffled labels.
    Evaluation uses the held-out factory's own shuffled labels."""
    ctr = {}
    for cn, d in clients.items():
        ctr[cn] = {
            "train_y": _shifted_labels({"time": d["train_time"], "machine": d["train_machine"], "y": d["train_y"]}),
            "val_y": _shifted_labels({"time": d["val_time"], "machine": d["val_machine"], "y": d["val_y"]}),
            "train_X": d["train_X"], "val_X": d["val_X"], "train_time": d["train_time"],
            "train_machine": d["train_machine"], "val_time": d["val_time"], "val_machine": d["val_machine"],
        }
    Xtr = np.concatenate([ctr[c]["train_X"] for c in ctr], axis=0)
    ys_tr = np.concatenate([ctr[c]["train_y"] for c in ctr], axis=0)
    Xva = np.concatenate([ctr[c]["val_X"] for c in ctr], axis=0)
    ys_va = np.concatenate([ctr[c]["val_y"] for c in ctr], axis=0)

    model, hist = am.train_local_mlp(
        INPUT_DIM, Xtr, ys_tr, Xva, ys_va, epochs=MAX_EPOCHS, batch_size=BATCH, lr=LR,
        dropout=DROPOUT, hidden=HIDDEN, patience=PATIENCE, seed=0, device=device, verbose=False)
    thr = am.pick_threshold(ys_va, am.predict_proba(model, Xva, device))

    ho_all = {"time": heldout_all["time"], "machine": heldout_all["machine"], "y": heldout_all["y"]}
    ho_test = {"time": heldout_test["time"], "machine": heldout_test["machine"], "y": heldout_test["y"]}
    y_ho_all = _shifted_labels(ho_all)
    y_ho_test = _shifted_labels(ho_test)

    p_all = am.predict_proba(model, heldout_all["X"], device)
    p_test = am.predict_proba(model, heldout_test["X"], device)
    tf = "+".join(clients.keys())
    rows = [
        _eval_row("NegControl", tf, 0, "heldout_all", y_ho_all, p_all, thr, heldout_all, failures, setting),
        _eval_row("NegControl", tf, 0, "heldout_test", y_ho_test, p_test, thr, heldout_test, failures, setting),
    ]
    pa, pv = am.pr_auc(y_ho_all, p_all), float(y_ho_all.mean())
    if verbose:
        print(f"    NegControl {setting} heldout_all PR-AUC={pa:.4f} prevalence={pv:.4f}")
    return rows


# --------------------------------------------------------------------------- #
# Ablation (feature groups, centralized, seed 0, held-out PR-AUC)
# --------------------------------------------------------------------------- #

def run_ablation(setting: str = "next24h", device=None, seed=0,
                 epochs: int = 25, verbose: bool = True) -> pd.DataFrame:
    device = device or get_device()
    clients = load_clients(ad.CLIENTS, setting=setting)
    ho = ad.load_processed(ad.HELDOUT, setting=setting)
    C = ad.FEATURE_NAMES
    rows = []
    for name, getcols in FEATURE_GROUPS.items():
        cols = getcols(C)
        def sl(X, cols=cols): return X if cols is None else X[:, cols]
        Xtr = np.concatenate([sl(clients[c]["train_X"]) for c in ad.CLIENTS], axis=0)
        ytr = np.concatenate([clients[c]["train_y"] for c in ad.CLIENTS], axis=0)
        Xva = np.concatenate([sl(clients[c]["val_X"]) for c in ad.CLIENTS], axis=0)
        yva = np.concatenate([clients[c]["val_y"] for c in ad.CLIENTS], axis=0)
        dim = Xtr.shape[1] if cols is None else len(cols)
        model, _ = am.train_local_mlp(dim, Xtr, ytr, Xva, yva, epochs=epochs, batch_size=BATCH,
                                      lr=LR, dropout=DROPOUT, hidden=HIDDEN, patience=6,
                                      seed=seed, device=device, verbose=False)
        p = am.predict_proba(model, sl(ho["all_X"]), device)
        pr = am.pr_auc(ho["all_y"], p)
        rows.append({"setting": setting, "feature_group": name, "n_features": dim,
                     "heldout_pr_auc": pr, "prevalence": float(ho["all_y"].mean())})
        if verbose:
            print(f"    Ablation[{setting}] {name:18s} n={dim:2d} heldout_pr_auc={pr:.4f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"azure_ablation_{setting}.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Communication cost
# --------------------------------------------------------------------------- #

def comm_cost_mb(rounds: int, n_clients: int) -> Dict[str, float]:
    n_params = sum(p.numel() for p in am.AzureMLP(INPUT_DIM, hidden=HIDDEN).parameters())
    state_bytes = n_params * 4
    per_round = 2 * n_clients * state_bytes / 1e6   # upload + download per client
    return {"n_params": n_params, "mb_per_round": per_round, "total_mb": per_round * rounds}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(setting: str, verbose: bool = True) -> pd.DataFrame:
    device = get_device()
    if verbose:
        print(f"device={device} input_dim={INPUT_DIM} setting={setting}")
    clients = load_clients(ad.CLIENTS, setting=setting)
    heldout_all, heldout_test = _heldout_sets(setting)
    failures = ad.load_failures()

    rows: List[dict] = []

    if verbose:
        print("LocalOnly ...")
    rows += run_local_only(clients, heldout_all, heldout_test, failures, device, setting, verbose)

    if verbose:
        print("Centralized ...")
    rows += run_centralized(clients, heldout_all, heldout_test, failures, device, setting, verbose)

    if verbose:
        print("FedAvg ...")
    for seed in SEEDS:
        r, h = run_federated(clients, heldout_all, heldout_test, failures, device, setting,
                             setup="FedAvg", mu=0.0, seed=seed, verbose=verbose)
        rows += r
        _save_history(setting, "FedAvg", 0.0, seed, h)

    if verbose:
        print("FedProx mu selection (seed 0) ...")
    mu_scores = {}
    mu_rows = {}
    for mu in MU_GRID:
        r, h = run_federated(clients, heldout_all, heldout_test, failures, device, setting,
                             setup="FedProx", mu=mu, seed=0, verbose=verbose)
        mu_scores[mu] = h["best_val_pr_auc"]
        mu_rows[mu] = r
        _save_history(setting, "FedProx", mu, 0, h)
    best_mu = max(mu_scores, key=mu_scores.get)
    if verbose:
        print(f"  mu scores={ {str(k): round(v, 4) for k, v in mu_scores.items()} } -> chosen mu={best_mu}")

    rows += mu_rows[best_mu]
    for seed in SEEDS:
        if seed == 0:
            continue
        r, h = run_federated(clients, heldout_all, heldout_test, failures, device, setting,
                             setup="FedProx", mu=best_mu, seed=seed, verbose=verbose)
        rows += r
        _save_history(setting, "FedProx", best_mu, seed, h)

    if verbose:
        print("LogReg ...")
    rows += run_logreg(clients, heldout_all, heldout_test, failures, setting, seed=0)

    if verbose:
        print("Negative control (shuffled labels) ...")
    rows += run_negative_control(clients, heldout_all, heldout_test, failures, device, setting, verbose)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _update_run_meta(setting, best_mu, mu_scores)
    if verbose:
        print(f"  setting={setting} done, {len(df)} rows")
    return df


def _save_history(setting: str, setup: str, mu: float, seed: int, hist: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"setting": setting, "setup": setup, "mu": mu, "seed": seed,
           "round": hist["round"], "weighted_val_pr_auc": hist["weighted_val_pr_auc"],
           "best_val_pr_auc": hist["best_val_pr_auc"], "final_val_pr_auc": hist["final_val_pr_auc"]}
    with open(RESULTS / f"azure_fl_histories_{setting}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


_META = None


def _update_run_meta(setting: str, best_mu: float, mu_scores: Dict[float, float]):
    global _META
    import os
    path = RESULTS / "azure_run_meta.json"
    if _META is None and os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{}")  # reset stale partial file
    if _META is None:
        _META = {}
    _META[setting] = {
        "chosen_mu": float(best_mu),
        "mu_scores": {str(k): v for k, v in mu_scores.items()},
        "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
    }
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        saved.update(_META)
    else:
        saved = _META
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def _smoke(setting: str = "next24h"):
    device = torch.device("cpu")
    clients = load_clients(ad.CLIENTS, setting=setting)
    heldout_all, heldout_test = _heldout_sets(setting)
    failures = ad.load_failures()
    r, h = run_federated(clients, heldout_all, heldout_test, failures, device, setting,
                         setup="FedAvg", mu=0.0, seed=0, verbose=True)
    nc = run_negative_control(clients, heldout_all, heldout_test, failures, device, setting, verbose=True)
    for row in r + nc:
        print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print("smoke OK")


if __name__ == "__main__":
    import sys
    setting = "next24h"
    if "--setting" in sys.argv:
        setting = sys.argv[sys.argv.index("--setting") + 1]
    if "--smoke" in sys.argv:
        _smoke(setting)
    else:
        df = run_all(setting, verbose=True)
        out = RESULTS / "azure_results_work.csv"
        df.to_csv(out, index=False)
        print(f"saved {len(df)} rows -> {out}")