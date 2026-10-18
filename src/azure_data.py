"""
Stage 8: Azure Predictive Maintenance dataset - loading, features, split.

This is a SEPARATE pipeline from the C-MAPSS stages and never imports or
modifies the C-MAPSS modules.

Factories = machine model (model1..model4) from PdM_machines.csv.
Clients   = model1, model2, model3.
Held-out  = model4 (fixed before any result).

One sample per machine per hour t. Two prediction settings share the same
features (information with time <= t only):
  - "next24h"   : label = 1 if that machine has any failure in (t, t + 24h]
  - "gap24_48h" : label = 1 if that machine has any failure in (t + 24h, t + 48h]
Features (per machine, computed only from data with time <= t):
  - rolling mean/std over 3h and 24h of volt, rotate, pressure, vibration (16)
  - counts of error1..error5 in the last 24h (5)
  - hours since the last maintenance for comp1..comp4 (4; if none yet, hours
    since the machine's first record)
  - machine age (1)
Normalization is LOCAL: mean/std from each factory's own train-period samples.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import set_seed  # noqa: F401  (kept for parity with other stages)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "azure"
PROC_DIR = PROJECT_ROOT / "data" / "processed_azure"

FACTORIES = ["model1", "model2", "model3", "model4"]
CLIENTS = ["model1", "model2", "model3"]
HELDOUT = "model4"

TELE = ["volt", "rotate", "pressure", "vibration"]
WINDOWS = [3, 24]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
COMPS = ["comp1", "comp2", "comp3", "comp4"]

FEATURE_NAMES = (
    [f"{c}_mean_{w}" for c in TELE for w in WINDOWS]
    + [f"{c}_std_{w}" for c in TELE for w in WINDOWS]
    + [f"err_{e}_last24h" for e in ERRORS]
    + [f"hours_since_maint_{c}" for c in COMPS]
    + ["age"]
)

TRAIN_END = pd.Timestamp("2015-09-01")
VAL_END = pd.Timestamp("2015-11-01")
HOUR_NS = 3600 * 1_000_000_000

# settings: label window = (t + gap_hours, t + gap_hours + horizon_hours]
SETTINGS = {
    "next24h": {"gap_h": 0, "horizon_h": 24},
    "gap24_48h": {"gap_h": 24, "horizon_h": 24},
}
ROLLING_H = 24  # first hours dropped per machine for rolling-window completeness


def window_ns(gap_h: int, horizon_h: int) -> Tuple[int, int]:
    return gap_h * HOUR_NS, (gap_h + horizon_h) * HOUR_NS


def _ns(series: pd.Series) -> np.ndarray:
    return series.values.astype("datetime64[ns]").astype("int64")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load the five tables, report basic quality, and clean sensibly."""
    tel = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    err = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    mnt = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    fail = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    mach = pd.read_csv(RAW_DIR / "PdM_machines.csv")

    if verbose:
        for name, df in [("telemetry", tel), ("errors", err), ("maintenance", mnt),
                         ("failures", fail), ("machines", mach)]:
            rng = (f"{df['datetime'].min()} .. {df['datetime'].max()}"
                   if "datetime" in df else "-")
            print(f"{name:12s} rows={len(df):7d}  date range: {rng}  "
                  f"missing={int(df.isna().sum().sum())}  "
                  f"duplicate rows={int(df.duplicated().sum())}")

    # --- duplicates ---
    tel = tel.drop_duplicates().drop_duplicates(["machineID", "datetime"], keep="first")
    err = err.drop_duplicates(["machineID", "datetime", "errorID"], keep="first")
    mnt = mnt.drop_duplicates(["machineID", "datetime", "comp"], keep="first")
    # multiple simultaneous component failures -> dedup on the event, not the part
    fail = fail.drop_duplicates(["machineID", "datetime"], keep="first")
    mach = mach.drop_duplicates("machineID", keep="first")

    # --- missing values ---
    tel = tel.dropna(subset=["datetime", "machineID", "volt", "rotate", "pressure", "vibration"])
    err = err.dropna(subset=["datetime", "machineID", "errorID"])
    mnt = mnt.dropna(subset=["datetime", "machineID", "comp"])
    fail = fail.dropna(subset=["datetime", "machineID", "failure"])
    mach = mach.dropna(subset=["machineID", "model", "age"])

    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return {"telemetry": tel, "errors": err, "maintenance": mnt,
            "failures": fail, "machines": mach}


# --------------------------------------------------------------------------- #
# Feature engineering (per machine, time <= t only)
# --------------------------------------------------------------------------- #

def build_features(raw: Dict[str, pd.DataFrame],
                   gap_h: int = 0, horizon_h: int = 24) -> pd.DataFrame:
    tel, err, mnt, fail, mach = (raw["telemetry"], raw["errors"],
                                 raw["maintenance"], raw["failures"], raw["machines"])

    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    out = tel[["machineID", "datetime"]].copy()

    # rolling mean/std of telemetry (per machine)
    for c in TELE:
        grp = tel.groupby("machineID")[c]
        for w in WINDOWS:
            out[f"{c}_mean_{w}"] = grp.rolling(w, min_periods=w).mean() \
                .reset_index(level=0, drop=True)
            out[f"{c}_std_{w}"] = grp.rolling(w, min_periods=w).std() \
                .reset_index(level=0, drop=True)
    out = out.sort_index()

    # event times per machine (sorted int64 ns)
    def by_machine(df: pd.DataFrame) -> Dict[int, np.ndarray]:
        d = {}
        for m, sub in df.groupby("machineID"):
            d[int(m)] = np.sort(_ns(sub["datetime"]))
        return d

    err_t = {e: by_machine(err[err["errorID"] == e][["machineID", "datetime"]]) for e in ERRORS}
    mnt_t = {c: by_machine(mnt[mnt["comp"] == c][["machineID", "datetime"]]) for c in COMPS}
    fail_t = by_machine(fail[["machineID", "datetime"]])

    n = len(tel)
    for e in ERRORS:
        out[f"err_{e}_last24h"] = 0.0
    for c in COMPS:
        out[f"hours_since_maint_{c}"] = np.nan
    label = np.zeros(n, dtype=np.int8)
    age = np.full(n, np.nan)

    age_map = dict(zip(mach["machineID"].astype(int), mach["age"].astype(float)))

    W = 24 * HOUR_NS
    gap_ns, win_end_ns = window_ns(gap_h, horizon_h)
    for m, pos in tel.groupby("machineID", sort=False).indices.items():
        m = int(m)
        t = _ns(tel["datetime"].iloc[pos])
        first = t[0]
        # error counts in (t-24h, t]
        for e in ERRORS:
            ev = err_t[e].get(m)
            if ev is None or len(ev) == 0:
                out.loc[out.index[pos], f"err_{e}_last24h"] = 0.0
                continue
            right = np.searchsorted(ev, t, side="right")
            left = np.searchsorted(ev, t - W, side="right")
            out.loc[out.index[pos], f"err_{e}_last24h"] = (right - left).astype(float)
        # hours since last maintenance per comp (or since first record)
        for c in COMPS:
            ev = mnt_t[c].get(m)
            if ev is None or len(ev) == 0:
                since = (t - first) / HOUR_NS
            else:
                idx = np.searchsorted(ev, t, side="right")
                last = np.where(idx > 0, ev[np.maximum(idx - 1, 0)], first)
                since = (t - last) / HOUR_NS
            out.loc[out.index[pos], f"hours_since_maint_{c}"] = since
        # label: any failure in (t+gap, t+gap+horizon]
        f = fail_t.get(m)
        if f is None or len(f) == 0:
            lab = np.zeros(len(t), dtype=np.int8)
        else:
            idx = np.searchsorted(f, t + gap_ns, side="right")   # first failure > t+gap
            lab = np.zeros(len(t), dtype=np.int8)
            valid = idx < len(f)
            lab[valid] = (f[idx[valid]] <= t[valid] + win_end_ns).astype(np.int8)
        label[pos] = lab
        age[pos] = age_map.get(m, np.nan)

    out["age"] = age
    out["label"] = label

    # drop the first 24h of every machine (incomplete rolling windows)
    first_t = out.groupby("machineID")["datetime"].transform("min")
    keep_first = (out["datetime"] - first_t) >= pd.Timedelta(hours=ROLLING_H)
    # drop the tail so every label window is complete
    global_max = out["datetime"].max()
    keep_last = out["datetime"] <= global_max - pd.Timedelta(hours=gap_h + horizon_h)
    out = out[keep_first & keep_last].reset_index(drop=True)

    # add model label
    model_map = dict(zip(mach["machineID"].astype(int), mach["model"]))
    out["model"] = out["machineID"].astype(int).map(model_map)
    out = out.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Split + local normalization + save
# --------------------------------------------------------------------------- #

def split_factory(sub: pd.DataFrame, gap_h: int = 0, horizon_h: int = 24) -> Dict[str, pd.DataFrame]:
    """Time split; (gap_h + horizon_h) of purge so no label window crosses splits."""
    margin = pd.Timedelta(hours=gap_h + horizon_h)
    t = sub["datetime"]
    train = sub[t < TRAIN_END]
    train = train[train["datetime"] + margin <= TRAIN_END]
    val = sub[(t >= TRAIN_END) & (t < VAL_END)]
    val = val[val["datetime"] + margin <= VAL_END]
    test = sub[t >= VAL_END]
    return {"train": train, "val": val, "test": test, "all": sub}


def _pack(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    return {
        "X": X,
        "y": df["label"].to_numpy(dtype=np.int64),
        "time": _ns(df["datetime"]),
        "machine": df["machineID"].to_numpy(dtype=np.int64),
    }


def build_and_save(setting: str = "next24h", verbose: bool = True) -> Dict[str, dict]:
    """Build, split, normalize and save every factory for a setting."""
    gap_h, horizon_h = SETTINGS[setting]["gap_h"], SETTINGS[setting]["horizon_h"]
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw(verbose=verbose)
    feats = build_features(raw, gap_h=gap_h, horizon_h=horizon_h)

    meta_all = {}
    for f in FACTORIES:
        sub = feats[feats["model"] == f].reset_index(drop=True)
        splits = split_factory(sub, gap_h=gap_h, horizon_h=horizon_h)

        # LOCAL normalization: mean/std from this factory's own train samples
        Xtr = splits["train"][FEATURE_NAMES].to_numpy(dtype=np.float64)
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std_safe = np.where(std < 1e-8, 1.0, std)

        arrays = {}
        for part, df in splits.items():
            p = _pack(df)
            p["X"] = ((p["X"].astype(np.float64) - mean) / std_safe).astype(np.float32)
            arrays[part] = p

        meta = {
            "setting": setting,
            "factory": f,
            "feature_names": FEATURE_NAMES,
            "mean": mean.tolist(),
            "std": std_safe.tolist(),
            "n_machines": int(sub["machineID"].nunique()),
            "counts": {part: int(len(arrays[part]["y"])) for part in arrays},
            "positives": {part: int(arrays[part]["y"].sum()) for part in arrays},
            "label_rate": {part: float(arrays[part]["y"].mean()) if len(arrays[part]["y"]) else 0.0
                           for part in arrays},
            "train_end": str(TRAIN_END), "val_end": str(VAL_END),
            "gap_h": gap_h, "horizon_h": horizon_h,
        }
        meta_all[f] = meta

        save = {"stats": meta}
        for part in arrays:
            for k, v in arrays[part].items():
                save[f"{part}_{k}"] = v
        out_dir = PROC_DIR / f / setting
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "arrays.npz", **save)

        with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        if verbose:
            print(f"[{setting}] {f}: machines={meta['n_machines']:3d}  "
                  + "  ".join(f"{p}={meta['counts'][p]}" for p in ["train", "val", "test", "all"])
                  + f"  train pos rate={meta['label_rate']['train']:.4f}")
    return meta_all


def load_processed(factory: str, setting: str = "next24h") -> Dict[str, np.ndarray]:
    with np.load(PROC_DIR / factory / setting / "arrays.npz", allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    stats = out.get("stats")
    if isinstance(stats, np.ndarray):
        stats = stats.item()
    out["stats"] = stats
    return out


def load_failures() -> pd.DataFrame:
    return load_raw(verbose=False)["failures"]


def count_failure_events(machines, t_min_ns: int, t_max_ns: int,
                         failures: Optional[pd.DataFrame] = None,
                         setting: str = "next24h") -> int:
    """Distinct (machine, failure) events whose datetime falls inside the label window
    of the last sample of the eval-set column."""
    gap_h, horizon_h = SETTINGS[setting]["gap_h"], SETTINGS[setting]["horizon_h"]
    failures = load_failures() if failures is None else failures
    sel = failures[failures["machineID"].isin(list(machines))]
    tt = _ns(sel["datetime"])
    lo = t_min_ns + gap_h * HOUR_NS
    hi = t_max_ns + (gap_h + horizon_h) * HOUR_NS
    mask = (tt > lo) & (tt <= hi)
    return int(sel.loc[mask, ["machineID", "datetime"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    set_seed(0)
    for setting in SETTINGS:
        meta = build_and_save(setting, verbose=True)
        print(f"\n[{setting}] Summary:")
        for f, m in meta.items():
            print(f"  {f}: label_rate(all)={m['label_rate']['all']:.4f}  "
                  f"label_rate(train)={m['label_rate']['train']:.4f}")
