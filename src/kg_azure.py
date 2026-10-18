"""
Stage 6 / Azure extension: per-factory maintenance knowledge graphs for Azure PdM (Schema v2).

Every machine model (factory) builds its OWN local graph from ITS OWN rows only,
restricted strictly to the TRAINING period (before 2015-09-01). The held-out factory
(model4) builds its graph from its own training-period rows only, and the graph is
used strictly for explanation facts, never for training or hyperparameter tuning.

Schema v2 Node types:
  - Machine (the factory's machines)
  - Component (comp1, comp2, comp3, comp4)
  - Sensor (volt, rotate, pressure, vibration)
  - ErrorCode (error1, error2, error3, error4, error5)
  - FailureMode (comp1_failure, comp2_failure, comp3_failure, comp4_failure)
  - Symptom (sensor deviations, e.g. rotate_falling, pressure_rising, vibration_rising)
  - Action (Replace compX, Inspect compX)

Relations:
  - Machine -has_component-> Component
  - FailureMode -affects-> Component
  - FailureMode -fixed_by-> Action
  - ErrorCode -precedes-> FailureMode (data-derived: lift >= 2.0, min_events >= 5 in 24h window)
  - Sensor -shows-> Symptom (data-derived)
  - Symptom -indicates-> FailureMode (data-derived: |z| > 2 in >= 50-60% of pre-failure windows)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "azure"
KG_DIR = PROJECT_ROOT / "results" / "kg"
CONFIG_DIR = PROJECT_ROOT / "configs"

FACTORIES = ["model1", "model2", "model3", "model4"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
FAILURE_MODES = [f"{c}_failure" for c in COMPONENTS]

TRAIN_END = pd.Timestamp("2015-09-01")
PRIMITIVE = (str, bool, int, float)


def load_schema(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "kg_schema.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_azure_config(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "kg_azure_config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_train_data(model_name: str) -> Dict[str, pd.DataFrame]:
    """
    Load raw Azure tables filtered STRICTLY to model_name and datetime < TRAIN_END.
    Ensures complete isolation: no other factory's data or post-train data is accessed.
    """
    mach = pd.read_csv(RAW_DIR / "PdM_machines.csv")
    model_machines = set(mach[mach["model"] == model_name]["machineID"].tolist())
    if not model_machines:
        raise ValueError(f"Unknown factory/model: {model_name}")

    tel = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    tel = tel[(tel["machineID"].isin(model_machines)) & (tel["datetime"] < TRAIN_END)]
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    err = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    err = err[(err["machineID"].isin(model_machines)) & (err["datetime"] < TRAIN_END)]

    mnt = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    mnt = mnt[(mnt["machineID"].isin(model_machines)) & (mnt["datetime"] < TRAIN_END)]

    fail = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    fail = fail[(fail["machineID"].isin(model_machines)) & (fail["datetime"] < TRAIN_END)]

    return {
        "machines": mach[mach["model"] == model_name],
        "telemetry": tel,
        "errors": err,
        "maintenance": mnt,
        "failures": fail,
    }


def build_azure_kg(model_name: str, config: Optional[dict] = None) -> nx.MultiDiGraph:
    """
    Build one Azure factory's knowledge graph using ONLY its own training rows.
    """
    cfg = config or load_azure_config()
    schema = load_schema()
    raw = _load_train_data(model_name)

    tel = raw["telemetry"].copy()
    err = raw["errors"]
    mnt = raw["maintenance"]
    fail = raw["failures"]
    mach = raw["machines"]

    G = nx.MultiDiGraph()
    G.graph["factory"] = model_name
    G.graph["dataset"] = "Azure Predictive Maintenance"
    G.graph["schema_version"] = 2
    G.graph["train_end"] = str(TRAIN_END)

    # 1. Machines
    machine_ids = sorted(mach["machineID"].tolist())
    for mid in machine_ids:
        G.add_node(f"Machine_{mid}", type="Machine", label=f"Machine_{mid}", machine_id=mid)

    # 2. Components
    for comp in COMPONENTS:
        G.add_node(comp, type="Component", label=comp)

    # 3. Sensors
    for s in SENSORS:
        G.add_node(s, type="Sensor", label=s)

    # 4. ErrorCodes
    for e in ERRORS:
        G.add_node(e, type="ErrorCode", label=e)

    # 5. FailureModes
    for comp in COMPONENTS:
        fm = f"{comp}_failure"
        G.add_node(fm, type="FailureMode", label=fm, component=comp)
        # Structural edge: FailureMode -affects-> Component
        G.add_edge(fm, comp, relation="affects", source="domain_knowledge")

    # 6. Machine -has_component-> Component (all machines have comp1..comp4)
    for mid in machine_ids:
        for comp in COMPONENTS:
            G.add_edge(f"Machine_{mid}", comp, relation="has_component", source="domain_knowledge")

    # 7. Actions: Inspect compX (domain) + Replace compX (data-derived from maintenance)
    for comp in COMPONENTS:
        fm = f"{comp}_failure"
        # Domain inspect action
        act_inspect = f"Inspect_{comp}"
        G.add_node(act_inspect, type="Action", label=f"Inspect {comp}")
        G.add_edge(fm, act_inspect, relation="fixed_by", source="domain_knowledge")

        # Check maintenance replacements following failures in this factory
        c_fails = fail[fail["failure"] == comp]
        replacements = 0
        for _, frow in c_fails.iterrows():
            sub_m = mnt[(mnt["machineID"] == frow["machineID"]) &
                        (mnt["datetime"] >= frow["datetime"]) &
                        (mnt["datetime"] <= frow["datetime"] + pd.Timedelta(hours=24)) &
                        (mnt["comp"] == comp)]
            if len(sub_m) > 0:
                replacements += 1

        act_replace = f"Replace_{comp}"
        G.add_node(act_replace, type="Action", label=f"Replace {comp}")
        G.add_edge(fm, act_replace, relation="fixed_by", source="data",
                   supporting_replacements=replacements, total_failures=len(c_fails))

    # 8. Data-derived: ErrorCode -precedes-> FailureMode
    # lift >= min_lift, min_events >= 5
    err_cfg = cfg["error_precedes"]
    win_h = err_cfg["window_hours"]
    min_lift = err_cfg["min_lift"]
    min_events = err_cfg["min_events"]

    # Calculate baseline probability of each error occurring in any 24h window
    total_hours = len(tel)
    base_error_rate = {}
    for e in ERRORS:
        e_sub = err[err["errorID"] == e]
        if len(e_sub) == 0 or total_hours == 0:
            base_error_rate[e] = 0.0
            continue
        # count hours with error e in past 24h
        has_e_count = 0
        for mid, m_tel in tel.groupby("machineID"):
            m_e = e_sub[e_sub["machineID"] == mid]["datetime"].sort_values().values
            if len(m_e) == 0:
                continue
            t_vals = m_tel["datetime"].sort_values().values
            idx_r = np.searchsorted(m_e, t_vals, side="right")
            idx_l = np.searchsorted(m_e, t_vals - np.timedelta64(win_h, "h"), side="right")
            has_e_count += int(((idx_r - idx_l) > 0).sum())
        base_error_rate[e] = has_e_count / max(1, total_hours)

    G.graph["base_error_rates"] = json.dumps(base_error_rate)

    # Correlate errors in 24h before each failure mode
    for comp in COMPONENTS:
        fm = f"{comp}_failure"
        c_fails = fail[fail["failure"] == comp]
        n_fails = len(c_fails)
        if n_fails == 0:
            continue

        for e in ERRORS:
            # count failures preceded by error e within win_h
            e_sub = err[err["errorID"] == e]
            supp_count = 0
            for _, frow in c_fails.iterrows():
                f_time = frow["datetime"]
                mid = frow["machineID"]
                pre_e = e_sub[(e_sub["machineID"] == mid) &
                              (e_sub["datetime"] >= f_time - pd.Timedelta(hours=win_h)) &
                              (e_sub["datetime"] <= f_time)]
                if len(pre_e) > 0:
                    supp_count += 1

            p_cond = supp_count / n_fails
            p_base = base_error_rate.get(e, 0.0)
            lift = (p_cond / p_base) if p_base > 0 else 0.0

            if lift >= min_lift and supp_count >= min_events:
                G.add_edge(e, fm, relation="precedes", source="data",
                           lift=float(lift), supporting_events=int(supp_count),
                           total_failures=int(n_fails), p_cond=float(p_cond), p_base=float(p_base))

    # 9. Data-derived: Sensor -shows-> Symptom -indicates-> FailureMode
    # Compute rolling 3h mean vs healthy baseline (no failure in next 72h)
    sym_cfg = cfg["sensor_symptom"]
    z_thr = sym_cfg["z_threshold"]
    healthy_margin = sym_cfg["healthy_margin_hours"]
    cons_ratio = sym_cfg.get("consistency_ratio", 0.50)

    for s in SENSORS:
        tel[f"{s}_roll3"] = tel.groupby("machineID")[s].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)

    # Determine healthy baseline windows
    fail_times = {m: sub["datetime"].values for m, sub in fail.groupby("machineID")}
    is_healthy = np.ones(len(tel), dtype=bool)
    for mid, group in tel.groupby("machineID"):
        f_times = fail_times.get(mid)
        if f_times is None or len(f_times) == 0:
            continue
        t_arr = group["datetime"].values
        idx_r = np.searchsorted(f_times, t_arr + np.timedelta64(healthy_margin, "h"), side="right")
        idx_l = np.searchsorted(f_times, t_arr, side="right")
        has_fail_next = (idx_r - idx_l) > 0
        is_healthy[group.index] = ~has_fail_next

    healthy_tel = tel[is_healthy]
    h_means = {s: float(healthy_tel[f"{s}_roll3"].mean()) for s in SENSORS}
    h_stds = {s: float(healthy_tel[f"{s}_roll3"].std()) for s in SENSORS}

    G.graph["healthy_means"] = json.dumps(h_means)
    G.graph["healthy_stds"] = json.dumps(h_stds)

    for comp in COMPONENTS:
        fm = f"{comp}_failure"
        c_fails = fail[fail["failure"] == comp]
        if len(c_fails) == 0:
            continue

        pre_idx = []
        for _, frow in c_fails.iterrows():
            mid = frow["machineID"]
            ftime = frow["datetime"]
            sub_win = tel[(tel["machineID"] == mid) &
                          (tel["datetime"] >= ftime - pd.Timedelta(hours=24)) &
                          (tel["datetime"] <= ftime)]
            pre_idx.extend(sub_win.index.tolist())

        pre_df = tel.loc[list(set(pre_idx))].dropna(subset=[f"{s}_roll3" for s in SENSORS])
        if len(pre_df) == 0:
            continue

        for s in SENSORS:
            z_vals = (pre_df[f"{s}_roll3"] - h_means[s]) / max(h_stds[s], 1e-6)
            pct_pos = float((z_vals > z_thr).mean())
            pct_neg = float((z_vals < -z_thr).mean())
            mean_z = float(z_vals.mean())

            direction = None
            ratio = 0.0
            if pct_pos >= cons_ratio:
                direction = "rising"
                ratio = pct_pos
            elif pct_neg >= cons_ratio:
                direction = "falling"
                ratio = pct_neg

            if direction is not None:
                sym_id = f"{s}_{direction}"
                if not G.has_node(sym_id):
                    G.add_node(sym_id, type="Symptom", label=sym_id, sensor=s, direction=direction)
                # Sensor -shows-> Symptom
                G.add_edge(s, sym_id, relation="shows", source="data")
                # Symptom -indicates-> FailureMode
                G.add_edge(sym_id, fm, relation="indicates", source="data",
                           mean_z=mean_z, consistency_ratio=ratio, supporting_windows=len(pre_df))

    return G


def _prim(v):
    if isinstance(v, PRIMITIVE):
        return v
    if v is None:
        return ""
    return json.dumps(v)


def save_azure_kg(G: nx.MultiDiGraph, model_name: str, out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    out_dir = Path(out_dir or KG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    G2 = G.copy()
    for k, v in list(G2.graph.items()):
        G2.graph[k] = _prim(v)
    for _, data in G2.nodes(data=True):
        for k, v in list(data.items()):
            data[k] = _prim(v)
    for _, _, _, data in G2.edges(data=True, keys=True):
        for k, v in list(data.items()):
            data[k] = _prim(v)

    graphml_path = out_dir / f"azure_{model_name}.graphml"
    pkl_path = out_dir / f"azure_{model_name}.pkl"
    nx.write_graphml(G2, graphml_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    return graphml_path, pkl_path


def load_azure_kg(model_name: str, out_dir: Optional[Path] = None, fmt: str = "pkl") -> nx.MultiDiGraph:
    out_dir = Path(out_dir or KG_DIR)
    if fmt == "pkl":
        with open(out_dir / f"azure_{model_name}.pkl", "rb") as f:
            return pickle.load(f)
    if fmt == "graphml":
        return nx.read_graphml(out_dir / f"azure_{model_name}.graphml", force_multigraph=True)
    raise ValueError("fmt must be 'pkl' or 'graphml'")


def build_all_azure(out_dir: Optional[Path] = None) -> Dict[str, nx.MultiDiGraph]:
    graphs = {}
    for m in FACTORIES:
        G = build_azure_kg(m)
        save_azure_kg(G, m, out_dir=out_dir)
        graphs[m] = G
    return graphs


def get_facts_azure(model_name: str, machine_id: int,
                     timestamp: Union[str, pd.Timestamp],
                     top_k: int = 5,
                     graph: Optional[nx.MultiDiGraph] = None,
                     raw_dir: Optional[Path] = None) -> List[str]:
    """
    Return numbered short English facts for one Azure machine at one time,
    using only data up to that timestamp.
    """
    ts = pd.Timestamp(timestamp)
    t_start_24h = ts - pd.Timedelta(hours=24)
    rdir = Path(raw_dir or RAW_DIR)

    G = graph or load_azure_kg(model_name)
    h_means = json.loads(G.graph["healthy_means"])
    h_stds = json.loads(G.graph["healthy_stds"])

    # Load machine events strictly up to ts
    err = pd.read_csv(rdir / "PdM_errors.csv", parse_dates=["datetime"])
    err = err[(err["machineID"] == machine_id) & (err["datetime"] >= t_start_24h) & (err["datetime"] <= ts)]

    tel = pd.read_csv(rdir / "PdM_telemetry.csv", parse_dates=["datetime"])
    # 3 hours of telemetry up to ts for rolling mean
    tel = tel[(tel["machineID"] == machine_id) & (tel["datetime"] >= ts - pd.Timedelta(hours=3)) & (tel["datetime"] <= ts)]
    tel = tel.sort_values("datetime")

    mnt = pd.read_csv(rdir / "PdM_maint.csv", parse_dates=["datetime"])
    mnt = mnt[(mnt["machineID"] == machine_id) & (mnt["datetime"] <= ts)]

    facts = []
    implicated_failure_modes = set()

    # 1. Active error codes in last 24h
    active_errors = err["errorID"].unique().tolist()
    for e in sorted(active_errors):
        # find what it precedes in this factory's graph
        precedes_fms = []
        for _, v, d in G.out_edges(e, data=True):
            if d.get("relation") == "precedes":
                precedes_fms.append((v, d.get("lift", 1.0)))
                implicated_failure_modes.add(v)

        if precedes_fms:
            fm_desc = ", ".join(f"{fm} (lift {lift:.1f}x)" for fm, lift in precedes_fms)
            facts.append(f"Active {e} in past 24h historically precedes: {fm_desc}")
        else:
            facts.append(f"Active {e} observed in past 24h")

    # 2. Active sensor deviations (|z| > 2)
    if len(tel) > 0:
        for s in SENSORS:
            val3 = tel[s].mean()
            mean = h_means[s]
            std = h_stds[s]
            z = (val3 - mean) / max(std, 1e-6)
            if abs(z) >= 2.0:
                direction = "rising" if z > 0 else "falling"
                sym_id = f"{s}_{direction}"
                # find what it indicates
                indicated_fms = []
                if G.has_node(sym_id):
                    for _, v, d in G.out_edges(sym_id, data=True):
                        if d.get("relation") == "indicates":
                            indicated_fms.append(v)
                            implicated_failure_modes.add(v)
                side = "above" if z > 0 else "below"
                if indicated_fms:
                    facts.append(f"Sensor {s} is {direction} ({abs(z):.1f} std {side} healthy baseline), indicating: {', '.join(indicated_fms)}")
                else:
                    facts.append(f"Sensor {s} is {direction} ({abs(z):.1f} std {side} healthy baseline)")

    # If neither error codes nor sensor deviations stand out, return honest fallback fact
    if len(active_errors) == 0 and len(implicated_failure_modes) == 0 and not any("Sensor" in f for f in facts):
        return ["[1] No error code or sensor deviation stands out"]

    # 3. Hours since last maintenance for components
    maint_hours = {}
    for comp in COMPONENTS:
        sub_m = mnt[mnt["comp"] == comp]
        if len(sub_m) > 0:
            last_dt = sub_m["datetime"].max()
            hrs = (ts - last_dt).total_seconds() / 3600.0
            maint_hours[comp] = hrs
        else:
            maint_hours[comp] = None

    # Highlight components with longest operating time or those implicated
    for comp in COMPONENTS:
        hrs = maint_hours.get(comp)
        fm = f"{comp}_failure"
        if hrs is not None and (hrs >= 1500 or fm in implicated_failure_modes):
            facts.append(f"{comp} has operated for {hrs:.0f} hours since last maintenance")

    # 4. Actions linked to implicated failure modes
    for fm in sorted(implicated_failure_modes):
        actions = []
        for _, act, d in G.out_edges(fm, data=True):
            if d.get("relation") == "fixed_by":
                actions.append(G.nodes[act].get("label", act))
        if actions:
            facts.append(f"{fm} is addressed by: {'; '.join(dict.fromkeys(actions))}")

    if not facts:
        return ["[1] No error code or sensor deviation stands out"]

    # Limit to top_k facts
    selected = facts[:top_k]
    return [f"[{i + 1}] {f}" for i, f in enumerate(selected)]


if __name__ == "__main__":
    graphs = build_all_azure()
    for m in FACTORIES:
        g = graphs[m]
        print(f"{m}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
