"""
Stage 6: per-factory maintenance knowledge graphs.

Every factory builds its OWN graph from its OWN data only. All factories share
one schema (ontology) loaded from configs/kg_schema.yaml; the numerical parts
(healthy baseline, degradation direction) are computed per factory from that
factory's TRAIN windows only. Nothing from FD004 is ever used to build or tune
the graphs of FD001-FD003.

The graph is static (components, sensors, symptoms, failure modes, actions,
machines). Per-window symptoms are computed on demand from the z-score of the
last N sensor values against the factory's healthy baseline.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
KG_DIR = PROJECT_ROOT / "results" / "kg"
CONFIG_DIR = PROJECT_ROOT / "configs"

FACTORIES = ["FD001", "FD002", "FD003", "FD004"]

PRIMITIVE = (str, bool, int, float)


def load_schema(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "kg_schema.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "kg_config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_arrays(factory: str) -> Dict[str, np.ndarray]:
    """Read only this factory's processed arrays."""
    with np.load(DATA_DIR / factory / "arrays.npz") as data:
        return {k: data[k] for k in data.files}


def _action_id(failure_mode: str, i: int) -> str:
    return f"action_{failure_mode}_{i}"


def _symptom_id(sensor: str, direction: int) -> str:
    return f"{sensor}_{'rising' if direction > 0 else 'falling'}"


def build_factory_kg(factory: str, schema: Optional[dict] = None) -> nx.MultiDiGraph:
    """
    Build one factory's knowledge graph from that factory's data only.

    Returns a networkx.MultiDiGraph whose node attribute "type" is one of the
    shared schema node types and whose edge attributes include "relation"
    (a shared schema relation) and "source" ("domain_knowledge" or "data").
    """
    schema = schema or load_schema()
    if factory not in schema["factory_failure_modes"]:
        raise ValueError(f"unknown factory {factory!r}")
    if not (DATA_DIR / factory / "arrays.npz").exists():
        raise FileNotFoundError(f"no processed data for {factory}")

    arrays = _load_arrays(factory)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    sensors = list(schema["sensors"].keys())
    sensor_meta = schema["sensors"]

    G = nx.MultiDiGraph()
    G.graph["factory"] = factory

    # --- Machines: engine ids found in this factory's data (all splits) ---
    units = sorted(set(arrays["unit_train"].tolist())
                   | set(arrays["unit_val"].tolist())
                   | set(arrays["unit_test"].tolist()))
    for u in units:
        G.add_node(f"Engine_{int(u)}", type="Machine", label=f"Engine_{int(u)}")

    # --- Components ---
    for c in schema["components"]:
        G.add_node(c, type="Component", label=c)

    # --- Sensors (shared schema, domain knowledge) ---
    for s in sensors:
        G.add_node(s, type="Sensor", label=s,
                   desc=sensor_meta[s]["desc"], component=sensor_meta[s]["component"])

    # --- Failure modes present in this factory ---
    fm_list = list(schema["factory_failure_modes"][factory])
    for fm in fm_list:
        G.add_node(fm, type="FailureMode", label=fm,
                   component=schema["failure_modes"][fm]["component"])
        # FailureMode -affects-> Component
        G.add_edge(fm, schema["failure_modes"][fm]["component"],
                   relation="affects", source="domain_knowledge")
        # FailureMode -fixed_by-> Action
        for i, act in enumerate(schema["failure_modes"][fm]["actions"]):
            aid = _action_id(fm, i)
            G.add_node(aid, type="Action", label=act)
            G.add_edge(fm, aid, relation="fixed_by", source="domain_knowledge")

    # --- Machine -has_component-> Component (domain knowledge) ---
    for u in units:
        for c in schema["components"]:
            G.add_edge(f"Engine_{int(u)}", c,
                       relation="has_component", source="domain_knowledge")

    # --- Component -monitored_by-> Sensor (domain knowledge) ---
    for s in sensors:
        G.add_edge(sensor_meta[s]["component"], s,
                   relation="monitored_by", source="domain_knowledge")

    # --- Data-derived healthy baseline (TRAIN windows with y == clip only) ---
    healthy_mask = y_train == 125
    last_train = X_train[:, -1, :]                     # last time step per window
    healthy = last_train[healthy_mask]
    healthy_mean = healthy.mean(axis=0)
    healthy_std = healthy.std(axis=0)

    # --- Data-derived degradation direction ---
    degraded_mask = y_train <= 15
    degraded = last_train[degraded_mask]
    diff = degraded.mean(axis=0) - healthy_mean
    direction = np.sign(diff).astype(int)

    # --- Symptom nodes from data: one per sensor with a clear direction ---
    sec_links = schema.get("secondary_failure_links", {}) or {}
    for i, s in enumerate(sensors):
        if healthy_std[i] < 1e-8:
            continue  # constant sensor -> no measurable symptom
        d = int(direction[i])
        if d == 0:
            continue
        sid = _symptom_id(s, d)
        G.add_node(sid, type="Symptom", label=sid, sensor=s, direction=d)
        G.add_edge(s, sid, relation="shows", source="data")
        # primary link: failure mode of the component this sensor monitors
        for fm in fm_list:
            if schema["failure_modes"][fm]["component"] == sensor_meta[s]["component"]:
                G.add_edge(sid, fm, relation="indicates", source="domain_knowledge")
        # secondary link (only where the schema justifies it)
        for fm in sec_links.get(s, []):
            if fm in fm_list:
                G.add_edge(sid, fm, relation="indicates_secondary",
                           source="domain_knowledge")

    # --- Keep the local statistics on the graph so get_facts can use them ---
    G.graph["sensors"] = json.dumps(sensors)
    G.graph["healthy_mean"] = json.dumps([float(x) for x in healthy_mean])
    G.graph["healthy_std"] = json.dumps([float(x) for x in healthy_std])
    G.graph["degradation_direction"] = json.dumps(
        {s: int(direction[i]) for i, s in enumerate(sensors)})
    G.graph["n_healthy_windows"] = int(healthy_mask.sum())
    G.graph["n_degraded_windows"] = int(degraded_mask.sum())
    return G


def _prim(v):
    """Make an attribute value safe for GraphML."""
    if isinstance(v, PRIMITIVE):
        return v
    if v is None:
        return ""
    return json.dumps(v)


def save_kg(G: nx.MultiDiGraph, factory: str, out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Save the graph as GraphML (attributes made GraphML-safe) and as a pickle."""
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

    graphml_path = out_dir / f"{factory}.graphml"
    pkl_path = out_dir / f"{factory}.pkl"
    nx.write_graphml(G2, graphml_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    return graphml_path, pkl_path


def load_kg(factory: str, out_dir: Optional[Path] = None,
            fmt: str = "pkl") -> nx.MultiDiGraph:
    """Load a saved graph. fmt is 'pkl' (default) or 'graphml'."""
    out_dir = Path(out_dir or KG_DIR)
    if fmt == "pkl":
        with open(out_dir / f"{factory}.pkl", "rb") as f:
            return pickle.load(f)
    if fmt == "graphml":
        return nx.read_graphml(out_dir / f"{factory}.graphml", force_multigraph=True)
    raise ValueError("fmt must be 'pkl' or 'graphml'")


# --------------------------------------------------------------------------- #
# Per-window symptom computation
# --------------------------------------------------------------------------- #

def healthy_stats_from_graph(G: nx.MultiDiGraph):
    sensors = json.loads(G.graph["sensors"])
    mean = np.array(json.loads(G.graph["healthy_mean"]), dtype=float)
    std = np.array(json.loads(G.graph["healthy_std"]), dtype=float)
    direction = {k: int(v) for k, v in json.loads(G.graph["degradation_direction"]).items()}
    return sensors, mean, std, direction


def sensor_zscores(G: nx.MultiDiGraph, window: np.ndarray, config: dict) -> Dict[str, float]:
    """z = (mean of last N steps - healthy mean) / healthy std, per sensor."""
    sensors, mean, std, _ = healthy_stats_from_graph(G)
    n = config["symptom"]["last_n_steps"]
    last = window[-n:, :].mean(axis=0)
    z = {}
    for i, s in enumerate(sensors):
        if std[i] < 1e-8:
            continue
        z[s] = float((last[i] - mean[i]) / std[i])
    return z


def active_symptoms(G: nx.MultiDiGraph, window: np.ndarray,
                    config: dict) -> List[dict]:
    """Active symptoms for one window, sorted by |z| descending."""
    _, _, _, direction = healthy_stats_from_graph(G)
    thr = config["symptom"]["z_threshold"]
    z = sensor_zscores(G, window, config)
    out = []
    for s, zv in z.items():
        d = direction.get(s, 0)
        if d == 0:
            continue
        if (d > 0 and zv > thr) or (d < 0 and zv < -thr):
            out.append({"sensor": s, "z": zv, "direction": d})
    out.sort(key=lambda a: abs(a["z"]), reverse=True)
    return out


def window_active_flags(G: nx.MultiDiGraph, X: np.ndarray, config: dict) -> np.ndarray:
    """Boolean array: does each window have at least one active symptom?"""
    sensors, mean, std, direction = healthy_stats_from_graph(G)
    n = config["symptom"]["last_n_steps"]
    thr = config["symptom"]["z_threshold"]
    last = X[:, -n:, :].mean(axis=1)          # (N, 14)
    flags = np.zeros(len(X), dtype=bool)
    for i, s in enumerate(sensors):
        if std[i] < 1e-8:
            continue
        d = direction.get(s, 0)
        if d == 0:
            continue
        z = (last[:, i] - mean[i]) / std[i]
        flags |= (z > thr) if d > 0 else (z < -thr)
    return flags


def risk_level(rul: float, config: dict) -> str:
    if rul <= config["risk"]["high_max_rul"]:
        return "HIGH"
    if rul <= config["risk"]["medium_max_rul"]:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #

def _failure_modes_of_symptom(G: nx.MultiDiGraph, symptom_id: str):
    res = []
    for _, t, d in G.out_edges(symptom_id, data=True):
        if d.get("relation") in ("indicates", "indicates_secondary"):
            res.append((t, d.get("relation")))
    return res


def get_facts(factory: str, split: str, index: int, predicted_rul: Optional[float] = None,
              top_k: Optional[int] = None, schema: Optional[dict] = None,
              config: Optional[dict] = None) -> List[str]:
    """
    Return numbered short English facts for one window.

    split in {"train", "val", "test"}; index indexes that split. The engine id
    comes from unit_<split>. If predicted_rul is given, the first fact states the
    predicted RUL and the risk level. The top_k sensors by |z| among the ACTIVE
    symptoms are used; if none is active a single honest fact is returned.
    """
    config = config or load_config()
    top_k = top_k or config["facts"]["top_k"]
    G = load_kg(factory)
    arrays = _load_arrays(factory)
    X, y, unit = arrays[f"X_{split}"], arrays[f"y_{split}"], arrays[f"unit_{split}"]
    window = X[index]
    engine_id = int(unit[index])

    facts: List[str] = []
    if predicted_rul is not None:
        facts.append(f"Predicted RUL is {predicted_rul:.0f} cycles -> "
                     f"risk level {risk_level(predicted_rul, config)}")

    actives = active_symptoms(G, window, config)[:top_k]
    if not actives:
        facts.append("No sensor deviates strongly from its healthy level")
        return [f"[{i + 1}] {t}" for i, t in enumerate(facts)]

    top_sensor = actives[0]["sensor"]
    comp = G.nodes[top_sensor].get("component")
    facts.append(f"Engine {engine_id} has component {comp}")

    for a in actives:
        s, d = a["sensor"], a["direction"]
        word = "rising" if d > 0 else "falling"
        facts.append(f"{G.nodes[s]['component']} is monitored by sensor {s} "
                     f"({G.nodes[s]['desc']})")
        side = "above" if a["z"] >= 0 else "below"
        facts.append(f"Sensor {s} is {word} ({abs(a['z']):.1f} std {side} healthy level)")

    seen_fm: List[str] = []
    for a in actives:
        s, d = a["sensor"], a["direction"]
        word = "rising" if d > 0 else "falling"
        for fm, rel in _failure_modes_of_symptom(G, _symptom_id(s, d)):
            if fm in seen_fm:
                continue
            seen_fm.append(fm)
            tag = "" if rel == "indicates" else " (secondary evidence)"
            facts.append(f"A {word} {s} indicates {fm}{tag}")

    for fm in seen_fm:
        acts = [G.nodes[t]["label"] for _, t, dd in G.out_edges(fm, data=True)
                if dd.get("relation") == "fixed_by"]
        acts = list(dict.fromkeys(acts))
        facts.append(f"{fm} is fixed by: " + "; ".join(acts))

    return [f"[{i + 1}] {t}" for i, t in enumerate(facts)]


def build_all(schema: Optional[dict] = None) -> Dict[str, nx.MultiDiGraph]:
    """Build and save all four factory graphs (each from its own data only)."""
    schema = schema or load_schema()
    graphs = {}
    for f in FACTORIES:
        G = build_factory_kg(f, schema=schema)
        save_kg(G, f)
        graphs[f] = G
    return graphs


if __name__ == "__main__":
    graphs = build_all()
    for f in FACTORIES:
        g = graphs[f]
        print(f"{f}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
              f"healthy windows {g.graph['n_healthy_windows']}")
    for line in get_facts("FD001", "test", 0, predicted_rul=15):
        print(line)
