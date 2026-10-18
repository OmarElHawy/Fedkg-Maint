"""
Stage 7: LLM-based explanations.

A single backend entry point, `generate(prompt, system=None)`, talks to a local
Ollama server (model configured in configs/llm_config.yaml). Every response is
cached in results/explanations/cache.jsonl (key = sha256 of model+system+prompt)
so reruns are free and an interrupted run resumes.

Predicted RUL comes from the final Federated learning model from Stage 4b
(FedProx, mu=0.01, seed 0) applied to the TEST windows (x125, as in training).
The true RUL is never placed in a prompt; it is only used for the stratified
evaluation selection and for reporting.

The rule-based checks (extract_claims / check_explanation) look for
unsupported entity claims, wrong numbers, wrong directions, invalid citations
and the insufficient-evidence behaviour. No LLM judge is used.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import requests

import torch

from src.models import LSTMRegressor
from src import kg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPL_DIR = RESULTS_DIR / "explanations"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CLIP_MAX = 125

EXPL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Config / prompts
# --------------------------------------------------------------------------- #


def load_llm_config(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "llm_config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Optional[Path] = None) -> dict:
    with open(path or (CONFIG_DIR / "prompts.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# LLM backend (single entry point) + JSONL cache
# --------------------------------------------------------------------------- #

_cache: Optional[Dict[str, str]] = None


def cache_path() -> Path:
    p = Path(load_llm_config()["cache_path"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def cache_key(model: str, prompt: str, system: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update((system or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _load_cache() -> Dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        p = cache_path()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                _cache[rec["key"]] = rec["response"]
    return _cache


def generate(prompt: str, system: Optional[str] = None) -> str:
    """
    Return a completion from the configured Ollama model.

    Cached by (model, system, prompt). Raises RuntimeError if Ollama is not
    reachable, so a failure is never silently faked.
    """
    cfg = load_llm_config()
    model = cfg["ollama"]["model"]
    gen = cfg["generation"]
    cache = _load_cache()
    key = cache_key(model, prompt, system)
    if key in cache:
        return cache[key]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": gen["temperature"],
            "seed": gen["seed"],
            "num_predict": gen["num_predict"],
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(
            cfg["ollama"]["base_url"].rstrip("/") + "/api/generate",
            json=payload, timeout=gen["timeout_s"],
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Ollama backend not reachable at {cfg['ollama']['base_url']}: {e}. "
            f"Start it with `ollama serve`; no outputs were produced."
        ) from e

    text = (resp.json().get("response") or "").strip()

    p = cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "model": model, "response": text}) + "\n")
    cache[key] = text
    return text


# --------------------------------------------------------------------------- #
# Predictions from the final FedProx model
# --------------------------------------------------------------------------- #


def load_predictions(
    checkpoint: Optional[str] = None,
    factory: Optional[str] = None,
    split: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Run the final FedProx global model on the TEST windows and return a frame:
    factory, split, index, engine, predicted_rul, risk_level, true_rul.

    predicted_rul = model output * 125 (unchanged from training). true_rul is
    only used for stratified evaluation selection / reporting, never in prompts.
    """
    cfg = load_llm_config()
    pred_cfg = cfg["predictions"]
    checkpoint = checkpoint or pred_cfg["checkpoint"]
    factory = factory or pred_cfg["factory"]
    split = split or pred_cfg["split"]
    clip_max = pred_cfg.get("clip_max", CLIP_MAX)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj

    model = LSTMRegressor(input_dim=14, hidden_size=64, num_layers=2, dropout=0.2)
    model.load_state_dict(state)
    model.to(device).eval()

    arrays = kg._load_arrays(factory)
    X = arrays[f"X_{split}"]
    y = arrays[f"y_{split}"]
    unit = arrays[f"unit_{split}"]

    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device)).cpu().numpy() * clip_max

    kcfg = kg.load_config()
    df = pd.DataFrame({
        "factory": factory,
        "split": split,
        "index": np.arange(len(y)),
        "engine": unit.astype(int),
        "predicted_rul": pred.astype(float),
        "true_rul": y.astype(float),
    })
    df["risk_level"] = [kg.risk_level(r, kcfg) for r in df["predicted_rul"]]
    return df


def predicted_rmse(df: pd.DataFrame, clip_max: int = CLIP_MAX) -> float:
    d = df["predicted_rul"].to_numpy() - np.minimum(df["true_rul"].to_numpy(), clip_max)
    return float(np.sqrt(np.mean(d ** 2)))


# --------------------------------------------------------------------------- #
# Prompt building + running
# --------------------------------------------------------------------------- #


def facts_for(row: pd.Series) -> List[str]:
    """Reference knowledge-graph facts for one engine (uses predicted RUL)."""
    return kg.get_facts(row["factory"], row["split"], int(row["index"]),
                        predicted_rul=float(row["predicted_rul"]))


def build_prompt(version: str, engine_info: dict,
                 facts: Optional[List[str]] = None) -> str:
    """
    Build the prompt for version 'A' (no KG) or 'B' (grounded with facts).
    Prompts live in configs/prompts.yaml and are frozen before the FD004 run.
    """
    prompts = load_prompts()
    version = version.upper()
    if version == "A":
        tmpl = prompts["version_a"]["template"]
        return tmpl.format(engine=engine_info["engine"],
                           rul=f"{engine_info['predicted_rul']:.0f}",
                           risk=engine_info["risk_level"])
    if version == "B":
        tmpl = prompts["version_b"]["template"]
        facts_block = "\n".join(facts or ["(none)"])
        return tmpl.format(engine=engine_info["engine"],
                           rul=f"{engine_info['predicted_rul']:.0f}",
                           risk=engine_info["risk_level"],
                           facts=facts_block)
    raise ValueError("version must be 'A' or 'B'")


def run_explanations(engines: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Generate one explanation per engine for the given version (cached).
    Returns a frame with the prompt, the explanation and the reference facts.
    """
    prompts = load_prompts()
    system = prompts.get("system")
    rows = []
    for _, row in engines.iterrows():
        facts = facts_for(row)
        prompt = build_prompt(
            version,
            {"engine": f"Engine_{int(row['engine'])}",
             "predicted_rul": float(row["predicted_rul"]),
             "risk_level": row["risk_level"]},
            facts,
        )
        text = generate(prompt, system=system)
        rows.append({
            "factory": row["factory"], "split": row["split"],
            "index": int(row["index"]), "engine": int(row["engine"]),
            "predicted_rul": float(row["predicted_rul"]),
            "risk_level": row["risk_level"], "true_rul": float(row["true_rul"]),
            "version": version.upper(), "prompt": prompt,
            "explanation": text, "ref_facts": facts,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rule-based checks
# --------------------------------------------------------------------------- #

_SCHEMA = kg.load_schema()
_KCFG = kg.load_config()
SENSORS = list(_SCHEMA["sensors"].keys())
COMPONENTS = list(_SCHEMA["components"])
FAILURE_MODES = list(_SCHEMA["failure_modes"].keys())

# sensor id -> short alias taken from the schema description ("T24, LPC ...")
SENSOR_ALIASES = {sid: meta["desc"].split(",")[0].strip()
                  for sid, meta in _SCHEMA["sensors"].items()}
ALIAS_TO_SENSOR = {a.lower(): s for s, a in SENSOR_ALIASES.items()}

_sensor_id_re = re.compile(r"\bs(\d{1,3})\b", re.IGNORECASE)
_rise_re = re.compile(
    r"\b(rising|increasing|increased|increases|rises|rise|upward|climbing|"
    r"higher|growing|elevated|up)\b", re.IGNORECASE)
_fall_re = re.compile(
    r"\b(falling|decreasing|decreased|decreases|drops|dropping|drop|declines|"
    r"declining|downward|lower|reduced|reduction|down)\b", re.IGNORECASE)
_rul_re = re.compile(
    r"(?:rul|remaining useful life|useful life|remaining life|life)\D{0,20}?(\d{1,3})",
    re.IGNORECASE)
_left_re = re.compile(
    r"(\d{1,3})\s*(?:cycles?|flights?|hours?)\s*(?:left|remaining|to go|of|until|before)",
    re.IGNORECASE)
_pred_re = re.compile(
    r"(?:predicted|predicts|prediction|estimate)\D{0,30}?(\d{1,3})\s*(?:cycles?)",
    re.IGNORECASE)
_risk_re = re.compile(
    r"risk(?:\s*level)?\s*(?:is|are|:|=)?\s*(high|medium|low)\b", re.IGNORECASE)
_risk_re2 = re.compile(r"\b(high|medium|low)\s+risk\b", re.IGNORECASE)
_std_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:std\b|standard deviations?)", re.IGNORECASE)
_cite_re = re.compile(r"\[(\d+)\]")

_ACTION_PATTERNS = {
    "Inspect compressor blades and clearances": [("compressor", "blade"), ("compressor", "clearance")],
    "Schedule HPC wash or replacement": [("wash",), ("hpc", "replacement"), ("hpc", "replace")],
    "Inspect fan blades": [("fan", "blade")],
    "Check fan balance and bearings": [("fan", "balance"), ("fan", "bearing"), ("balance",), ("bearing",)],
}
_GENERIC_SAFE = re.compile(
    r"\b(routine inspection|regular inspection|scheduled inspection|"
    r"monitor the engine|continue monitoring|routine maintenance|"
    r"monitoring|no immediate action|no action necessary)\b", re.IGNORECASE)
_INSUFFICIENT = re.compile(
    r"(not enough (?:evidence|information|data|detail)|insufficient "
    r"(?:evidence|information|data)|lack of (?:facts|evidence|information)|"
    r"no additional information|no information|no clear evidence|"
    r"cannot (?:determine|identify|conclude|recommend)|"
    r"unable to (?:determine|recommend|identify)|"
    r"not possible to (?:determine|recommend|identify)|"
    r"no specific (?:sensor|evidence|action)|no sensor deviates)", re.IGNORECASE)


def _normalise_fm(text: str) -> set:
    low = text.lower().replace("_", " ")
    found = set()
    for fm in FAILURE_MODES:
        if fm.lower().replace("_", " ") in low:
            found.add(fm)
    return found


def _detect_actions(text: str) -> set:
    low = text.lower()
    found = set()
    for action, patterns in _ACTION_PATTERNS.items():
        for pat in patterns:
            if all(kw in low for kw in pat):
                found.add(action)
                break
    return found


def extract_claims(text: str) -> Dict[str, list]:
    """
    Extract entity claims from free text: sensors, components, failure modes and
    actions. Sensors are any token 's<digits>' (so made-up ids like s99 are
    caught) plus the short names (T24, Ps30, ...).
    """
    sensors = []
    for m in _sensor_id_re.finditer(text):
        sensors.append("s" + m.group(1))
    low = text.lower()
    for alias, sid in ALIAS_TO_SENSOR.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            sensors.append(sid)
    # de-dup, keep order
    sensors = list(dict.fromkeys(sensors))

    components = [c for c in COMPONENTS
                  if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE)]
    failure_modes = sorted(_normalise_fm(text))
    actions = sorted(_detect_actions(text))
    generic = bool(_GENERIC_SAFE.search(text))
    return {"sensors": sensors, "components": components,
            "failure_modes": failure_modes, "actions": actions,
            "generic_action": generic}


def _ref_entities(ref_facts: List[str]) -> Dict:
    ref_text = "\n".join(ref_facts)
    ref_sensors = set(extract_claims(ref_text)["sensors"])
    ref_components = {c for c in COMPONENTS
                      if re.search(r"\b" + re.escape(c) + r"\b", ref_text, re.IGNORECASE)}
    ref_fms = _normalise_fm(ref_text)
    # a failure mode in the facts implies its component (e.g. HPC_degradation -> HPC)
    for fm in ref_fms:
        ref_components.add(_SCHEMA["failure_modes"][fm]["component"])
    ref_actions = _detect_actions(ref_text)

    ref_direction, ref_std = {}, {}
    for line in ref_facts:
        m = re.search(r"Sensor (s\d+)\s+is\s+(rising|falling)\s*\((\d+(?:\.\d+)?)\s*std",
                      line)
        if m:
            sid = m.group(1)
            ref_direction[sid] = 1 if m.group(2) == "rising" else -1
            ref_std[sid] = float(m.group(3))
    quiet = "no sensor deviates" in ref_text.lower()
    return {"sensors": ref_sensors, "components": ref_components,
            "failure_modes": ref_fms, "actions": ref_actions,
            "direction": ref_direction, "std": ref_std, "quiet": quiet,
            "text": ref_text}


def _sentence_split(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?\n])", text) if s.strip()]


def check_explanation(text: str, ref_facts: List[str], predicted_rul: float,
                      risk_level: str, version: str = "B") -> dict:
    """
    Rule-based check of one explanation against the reference facts + model output.
    Returns counts/flags; has_hallucination = unsupported claims or wrong
    number or wrong direction.
    """
    claims = extract_claims(text)
    ref = _ref_entities(ref_facts)

    unsupported = []

    def _sensor_supported(sid: str) -> bool:
        return sid in ref["sensors"]

    for sid in claims["sensors"]:
        if not _sensor_supported(sid):
            unsupported.append({"kind": "sensor", "value": sid})
    for c in claims["components"]:
        if c not in ref["components"]:
            unsupported.append({"kind": "component", "value": c})
    for fm in claims["failure_modes"]:
        if fm not in ref["failure_modes"]:
            unsupported.append({"kind": "failure_mode", "value": fm})
    for a in claims["actions"]:
        if a not in ref["actions"]:
            unsupported.append({"kind": "action", "value": a})

    n_claims = (len(claims["sensors"]) + len(claims["components"])
                + len(claims["failure_modes"]) + len(claims["actions"])
                + (1 if claims["generic_action"] else 0))

    # --- numbers ---
    reasons = []
    rr = float(round(predicted_rul))
    cand = ([int(x) for x in _rul_re.findall(text)]
            + [int(x) for x in _left_re.findall(text)]
            + [int(x) for x in _pred_re.findall(text)])
    number_ok = True
    for n in cand:
        if abs(n - rr) > 0.5:
            number_ok = False
            reasons.append(f"RUL {n} != predicted {rr}")

    if risk_level:
        for rx in (_risk_re, _risk_re2):
            for val in rx.findall(text):
                if val.lower() != risk_level.lower():
                    number_ok = False
                    reasons.append(f"risk {val} != {risk_level}")

    ref_std_vals = list(ref["std"].values())
    for s in _std_re.findall(text):
        v = float(s)
        if ref_std_vals and not any(abs(v - rv) <= 0.1 for rv in ref_std_vals):
            number_ok = False
            reasons.append(f"std {v} not in facts {ref_std_vals}")

    # --- directions ---
    direction_ok = True
    n_direction_claims = 0
    for sent in _sentence_split(text):
        sids = extract_claims(sent)["sensors"]
        if not sids:
            continue
        is_rise = bool(_rise_re.search(sent))
        is_fall = bool(_fall_re.search(sent))
        stated = 1 if (is_rise and not is_fall) else (-1 if (is_fall and not is_rise) else 0)
        if stated == 0:
            continue
        for sid in sids:
            if sid in ref["direction"]:
                n_direction_claims += 1
                if ref["direction"][sid] != stated:
                    direction_ok = False
                    reasons.append(f"sensor {sid} direction wrong")

    # --- citations (version B) ---
    cites = [int(x) for x in _cite_re.findall(text)]
    citation_valid_rate = float("nan")
    if cites:
        valid = sum(1 for c in cites if 1 <= c <= len(ref_facts))
        citation_valid_rate = valid / len(cites)

    # --- insufficient-evidence behaviour (quiet engines, version B) ---
    insufficient_evidence_handled = None
    if version.upper() == "B" and ref["quiet"]:
        said = bool(_INSUFFICIENT.search(text))
        named_sensor = len(claims["sensors"]) > 0
        insufficient_evidence_handled = bool(said and not named_sensor)

    has_hallucination = bool(len(unsupported) > 0 or not number_ok or not direction_ok)

    return {
        "n_claims": n_claims,
        "n_unsupported": len(unsupported),
        "unsupported": unsupported,
        "has_hallucination": has_hallucination,
        "number_ok": number_ok,
        "direction_ok": direction_ok,
        "n_direction_claims": n_direction_claims,
        "citation_valid_rate": citation_valid_rate,
        "n_citations": len(cites),
        "insufficient_evidence_handled": insufficient_evidence_handled,
        "quiet": ref["quiet"],
        "reasons": reasons,
    }


def check_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply check_explanation to every row of an explanations frame."""
    recs = []
    for _, r in df.iterrows():
        res = check_explanation(r["explanation"], r["ref_facts"],
                                r["predicted_rul"], r["risk_level"], r["version"])
        res["engine"] = int(r["engine"])
        res["version"] = r["version"]
        recs.append(res)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #

_FACTS = [
    "[1] Predicted RUL is 7 cycles -> risk level HIGH",
    "[2] HPC is monitored by sensor s11 (Ps30, HPC static pressure)",
    "[3] Sensor s11 is rising (4.4 std above healthy level)",
    "[4] A rising s11 indicates HPC_degradation",
    "[5] HPC_degradation is fixed by: Inspect compressor blades and clearances; "
    "Schedule HPC wash or replacement",
]
_QUIET_FACTS = [
    "[1] Predicted RUL is 120 cycles -> risk level LOW",
    "[2] No sensor deviates strongly from its healthy level",
]


def _self_tests() -> None:
    # 1. a made-up sensor name must be flagged unsupported
    r = check_explanation("The s99 sensor is failing and the engine is at risk.",
                          _FACTS, 7, "HIGH")
    assert any(u["kind"] == "sensor" and u["value"] == "s99" for u in r["unsupported"])
    assert r["has_hallucination"]

    # 2. a correct grounded sentence must pass
    good = ("The predicted RUL is 7 cycles (risk: HIGH) [1]. Sensor s11 is "
            "rising [3], and a rising s11 indicates HPC_degradation [4]. "
            "Inspect compressor blades and clearances [5].")
    r = check_explanation(good, _FACTS, 7, "HIGH", version="B")
    assert r["n_unsupported"] == 0, r["unsupported"]
    assert r["number_ok"] and r["direction_ok"]
    assert not r["has_hallucination"]
    assert r["citation_valid_rate"] == 1.0

    # 3. wrong RUL number is caught
    r = check_explanation("The predicted RUL is 50 cycles.", _FACTS, 7, "HIGH")
    assert not r["number_ok"]

    # 4. wrong risk level is caught
    r = check_explanation("The risk level is LOW.", _FACTS, 7, "HIGH")
    assert not r["number_ok"]

    # 5. wrong direction is caught
    r = check_explanation("Sensor s11 is falling.", _FACTS, 7, "HIGH")
    assert not r["direction_ok"]

    # 6. component not in facts is unsupported
    r = check_explanation("The Fan is degrading.", _FACTS, 7, "HIGH")
    assert any(u["kind"] == "component" and u["value"] == "Fan" for u in r["unsupported"])

    # 7. invalid citation is caught
    r = check_explanation("Sensor s11 is rising [9].", _FACTS, 7, "HIGH", version="B")
    assert r["citation_valid_rate"] == 0.0

    # 8. quiet engine + sensor named -> NOT handled
    r = check_explanation("Sensor s11 is rising, likely HPC_degradation.",
                          _QUIET_FACTS, 120, "LOW", version="B")
    assert r["quiet"] and r["insufficient_evidence_handled"] is False

    # 9. quiet engine + honest answer -> handled
    r = check_explanation("There is not enough evidence; recommend a routine inspection.",
                          _QUIET_FACTS, 120, "LOW", version="B")
    assert r["insufficient_evidence_handled"] is True
    assert not r["has_hallucination"]

    # 10. generic safe action counts as supported
    r = check_explanation("Predicted RUL is 7 cycles. Recommend a routine inspection.",
                          _FACTS, 7, "HIGH")
    assert not any(u["kind"] == "action" for u in r["unsupported"])
    print("explain.py: all self-tests passed")


if __name__ == "__main__":
    _self_tests()
