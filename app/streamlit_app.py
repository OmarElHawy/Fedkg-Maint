"""FedKG-Maint demo app (GPU-free, Ollama-free, no LLM calls).

Reads precomputed artifacts only:
  - results/demo/fd004_demo_data.csv   per-engine predictions + KG facts + A/B explanations
  - results/final/*.csv and *.png      reported numbers and figures (regenerate with
                                       `python src/evaluate.py` if needed)

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DEMO_CSV = ROOT / "results" / "demo" / "fd004_demo_data.csv"
FINAL = ROOT / "results" / "final"

# Edit this to show a "GitHub repository" link under the title. Empty = no link shown.
GITHUB_URL = "https://github.com/OmarElHawy/fedkg-maint"

st.set_page_config(page_title="FedKG-Maint demo", layout="wide")
st.title("FedKG-Maint — federated maintenance demo")
if GITHUB_URL:
    st.markdown(f"[GitHub repository]({GITHUB_URL})")
st.caption(
    "Precomputed results only; no GPU, no Ollama, no LLM calls. "
    "All values are the ones in results/final/ (recomputed by `python src/evaluate.py`)."
)

_SYMPTOM_RE = re.compile(r"^\[(\d+)\] Sensor s\d+ is (rising|falling)", re.MULTILINE)


@st.cache_data
def load_demo() -> pd.DataFrame:
    df = pd.read_csv(DEMO_CSV).copy()
    df["n_symptoms"] = df["kg_facts"].map(lambda g: len(_SYMPTOM_RE.findall(g)))
    return df


@st.cache_data
def load_final_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FINAL / name)


def default_engine(demo: pd.DataFrame) -> int:
    cand = demo[(demo["risk_level"].isin(["HIGH", "MEDIUM"])) & (demo["n_symptoms"] >= 2)]
    cand = cand.sort_values(["n_symptoms", "engine"], ascending=[False, True])
    return int(cand.iloc[0]["engine"]) if len(cand) else int(demo.iloc[0]["engine"])


def verdict_text(verdict: str) -> str:
    return "✅ supported" if verdict == "supported" else "❌ unsupported claims found"


def checker_summary(n_unsupported: int, n_claims: int) -> str:
    if n_claims <= 0:
        return "Checker found no claims it could check."
    unsup = "claim" if n_unsupported == 1 else "claims"
    checked = "claim" if n_claims == 1 else "claims"
    return f"Checker found {n_unsupported} unsupported {unsup} out of {n_claims} {checked} it could check."


def risk_filter_options(demo: pd.DataFrame) -> dict:
    return {
        "All": demo,
        "Near failure (HIGH)": demo[demo["risk_level"] == "HIGH"],
        "Medium": demo[demo["risk_level"] == "MEDIUM"],
        "Healthy-looking (no symptoms)": demo[demo["n_symptoms"] == 0],
    }


tab1, tab2 = st.tabs(["Explanations A vs B", "Results"])

with tab1:
    demo = load_demo()

    st.info(
        "**How to read this page**\n"
        "- The engine comes from **FD004**, a factory the model never saw during training.\n"
        "- The model predicts the **remaining useful life (RUL)** in cycles.\n"
        "- **Version A** gets only the prediction; **version B** also gets facts from the knowledge graph.\n"
        "- The **checker** looks for claims that the facts do not support."
    )

    st.subheader("FD004 engines — FedProx (seed 0) predictions with grounded vs ungrounded explanations")

    filter_opts = risk_filter_options(demo)
    filter_name = st.selectbox("Filter", list(filter_opts), index=0)
    sub = filter_opts[filter_name]

    default = default_engine(demo)
    engines = sorted(sub["engine"].unique().tolist())
    init_index = engines.index(default) if default in engines else 0
    sel = st.selectbox("Engine (FD004 test)", engines, index=init_index)

    row = demo[demo["engine"] == int(sel)].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted RUL", f"{row['predicted_rul']:.0f} cycles")
    m2.metric("True RUL", f"{row['true_rul']:.0f} cycles")
    m3.metric("Risk level", row["risk_level"])

    pred = int(round(float(row["predicted_rul"])))
    truth = int(round(float(row["true_rul"])))
    off = abs(pred - truth)
    st.markdown(f"**Predicted {pred}, true {truth} (off by {off} cycles)**")

    st.markdown(f"**Version A check:** {verdict_text(row['check_A'])}")
    st.markdown(f"**Version B check:** {verdict_text(row['check_B'])}")
    st.caption(
        "How the checker works: it compares every sensor, component and action named "
        "in the explanation against the knowledge-graph facts of that engine, and flags "
        "anything the facts do not support."
    )

    if int(row["n_symptoms"]) == 0:
        st.info("No sensor deviates strongly. A good explanation should say the evidence is not enough.")

    with st.expander("Knowledge-graph facts used by version B", expanded=True):
        st.text(row["kg_facts"])

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### Version A — without KG")
        st.markdown(checker_summary(int(row["n_unsupported_claims_A"]), int(row["n_claims_A"])))
        st.text(row["explanation_A"])
    with cB:
        st.markdown("### Version B — with KG")
        st.markdown(checker_summary(int(row["n_unsupported_claims_B"]), int(row["n_claims_B"])))
        st.text(row["explanation_B"])

    expl = load_final_csv("explanations_overall.csv")
    a_rates = expl.loc[expl["version"] == "A", "hallucination_rate"]
    b_rates = expl.loc[expl["version"] == "B", "hallucination_rate"]
    stats = load_final_csv("explanation_stats.csv")
    ps = stats["p_mcnemar"].dropna().tolist()

    st.info(
        f"**Supported by the tests:** over three samples of 60 FD004 cases each, the "
        f"rule-based checker flags {100 * a_rates.min():.1f}%–{100 * a_rates.max():.1f}% "
        f"of version-A explanations but only 0.0%–{100 * b_rates.max():.1f}% of version-B "
        f"explanations (McNemar p = {ps[0]:.4f} / {ps[1]:.4f}). A single engine shown above is "
        f"only an example (descriptive only); the rates are the tested aggregates."
    )

with tab2:
    st.subheader("Main results (results/final/)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**C-MAPSS FD004 RMSE (5 seeds)**")
        st.dataframe(
            load_final_csv("cmapss_main.csv")[["setup", "fd004_rmse_mean", "fd004_rmse_std", "n_seeds"]],
            hide_index=True, use_container_width=True,
        )
        st.markdown("**DP / epsilon**")
        st.dataframe(load_final_csv("dp_overview.csv"), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Azure held-out PR-AUC**")
        az = load_final_csv("azure_main.csv")
        st.dataframe(
            az[az["eval_set"] == "heldout_test"][["setting", "setup", "pr_auc_mean", "pr_auc_std"]],
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Hallucination rate (A vs B)**")
        expl = load_final_csv("explanations_overall.csv")[["sample", "version", "hallucination_rate"]]
        expl = expl.pivot(index="sample", columns="version", values="hallucination_rate")
        st.dataframe(expl, use_container_width=True)

    st.markdown("**Figures**")
    for img, cap in [
        (FINAL / "cmapss_fd004_rmse.png", "FD004 RMSE by method (5 seeds)"),
        (FINAL / "azure_main.png", "Azure PR-AUC by method and setting"),
        (FINAL / "explanation_hallucination.png", "Hallucination rate, A vs B (bootstrap 95% CI)"),
        (FINAL / "dp_epsilon_tradeoff.png", "Privacy–accuracy trade-off"),
    ]:
        if img.exists():
            st.image(str(img), caption=cap, use_container_width=True)

    st.caption("Regenerate these tables and figures with `python src/evaluate.py` (read-only).")