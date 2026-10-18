# FedKG-Maint: Federated Predictive Maintenance with a Shared Knowledge Graph

**License: MIT (see LICENSE)**

A full vertical pipeline for **federated predictive maintenance with optional
differential privacy and knowledge-graph-grounded LLM explanations**. Three
simulated "factories" (NASA C-MAPSS turbofan fleets FD001/FD002/FD003) train an
LSTM RUL model cooperatively **without sharing raw telemetry**; a fourth,
unseen fleet (FD004, 248 engines) is used only for evaluation. On that unseen
factory FedProx (30 rounds, 5-seed average) reaches **FD004 RMSE
17.905 ± 0.963 cycles**. This shows no detectable difference from the centralized
baseline (17.851 ± 0.950, Holm p = 0.947; with only 5 seeds a difference of about one RMSE point cannot be ruled out), and it is clearly better than FedAvg
(20.240 ± 0.644, Holm p = 0.034). Client-level DP with Rényi accounting costs
accuracy as expected (ε ≈ 1.75e4 → RMSE 18.591; ε ≈ 1.76e3 → RMSE 22.904).
Explanations generated from the same model hallucinate **far less when
grounded in a shared knowledge graph** (B: 0.0–3.3% vs A: 21.7–30.0% per 60
explanations, McNemar p = 0.0074/0.0001). The same stack transfers to the
Azure Predictive Maintenance dataset (26 features, binary failure-window
prediction): federated models beat local-only on the held-out machine model
(next-24h PR-AUC 0.986 vs 0.945). Every number above is recomputed read-only
from `results/final/` by `python src/evaluate.py`, with an exact
reproducibility gate (regression `val30` reproduced with absolute difference
0.0).

## Architecture

```mermaid
flowchart LR
    F1[Factory FD001] --> P1[local preprocessing + local KG]
    F2[Factory FD002] --> P2[local preprocessing + local KG]
    F3[Factory FD003] --> P3[local preprocessing + local KG]
    P1 --> R{{round r: server aggregates clipped deltas}}
    P2 --> R
    P3 --> R
    R --> D{{optional DP: clip C=0.32 + Gaussian noise σ}}
    D --> G[global FedProx model · 30 rounds]
    G --> Y[per-engine prediction · unseen FD004]
    Y --> K[KG facts · current active symptoms + actions]
    Y --> E[rule-based evaluation · RMSE / engine MAE]
    K --> L[LLM explanation · versions A (no KG) / B (with KG)]
    L --> C[hallucination checker · supported / unsupported claims]
```

## Key results

All values from `results/final/` (mean ± std over 5 seeds; A = version without
the KG, B = version with the KG). Claim labels follow
`report/findings_and_limitations.md`.

| C-MAPSS FD004 RMSE (cycles) | mean ± std |
|------------------------------|------------:|
| Local FD001 / FD002 / FD003  | 22.911 ± 0.785 / 25.766 ± 2.008 / 20.799 ± 0.530 |
| Centralized (pooled)          | 17.851 ± 0.950 |
| FedAvg                       | 20.240 ± 0.644 |
| **FedProx (μ=0.01)**          | **17.905 ± 0.963** |
| DP σ=0.03 (ε=17543)          | 18.591 ± 0.703 |
| DP σ=0.1 (ε=1763)            | 22.904 ± 4.203 |

- FedProx ≈ Centralized (Holm p = 0.947) and FedProx < FedAvg (Holm p = 0.034);
  FedAvg < Centralized significantly (Holm p = 0.026) — **supported by the tests**.
- Engine-level (248 engines, mean absolute error): Centralized 12.89,
  FedProx 14.14, FedAvg 14.40, DP0.03 14.85, DP0.1 18.17; FedProx vs FedAvg
  not significant (Wilcoxon p = 0.86) — RMSE vs JMAE disagree by design
  (`findings_and_limitations.md` limitation 7) — **supported by the tests**.
- Azure held-out-test PR-AUC (next24h): LocalOnly 0.945, Centralized 0.974,
  FedAvg 0.986, FedProx 0.975; the harder **gap24_48h** task: 0.181 / 0.274 /
  0.287 / 0.287 — **supported by the tests** (FL & Centralized beat LocalOnly,
  Holm p < 0.05; neither FL variant beats Centralized, Holm p > 0.16; with only 5
  seeds the bootstrap CIs are narrow and optimistic, so the Holm-corrected
  paired t-test is the primary evidence).
- Explanations hallucination rate (60 pairs/sample, rule-based checker):
  **A (no KG) 0.217–0.300, B (with KG) 0.000–0.033**; McNemar p = 0.0074
  and p = 0.0001 — **supported by the tests**.
- Communication cost over 30 rounds: C-MAPSS LSTM 0.431 MB/client/round
  (12.9 MB total); Azure MLP 0.143 MB/client/round (4.3 MB total).

![FD004 RMSE by method](results/final/cmapss_fd004_rmse.png)

*Fig. 1 — `results/final/cmapss_fd004_rmse.png`.*

![Azure PR-AUC](results/final/azure_main.png)

*Fig. 2 — `results/final/azure_main.png`.*

![Hallucination rate A vs B](results/final/explanation_hallucination.png)

*Fig. 3 — `results/final/explanation_hallucination.png`.*

### Grounded vs ungrounded explanation (engine 26, FD004 test)

Same engine, same model output (`predicted RUL 81, true RUL 92, risk LOW`),
same LLM. Facts available to version B: `[1] Predicted RUL is 81 cycles →
risk level LOW`, `[2] No sensor deviates strongly from its healthy level`.
Source rows: `results/explanations_repeat/seed43/explanations_A.csv` /
`explanations_B.csv` (also in `results/demo/fd004_demo_data.csv`).

| Version A (without KG) | Version B (with KG) |
|---|---|
| *"…we recommend a routine inspection and cleaning of the engine's compressor blades after the next 50 cycles."* — a **compressor-blade cleaning** not supported by any fact → 1 unsupported claim, checker verdict: **unsupported**. | *"The prediction of 81 cycles to RUL is based on the model's output, which cites a risk level of LOW ([1]). This risk level is further supported by the fact that no sensor deviates strongly from its healthy level ([2]). …There is not enough evidence to determine the cause of the predicted RUL or to recommend any specific action. A routine inspection … is recommended."* — all entities in facts, citation in range, verdict: **supported**. |

*Single-pair comparison is descriptive only; the aggregate is the McNemar test above.*

## Install

Requires Python 3.10+, and PyTorch installed **separately** (platform/CUDA
specific — it is deliberately not pinned in `requirements.txt`).

```powershell
git clone <this-repo> FedKG-Maint
cd FedKG-Maint
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows; `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
# then install torch per https://pytorch.org/get-started/locally/ , e.g.:
pip install torch                # CPU build; for CUDA use the torch.org command
```

## Run

> Raw telemetry is **not** included in the repo (see *Datasets*). To reproduce
> local training from scratch, place the C-MAPSS zip in `data/raw/` and the
> Azure CSVs in `data/raw/azure/` first — otherwise all checkpoints and results
> are already in `results/`.

```powershell
# 1) Optional: rebuild the knowledge graphs (one per factory) — read-only steps
python src/kg.py

# 2) READ-ONLY final evaluation — recomputes everything in results/final/
#    (including stats_azure.csv) from saved checkpoints/logs.
python src/evaluate.py

# 2b) Optional: check human evaluation sheet ratings
python scripts/summarize_human_eval.py

# 3) Demo app (no GPU, no Ollama, no LLM calls) — shows per-engine FD004
#    predictions, KG facts, explanations A/B, and the results tables/figures.
streamlit run app/streamlit_app.py

# 4) Reproduce / extend training and DP runs
python src/run_stage9.py                        # heavy runner (GPU), all sections
python src/run_stage9.py --only regression      # or a single section
```

**Explanations (optional, needs a local LLM).** The LLM backend is Ollama
running the small instruct model `llama3.2:latest` (`configs/llm_config.yaml` —
the only model installed on the development machine; nothing was downloaded).
If you want to re-run the explanation stage:

```powershell
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve
python src/run_stage9.py --only explanations   # needs the model serving at :11434
```

All explanation runs are cached (`results/explanations/cache.jsonl`), so reruns
are free and resume-safe.

## Reproduce

See **`docs/REPRODUCE.md`** for full reproduction details.

### The Two Master Notebooks
The pipeline is consolidated into two self-contained master notebooks under `notebooks/` (paired with percent-format scripts in `notebooks/source/` via `jupytext`):
1. **`notebooks/01_FedKG_Maint_CMAPSS.ipynb`**: C-MAPSS federated learning pipeline (FD001–FD003 clients, FD004 held-out; local/centralized baselines; FedAvg vs FedProx μ tuning; DP accuracy/privacy trade-off; per-factory KGs; grounded LLM explanations A vs B; regression gate).
2. **`notebooks/02_FedKG_Maint_Azure_and_Final_Evaluation.ipynb`**: Azure Predictive Maintenance transfer (binary FL, negative control and ablations, Schema v2 KG with ErrorCode precedence edges and active fact validation) and final multi-seed statistical evaluation (Holm correction, communication cost, requirement checklist R1–R17).

### Notebook Execution Flags
The first cell of each notebook defines three flags:
- `RERUN_HEAVY = False` *(default)*: loads saved results from `results/final/` and reproduces all tables and figures. Runs **without raw data, without a GPU, and without Ollama** in ~12 seconds.
- `RERUN_HEAVY = True`: retrains from raw data (estimated runtime printed at top; outputs written to `results_rerun/` unless `OVERWRITE = True`). Requires raw datasets in `data/raw/` or configured via `FEDKG_DATA_DIR`.
- `OVERWRITE = False`: safety guard preventing overwriting existing results when `RERUN_HEAVY = True`.
- `SMOKE = True`: runs a tiny real training using the clean functions in `src/` (C-MAPSS: 2 federated FedProx rounds on FD001–FD003 evaluated on FD004, plus one DP round with σ=0.03; Azure: 2 FedAvg rounds on model1–3 evaluated on model4; explanations: 3 engines from cache). It needs the data (`FEDKG_DATA_DIR` or processed data) and a CPU or GPU, takes ~1–2 minutes, writes strictly to `results_smoke/`, and its scores (e.g. FD004 RMSE ~88.7, model4 PR-AUC ~0.008) are intentionally poor and not comparable to the saved results.

### External Data Configuration (`FEDKG_DATA_DIR`)
Raw telemetry data is omitted from the repository to keep the footprint small (~9 MB). If you wish to retrain from scratch (`RERUN_HEAVY = True`), set the environment variable pointing to your raw data directory:
```powershell
$env:FEDKG_DATA_DIR = "path\to\raw_data"
```

### Interactive Demo App
The Streamlit web demo requires neither GPU nor Ollama:
```powershell
streamlit run app/streamlit_app.py
```

## Folder structure

```
configs/           YAML configs: FL, DP, KG, LLM, Azure (single source of truth)
src/               Pipeline modules (data, models, federated, privacy, kg, explain, evaluate...)
notebooks/         Two master Jupyter notebooks (01_FedKG_Maint_CMAPSS.ipynb, 02_FedKG_Maint_Azure_and_Final_Evaluation.ipynb)
notebooks/source/  Jupytext percent scripts paired with notebooks
results/
  final/           All reported numbers + figures (regenerated by src/evaluate.py, incl. stats_azure.csv)
  explanations*/   LLM outputs, cached generations, per-seed repeats
  demo/            Precomputed demo table for the Streamlit app
app/               Streamlit demo (GPU-free, Ollama-free)
scripts/           Helper scripts (make_demo_data.py, summarize_human_eval.py)
report/            Project reports: FedKG-Maint_Report.md, findings_and_limitations.md, privacy_discussion.md
docs/              Reproduction guide (docs/REPRODUCE.md)
models/            Single small demo checkpoint (fedprox_fd_seed0.pt)
```

## Limitations (in five lines)

1. Only **5 seeds** for the paired RMSE tests — 4 degrees of freedom; engine-level
   (248-engine) tests and repeated curves mitigate but do not remove this.
2. "Factories" are slices of public benchmarks, not real sites — claims are
   benchmark-level.
3. Client-level DP with 3 clients admits only a weak guarantee: ε is 1.75e4 at
   usable accuracy; there is **no** tested point with both small ε and full
   accuracy.
4. Explanations are judged by a **rule-based factual checker**, not a
   semantic-quality judge, and generation is non-deterministic (Ollama).
5. Engine-level JMAE and window-level RMSE disagree (FedProx ties FedAvg on
   the former but wins on the latter); both are reported.

## Datasets

- **NASA Turbofan (C-MAPSS)**: Saxena, Goebel, Simon and Eklund, *Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, IEEE
  PHM 2008 — dataset page:
  <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
  (`.zip` copy is in the repo root; raw data is not committed to git).
- **Azure Predictive Maintenance (Kaggle)**:
  <https://www.kaggle.com/datasets/vinculum-in-datascience/azure-predictive-maintenance>
  (raw CSVs go to `data/raw/azure/`; not committed).

Both datasets are public benchmark data. Check the license and terms on each dataset page before reuse.

> Torch note: `requirements.txt` intentionally omits `torch` because the wheel
> differs per platform/CUDA — install it first per pytorch.org. `streamlit` is
> used only by the demo app.