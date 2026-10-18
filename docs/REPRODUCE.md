# Reproducing FedKG-Maint

This guide lists the exact order in which the pipeline was built and can be
re-run. Raw data is **not** committed; checkpoints and all result tables/figures
**are** present, so the last two steps (`evaluate.py`, the demo) run without any
data or GPU.

**License: MIT (see LICENSE)**

## Machine context

- Windows + PowerShell, Python 3.10+. PyTorch is installed separately
  (not in `requirements.txt`); the training stages want a CUDA GPU.
- The LLM stage needs Ollama serving `llama3.2:latest` on `localhost:11434`
  (for generation only; all generated texts are already cached in
  `results/explanations/cache.jsonl`).

## Notebooks & Reproduction Pipeline

The experimental pipeline is consolidated into two comprehensive notebooks under `notebooks/` (paired with scripts in `notebooks/source/` via `jupytext`):

| Notebook | Focus | Key Stages & Checks Covered |
|---|---|---|
| `notebooks/01_FedKG_Maint_CMAPSS.ipynb` | C-MAPSS RUL | Data & factories (FD001–3 clients, FD004 held-out), preprocessing, baselines, FedAvg/FedProx (client-val μ tuning), DP privacy-accuracy trade-off, factory KGs, grounded LLM explanations A vs B, regression gate (16.5021) |
| `notebooks/02_FedKG_Maint_Azure_and_Final_Evaluation.ipynb` | Azure Transfer & Final Eval | Azure PdM data, leakage checks (negative control, ablations), LocalOnly/Centralized/FedAvg/FedProx for both settings (`next24h` and `gap24_48h`), Schema v2 KG (ErrorCode nodes, lift, active fact validation), 5-seed statistical evaluation with Holm correction, communication cost, requirements checklist R1–R17 |

### Notebook Execution Flags (First Cell)
Both notebooks contain three flags controlling execution:
- **`RERUN_HEAVY = False`** *(default)*: fast reproduction (~12 s) loading precomputed results from `results/final/` to rebuild all tables, charts, and assertions. Requires **no raw data, no GPU, and no local LLM/Ollama**.
- **`RERUN_HEAVY = True`**: retrains models from raw datasets. Writes to `results_rerun/` unless `OVERWRITE = True` is also set.
- **`OVERWRITE = False`**: safety guard preventing overwriting results when `RERUN_HEAVY = True`.
- **`SMOKE = True`**: runs a tiny real training using the clean functions in `src/` (C-MAPSS: 2 federated FedProx rounds on FD001–FD003 evaluated on FD004, plus one DP round with σ=0.03; Azure: 2 FedAvg rounds on model1–3 evaluated on model4; explanations: 3 engines from cache). It requires the data (`FEDKG_DATA_DIR` or processed data) and a CPU or GPU, takes ~1–2 minutes, writes strictly to `results_smoke/`, and its scores are intentionally poor and not comparable to the saved results.

### Setting `FEDKG_DATA_DIR`
When training from scratch (`RERUN_HEAVY = True`), set the environment variable pointing to the raw datasets folder (containing `cmapss/` and `azure/`):
```powershell
$env:FEDKG_DATA_DIR = "path\to\raw_data"
```

To execute a notebook from the command line:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/01_FedKG_Maint_CMAPSS.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/02_FedKG_Maint_Azure_and_Final_Evaluation.ipynb
```

## Stage-9 heavy runner (recorded gates)

`src/run_stage9.py` re-runs the heavy gates and the repeats; each section is
independently selectable with `--only`:

```powershell
python src/run_stage9.py --only regression      # reproduces val30 = 16.502083720307283
python src/run_stage9.py --only task0           # Azure labeler cross-check (2659 positives)
python src/run_stage9.py --only task1_cmapss    # seeds 3-4 runs for the 5-seed comparisons
python src/run_stage9.py --only backfill        # DP sigma{0.03,0.1} seeds 3-4
python src/run_stage9.py --only task2           # number-of-factories ablation
python src/run_stage9.py --only task3           # engine-level inference + rounds-within-5% 
python src/run_stage9.py --only dp_curves       # full DP sigma sweep curves
python src/run_stage9.py --only azure           # Azure 5-seed runs + ablation + convergence
python src/run_stage9.py --only explanations    # explanation repeats for samples 43/44
```

Recorded gates:

- **Regression**: `expected` = `reproduced` = `16.502083720307283`, `abs_diff =
  0.0`, `passed = true`, **runtime 161.1 s** (`results/final/regression.json`).
- **Task 0**: 2,659 positive train windows in all four (setting × machines)
  configurations; all four cross-checks `match = TRUE`
  (`results/final/task0.csv`).

## Final read-only evaluation (no data, no GPU required)

This is the single command that regenerates everything in `results/final/`
(checkpoints, raw logs and saved explanation CSVs already exist):

```powershell
python src/evaluate.py          # full run (also: python src/evaluate.py --only cmapss)
```

It writes/overwrites only `results/final/*` (including `stats_azure.csv`, `stats_cmapss.csv`, etc.) and `results/checkpoints/` copies;
it never touches training code outputs. On a CPU-only machine it takes a few
minutes (engine-level scoring runs on the 248 FD004 engines).

To inspect human evaluation ratings:

```powershell
python scripts/summarize_human_eval.py  # prints 'not scored yet' if unfilled
```

## Notebook reproduction

Every `notebooks/NN_x.py` is a jupytext source of the matching `.ipynb`. To
re-execute the notebooks:

```powershell
pip install jupytext nbformat
jupytext --sync --pipe jupyter nbconvert --to notebook --execute notebooks/
```

## Demo data (reusable, GPU-free)

```powershell
python scripts/make_demo_data.py     # regenerates results/demo/fd004_demo_data.csv
streamlit run app/streamlit_app.py   # opens the app
```

`make_demo_data.py` reads only saved artifacts and verifies the demo
predictions against the final FedProx seed-0 checkpoint (gate printed at the
end).

**Note:** `configs/llm_config.yaml` refers to `results/checkpoints/FedProx_4b_seed0.pt`, which is not included in the upload (checkpoints are large); the demo and the notebooks in saved mode use precomputed predictions, and `RERUN_HEAVY = True` regenerates the checkpoint.