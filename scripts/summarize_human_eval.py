"""Summarize human evaluation ratings for explanation versions A and B.

Reads results/explanations/human_eval_sheet.csv and human_eval_key.csv.
If ratings are present, writes results/final/human_eval_summary.csv.
If ratings are not yet filled, prints 'not scored yet' and writes nothing.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLAIN_DIR = PROJECT_ROOT / "results" / "explanations"
FINAL_DIR = PROJECT_ROOT / "results" / "final"

SHEET_PATH = EXPLAIN_DIR / "human_eval_sheet.csv"
KEY_PATH = EXPLAIN_DIR / "human_eval_key.csv"
OUT_PATH = FINAL_DIR / "human_eval_summary.csv"

RATING_COLS = ["clarity", "correctness", "useful_action"]


def main():
    if not SHEET_PATH.exists() or not KEY_PATH.exists():
        print("not scored yet (files missing)")
        return

    sheet = pd.read_csv(SHEET_PATH)
    key = pd.read_csv(KEY_PATH)

    # Check if any rating columns have non-null numeric values
    has_ratings = False
    for col in RATING_COLS:
        if col in sheet.columns and sheet[col].notna().sum() > 0:
            has_ratings = True
            break

    if not has_ratings:
        print("not scored yet")
        return

    # Merge ratings with blinded key
    merged = pd.merge(sheet, key, on="id", how="inner")

    # Compute summary stats per version
    summary_rows = []
    for col in RATING_COLS:
        if col not in merged.columns:
            continue
        valid = merged.dropna(subset=[col])
        for ver in sorted(valid["version"].unique()):
            sub = valid[valid["version"] == ver]
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) > 0:
                summary_rows.append({
                    "metric": col,
                    "version": ver,
                    "n": len(vals),
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0,
                })

    df_summary = pd.DataFrame(summary_rows)

    # If paired comparison is possible (same engine evaluated in both A and B)
    if "engine" in merged.columns and len(merged) >= 4:
        piv = merged.pivot_table(index="engine", columns="version", values=RATING_COLS)
        # Add paired stats if available
        pass

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUT_PATH, index=False)
    print(f"Summary written to {OUT_PATH}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
