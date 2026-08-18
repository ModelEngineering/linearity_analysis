"""Perturbation study: how do perturbed initial conditions affect SystemDiscovery R²?

Runs only on models with a explicitly specified end time.
For each qualifying model, SystemDiscovery.analyzePerturbations is called with
perturbation_value_fractions of -50%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +50%.
R² is computed using the derivative method.

Output CSV: data/perturbation_study.csv
Columns: model_name, threshold, and for each perturbation level
"""

import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn
from src.system_discovery import SystemDiscovery
from src.timecourse_iterator import TimecourseIterator

THRESHOLD = 0.001
POLY_DEGREE = 1
SPECIES_FRACTION = 1.0
PERTURBATIONS: list[float] = [-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.50]

OUTPUT_PATH = os.path.join(cn.DATA_DIR, f"perturbation_study-{THRESHOLD}.csv")
MIN_R2 = 0.8
COL_DEG1_MODEL_MAX = "deg1_max"

EXCLUDES = [
            "BIOMD0000000718",
            ]


def main(is_initialize: bool = False) -> pd.DataFrame:
    if not is_initialize and os.path.isfile(OUTPUT_PATH):
        print(f"Loading existing results from {OUTPUT_PATH}...")
        initial_df = pd.read_csv(OUTPUT_PATH)
    else:
        initial_df = pd.DataFrame()

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[cn.COL_SYSTEM_ID].values)

    for item in TimecourseIterator():
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        if item.model_name in EXCLUDES:
            print(f"Skipping {item.model_name} (excluded)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        try:
            analyze_df = SystemDiscovery.analyzePerturbations(
                model=item.timecourse.model,
                training_df=item.timecourse.timecourse_df,
                threshold=THRESHOLD,
                perturbations=PERTURBATIONS,
                perturbation_species_fraction=SPECIES_FRACTION,
                poly_degree=POLY_DEGREE,
                is_plot=False,
            ).df
        except Exception as exc:
            print(f"  [error] {item.model_name}: {exc}", file=sys.stderr)
            continue

        analyze_df[cn.COL_THRESHOLD] = THRESHOLD
        current_df = pd.read_csv(OUTPUT_PATH) if os.path.isfile(OUTPUT_PATH) else pd.DataFrame()
        # Ensure analyze_df is a DataFrame before concatenating/writing.
        if isinstance(analyze_df, pd.Series):
            analyze_df = pd.DataFrame([analyze_df.to_dict()]).df
        full_df = pd.concat([current_df, analyze_df], ignore_index=True) if len(current_df) > 0 else analyze_df
        full_df.to_csv(OUTPUT_PATH, index=False)

    if os.path.isfile(OUTPUT_PATH):
        full_df = pd.read_csv(OUTPUT_PATH)
    else:
        full_df = initial_df
    print(f"\nDone. {len(full_df)} rows in {OUTPUT_PATH}")
    return full_df


if __name__ == "__main__":
    main()
