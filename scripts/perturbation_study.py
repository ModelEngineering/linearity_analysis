"""Perturbation study: how do perturbed initial conditions affect SystemDiscovery R²?

Runs only on models whose deg1_min >= 0.9 in evaluate_monomial_models-0.01.csv.
For each qualifying model, SystemDiscovery.analyzePerturbations is called with
perturbation_value_fractions of -50%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +50%.
R² is computed using the derivative method.

Output CSV: data/perturbation_study.csv
Columns: model_name, threshold, r2_-50, r2_-20, r2_-10, r2_-05, r2_0,
         r2_+05, r2_+10, r2_+20, r2_+50
"""

import os
import sys

import pandas as pd

import src.constants as cn
from src.system_discovery import SystemDiscovery
from src.timecourse_iterator import TimecourseIterator

THRESHOLD = 0.01
POLY_DEGREE = 1
SPECIES_FRACTION = 1.0
PERTURBATIONS: list[float] = [-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.50]

SOURCE_PATH = os.path.join(cn.DATA_DIR, "evaluate_monomial_models-0.01.csv")
OUTPUT_PATH = os.path.join(cn.DATA_DIR, "perturbation_study.csv")
MIN_DEG1 = 0.9


def main(is_initialize: bool = False) -> pd.DataFrame:
    source_df = pd.read_csv(SOURCE_PATH)
    model_names: set[str] = set(
        source_df.loc[source_df["deg1_min"] >= MIN_DEG1, cn.COL_MODEL_NAME]
    )
    print(f"Models with deg1_min >= {MIN_DEG1}: {len(model_names)}")

    if not is_initialize and os.path.isfile(OUTPUT_PATH):
        print(f"Loading existing results from {OUTPUT_PATH}...")
        initial_df = pd.read_csv(OUTPUT_PATH)
    else:
        initial_df = pd.DataFrame()

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[cn.COL_MODEL_NAME].values)

    rows: list[pd.Series] = []
    for item in TimecourseIterator():
        if item.model_name not in model_names:
            continue
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        try:
            series = SystemDiscovery.analyzePerturbations(
                model=item.timecourse.model,
                training_df=item.timecourse.timecourse_df,
                threshold=THRESHOLD,
                perturbations=PERTURBATIONS,
                perturbation_species_fraction=SPECIES_FRACTION,
                poly_degree=POLY_DEGREE,
                is_plot=False,
            )
        except Exception as exc:
            print(f"  [error] {item.model_name}: {exc}", file=sys.stderr)
            continue
        rows.append(series)
        full_df = pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        full_df.to_csv(OUTPUT_PATH, index=False)

    full_df = (
        pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        if rows
        else initial_df
    )
    print(f"\nDone. {len(full_df)} rows in {OUTPUT_PATH}")
    return full_df


if __name__ == "__main__":
    main()
