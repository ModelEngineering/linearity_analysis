"""Perturbation study: how do perturbed initial conditions affect SystemDiscovery R²?

Runs only on models whose deg1_min >= 0.9 in evaluate_monomial_models-0.01.csv.
For each model, a degree-1 SystemDiscovery is fitted on the unperturbed timecourse,
then evaluated against test timecourses at perturbation_value_fraction of
0%, 5%, 10%, 20%, and 50% (all species perturbed).

The R² for a model at a given perturbation is the minimum clamped R² across species.

Output CSV: data/perturbation_study.csv
Columns: model_name, r2_0, r2_5, r2_10, r2_20, r2_50
"""

import os
import sys

import pandas as pd

import src.constants as cn
from src.system_discovery import SystemDiscovery
from src.timecourse import Timecourse
from src.timecourse_iterator import TimecourseIterator

THRESHOLD = 0.01
POLY_DEGREE = 1
SPECIES_FRACTION = 1.0
PERTURBATIONS: list[tuple[float, str]] = [
    (0.00, "r2_0"),
    (0.05, "r2_5"),
    (0.10, "r2_10"),
    (0.20, "r2_20"),
    (0.50, "r2_50"),
]

SOURCE_PATH = os.path.join(cn.DATA_DIR, "evaluate_monomial_models-0.01.csv")
OUTPUT_PATH = os.path.join(cn.DATA_DIR, "perturbation_study.csv")
MIN_DEG1 = 0.9


def _evaluate_model(model_name: str, timecourse: Timecourse) -> dict:
    row: dict = {cn.COL_MODEL_NAME: model_name}
    for _, col in PERTURBATIONS:
        row[col] = float("nan")

    training_df = timecourse.timecourse_df
    start_time = float(training_df.index[0])
    end_time = float(training_df.index[-1])
    num_point = len(training_df)

    try:
        sd = SystemDiscovery(training_df, threshold=THRESHOLD, poly_degree=POLY_DEGREE)
        sd.fit()
    except Exception as exc:
        print(f"  [fit] {model_name}: {exc}", file=sys.stderr)
        return row

    for p, col in PERTURBATIONS:
        try:
            tc = Timecourse(
                model=timecourse.model,
                start_time=start_time,
                end_time=end_time,
                num_point=num_point,
                perturbation_value_fraction=p,
                perturbation_species_fraction=SPECIES_FRACTION,
            )
            row[col] = sd.minR2(test_df=tc.timecourse_df)
        except Exception as exc:
            print(f"  [p={p}] {model_name}: {exc}", file=sys.stderr)

    return row


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
        pd.DataFrame(columns=[cn.COL_MODEL_NAME]).to_csv(OUTPUT_PATH, index=False)

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[cn.COL_MODEL_NAME].values)

    rows: list[dict] = []
    for item in TimecourseIterator():
        if item.model_name not in model_names:
            continue
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        row = _evaluate_model(item.model_name, item.timecourse)
        rows.append(row)
        full_df = pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        full_df.to_csv(OUTPUT_PATH, index=False)

    full_df = pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True) if rows else initial_df
    print(f"\nDone. {len(full_df)} rows in {OUTPUT_PATH}")
    return full_df


if __name__ == "__main__":
    main()
