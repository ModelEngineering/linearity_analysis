"""Evaluate SystemDiscovery with poly_degree 0, 1, 2 over all BioModel timecourses.

Output CSV written to data/evaluate_monomial_models.csv with columns:
    model_name, num_species, num_reaction, num_perturbation, num_timepoint,
    deg0_min, deg0_median, deg0_max,
    deg1_min, deg1_median, deg1_max,
    deg2_min, deg2_median, deg2_max
"""

import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.system_discovery import SystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

DEGREES = [0, 1, 2]
OUTPUT_PATH = os.path.join(cn.DATA_DIR, "evaluate_monomial_models.csv")
EXCLUDES: list[str] = [
]



def _score_cols(deg: int) -> tuple[str, str, str, str, str]:
    return f"deg{deg}_min", f"deg{deg}_median", f"deg{deg}_max",  \
        f"deg{deg}_values", f"deg{deg}_num_nonzero_term"


def _evaluate_model(model_name: str, timecourse, *,
        threshold: float=0.01) -> dict:
    row: dict = {
        cn.COL_MODEL_NAME: model_name,
        cn.COL_NUM_SPECIES: timecourse.model.num_species,
        cn.COL_NUM_REACTION: timecourse.model.num_reaction,
        cn.COL_NUM_PERTURBATION: timecourse.num_perturbation,
        cn.COL_NUM_TIMEPOINT: timecourse.num_timepoint,
    }
    for deg in DEGREES:
        for col in _score_cols(deg):
            row[col] = float("nan")

    for deg in DEGREES:
        try:
            sd = SystemDiscovery(
                timecourse.timecourse_df,
                poly_degree=deg,
                include_bias=True,
                threshold=threshold,
            )
            sd.fit()
            info = sd.score()[0]
            min_col, median_col, max_col, values_col, num_nonzero_term_col = _score_cols(deg)
            row[min_col] = info.min
            row[median_col] = info.p50
            row[max_col] = info.max
            row[values_col] = info.values
            row[num_nonzero_term_col] = info.num_nonzero_term
        except Exception as exc:
            print(f"  [deg={deg}] {model_name}: {exc}", file=sys.stderr)

    return row


def main(threshold: float = 1, is_initialize: bool = False) -> pd.DataFrame:
    """


    Args:
        threshold (float, optional): _description_. Defaults to 0.01.
        is_initialize (bool, optional): Overwrite existing output file.
    """
    if not is_initialize:
        if os.path.isfile(OUTPUT_PATH):
            print(f"Loading existing results from {OUTPUT_PATH}...")
            try:
                initial_df = pd.read_csv(OUTPUT_PATH)
            except Exception as exc:
                print(f"Error reading {OUTPUT_PATH}: {exc}", file=sys.stderr)
                initial_df = pd.DataFrame()
                initial_df.to_csv(OUTPUT_PATH, index=False)
        else:
            print(f"Initializing {OUTPUT_PATH} with header...")
            initial_df = pd.DataFrame()
            initial_df.to_csv(OUTPUT_PATH, index=False) 
    else:
        initial_df = pd.DataFrame()
    # Process models not already present
    rows:list = []
    full_df = pd.DataFrame()
    for item in TimecourseIterator(num_model=-1):
        if item.model_name in EXCLUDES:
            print(f"Skipping {item.model_name} (in EXCLUDES)", flush=True)
            continue
        if len(initial_df) > 0 and item.model_name in initial_df[cn.COL_MODEL_NAME].values:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        row = _evaluate_model(item.model_name, item.timecourse, threshold=threshold)
        rows.append(row)
        new_df = pd.DataFrame(rows)
        full_df = pd.concat([initial_df, new_df], ignore_index=True)
        full_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(initial_df)} rows to {OUTPUT_PATH}")
    return full_df


if __name__ == "__main__":
    main()
