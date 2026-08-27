"""Assess PiecewiseSystemDiscovery accuracy across all BioModels.

For each serialized Timecourse, fits a PiecewiseSystemDiscovery with
NUM_CHANGE_POINT change points and records score() fields in a CSV.

Usage:
    source activate.sh
    python scripts/analyze_piecewise_system_discovery.py [--num_change_point N]

Output CSV: data/piecewise_system_discovery_scores-<N>.csv
Columns: model_name, score_min, score_median, score_max, num_nonzero_term
"""

import argparse
import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn
from src.piecewise_system_discovery import PiecewiseSystemDiscovery
from src.timecourse_iterator import TimecourseIterator

COL_MODEL_NAME = cn.COL_MODEL_NAME
COL_SCORE_MIN = "score_min"
COL_SCORE_MEDIAN = "score_median"
COL_SCORE_MAX = "score_max"
COL_NUM_NONZERO_TERM = "num_nonzero_term"

OUTPUT_TEMPLATE = os.path.join(cn.DATA_DIR, "piecewise_system_discovery_scores-{}.csv")


def _output_path(num_change_point: int) -> str:
    return OUTPUT_TEMPLATE.format(num_change_point)


def main(num_change_point: int = 2, is_initialize: bool = False) -> pd.DataFrame:
    output_path = _output_path(num_change_point)

    if not is_initialize and os.path.isfile(output_path):
        print(f"Loading existing results from {output_path}...")
        initial_df = pd.read_csv(output_path)
    else:
        initial_df = pd.DataFrame()

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[COL_MODEL_NAME].values)

    rows: list[dict] = []
    for item in TimecourseIterator():
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        try:
            psd = PiecewiseSystemDiscovery(
                item.timecourse,
                max_changepoint=num_change_point,
            ).fit()
            info = psd.score()
        except Exception as exc:
            print(f"  [error] {item.model_name}: {exc}", file=sys.stderr)
            continue
        rows.append({
            COL_MODEL_NAME: item.model_name,
            COL_SCORE_MIN: info.min,
            COL_SCORE_MEDIAN: info.median,
            COL_SCORE_MAX: info.max,
            COL_NUM_NONZERO_TERM: info.num_nonzero_term,
        })
        full_df = pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        full_df.to_csv(output_path, index=False)

    full_df = (
        pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        if rows
        else initial_df
    )
    print(f"\nDone. {len(full_df)} rows written to {output_path}")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_change_point", type=int, default=2,
        help="Number of change points for PiecewiseSystemDiscovery (default: 2)",
    )
    parser.add_argument(
        "--initialize", action="store_true",
        help="Ignore existing results and start fresh",
    )
    args = parser.parse_args()
    main(num_change_point=args.num_change_point, is_initialize=args.initialize)
