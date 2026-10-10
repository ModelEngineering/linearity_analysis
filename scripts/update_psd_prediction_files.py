#!/usr/bin/env python3
'''Append an "auc" column to every ``data/piecewise_predictions*.csv`` file.

For each row, AUC is computed from that row's own changepoints via :class:`NLCurve`:

* If the changepoints list is empty (``"[]"``), AUC is set to **1.0** -- the
  value of a single-segment curve over the full timecourse.
* Otherwise boundaries are built from the parsed changepoints as
  ``[0, *sorted(changepoints), NUM_POINT - 1]``, an :class:`NLCurve` is
  constructed, and :meth:`~src.nl_curve.NLCurve.calculateAUC` returns the value.

Each CSV is backed up to ``<path>.bak`` before being overwritten in place so a
re-run can be reverted easily if something looks wrong.

Usage::

    python scripts/update_psd_prediction_files.py [--dry-run] [--verbose]
'''
import src.constants as cn  # noqa: E402
import src.util as util  # type: ignore
from src.nl_curve import NLCurve # type: ignore
from src.psd_prediction_files_iterator import PSDPredictionFilesIterator  # type: ignore

import argparse
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import shutil
from collections import namedtuple


UpdateTask = namedtuple("UpdateTask", ["column", "func"])


#############################
# Column editing functions
#############################

def _addAUCColumn(df) -> int:
    """Append an ``auc`` column to *df* in place

    Args:
        df (pd.DataFrame): DataFrame to update
    Returns:
        int: Number of rows updated

    Note that this function may fail if the changepoints list is empty or invalid, in which case the caller should handle the exception.
    """
    df[cn.COL_AUC] = df.apply(lambda row: NLCurve(row[cn.COL_BOUNDARIES], name=row[cn.COL_SYSTEM_ID]).auc, axis=1)
    return len(df)

UPDATE_TASKS = [
    UpdateTask(column=cn.COL_AUC, func=_addAUCColumn),
]

############################
# Main script
############################

def main(is_report: bool = False):

    ##
    def message(msg: str) -> None:
        if is_report:
            print(msg)

    psd_iterator = PSDPredictionFilesIterator(is_pkl=True)

    ok_count = 0
    fail_count = 0
    success_count = 0

    for item in psd_iterator:
        # Save the file
        df = item.df
        if df.empty:
            message("  SKIP: empty DataFrame after reading.")
            continue
        else:
            success_count += 1
        # Process the updates
        n_rows = 0
        for task in UPDATE_TASKS:
            if task.column in df.columns:
                message(f"    SKIP: already has '{task.column}' column ({len(df)} rows).")
                ok_count += 1
                continue
            try:
                n_rows = task.func(df)
                message(f"  SUCCESS: {task.column} computation succeeded for {item.filepath} ({n_rows} rows).")
            except Exception as exc:
                message(f"  FAIL: {task.column} computation failed for {item.filepath}: {exc}; keeping backup.")
                fail_count += 1
                continue
        # Back up the original file before mutating it.
        filepath = Path(item.csv_path)
        backup_path = Path(filepath.with_suffix(filepath.suffix + ".bak"))
        shutil.copy2(filepath, backup_path)
        if is_report:
            message(f"    backed up -> {backup_path.name}")
        # Write the updated DataFrame back to the original file.
        df.to_csv(filepath, index=False)
        #
        if is_report:
            message(f"Updated {filepath.name} ({n_rows} rows)")

    if is_report:
        print(f"Successfully processed {success_count} files: {ok_count} OK, {fail_count} FAIL.")   
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append an 'auc' column to every piecewise_predictions CSV.",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-file and progress output.")
    args = parser.parse_args()
    args = parser.parse_args()
    sys.exit(main(is_report=args.verbose))
