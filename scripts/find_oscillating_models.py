'''Find oscillation frequencies in each BioModel timecourse via FFT.

Usage:
    python scripts/find_oscillating_models.py --first_model_num 0 --last_model_num -1

For each model's ``timecourse_df``, runs :func:`src.oscillation_detector.findOscillations` on
each species column and writes one CSV row per ``(model, species)`` pair to
``data/find_oscillating_models.csv``.
'''


import src.constants as cn  # type: ignore
from src.oscillation_detector import findOscillations  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import argparse
import os
import pandas as pd  # type: ignore
from typing import List, Optional


OUTPUT_PATH = os.path.join(cn.DATA_DIR, "find_oscillating_models.csv")

def _addEntry(result_dct, model_name: str,
        species_name: Optional[str] = None,
        frequencies: Optional[List[float]] = None,
        endtime: Optional[float] = None) -> None:
    """Add a an entry to the results dictionary.

    Parameters
    ----------
    dct : dict
        Dictionary of lists for each column.
    model_name : str
        Model name (system ID).
    species_name : str
        Species name.
    frequencies : list
        List of detected oscillation frequencies in Hz.
    """
    result_dct[cn.COL_SYSTEM_ID].append(model_name)
    result_dct[cn.COL_SPECIES_NAME].append(species_name)
    result_dct[cn.COL_FREQUENCIES].append(frequencies)
    result_dct[cn.COL_ENDTIME].append(endtime)


def processModels(
    first_model_num: int,
    last_model_num: int,
    output_path: str = OUTPUT_PATH,
    is_initialize: bool = True,
) -> None:
    """Find oscillation frequencies per species in each serialized timecourse.

    Parameters
    ----------
    first_model_num : int
        First model number (inclusive).  ``0`` starts from the beginning.
    last_model_num : int
        Last model number (inclusive).  Negative values mean "all remaining".
    output_path : str, optional
        Path for the resulting CSV.  Defaults to ``data/find_oscillating_models.csv``.
    is_initialize : bool, optional
        If ``True``, overwrites the output file.  If ``False``, append
        to the existing file.
    """
    # Get initial if not initializing
    if not is_initialize and os.path.isfile(output_path):
        current_df = pd.read_csv(output_path)
        existing_model_names = set(current_df[cn.COL_SYSTEM_ID].unique())
    else:
        current_df = pd.DataFrame()
        existing_model_names = set()
    # Collect the results
    iterator = TimecourseIterator(
        first_model_num=first_model_num,
        last_model_num=last_model_num,
        is_report=True,
    )
    result_dct = {n: [] for n in [cn.COL_SYSTEM_ID, cn.COL_SPECIES_NAME,
            cn.COL_FREQUENCIES, cn.COL_ENDTIME]}
    for item in iterator:
        model_name = item.model_name
        if model_name in existing_model_names:
            print(f"Skipping {model_name} (already processed)")
            continue
        # Process the timecourse_df for each species
        timecourse_df = item.timecourse.timecourse_df
        if timecourse_df.empty or len(timecourse_df.columns) == 0:
            print(f"Empty timecourse_df {model_name}.")
            _addEntry(result_dct, model_name)
            continue
        else:
            for species in timecourse_df.columns:
                col_df = timecourse_df[[species]]
                frequencies = findOscillations(col_df)
                _addEntry(result_dct, model_name, species, frequencies,
                        item.timecourse.timecourse_df.index.to_numpy()[-1])
        # Write results for this model
        added_df = pd.DataFrame(result_dct)
        full_df = pd.concat([current_df, added_df], ignore_index=True)
        full_df.to_csv(output_path, index=False)
        current_df = full_df.copy()

    print(f"Wrote {len(current_df)} rows to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find oscillating species in BioModel timecourses via FFT.",
    )
    parser.add_argument(
        "--first_model_num", type=int, default=0,
        help="First model number to process (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--last_model_num", type=int, default=-1,
        help="Last model number to process (inclusive). -1 = all. Default: -1.",
    )
    parser.add_argument(
        "--initialize", action="store_true",
        help="Indicates if output csv file should be initialized (overwritten) or appended to. Default: False.",
    )
    args = parser.parse_args()

    processModels(
        first_model_num=args.first_model_num,
        last_model_num=args.last_model_num,
        is_initialize=args.initialize,
    )