'''Utilities'''

from pathlib import Path
import src.constants as cn

import pandas as pd  # type: ignore
from typing import Optional


PATTERN = "piecewise_predictions__numpoint*.csv"
PSD_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def getPSDPredictionDF(max_fractional_reduction: str, repeat: Optional[int] = None) -> pd.DataFrame:
    """Get a DataFrame from piecewise prediction CSVs filtered by max_fractional_reduction and repeat.

    Args:
        max_fractional_reduction (str): _description_
        repeat (int, optional): Which repetition file to use. If None, then all repetitions are used. Defaults to None.

    Raises:
        FileNotFoundError: If no file is found

    Returns:
        pd.DataFrame
    """
    file_selection1_str = f"__maxreduction_{max_fractional_reduction}"
    if repeat is not None:
        file_selection2_str = f"__repeat_{repeat}"
    else:
        file_selection2_str = None

    csv_files = sorted(PSD_DATA_DIR.glob(PATTERN))
    sections = [file_selection1_str]
    if isinstance(file_selection2_str, str):
        sections.append(file_selection2_str)
    for file_selection_str in sections:
        csv_files = [f for f in csv_files if file_selection_str in f.name]
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{PATTERN}' found in {PSD_DATA_DIR}")
    # Get the data
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df[cn.COL_CSV_FILE] = path.name
        frames.append(df)
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df = merged_df[merged_df["aggregation_type"] == "model"].reset_index(drop=True)
    # Check result
    sel = merged_df["max_fractional_reduction"] == float(max_fractional_reduction)
    if not sel.all():
        raise ValueError(
            f"Some rows in the merged DataFrame do not match max_fractional_reduction={max_fractional_reduction}"
        )
    return merged_df

def codedstr2Dict(codedstr: str) -> dict:
    """Convert a coded string to a dictionary.

    Args:
        codedstr (str): Coded string in the format "key1_value1__key2_value2"

    Returns:
        dict: Dictionary with keys and values extracted from the coded string.
    """
    result = {}
    for part in codedstr.split("__"):
        if "_" in part:
            key, value = part.split("_", 1)
            try:
                result[key] = eval(value)
            except:
                result[key] = value
    return result