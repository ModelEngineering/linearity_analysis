'''Utilities'''

from pathlib import Path
import src.constants as cn

import pandas as pd  # type: ignore
from typing import List, Optional


PATTERN = "piecewise_predictions__numpoint*.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def getPSDPreditionDF(max_fractional_reduction: str, repeat: int = 1) -> pd.DataFrame:
    sel1_str = f"__maxreduction_{max_fractional_reduction}"
    sel2_str = f"__repeat_{repeat}"

    csv_files = sorted(DATA_DIR.glob(PATTERN))
    for sel_str in [sel1_str, sel2_str]:
        csv_files = [f for f in csv_files if sel_str in f.name]
    if not csv_files:
        raise SystemExit(f"No files matching '{PATTERN}' found in {DATA_DIR}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df[cn.COL_CSV_FILE] = path.name
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df = merged_df[merged_df["aggregation_type"] == "model"].reset_index(drop=True)
    return merged_df