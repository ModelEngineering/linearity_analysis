'''Script that creates a CSV file of BioModels end times.'''

import src.constants as cn  # type: ignore
from src.biomodels_iterator import BiomodelsIterator, BiomodelsItem # type: ignore
from src.trajectory import Trajectory  # type: ignore

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
from typing import List, Dict, Tuple, Optional

EXCLUDED_MODELS = ["BIOMD0000000268",
        "BIOMD0000000625",]


def main(output_path: str = cn.CALCULATED_ENTIMES_PATH, is_report: bool = True) -> pd.DataFrame:
    '''
    Main function.

    Parameters
    ----------
    output_path : str
        Path to the output CSV file. Defaults to cn.CALCULATED_ENTIMES_PATH
    is_report : bool
        Whether the model is a report. --- IGNORE ---
    '''
    # Initialize a dictionary to store the results
    result_dct: dict = {c:[] for c in [cn.COL_MODEL_NAME, cn.COL_ENDTIME_SOURCE, cn.COL_ENDTIME]}
    # Loop through each SBML directory and get the end time
    endtime_df = pd.DataFrame(result_dct)
    for item in BiomodelsIterator(excluded_models=EXCLUDED_MODELS, existing_csv_path=output_path,
            is_report=is_report):
        model_name = item.model_name
        if len(item.sbml_paths) == 0:
            continue
        #
        try:
            trajectory = Trajectory.makeBiomodel(item.model_num)
        except Exception as e:
            print(f"Error occurred while processing model {model_name}: {e}")
            continue
        #
        result_dct[cn.COL_MODEL_NAME].append(model_name)
        result_dct[cn.COL_ENDTIME].append(trajectory.end_time)
        result_dct[cn.COL_ENDTIME_SOURCE].append(trajectory.end_time_source)
        existing_df : pd.DataFrame = item.existing_df
        df = pd.DataFrame(result_dct)
        if len(df) > 0 and len(existing_df) > 0:
            endtime_df= pd.concat([existing_df, df], ignore_index=True)
        elif len(df) > 0:
            endtime_df = df
        else:
            endtime_df = existing_df
        endtime_df.to_csv(output_path, index=False)
    # Create a DataFrame from the list of end times
    if os.path.exists(output_path):
        endtime_df = pd.read_csv(output_path)
    else:
        existing_df = pd.DataFrame()
    return endtime_df


if __name__ == "__main__":
    main()