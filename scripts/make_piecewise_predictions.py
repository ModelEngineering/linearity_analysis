'''Creates piecewise predictions for elements of BioModels.'''

"""
Usage:
    python scripts/make_piecewise_predictions.py --first_model_num 0 --last_model_num 100 --min_segment_length 50
"""

import src.constants as cn  # type: ignore
from src.score import Score  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator, TimecourseIteratorItem  # type: ignore

import argparse
import os
import pandas as pd  # type: ignore
from typing import List, Optional

EXCLUDED_MODELS: List[str] = [
    "BIOMD0000000339",
]
MAX_CHANGPOINTS = [0, 1, 5, 10, 12, 15, 17, 18, 19, 20]  # Maximum number of change points to consider in the piecewise model.
MAX_FRACTIONAL_REDUCTION = 0.01  # Maximum fractional reduction in the sum of squared errors required to accept a new change point.
COEFFICIENT_THRESHOLD = 0.001  # Threshold for coefficient magnitude to consider a species as linear.

#################################################################
# Preliminaries
#################################################################
if os.path.isfile(os.path.join(cn.DATA_DIR, "badmodels.txt")):
    with open(os.path.join(cn.DATA_DIR, "badmodels.txt"), "r") as f:
        for line in f:
            model_name = line.strip()
            if model_name and not model_name.startswith("#"):
                EXCLUDED_MODELS.append(model_name.upper())


#################################################################
# Processing for a single model and parameters
#################################################################
def processModel(
        item: TimecourseIteratorItem,
        max_changepoint: int,
        min_segment_length: int,
        coefficient_threshold: float,
        max_fractional_reduction: float = MAX_FRACTIONAL_REDUCTION,
) -> Optional[pd.DataFrame]:
    """
    Process a single item

    Args:
        item (TimecourseIteratorItem): Information on current model and its timecourse.
        max_changepoint (int): Maximum number of change points to consider.
        min_segment_length (int): Minimum length of each segment in the piecewise model. 
        max_fractional_reduction (float): Maximum fractional reduction in the sum of squared errors required to accept a new change point.
        coefficient_threshold (float): Threshold for the coefficient of determination (R-squared) to consider a model valid.

    Returns:
        Optional[pd.DataFrame]: _description_
    """
    model_name = item.model_name
    df = item.timecourse.timecourse_df
    try:
        psd = PiecewiseSystemDiscovery(df,
                max_changepoint=max_changepoint,
                min_segment_length=min_segment_length,
                model_name=model_name,
                coefficient_threshold=coefficient_threshold,
                max_fractional_reduction=max_fractional_reduction,
        )
        psd.fit()
        pred_df = psd.predict()
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return None
    if pred_df is None:
        print(f"Skipping {item.model_name} (no new data)")
        return None
    # Create the score
    score = Score()
    score.add(df, pred_df, system_id=model_name)
    accuracy_df = score.score_df
    # Augment the dataframe with additional columns for the model
    accuracy_df[cn.COL_SYSTEM_ID] = model_name
    accuracy_df[cn.COL_MAX_CHANGEPOINT] = max_changepoint
    accuracy_df[cn.COL_MIN_SEGMENT_LENGTH] = min_segment_length
    accuracy_df[cn.COL_MAX_FRACTIONAL_REDUCTION] = max_fractional_reduction
    accuracy_df[cn.COL_COEFFICIENT_THRESHOLD] = coefficient_threshold
    accuracy_df[cn.COL_NUM_CHANGEPOINT] = psd.num_changepoint  # Number of change points detected in the piecewise model. 
    #
    return accuracy_df


#################################################################
# Iterate across models and make piecewise predictions
#################################################################
def main(
        process_idx: int = 0,
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
        is_initialize: bool = False, # Ignore existing serialized Timecourse when initializing (for testing).
        coefficient_threshold: float = COEFFICIENT_THRESHOLD,
        min_segment_length: int = 50,
        max_fractional_reduction: float = MAX_FRACTIONAL_REDUCTION,  # 0 means "accept any ASS reduction" — aggressive batch mode across thousands of models.
        output_path: str = cn.PIECEWISE_PREDICTIONS_PATH,
) -> None:
    '''
    Main function to make piecewise predictions. Iterate across models in the timecourse zip file, and for each model, fit a piecewise linear model
    and make predictions.

    Parameters
    ----------
    process_idx : int
        Index of the current process (for parallel processing). 
    first_model_num : int
        First model number to include (inclusive).
    last_model_num : int
        Last model number to include (inclusive).
    is_initialize : bool
        Whether to initialize the output file.
    coefficient_threshold : float
        Threshold for coefficient magnitude to consider a species as linear.
    min_segment_length : int
        Minimum length of each segment in the piecewise model.
    max_fractional_reduction : float
        Maximum fractional reduction in the sum of squared errors required to accept a new change point.
    output_path : str
        Path to the output file. Is modified by the process index
    '''
    output_path = output_path.replace(".csv", f"_{process_idx}.csv")
    if os.path.isfile(output_path) and (not is_initialize):
        initial_df = pd.read_csv(output_path)
        existing_model_names = set(initial_df[cn.COL_SYSTEM_ID].unique())
    else:
        existing_model_names = set()
        initial_df = pd.DataFrame()
    # Process the max_changepoint values in order, so that the output file is sorted by max_changepoint.
    for item in TimecourseIterator(
            first_model_num=first_model_num,
            last_model_num=last_model_num):
        # See if this is a model to skip
        if item.model_name in existing_model_names:
            print(f"Skipping {item.model_name} (already processed)")
            continue
        if item.model_name.upper() in EXCLUDED_MODELS:
            print(f"Skipping {item.model_name} (excluded)")
            continue
        # Process the model for each max_changepoint valuea
        for max_changepoint in MAX_CHANGPOINTS:
            pred_df = None
            msg = f"Processing {item.model_name} (max_changepoint={max_changepoint})"
            print(msg)
            pred_df = processModel(item,
                    max_changepoint=max_changepoint,
                    min_segment_length=min_segment_length,
                    max_fractional_reduction=max_fractional_reduction,
                    coefficient_threshold=coefficient_threshold,
                    )
            if pred_df is None:
                print(f"Skipping {item.model_name} {max_changepoint}--no predicturion.")
                continue
            if pred_df is not None:
                initial_df = pd.concat([initial_df, pred_df], ignore_index=True)
                initial_df.to_csv(output_path, index=False)

    # Persist results to disk.
    initial_df.to_csv(output_path, index=False)


#################################################################
# Collect parameters and run the script.
#################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Piecewise predictions for BioModels.")
    parser.add_argument("--process_idx", type=int, default=0)
    parser.add_argument("--first_model_num", type=int, default=0)
    parser.add_argument("--last_model_num", type=int, default=int(1e9))
    parser.add_argument("--initialize", action="store_true",
                        help="Reset output file to empty (reprocess all models).")
    parser.add_argument("--min_segment_length", type=int, default=50)
    parser.add_argument("--max_fractional_reduction", type=float, default=MAX_FRACTIONAL_REDUCTION,
                        help="Maximum fractional reduction in accuracy to eliminate a changepoint.")
    parser.add_argument("--coefficient_threshold", type=float, default=COEFFICIENT_THRESHOLD,
                        help="Threshold for coefficient magnitude to consider a species as linear.")
    args = parser.parse_args()
    main(
            process_idx=args.process_idx,
            first_model_num=args.first_model_num,
            last_model_num=args.last_model_num,
            is_initialize=args.initialize,
            min_segment_length=args.min_segment_length,
            max_fractional_reduction=args.max_fractional_reduction,
            coefficient_threshold=args.coefficient_threshold,
        )