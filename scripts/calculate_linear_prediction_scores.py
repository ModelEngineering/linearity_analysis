"""
Analyze SystemDiscovery prediction accuracy across BioModels for those with SEDML endtimes.
This script is intended to be run in parallel across multiple processes,
    each handling a distinct slice of the available models.

Usage:
    python analyze_linear_predictor.py <first_model_num> <last_model_num> 


Each instance processes a distinct slice of available BioModels and writes
results to a per-instance CSV (linear_predictor_scores2_<process_index>.csv).
"""
import src.constants as cn # type: ignore
from src.system_discovery import SystemDiscovery # type: ignore
from src.score import Score # type: ignore
from src.biomodels_iterator import BiomodelsIterator, getBiomodelsEndtimes # type: ignore
from src.timecourse_iterator import TimecourseIterator # type: ignore
from src.timecourse import Timecourse # type: ignore
from src.model import Model # type: ignore

import argparse
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os

SERIALIZATION_PATH = os.path.join(cn.DATA_DIR, "linear_predictor_scores.csv")

EXCLUDED_MODELS: list[str] = [
    "BIOMD0000000014",  # Errors "too much work"
    "BIOMD0000000035",  # Errors "too much work"
    "BIOMD0000000036",  # Errors "too much work"
    "BIOMD0000000079",  # Errors "too much work"
    "BIOMD0000000088",  # Errors "too much work"

    "BIOMD0000000072", 
    "BIOMD0000000566", 
    "BIOMD0000000567", 
    "BIOMD0000000666", 
    "BIOMD0000000794", 
    "BIOMD0000000866", 
    "BIOMD0000001012", 
    "BIOMD0000001000", 
    "BIOMD0000000144", 
    "BIOMD0000000151", 
    "BIOMD0000000432", 
    "BIOMD0000000450", 
    "BIOMD0000000514", 
    "BIOMD0000000606", 
]


def _getModelNums() -> list[int]:
    """Return sorted list of all BIOMD model numbers found in the biomodels directory."""
    model_nums = []
    for d in os.listdir(cn.BIOMODELS_DIR):
        if os.path.isdir(os.path.join(cn.BIOMODELS_DIR, d)) and "BIOMD" in d:
            num = BiomodelsIterator.extractModelNum(d)
            if num > 0:
                model_nums.append(num)
    return sorted(model_nums)


def _getChunk(all_model_nums: list[int], num_processes: int,
        process_index: int) -> tuple[int, int]:
    """
    Return (first_model_num, last_model_num) for this process's slice.

    Parameters
    ----------
    all_model_nums : list[int]
        Sorted list of all available model numbers.
    num_processes : int
        Total number of parallel processes.
    process_index : int
        1-based index of this process.

    Returns
    -------
    tuple[int, int]
        First and last model numbers (inclusive) for this process, or
        (0, -1) when no models are assigned to this index.
    """
    chunk_size = math.ceil(len(all_model_nums) / num_processes)
    start_idx = (process_index - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_model_nums))
    if start_idx >= len(all_model_nums):
        return 0, -1
    chunk = all_model_nums[start_idx:end_idx]
    return chunk[0], chunk[-1]

def processModels(first_model_num: int, last_model_num: int,
        threshold: float=cn.SYSTEM_DISCOVERY_THRESHOLD,
        serialization_path: str=SERIALIZATION_PATH) -> None:
    """Processes all the models in the range of first to last.

    Args:
        first_model_num (int):
        last_model_num (int):
        threshold (float): The threshold for the system discovery.
    """
    """ print(f"Process {process_index}/{num_processes}: "
        f"models {first_model_num}–{last_model_num} "
        f"({last_model_num - first_model_num + 1} in range)") """
    endtime_dct = getBiomodelsEndtimes(is_include_endtime_source=True)

    score = Score(serialization_path=serialization_path)
    if len(score.score_df) > 0:
        if "description" in score.score_df.columns:
            existing_models = set(score.score_df["description"].unique())
        else:
            existing_models = set(score.score_df[cn.COL_SYSTEM_ID].unique())
    else:
        existing_models = set()
    excluded_models = list(set(EXCLUDED_MODELS) | existing_models)
    iterator = BiomodelsIterator(
            excluded_models=excluded_models,
            first_model_num=first_model_num,
            last_model_num=last_model_num)
    timecourse_iterator = TimecourseIterator()
    #
    for item in iterator:
        model_name = item.model_name
        if (model_name not in endtime_dct)  \
                or (endtime_dct[model_name][1] != cn.ENDTIME_SOURCE_SEDML):
            print(f"Not a model with a SEDML endtime: {model_name} — skipping.")
            continue
        # Get the timecourse or created it if not present
        found_timecourse = False
        try:
            timecourse = timecourse_iterator.getTimecourse(model_name)
            found_timecourse = True
        except Exception as e:
            print(f"Timecourse not found. Creating it.")
        # Construct Timecourse if not found
        if not found_timecourse:
            try:
                model = Model.makeBiomodel(model_name=model_name)
                timecourse = Timecourse(model, num_point=1000)
                _ = timecourse.timecourse_df  # Force creation of the timecourse_df
                timecourse.serialize()
            except Exception as e:
                print(f"Error occurred while creating timecourse for model {model_name}: {e}")
                continue
        try:
            timecourse = timecourse_iterator.getTimecourse(model_name)
            discovery = SystemDiscovery.makeBiomodel(
                    model_name=model_name, timecourse=timecourse,
                    threshold=threshold)
            discovery.fit()
            prediction_df = discovery.predict()
            score.add(timecourse.timecourse_df, prediction_df, system_id=model_name)
        except Exception as e:
            print(f"Error occurred while processing model {model_name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze BioModels linear predictor (one shard of a parallel run).")
    parser.add_argument("--first_model", type=int, default=1,
            help="First model number to process.")
    parser.add_argument("--last_model", type=int, default=-1,
            help="Last model number to process.")
    args = parser.parse_args()
    #
    first_model : int = args.first_model
    last_model : int = args.last_model
    if last_model < 0:
        last_model = 2000
    processModels(first_model, last_model, threshold=0.001)
    raise SystemExit(0)