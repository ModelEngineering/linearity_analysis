"""
Analyze SystemDiscovery prediction accuracy across BioModels, parallelized across processes.

Usage:
    python analyze_linear_predictor.py <num_processes> <process_index>

    num_processes : total number of parallel instances
    process_index : 1-based index of this instance

Each instance processes a distinct slice of available BioModels and writes
results to a per-instance CSV (linear_predictor_scores2_<process_index>.csv).
"""
import src.constants as cn # type: ignore
from src.system_discovery import SystemDiscovery # type: ignore
from src.score import Score # type: ignore
from src.biomodels_iterator import BiomodelsIterator, getBiomodelsEndtimes # type: ignore
from src.timecourse_iterator import TimecourseIterator # type: ignore

import argparse
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os

EXCLUDED_MODELS: list[str] = [
    "BIOMD0000000014",  # Errors "too much work"
    "BIOMD0000000035",  # Errors "too much work"
    "BIOMD0000000036",  # Errors "too much work"
    "BIOMD0000000079",  # Errors "too much work"
    "BIOMD0000000088",  # Errors "too much work"

    "BIOMD0000000072", 
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

def processModels(first_model_num: int, last_model_num: int, process_index: int,
        num_processes: int) -> None:
    """Processes all the models in the range of first to last.

    Args:
        first_model_num (int): _description_
        last_model_num (int): _description_
        process_index (int): _description_
        num_processes (int): _description_
    """
    """ print(f"Process {process_index}/{num_processes}: "
        f"models {first_model_num}–{last_model_num} "
        f"({last_model_num - first_model_num + 1} in range)") """
    endtime_dct = getBiomodelsEndtimes(is_include_endtime_source=True)
    serialization_path = os.path.join(
            cn.DATA_DIR, f"linear_predictor_scores_{process_index}.csv")

    score = Score(serialization_path=serialization_path)
    if os.path.exists(score._serialization_path):
        existing_models = set(score.score_df["description"].unique())
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
        try:
            timecourse = timecourse_iterator.getTimecourse(model_name)
            discovery = SystemDiscovery.makeBiomodel(
                    model_name=model_name, timecourse=timecourse)
            discovery.fit()
            prediction_df = discovery.predict()
            score.addTestResult(
                    timecourse.timecourse_df, prediction_df,
                    description=model_name)
        except Exception as e:
            print(f"Error occurred while processing model {model_name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze BioModels linear predictor (one shard of a parallel run).")
    parser.add_argument("num_processes", type=int,
            help="Total number of parallel process instances.")
    parser.add_argument("process_index", type=int,
            help="1-based index of this process instance.")
    args = parser.parse_args()
    #
    num_processes: int = args.num_processes
    process_index: int = args.process_index
    #
    if process_index > num_processes or process_index < 1:
        raise ValueError(
            f"process_index ({process_index}) must be in [1, num_processes={num_processes}].")

    all_model_nums = _getModelNums()
    first_model_num, last_model_num = _getChunk(all_model_nums, num_processes, process_index)
    if last_model_num < first_model_num:
        print(f"No models assigned to process {process_index} — exiting.")
    processModels(first_model_num, last_model_num, process_index, num_processes)
    raise SystemExit(0)