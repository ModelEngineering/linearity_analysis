"""
Creates ClusteredJacobianCollection objects for BioModels.
"""
import src.constants as cn
from trajectory import Trajectory # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from biomodels_iterator import BiomodelsIterator  # type: ignore

import os
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Tuple

import numpy as np # type: ignore
import tellurium as te # type: ignore
from src.l_roadrunner import LRoadrunner # type: ignore

OUTPUT_DATA_FILE = os.path.join(cn.DATA_DIR, "model_linearity_analysis_data.csv")


class BiomodelsCluster:
    """Analyzes linearity of SBML or Antimony models by collecting Jacobians over time."""

    def __init__(
        self,
        model_name: str,
        start_time: float = 0,
        end_time: float = np.nan,
        num_point: int = 100,
        diameter_metric: str = cn.DIAMETER_IVP,
    ) -> None:
        """
        Initialize a BiomodelsCluster with a model and simulation parameters.

        Parameters
        ----------
        model_name : str
            Name of the model to analyze.
        start_time : float
            Simulation start time (default: 0).
        end_time : float, optional
            Simulation end time (default: 10).
        num_point : int
            Number of simulation timepoints (default: 100).
        """
        self.model_name = model_name
        self.start_time = start_time
        if np.isnan(end_time):
            end_time = LRoadrunner.endtime_dct.get(model_name, cn.END_TIME) # type: ignore
        self.end_time = end_time
        self.num_point = num_point
        self._diameter_metric = diameter_metric
        #
        self.sbml_str = self._getSbml()
        self.l_roadrunner = LRoadrunner(self.sbml_str, start_time=start_time,
                end_time=end_time, num_point=num_point)
        self._jacobian_collection = Trajectory(self.l_roadrunner,
                diameter_metric=diameter_metric)

    @staticmethod
    def _report(text: str):
        """Print a report message if reporting is enabled."""
        print(text)

    def _getSbml(self) -> str:
        """Search for an SBML file in the given directory and return its contents as a string."""
        dir_path = os.path.join(cn.BIOMODELS_DIR, self.model_name)
        sbml_files = os.listdir(dir_path)
        sbml_files = [f for f in sbml_files 
                if f.endswith(".xml") and not "manifest" in f.lower()]
        if not sbml_files:
            raise FileNotFoundError("No SBML file found for model: " + self.model_name)
        with open(os.path.join(dir_path, sbml_files[0]), "r") as f:
            return f.read()
        
    def cluster(self, n_cluster: int, is_sequential_partition: bool = True) -> TrajectoryCollection:
        """Partition the Jacobians into clusters and return a ClusteredJacobianCollection."""
        if is_sequential_partition:
            trajectory_collection = TrajectoryCollection(
                    self._jacobian_collection.sequentialPartition(
                    n_cluster=n_cluster))
        else:
            trajectory_collection = TrajectoryCollection(
                    self._jacobian_collection.nonsequentialPartition(
                    n_cluster=n_cluster))
        return trajectory_collection

    @classmethod
    def clusterAnalysis(
        cls,
        directory: str = cn.BIOMODELS_DIR,
        output_data_file: str = OUTPUT_DATA_FILE,
        excluded_models: Optional[List[str]] = None,
        start_time: float = cn.START_TIME,
        num_point: int = cn.NUM_POINTS,
        n_cluster: int = 1,
        is_report: bool = True,
        is_sequential_partition: bool = True,
        diameter_metric: str = cn.DIAMETER_IVP,
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
    ) -> pd.DataFrame:
        """
        For each model in BioModels, partition its Jacobians into n_cluster clusters and save
        the max CV of the clusters to a CSV.
        Two partitionation methods are available:
            k-means clustering (partitionJacobians) and sequential partitioning

        Parameters
        ----------
        directory : str
            Path to the directory containing BioModel subdirectories. Defaults to cn.BIOMODELS_DIR.
        start_time : float
            The start time for the simulation.
        output_data_file : str
            Path to the CSV file where results will be saved.
        excluded_models : Optional[List[str]]
            List of model identifiers to exclude from processing.
        end_time : float
            The end time for the simulation. If NaN, it will be determined from LRoadrunner's endtime_dct or default to 10.
        num_point : int
            The number of time points to simulate.
        n_cluster : int
            Number of clusters of Jacobians for timepoints to use for k-means clustering.
        is_report : bool
            Whether to print report messages during processing.
        is_sequential_partition : bool
            Whether to use sequential partitioning instead of k-means clustering.
        diameter_metric : str
            The metric to use for calculating the diameter of each cluster.
        first_model_num : int
            The first model number to include (inclusive).
        last_model_num : int
            The last model number to include (inclusive).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results for each model.
        """
        excluded_models = excluded_models if excluded_models is not None else []
        if excluded_models is None:
            excluded_models = []
        iterator = BiomodelsIterator(
            biomodels_dir=directory,
            excluded_models=excluded_models,
            existing_csv_path=output_data_file,
            is_report=is_report,
            first_model_num=first_model_num,
            last_model_num=last_model_num,
        )
        existing_df = iterator._existing_df
        ##
        def _write_csv(result_dct: Dict[str, float]) -> pd.DataFrame:
            """Write the given results to the output CSV, appending to existing data."""
            df = pd.DataFrame(result_dct)
            df = pd.concat([existing_df, df], ignore_index=False) if not existing_df.empty else df
            df.set_index(cn.COL_MODEL_NAME, inplace=True)
            df.to_csv(output_data_file, header=True, index=True)
            return df
        ##
        # Iterate over models and append results to CSV after each model is processed
        col_names = list(set(cn.COL_NAMES) - {cn.COL_ENDTIME_SOURCE})
        result_dct: dict = {c: [] for c in col_names}
        for item in iterator:
            try:
                biomodels_cluster = BiomodelsCluster(item.model_name, 
                        start_time=start_time, num_point=num_point,
                        diameter_metric=diameter_metric)
            except Exception as e:
                print(e)
                continue
            cjc = biomodels_cluster.cluster(n_cluster=n_cluster, is_sequential_partition=is_sequential_partition)
            result_dct[cn.COL_MODEL_NAME].append(item.model_name)
            result_dct[cn.COL_MAXCV].append(cjc.max_cv)
            result_dct[cn.COL_ENDTIME].append(biomodels_cluster.end_time)
            result_df = _write_csv(result_dct)
        #
        result_df = _write_csv(result_dct)
        return result_df