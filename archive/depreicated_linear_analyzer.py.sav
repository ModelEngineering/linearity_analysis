"""
Module for analyzing linearity of SBML and Antimony models via Jacobian analysis.
"""
import src.constants as cn
from trajectory import Trajectory # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from biomodels_iterator import BiomodelsIterator  # type: ignore

import os
import pandas as pd # type: ignore
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np # type: ignore
import tellurium as te # type: ignore
from src.l_roadrunner import LRoadrunner # type: ignore

OUTPUT_DATA_FILE = os.path.join(cn.DATA_DIR, "model_linearity_analysis_data.csv")


class LinearAnalyzer:
    """Analyzes linearity of SBML or Antimony models by collecting Jacobians over time."""

    def __init__(
        self,
        model: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
        num_point: int = 100,
        sedml_str: Optional[str] = None,
    ) -> None:
        """
        Initialize a LinearAnalyzer with a model and simulation parameters.

        Parameters
        ----------
        model : str
            SBML XML string or Antimony model string.
        start_time : float
            Simulation start time (default: 0).
        end_time : float, optional
            Simulation end time (default: 10).
        num_point : int
            Number of simulation timepoints (default: 100).
        sedml_str : str, optional
            SED-ML string for simulation setup (default: None).
        """
        self.start_time = start_time
        self.end_time = end_time
        self.num_point = num_point
        self.sedml_str = sedml_str
        self.l_roadrunner = LRoadrunner(model, start_time=start_time,
                end_time=end_time, num_point=num_point)
        self._jacobian_collection = Trajectory(self.l_roadrunner)
        self.model = model

    @staticmethod
    def _report(text: str):
        """Print a report message if reporting is enabled."""
        print(text)

    @classmethod
    def _getSedml(cls, directory: str) -> Optional[str]:
        """Search for a SED-ML file in the given directory and return its contents as a string."""
        sedml_files = glob.glob(os.path.join(directory, "*.sedml"))
        if not sedml_files:
            return None
        with open(sedml_files[0], "r") as f:
            return f.read()

    @classmethod
    def partitionBiomodelsJacobians(
        cls,
        directory: str = cn.BIOMODELS_DIR,
        output_data_file: str = OUTPUT_DATA_FILE,
        excluded_models: Optional[List[str]] = None,
        n_cluster: int = 3,
        is_sequential_partition: bool = False,
        is_report: bool = True,
    ) -> pd.DataFrame:
        """
        For each model in BioModels, partition its Jacobians into n_cluster clusters and save
        the max CV of the clusters to a CSV.
        Two partitionation methods are available:
            k-means clustering (partitionJacobians) and sequential partitioning

        Parameters
        ----------
        directory : str
            Path to the directory containing BioModel subdirectories. Defaults to
            the local temp-biomodels/final directory.
        output_data_file : str  where the CSV file containing the CV results will be saved.
        excluded_models : Optional[List[str]]
            List of model identifiers to exclude from processing.
        n_cluster : int
            Number of clusters of Jacobians for timepoints to use for k-means clustering.
        is_sequential_partition : bool
            Whether to use sequential partitioning instead of k-means clustering.
        is_report : bool
            Whether to report progress during processing.

        Returns
        -------
        pd.DataFrame
            DataFrame containing
                index: model identifiers (subdirectory names)
                max_cv: max CV of the Jacobian clusters for each model.
                end_time: end time used for each model's simulation.
        """
        if excluded_models is None:
            excluded_models = []
        iterator = BiomodelsIterator(
            biomodels_dir=directory,
            excluded_models=excluded_models,
            existing_csv_path=output_data_file,
            is_report=is_report,
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

            clustered_jacobian_collection = cls.makeBiomodelCusteredJacobianCollection(
                item.model_name,
                directory=directory,
                n_cluster=n_cluster,
                is_sequential_partition=is_sequential_partition,
                is_report=is_report,
            )
            result_dct[cn.COL_MODEL_NAME].append(item.model_name)
            result_dct[cn.COL_MAXCV].append(clustered_jacobian_collection.max_cv)
            result_dct[cn.COL_ENDTIME].append(clustered_jacobian_collection.l_roadrunner.end_time)
            result_df = _write_csv(result_dct)
        #
        result_df = _write_csv(result_dct)
        return result_df
    
    @classmethod
    def makeBiomodelCusteredJacobianCollection(cls,
        model_name: str,
        directory: str = cn.BIOMODELS_DIR,
        start_time: float = cn.START_TIME,
        end_time: float = np.nan,
        num_point: int = cn.NUM_POINTS,
        n_cluster: int = 1,
        is_sequential_partition: bool = True,
        is_report: bool = True,
    ) -> TrajectoryCollection:
        """
        Create a ClusteredJacobianCollection for a single Biomodel.

        Parameters
        ----------
        model_name : str
            The BioModel identifier (e.g. 'BIOMD0000000001').
        directory : str
            Path to the directory containing BioModel subdirectories.
            Defaults to cn.BIOMODELS_DIR.
        start_time : float
            The start time for the simulation.
        num_points : int
            The number of time points to simulate.
        n_cluster : int
            The number of clusters to partition the Jacobians into.
        is_sequential_partition : bool
            Whether to use sequential partitioning instead of k-means clustering.
        is_report : bool
            Whether to print report messages during processing.

        Returns
        -------
        ClusteredJacobianCollection
            A ClusteredJacobianCollection instance containing the clustered Jacobian collections for the model.
        """
        clustered_jacobian_collection = TrajectoryCollection([])
        model_dir = os.path.join(directory, model_name)
        item = BiomodelsIterator.getBiomodelInfo(model_dir)
        sbml_file = item.sbml_paths[0]
        if np.isnan(end_time):
            end_time = LRoadrunner.endtime_dct.get(item.model_name, np.nan) # type: ignore
        if np.isnan(end_time): # type: ignore
            if is_report:
                cls._report(f"Error processing {model_name}: skipping due to missing end time.")
            return clustered_jacobian_collection
        try:
            with open(sbml_file, "r") as f:
                sbml_str = f.read()
            analyzer = cls(sbml_str, start_time=start_time, num_point=num_point, end_time=end_time)
            if is_sequential_partition:
                clustered_jacobian_collection = TrajectoryCollection(
                        analyzer._jacobian_collection.sequentialPartition(
                                n_cluster=n_cluster))
            else:
                clustered_jacobian_collection = TrajectoryCollection(
                        analyzer._jacobian_collection.nonsequentialPartition(
                                n_cluster=n_cluster))
        except Exception as e:
            if is_report:
                cls._report(f"Error processing {model_name}: {e}")
            return clustered_jacobian_collection
        #
        return clustered_jacobian_collection