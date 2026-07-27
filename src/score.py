"""Abstrct class for manage scores of predictions against true timecourses."""

"""
Contains code common to all score types, including ARE and R².
"""

from src.dataframe_serializer import DataframeSerializer  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Optional  # type: ignore

AGGREGATION_DESCRIPTION = "description"
AGGREGATION_MEAN = "mean"
AGGREGATION_MIN = "min"
AGGREGATION_MAX = "max"
AGGREGATION_COUNT = "count"
AGGREGATION_TYPE = "aggregation_type"  # model or species name
DEFAULT_PERCENTILES = [25.0, 30.0, 50.0, 80.0, 95.0, 99.0]
METRIC_TYPE_ARE = "are"
METRIC_TYPE_R2 = "r2"  # r-squared


#########################################
class ScoreInfo(object):

    """A container for storing ARE (Absolute Relative Error) scores."""

    METRICS: List[str] = []

    def __init__(self,
            description: str = "",
            aggregation_type: str = "",  # "model" or species name
            **kw_metrics) -> None:
        # Validate the metrics
        expected_metrics = list(self.METRICS)
        for metric in self.METRICS:
            if metric not in kw_metrics:
                raise ValueError(f"Missing required metric: {metric}")
            expected_metrics.remove(metric)
        if expected_metrics:
            raise ValueError(f"Unexpected metrics: {expected_metrics}")
        # Store the metrics
        self.description = description
        self.aggregation_type = aggregation_type
        for metric in self.METRICS:
            setattr(self, metric, kw_metrics[metric])


#########################################
class Score:
    """Scores prediction timecourses against true timecourses.

    ARE = (prediction - true) / true.
    Results are stored as ScoreInfo objects and persisted to CSV.
    """

    def __init__(self, serialization_path: str = "",
            is_ignore_first_prediction: bool = True,
            is_initialize: bool = False,
            ) -> None:
        """
        Parameters
        ----------
        serialization_path : str
            Path to a CSV file for persistence.
        is_ignore_first_prediction : bool
            Whether to ignore the first prediction when computing scores, since it may be an outlier.
        is_initialize : bool
            Whether to initialize the CSV file by writing an empty DataFrame with the appropriate columns.
        """
        if len(serialization_path) == 0:
            raise ValueError("serialization_path must be provided.")
        self._serializer = DataframeSerializer(serialization_path,
                is_initialize=is_initialize)
        self._is_ignore_first_prediction = is_ignore_first_prediction
        if is_initialize:
            self._serializer.serialize([])

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._serializer.dataframe

    @property
    def serialization_path(self) -> str:
        return self._serializer.serialization_path

    @property
    def score_df(self) -> pd.DataFrame:
        return self._serializer.dataframe
    
    def addTestResult(self,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            description: str = "",
            metric_type: str = METRIC_TYPE_ARE) -> None:
        """
        Adds a test result by computing ScoreInfo from true and prediction timecourses.

        Parameters
        ----------
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with same structure as true_timecourse_df.
        description : str
            Descriptive label for this test result.
        metric_type : str
            Type of metric to compute (ARE or R-squared).
        """
        score_infos = self.makeScoreInfo(description, true_timecourse_df,
                prediction_timecourse_df)
        self._serializer.serialize([info.__dict__ for info in score_infos])

    def makeScoreInfo(self,
            description: str,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            ) -> List[ScoreInfo]:
        """Computes a list of ScoreInfo: one model-level and one per species.

        Parameters
        ----------
        description : str
            Descriptive label stored in each ScoreInfo.
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with the same structure.

        Returns
        -------
        List[ScoreInfo]
            First element covers all species/timepoints (aggregation_type="model");
            subsequent elements cover individual species.
        """
        raise NotImplementedError("makeScoreInfo must be implemented in subclasses.")

    def plotCDF(self, 
            metric_name: str,
            is_plot_model: bool = True,
            is_plot_species: bool = True,
            is_plot: bool = True,
            **plt_kwargs) -> PlotOptions:
        """
        Plots the CDF

        Args:
            metric_name (str): Name of the metric to plot.
            is_plot_model (bool, optional): Whether to plot model-level aggregation. Defaults to True.
            is_plot_species (bool, optional): Whether to plot species-level aggregation. Defaults to True.
            is_plot (bool, optional): Whether to display the plot. Defaults to True.
            **plt_kwargs: Additional keyword arguments for PlotOptions.
        """
        # Get the data
        df = self.dataframe
        # Do the plot
        kwargs = dict(plt_kwargs)
        title = ""
        if is_plot_model and is_plot_species:
            legend = ["model", "species"]
            kwargs.setdefault("legend", legend)
            title = "CDFs"
        elif is_plot_model:
            title = "CDF for models"
        elif is_plot_species:
            title = "CDF for species"
        else:
            raise ValueError("At least one of is_plot_model or is_plot_species must be True.")
        kwargs.setdefault("title", title)
        kwargs.setdefault("xlabel", metric_name)
        kwargs.setdefault("ylabel", "fraction")
        plot_options = PlotOptions(**kwargs)
        ##
        def doPlot(value_arr: np.ndarray, xlabel: str):
            sorted_arr = np.sort(value_arr)
            length = len(sorted_arr)
            yv = np.array(range(length))/length
            plt.plot(sorted_arr, yv)
            if not is_plot:
                plt.close()
        ##
        if is_plot_model:
            if AGGREGATION_TYPE in df.columns:
                model_df = df[df[AGGREGATION_TYPE] == "model"]
                if not model_df.empty and metric_name in model_df.columns:
                    value_arr = np.array(model_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        if is_plot_species:
            if AGGREGATION_TYPE in df.columns:
                species_df = df[df[AGGREGATION_TYPE] == "species"]
                if not species_df.empty and metric_name in species_df.columns:
                    value_arr = np.array(species_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        plot_options.apply()
        #
        return plot_options

    @staticmethod
    def makePercentileName(percentile: float) -> str:
        """Returns the name of a percentile column.

        Parameters
        ----------
        percentile : float
            Percentile value (e.g., 25.0, 50.0).

        Returns
        -------
        str
            Name of the percentile column (e.g., "p25", "p50").
        """
        return f"p{int(percentile)}"