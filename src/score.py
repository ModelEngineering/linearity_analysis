'''Calculates the Accuracy score for predictions.'''

"""
Accuracy score is a pointwise score for each species and timepoint. At each timepoint, we calculate
the Absolute Percentage Error (APE) between the predicted and true values that is clipped to the range [0, 1].
APE = clip(abs(prediction - true) / true), 0, 1) with sentinel -1 for invalid (zero or non-finite) values.
Accuracy = 1 - APE, with sentinel -1 for invalid (zero or non-finite) values.
"""

from src.plot_options import PlotOptions  # type: ignore
import src.constants as cn  # type: ignore
from src.statistic_calculator import StatisticCalculator # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class Score:
    """Scores prediction timecourses against true timecourses using zero-floor Accuracy.

    Results are accumulated via StatisticAccumulator and persisted to CSV.
    """
    SERIALIZATION_PATH = "score.csv"  # Default path for persistence, can be overridden in subclasses.

    def __init__(self, serialization_path: str = "", is_persist: bool = True,
            accuracy_percentile_for_model_aggregation: int = 90) -> None:
        """
        Parameters
        ----------
        serialization_path : str
            Path to a CSV file for persistence.
        is_persist : bool
            Whether to persist the DataFrame to the CSV file.
        accuracy_percentile_for_model_aggregation : int
            Percentile of accuracy values to use for model-level aggregation (0-100). Default: 90.
        """
        self._is_persist = is_persist
        if len(serialization_path) == 0:
            serialization_path = self.SERIALIZATION_PATH
        self._serialization_path = serialization_path
        self._accuracy_percentile_for_model_aggregation = accuracy_percentile_for_model_aggregation
        #
        self.score_df = pd.DataFrame()

    @property
    def serialization_path(self) -> str:
        return self._serialization_path

    @staticmethod 
    def calculateAccuracy(true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes a [0, 1] clipped accuracy score for each species and timepoint. The accuracy is defined as
        Accuracy = max(0, 1 - (abs(prediction - true) / true)).
        If the true value is undefined or zero, Accuracyis set to -1 as a sentinel
        indicating an invalid measurement (these values are excluded from aggregation).

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same structure as the input dataframes, containing the Accuracy values.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ape_df = (prediction_df - true_df) / true_df
            ape_df = ape_df.abs()
            # Clip valid values to [0, 1] range first.
            ape_df = ape_df.clip(lower=0, upper=1)
            # Then mark undefined/zero-true values as -1 sentinel AFTER clipping.
            # This must happen after clip so that the -1 sentinel is not converted
            # to 0 by clip(lower=0), which would mask bad predictions from aggregation.
            invalid_mask = (true_df == 0) | ~np.isfinite(ape_df)
            accuracy_df = 1 - ape_df
            accuracy_df = accuracy_df.where(~invalid_mask, other=-1.0)
        max_value = accuracy_df.max().max()
        if max_value > 1.0:
            raise ValueError(f"Accuracy values should not exceed 1.0, but found")
        return accuracy_df

    @classmethod
    def calculateAccuracyPercentile(cls, true_df: pd.DataFrame, prediction_df: pd.DataFrame,
            percentile: int=90) -> pd.Series:
        """
        Calculates the percentile of Accuracy values for each species across all timepoints.

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.
        percentile : int
            The percentile to compute (0-100). Default is 90.

        Returns
        -------
        pd.Series
            A Series containing the Accuracy values for each timepoint and species.
        """
        accuracy_df = cls.calculateAccuracy(true_df, prediction_df)
        return accuracy_df.quantile(q=percentile / 100, axis=0)

    def add(self,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            system_id: str = "",
            ) -> pd.DataFrame:
        """Computes scores and accumulates statistics for model-level and per-species aggregations.

        Parameters
        ----------
        system_id : str
            Descriptive ID stored in each aggregation row.
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with the same structure.

        Returns
        -------
        pd.DataFrame
            The full score DataFrame after this addition.
        """
        statistic_calculator = StatisticCalculator()
        # Calculate model-level aggregation (across all species and timepoints)
        model_accuracy_ser = self.calculateAccuracyPercentile(true_timecourse_df, prediction_timecourse_df, self._accuracy_percentile_for_model_aggregation)
        statistic_calculator.add(cn.COL_AGGREGATION_TYPE_MODEL, model_accuracy_ser.to_numpy())
        # Species level aggregations (one per species column, across all timepoints)
        accuracy_df: pd.DataFrame = self.calculateAccuracy(true_timecourse_df, prediction_timecourse_df)
        species_names = list(accuracy_df.columns)
        for species_name in species_names:
            statistic_calculator.add(species_name, accuracy_df[species_name].to_numpy())
        # Add the system ID
        score_df = statistic_calculator.dataframe.copy()
        score_df[cn.COL_SYSTEM_ID] = system_id
        score_df = score_df.rename(columns={cn.COL_LABEL: cn.COL_AGGREGATION_TYPE})
        self.score_df = pd.concat([self.score_df, score_df], ignore_index=True)
        # Serialize the accumulated statistics
        if self._is_persist:
            self.score_df.to_csv(self.serialization_path, index=False)
        return self.score_df

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
        df = self.score_df
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
            if cn.COL_AGGREGATION_TYPE in df.columns:
                model_df = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
                if not model_df.empty and metric_name in model_df.columns:
                    value_arr = np.array(model_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        if is_plot_species:
            if cn.COL_AGGREGATION_TYPE in df.columns:
                species_df = df[df[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
                if not species_df.empty and metric_name in species_df.columns:
                    value_arr = np.array(species_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        plot_options.apply()
        #
        return plot_options