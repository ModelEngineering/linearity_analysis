'''Calculates the MAPE (minus absolute percentage error) score for predictions against true timecourses.'''

"""
This module calculates the zero floor, Minus Absolute Percentage Error (MAPE) score for predictions against true timecourses.
These are pointwise scores for each species and timepoint, which are then aggregated to provide a model-level score.
The formula for the MAPE score is:
    MAPE = max(0, 1 - (abs(prediction - true) / true))
"""

from src.plot_options import PlotOptions  # type: ignore
import src.constants as cn  # type: ignore
from src.statistic_calculator import StatisticCalculator # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class Score:
    """Scores prediction timecourses against true timecourses using zero-floor MAPE.

    MAPE = 1 - |prediction - true| / true, with sentinel -1 for invalid (zero or non-finite) values.
    Results are accumulated via StatisticAccumulator and persisted to CSV.
    """
    SERIALIZATION_PATH = "score.csv"  # Default path for persistence, can be overridden in subclasses.

    def __init__(self, serialization_path: str = "",
            is_initialize: bool = False,
            is_persist: bool = True,
            ) -> None:
        """
        Parameters
        ----------
        serialization_path : str
            Path to a CSV file for persistence.
        is_initialize : bool
            Whether to initialize the CSV file by writing an empty DataFrame with the appropriate columns.
        is_persist : bool
            Whether to persist the DataFrame to the CSV file.
        """
        self._is_persist = is_persist
        if len(serialization_path) == 0:
            serialization_path = self.SERIALIZATION_PATH
        self._serialization_path = serialization_path
        #
        self.score_df = pd.DataFrame()

    @property
    def serialization_path(self) -> str:
        return self._serialization_path

    @staticmethod 
    def calculateMAPE(true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the zero floor, Minus Absolute Percentage Error (MAPE) between true and prediction dataframes.
        MAPE = max(0, 1 - (abs(prediction - true) / true)).
        If the true value is undefined or zero, the MAPE is set to -1 as a sentinel
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
            A DataFrame with the same structure as the input dataframes, containing the MAPE values.
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
            mape_df = 1 - ape_df
            mape_df = mape_df.where(~invalid_mask, other=-1.0)
        max_value = mape_df.max().max()
        if max_value > 1.0:
            raise ValueError(f"MAPE values should not exceed 1.0, but found")
        return mape_df

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
        mape_df: pd.DataFrame = self.calculateMAPE(true_timecourse_df, prediction_timecourse_df)
        statistic_calculator.add(cn.COL_AGGREGATION_TYPE_MODEL, mape_df.values.flatten())
        # Species level aggregations (one per species column, across all timepoints)
        species_names = list(mape_df.columns)
        for species_name in species_names:
            statistic_calculator.add(species_name, mape_df[species_name].to_numpy())
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