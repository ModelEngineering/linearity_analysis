'''Calculates the MAPE (minus absolute percentage error) score for predictions against true timecourses.'''

"""
This module calculates the zero floor, Minus Absolute Percentage Error (MAPE) score for predictions against true timecourses.
These are pointwise scores for each species and timepoint, which are then aggregated to provide a model-level score.
The formula for the MAPE score is:
    MAPE = max(0, 1 - (abs(prediction - true) / true))
"""

from src.dataframe_serializer import DataframeSerializer  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
import src.constants as cn  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

MEAN = "mean"
MIN = "min"
MAX = "max"
COUNT = "count"

#########################################
class _StatisticAccumulator(object):
    STATISTICS = ["mean", "min", "max", "count", "invalid_count", "p25", "p30",
            "p50", "p80", "p95", "p99"]
    # count is the number of valid values used in calculating the statistics.
    # invalid_count is the number of invalid (sentinel -1) values excluded from aggregation.

    """A container for storing statistics accumulated in a dictionary"""
    def __init__(self) -> None:
        self.statistic_dct: dict = {n: [] for n in self.STATISTICS}
        self.statistic_dct[cn.AGGREGATION_TYPE] = []
        self.statistic_dct[cn.DESCRIPTION] = []

    def add(self,
            value_arr: np.ndarray = np.array([]),
            label: str = "",
            aggregation_type: str = "",  # "model" or species name
    ) -> None:
        """Computes statistics and adds to cumulative values.

        Parameters
        ----------
        value_arr : np.ndarray
            1D Array of MAPE metric values. Negative sentinel values (-1)
            indicate invalid measurements and are excluded from aggregation.
        label : str
            Descriptive label for this aggregation.
        aggregation_type : str
            Type of aggregation, either "model" or a species name.

        Returns
        -------
        None
            Modifies internal state (self.statistic_dct) only.
        """
        value_arr = value_arr.flatten()  # Flatten to 1D for aggregation
        LARGE_VAL = 1e6
        # Filter out negative sentinel values (invalid/undefined true values).
        valid_mask = value_arr >= 0
        invalid_count = int(np.sum(~valid_mask))
        valid_arr = value_arr[valid_mask].copy()

        count = int(len(valid_arr))

        # If no input values at all, still record a row so the CSV reflects that data was collected.
        if len(value_arr) == 0:
            self.statistic_dct[cn.DESCRIPTION].append(label)
            self.statistic_dct[cn.AGGREGATION_TYPE].append(aggregation_type)
            self.statistic_dct[MEAN].append(0.0)
            self.statistic_dct[MIN].append(0.0)
            self.statistic_dct[MAX].append(0.0)
            self.statistic_dct[COUNT].append(0)
            self.statistic_dct["invalid_count"].append(invalid_count)
            for p in [p for p in self.STATISTICS if p.startswith("p")]:
                self.statistic_dct[p].append(0.0)
            return

        # If all values are invalid, still record a row so the CSV reflects that data was collected.
        if count == 0:
            self.statistic_dct[cn.DESCRIPTION].append(label)
            self.statistic_dct[cn.AGGREGATION_TYPE].append(aggregation_type)
            self.statistic_dct[MEAN].append(0.0)
            self.statistic_dct[MIN].append(0.0)
            self.statistic_dct[MAX].append(0.0)
            self.statistic_dct[COUNT].append(0)
            self.statistic_dct["invalid_count"].append(invalid_count)
            for p in [p for p in self.STATISTICS if p.startswith("p")]:
                self.statistic_dct[p].append(0.0)
            return

        # Replace remaining NaN/inf/large values with LARGE_VAL for aggregation.
        sel = np.isnan(valid_arr) | np.isinf(valid_arr) | (valid_arr > LARGE_VAL)
        valid_arr[sel] = LARGE_VAL
        # Update dictionary
        self.statistic_dct[cn.DESCRIPTION].append(label)
        self.statistic_dct[cn.AGGREGATION_TYPE].append(aggregation_type)
        self.statistic_dct[MEAN].append(float(np.nanmean(valid_arr)))
        self.statistic_dct[MIN].append(float(np.nanmin(valid_arr)))
        self.statistic_dct[MAX].append(float(np.nanmax(valid_arr)))
        self.statistic_dct[COUNT].append(count)
        self.statistic_dct["invalid_count"].append(invalid_count)
        percentiles = [p for p in self.STATISTICS if p.startswith("p")]
        for p in percentiles:
            self.statistic_dct[p].append(float(np.nanpercentile(valid_arr, int(p[1:]))))


#########################################
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
        if len(serialization_path) == 0:
            serialization_path = self.SERIALIZATION_PATH
        self._serializer = DataframeSerializer(serialization_path,
                is_initialize=is_initialize, is_persist=is_persist)
        #
        self.statistic_accumulator = _StatisticAccumulator()

    @property
    def serialization_path(self) -> str:
        return self._serializer.serialization_path

    @property
    def score_df(self) -> pd.DataFrame:
        return self._serializer.dataframe

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
            are_df = (prediction_df - true_df) / true_df
            are_df = are_df.abs()
            # Clip valid values to [0, 1] range first.
            are_df = are_df.clip(lower=0, upper=1)
            # Then mark undefined/zero-true values as -1 sentinel AFTER clipping.
            # This must happen after clip so that the -1 sentinel is not converted
            # to 0 by clip(lower=0), which would mask bad predictions from aggregation.
            invalid_mask = (true_df == 0) | ~np.isfinite(are_df)
            are_df = are_df.where(~invalid_mask, other=-1.0)
        return 1 - are_df

    def add(self,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            label: str = "",
            ) -> pd.DataFrame:
        """Computes MAPE scores and accumulates statistics for model-level and per-species aggregations.

        Parameters
        ----------
        label : str
            Descriptive label stored in each aggregation row.
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with the same structure.

        Returns
        -------
        pd.DataFrame
            The full score DataFrame after this addition.
        """
        score_df: pd.DataFrame = self.calculateMAPE(true_timecourse_df, prediction_timecourse_df)

        # Record how many rows were in the accumulator BEFORE this call.
        dct = self.statistic_accumulator.statistic_dct
        start_count = len(dct[cn.DESCRIPTION])

        # Model level aggregation (all species and timepoints combined)
        model_arr = np.asarray(score_df.values, dtype=float)
        self.statistic_accumulator.add(model_arr,
                aggregation_type=cn.AGGREGATION_TYPE_MODEL,
                label=label)

        # Species level aggregations (one per species column, across all timepoints)
        species_names = list(score_df.columns)
        for species_name in species_names:
            species_arr = np.asarray(score_df[species_name].values, dtype=float)
            self.statistic_accumulator.add(species_arr,
                    aggregation_type=species_name,
                    label=label)

        # Build list of dicts from only the NEW rows added in this call.
        end_count = len(dct[cn.DESCRIPTION])
        new_rows: list[dict] = []
        for i in range(start_count, end_count):
            row: dict = {}
            for key in dct:
                if isinstance(dct[key], list):
                    row[key] = dct[key][i]
            new_rows.append(row)

        # Persist the new rows to the DataFrame/CSV.
        self._serializer.serializeDct(new_rows)
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
            if cn.AGGREGATION_TYPE in df.columns:
                model_df = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
                if not model_df.empty and metric_name in model_df.columns:
                    value_arr = np.array(model_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        if is_plot_species:
            if cn.AGGREGATION_TYPE in df.columns:
                species_df = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_SPECIES]
                if not species_df.empty and metric_name in species_df.columns:
                    value_arr = np.array(species_df[metric_name].values)
                    doPlot(value_arr, xlabel=metric_name)
        plot_options.apply()
        #
        return plot_options