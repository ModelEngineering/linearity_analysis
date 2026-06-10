"""Scores prediction timecourses against true timecourses using absolute relative error (ARE)."""

from src.dataframe_serializer import DataframeSerializer  # type: ignore

import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Optional  # type: ignore

AGGREGATION_DESCRIPTION = "description"
AGGREGATION_MEAN = "mean"
AGGREGATION_MIN = "min"
AGGREGATION_MAX = "max"
AGGREGATION_MEDIAN = "median"
AGGREGATION_PERCENTILE_25 = "p25"
AGGREGATION_PERCENTILE_30 = "p30"
AGGREGATION_PERCENTILE_75 = "p75"
AGGREGATION_PERCENTILE_95 = "p95"
AGGREGATION_PERCENTILE_99 = "p99"
AGGREGATION_COUNT = "count"
AGGREGATION_TYPE = "aggregation_type"
RESULT_IDX = "result_idx"
DEFAULT_AGGREGATIONS = [AGGREGATION_MEAN, AGGREGATION_MIN, AGGREGATION_MAX,
        AGGREGATION_PERCENTILE_25, AGGREGATION_PERCENTILE_30,
        AGGREGATION_PERCENTILE_75, AGGREGATION_PERCENTILE_95]
DEFAULT_PERCENTILES = [25.0, 30.0, 75.0, 95.0, 99.0]
SERIALIZATION_PATH = "score_serialization.csv"

_SCOREINFO_COLS = [RESULT_IDX, AGGREGATION_DESCRIPTION, AGGREGATION_TYPE,
        AGGREGATION_MEAN, AGGREGATION_MIN, AGGREGATION_MAX, AGGREGATION_MEDIAN,
        AGGREGATION_PERCENTILE_25, AGGREGATION_PERCENTILE_30,
        AGGREGATION_PERCENTILE_75, AGGREGATION_PERCENTILE_95,
        AGGREGATION_PERCENTILE_99, AGGREGATION_COUNT]
_PERCENTILE_ATTRS = {AGGREGATION_PERCENTILE_25, AGGREGATION_PERCENTILE_30,
        AGGREGATION_PERCENTILE_75, AGGREGATION_PERCENTILE_95,
        AGGREGATION_PERCENTILE_99}



#########################################
class ScoreInfo(object):

    def __init__(self,
            description: str = "",
            aggregation_type: str = "",
            mean: float = np.nan,
            min: float = np.nan,
            max: float = np.nan,
            median: float = np.nan,
            p25: float = np.nan,
            p30: float = np.nan,
            p75: float = np.nan,
            p95: float = np.nan,
            p99: float = np.nan,
            count: int = -1) -> None:
        self.description = description
        self.aggregation_type = aggregation_type
        self.mean = mean
        self.min = min
        self.max = max
        self.median = median
        self.p25 = p25
        self.p30 = p30
        self.p75 = p75
        self.p95 = p95
        self.p99 = p99
        self.count = count


#########################################
class Score:
    """Scores prediction timecourses against true timecourses.

    ARE = (prediction - true) / true.
    Results are stored as ScoreInfo objects and persisted to CSV.
    """

    def __init__(self, serialization_path: str = SERIALIZATION_PATH,
            is_ignore_first_prediction: bool = True,
            is_initialize: bool = False,
            min_true_value: float = 0.01) -> None:
        """
        Parameters
        ----------
        serialization_path : str
            Path to a CSV file for persistence.
        is_ignore_first_prediction : bool
            Whether to ignore the first prediction when computing scores, since it may be an outlier.
        is_initialize : bool
            Whether to initialize the CSV file by writing an empty DataFrame with the appropriate columns.
        min_true_value : float
            Minimum value for true timecourse to avoid division by zero.
        """
        self._min_true_value = min_true_value
        self._serializer = DataframeSerializer(serialization_path,
                is_initialize=is_initialize)
        self._serialization_path = serialization_path
        self._is_ignore_first_prediction = is_ignore_first_prediction
        if is_initialize:
            self._serializer.serialize([])

    @property
    def score_df(self) -> pd.DataFrame:
        return self._serializer.dataframe

    def addTestResult(self, true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            description: str = "") -> None:
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
        """
        score_infos = self.makeScoreInfo(description, true_timecourse_df,
                prediction_timecourse_df)
        self._serializer.serialize([info.__dict__ for info in score_infos])

    @classmethod
    def deserialize(cls, path: str) -> 'Score':
        """Reconstructs a Score from a CSV written by serialize().

        Parameters
        ----------
        path : str
            Path to the CSV file previously written by serialize().

        Returns
        -------
        Score
            Restored Score with _serialization_path set to path.
        """
        score = cls(serialization_path=path)
        return score

    def _computeARE(self, true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Computes ARE = abs(prediction - true) / true; NaN where true == 0."""
        with np.errstate(divide='ignore', invalid='ignore'):
            are_arr = np.where(
                    true_df.values <= self._min_true_value,
                    np.nan,
                    np.abs(prediction_df - true_df) / np.abs(true_df.values)
            )
        if self._is_ignore_first_prediction:
            first_idx = 1
        else:
            first_idx = 0
        idx_arr = np.array(true_df.index)[first_idx:]
        return pd.DataFrame(are_arr[first_idx:, :], index=idx_arr, columns=true_df.columns)

    def plot(self, is_model_aggregation: bool = True,
            column_name: str = AGGREGATION_MEAN,
            ax = None,
            num_bin: int = 20) -> mfigure.Figure:  # type: ignore
        """Histogram plot of time aggregations over species.

        Parameters
        ----------
        is_model_aggregation : bool
            Whether to plot model-level aggregation (True) or species-level (False).
        column_name : str
            Name of the column to plot.
        ax : Optional[plt.Axes]
            Axes to draw on. A new figure is created if None.
        num_bin : int
            Number of bins for the histogram.

        Returns
        -------
        mfigure.Figure
        """
        if is_model_aggregation:
            aggregation_type = "model"
        else:
            aggregation_type = "species"
        if is_model_aggregation:
            agg_df = self.score_df[self.score_df[AGGREGATION_TYPE] == "model"]
        else:
            agg_df = self.score_df[self.score_df[AGGREGATION_TYPE] != "model"]
        arr = agg_df[column_name].values
        #arr = np.min(arr, 1)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        assert isinstance(fig, mfigure.Figure)
        ax.hist(arr, bins=num_bin)
        ax.set_xlabel(f"{column_name} abs relative error")
        ax.set_ylabel("count")
        ax.set_title(aggregation_type)
        fig.tight_layout()
        plt.show()
        return fig

    def makeScoreInfo(self,
            description: str,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame) -> List[ScoreInfo]:
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
        are_df = self._computeARE(true_timecourse_df, prediction_timecourse_df)
        # Model level aggregation
        score_info = self._makeBasicScoreInfo(are_df)
        score_info.description = description
        score_info.aggregation_type = "model"
        score_infos = [score_info]
        # Species level aggregations
        for species_name in are_df.columns:
            species_are_df = pd.DataFrame(are_df[species_name])
            score_info = self._makeBasicScoreInfo(species_are_df)
            score_info.description = description
            score_info.aggregation_type = species_name
            score_infos.append(score_info)
        return score_infos

    def _makeBasicScoreInfo(self, are_df: pd.DataFrame) -> ScoreInfo:
        """Computes a ScoreInfo from the DataFrame

        Parameters
        ----------
        are_df : pd.DataFrame
            DataFrame containing the ARE values.

        Returns
        -------
        ScoreInfo
            A ScoreInfo instance with the aggregated statistics.
        """
        LARGE_ARE = 1e6
        #
        are_arr = are_df.values.flatten()
        sel = np.isnan(are_arr) | np.isinf(are_arr) | (are_arr > LARGE_ARE)
        are_arr[sel] = LARGE_ARE
        count = int(np.sum(~np.isnan(are_arr)))
        score_info = ScoreInfo(
                mean=float(np.nanmean(are_arr)),
                min=float(np.nanmin(are_arr)),
                max=float(np.nanmax(are_arr)),
                median=float(np.nanmedian(are_arr)),
                p25=float(np.nanpercentile(are_arr, 25)),
                p30=float(np.nanpercentile(are_arr, 30)),
                p75=float(np.nanpercentile(are_arr, 75)),
                p95=float(np.nanpercentile(are_arr, 95)),
                p99=float(np.nanpercentile(are_arr, 99)),
                count=count)
        return score_info
