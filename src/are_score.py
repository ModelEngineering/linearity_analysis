"""Manage scores of predictions against true timecourses."""

"""
Calculates, aggregates and persists scores of
predictions against true timecourses. Scores are computed for each
univariate timecourse (species) and aggregated across species. The score is a
measure of error between the predicted and true timecourses. Two
measures of error are supported: absolute relative error (ARE) and
r-squared. ARE is defined as (prediction - true) / true. R-squared is defined as
1 - (sum((prediction - true)^2) / sum((true - mean(true)).
"""

from src.dataframe_serializer import DataframeSerializer  # type: ignore
from src.score import ScoreInfo, Score

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import cast, List  # type: ignore

SERIALIZATION_PATH = "are_score.csv"


#########################################
class AREScoreInfo(ScoreInfo):
    """A container for storing ARE (Absolute Relative Error) scores."""

    # Must use same name as AREScoreInfo.METRICS to ensure compatibility with ScoreInfo constructor
    PERCENTILES: List[float] = [25.0, 30.0, 50.0, 80.0, 95.0, 99.0]
    METRICS: List[str] = ["mean", "min", "max", "count"] + [Score.makePercentileName(p) for p in PERCENTILES]

    def __init__(self,
            description: str = "",
            aggregation_type: str = "",  # "model" or species name
            **kw_metrics) -> None:
        kwargs = dict(kw_metrics)
        for metric in AREScoreInfo.METRICS:
            kwargs.setdefault(metric, float("nan"))
        super().__init__(description=description,
                aggregation_type=aggregation_type,
                **kwargs)


# Module-level metrics list (computed after class definition to avoid forward reference)
ARE_METRICS: List[str] = ["mean", "min", "max", "count"]  \
        + [Score.makePercentileName(percentile) for percentile in AREScoreInfo.PERCENTILES]


#########################################
class AREScore(Score):
    """Scores prediction timecourses against true timecourses.

    ARE = (prediction - true) / true.
    Results are stored as ScoreInfo objects and persisted to CSV.
    """

    def __init__(self, serialization_path: str = SERIALIZATION_PATH,
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
        super().__init__(serialization_path=serialization_path,
                is_ignore_first_prediction=is_ignore_first_prediction,
                is_initialize=is_initialize)

    def _computeARE(self, true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the absolute relative error (ARE) between true and prediction dataframes.
        ARE = abs((prediction - true) / true).
        Returns a DataFrame with the same structure as the input dataframes.
        If the true value is undefined or zero, the ARE is set to -1 as a sentinel
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
            A DataFrame with the same structure as the input dataframes, containing the ARE values.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            are_df = (prediction_df - true_df) / true_df
            are_df = are_df.abs()
            # Mark undefined/zero-true values as -1 sentinel (before clipping,
            # since clip(lower=0) would turn -1 into 0 and mask bad predictions).
            are_df = are_df.where(are_df >= 0, other=-1.0)
            are_df = are_df.clip(lower=0, upper=1)
        if self._is_ignore_first_prediction:
            first_idx = 1
        else:
            first_idx = 0
        return are_df.iloc[first_idx:, :]

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
        score_df: pd.DataFrame = self._computeARE(true_timecourse_df, prediction_timecourse_df)

        # Model level aggregation (all species and timepoints combined)
        model_arr = np.asarray(score_df.values.flatten(), dtype=float)
        score_info = self._makeBasicScoreInfo(model_arr)
        score_info.description = description
        score_info.aggregation_type = "model"
        score_infos = [score_info]

        # Species level aggregations (one per species column, across all timepoints)
        species_names = list(score_df.columns)
        for species_name in species_names:
            species_arr = np.asarray(score_df[species_name].values, dtype=float)
            score_info = self._makeBasicScoreInfo(species_arr)
            score_info.description = description
            score_info.aggregation_type = species_name
            score_infos.append(score_info)
        return cast(List[ScoreInfo], score_infos)

    def _makeBasicScoreInfo(self, arr: np.ndarray) -> AREScoreInfo:
        """Computes a ScoreInfo from an array of values.

        Parameters
        ----------
        arr : np.ndarray
            Array of metric values (ARE or R²). Negative sentinel values (-1)
            indicate invalid measurements and are excluded from aggregation.

        Returns
        -------
        AREScoreInfo
            A ScoreInfo instance with the aggregated statistics.
        """
        LARGE_VAL = 1e6
        # Filter out negative sentinel values (invalid/undefined true values).
        valid_mask = arr >= 0
        valid_arr = arr[valid_mask].copy()

        count = int(len(valid_arr))
        if count == 0:
            return AREScoreInfo(
                description="",
                aggregation_type="",
                mean=float("nan"),
                min=float("nan"),
                max=float("nan"),
                count=0,
                **{Score.makePercentileName(p): float("nan") for p in AREScoreInfo.PERCENTILES}
            )

        # Replace remaining NaN/inf/large values with LARGE_VAL for aggregation.
        sel = np.isnan(valid_arr) | np.isinf(valid_arr) | (valid_arr > LARGE_VAL)
        valid_arr[sel] = LARGE_VAL

        kw_percentile = {Score.makePercentileName(p): float(np.nanpercentile(valid_arr, p))
                for p in AREScoreInfo.PERCENTILES}
        score_info = AREScoreInfo(
                mean=float(np.nanmean(valid_arr)),
                min=float(np.nanmin(valid_arr)),
                max=float(np.nanmax(valid_arr)),
                count=count,
                **kw_percentile  # type: ignore
        )
        return score_info
