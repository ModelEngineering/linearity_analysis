"""Manage scores of predictions against true timecourses using R-squared."""

from src.dataframe_serializer import DataframeSerializer  # type: ignore
from src.score import ScoreInfo, Score

import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import cast, List, Optional  # type: ignore

R2_METRICS = ["mean", "min", "max", "count"]  \
        + [f"p{int(percentile)}" for percentile in [25.0, 30.0, 50.0, 80.0, 95.0, 99.0]]
SERIALIZATION_PATH = "r2_score.csv"



#########################################
class R2ScoreInfo(ScoreInfo):
    """A container for storing R2 (R-squared) scores."""

    # Must use same name as AREScoreInfo.METRICS to ensure compatibility with ScoreInfo constructor
    PERCENTILES = [25.0, 30.0, 50.0, 80.0, 95.0, 99.0]
    METRICS = ["mean", "min", "max", "count"] + [Score.makePercentileName(p) for p in PERCENTILES]

    def __init__(self,
            description: str = "",
            aggregation_type: str = "",  # "model" or species name
            **kw_metrics) -> None:
        kwargs = dict(kw_metrics)
        for metric in R2ScoreInfo.METRICS:
            kwargs.setdefault(metric, float("nan"))
        super().__init__(description=description,
                aggregation_type=aggregation_type,
                **kwargs)


#########################################
class R2Score(Score):
    """Scores prediction timecourses against true timecourses.

    R2 = 1 - std(prediction - true) / std(true)
    Results are stored as R2ScoreInfo objects and persisted to CSV.
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

    def _computeR2(self, true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.Series:
        """
        Computes the R-squared value per species between true and prediction dataframes.
        R² = 1 - sum((prediction - true)^2) / sum((true - mean(true))^2)
        Returns a Series indexed by species name with one R² value per column.
        If the denominator is zero or undefined, R² is set to -1.

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure.  
        
        Returns
        -------
        pd.Series
            R² values indexed by species name.
        """
        ss_res = np.sum((prediction_df - true_df) ** 2, axis=0)
        ss_tot = np.sum((true_df - true_df.mean()) ** 2, axis=0)

        r2_ser = 1 - ss_res / ss_tot
        valid = r2_ser > 0
        r2_ser[~valid] = -1.0  # Set R² to -1 for undefined cases
        return r2_ser

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
        score_ser: pd.Series = self._computeR2(true_timecourse_df, prediction_timecourse_df)
        # Model level aggregation (all species combined)
        if len(score_ser) == 0:
            model_arr = np.array([], dtype=float)
        else:
            model_arr = np.asarray(score_ser.values, dtype=float)
            species_names = score_ser.index.tolist()
        # Model aggregations
        r2_score_info = self._makeR2ScoreInfo(model_arr)
        r2_score_info.description = description
        r2_score_info.aggregation_type = "model"
        score_infos = [r2_score_info]
        # Species level aggregations
        species_names = score_ser.index.tolist()
        for species_name in species_names:
            species_val = np.array(float(score_ser[species_name]))
            r2_score_info = self._makeR2ScoreInfo(np.array([species_val], dtype=float))
            r2_score_info.description = description
            r2_score_info.aggregation_type = species_name
            score_infos.append(r2_score_info)
        return cast(List[ScoreInfo], score_infos)

    def _makeR2ScoreInfo(self, arr: np.ndarray) -> R2ScoreInfo:
        """Computes a ScoreInfo for R2

        Parameters
        ----------
        arr : np.ndarray
            Array of metric values (R2 or R²).

        Returns
        -------
        R2ScoreInfo
            A ScoreInfo instance with the aggregated statistics.
        """
        LARGE_VAL = 1e6
        # Make a writable copy to avoid "assignment destination is read-only" errors
        # when the input comes from pandas Series values (which can be read-only views).
        arr = np.array(arr, dtype=float)
        # Handle empty array case
        if len(arr) == 0:
            return R2ScoreInfo(
                description="",
                aggregation_type="",
                mean=float("nan"),
                min=float("nan"),
                max=float("nan"),
                count=0,
                **{Score.makePercentileName(p): float("nan") for p in R2ScoreInfo.PERCENTILES}
            )
        sel = np.isnan(arr) | np.isinf(arr) | (arr > LARGE_VAL)
        arr[sel] = LARGE_VAL
        count = int(np.sum(~np.isnan(arr)))
        # Compute percentiles
        kw_percentile = {Score.makePercentileName(p): float(np.nanpercentile(arr, p))
                for p in R2ScoreInfo.PERCENTILES}
        r2_score_info = R2ScoreInfo(
                mean=float(np.nanmean(arr)),
                min=float(np.nanmin(arr)),
                max=float(np.nanmax(arr)),
                count=count,
                **kw_percentile  # type: ignore
        )
        return r2_score_info
