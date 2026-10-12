'''Analyzes predictions from a Piecewise System Discovery object.'''

import src.constants as cn  # type: ignore
from src.psd_prediction_files_iterator import PSDPredictionFilesIterator  # type: ignore

import pandas as pd  # type: ignore

DEFAULT_MAX_CHANGEPOINT = 5000


class PSDPredictionAnalyzer:

    def __init__(self, is_pkl: bool = True) -> None:
        """Construct a PSDPredictionAnalyzer object.

        Sets up an iterator over piecewise prediction pickle files for use by analysis methods.

        Args:
            is_pkl (bool): If True, the iterator will look for pickle files; if
        """
        self.psd_iterator = PSDPredictionFilesIterator(is_pkl=is_pkl)

    def analyzeAUCConsistency(self, system_id: str) -> pd.DataFrame:
        """Analyze the consistency of the AUC (Area Under the Curve) for a given system.

        Args:
            system_id (str): The identifier of the system being analyzed.
        Returns:
            pd.DataFrame: A DataFrame containing the analysis results, including the AUC values and
            their consistency metrics.
        """
        def _filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            filtered_columns = [cn.COL_SYSTEM_ID, cn.COL_AUC, cn.COL_P50, cn.COL_NUM_CHANGEPOINT,
                    cn.COL_MAX_CHANGEPOINT, cn.COL_MAX_FRACTIONAL_REDUCTION,
                    cn.COL_SYSTEM_ID
                ]
            mask = df[cn.COL_SYSTEM_ID] == system_id
            mask &= df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
            mask &= df[cn.COL_MAX_CHANGEPOINT] == DEFAULT_MAX_CHANGEPOINT
            return df[mask][filtered_columns]

        filtered_df = self.psd_iterator.query(data_frame_filter=_filter_dataframe)

        grouped_df = filtered_df.groupby(cn.COL_MAX_FRACTIONAL_REDUCTION).agg({
            cn.COL_AUC: [cn.COL_MEAN, cn.COL_STD, cn.COL_COUNT],
        })
        grouped_df = grouped_df.fillna(0)

        return grouped_df