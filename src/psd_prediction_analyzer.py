'''Analyzes predictions from a Piecewise System Discovery object.'''

import src.constants as cn  # type: ignore
from psd_prediction_file_iterator import PSDPredictionFileIterator  # type: ignore

import pandas as pd  # type: ignore

DEFAULT_MAX_CHANGEPOINT = 5000


class PSDPredictionAnalyzer:

    def __init__(self, is_pkl: bool = True, data_dir: str = cn.PSD_DATA_DIR) -> None:
        """Construct a PSDPredictionAnalyzer object.

        Sets up an iterator over piecewise prediction pickle files for use by analysis methods.

        Args:
            is_pkl (bool): If True, the iterator will look for pickle files; if
                is_pkl (bool): If True, the iterator will look for pickle files; if False, it will look for CSV files.
            data_dir (str): The directory containing the data files.
        """
        self.data_dir = data_dir
        self.psd_iterator = PSDPredictionFileIterator(is_pkl=is_pkl, data_dir=data_dir)

    def getAUCConsistencyDF(self) -> pd.DataFrame:
        """Gets data for analyzing the consistency of the AUC (Area Under the Curve) for a given system.

        Returns:
            pd.DataFrame: A DataFrame containing the analysis results, including the AUC values and
            their consistency metrics. Only provides "model" aggregation type and max_changepoint=5000.
        """
        def _filter_dataframe(df: pd.DataFrame, dct: dict) -> pd.DataFrame:
            filtered_columns = [cn.COL_SYSTEM_ID, cn.COL_AUC, cn.COL_P50, cn.COL_NUM_CHANGEPOINT,
                    cn.COL_MAX_CHANGEPOINT, cn.COL_MAX_FRACTIONAL_REDUCTION,
                    cn.COL_AGGREGATION_TYPE,
                ]
            try:
                mask = df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
                mask &= df[cn.COL_MAX_CHANGEPOINT] == DEFAULT_MAX_CHANGEPOINT
                result_df = df[mask][filtered_columns]
                return result_df
            except Exception as e:
                raise ValueError(f"Error filtering DataFrame for {dct}: {e}")

        filtered_df = self.psd_iterator.query(data_frame_filter=_filter_dataframe,
                data_dir=self.data_dir)
        return filtered_df

    def analyzeAUCConsistency(self, system_id: str) -> pd.DataFrame:
        """Analyze the consistency of the AUC (Area Under the Curve) for a given system.

        Args:
            system_id (str): The identifier of the system being analyzed.
        Returns:
            pd.DataFrame: A DataFrame containing the analysis results, including the AUC values and
            their consistency metrics.
        """
        filtered_df = self.getAUCConsistencyDF()
        filtered_df = filtered_df[filtered_df[cn.COL_SYSTEM_ID] == system_id]
        grouped_df = filtered_df.groupby(cn.COL_MAX_FRACTIONAL_REDUCTION).agg({
            cn.COL_AUC: [cn.COL_MEAN, cn.COL_STD, cn.COL_COUNT],
        })
        grouped_df = grouped_df.fillna(0)

        return grouped_df