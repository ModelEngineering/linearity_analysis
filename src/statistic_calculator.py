'''Calculates univariate descriptive statistics'''

# Statistics that begin with 'p' are percentiles, e.g. p25 is the 25th percentile
# of the valid values used in calculating the statistics.
# Assumes that all statistics
"""
Usage:
    statistics = StatisticCalculator()
    statistics.add(label="model1", value_arr=np.array([1, 2, 3, -1, 4]))
    statistics.add(label="model2", value_arr=np.array([5, 6, 7, 8, 9]))
    df = statistics.dataframe
"""


import src.constants as cn  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

LARGE_VAL = 1e6


#########################################
class StatisticCalculator(object):
    # count is the number of valid values used in calculating the statistics.
    # invalid_count is the number of invalid (sentinel -1) values excluded from aggregation.

    """A container for storing statistics accumulated in a dictionary"""
    def __init__(self) -> None:
        self.statistic_dct: dict = {n: [] for n in cn.STATISTICS}
        self.statistic_dct[cn.COL_LABEL] = []

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame of the accumulated statistics.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the accumulated statistics.
        """
        return pd.DataFrame(self.statistic_dct)
    
    def _is_percentile(self, stat_name: str) -> bool:
        """Returns True if the statistic name is a percentile (e.g. p25, p50, etc.)"""
        return stat_name.startswith("p") and stat_name[1:].isdigit() and 0 <= int(stat_name[1:]) <= 100

    def add(self,
            label: str,
            value_arr: np.ndarray = np.array([]),
            is_non_negative: bool = True,
    ) -> None:
        """Computes statistics and accumulates them in the internal dictionary (self.statistic_dct).

        Parameters
        ----------
        label : str
            Descriptive label for this aggregation.
        value_arr : np.ndarray
            1D Array of values. Negative sentinel values (-1)
            indicate invalid measurements and are excluded from aggregation.
        is_non_negative : bool
            If True, only non-negative values are considered valid for aggregation.

        Returns
        -------
        None
            Modifies internal state (self.statistic_dct) only.
        """
        value_arr = value_arr.flatten()  # Flatten to 1D for aggregation
        # Filter out NaN, inf, and optionally negative values.
        if is_non_negative:
            valid_mask = [not (np.isnan(v) or np.isinf(v)) and v >= 0 for v in value_arr]
        else:
            valid_mask = [not (np.isnan(v) or np.isinf(v)) for v in value_arr]
        count = int(np.sum(valid_mask))  # Count of valid values used in aggregation
        invalid_count = len(value_arr) - int(np.sum(valid_mask))

        # If no input values at all, still record a row so the CSV reflects that data was collected.
        if (len(value_arr) == 0) or (count == 0):
            self.statistic_dct[cn.COL_LABEL].append(label)
            self.statistic_dct[cn.COL_MEAN].append(np.nan)
            self.statistic_dct[cn.COL_MIN].append(np.nan)
            self.statistic_dct[cn.COL_MAX].append(np.nan)
            self.statistic_dct[cn.COL_COUNT].append(0)
            self.statistic_dct[cn.COL_INVALID_COUNT].append(invalid_count)
            for p in [p for p in cn.STATISTICS if self._is_percentile(p)]:
                self.statistic_dct[p].append(np.nan)
            return

        # Replace large values with LARGE_VAL for aggregation.
        valid_arr = value_arr[valid_mask].copy()
        sel = valid_arr > LARGE_VAL
        valid_arr[sel] = LARGE_VAL
        # Update dictionary
        self.statistic_dct[cn.COL_LABEL].append(label)
        self.statistic_dct[cn.COL_MEAN].append(float(np.nanmean(valid_arr)))
        self.statistic_dct[cn.COL_MIN].append(float(np.nanmin(valid_arr)))
        self.statistic_dct[cn.COL_MAX].append(float(np.nanmax(valid_arr)))
        self.statistic_dct[cn.COL_COUNT].append(count)
        self.statistic_dct[cn.COL_INVALID_COUNT].append(invalid_count)
        percentiles = [p for p in cn.STATISTICS if self._is_percentile(p)]
        for p in percentiles:
            self.statistic_dct[p].append(float(np.nanpercentile(valid_arr, int(p[1:]))))