'''Constructs the non-linearity curve (NLCurve) for analyzing segments of multivariate timecourses.'''

"""
This module analyzes segment lengths induced by changepoints in a timecourse.
The NLCurve is a plot of the culmulative fraction of the timecourse that is contained in segments of length
less than or equal to a given length.
The area under the NLCurve is a measure of the non-linearity of the timecourse.
"""

import ast  # type: ignore
import src.constants as cn
from src.model import Model  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os  # type: ignore
import pandas as pd # type: ignore
from typing import List, Union, Optional, cast # type: ignore


class NLCurve(object):

    def __init__(self, changepoints: List[Union[int, float]]) -> None:
        self._changepoints = list(np.sort(changepoints))
        self._segment_length_arr = np.diff(self._changepoints)
        #
        self.nl_curve = self._makeNLCurve()

    def copy(self) -> 'NLCurve':
        """Creates a copy of the NLCurve instance.

        Returns:
            A new NLCurve instance with the same changepoints.
        """
        return NLCurve(self._changepoints)

    def dist(self, other: 'NLCurve') -> float:
        """Computes the distance between two NLCurves.

        Args:
            other: Another NLCurve instance to compare with.

        Returns:
            A float representing the distance between the two NLCurves.
        """
        common_idx = np.union1d(self.nl_curve.index.to_numpy(), other.nl_curve.index.to_numpy())  # type: ignore
        a = self.nl_curve.reindex(common_idx, fill_value=0)
        b = other.nl_curve.reindex(common_idx, fill_value=0)
        return float(np.linalg.norm(a.to_numpy() - b.to_numpy()))

    def reindex(self, other_ser: pd.Series, this_ser : Optional[pd.Series] = None) -> pd.Series:
        """Reindex the NLCurve

        Args:
            other_ser: A pandas Series with segment lengths as the index.
        """
        if this_ser is None:
            this_ser = self.nl_curve.copy()
        indexes = np.union1d(this_ser.index.to_numpy(), other_ser.index.to_numpy())  # type: ignore
        this_ser = this_ser.reindex(indexes, fill_value=0)
        return this_ser

    def _makeNLCurve(self) -> pd.Series:
        """
        Computes the NLCurve, the cumulative area function (CAF) of segment lengths.

        Duplicate segment lengths are grouped and their areas summed so that each
        unique length appears exactly once in the result index.

        Returns:
            pd.Series:
                index: unique segment lengths (sorted ascending)
                values: cumulative fraction of total time contained in segments
                        with length less than or equal to the corresponding index value
        """
        if len(self._segment_length_arr) == 0:
            return pd.Series(dtype=float)

        segment_areas = np.sort(self._segment_length_arr)
        total_area = float(np.sum(segment_areas))
        # Use segment lengths as both index and data, group duplicates by summing
        # their areas, then sort ascending by length.
        grouped = (
            pd.Series(segment_areas, index=segment_areas)
            .groupby(level=0)
            .sum()
            .sort_index()
        )
        caf_ser = (grouped / total_area).cumsum()
        return caf_ser

    def plotNLCurve(self) -> None:
        """Plots the NLCurve for the segment lengths."""
        cdf = self._makeNLCurve()
        plt.step(cdf.index, cdf.to_numpy(), where='post')
        plt.xlabel('Segment Length')
        plt.ylabel('Cumulative Area')
        plt.title('Segment Area CDF')
        plt.grid()
        plt.show()

    @staticmethod
    def _parseChangepoints(value) -> List[Union[int, float]]:
        """Parse a changepoints cell value into a list of numbers.

        Handles both Python lists (already parsed from pickle/JSON) and string
        representations like "[0.0, 2.5]" that come from CSV round-trips via pandas.

        Args:
            value: A Python list or a string representation of one.

        Returns:
            List of int/float values suitable for SegmentAnalyzer.__init__.

        Raises:
            ValueError: If the value cannot be converted to a list of numbers.
        """
        if isinstance(value, str):
            try:
                return cast(List[Union[int, float]], ast.literal_eval(value))
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Cannot parse changepoints string '{value}': {e}"
                ) from e
        if isinstance(value, list):
            return value
        raise ValueError(
            f"Expected changepoints to be a list or string, got "
            f"{type(value).__name__}: {value!r}"
        )

    @staticmethod
    def getChangepoints(path: str) -> pd.DataFrame:
        """Reads a changepoints CSV file and returns only the relevant columns.

        The changepoints column may contain either Python lists (e.g., from pickle)
        or string representations like "[0.0, 2.5]" (from CSV). Both are preserved
        as-is; downstream consumers should call _parseChangepoints if needed.

        Args:
            path: Path to a CSV file with changepoints data.

        Returns:
            pd.DataFrame with columns: changepoints, aggregation_type, system_id.

        Raises:
            ValueError: If the path does not exist or required columns are missing.
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        df = pd.read_csv(path)
        missing_cols = [c for c in (
            cn.COL_CHANGEPOINTS, cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID
        ) if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV file {path} is missing required columns: {missing_cols}"
            )
        return df[[cn.COL_CHANGEPOINTS, cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID]]

    @classmethod
    def fromChangepoints(cls, path: str, model_num: int,
                        species_name: Optional[str] = None) -> 'NLCurve':
        """Creates a NLCurve from changepoints stored in a CSV file.

        Filters the CSV rows to match the given ``model_num`` and aggregation type
        (species name or ``cn.COL_AGGREGATION_TYPE_MODEL``), extracts the single
        matching changepoint series, parses it into a list of numbers, and returns
        a new NLCurve instance.

        Args:
            path: Path to a CSV file with changepoints data.
            model_num: The BioModel number (e.g., 1 matches system_id "BIOMD0000000001").
            species_name: If provided, select the species-specific aggregation row;
                otherwise select the model-level aggregation row.

        Returns:
            A new NLCurve initialized with the parsed changepoints list.

        Raises:
            ValueError: If no single matching row is found for the given filters,
                or if the changepoints value cannot be parsed into a list of numbers.
        """
        df = cls.getChangepoints(path)
        system_id = Model.getBiomodelName(model_num)
        system_df = df[df[cn.COL_SYSTEM_ID] == system_id]
        if species_name is not None:
            ser = system_df[
                system_df[cn.COL_AGGREGATION_TYPE] == species_name
            ][cn.COL_CHANGEPOINTS].values
        else:
            ser = system_df[
                system_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
            ][cn.COL_CHANGEPOINTS].values
        if len(ser) != 1:
            raise ValueError(
                f"Expected one changepoint series for model {model_num} and "
                f"species {species_name}, but found {len(ser)}."
            )
        raw = ser[0]
        changepoints = cls._parseChangepoints(raw)
        return cls(changepoints)
