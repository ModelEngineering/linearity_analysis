'''Constructs the non-linearity curve (NLCurve) for analyzing segments of multivariate timecourses.'''

"""
This module analyzes segment lengths from changepoints.
A changepoint is a transition between segments.
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
        self._changepoints = changepoints
        self._segment_length_arr = np.diff(changepoints)

    def makeSegmentAreaCDF(self) -> pd.Series:
        """Calculates the CDF of area for the segment lengths.

        Returns:
            pd.Series:
                index: unique segment lengths
                values: cumulative area of segment lengths up to that length
        """
        segment_areas = self._segment_length_arr
        cdf = pd.Series(np.sort(segment_areas)).value_counts(normalize=True).sort_index()
        cdf = cdf * cdf.index.to_numpy()  # weight each probability by its length
        cdf = cdf.cumsum()  # cumulative sum to get the CDF
        return cdf

    def plotSegmentAreaCDF(self) -> None:
        """Plots the CDF of area for the segment lengths."""
        cdf = self.makeSegmentAreaCDF()
        plt.step(cdf.index, cdf.to_numpy(), where='post')
        plt.xlabel('Segment Length')
        plt.ylabel('Cumulative Area')
        plt.title('Segment Area CDF')
        plt.grid()
        plt.show()

    @staticmethod
    def _parse_changepoints_value(value) -> List[Union[int, float]]:
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
    def getChangpoints(path: str) -> pd.DataFrame:
        """Reads a changepoints CSV file and returns only the relevant columns.

        The changepoints column may contain either Python lists (e.g., from pickle)
        or string representations like "[0.0, 2.5]" (from CSV). Both are preserved
        as-is; downstream consumers should call _parse_changepoints_value if needed.

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
    def fromChangpoints(cls, path: str, model_num: int,
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
        df = cls.getChangpoints(path)
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
        changepoints = cls._parse_changepoints_value(raw)
        return cls(changepoints)
