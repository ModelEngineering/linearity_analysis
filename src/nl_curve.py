'''Constructs the non-linearity curve (NLCurve) for analyzing segments of multivariate timecourses.'''

"""
This module analyzes segment lengths induced by changepoints in a timecourse.
The segment frequency is a frequency of count of each segment.
The segment density is a fraction of the total timecourse length that is
    contained in segments of each length.
The cumulative area function (CAF) is a cumulative sum of the segment density, and is the NLCurve.
Two CAF curves are compatible if there last boundary is the same.
The distance between two CAF curves is the area between them.
"""

import ast  # type: ignore
import re  # type: ignore
import src.constants as cn
from src.model import Model  # type: ignore
import matplotlib  # type: ignore
import matplotlib.axes  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os  # type: ignore
import pandas as pd # type: ignore
from typing import List, Union, Optional, cast, Tuple # type: ignore


_NP_INT64_PATTERN = re.compile(r'np\.int64\(([^)]*)\)')


class NLCurve(object):

    def __init__(self, boundaries: List[Union[int, float]], name: Optional[str] = None) -> None:
        """

        Args:
            boundaries (List[Union[int, float]]): Endpoints of segment
                (includes the first and last timepoints of the timecourse). Must be sorted in ascending order.
        """
        if len(boundaries) < 2:
            raise ValueError("At least two boundaries are required to define segments.")
        self._name = name
        self._boundaries = list(np.sort(boundaries))
        self._segment_arr = np.diff(self._boundaries)  # lengths of segments between boundaries
        self.curve_ser = self._makeNLCurve()

    def __repr__(self) -> str:
        return f"NLCurve(NLCurve={self.curve_ser})"

    def calculateAUC(self) -> float:
        """Calculates the area under the NLCurve.

        Returns:
            float: Area under the NLCurve.
        """
        # Differences in x
        dx = np.diff(self.curve_ser.index.to_numpy()) 
        total_integral = np.sum(self.curve_ser.iloc[:-1].values * dx)
        total_integral += self.curve_ser.iloc[-1] * (1 - self.curve_ser.index[-1])  # Add the last segment to reach x=1
        return total_integral

    def copy(self) -> 'NLCurve':
        """Creates a copy of the NLCurve instance.

        Returns:
            A new NLCurve instance with the same boundaries.
        """
        return NLCurve(self._boundaries, name=self._name)

    def _isCompatible(self, other: 'NLCurve') -> None:
        """Checks if two NLCurves are compatible for distance computation.

        Args:
            other: Another NLCurve instance to compare with.:w


        Returns:
            None
        """
        if self._boundaries[-1] != other._boundaries[-1]:
            raise ValueError("Both NLCurves must have the same final boundary to compute distance.")

    def dist(self, other: 'NLCurve') -> float:
        """Computes the distance between two NLCurves as the of the area between their curves.

        Args:
            other: Another NLCurve instance to compare with.

        Returns:
            A float representing the distance between the two NLCurves.
        """
        # Curves must have the same findal boundary to compute distance
        self._isCompatible(other)
        # Get common indices
        a = self.mergeIndex(other)
        b = other.mergeIndex(self)
        diff_ser = (a - b).abs()
        distance_ser = diff_ser * diff_ser.index.to_numpy()  # Multiply by the normalized segment lengths to get area   
        distance = distance_ser.sum()
        return distance

    def deprecatedDist(self, other: 'NLCurve') -> float:
        """Computes the distance between two NLCurves as the of the area between their curves.

        Args:
            other: Another NLCurve instance to compare with.

        Returns:
            A float representing the distance between the two NLCurves.
        """
        # Curves must have the same findal boundary to compute distance
        self._isCompatible(other)
        end_boundary = self._boundaries[-1]
        # Get common indices
        a = self.makeMergedSegmentFrequencySer(other)
        b = other.makeMergedSegmentFrequencySer(self)
        diff_ser = (a - b).abs()
        diff_ser.index = diff_ser.index.to_numpy() / end_boundary  # Normalize by the boundary
        distance_ser = diff_ser * diff_ser.index.to_numpy()  # Multiply by the normalized segment lengths to get area   
        distance = distance_ser.sum()
        return distance

    @staticmethod
    def _makeSegmentFrequencySer(boundaries) -> pd.Series:
        """Creates a pd.Series representing the frequency of each segment length implied
        by the given boundaries.
        
        Args:
            boundaries: List of segment boundaries.

        Returns:
            A pd.Series with unique segment lengths as index and their frequencies as values.
        """
        if len(boundaries) < 2:
            return pd.Series(dtype=float)
        segment_lengths = np.diff(boundaries)
        unique_lengths, counts = np.unique(segment_lengths, return_counts=True)
        freq_ser = pd.Series(counts, index=unique_lengths, dtype=float)
        return freq_ser

    def _makeSegmentDensitySer(self, segment_frequency_ser: pd.Series) -> pd.Series:
        """Creates a pd.Series that indicates the fraction of the total segment length
        that is attributed to each unique segment length. Segment lengths are normalized by the
        boundary size (last boundary value).

        Args:
            segment_frequency_ser: A pd.Series with unique segment lengths as index and their frequencies as values.

        Returns:
            A pd.Series with unique segment lengths as index and their densities as values.
            Segment lengths are normalized by the boundary size (last boundary value)
        """
        if segment_frequency_ser.empty:
            return pd.Series(dtype=float)
        density_ser = segment_frequency_ser.copy()
        density_ser.index = density_ser.index.to_numpy() / self._boundaries[-1]  # Normalize by the last boundary
        density_ser = density_ser*density_ser.index.to_numpy()  # Multiply by the normalized segment lengths to get density
        density_ser = density_ser.sort_index()
        return density_ser

    def mergeIndex(self, other: 'NLCurve') -> pd.Series:
        """Reindexes this NLCurve's series to include the union of its index and the other NLCurve's


        Args:
            other: Another NLCurve instance whose index will be merged with this one.

        Returns:
            A pd.Series with the merged index and values from this NLCurve's series,
        """
        self._isCompatible(other)
        daf_ser = self.curve_ser.diff()
        index0 = self.curve_ser.index.to_numpy()[0]
        daf_ser[index0] = self.curve_ser.iloc[0]  # Set the first value to the original curve's first value
        new_indexes = np.array(np.union1d(daf_ser.index.to_numpy(), other.curve_ser.index.to_numpy()))
        reindexed_daf_ser = daf_ser.reindex(new_indexes, fill_value=0)
        return reindexed_daf_ser.cumsum()

    def makeMergedSegmentFrequencySer(self, other: 'NLCurve') -> pd.Series:
        """Merges the boundaries of the two NLCurves, preserving the segment frequencies
        of the original NLCurve


        Args:
            other: Another NLCurve instance whose index will be merged with this one.

        Returns:
            A pd.Series - Segment frequencies of this NLCurve reindexed to include the union of its index and the other NLCurve's index.
        """
        # Calculate density
        other_segment_frequency_ser = other._makeSegmentFrequencySer(other._boundaries)
        this_segment_frequency_ser = self._makeSegmentFrequencySer(self._boundaries)
        new_indexes = np.array(np.union1d(this_segment_frequency_ser.index.to_numpy(),
                other_segment_frequency_ser.index.to_numpy()))
        reindexed_this_segment_frequency_ser = this_segment_frequency_ser.reindex(new_indexes,
                fill_value=0)
        return reindexed_this_segment_frequency_ser

    def deprecatedMakeMergedDensityCurve(self, other: 'NLCurve') -> pd.Series:
        """Reindexes this NLCurve's series to include the union of its index and the other NLCurve's


        Args:
            other: Another NLCurve instance whose index will be merged with this one.

        Returns:
            A pd.Series with the merged index and values from this NLCurve's series,
        """
        # Calculate density
        other_range = other._boundaries[-1]
        this_range = self._boundaries[-1]
        this_daf_ser = self._makeNLDensity()
        other_daf_ser = other._makeNLDensity()
        # Adjust the indicies to reflect the timecourse lengths
        this_daf_ser.index = this_daf_ser.index.to_numpy() * this_range
        other_daf_ser.index = other_daf_ser.index.to_numpy() * other_range
        # Reindex the density
        max_index = max(self._boundaries[-1], other._boundaries[-1])
        indexes = np.array(np.union1d(this_daf_ser.index.to_numpy(), other_daf_ser.index.to_numpy()))
        this_daf_ser = this_daf_ser.reindex(indexes, fill_value=0)
        this_daf_ser.index = this_daf_ser.index.to_numpy() / max_index
        return this_daf_ser

    def _makeNLDensity(self) -> pd.Series:
        """
        Computes the NLDensity, the density function of segment lengths.

        Returns:
            pd.Series:
                index: unique segment lengths (sorted ascending)
                values: fraction of total time contained in segments with length
                        equal to the corresponding index value
        """
        if len(self._segment_arr) == 0:
            return pd.Series(dtype=float)
        segment_areas = np.sort(self._segment_arr)
        total_area = float(np.sum(segment_areas))
        # Use segment lengths as both index and data, group duplicates by summing
        # their areas, then sort ascending by length.
        grouped = (
            pd.Series(segment_areas, index=segment_areas / total_area, dtype=float)
            .groupby(level=0)
            .sum()
            .sort_index()
        )
        density_ser = grouped / total_area
        return density_ser

    def _makeNLCurve(self) -> pd.Series:
        """
        Computes the NLCurve, the cumulative area function (CAF) of segment lengths.

        Returns:
            pd.Series:
                index: unique segment lengths normalized by total segment length (sorted ascending)
                values: cumulative fraction of boundary end time accounted for by segments
                        with length less than or equal to the corresponding index value
        """
        segment_frequency_ser = self._makeSegmentFrequencySer(self._boundaries)
        curve_ser = self._makeSegmentDensitySer(segment_frequency_ser)
        curve_ser = curve_ser.cumsum()
        return curve_ser

    def plotNLCurve(self, data_src: Optional[str] = None, ax=None) -> matplotlib.axes.Axes:
        """Plots the NLCurve for the segment lengths.

        Args:
            data_src: Optional string to include in the plot title.
            ax: Optional matplotlib Axes object to plot on. If None, a new figure and axes are created.

        Returns:
            matplotlib.axes.Axes: The Axes object containing the plot.
        """
        ##
        def makeLegend() -> str:
            if data_src is not None:
                legend_text = f"{data_src}: {self.calculateAUC():.2f}"
            else:
                legend_text = f"Model: {self.calculateAUC():.2f}"
            return legend_text
        ##
        if ax is None:
            ax = plt.gca()
        xv = self.curve_ser.index.to_numpy()
        ax.step(xv, self.curve_ser.to_numpy(), where='post')
        ax.set_xlabel('Normalized Segment Length')
        ax.set_ylabel('Cumulative Area')
        name = "NLCurve" if self._name is None else f"BioModel {self._name}"
        #ax.set_title(f"{name}: AUC = {self.calculateAUC():.2f}")
        ax.set_title(f"{name}")
        if ax.get_legend() is not None:
            legend_texts = [text.get_text() for text in ax.get_legend().get_texts()] # type: ignore
            legend_text = makeLegend()
            legend_texts.append(legend_text)
            ax.legend(legend_texts)
        else:
            legend_text = makeLegend()
            ax.legend([legend_text])
        ax.grid()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

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
            candidates = [value]
            normalized = _NP_INT64_PATTERN.sub(r'\1', value)
            if normalized != value:
                candidates.append(normalized)
            last_err: Optional[Exception] = None
            for s in candidates:
                try:
                    return cast(List[Union[int, float]], ast.literal_eval(s))
                except (ValueError, SyntaxError) as e:  # noqa: PERF203
                    last_err = e
            raise ValueError(
                f"Cannot parse changepoints string '{value}': {last_err!r}"
            ) from last_err
        if isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                return value
        raise ValueError(
            f"Expected changepoints to be a list or string, got "
            f"{type(value).__name__}: {value!r}"
        )

    @staticmethod
    def getDataframeColumns(path: str) -> pd.DataFrame:
        """Reads a PiecewisePredictions CSV file and construct the columns needed for NLCurve analysis.

        The changepoints column may contain either Python lists (e.g., from pickle)
        or string representations like "[0.0, 2.5]" (from CSV). Both are preserved
        as-is; downstream consumers should call _parseChangepoints if needed.

        Args:
            path: Path to a CSV file with changepoints data.

        Returns:
            pd.DataFrame with columns: boundaries, aggregation_type, system_id.

        Raises:
            ValueError: If the path does not exist or required columns are missing.
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        df = pd.read_csv(path)
        missing_cols = [c for c in (
            cn.COL_CHANGEPOINTS, cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID, cn.COL_COUNT
        ) if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV file {path} is missing required columns: {missing_cols}"
            )
        df[cn.COL_BOUNDARIES] = df.apply(
            lambda row: [0.0] + NLCurve._parseChangepoints(row[cn.COL_CHANGEPOINTS]) 
                    + [row[cn.COL_COUNT] -1],
            axis=1
        )
        return df[[cn.COL_BOUNDARIES, cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID]]

    @classmethod
    def fromPSDPredictions(cls, csv_path: str, model_num: int,
                        species_name: Optional[str] = None,
                        df: Optional[pd.DataFrame] = None) -> Tuple['NLCurve', pd.DataFrame]:
        """Creates a NLCurve from data in a PiecewisePredictions CSV file.

        Filters the CSV rows to match the given ``model_num`` and aggregation type
        (species name or ``cn.COL_AGGREGATION_TYPE_MODEL``), extracts the single
        matching changepoint series, parses it into a list of numbers, and returns
        a new NLCurve instance.

        Args:
            csv_path: Path to a CSV file with changepoints data.
            model_num: The BioModel number (e.g., 1 matches system_id "BIOMD0000000001").
            species_name: If provided, select the species-specific aggregation row;
                otherwise select the model-level aggregation row.
            df: Optional pre-loaded DataFrame to use instead of reading from CSV.

        Returns:
            A tuple containing the new NLCurve initialized with the parsed changepoints list and the DataFrame used.

        Raises:
            ValueError: If no single matching row is found for the given filters,
                or if the changepoints value cannot be parsed into a list of numbers.
        """
        if df is None:
            df = cls.getDataframeColumns(csv_path)
        system_id = Model.getBiomodelName(model_num)
        system_df = df[df[cn.COL_SYSTEM_ID] == system_id]
        if species_name is not None:
            ser = system_df[system_df[cn.COL_AGGREGATION_TYPE] == species_name][cn.COL_BOUNDARIES]
        else:
            ser = system_df[system_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL][cn.COL_BOUNDARIES]
        if len(ser) == 0:
            raise ValueError(
                f"Expected exactly a matching row for model_num={model_num}, "
                f"species_name={species_name}, found {len(ser)}."
            )
        boundaries = ser.iloc[0]
        return (cls(boundaries, name=str(model_num)), df)