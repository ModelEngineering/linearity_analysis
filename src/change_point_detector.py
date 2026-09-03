import heapq
from typing import Any, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import collections

from src.plot_options import PlotOptions  # type: ignore


ChangePointInfo = collections.namedtuple('ChangePointInfo',
        ['splice_start', 'splice_end', 'adj_sum_sq'])

class ChangePointDetector:
    """
    Detects change points in a sequence of univariate data to minimize the 
    adjusted sum of squares.
    """

    def __init__(self, data_arr: np.ndarray, max_changepoint: int = 2, min_fractional_reduction: float = 0.01,
            min_segment_length: int = 1, is_random_changepoints: bool = False) -> None:
        """
        Initialize the ChangePointDetector with data and change-point detection parameters.

        Args:
            data (np.ndarray): A numpy array of floats.
            max_changepoint (int, optional): Maximum number of change points to detect. Defaults to 2.
            min_fractional_reduction (float, optional): Minimum fractional reduction in ASS required. Defaults to 0.01.
            min_segment_length (int, optional): Minimum length of segments to consider for splitting. Defaults to 1.
            is_random_changepoints (bool, optional): Whether to use random change points. Defaults to False.
        """
        self.data_arr = np.asarray(data_arr, dtype=float)
        self.length = len(self.data_arr)
        self.is_random_changepoints = is_random_changepoints

        # Change-point detection parameters set during construction.
        self.max_changepoint = max_changepoint
        self.min_fractional_reduction = min_fractional_reduction
        self.min_segment_length = min_segment_length

        # Precalculate prefix sums and prefix sums of squares for efficient computation
        self.cumulative_sum_arr = np.cumsum(self.data_arr)
        self.cumulative_sumsq_arr = np.cumsum(self.data_arr**2)
        
        # Instance variables to be set during fit()
        self.subsequences: List[ChangePointInfo] = []
        self.total_reduction: float = 0.0

    def _get_sum(self, start: int, end: int) -> float:
        """Calculate the sum of elements in range [start, end)."""
        if start == 0:
            return self.cumulative_sum_arr[end - 1]
        return self.cumulative_sum_arr[end - 1] - self.cumulative_sum_arr[start - 1]

    def _get_sum_sq(self, start: int, end: int) -> float:
        """Calculate the sum of squares of elements in range [start, end)."""
        if start == 0:
            return self.cumulative_sumsq_arr[end - 1]
        return self.cumulative_sumsq_arr[end - 1] - self.cumulative_sumsq_arr[start - 1]

    def _get_adj_sum_sq(self, start: int, end: int) -> float:
        """
        Calculate the adjusted sum of squares (ASS) for range [start, end).
        ASS = sum(x_i^2) - (sum(x_i)^2 / N)
        """
        length = end - start
        if length <= 0:
            return 0.0
        
        cumulative_sum = self._get_sum(start, end)
        cumulative_ssq = self._get_sum_sq(start, end)
        
        return cumulative_ssq - (cumulative_sum**2 / length)

    def _find_best_split(self, start: int, end: int, min_segment_length: int = 1) -> Tuple[int, float]:
        """
        Find the optimal single split point for the range [start, end).
        
        Returns:
            Tuple[int, float]: (best_split_idx, reduction_in_ASS)
        """
        length = end - start
        if length < 2 * min_segment_length:
            return -1, 0.0

        # The sum of the total range for the reduction calculation
        s_total = self._get_sum(start, end)
        ass_total = self._get_adj_sum_sq(start, end)

        best_s_n = -float('inf')
        best_k = -1

        # Split index k: data[start:k] and data[k:end]
        # k must be such that k - start >= min_segment_length and end - k >= min_segment_length
        for k in range(start + min_segment_length, end - min_segment_length + 1):
            # First partition: [start, k), length k - start
            sum_1 = self._get_sum(start, k)
            # Second partition: [k, end), length end - k
            sum_2 = self._get_sum(k, end)
            
            # s_n = (sum_1^2 / len_1) + (sum_2^2 / len_2)
            s_n = (sum_1**2 / (k - start)) + (sum_2**2 / (end - k))
            
            if s_n > best_s_n:
                best_s_n = s_n
                best_k = k

        if best_k == -1:
            return -1, 0.0

        # Reduction = (sum_1^2 / len_1) + (sum_2^2 / len_2) - (sum_total^2 / length)
        # which is exactly s_n - (s_total^2 / length)
        reduction = best_s_n - (s_total**2 / length)
        
        return best_k, reduction

    def _makeRandomChangePoints(self, num_points: int) -> List[int]:
        """
        Generate random change points within the data range.

        Args:
            num_points (int): Number of random change points to generate.

        Returns:
            List[int]: Sorted list of random change point indices.
        """
        if num_points <= 0 or self.length == 0:
            return []

        # Ensure that change points are within the valid range
        valid_range = self.length - 1  # Change points can be from 1 to length-1
        if num_points >= valid_range:
            return list(range(1, self.length))  # All possible change points

        random_points = np.random.choice(range(1, self.length), size=num_points, replace=False)
        return sorted(random_points)

    def fit(self) -> None:
        """Find change points based on parameters set during construction.

        Args:
            (none — uses self.max_changepoint, self.min_fractional_reduction,
             and self.min_segment_length).
        """
        if self.length == 0:
            self.subsequences = []
            self.total_reduction = 0.0
            return

        # Calculate initial ASS for the entire sequence
        a_total = self._get_adj_sum_sq(0, self.length)

        # Binary Segmentation with max-heap: always split the segment with the
        # largest reduction first so the fixed-K budget is spent optimally.
        # Each entry: (-reduction, start, end, k) — negated for min-heap ordering.
        heap: List[Tuple[float, int, int, int]] = []
        k_init, red_init = self._find_best_split(0, self.length, self.min_segment_length)
        if red_init > 0:
            heapq.heappush(heap, (-red_init, 0, self.length, k_init))

        change_points: List[int] = []
        total_abs_reduction = 0.0

        while heap and len(change_points) < self.max_changepoint:
            neg_red, start, end, k = heapq.heappop(heap)
            if end - start < 2 * self.min_segment_length:
                continue  # Skip segments that are too short to split into two min_segment_length pieces
            reduction = -neg_red

            if reduction <= self.min_fractional_reduction * a_total:
                break  # Heap is ordered by decreasing reduction; no future entry qualifies

            change_points.append(k)
            total_abs_reduction += reduction

            for seg_start, seg_end in ((start, k), (k, end)):
                if seg_end - seg_start >= 2 * self.min_segment_length:
                    k_sub, red_sub = self._find_best_split(seg_start, seg_end, self.min_segment_length)
                    if red_sub > 0:
                        heapq.heappush(heap, (-red_sub, seg_start, seg_end, k_sub))


        # Finalize results
        sorted_points = sorted(change_points)
        
        # Construct partitions
        partitions = []
        prev_point = 0
        for pt in sorted_points:
            changepoint_info = ChangePointInfo(
                splice_start=prev_point, splice_end=pt, adj_sum_sq=self._get_adj_sum_sq(prev_point, pt))
            partitions.append(changepoint_info)
            prev_point = pt
        changepoint_info = ChangePointInfo(
                splice_start=prev_point, splice_end=self.length, adj_sum_sq=self._get_adj_sum_sq(prev_point, self.length))
        partitions.append(changepoint_info)

        self.subsequences = partitions
        self.total_reduction = total_abs_reduction

    def plot(self, **kwargs: Any) -> PlotOptions:
        """Plot the data sequence with red dashed vertical lines at change points.

        Parameters
        ----------
        **kwargs
            Forwarded to PlotOptions (title, xlabel, ylabel, legend, xlim, ylim,
            model_name, fig, ax).  Defaults: xlabel="index", ylabel="value".

        Returns
        -------
        PlotOptions
            Wraps the figure and axes.
        """
        kwargs.setdefault("xlabel", "index")
        kwargs.setdefault("ylabel", "value")
        kwargs.setdefault("legend", False)
        po = PlotOptions(**kwargs)
        ax = po.ax
        ax.plot(np.arange(self.length), self.data_arr, color="steelblue", marker="o", lw=1.0)  # type: ignore
        for subseq in self.subsequences[1:]:
            ax.axvline(subseq.splice_start, color="red", linestyle=":", lw=2.5)  # type: ignore
        po.apply()
        return po
