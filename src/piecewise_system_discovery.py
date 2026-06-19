"""Piecewise system discovery: detects Jacobian-based change points in a
Timecourse and fits a separate SystemDiscovery model to each segment,
blending predictions across segment boundaries with a Gaussian kernel.

See docs/piecewise_system_discovery.md for the full design.
"""

import bisect
from typing import Any, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from src.scaler import Scaler  # type: ignore
from src.system_discovery import ScoreInfo, SystemDiscovery  # type: ignore
from src.timecourse import Timecourse  # type: ignore

NULL_DF = pd.DataFrame()


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-point segments."""

    def __init__(
        self,
        timecourse: Timecourse,
        num_change_point: int = 2,
        min_segment_length: int = 100,
        change_point_threshold: float = 0.1,
        fit_kernel_bandwidth: float = 1.0,
        predict_kernel_bandwidth: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.timecourse = timecourse
        self.num_change_point = num_change_point
        self.min_segment_length = min_segment_length
        self.change_point_threshold = change_point_threshold
        self.fit_kernel_bandwidth = fit_kernel_bandwidth
        self.predict_kernel_bandwidth = predict_kernel_bandwidth
        self._kwargs = kwargs

        self._segment_models: List[SystemDiscovery] = []
        self._segment_boundaries: List[Tuple[float, float]] = []
        self._segment_lengths: List[int] = []
        self._scaler: Scaler
        self._is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")

    @staticmethod
    def _gaussianSmooth(times: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray:
        """Nadaraya-Watson Gaussian kernel smoothing of `values` over `times`."""
        delta_arr = times[:, np.newaxis] - times[np.newaxis, :]
        weight_arr = np.exp(-0.5 * (delta_arr / bandwidth) ** 2)
        return (weight_arr @ values) / weight_arr.sum(axis=1)

    def _computeChangePointSignal(self, timecourse_df: pd.DataFrame,
            jacobian_collection_arr: np.ndarray) -> np.ndarray:
        """fit() steps 2-3: normalized-Jacobian Frobenius distance, smoothed.

        Returns one value per interior candidate split index 1..num_point-1
        (signal[k] corresponds to split index k+1: the point where a new
        segment would start if a change point were placed there).
        """
        num_species = timecourse_df.shape[1]
        std_arr = timecourse_df.to_numpy(dtype=float).std(axis=0, ddof=1)
        safe_std_arr = np.where(np.isclose(std_arr, 0.0), 1.0, std_arr)
        norm_jacobian_arr = jacobian_collection_arr * (
                safe_std_arr[np.newaxis, np.newaxis, :]
                / safe_std_arr[np.newaxis, :, np.newaxis])
        diff_arr = norm_jacobian_arr[1:] - norm_jacobian_arr[:-1]
        raw_signal_arr = np.linalg.norm(
                diff_arr.reshape(diff_arr.shape[0], -1), axis=1) / (num_species ** 2)
        split_time_arr = timecourse_df.index.to_numpy(dtype=float)[1:]
        return self._gaussianSmooth(split_time_arr, raw_signal_arr, self.fit_kernel_bandwidth)

    def _detectChangePoints(self, signal_arr: np.ndarray, num_point: int) -> List[int]:
        """fit() step 4. signal_arr[k] is the signal for split index k+1
        (the time-grid index at which a new segment would begin).

        Returns a sorted (by time) list of accepted interior split indices.
        """
        candidate_index_arr = np.arange(1, num_point)
        order_arr = np.argsort(-signal_arr, kind="stable")
        accepted: List[int] = []
        for rank in order_arr:
            signal_value = signal_arr[rank]
            if signal_value < self.change_point_threshold:
                break
            split_idx = int(candidate_index_arr[rank])
            pos = bisect.bisect_left(accepted, split_idx)
            left_bound = accepted[pos - 1] if pos > 0 else 0
            right_bound = accepted[pos] if pos < len(accepted) else num_point
            if (split_idx - left_bound) < self.min_segment_length:
                continue
            if (right_bound - split_idx) < self.min_segment_length:
                continue
            accepted.insert(pos, split_idx)
            if len(accepted) == self.num_change_point:
                break
        return accepted
