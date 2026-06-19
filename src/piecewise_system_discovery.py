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

    def fit(self) -> "PiecewiseSystemDiscovery":
        """fit() steps 1-4: detect change points, fit per-segment models."""
        raw_df = self.timecourse.timecourse_df
        jacobian_collection_arr = self.timecourse.jacobian_collection_arr
        num_point = raw_df.shape[0]
        time_arr = raw_df.index.to_numpy(dtype=float)

        signal_arr = self._computeChangePointSignal(raw_df, jacobian_collection_arr)
        split_index_list = self._detectChangePoints(signal_arr, num_point)
        boundary_index_arr = [0] + split_index_list + [num_point]

        self._segment_models = []
        self._segment_boundaries = []
        self._segment_lengths = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            segment_df = raw_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < num_point else time_arr[-1]
            self._segment_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._segment_lengths.append(hi - lo)
            model = SystemDiscovery(segment_df, **self._kwargs).fit()
            self._segment_models.append(model)

        self._is_fitted = True
        return self

    def predict_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """Blend per-segment derivative predictions at (t, x) with a Gaussian
        kernel over each segment's midpoint. See docs/piecewise_system_discovery.md.
        """
        self._require_fitted()
        midpoint_arr = np.array(
                [0.5 * (start + end) for start, end in self._segment_boundaries])
        weight_arr = np.exp(-0.5 * ((t - midpoint_arr) / self.predict_kernel_bandwidth) ** 2)
        derivative_arr = np.array([
                model.predictOneStepDerivative(x) for model in self._segment_models])
        return (weight_arr[:, np.newaxis] * derivative_arr).sum(axis=0) / weight_arr.sum()

    def predict(self, test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame:
        """Integrate the blended ODE forward and return predicted concentrations."""
        self._require_fitted()
        if test_df is not NULL_DF:
            x0 = test_df.to_numpy(dtype=float)[0, :]
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            raw_df = self.timecourse.timecourse_df
            x0 = raw_df.to_numpy(dtype=float)[0, :]
            time_arr = raw_df.index.to_numpy(dtype=float)

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.predict_derivative(t, x)

        sol = solve_ivp(
                rhs,
                t_span=(time_arr[0], time_arr[-1]),
                y0=x0,
                t_eval=time_arr,
                method="Radau",
                rtol=1e-6,
                atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        species_names = self._segment_models[0].species_names
        return pd.DataFrame(sol.y.T, index=time_arr, columns=species_names)

    def score(self) -> ScoreInfo:
        """Length-weighted aggregation of per-segment ScoreInfo. See
        docs/piecewise_system_discovery.md `score()` section."""
        self._require_fitted()
        weighted_values: List[float] = []
        num_nonzero_term = 0
        for model, length in zip(self._segment_models, self._segment_lengths):
            info = model.score()
            weighted_values.extend(info.values * length)
            num_nonzero_term += info.num_nonzero_term
        return ScoreInfo(
                min=float(np.min(weighted_values)),
                median=float(np.median(weighted_values)),
                max=float(np.max(weighted_values)),
                values=weighted_values,
                num_nonzero_term=num_nonzero_term,
        )
