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
