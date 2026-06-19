"""Tests for PiecewiseSystemDiscovery in piecewise_system_discovery.py."""

import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from model import Model  # type: ignore
from timecourse import Timecourse  # type: ignore
from piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore

IGNORE_TESTS = False

_TWO_SPECIES_ANTIMONY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


# ---------------------------------------------------------------------------
# Synthetic two-regime fixture: a 2-species linear decay chain whose rate
# constants change sharply at t=5, built directly with solve_ivp (no
# tellurium) so that the ground-truth Jacobian per regime is known exactly.
# Segment A (t in [0, 5)):  dS1/dt = -0.5*S1            ;  dS2/dt = 0.5*S1 - 0.3*S2
# Segment B (t in [5, 10)): dS1/dt = -0.05*S1            ;  dS2/dt = 0.05*S1 - 0.05*S2
# ---------------------------------------------------------------------------
_RATE_A = (0.5, 0.3)
_RATE_B = (0.05, 0.05)
_SPLIT_TIME = 5.0
_END_TIME = 10.0
_NUM_POINT_PER_SEGMENT = 100


def _segment_ode(rates):
    a, b = rates
    def f(_t, x):
        return [-a * x[0], a * x[0] - b * x[1]]
    return f


def _jacobian(rates) -> np.ndarray:
    a, b = rates
    return np.array([[-a, 0.0], [a, -b]])


def _makeTwoRegimeTimecourse(min_segment_length: int = 10) -> Timecourse:
    t_a = np.linspace(0.0, _SPLIT_TIME, _NUM_POINT_PER_SEGMENT, endpoint=False)
    sol_a = solve_ivp(_segment_ode(_RATE_A), [0.0, _SPLIT_TIME], [10.0, 0.0],
            t_eval=t_a, rtol=1e-10, atol=1e-12)
    x_split = sol_a.y[:, -1]
    t_b = np.linspace(_SPLIT_TIME, _END_TIME, _NUM_POINT_PER_SEGMENT)
    sol_b = solve_ivp(_segment_ode(_RATE_B), [_SPLIT_TIME, _END_TIME], x_split,
            t_eval=t_b, rtol=1e-10, atol=1e-12)
    time_arr = np.concatenate([t_a, t_b])
    data_arr = np.concatenate([sol_a.y.T, sol_b.y.T], axis=0)
    timecourse_df = pd.DataFrame(data_arr, index=time_arr, columns=["S1", "S2"])

    jac_a = np.tile(_jacobian(_RATE_A), (len(t_a), 1, 1))
    jac_b = np.tile(_jacobian(_RATE_B), (len(t_b), 1, 1))
    jacobian_collection_arr = np.concatenate([jac_a, jac_b], axis=0)

    model = Model(_TWO_SPECIES_ANTIMONY, model_name="test_model")
    return Timecourse(
        model=model,
        timecourse_df=timecourse_df,
        jacobian_collection_arr=jacobian_collection_arr,
    )


class TestPiecewiseSystemDiscoveryConstructor(unittest.TestCase):
    """Tests for the constructor and pre-fit state."""

    def test_stores_constructor_params(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(
            tc, num_change_point=1, min_segment_length=10,
            change_point_threshold=0.01, fit_kernel_bandwidth=0.5,
            predict_kernel_bandwidth=0.5,
        )
        self.assertEqual(psd.num_change_point, 1)
        self.assertEqual(psd.min_segment_length, 10)
        self.assertAlmostEqual(psd.change_point_threshold, 0.01)
        self.assertAlmostEqual(psd.fit_kernel_bandwidth, 0.5)
        self.assertAlmostEqual(psd.predict_kernel_bandwidth, 0.5)

    def test_kwargs_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, threshold=0.5, poly_degree=1)
        self.assertEqual(psd._kwargs, {"threshold": 0.5, "poly_degree": 1})  # pylint: disable=protected-access

    def test_not_fitted_initially(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        self.assertFalse(psd._is_fitted)  # pylint: disable=protected-access
        self.assertEqual(psd._segment_models, [])  # pylint: disable=protected-access
        self.assertEqual(psd._segment_boundaries, [])  # pylint: disable=protected-access

    def test_require_fitted_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd._require_fitted()  # pylint: disable=protected-access


class TestGaussianSmooth(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._gaussianSmooth."""

    def test_constant_values_unchanged(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=1.0)  # pylint: disable=protected-access
        np.testing.assert_allclose(result, values)

    def test_self_weight_dominates_for_small_bandwidth(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 0.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=0.01)  # pylint: disable=protected-access
        np.testing.assert_allclose(result, values, atol=1e-6)

    def test_smoothing_blends_neighbors_for_large_bandwidth(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 0.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=100.0)  # pylint: disable=protected-access
        self.assertAlmostEqual(result[0], result[1], delta=0.5)
        self.assertAlmostEqual(result[1], result[2], delta=0.5)


class TestComputeChangePointSignal(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._computeChangePointSignal."""

    def test_signal_length(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        self.assertEqual(len(signal), len(tc.timecourse_df) - 1)

    def test_signal_peaks_near_regime_split(self) -> None:
        """With a small smoothing bandwidth, the largest signal should occur
        near t=5, where the Jacobian changes sharply."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        split_time_arr = tc.timecourse_df.index.to_numpy(dtype=float)[1:]
        peak_time = split_time_arr[int(np.argmax(signal))]
        self.assertAlmostEqual(peak_time, _SPLIT_TIME, delta=0.5)

    def test_signal_near_zero_within_constant_regime(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        split_time_arr = tc.timecourse_df.index.to_numpy(dtype=float)[1:]
        mid_regime_a = np.argmin(np.abs(split_time_arr - 2.0))
        self.assertLess(signal[mid_regime_a], 0.01)


if __name__ == "__main__":
    unittest.main()
