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


class TestDetectChangePoints(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._detectChangePoints."""

    def test_no_change_point_below_threshold(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, change_point_threshold=1e6)
        signal = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [])

    def test_single_clear_change_point_detected(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=2, change_point_threshold=0.05)
        # num_point=10, candidates are split indices 1..9 (signal indices 0..8).
        # A clear spike at split index 5 (signal index 4).
        signal = np.array([0.01, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [5])

    def test_rejects_candidate_violating_min_segment_length(self) -> None:
        """A spike at split index 1 would create a 1-point left segment,
        violating min_segment_length=3; it must be skipped."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=3, change_point_threshold=0.05)
        signal = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [])

    def test_stops_at_num_change_point(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=2, change_point_threshold=0.05)
        # Two clear spikes; only the larger (split index 5) should be kept
        # since num_change_point=1.
        signal = np.array([0.01, 0.01, 0.01, 0.8, 1.0, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [5])

    def test_two_change_points_sorted_by_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=2,
                min_segment_length=2, change_point_threshold=0.05)
        signal = np.array([0.01, 0.01, 0.8, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [3, 6])

    def test_continues_past_length_rejection_to_next_candidate(self) -> None:
        """The highest-signal candidate (split index 1) is above threshold
        but rejected for violating min_segment_length=3 (left segment would
        be only 1 point). The loop must continue scanning rather than stop,
        and accept the next above-threshold candidate (split index 4,
        signal 0.3) instead."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=3, change_point_threshold=0.05)
        signal = np.array([1.0, 0.01, 0.01, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [4])


class TestFit(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.fit()."""

    def test_returns_self(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite")
        result = psd.fit()
        self.assertIs(result, psd)

    def test_sets_fitted_flag(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertTrue(psd._is_fitted)  # pylint: disable=protected-access

    def test_detects_two_segments_for_clear_regime_change(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 2)  # pylint: disable=protected-access
        self.assertEqual(len(psd._segment_boundaries), 2)  # pylint: disable=protected-access

    def test_segment_boundary_near_split_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        boundary_time = psd._segment_boundaries[0][1]  # pylint: disable=protected-access
        self.assertAlmostEqual(boundary_time, _SPLIT_TIME, delta=0.5)

    def test_segments_default_to_is_normalize_true(self) -> None:
        """SystemDiscovery's own default applies when kwargs doesn't override it."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertTrue(model._is_normalize)  # pylint: disable=protected-access

    def test_segments_honor_explicit_is_normalize_override(self) -> None:
        """An explicit is_normalize in kwargs passes through unmodified."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite",
                is_normalize=False).fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertFalse(model._is_normalize)  # pylint: disable=protected-access

    def test_segment_coefficients_are_physical_units(self) -> None:
        """Regression guard for the units-mismatch bug: with the default
        is_normalize=True and raw per-segment data, the fitted cross-term
        coefficient (S1 in dS2/dt) must match the true physical-units rate
        constant, not be inflated by a global std(S1)/std(S2) ratio."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        summary_a = psd._segment_models[0].summary()  # pylint: disable=protected-access
        s1_in_ds2dt = float(summary_a.loc["S1", "dS2/dt"])
        self.assertAlmostEqual(s1_in_ds2dt, _RATE_A[0], delta=0.05)

    def test_segment_models_are_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertTrue(model._is_fitted)  # pylint: disable=protected-access

    def test_no_change_point_yields_single_segment(self) -> None:
        """A very high threshold rejects all candidates; entire timecourse is one segment."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                change_point_threshold=1e6, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 1)  # pylint: disable=protected-access
        start, end = psd._segment_boundaries[0]  # pylint: disable=protected-access
        self.assertAlmostEqual(start, tc.timecourse_df.index[0])
        self.assertAlmostEqual(end, tc.timecourse_df.index[-1])

    def test_all_segments_too_short_yields_single_segment(self) -> None:
        """min_segment_length larger than any achievable segment also collapses
        to a single segment (every candidate gets rejected)."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=1000, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 1)  # pylint: disable=protected-access


def _fitTwoRegimePsd(**overrides) -> "PiecewiseSystemDiscovery":
    tc = _makeTwoRegimeTimecourse()
    params = dict(num_change_point=1, min_segment_length=10,
            change_point_threshold=0.05, fit_kernel_bandwidth=0.05,
            predict_kernel_bandwidth=0.2, poly_degree=1, differentiation="finite")
    params.update(overrides)
    return PiecewiseSystemDiscovery(tc, **params).fit()


class TestPredictDerivative(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.predict_derivative."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.predict_derivative(0.0, np.array([10.0, 0.0]))

    def test_returns_correct_shape(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict_derivative(2.0, np.array([8.0, 1.0]))
        self.assertEqual(result.shape, (2,))

    def test_deep_in_segment_a_matches_segment_a_model(self) -> None:
        """Far from the boundary, Gaussian weighting should make the blended
        derivative closely match the nearest (dominant) segment's own
        derivative evaluator."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        x = np.array([8.0, 1.0])
        blended = psd.predict_derivative(0.5, x)
        segment_a_only = psd._segment_models[0].predictOneStepDerivative(x)  # pylint: disable=protected-access
        np.testing.assert_allclose(blended, segment_a_only, atol=0.3)

    def test_deep_in_segment_b_matches_segment_b_model(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        x = np.array([2.0, 3.0])
        blended = psd.predict_derivative(9.5, x)
        segment_b_only = psd._segment_models[1].predictOneStepDerivative(x)  # pylint: disable=protected-access
        np.testing.assert_allclose(blended, segment_b_only, atol=0.3)


class TestPredict(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.predict."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.predict()

    def test_default_predict_returns_dataframe_matching_training_grid(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        self.assertIsInstance(result, pd.DataFrame)
        np.testing.assert_allclose(
                result.index.to_numpy(dtype=float),
                psd.timecourse.timecourse_df.index.to_numpy(dtype=float))

    def test_predict_columns_are_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        self.assertEqual(list(result.columns), ["S1", "S2"])

    def test_predict_starts_at_initial_condition(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        x0 = psd.timecourse.timecourse_df.to_numpy(dtype=float)[0, :]
        np.testing.assert_allclose(result.iloc[0].to_numpy(), x0, atol=1e-6)

    def test_predict_tracks_true_trajectory_reasonably(self) -> None:
        """The blended piecewise prediction should stay within a modest
        absolute tolerance of the true synthetic trajectory."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        true_df = psd.timecourse.timecourse_df
        max_abs_error = (result.to_numpy() - true_df.to_numpy()).__abs__().max()
        self.assertLess(max_abs_error, 2.0)

    def test_predict_with_test_df_uses_its_initial_condition(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        test_df = pd.DataFrame(
                {"S1": [3.0, 2.0], "S2": [1.0, 1.5]}, index=[0.0, 1.0])
        result = psd.predict(test_df)
        np.testing.assert_allclose(result.iloc[0].to_numpy(), [3.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(
                result.index.to_numpy(dtype=float), test_df.index.to_numpy(dtype=float))


class TestScore(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.score."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.score()

    def test_num_nonzero_term_is_sum_across_segments(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        expected = sum(m.score().num_nonzero_term for m in psd._segment_models)  # pylint: disable=protected-access
        self.assertEqual(psd.score().num_nonzero_term, expected)

    def test_values_length_is_weighted_by_segment_length(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        expected_length = sum(
                len(m.score().values) * length
                for m, length in zip(psd._segment_models, psd._segment_lengths))  # pylint: disable=protected-access
        self.assertEqual(len(psd.score().values), expected_length)

    def test_min_median_max_match_manual_computation(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        weighted_values: list = []
        for model, length in zip(psd._segment_models, psd._segment_lengths):  # pylint: disable=protected-access
            weighted_values.extend(model.score().values * length)
        info = psd.score()
        self.assertAlmostEqual(info.min, min(weighted_values))
        self.assertAlmostEqual(info.max, max(weighted_values))
        self.assertAlmostEqual(info.median, float(np.median(weighted_values)))

    def test_single_segment_score_matches_underlying_model(self) -> None:
        """With no change points, score() should reduce to the single
        segment model's own score()."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd(change_point_threshold=1e6)
        underlying = psd._segment_models[0].score()  # pylint: disable=protected-access
        result = psd.score()
        self.assertAlmostEqual(result.min, underlying.min)
        self.assertAlmostEqual(result.max, underlying.max)
        self.assertEqual(result.num_nonzero_term, underlying.num_nonzero_term)


class TestPrintEquations(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.printEquations / __str__."""

    def test_print_equations_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.printEquations()

    def test_str_contains_one_header_per_segment(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        self.assertEqual(text.count("Segment"), len(psd._segment_models))  # pylint: disable=protected-access

    def test_str_contains_segment_time_ranges(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        for start, end in psd._segment_boundaries:  # pylint: disable=protected-access
            self.assertIn(f"{start:.1f}", text)
            self.assertIn(f"{end:.1f}", text)

    def test_str_contains_species_derivative_lines(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        self.assertIn("dS1/dt", text)
        self.assertIn("dS2/dt", text)

    def test_print_equations_runs_without_error(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        psd.printEquations()  # smoke test: just confirm no exception


if __name__ == "__main__":
    unittest.main()
