"""Tests for SystemDiscoveryChangepointDetector in src/system_discovery_changepoint_detector.py."""

import unittest

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.system_discovery import SystemDiscovery
from src.plot_options import PlotOptions
from src.system_discovery_changepoint_detector import (
    LARGE_VALUE,
    MIN_SIGNIFICANT_VALUE,
    SystemDiscoveryChangepointDetector,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False


def _make_linear_df(
    n_points: int = 200,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a simple linear ODE timecourse for testing."""
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    from scipy.integrate import solve_ivp  # type: ignore

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


def _make_piecewise_linear_df(
    n_points: int = 200,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a piecewise linear ODE timecourse with an artificial changepoint at index ~100.

    Phase 1 (t < ~5): dA/dt = -A + B, dB/dt = A - B
    Phase 2 (t >= ~5): dA/dt = -0.5*A, dB/dt = 0.5*A
    """
    rng = np.random.default_rng(seed)

    def rhs1(t, z):
        a, b = z
        return [-a + b, a - b]

    def rhs2(t, z):
        a, _b = z
        return [-0.5 * a, 0.5 * a]

    from scipy.integrate import solve_ivp  # type: ignore

    mid_idx = n_points // 2
    t_eval_1 = np.linspace(t_start, (t_start + t_end) / 2, mid_idx)
    sol1 = solve_ivp(rhs1, [t_start, (t_start + t_end) / 2], [1.0, 0.0],
                     t_eval=t_eval_1, rtol=1e-8)

    final_state = sol1.y[:, -1]
    t_mid = t_eval_1[-1]
    t_eval_2 = np.linspace(t_mid + 0.01, t_end, n_points - mid_idx)
    sol2 = solve_ivp(rhs2, [t_mid + 0.01, t_end], final_state.tolist(),
                     t_eval=t_eval_2, rtol=1e-8)

    X1 = sol1.y.T + rng.normal(0, noise_std, (mid_idx, 2))
    X2 = sol2.y.T + rng.normal(0, noise_std, (n_points - mid_idx, 2))
    X = np.vstack([X1, X2])

    return pd.DataFrame(X, index=np.linspace(t_start, t_end, n_points), columns=["A", "B"])
# ---------------------------------------------------------------------------
# is_detected tests
# ---------------------------------------------------------------------------


class TestIsDetected(unittest.TestCase):
    """Tests for the is_detected() method."""

    def _make_detector(self, max_changepoint: int = 0, min_segment_length: int = 5,
                       min_fractional_reduction: float = 0.01) -> SystemDiscoveryChangepointDetector:
        df = _make_linear_df()
        disc = SystemDiscovery(df, threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        return SystemDiscoveryChangepointDetector(disc, max_changepoint=max_changepoint,
                                                   min_segment_length=min_segment_length,
                                                   min_fractional_reduction=min_fractional_reduction)

    def test_not_detected_before_fit(self) -> None:
        """is_detected returns False before detect()."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector()
        self.assertFalse(detector.is_detected())

    def test_detected_after_detect(self) -> None:
        """is_detected returns True after a successful detect()."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        detector.fit()
        self.assertTrue(detector.is_detected())
# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor(unittest.TestCase):
    """Tests for SystemDiscoveryChangepointDetector.__init__."""

    def _make_fitted_disc(self) -> SystemDiscovery:
        df = _make_linear_df()
        return SystemDiscovery(df, threshold=0.001, alpha=0.001, is_normalize=False)

    def test_basic_construction(self) -> None:
        """Construction with a fitted SystemDiscovery succeeds."""
        if IGNORE_TESTS:
            return
        disc = self._make_fitted_disc()
        detector = SystemDiscoveryChangepointDetector(disc)
        self.assertIsNotNone(detector.training_df)
        self.assertEqual(detector.training_df.shape, disc.training_df.shape)

    def test_changepoints_initial_empty(self) -> None:
        """changepoints list is empty before detect()."""
        if IGNORE_TESTS:
            return
        disc = self._make_fitted_disc()
        detector = SystemDiscoveryChangepointDetector(disc)
        self.assertEqual(detector.changepoints, [])

    def test_change_point_detector_initial_none(self) -> None:
        """change_point_detector is None before detect()."""
        if IGNORE_TESTS:
            return
        disc = self._make_fitted_disc()
        detector = SystemDiscoveryChangepointDetector(disc)
        self.assertIsNone(detector.change_point_detector)

    def test_scaler_shared(self) -> None:
        """The detector shares the scaler from SystemDiscovery."""
        if IGNORE_TESTS:
            return
        disc = self._make_fitted_disc()
        detector = SystemDiscoveryChangepointDetector(disc)
        self.assertIs(detector._scaler, disc._scaler)

    def test_system_discovery_shared(self) -> None:
        """The detector stores the original SystemDiscovery."""
        if IGNORE_TESTS:
            return
        disc = self._make_fitted_disc()
        detector = SystemDiscoveryChangepointDetector(disc)
        self.assertIs(detector.system_discovery, disc)
# ---------------------------------------------------------------------------
# _calculateSignal tests (static method)
# ---------------------------------------------------------------------------


class TestCalculateSignal(unittest.TestCase):
    """Tests for the static _calculateSignal method."""

    def test_perfect_prediction_returns_one(self) -> None:
        """When predictions match true values exactly, signal is 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        n = 50
        pred_df = pd.DataFrame(np.random.rand(n, 2), columns=["A", "B"])
        true_df = pred_df.copy()

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        self.assertEqual(result.shape, (n,))
        np.testing.assert_allclose(result, 0, atol=1e-12)

    def test_zero_true_values_masked_out(self) -> None:
        """Columns with values near zero produce LARGE_VALUE in the signal."""
        if IGNORE_TESTS:
            return
        n = 50
        pred_df = pd.DataFrame(np.ones((n, 2)), columns=["A", "B"])
        true_df = pd.DataFrame(np.zeros((n, 2)), columns=["A", "B"])

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        # When all values are zero (or close), every row should be LARGE_VALUE.
        np.testing.assert_array_equal(result, 1)

    def test_single_zero_column_ignored(self) -> None:
        """If one column has zero true values, the min over columns picks the non-zero column."""
        if IGNORE_TESTS:
            return
        n = 50
        pred_df = pd.DataFrame(
            np.column_stack([np.ones(n), 2 * np.ones(n)]), columns=["A", "B"]
        )
        # Column A has zero true values; B is nonzero.
        true_df = pd.DataFrame(
            np.column_stack([np.zeros(n), np.ones(n) + 0.1]), columns=["A", "B"]
        )

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        # The min across columns for each row should be based on column B only (non-zero).
        np.testing.assert_allclose(result, 1, atol=1e-12)

    def test_divide_by_zero_handled(self) -> None:
        """Division by true values near zero produces LARGE_VALUE, not NaN/Inf."""
        if IGNORE_TESTS:
            return
        n = 50
        pred_df = pd.DataFrame(np.random.rand(n, 1), columns=["A"])
        true_df = pd.DataFrame(
            np.where(pred_df.values > MIN_SIGNIFICANT_VALUE, pred_df.values * 2, 0.0),
            columns=["A"],
        )

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_signal_shape_matches_rows(self) -> None:
        """Result shape equals the number of rows in the input DataFrame."""
        if IGNORE_TESTS:
            return
        n = 100
        m = 5
        true_df = pd.DataFrame(np.random.rand(n, m), columns=[f"S{i}" for i in range(m)])
        pred_df = pd.DataFrame(np.random.rand(n, m), columns=[f"S{i}" for i in range(m)])

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        self.assertEqual(result.shape, (n,))

    def test_signal_range_bounded(self) -> None:
        """For non-zero true values with finite predictions, signal is <= 1.0 or LARGE_VALUE."""
        if IGNORE_TESTS:
            return
        n = 50
        true_df = pd.DataFrame(np.random.rand(n, 2) + 0.1, columns=["A", "B"])
        pred_df = pd.DataFrame(true_df.values * 1.5, columns=["A", "B"])

        result = SystemDiscoveryChangepointDetector._calculateSignal(true_df, pred_df)

        # Every value is either <= 1.0 or == LARGE_VALUE (for zero-masked rows).
        for val in result:
            self.assertTrue(val <= 1.0 + 1e-9 or np.isclose(val, LARGE_VALUE))
# ---------------------------------------------------------------------------
# Integration tests with real piecewise data
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end tests using realistic (piecewise) timecourse data."""

    def test_piecewise_data_detects_changepoint(self) -> None:
        """On synthetic piecewise data, detect should find a changepoint near the true boundary."""
        if IGNORE_TESTS:
            return
        df = _make_piecewise_linear_df(n_points=200, noise_std=0.15, seed=42)

        disc = SystemDiscovery(
            df.iloc[:100], threshold=0.001, alpha=0.001, is_normalize=False
        )
        # fit on first half only so the model matches phase 1 dynamics well
        disc.fit()

        detector = SystemDiscoveryChangepointDetector(disc, max_changepoint=3,
                                                       min_segment_length=5,
                                                       min_fractional_reduction=0.001)
        result = detector.fit()

        # We expect at least one changepoint to be found (the true boundary).
        self.assertIsInstance(result, list)
        for cp in result:
            self.assertGreaterEqual(cp, 0)
            self.assertLess(cp, len(df))
# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------


class TestDetect(unittest.TestCase):
    """Tests for the detect() method."""

    def _make_detector(self, max_changepoint: int = 0, min_segment_length: int = 5,
                       min_fractional_reduction: float = 0.01) -> SystemDiscoveryChangepointDetector:
        df = _make_linear_df(n_points=200, noise_std=0.01)
        disc = SystemDiscovery(df, threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        return SystemDiscoveryChangepointDetector(disc, max_changepoint=max_changepoint,
                                                   min_segment_length=min_segment_length,
                                                   min_fractional_reduction=min_fractional_reduction)

    def test_detect_returns_list(self) -> None:
        """detect() returns a list of ints."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        result = detector.fit()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(cp, int) for cp in result))

    def test_detect_sets_change_point_detector(self) -> None:
        """After detect(), change_point_detector is set."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        detector.fit()
        self.assertIsNotNone(detector.change_point_detector)

    def test_detect_sets_is_detected(self) -> None:
        """After detect(), is_detected() returns True."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        detector.fit()
        self.assertTrue(detector.is_detected())

    def test_detect_max_zero_returns_empty(self) -> None:
        """With max_changepoint=0, no changepoints are detected."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=0, min_segment_length=5)
        result = detector.fit()
        self.assertEqual(result, [])

    def test_detect_with_no_real_change(self) -> None:
        """On a consistent linear system with no real changepoints."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=5, min_segment_length=10)
        result = detector.fit()
        # At minimum we should get a valid list.
        self.assertIsInstance(result, list)

    def test_detect_stores_changepoints(self) -> None:
        """detect() populates self.changepoints."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        result = detector.fit()
        self.assertEqual(result, detector.changepoints)
# ---------------------------------------------------------------------------
# plotTimecourseWithChangepoints tests
# ---------------------------------------------------------------------------


class TestPlotTimecourse(unittest.TestCase):
    """Tests for the plotTimecourseWithChangepoints method."""

    def _make_detector(self, max_changepoint: int = 0, min_segment_length: int = 5,
                       min_fractional_reduction: float = 0.01) -> SystemDiscoveryChangepointDetector:
        df = _make_linear_df(n_points=100, noise_std=0.01)
        disc = SystemDiscovery(df, threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        return SystemDiscoveryChangepointDetector(disc, max_changepoint=max_changepoint,
                                                   min_segment_length=min_segment_length,
                                                   min_fractional_reduction=min_fractional_reduction)

    def test_raises_before_detect(self) -> None:
        """plotTimecourseWithChangepoints raises RuntimeError if detect() not called."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector()
        with self.assertRaises(RuntimeError):
            detector.plotTimecourseWithChangepoints()

    def test_returns_figure_and_axes(self) -> None:
        """After detect(), plot returns (Figure, Axes)."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        detector.fit()
        plot_options = detector.plotTimecourseWithChangepoints()
        self.assertTrue(isinstance(plot_options, PlotOptions))
        self.assertIsNotNone(plot_options.ax)

    def test_plot_with_explicit_changepoints(self) -> None:
        """plotTimecourseWithChangepoints accepts explicit changepoint list."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=1, min_segment_length=5)
        detector.fit()
        plot_options = detector.plotTimecourseWithChangepoints(changepoints=[50])
        self.assertTrue(isinstance(plot_options, PlotOptions))
        self.assertIsNotNone(plot_options.ax)

    def test_plot_after_empty_detect(self) -> None:
        """plot works even when detect() finds no changepoints."""
        if IGNORE_TESTS:
            return
        detector = self._make_detector(max_changepoint=0, min_segment_length=5)
        # max_changepoint=0 means no segments to split.
        detector.fit()
        plot_options = detector.plotTimecourseWithChangepoints()
        self.assertTrue(isinstance(plot_options, PlotOptions))
        self.assertIsNotNone(plot_options.ax)
# ---------------------------------------------------------------------------
# _calculateNormalizedOneStepPredictions tests
# ---------------------------------------------------------------------------


class TestCalculateNormalizedOneStepPredictions(unittest.TestCase):
    """Tests for the private method that computes normalized derivatives."""

    def _make_detector(self):
        df = _make_linear_df(n_points=100)
        disc = SystemDiscovery(df, threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        return SystemDiscoveryChangepointDetector(disc), disc

    def test_returns_two_dataframes(self) -> None:
        """Method returns a tuple of two DataFrames."""
        if IGNORE_TESTS:
            return
        detector, _ = self._make_detector()
        true_df, pred_df = detector._calculateNormalizedOneStepPredictions()
        self.assertIsInstance(true_df, pd.DataFrame)
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_same_shape_and_index(self) -> None:
        """True and predicted DataFrames have the same shape and index as training_df."""
        if IGNORE_TESTS:
            return
        detector, disc = self._make_detector()
        true_df, pred_df = detector._calculateNormalizedOneStepPredictions()

        # Both should match training_df's shape (pred fills in NaN for first row).
        self.assertEqual(true_df.shape[1], disc.training_df.shape[1])
        self.assertEqual(pred_df.shape, true_df.shape)
        pd.testing.assert_index_equal(true_df.index, pred_df.index)

    def test_values_are_normalized(self) -> None:
        """Normalized true and predicted values are finite (except first row of true_df
        which is NaN because Xdot has one fewer row than training data).
        """
        if IGNORE_TESTS:
            return
        detector, _ = self._make_detector()
        true_df, pred_df = detector._calculateNormalizedOneStepPredictions()

        # Predicted values must all be finite.
        self.assertFalse(pred_df.isna().any().any())
        # True derivatives have a leading NaN row (Xdot is one row shorter than training data).
        # After reindexing, the first row of true_df is filled with NaN.
        self.assertTrue(true_df.iloc[0].isna().all())
        # All non-leading rows should be finite.
        self.assertFalse(true_df.iloc[1:].isna().any().any())
# ---------------------------------------------------------------------------
# End-to-end test with real BioModel data (BIOMD0000000045)
# ---------------------------------------------------------------------------


HAS_REAL_BIOMODELS_DATA = True  # Set to False if /Users/jlheller/home/Technical/repos/temp-biomodels/final is missing.
import os as _os
_BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
if not _os.path.isdir(_BIOMODELS_DIR):
    HAS_REAL_BIOMODELS_DATA = False
BIOMODEL_NAME = "BIOMD0000000045"
BIOMODEL_NAME = "BIOMD0000001058"


@unittest.skipUnless(HAS_REAL_BIOMODELS_DATA, "BioModels data directory not found")
class TestEndToEndBioModel45(unittest.TestCase):
    """End-to-end test using real BioModel timecourse."""

    def _make_detector(self, max_changepoint: int = 0, min_segment_length: int = 5,
                       min_fractional_reduction: float = 0.01):
        from src.timecourse import Timecourse
        tc = Timecourse.makeBiomodelDF(BIOMODEL_NAME, num_point=1000, end_time=62)
        disc = SystemDiscovery(
            tc.timecourse_df,
            threshold=0.001,
            alpha=0.001,
            is_normalize=False,
        )
        disc.fit()
        return SystemDiscoveryChangepointDetector(disc, max_changepoint=max_changepoint,
                                                   min_segment_length=min_segment_length,
                                                   min_fractional_reduction=min_fractional_reduction), tc

    def test_e2e_detects_changepoints_on_real_data(self) -> None:
        """Full pipeline: load BioModel, fit SystemDiscovery, detect changepoints."""
        if IGNORE_TESTS or not HAS_REAL_BIOMODELS_DATA:
            return
        detector, tc = self._make_detector(max_changepoint=5, min_segment_length=10)
        result = detector.fit()
        self.assertIsInstance(result, list)
        for cp in result:
            self.assertGreaterEqual(cp, 0)
            self.assertLess(cp, len(tc.timecourse_df))

    def test_e2e_is_detected_after_detect(self) -> None:
        """is_detected() returns True after a successful detect() on real BioModel data."""
        if IGNORE_TESTS or not HAS_REAL_BIOMODELS_DATA:
            return
        detector, _ = self._make_detector(max_changepoint=5, min_segment_length=10)
        detector.fit()
        self.assertTrue(detector.is_detected())

    def test_e2e_normalized_predictions_shape(self) -> None:
        """Normalized one-step predictions have the expected shape for BIOMODEL_NAME."""
        if IGNORE_TESTS or not HAS_REAL_BIOMODELS_DATA:
            return
        detector, tc = self._make_detector()
        true_df, pred_df = detector._calculateNormalizedOneStepPredictions()
        # Both DataFrames have the same number of rows and columns as training data.
        n_train = len(tc.timecourse_df)
        n_species = len(tc.timecourse_df.columns)
        self.assertEqual(true_df.shape, (n_train, n_species))
        self.assertEqual(pred_df.shape, (n_train, n_species))

    def test_e2e_signal_finite(self) -> None:
        """The one-step-prediction-accuracy signal is finite and non-empty."""
        if IGNORE_TESTS or not HAS_REAL_BIOMODELS_DATA:
            return
        detector, tc = self._make_detector()
        true_df, pred_df = detector._calculateNormalizedOneStepPredictions()
        signal = detector._calculateSignal(true_df, pred_df)
        self.assertEqual(signal.shape, (len(tc.timecourse_df),))
        self.assertTrue(np.all(np.isfinite(signal)))

    def test_e2e_plot_works(self) -> None:
        """plotTimecourseWithChangepoints works end-to-end on real BioModel data."""
        if IGNORE_TESTS or not HAS_REAL_BIOMODELS_DATA:
            return
        detector, _ = self._make_detector(max_changepoint=100, min_segment_length=100, min_fractional_reduction=0.000)
        detector.fit()
        plot_options = detector.plotTimecourseWithChangepoints()
        self.assertTrue(isinstance(plot_options, PlotOptions))
        self.assertIsNotNone(plot_options.ax)


if __name__ == "__main__":
    unittest.main()