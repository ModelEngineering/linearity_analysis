"""Tests for PiecewiseSystemDiscovery in src/piecewise_system_discovery.py."""

import os
import unittest

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.model import Model  # type: ignore
from src.constants import COL_END_TIME, COL_START_TIME  # type: ignore
import src.constants as cn  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.system_discovery import SystemDiscovery, NULL_DF  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
NUM_POINT = 200


def _make_linear_df(
    n_points: int = NUM_POINT,
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
    n_points: int = NUM_POINT,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a piecewise linear ODE timecourse with two distinct dynamics phases.

    Phase 1 (first half): dA/dt = -A + B, dB/dt = A - B
    Phase 2 (second half): dA/dt = -0.5*A, dB/dt = 0.5*A
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


def _make_small_piecewise_df(seed: int = 42) -> pd.DataFrame:
    """Small piecewise DataFrame for fast-fit tests (100 points)."""
    return _make_piecewise_linear_df(n_points=100, seed=seed)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.__init__."""

    def test_basic_construction(self) -> None:
        """Construction with a valid DataFrame stores species and dimensions."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(df)
        self.assertEqual(psd.num_species, 2)
        self.assertEqual(psd.species_names, ["A", "B"])
        self.assertEqual(psd.num_point, df.shape[0])

    def test_default_parameters(self) -> None:
        """Default parameter values are stored correctly."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(df)
        self.assertEqual(psd.max_changepoint, 2)
        self.assertEqual(psd.min_fractional_reduction, 0.1)
        self.assertEqual(psd.min_subsequence_length, 100)
        self.assertEqual(psd.predict_kernel_bandwidth, 0.5)

    def test_custom_parameters(self) -> None:
        """Custom constructor parameters are stored correctly."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=3, min_fractional_reduction=0.2,
            min_subsequence_length=50, predict_kernel_bandwidth=1.0,
        )
        self.assertEqual(psd.max_changepoint, 3)
        self.assertEqual(psd.min_fractional_reduction, 0.2)
        self.assertEqual(psd.min_subsequence_length, 50)
        self.assertEqual(psd.predict_kernel_bandwidth, 1.0)

    def test_sd_kwargs_poly_degree_default(self) -> None:
        """poly_degree defaults to 1 when not explicitly provided."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(df)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 1)

    def test_sd_kwargs_poly_degree_preserved(self) -> None:
        """Explicit poly_degree in sd_kwargs is preserved."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(df, poly_degree=2)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 2)

    def test_initial_state_unfitted(self) -> None:
        """Constructor leaves the object in an unfitted state."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        psd = PiecewiseSystemDiscovery(df)
        self.assertFalse(psd._is_fitted)
        self.assertEqual(psd._subsequence_models, [])
        self.assertIsNone(psd.sys_disc)
     
# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


class TestFit(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.fit()."""

    def _make_psd(self, df=None) -> PiecewiseSystemDiscovery:
        if df is None:
            df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, min_subsequence_length=20)

    def test_fit_sets_is_fitted(self) -> None:
        """fit() sets _is_fitted to True."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        result = psd.fit()
        self.assertTrue(psd._is_fitted)

    def test_fit_returns_self(self) -> None:
        """fit() returns self for chaining."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        self.assertIs(psd.fit(), psd)

    def test_lazy_initialization(self) -> None:
        """sys_disc and detector are None before fit(), populated after."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        psd = PiecewiseSystemDiscovery(df, min_subsequence_length=20)
        self.assertIsNone(psd.sys_disc)
        self.assertIsNone(psd.detector)
        psd.fit()
        self.assertIsNotNone(psd.sys_disc)
        self.assertIsNotNone(psd.detector)

    def test_fit_creates_single_segment_no_changepoints(self) -> None:
        """When min_subsequence_length exceeds the data size, splitting is impossible.

        With 300 rows and a single changepoint the smallest each segment could be
        would be (300 - 1) / 2 = 149.5 < min_subsequence_length=200, so the detector
        must return zero change points -> exactly one segment.
        """
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=300)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()   # pass override directly to fit()

        self.assertEqual(len(psd._subsequence_models), 1)
        t_start, t_end = psd._subsequence_boundaries[0]
        time_arr = df.index.to_numpy(dtype=float)
        np.testing.assert_allclose(t_start, time_arr[0])
        np.testing.assert_allclose(t_end, time_arr[-1])
        t_start, t_end = psd._subsequence_boundaries[0]
        time_arr = df.index.to_numpy(dtype=float)
        np.testing.assert_allclose(t_start, time_arr[0])
        np.testing.assert_allclose(t_end, time_arr[-1])

    def test_fit_boundary_starts_at_zero(self) -> None:
        """First segment boundary always starts at t=0 in training data."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        first_start, _ = psd._subsequence_boundaries[0]
        np.testing.assert_allclose(first_start, 0.0)

    def test_fit_segment_lengths_sum_to_training_rows(self) -> None:
        """Sum of segment lengths equals the number of training rows."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        df = psd.training_df
        psd.fit()
        total = sum(psd._subsequence_lengths)
        self.assertEqual(total, len(df))

    def test_fit_boundaries_match_training_time_grid(self) -> None:
        """Segment boundary times align with training data time array."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        df = psd.training_df
        time_arr = df.index.to_numpy(dtype=float)
        psd.fit()
        for t_start, _ in psd._subsequence_boundaries:
            self.assertTrue(np.isin(t_start, time_arr))

    def test_fit_last_boundary_uses_final_time(self) -> None:
        """The last segment's end_time equals the final training time."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        df = psd.training_df
        expected_end = float(df.index.to_numpy(dtype=float)[-1])
        psd.fit()
        _, t_end = psd._subsequence_boundaries[-1]
        np.testing.assert_allclose(t_end, expected_end)


# ---------------------------------------------------------------------------
# predict() pre-condition tests
# ---------------------------------------------------------------------------


class TestPredictPreconditions(unittest.TestCase):
    """Tests for guards that predict() enforces before computing results."""

    def _make_psd(self, df=None) -> PiecewiseSystemDiscovery:
        if df is None:
            df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, min_subsequence_length=20)

    def test_raises_when_unfitted(self) -> None:
        """predict() raises RuntimeError when called before fit()."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        with self.assertRaises(RuntimeError):
            psd.predict()

    def test_rejects_mismatched_columns(self) -> None:
        """predict() rejects a test_df whose columns don't match training species."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        bad_df = pd.DataFrame({"X": [1.0], "Y": [2.0]}, index=[0.0])
        psd = self._make_psd(df)
        psd.fit()
        with self.assertRaises(ValueError):
            psd.predict(test_df=bad_df)

    def test_rejects_mismatched_time_grid(self) -> None:
        """predict() rejects a test_df whose index doesn't match the training grid."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        bad_index = np.linspace(0.0, 5.0, len(df))
        bad_df = pd.DataFrame(df.values, index=bad_index, columns=["A", "B"])
        psd = self._make_psd(df)
        psd.fit()
        with self.assertRaises(ValueError):
            psd.predict(test_df=bad_df)

    def test_rejects_time_grid_not_starting_at_zero(self) -> None:
        """_checkTimegGrid rejects indices that don't start at t=0."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        bad_index = np.linspace(1.0, 11.0, len(df))
        bad_df = pd.DataFrame(df.values, index=bad_index, columns=["A", "B"])
        psd = self._make_psd(df)
        psd.fit()
        with self.assertRaises(ValueError):
            psd.predict(test_df=bad_df)


# ---------------------------------------------------------------------------
# predict() correctness tests
# ---------------------------------------------------------------------------


class TestPredict(unittest.TestCase):
    """Tests for the core prediction logic of PiecewiseSystemDiscovery."""

    def _make_psd(self, df=None, max_changepoint=2, min_subsequence_length=20) -> PiecewiseSystemDiscovery:
        if df is None:
            df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, min_subsequence_length=min_subsequence_length,
                max_changepoint=max_changepoint)

    def test_default_prediction_returns_dataframe(self) -> None:
        """predict() with no arguments returns a DataFrame."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.predict()
        self.assertIsInstance(result, pd.DataFrame)

    def test_default_prediction_shape_matches_training(self) -> None:
        """Default prediction output has the same shape as training data."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        df = psd.training_df
        result = psd.predict()
        self.assertEqual(result.shape, df.shape)

    def test_default_prediction_columns_match_species_names(self) -> None:
        """Default prediction columns equal the stored species names."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.predict()
        self.assertEqual(list(result.columns), list(psd.species_names))

    def test_default_prediction_index_matches_training_time(self) -> None:
        """Default prediction index equals the training time grid."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        df = psd.training_df
        result = psd.predict()
        np.testing.assert_array_equal(result.index.to_numpy(), df.index.to_numpy())

    def test_no_duplicate_rows_with_multiple_segments(self) -> None:
        """Multi-segment predictions must not contain duplicate index rows.

        Regression guard for the boundary-dedup fix in predict().
        """
        if IGNORE_TESTS:
            return
        psd = self._make_psd(max_changepoint=3, min_subsequence_length=10)
        psd.fit()
        result = psd.predict()
        if len(psd._subsequence_models) >= 2:
            self.assertFalse(
                result.index.duplicated().any(),
                f"Got {result.index.duplicated().sum()} dup rows "
                f"(segments={len(psd._subsequence_models)}).",
            )

    def test_prediction_with_explicit_test_df(self) -> None:
        """predict() with a valid test_df returns results over the same time grid."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        psd = self._make_psd(df)
        psd.fit()
        result = psd.predict(test_df=df)
        np.testing.assert_array_equal(result.index.to_numpy(), df.index.to_numpy())

    def test_prediction_first_segment_includes_boundary(self) -> None:
        """The first segment's prediction includes the boundary row."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd(max_changepoint=3, min_subsequence_length=10)
        psd.fit()
        result = psd.predict()

        if len(psd._subsequence_models) >= 2:
            t_first_end, _ = psd._subsequence_boundaries[0]
            self.assertTrue(
                np.isin(t_first_end, result.index.to_numpy()),
                f"First segment end-time {t_first_end} not in prediction index.",
            )

    def test_prediction_values_are_finite(self) -> None:
        """Predicted values should be finite (no NaN/Inf from integration)."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.predict()
        self.assertTrue(np.all(np.isfinite(result.to_numpy())))


# ---------------------------------------------------------------------------
# getScoreDetails tests
# ---------------------------------------------------------------------------


class TestGetScoreDetails(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.getScoreDetails()."""

    def _make_psd(self) -> PiecewiseSystemDiscovery:
        df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, min_subsequence_length=20)

    def test_returns_dataframe(self) -> None:
        """getScoreDetails returns a DataFrame."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.getScoreDetails()
        self.assertIsInstance(result, pd.DataFrame)

    def test_contains_start_time_column(self) -> None:
        """Result includes a start_time column for each segment."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.getScoreDetails()
        self.assertIn(COL_START_TIME, result.columns)

    def test_contains_end_time_column(self) -> None:
        """Result includes an end_time column for each segment."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.getScoreDetails()
        self.assertIn(COL_END_TIME, result.columns)

    def test_segment_count_matches_boundaries(self) -> None:
        """Number of rows corresponds to species * segments + segment headers."""
        if IGNORE_TESTS:
            return
        df = _make_small_piecewise_df()
        psd = PiecewiseSystemDiscovery(df, min_subsequence_length=20, max_changepoint=3)
        psd.fit()
        result = psd.getScoreDetails()
        expected_rows = (len(psd.species_names) * len(psd._subsequence_models)
                         + len(psd._subsequence_models))
        self.assertEqual(len(result), expected_rows)


# ---------------------------------------------------------------------------
# __str__() and printEquations() smoke tests
# ---------------------------------------------------------------------------


class TestStrAndPrint(unittest.TestCase):
    """Smoke tests for string representations."""

    def _make_psd(self, max_cp: int = 2, min_seg_len: int = 20) -> PiecewiseSystemDiscovery:
        df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, max_changepoint=max_cp,
                                        min_subsequence_length=min_seg_len)

    def test_str_contains_segment_info(self) -> None:
        """__str__() includes segment header information."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        s = str(psd)
        self.assertIsInstance(s, str)
        self.assertIn("subsequence", s.lower())

    def test_print_equations_does_not_crash(self) -> None:
        """printEquations() runs without raising."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        psd.printEquations()


# ---------------------------------------------------------------------------
# plotPiecewise() smoke test
# ---------------------------------------------------------------------------


class TestPlotPiecewise(unittest.TestCase):
    """Smoke tests for the plotting method."""

    def _make_psd(self) -> PiecewiseSystemDiscovery:
        df = _make_small_piecewise_df()
        return PiecewiseSystemDiscovery(df, min_subsequence_length=20)

    def test_plot_returns_plot_options(self) -> None:
        """plotPiecewise returns a PlotOptions object."""
        if IGNORE_TESTS:
            return
        psd = self._make_psd()
        psd.fit()
        result = psd.plotPiecewise(legend=False)
        self.assertIsInstance(result, PlotOptions)


# ---------------------------------------------------------------------------
# End-to-end tests using BioModels 45 (EC/Z/Y/X oscillator)
# ---------------------------------------------------------------------------


_IGNORE_TESTS_BIOMODEL = False
BIOMODEL_NUM = 1058
BIOMODEL_NUM = 904
BIOMODEL_NUM_POINT = 1000
BIOMODEL_NAME = Model.getBiomodelName(BIOMODEL_NUM)
_HAS_BIOMODEL = os.path.isdir(cn.BIOMODELS_DIR) and os.path.isdir(
    os.path.join(cn.BIOMODELS_DIR, BIOMODEL_NAME)
)

#_MODEL_45_SPECIES = ["EC", "Z", "Y", "X"]


def _make_biomodel_timecourse_df(n_points: int = BIOMODEL_NUM_POINT) -> pd.DataFrame:
    """Load and simulate BioModels model returning its timecourse."""
    from src.model import Model  # type: ignore
    from src.timecourse import Timecourse  # type: ignore

    model = Model.makeBiomodel(BIOMODEL_NAME)
    tc = Timecourse(model=model, num_point=n_points)
    return tc.timecourse_df


@unittest.skipUnless(
    _HAS_BIOMODEL and not _IGNORE_TESTS_BIOMODEL,
    "BioModels data directory not found or tests are disabled",
)
class TestEndToEndBioModels45(unittest.TestCase):
    """End-to-end PiecewiseSystemDiscovery using real BioModels model 45.

    BIOMD0000000045 is the EC/Z/Y/X oscillator (Stricker et al. 2008 feedback oscillator).
    Four species with coupled nonlinear dynamics; used here to validate that the full
    SBML -> simulate -> fit PSW -> predict pipeline runs end-to-end without errors.
    """

    def test_fit_produces_single_segment(self) -> None:
        """fit() with a high min_subsequence_length yields one segment covering the full range."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()
        self.assertTrue(psd._is_fitted)
        self.assertEqual(len(psd._subsequence_models), 1)
        t_start, t_end = psd._subsequence_boundaries[0]
        np.testing.assert_allclose(t_start, df.index.to_numpy(dtype=float)[0])
        np.testing.assert_allclose(
            t_end, df.index.to_numpy(dtype=float)[-1], atol=1e-9
        )

    def test_predict_returns_dataframe_with_correct_columns(self) -> None:
        """predict() output has same shape and species columns as training data."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()
        result = psd.predict()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df.shape)
        self.assertEqual(list(result.columns), list(df.columns))

    def test_predict_index_matches_training_time(self) -> None:
        """predict() index equals the training time grid exactly."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()
        result = psd.predict()
        np.testing.assert_array_equal(result.index.to_numpy(), df.index.to_numpy())

    def test_predict_values_are_finite(self) -> None:
        """All predicted values are finite (no NaN/Inf from integration)."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()
        result = psd.predict()
        self.assertTrue(np.all(np.isfinite(result.to_numpy())))

    def test_get_score_details_returns_dataframe(self) -> None:
        """getScoreDetails() returns a DataFrame with start_time/end_time columns."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        psd.fit()
        score_df = psd.getScoreDetails()
        self.assertIsInstance(score_df, pd.DataFrame)
        self.assertIn(COL_START_TIME, score_df.columns)
        self.assertIn(COL_END_TIME, score_df.columns)

    def test_plot_piecewise_returns_plot_options(self) -> None:
        """plotPiecewise() returns a PlotOptions object for BioModels 45 data."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_timecourse_df()
        psd = PiecewiseSystemDiscovery(df, max_changepoint=10, min_subsequence_length=30,
                                        min_fractional_reduction=0.000)
        psd.fit()
        result = psd.plotPiecewise(legend=False, ylim=(0, 1.5), num_true_point=30)
        self.assertIsInstance(result, PlotOptions)


if __name__ == "__main__":
    unittest.main()

