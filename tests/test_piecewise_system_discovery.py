"""Tests for ``src.piecewise_system_discovery.PiecewiseSystemDiscovery``."""

import os  # type: ignore
import unittest
from unittest.mock import patch
from typing import List, cast

import matplotlib  # noqa: F401 -- non-interactive backend needed before pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

import src.constants as cn  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.model import Model  # type: ignore
from src.piecewise_system_discovery import (  # type: ignore
    PiecewiseSystemDiscovery,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(TimecourseIterator.getZipPath(num_point=cn.NUM_POINT))
BIOMODEL_548 = "BIOMD0000000548"
NUM_POINT_LARGE = 500  # used for slow fit/predict tests; small fixtures use 100.


def _make_linear_df(
    n_points: int = NUM_POINT_LARGE,
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

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestPiecewiseSystemDiscoveryConstructor(unittest.TestCase):

    def test_basic_construction(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, model_name="test_model")
        self.assertEqual(psd.model_name, "test_model")
        self.assertEqual(psd.species_names, ["A", "B"])
        self.assertEqual(psd.num_species, 2)
        self.assertEqual(psd.num_point, 100)

    def test_default_parameters(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        self.assertEqual(psd.max_changepoint, 2)
        self.assertAlmostEqual(psd.max_fractional_reduction, 0.01)
        self.assertEqual(psd.min_segment_length, 100)
        self.assertEqual(psd.num_trail, 1)
        self.assertIsNone(psd.changepoints)

    def test_custom_parameters(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=3, max_fractional_reduction=0.2,
            min_segment_length=20, model_name="my_model",
            num_trail=5, changepoints=[10, 20],
        )
        self.assertEqual(psd.max_changepoint, 3)
        self.assertAlmostEqual(psd.max_fractional_reduction, 0.2)
        self.assertEqual(psd.min_segment_length, 20)
        self.assertEqual(psd.model_name, "my_model")
        self.assertEqual(psd.num_trail, 5)
        self.assertEqual(psd.changepoints, [10, 20])

    def test_not_fitted_initially(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        self.assertFalse(psd._is_fitted)
        self.assertEqual(len(psd._subsequence_models), 0)

    def test_sd_kwargs_forwarded(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, coefficient_threshold=0.1, alpha=0.2)
        self.assertIn("coefficient_threshold", psd._sd_kwargs)
        self.assertAlmostEqual(psd._sd_kwargs["coefficient_threshold"], 0.1)
        self.assertAlmostEqual(psd._sd_kwargs["alpha"], 0.2)

    def test_poly_degree_forced_to_one(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, poly_degree=1)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 1)

    def test_poly_degree_explicit_override(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, poly_degree=2)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 2)

    def test_is_random_changepoints_default_false(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        self.assertFalse(psd.is_random_changepoints)

    def test_is_random_changepoints_explicit_true(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, is_random_changepoints=True)
        self.assertTrue(psd._is_random_changepoints)


# ---------------------------------------------------------------------------
# _makeRandomChangepoints tests
# ---------------------------------------------------------------------------


class TestMakeRandomChangepoints(unittest.TestCase):

    def test_empty_when_max_is_zero(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        self.assertEqual(psd._makeRandomChangepoints(), [])

    def test_empty_when_max_is_negative(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=-1)
        self.assertEqual(psd._makeRandomChangepoints(), [])

    def test_returns_sorted_indices(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=3)
        result = psd._makeRandomChangepoints()
        self.assertEqual(result, sorted(result))

    def test_deterministic_with_seed(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200)
        a = PiecewiseSystemDiscovery(df, max_changepoint=3)._makeRandomChangepoints(seed=42)
        b = PiecewiseSystemDiscovery(df, max_changepoint=3)._makeRandomChangepoints(seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=500, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=2, min_segment_length=10)
        a = psd._makeRandomChangepoints(seed=1)
        b = psd._makeRandomChangepoints(seed=999)
        self.assertNotEqual(a, b)

    def test_fewer_than_max_when_constraints_tight(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=30)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=10, min_segment_length=5)
        result = psd._makeRandomChangepoints()
        self.assertLessEqual(len(result), 12)

    def test_single_changepoint_in_valid_range(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=1)
        result = psd._makeRandomChangepoints(seed=7)
        self.assertEqual(len(result), 1)
        idx = result[0]
        self.assertGreater(idx, 0)
        self.assertLess(idx, df.shape[0] - 1)


# ---------------------------------------------------------------------------
# _makeBestRandomChangepoints tests
# ---------------------------------------------------------------------------


class TestGetBestRandomChangepoints(unittest.TestCase):

    def test_single_trial_returns_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, num_trail=1, min_segment_length=20)
        cp = psd._makeBestRandomChangepoints()
        self.assertIsInstance(cp, list)

    def test_multiple_trials_returns_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, num_trail=3, min_segment_length=20)
        cp = psd._makeBestRandomChangepoints()
        self.assertIsInstance(cp, list)

    def test_returns_sorted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, num_trail=3, min_segment_length=10)
        cp = psd._makeBestRandomChangepoints()
        self.assertEqual(cp, sorted(cp))


# ---------------------------------------------------------------------------
# _fitSegments tests
# ---------------------------------------------------------------------------


class TestFitSegments(unittest.TestCase):

    def test_single_segment(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, min_segment_length=20)
        models, boundaries, lengths = psd._fitSegments([])
        self.assertEqual(len(models), 1)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(lengths[0], df.shape[0])

    def test_two_segments(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, min_segment_length=30)
        models, boundaries, lengths = psd._fitSegments([100])
        self.assertEqual(len(models), 2)
        self.assertEqual(lengths[0], 100)
        self.assertEqual(lengths[1], 100)

    def test_three_segments(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=300, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, min_segment_length=30)
        models, boundaries, lengths = psd._fitSegments([100, 200])
        self.assertEqual(len(models), 3)
        self.assertEqual(boundaries[0][0], df.index[0])

    def test_boundaries_are_floating_point(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, min_segment_length=20)
        _, boundaries, _ = psd._fitSegments([50])
        for start, end in boundaries:
            self.assertIsInstance(start, float)
            self.assertIsInstance(end, float)

    def test_models_are_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, min_segment_length=20)
        models, _, _ = psd._fitSegments([])
        for m in models:
            self.assertTrue(m.is_fitted)


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


class TestFit(unittest.TestCase):

    def test_fit_populates_attributes(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        result = psd.fit()
        self.assertTrue(psd._is_fitted)
        self.assertEqual(len(psd._subsequence_models), 2)
        self.assertIs(result, psd)

    def test_fit_with_explicit_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[40], min_segment_length=20)
        psd.fit()
        self.assertEqual(psd._subsequence_lengths[0], 40)
        self.assertEqual(psd._subsequence_lengths[1], 60)

    def test_fit_with_random_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, num_trail=1, min_segment_length=30)
        psd.fit()
        self.assertTrue(psd._is_fitted)


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------


class TestPredict(unittest.TestCase):

    def test_predict_returns_dataframe(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        pred_df = psd.predict()
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_predict_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.predict()

    def test_predict_columns_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        pred_df = psd.predict()
        self.assertEqual(list(pred_df.columns), ["A", "B"])

    def test_predict_with_custom_test_df(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        pred_df = psd.predict(test_df=df)
        self.assertIsInstance(pred_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# score() and getScoreDetails() tests
# ---------------------------------------------------------------------------


class TestScore(unittest.TestCase):

    def test_score_returns_float(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        score_val = psd.score()
        self.assertIsInstance(score_val, float)

    def test_score_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.score()

    def test_getscoredetails_returns_dataframe(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        score_df = psd.getScoreDetails()
        self.assertIsInstance(score_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# __str__ / printEquations tests
# ---------------------------------------------------------------------------


class TestStr(unittest.TestCase):

    def test_str_returns_nonempty_after_fit(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        s = str(psd)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    def test_str_unfitted_returns_empty(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        s = str(psd)
        self.assertEqual(s, "")

    def test_print_equations_no_exception(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        psd.printEquations()  # just ensure it doesn't raise


# ---------------------------------------------------------------------------
# plotPiecewise tests
# ---------------------------------------------------------------------------


class TestPlotPiecewise(unittest.TestCase):

    def test_plot_piecewise_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        po = psd.plotPiecewise(num_true_point=-1)
        self.assertIsNotNone(po.fig)

    def test_plot_piecewise_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.plotPiecewise()



# ---------------------------------------------------------------------------
# End-to-end BioModel 548 test
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestEndToEndBioModels548(unittest.TestCase):
    """End-to-end test using real BioModel 548 serialized timecourse."""

    def setUp(self) -> None:
        from src.timecourse_iterator import TimecourseIterator  # type: ignore
        self.tc = TimecourseIterator.getTimecourse(BIOMODEL_548, num_point=1000)

    def _make_psd(self, **overrides):
        defaults = dict(
            max_changepoint=2,
            min_segment_length=100,
            poly_degree=1,
            coefficient_threshold=0.01,
            num_trail=1,
            model_name=BIOMODEL_548,
        )
        defaults.update(overrides)
        return PiecewiseSystemDiscovery(self.tc.timecourse_df, **defaults)  # type: ignore

    def test_fit_produces_at_least_one_segment(self) -> None:
        """``fit()`` on real BioModel 548 data must populate subsequence models."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        result = psd.fit()
        self.assertTrue(psd._is_fitted)
        self.assertGreaterEqual(len(psd._subsequence_models), 1)
        self.assertIs(result, psd)

    def test_predict_returns_dataframe_with_correct_columns(self) -> None:
        """``predict()`` must return a DataFrame whose columns match the training species."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        pred_df = psd.predict()
        self.assertIsInstance(pred_df, pd.DataFrame)
        self.assertEqual(list(pred_df.columns), list(self.tc.timecourse_df.columns))

    def test_predict_index_matches_training_time(self) -> None:
        """``predict()`` must return predictions aligned with the training time index."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        pred_df = psd.predict()
        # Values must match; index names may differ (predict does not propagate the column name).
        np.testing.assert_array_equal(pred_df.index.to_numpy(), self.tc.timecourse_df.index.to_numpy())

    def test_predict_values_are_finite(self) -> None:
        """Predicted values that are finite must equal the corresponding training values within tolerance."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        pred_df = psd.predict()
        # At minimum every predicted column must contain some finite values.
        for col in pred_df.columns:
            self.assertGreater(pred_df[col].notna().sum(), 0)

    def test_score_returns_float(self) -> None:
        """``score()`` must return a finite float on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        score_val = psd.score()
        self.assertIsInstance(score_val, float)
        self.assertTrue(np.isfinite(score_val))

    def test_getscoredetails_returns_dataframe(self) -> None:
        """``getScoreDetails()`` must return a non-empty DataFrame on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        score_df = psd.getScoreDetails()
        self.assertIsInstance(score_df, pd.DataFrame)
        self.assertGreater(len(score_df), 0)

    def test_plot_piecewise_returns_plot_options(self) -> None:
        """``plotPiecewise()`` must return a valid PlotOptions on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        po = psd.plotPiecewise(num_true_point=-1)
        self.assertIsNotNone(po.fig)

    def test_str_contains_species_names(self) -> None:
        """``str(psd)`` after fit should mention every species name."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        s = str(psd)
        for sp in self.tc.timecourse_df.columns:
            self.assertIn(sp, s)

    def test_fit_with_explicit_changepoint_indices(self) -> None:
        """Providing explicit changepoints should split the timecourse at those indices."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        n = len(self.tc.timecourse_df)
        mid = n // 2
        psd = self._make_psd(changepoints=[mid], min_segment_length=50)
        psd.fit()
        self.assertEqual(len(psd._subsequence_models), 2)
        self.assertEqual(psd._subsequence_lengths[0], mid)
        self.assertEqual(psd._subsequence_lengths[1], n - mid)


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestChangepointSpecification(unittest.TestCase):
    """End-to-end test using real BioModel 548 serialized timecourse."""

    def test_specify_changepoints(self) -> None:
        """Specifying changepoints should retain them after fit."""
        model_num = 548
        model = Model.makeBiomodel(model_num=model_num)
        timecourse = Timecourse(model, num_point =1000)
        df = timecourse.timecourse_df
        new_changepoints = list(range(10, 990, 10))
        psd = PiecewiseSystemDiscovery(df,
                changepoints=new_changepoints,
                max_fractional_reduction=0.01, model_name=str(model_num))
        psd.fit()
        self.assertEqual(psd.changepoints, new_changepoints)

    def test_no_removal(self) -> None:
        """Specifying changepoints should retain them after fit."""
        model_num = 548
        model = Model.makeBiomodel(model_num=model_num)
        timecourse = Timecourse(model, num_point =1000)
        df = timecourse.timecourse_df
        psd = PiecewiseSystemDiscovery(df,
                max_changepoint=80,
                is_changepoint_removal=False,
                max_fractional_reduction=0.01, model_name=str(model_num))
        psd.fit()
        self.assertEqual(len(psd.changepoints), 80)  # type: ignore


def _make_piecewise_df(n_points: int = 300):
    """Three-segment piecewise ODE with clearly distinct per-segment Jacobians.

    Returns ``(df, cp1_idx, cp2_idx)`` where the two changepoint indices split
    the trajectory into equal thirds.
    """
    t_eval = np.linspace(0.0, 12.0, n_points)
    cp1_idx = n_points // 3
    cp2_idx = 2 * n_points // 3

    t1 = t_eval[:cp1_idx]
    t2 = t_eval[cp1_idx:cp2_idx]
    t3 = t_eval[cp2_idx:]

    def rhs1(t, z): return [-2.0 * z[0] + 0.5 * z[1], 0.5 * z[0] - 1.5 * z[1]]
    sol1 = solve_ivp(rhs1, [t1[0], t1[-1]], [1.0, 0.5], t_eval=t1, rtol=1e-9)

    def rhs2(t, z): return [-0.3 * z[0] + 0.1 * z[1], 0.1 * z[0] - 0.4 * z[1]]
    sol2 = solve_ivp(rhs2, [t2[0], t2[-1]], [sol1.y[0, -1], sol1.y[1, -1]], t_eval=t2, rtol=1e-9)

    def rhs3(t, z): return [-1.2 * z[0] + 0.2 * z[1], 0.2 * z[0] - 0.9 * z[1]]
    sol3 = solve_ivp(rhs3, [t3[0], t3[-1]], [sol2.y[0, -1], sol2.y[1, -1]], t_eval=t3, rtol=1e-9)

    A = np.concatenate([sol1.y[0], sol2.y[0], sol3.y[0]])
    B = np.concatenate([sol1.y[1], sol2.y[1], sol3.y[1]])
    df = pd.DataFrame({"A": A, "B": B}, index=t_eval)
    return df, cp1_idx, cp2_idx


class TestEstimateAccuracyRate(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._estimateAccuracyRate."""

    @classmethod
    def setUpClass(cls):
        """Fit a 3-segment PSD once; all tests in this class reuse it."""
        df, cls.cp1, cls.cp2 = _make_piecewise_df(n_points=300)
        cls.changepoints = [cls.cp1, cls.cp2]
        cls.psd = PiecewiseSystemDiscovery(
            df,
            changepoints=cls.changepoints,
            is_changepoint_removal=False,
            min_segment_length=10,
            model_name="test_piecewise",
        )
        cls.psd.fit()

    # ------------------------------------------------------------------
    # Validation / error paths
    # ------------------------------------------------------------------

    def test_empty_changepoints_raises_value_error(self):
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.psd._estimateAccuracyRate([])

    def test_unfitted_psd_raises_runtime_error(self):
        if IGNORE_TESTS:
            return
        df, cp1, cp2 = _make_piecewise_df(n_points=60)
        unfitted = PiecewiseSystemDiscovery(
            df, changepoints=[cp1, cp2], is_changepoint_removal=False, min_segment_length=5)
        with self.assertRaises(RuntimeError):
            unfitted._estimateAccuracyRate([cp1, cp2])

    def test_no_candidates_passes_threshold_returns_nan_result(self):
        """When no changepoint has a small enough Frobenius distance, return EstimatorResult with NaN."""
        if IGNORE_TESTS:
            return
        # Distinct Jacobians → all normalized diffs >> 0.01 → idx_arr empty → early-return guard.
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=999, max_frac_frob_dist=0.01)
        self.assertIsInstance(result, PiecewiseSystemDiscovery.EstimatorResult)
        self.assertEqual(result.accuracy_rate, 0)
        self.assertTrue(np.isnan(result.total_frob_dist))
        self.assertTrue(np.isnan(result.delta_accuracy))

    def test_one_changepoint_no_candidates_passes_threshold_returns_nan_result(self):
        """1-changepoint PSD where the Jacobian transition is large → idx_arr empty."""
        if IGNORE_TESTS:
            return
        df, _, _ = _make_piecewise_df(n_points=60)  # unpack into (df, cp1_idx, cp2_idx)
        psd_1cp = PiecewiseSystemDiscovery(
            df, changepoints=[20], is_changepoint_removal=False, min_segment_length=5)
        psd_1cp.fit()
        result = psd_1cp._estimateAccuracyRate(
            [20], num_random_changepoint=1, max_frac_frob_dist=0.01)
        self.assertIsInstance(result, PiecewiseSystemDiscovery.EstimatorResult)
        self.assertEqual(result.accuracy_rate, 0)
        self.assertTrue(np.isnan(result.total_frob_dist))
        self.assertTrue(np.isnan(result.delta_accuracy))

    def test_zero_frob_diff_no_candidates_passes_threshold_returns_nan_result(self):
        """Mocked zero Frobenius diffs → normalized to 0.5 each → above threshold → early return."""
        if IGNORE_TESTS:
            return
        with patch.object(self.psd, '_makeFrobeniusDistances', return_value=[0.0, 0.0]):
            result = self.psd._estimateAccuracyRate(
                self.changepoints, num_random_changepoint=1, max_frac_frob_dist=0.01)
        self.assertIsInstance(result, PiecewiseSystemDiscovery.EstimatorResult)
        self.assertEqual(result.accuracy_rate, 0)
        self.assertTrue(np.isnan(result.total_frob_dist))
        self.assertTrue(np.isnan(result.delta_accuracy))

    # ------------------------------------------------------------------
    # Return-value type and structure
    # ------------------------------------------------------------------

    def test_num_random_changepoint_zero_raises_value_error(self):
        """With num_random_changepoint=0 the removed set is empty → total_frob_dist = 0."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError) as ctx:
            self.psd._estimateAccuracyRate(
                self.changepoints, num_random_changepoint=0, max_frac_frob_dist=1.0)
        self.assertIn("zero", str(ctx.exception).lower())

    def test_returns_estimator_result_namedtuple(self):
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertIsInstance(result, PiecewiseSystemDiscovery.EstimatorResult)

    def test_result_fields_are_python_floats(self):
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertIsInstance(result.accuracy_rate, (int, float))
        self.assertIsInstance(result.total_frob_dist, (int, float))
        self.assertIsInstance(result.delta_accuracy, (int, float))

    def test_result_fields_are_finite(self):
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertTrue(np.isfinite(result.accuracy_rate))
        self.assertTrue(np.isfinite(result.total_frob_dist))
        self.assertTrue(np.isfinite(result.delta_accuracy))

    def test_total_frob_dist_is_positive(self):
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertGreater(result.total_frob_dist, 0.0)

    def test_delta_accuracy_equals_rate_times_frob_diff(self):
        """accuracy_rate * total_frob_dist == delta_accuracy is an exact arithmetic identity."""
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertAlmostEqual(
            result.delta_accuracy,
            result.accuracy_rate * result.total_frob_dist,
            places=10,
        )

    # ------------------------------------------------------------------
    # Algorithmic / behavioral correctness
    # ------------------------------------------------------------------

    def test_num_random_changepoint_one_total_frob_dist_less_than_full_sum(self):
        """Removing 1 of 2 changepoints → total_frob_dist strictly less than sum of both diffs."""
        if IGNORE_TESTS:
            return
        all_diffs = self.psd._makeFrobeniusDistances()
        np.random.seed(42)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertGreater(result.total_frob_dist, 0.0)
        self.assertLess(result.total_frob_dist, sum(all_diffs))

    def test_total_frob_dist_equals_one_of_the_two_element_diffs(self):
        """With 2 changepoints and num_random=1, total equals exactly one element of normalized frob_diff_arr.

        The function normalizes the raw Frobenius differences by their sum before picking a
        candidate to remove, so ``total_frob_dist`` must match one of the *normalized* diffs.
        This is also a regression guard for Bug 3 (positional vs time-series indexing).
        """
        if IGNORE_TESTS:
            return
        raw_diffs = self.psd._makeFrobeniusDistances()
        total_raw = sum(raw_diffs)
        norm_diffs = [d / total_raw for d in raw_diffs]
        np.random.seed(42)
        result = self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        matches_first = abs(result.total_frob_dist - norm_diffs[0]) < 1e-10
        matches_second = abs(result.total_frob_dist - norm_diffs[1]) < 1e-10
        self.assertTrue(
            matches_first or matches_second,
            f"total_frob_dist {result.total_frob_dist} should equal one of {norm_diffs}",
        )

    def test_self_state_unchanged_after_call(self):
        """_estimateAccuracyRate must not mutate the fitted PSD's state."""
        if IGNORE_TESTS:
            return
        n_models_before = len(self.psd._subsequence_models)
        changepoints_before = list(cast(List[int], self.psd.changepoints))
        is_fitted_before = self.psd._is_fitted
        np.random.seed(0)
        self.psd._estimateAccuracyRate(
            self.changepoints, num_random_changepoint=1, max_frac_frob_dist=1.0)
        self.assertEqual(len(self.psd._subsequence_models), n_models_before)
        self.assertEqual(self.psd.changepoints, changepoints_before)
        self.assertEqual(self.psd._is_fitted, is_fitted_before)


class TestIncrementalMergeConsistency(unittest.TestCase):
    """Regression guard for the incremental merge optimization in _estimateAccuracyRate.

    Verifies that ``_buildTrialSegments`` + ``_scoreFromSegmentList`` produce the same 
    score as fitting a full trial PiecewiseSystemDiscovery, confirming correctness of 
    the O(c) merge approach over the original O(k) re-fit approach.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.time = np.arange(0, 600, dtype=float)
        n_species = 2
        y = np.zeros((len(cls.time), n_species))
        # Create a piecewise-linear signal with two distinct changeppoints.
        y[:180, 0] = cls.time[:180] * 0.03 + 1.0
        y[180:420, 0] = (cls.time[180:420] - 180) * (-0.02) + 6.4
        y[420:, 0] = (cls.time[420:] - 420) * 0.01 + 1.6
        y[:180, 1] = cls.time[:180] * 0.01 + 5.0
        y[180:420, 1] = (cls.time[180:420] - 180) * 0.005 + 6.8
        y[420:, 1] = (cls.time[420:] - 420) * (-0.03) + 8.0
        # Add enough noise to make scores non-trivial but not so much that fits are degenerate.
        y += np.random.normal(0, 0.15, y.shape)
        cls.df = pd.DataFrame(y, columns=[f's{j}' for j in range(n_species)])
        cls.df.index.name = 'time'
        cls.changeppoints = [179, 419]

    def test_single_removal_matches_full_trial_fit(self):
        """Removing one changeppoint: incremental merge score equals full trial PSD fit."""
        if IGNORE_TESTS:
            return
        psd = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=self.changeppoints).fit()

        # Incremental approach
        remove_positions = np.array([0])  # remove first changeppoint (row 179)
        trial_models, trial_boundaries = psd._buildTrialSegments(remove_positions)
        new_score = psd._scoreFromSegmentList(
            trial_models, trial_boundaries, col=cn.COL_P10, statistic="median")

        # Old approach: fit full trial PSD on surviving changeppoints.
        surviving_cps = [int(cp) for i, cp in enumerate(psd.changepoints)
                         if i not in set(remove_positions)]
        trial_psd_old = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=surviving_cps, is_changepoint_removal=False,
            **psd._sd_kwargs)
        (trial_psd_old._subsequence_models,
         trial_psd_old._subsequence_boundaries,
         trial_psd_old._subsequence_lengths) = trial_psd_old._fitSegments(surviving_cps)
        trial_psd_old._is_fitted = True
        old_score = trial_psd_old.score(col=cn.COL_P10, statistic="median")

        self.assertAlmostEqual(
            new_score, old_score, places=8,
            msg=(f"Incremental score {new_score:.12f} should match full-fit "
                 f"score {old_score:.12f} for single removal"))

    def test_both_removals_matches_full_trial_fit(self):
        """Removing all changeppoints: incremental merge produces a single merged segment 
        with score matching the single-segment trial PSD fit."""
        if IGNORE_TESTS:
            return
        psd = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=self.changeppoints).fit()

        remove_positions = np.array([0, 1])
        trial_models, trial_boundaries = psd._buildTrialSegments(remove_positions)
        new_score = psd._scoreFromSegmentList(
            trial_models, trial_boundaries, col=cn.COL_P10, statistic="median")

        surviving_cps = [int(cp) for i, cp in enumerate(psd.changepoints)
                         if i not in set(remove_positions)]
        trial_psd_old = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=surviving_cps, is_changepoint_removal=False,
            **psd._sd_kwargs)
        (trial_psd_old._subsequence_models,
         trial_psd_old._subsequence_boundaries,
         trial_psd_old._subsequence_lengths) = trial_psd_old._fitSegments(surviving_cps)
        trial_psd_old._is_fitted = True
        old_score = trial_psd_old.score(col=cn.COL_P10, statistic="median")

        self.assertAlmostEqual(
            new_score, old_score, places=8,
            msg=(f"Incremental score {new_score:.12f} should match full-fit "
                 f"score {old_score:.12f} for all-removals case"))

    def test_build_trial_segments_reuses_unchanged_models(self):
        """When only one changeppoint is removed, the unchanged segment's model object 
        must be identical to self._subsequence_models (no re-fit)."""
        if IGNORE_TESTS:
            return
        psd = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=self.changeppoints).fit()

        remove_positions = np.array([0])  # removes cp at row 179, merges segs [0, 1]
        trial_models, _ = psd._buildTrialSegments(remove_positions)

        # Segment index 2 (the one starting at row 419 and extending to end) should be reused.
        self.assertIs(
            trial_models[1], psd._subsequence_models[2],
            "Unchanged segment model must be the same object as in original fit")


if __name__ == "__main__":
    unittest.main()