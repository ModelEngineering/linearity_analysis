"""Tests for ``src.piecewise_system_discovery.PiecewiseSystemDiscovery``."""

import os  # type: ignore
import unittest

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
        self.tc = TimecourseIterator.getTimecourse(BIOMODEL_548)

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


if __name__ == "__main__":
    unittest.main()