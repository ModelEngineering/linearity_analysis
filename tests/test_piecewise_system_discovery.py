"""Tests for ``src.piecewise_system_discovery.PiecewiseSystemDiscovery``."""

import unittest

import matplotlib  # noqa: F401 -- non-interactive backend needed before pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

import src.constants as cn  # type: ignore
from src.piecewise_system_discovery import (  # type: ignore
    PiecewiseSystemDiscovery,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
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
        self.assertTrue(psd.is_random_changepoints)


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

    def test_raises_when_max_exceeds_num_point(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=100)
        with self.assertRaises(ValueError):
            psd._makeRandomChangepoints()

    def test_returns_sorted_indices(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=200)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=3)
        result = psd._makeRandomChangepoints()
        self.assertEqual(result, sorted(result))

    def test_indices_respect_min_segment_length(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=500)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=4, min_segment_length=50)
        result = psd._makeRandomChangepoints(seed=123)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                self.assertGreaterEqual(abs(result[j] - result[i]), psd.min_segment_length)

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
        self.assertLessEqual(len(result), 6)

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
# getScoreSummary tests
# ---------------------------------------------------------------------------


class TestGetScoreSummary(unittest.TestCase):

    def test_getscoresummary_returns_score_summary(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50], min_segment_length=20)
        psd.fit()
        summary = psd.getScoreSummary()
        self.assertIsInstance(summary, PiecewiseSystemDiscovery._ScoreSummary)

    def test_getscoresummary_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.getScoreSummary()


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
# _makeChangePointsIteratively tests
# ---------------------------------------------------------------------------


class TestMakeChangepointIteratively(unittest.TestCase):

    def test_max_changepoint_zero_returns_empty(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0, min_segment_length=10)
        cps = psd._makeChangePointsIteratively()
        self.assertEqual(cps, [])

    def test_max_changepoint_exceeds_num_point_raises(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=100, min_segment_length=5)
        with self.assertRaises(ValueError):
            psd._makeChangePointsIteratively()

    def test_generous_threshold_removes_all_on_smooth_data(self) -> None:
        if IGNORE_TESTS:
            return
        # With a very generous threshold every candidate removal is acceptable; on smooth
        # single-regime data all evenly-spaced changepoints should be pruned.
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=10,
            max_fractional_reduction=10.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangePointsIteratively()
        self.assertEqual(cps, [], "generous threshold should prune all on smooth data")

    def test_negative_threshold_keeps_all_init_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        # max_fractional_reduction=-1.0 only permits removals that improve accuracy by > 1.0,
        # which is impossible on a [0, 1] score scale; so every init changepoint survives.
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=5,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangePointsIteratively()
        # All four evenly-spaced init changepoints should survive since nothing can be removed.
        self.assertEqual(cps, [20, 40, 60, 80])

    def test_respects_min_segment_length(self) -> None:
        if IGNORE_TESTS:
            return
        # Use a negative threshold so no changepoint is ever pruned; verify the init
        # evenly-spaced candidates were filtered to satisfy min_segment_length.
        df = _make_linear_df(n_points=12, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=5, min_segment_length=4,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangePointsIteratively()
        self.assertLessEqual(len(cps), 5)
        if len(cps) >= 2:
            diffs = [b - a for a, b in zip(cps, cps[1:])]
            self.assertTrue(all(d >= 4 for d in diffs))

    def test_tighter_threshold_prunes_at_least_as_many(self) -> None:
        if IGNORE_TESTS:
            return
        # Robust ordering check: with max_fractional_reduction=0.0 the method is strictly
        # more selective than with 10.0, so it must keep >= as many changepoints.
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd_tight = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=10,
            max_fractional_reduction=0.0, poly_degree=1, is_normalize=False,
        )
        cps_tight = psd_tight._makeChangePointsIteratively()

        psd_generous = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=10,
            max_fractional_reduction=10.0, poly_degree=1, is_normalize=False,
        )
        cps_generous = psd_generous._makeChangePointsIteratively()

        self.assertGreaterEqual(len(cps_tight), len(cps_generous))

    def test_uses_instance_max_fractional_reduction(self) -> None:
        if IGNORE_TESTS:
            return
        # Confirm the method reads from self.max_fractional_reduction (set via constructor)
        # rather than a per-call argument, by constructing two instances that differ only in
        # this attribute and verifying their pruning behavior diverges.
        df = _make_linear_df(n_points=100, noise_std=0.0)

        psd_tight = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=10,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps_tight = psd_tight._makeChangePointsIteratively()

        psd_generous = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=10,
            max_fractional_reduction=10.0, poly_degree=1, is_normalize=False,
        )
        cps_generous = psd_generous._makeChangePointsIteratively()

        # Tight keeps all; generous prunes to empty on smooth data.
        self.assertEqual(cps_tight, [20, 40, 60, 80])
        self.assertEqual(cps_generous, [])


if __name__ == "__main__":
    unittest.main()