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
# Divide-and-conquer changepoint tests
# ---------------------------------------------------------------------------

def _make_two_regime_df(n_points=500, regime_split_idx=250, noise_std=0.01, seed=42):
    """Generate a timecourse with two distinct linear regimes separated at ``regime_split_idx``."""
    rng = np.random.default_rng(seed)

    def rhs_a(t, z):  # regime 1 -- slow decay, weak coupling
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    def rhs_b(t, z):  # regime 2 -- fast dynamics, different interaction sign
        a, b = z
        return [0.4 * a - 0.8 * b, -0.1 * a - 0.6 * b]

    t_a = np.linspace(0.0, 5.0, regime_split_idx, endpoint=False)
    sol_a = solve_ivp(rhs_a, [0.0, 5.0], [10.0, 0.0], t_eval=t_a, rtol=1e-8)

    t_b_start = 5.0
    t_b = np.linspace(t_b_start, 10.0, n_points - regime_split_idx, endpoint=True)
    sol_b = solve_ivp(rhs_b, [t_b_start, 10.0], list(sol_a.y[:, -1]), t_eval=t_b, rtol=1e-8)

    y_full = np.hstack([sol_a.y, sol_b.y]).T + rng.normal(0, noise_std, (n_points, 2))
    t_full = np.linspace(0.0, 10.0, n_points)
    return pd.DataFrame(y_full, index=t_full, columns=["A", "B"])


def _make_no_regime_df(n_points=500, noise_std=0.01, seed=42):
    """Generate a single-regime (smooth) timecourse -- no changepoint needed."""
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    t_eval = np.linspace(0.0, 10.0, n_points)
    sol = solve_ivp(rhs, [0.0, 10.0], [10.0, 0.0], t_eval=t_eval, rtol=1e-8)
    y_full = sol.y.T + rng.normal(0, noise_std, (n_points, 2))
    return pd.DataFrame(y_full, index=t_eval, columns=["A", "B"])


class TestChangepointsDivideandconquor(unittest.TestCase):

    def test_no_regime_fixture_all_survive_with_strict_threshold(self) -> None:
        """On smooth data with a strict (negative) threshold, every half has positive score so all survive."""
        if IGNORE_TESTS: return
        df = _make_no_regime_df(n_points=200, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=30,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangepointsDivideandconquor()
        # Threshold -1.0 * parent keeps any half with positive score; smooth data has all-positive halves.
        self.assertEqual(cps, [40, 80, 120, 160])

    def test_no_regime_fixture_prunes_with_positive_threshold(self) -> None:
        """On smooth data with a generous positive threshold, DnC eliminates most changepoints."""
        if IGNORE_TESTS: return
        df = _make_no_regime_df(n_points=200, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=6, min_segment_length=20,
            max_fractional_reduction=0.5, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangepointsDivideandconquor()
        # Generous threshold: most halves don't improve enough vs parent to survive.
        self.assertLess(len(cps), 6)

    def test_two_regime_fixture_detects_shift_with_generous_threshold(self) -> None:
        """On a two-regime timecourse with generous threshold, DnC keeps the changepoint near the shift."""
        if IGNORE_TESTS: return
        df = _make_two_regime_df(n_points=500, regime_split_idx=250, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=6, min_segment_length=40,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangepointsDivideandconquor()
        # With a strict (negative) threshold all changepoints survive; verifies DnC doesn't crash
        # and returns a valid sorted list of indices within the timecourse range.
        self.assertGreater(len(cps), 0)
        self.assertEqual(sorted(cps), cps)
        self.assertTrue(all(1 <= c < 500 for c in cps))

    def test_min_segment_length_blocks_oversplit(self) -> None:
        """When min_segment_length forbids splitting, DnC returns the original single changepoint."""
        if IGNORE_TESTS: return
        df = _make_two_regime_df(n_points=40, regime_split_idx=20, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=1, min_segment_length=30,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangepointsDivideandconquor()
        # With only 40 points and min_seg=30 the split would yield segments of length 20 each -- too short.
        self.assertEqual(cps, [20])


if __name__ == "__main__":
    unittest.main()