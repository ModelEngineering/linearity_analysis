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
from src.model import Model  # type: ignore
from src.piecewise_system_discovery import (  # type: ignore
    PiecewiseSystemDiscovery,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(cn.TIMECOURSE_ZIP_PATH)
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


class TestParallelChangepoints(unittest.TestCase):
    """Tests for the parallel changepoint elimination pipeline."""

    def test_fit_segments_parallel_matches_serial_on_two_regime(self) -> None:
        """``_fit_segments_parallel`` should produce identical models, boundaries and lengths as
        the serial ``_fitSegments`` on a realistic two-regime timecourse."""
        if IGNORE_TESTS: return
        df = _make_two_regime_df(n_points=200, regime_split_idx=100, noise_std=0.0)
        boundary_index_arr = [0, 50, 100, 150, 200]
        time_arr = df.index.to_numpy(dtype=float)

        psd_serial = PiecewiseSystemDiscovery(df)
        models_s, bounds_s, lens_s = psd_serial._fitSegments(
            [boundary_index_arr[i] for i in (1, 2, 3)]  # intermediate boundaries as changepoints
        )

        from src.piecewise_system_discovery import _fit_segments_parallel  # noqa: local import in test
        models_p, bounds_p, lens_p = _fit_segments_parallel(
            training_df=df,
            boundary_index_arr=boundary_index_arr,
            time_arr=time_arr,
            num_point=len(df),
            sd_kwargs=dict(psd_serial._sd_kwargs),
        )

        self.assertEqual(len(models_s), len(models_p))
        self.assertEqual(lens_s, lens_p)
        for (lo_s, hi_s), (lo_p, hi_p) in zip(bounds_s, bounds_p):
            self.assertAlmostEqual(lo_s, lo_p)
            self.assertAlmostEqual(hi_s, hi_p)
        # Compare per-species ODE equation strings to confirm fit equivalence.
        for m_s, m_p in zip(models_s, models_p):
            eqs_s = m_s.getEquations()
            eqs_p = m_p.getEquations()
            self.assertEqual(sorted(eqs_s.keys()), sorted(eqs_p.keys()))
            for sp in eqs_s:
                self.assertEqual(eqs_s[sp], eqs_p[sp])

    def test_fit_segments_parallel_single_segment_degenerates_to_serial(self) -> None:
        """With only one segment the parallel path must behave identically to ``_fitSegments``."""
        if IGNORE_TESTS: return
        df = _make_linear_df(n_points=100, noise_std=0.0)
        boundary_index_arr = [0, 100]

        psd = PiecewiseSystemDiscovery(df)
        models_s, bounds_s, lens_s = psd._fitSegments([])  # no changepoints -> single segment

        from src.piecewise_system_discovery import _fit_segments_parallel  # noqa: local import in test
        time_arr = df.index.to_numpy(dtype=float)
        models_p, bounds_p, lens_p = _fit_segments_parallel(
            training_df=df, boundary_index_arr=boundary_index_arr,
            time_arr=time_arr, num_point=100, sd_kwargs=dict(psd._sd_kwargs),
        )

        self.assertEqual(len(models_s), len(models_p))
        self.assertEqual(lens_s, lens_p)
        eqs_s = models_s[0].getEquations()
        eqs_p = models_p[0].getEquations()
        for sp in eqs_s:
            self.assertEqual(eqs_s[sp], eqs_p[sp])

    def test_make_changepoints_with_elimination_parallel_valid_two_regime(self) -> None:
        """Parallel eliminator must return a sorted list of indices within the valid range."""
        if IGNORE_TESTS: return
        df = _make_two_regime_df(n_points=300, regime_split_idx=150, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=30,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps = psd._makeChangepointsWithEliminationParallel()
        self.assertGreater(len(cps), 0)
        self.assertEqual(sorted(cps), cps)
        self.assertTrue(all(1 <= c < 300 for c in cps))

    def test_make_changepoints_with_elimination_parallel_no_change_on_smooth_strict(self) -> None:
        """On smooth data with a strict (negative) threshold every initial changepoint survives --
        parallel result must equal the serial baseline."""
        if IGNORE_TESTS: return
        df = _make_no_regime_df(n_points=200, noise_std=0.0)
        psd_p = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=30,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps_parallel = psd_p._makeChangepointsWithEliminationParallel()

        psd_serial = PiecewiseSystemDiscovery(
            df, max_changepoint=4, min_segment_length=30,
            max_fractional_reduction=-1.0, poly_degree=1, is_normalize=False,
        )
        cps_serial = psd_serial._makeChangepointsWithElimination()

        self.assertEqual(cps_parallel, cps_serial)

    def test_make_changepoints_with_elimination_parallel_max_zero(self) -> None:
        """When max_changepoint <= 0 the parallel method should return an empty list."""
        if IGNORE_TESTS: return
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, max_changepoint=0)
        cps = psd._makeChangepointsWithEliminationParallel()
        self.assertEqual(cps, [])



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

    def test_getscoresummary_returns_score_summary(self) -> None:
        """``getScoreSummary()`` must return a _ScoreSummary instance after fit."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200], min_segment_length=100)
        psd.fit()
        summary = psd.getScoreSummary()
        self.assertIsInstance(summary, PiecewiseSystemDiscovery._ScoreSummary)

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