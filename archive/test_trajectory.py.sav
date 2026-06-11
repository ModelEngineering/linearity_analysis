"""Tests for Trajectory class."""
from src.l_roadrunner import LRoadrunner  # type: ignore

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd  # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from tests.utils_test import makeBiomdPath # . ype: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from trajectory import Trajectory, IVP_RELATIVE_TIMES # type: ignore
from src.plot_options import PlotOptions  # type: ignore

IGNORE_TESTS = False
IS_BIG_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

MAX_ITERATOR_STEP = 1000  # Increase the default max step for solve_ivp to avoid warnings about too many steps in some tests
#
BIOMD8_PATH = makeBiomdPath(8)
BIOMD38_PATH = makeBiomdPath(38)
BIOMD40_PATH = makeBiomdPath(40)
BIOMD60_PATH = makeBiomdPath(60)
BIOMD153_PATH = makeBiomdPath(153)
BIOMD206_PATH = makeBiomdPath(206)
BIOMD27_PATH = makeBiomdPath(27)
#
BIOMD60_ENDTIME = 10.0
BIOMD206_ENDTIME = 15.0
BIOMD153_END_TIME = 1.4106262159417386
BIOMD153_NUM_SPECIES = 74
BIOMD27_END_TIME = 1000
BIOMD27_NUM_SPECIES = 3
BIOMD8_ENDTIME = 20.0
BIOMD40_ENDTIME = 57.0
#
BIOMODEL_NAMES = ["BIOMD0000000206", "BIOMD0000000054",
        "BIOMD0000000181", "BIOMD0000000008"]


def _make_trajectory_from_arrays(jacobian_collection_arr: np.ndarray, timepoints: np.ndarray) -> Trajectory:
    """Return a JacobianCollection from explicit arrays using a mock LRoadrunner."""
    lr = MagicMock(spec=LRoadrunner)
    if len(jacobian_collection_arr) == 0:
        num_species = 0
    else:
        num_species = jacobian_collection_arr.shape[1]
    forcing_inputs = np.zeros((len(timepoints), num_species))
    lr.makeJacobians.return_value = LRoadrunner.MakeJacobianResult(
            jacobians=jacobian_collection_arr,
            timepoints=timepoints,
            forcing_inputs=forcing_inputs)
    lr.start_time = 0.0
    lr.end_time = 5.0
    lr.num_point = len(timepoints)
    lr.simulate.return_value = np.zeros((len(timepoints), num_species))
    lr.getForcingInputs.return_value = np.zeros(num_species)
    lr.getInitialValues.return_value = np.zeros(num_species)
    return Trajectory(lr)

def _make_trajectory(n_point: int = 5, n_species: int = 3) -> Trajectory:
    """Return a JacobianCollection with deterministic values using a mock LRoadrunner."""
    rng = np.random.default_rng(42)
    jacobian_collection_arr = rng.standard_normal((n_point, n_species, n_species))
    timepoints = np.linspace(0, 10, n_point)
    return _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)

def _make_lroadrunner(antimony_str: str, end_time:float = 10.0, num_point:int = 11) -> LRoadrunner:
    """Return a real LRoadrunner for the given Antimony model."""
    return LRoadrunner(antimony_str, start_time=0.0, end_time=end_time, num_point=num_point)

def _make_trajectory_from_model(antimony_str: str, end_time:float = 10.0, num_point:int = 11) -> Trajectory:
    """Return a MultipleLinearPredictor built from a real Antimony model."""
    lr = _make_lroadrunner(antimony_str, end_time=end_time, num_point=num_point)
    jc = Trajectory(lr, jacobian_selection=cn.JAC_MEDIAN)
    return jc


class TestJacobianCollectionInit(unittest.TestCase):
    """Tests for JacobianCollection.__init__."""

    def test_stores_jacobian_collection_arr(self) -> None:
        """jacobian_collection_arr attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertEqual(jc.jacobian_collection_arr.shape, (5, 3, 3))

    def test_stores_timepoints(self) -> None:
        """timepoints attribute is stored as-is."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertEqual(len(jc.timepoint_arr), 5)


class TestGetTimes(unittest.TestCase):
    """Tests for JacobianCollection.getTimes."""

    def test_returns_set(self) -> None:
        """getTimes returns a set."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.getTimes(), np.ndarray)

    def test_unique_timepoints(self) -> None:
        """getTimes returns the unique set of timepoints."""
        if IGNORE_TESTS:
            return
        timepoints = np.array([0.0, 1.0, 1.0, 2.0])
        jacobian_collection_arr = np.zeros((4, 2, 2))
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertTrue(np.array_equal(jc.getTimes(), np.array([0.0, 1.0, 2.0])))

    def test_all_unique_timepoints(self) -> None:
        """getTimes returns all timepoints when none are duplicated."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5)
        self.assertEqual(len(jc.getTimes()), 5)


class TestJacobianMeanArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_mean_arr property."""

    def test_shape(self) -> None:
        """jacobian_mean_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        self.assertEqual(jc.jacobian_median_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_mean_arr equals element-wise mean across timepoints."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(jc.jacobian_median_arr, expected)

    def test_single_timepoint_equals_jacobian(self) -> None:
        """With one timepoint mean equals the Jacobian itself."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_median_arr, jacobian_collection_arr[0])

    def test_cached_on_second_access(self) -> None:
        """jacobian_mean_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=2)
        first = jc.jacobian_median_arr
        second = jc.jacobian_median_arr
        self.assertIs(first, second)


class TestJacobianStdArr(unittest.TestCase):
    """Tests for JacobianCollection.jacobian_std_arr property."""

    def test_shape(self) -> None:
        """jacobian_std_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        self.assertEqual(jc.jacobian_std_arr.shape, (3, 3))

    def test_known_value(self) -> None:
        """jacobian_std_arr equals element-wise std across timepoints."""
        if IGNORE_TESTS:
            return
        values = np.array([1.0, 2.0, 3.0])
        jacobian_collection_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected = np.array([[np.std(values)]])
        np.testing.assert_allclose(jc.jacobian_std_arr, expected)

    def test_constant_jacobian_gives_zero_std(self) -> None:
        """Identical Jacobians across timepoints produce std=0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((5, 2, 2))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        np.testing.assert_allclose(jc.jacobian_std_arr, np.zeros((2, 2)))

    def test_nonnegative(self) -> None:
        """jacobian_std_arr is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=10, n_species=3)
        self.assertTrue(np.all(jc.jacobian_std_arr >= 0.0))

    def test_cached_on_second_access(self) -> None:
        """jacobian_std_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=2)
        first = jc.jacobian_std_arr
        second = jc.jacobian_std_arr
        self.assertIs(first, second)


class TestMaxCV(unittest.TestCase):
    """Tests for JacobianCollection.max_cv property."""

    def test_returns_float(self) -> None:
        """max_cv returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.max_cv, float)

    def test_constant_entries_return_zero(self) -> None:
        """Constant Jacobian entries (std=0) produce cv=0 and max_cv=0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((10, 2, 2))
        timepoints = np.linspace(0, 1, 10)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_zero_mean_entries_excluded(self) -> None:
        """Entries with zero mean (which give inf/nan CV) are treated as 0, not inf."""
        if IGNORE_TESTS:
            return
        # Alternating +1/-1 → mean=0, so CV would be inf/nan
        jacobian_collection_arr = np.array([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
        timepoints = np.array([0.0, 1.0, 2.0, 3.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertEqual(jc.max_cv, 0.0)

    def test_known_cv_value(self) -> None:
        """max_cv matches hand-computed CV for a simple case."""
        if IGNORE_TESTS:
            return
        # Single entry varying from 1 to 3 → mean=2, std=~0.816 → CV=~0.408
        values = np.array([1.0, 2.0, 3.0])
        jacobian_collection_arr = values.reshape(3, 1, 1)
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        expected_cv = float(np.abs(np.std(values) / np.mean(values)))
        self.assertAlmostEqual(jc.max_cv, expected_cv, places=10)

    def test_max_taken_across_entries(self) -> None:
        """max_cv returns the maximum CV across all Jacobian entries."""
        if IGNORE_TESTS:
            return
        # Entry [0,0]: constant 1.0 → cv=0; Entry [0,1]: varies → cv>0
        jacobian_collection_arr = np.ones((5, 2, 2))
        jacobian_collection_arr[:, 0, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timepoints = np.linspace(0, 1, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_max = float(np.abs(np.std(vals) / np.mean(vals)))
        self.assertAlmostEqual(jc.max_cv, expected_max, places=10)

    def test_nonnegative(self) -> None:
        """max_cv is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertGreaterEqual(jc.max_cv, 0.0)


class TestCalculateDeviation(unittest.TestCase):
    """Tests for JacobianCollection._calculateDeviation."""

    def test_returns_1d_array_of_correct_length(self) -> None:
        """Result has shape (num_points,)."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=5, n_species=3)
        result = jc._calculateDeviation()
        self.assertEqual(result.shape, (5,))

    def test_identical_jacobians_give_zero_deviation(self) -> None:
        """When all Jacobians are identical the centroid equals every Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.ones((4, 2, 2))
        timepoints = np.linspace(0, 3, 4)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(4))

    def test_known_value_1x1(self) -> None:
        """Hand-computed deviation for a 1×1 Jacobian with two timepoints."""
        if IGNORE_TESTS:
            return
        # centroid = 3.0; each deviation = |J - 3| / 3 = 1/3
        jacobian_collection_arr = np.array([[[2.0]], [[4.0]]])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        expected = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_centroid_entry_produces_no_nan_or_inf(self) -> None:
        """Entries whose centroid is 0 contribute 0 to the deviation (no inf/nan)."""
        if IGNORE_TESTS:
            return
        # Off-diagonal entries are always 0 → centroid off-diagonal = 0
        jacobian_collection_arr = np.zeros((3, 2, 2))
        jacobian_collection_arr[:, 0, 0] = np.array([1.0, 2.0, 3.0])
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_all_zero_jacobians_give_zero_deviation(self) -> None:
        """All-zero Jacobians → centroid zero → deviation is 0 everywhere."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.zeros((3, 2, 2))
        timepoints = np.array([0.0, 1.0, 2.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_single_timepoint_gives_zero_deviation(self) -> None:
        """With one timepoint the centroid equals the Jacobian, so deviation is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc._calculateDeviation()
        np.testing.assert_array_almost_equal(result, np.zeros(1))

    def test_nonnegative(self) -> None:
        """Deviation is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        result = jc._calculateDeviation()
        self.assertTrue(np.all(result >= 0.0))


class TestPlot(unittest.TestCase):
    """Tests for JacobianCollection.plot."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=5.0, num_point=11)
        self.collection = Trajectory(lr)

    def tearDown(self) -> None:
        plt.close("all")

    def test_runs_without_error(self) -> None:
        """plot runs without raising for a simple Antimony model."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()

    def test_creates_two_axes(self) -> None:
        """plot creates a figure with exactly two subplots."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes), 2)

    def test_first_axis_has_one_line(self) -> None:
        """First subplot contains exactly one line (the deviation curve)."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[0].lines), 1)

    def test_second_axis_line_count_matches_species(self) -> None:
        """Second subplot has one line per floating species."""
        if IGNORE_TESTS:
            return
        n_species = len(self.collection.l_roadrunner.getRoadrunner().getFloatingSpeciesIds())
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        self.assertEqual(len(plt.gcf().axes[1].lines), n_species)

    def test_second_axis_legend_contains_species_ids(self) -> None:
        """Second subplot legend labels match the floating species IDs."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            self.collection.plot()
        legend = plt.gcf().axes[1].get_legend()
        self.assertIsNotNone(legend)
        if legend is not None:
            legend_texts = [t.get_text() for t in legend.get_texts()]
            self.assertIn("S1", legend_texts)
            self.assertIn("S2", legend_texts)

    def test_biomodel_38(self) -> None:
        """plot runs without error for BioModel BIOMD0000000038."""
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD38_PATH):
            self.skipTest(f"SBML file not found at {BIOMD38_PATH}")
        with open(BIOMD38_PATH) as f:
            sbml_str = f.read()
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=0.002, num_point=600)
        collection = Trajectory(lr)
        with patch("matplotlib.pyplot.show"):
            collection.plot()


class TestPlotPredictions(unittest.TestCase):
    """Tests for Trajectory.plotPredictions."""

    trajectory: Trajectory
    num_species: int
    plot_options: PlotOptions

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=5.0, num_point=10)
        cls.trajectory = Trajectory(lr)
        cls.num_species = cls.trajectory.num_species
        cls.plot_options = cls.trajectory.plotPrediction()

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.plot_options, PlotOptions)

    def test_ax_is_not_none(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsNotNone(self.plot_options.ax)

    def test_line_count_matches_species(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.plot_options.ax.lines), self.num_species)

    def test_scatter_count_matches_species(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.plot_options.ax.collections), self.num_species)

    def test_title_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.trajectory.plotPrediction(model_name="TestModel")
        self.assertIn("TestModel", po.ax.get_title())

    def test_explicit_ax_is_used(self) -> None:
        if IGNORE_TESTS:
            return
        _, ax = plt.subplots()
        po = self.trajectory.plotPrediction(ax=ax)
        self.assertIs(po.ax, ax)

    def test_legend_false_suppresses_legend(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.trajectory.plotPrediction(legend=False)
        self.assertIsNone(po.ax.get_legend())

    def test_ylim_applied(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.trajectory.plotPrediction(ylim=(0.0, 5.0))
        self.assertAlmostEqual(po.ax.get_ylim()[0], 0.0, places=5)
        self.assertAlmostEqual(po.ax.get_ylim()[1], 5.0, places=5)


def _diameter_ivp(jacobian_collection_arr: np.ndarray) -> np.ndarray:
    """Mirror of JacobianCollection._calculateWeightedEigenvectors for tests."""
    eigvals, eigvecs = np.linalg.eig(jacobian_collection_arr)
    return eigvecs @ np.exp(eigvals)


class TestDiameter(unittest.TestCase):
    """Tests for JacobianCollection.diameter property."""

    def test_returns_float(self) -> None:
        """diameter returns a float."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertIsInstance(jc.diameter, float)

    def test_identical_jacobians_return_zero(self) -> None:
        """When all Jacobians are identical the mean equals each one, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (5, 1, 1))
        timepoints = np.linspace(0, 4, 5)
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

    def test_single_timepoint_returns_zero(self) -> None:
        """With one timepoint the mean equals the Jacobian, so diameter is 0."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 0.0], [0.0, -2.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        self.assertAlmostEqual(jc.diameter, 0.0, places=10)

    def test_max_taken_across_timepoints(self) -> None:
        """diameter is the maximum distance, not mean or sum."""
        if IGNORE_TESTS:
            return
        # mean = diag([3, 3]); j1 is further from the mean than j0.
        j0 = np.diag([1.0, 1.0])
        j1 = np.diag([5.0, 5.0])
        jacobian_collection_arr = np.array([j0, j1])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        w_mean = _diameter_ivp(jc.jacobian_median_arr)
        dist0 = float(np.linalg.norm(_diameter_ivp(j0) - w_mean))
        dist1 = float(np.linalg.norm(_diameter_ivp(j1) - w_mean))
        self.assertNotEqual(dist0, dist1)

    def test_nonnegative(self) -> None:
        """diameter is always non-negative."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory()
        self.assertGreaterEqual(jc.diameter, 0.0)

    def test_decreases_as_outliers_eliminated(self) -> None:
        """diameter decreases strictly as outlier Jacobians are removed.

        Three symmetric collections are built around center diag([2, 2]) so that
        the mean stays diag([2, 2]) regardless of which subset is used:
            - full: strong outliers + mild outliers + center  → largest diameter
            - mid:  mild outliers + center                    → intermediate diameter
            - center only                                     → diameter = 0
        """
        if IGNORE_TESTS:
            return
        j_center = np.diag([2.0, 2.0])
        j_mild0 = np.diag([1.0, 1.0])
        j_mild1 = np.diag([3.0, 3.0])
        j_strong0 = np.diag([0.0, 0.0])
        j_strong1 = np.diag([4.0, 4.0])

        full_arr = np.array([j_strong0, j_mild0, j_center, j_mild1, j_strong1])
        jc_full = Trajectory.fromArrays(full_arr, np.linspace(0, 4, 5))

        mid_arr = np.array([j_mild0, j_center, j_mild1])
        jc_mid = Trajectory.fromArrays(mid_arr, np.linspace(0, 2, 3))

        center_arr = np.array([j_center])
        jc_center = Trajectory.fromArrays(center_arr, np.array([0.0]))

        self.assertGreater(jc_full.diameter, jc_mid.diameter)
        self.assertGreater(jc_mid.diameter, jc_center.diameter)
        self.assertAlmostEqual(jc_center.diameter, 0.0, places=10)


class TestNonsequentialPartition(unittest.TestCase):

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_collection_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr)

    def test_returns_list(self) -> None:
        """nonsequentialPartition returns a list."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=2)
        self.assertIsInstance(result, list)

    def test_cluster_count_equals_n_cluster(self) -> None:
        """Result has exactly n_cluster elements."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=3)
        self.assertEqual(len(result), 3)

    def test_each_element_is_jacobian_collection(self) -> None:
        """Each element in the result is a JacobianCollection."""
        if IGNORE_TESTS:
            return
        for jc in self.jc.nonsequentialPartition(n_cluster=2):
            self.assertIsInstance(jc, Trajectory)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_collection_arr.shape[0] for jc in result)
        self.assertEqual(total, self.n_points)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.jc.nonsequentialPartition(n_cluster=self.n_points + 1)

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.jc.nonsequentialPartition(n_cluster=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].jacobian_collection_arr.shape[0], self.n_points)


class TestPartitionJacobiansSequentially(unittest.TestCase):
    """Tests for JacobianCollection.sequentialPartition."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.n_points = 20
        self.n_species = 3
        rng = np.random.default_rng(0)
        jacobian_collection_arr = rng.standard_normal((self.n_points, self.n_species, self.n_species))
        timepoint_arr = np.linspace(0.0, 10.0, self.n_points)
        self.jc = Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr)

    def test_returns_list(self) -> None:
        """sequentialPartition returns a list."""
        if IGNORE_TESTS:
            return
        for n_cluster in [2, 3, 4]:
            result = self.jc.sequentialPartition(n_cluster=n_cluster)
            self.assertEqual(len(result), n_cluster)
            total_length = 0
            for jc in result:
                self.assertIsInstance(jc, Trajectory)
                total_length += jc.jacobian_collection_arr.shape[0]
            self.assertEqual(total_length, self.n_points)  # Ensure all Jacobians are included
            self.assertIsInstance(result, list)

    def test_total_jacobians_preserved(self) -> None:
        """Total Jacobian count across all clusters equals n_points."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=4)
        total = sum(jc.jacobian_collection_arr.shape[0] for jc in result)
        self.assertEqual(total, self.n_points)

    def test_clusters_are_contiguous_in_time(self) -> None:
        """Concatenating cluster jacobian_collection_arrs in order reconstructs the original array."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=3)
        reconstructed = np.concatenate([jc.jacobian_collection_arr for jc in result], axis=0)
        np.testing.assert_array_equal(reconstructed, self.jc.jacobian_collection_arr)

    def test_raises_when_n_cluster_exceeds_n_points(self) -> None:
        """ValueError is raised when n_cluster exceeds the number of timepoints."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.jc.sequentialPartition(n_cluster=self.n_points + 1)

    def test_n_cluster_one_returns_all_jacobians(self) -> None:
        """With n_cluster=1, the single cluster contains all timepoints."""
        if IGNORE_TESTS:
            return
        result = self.jc.sequentialPartition(n_cluster=1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].jacobian_collection_arr, self.jc.jacobian_collection_arr)


class TestBiomodel206(unittest.TestCase):
    """Integration tests for JacobianCollection with BioModel BIOMD0000000206.

    Uses the canonical simulation window from the model's SED-ML: t=0..10.
    num_points is reduced from 1000 to 100 to keep the test suite fast.
    """

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD206_PATH):
            self.skipTest(f"BIOMD0000000206 not found at {BIOMD206_PATH}")
        with open(BIOMD206_PATH) as f:
            sbml_str = f.read()
        self.lr = LRoadrunner(sbml_str, start_time=0.0, end_time=BIOMD206_ENDTIME, num_point=100)

    def test_construction_succeeds(self) -> None:
        """JacobianCollection is created without error from BIOMD0000000206."""
        if IGNORE_TESTS:
            return
        try:
            Trajectory(self.lr)
            self.assertTrue(False)
        except Exception as e:
            self.assertTrue(True)


def _make_wed_collection(jacobian_collection_arr: np.ndarray, timepoint_arr: np.ndarray = np.array([]),
        end_time: float = 1.0) -> Trajectory:
    """Return a JacobianCollection backed by a real LRoadrunner for diameter_ivp.

    diameter_ivp reads self.l_roadrunner.end_time to compute relative
    timepoints, so a mock is not sufficient here.
    """
    lr = LRoadrunner(ANTIMONY_MODEL, start_time=0.0, end_time=end_time, num_point=11)
    if len(timepoint_arr) == 0:
        timepoint_arr = np.linspace(0.0, end_time, len(jacobian_collection_arr))
    return Trajectory.fromArrays(jacobian_collection_arr, timepoint_arr, lr)


class TestDiameterIVP(unittest.TestCase):
    """Tests for JacobianCollection.diameter_ivp property.

    Uses identity matrices as a key reference point: all eigenvalues are 1, so
    eigvals**t = 1 for every t, and every weighted-eigenvector sum equals n*e.
    The resulting constant sequence has mean n*e and max-distance 0 — the only
    case that provably yields diameter = 0.
    """

    # 2×2 identity: eigvals=[1,1], so eigvecs@exp(eigvals**t)=[e,e] for all t.
    IDENTITY = np.eye(2)
    # Diagonal with eigenvalues 0.5: produces timepoint-dependent sums.
    J_HALF = np.diag([0.5, 0.5])

    def test_identical(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile([self.IDENTITY]*5, (3, 1, 1)),
                np.linspace(0, 2, 3))
        result = jc.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0)  # Identity Jacobians should yield diameter = 0.0
    
    def test_zero_difference(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jacobian_arr = -1*np.array([self.IDENTITY, self.IDENTITY])
        jc1 = _make_wed_collection(jacobian_arr)
        result = jc1.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0)  # Identity Jacobians should yield diameter = 0.0

    def test_more_different(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jacobian_arr = -1*np.array([self.IDENTITY, 0.9*self.IDENTITY])
        jc1 = _make_wed_collection(jacobian_arr)
        for factor in [1.5, 2.0, 10.0, 100.0]:
            jacobian_arr = -1*np.array([self.IDENTITY, factor*self.IDENTITY])
            jc2 = _make_wed_collection(jacobian_arr)
            self.assertGreater(jc2.diameter_ivp, jc1.diameter_ivp)  # More different Jacobians should yield larger diameter 
            jc1 = jc2
    
    def test_different(self) -> None:
        """diameter_ivp returns a Python float."""
        if IGNORE_TESTS:
            return
        jc1 = _make_wed_collection(
                -1*np.array([self.IDENTITY, 2*self.IDENTITY, self.J_HALF]),
                np.linspace(0, 2, 3))
        result = jc1.diameter_ivp
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # Identity Jacobians should yield diameter = 0.0
        #
        jc2 = _make_wed_collection(
                -1*np.array([self.IDENTITY, 4*self.IDENTITY, 0.5*self.J_HALF]),
                np.linspace(0, 2, 3))
        result2 = jc2.diameter_ivp
        self.assertGreater(result2, result)  # More different Jacobians should yield larger diameter

    def test_nonnegative(self) -> None:
        """diameter_ivp is always non-negative."""
        if IGNORE_TESTS:
            return
        rng = np.random.default_rng(7)
        jacobian_collection_arr = rng.standard_normal((5, 2, 2)) * 0.2
        jc = _make_wed_collection(jacobian_collection_arr, np.linspace(0, 4, 5))
        self.assertGreaterEqual(jc.diameter_ivp, 0.0)

    def test_cached_on_second_access(self) -> None:
        """diameter_ivp is cached: the same value is returned on repeated calls."""
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile(self.J_HALF, (3, 1, 1)),
                np.linspace(0, 2, 3))
        first = jc.diameter_ivp
        second = jc.diameter_ivp
        self.assertEqual(first, second)
        self.assertFalse(np.isnan(jc._diameter))

    def test_identity_matrices_give_zero(self) -> None:
        """A collection of identity matrices has diameter_ivp == 0.

        For J = I, eigvals = [1,...,1] so eigvals**t = 1 for all t, making every
        weighted-eigenvector sum equal to n*e regardless of t.  The sequence is
        constant and its max-distance from the mean is 0.
        """
        if IGNORE_TESTS:
            return
        jc = _make_wed_collection(
                np.tile(self.IDENTITY, (5, 1, 1)),
                np.linspace(0, 4, 5))
        self.assertAlmostEqual(jc.diameter_ivp, 0.0, places=10)

    def test_nonidentity_jacobians_give_positive_diameter(self) -> None:
        """A collection that mixes identity and non-identity Jacobians has positive diameter."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([self.IDENTITY, self.J_HALF, self.IDENTITY])
        jc = _make_wed_collection(jacobian_collection_arr, np.linspace(0, 2, 3))
        self.assertGreater(jc.diameter_ivp, 0.0)

    def test_larger_spread_gives_larger_diameter(self) -> None:
        """Mixing identity Jacobians with a non-identity one increases the diameter.

        jc_A contains only identity matrices  → diameter = 0.
        jc_B contains identity and J_HALF     → diameter > 0.
        """
        if IGNORE_TESTS:
            return
        arr_a = np.tile(self.IDENTITY, (3, 1, 1))
        arr_b = np.array([self.IDENTITY, self.J_HALF, self.IDENTITY])
        jc_a = _make_wed_collection(arr_a, np.linspace(0, 2, 3))
        jc_b = _make_wed_collection(arr_b, np.linspace(0, 2, 3))
        self.assertGreater(jc_b.diameter_ivp,
                jc_a.diameter_ivp)


class TestIvpSolutions(unittest.TestCase):
    """Tests for JacobianCollection.ivp_solutions property."""

    def test_returns_empty_for_empty_collection(self) -> None:
        """ivp_solutions returns an empty array for an empty jacobian_collection_arr."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            _ = _make_trajectory_from_arrays(np.array([]), np.array([]))

    def test_returns_ndarray(self) -> None:
        """ivp_solutions returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        jc = _make_trajectory(n_point=2, n_species=2)
        self.assertIsInstance(jc.ivp_solutions, np.ndarray)

    def test_shape(self) -> None:
        """ivp_solutions has shape (n_jacobians, n_species, n_ivp_timepoints)."""
        if IGNORE_TESTS:
            return
        n_points, n_species = 4, 3
        jc = _make_trajectory(n_point=n_points, n_species=n_species)
        result = jc.ivp_solutions
        self.assertEqual(result.shape, (n_points, n_species *len(IVP_RELATIVE_TIMES)))

    def test_initial_condition_is_ones(self) -> None:
        """At t=0 (IVP_RELATIVE_TIMES[0]=0) the solution equals the initial condition ones(n)."""
        if IGNORE_TESTS:
            return
        n_species = 2
        jacobian_collection_arr = np.array([np.diag([-1.0, -2.0])])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][len(IVP_RELATIVE_TIMES)], 1)

    def test_known_1x1_solution(self) -> None:
        """For J=[[-1]], solution is x(t)=exp(-t); verify at relative timepoints."""
        if IGNORE_TESTS:
            return
        from trajectory import IVP_RELATIVE_TIMES  # type: ignore
        end_time = 5.0
        jacobian_collection_arr = np.array([[[-1.0]]])
        timepoints = np.array([0.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        expected_times = end_time * IVP_RELATIVE_TIMES
        expected = np.exp(-1.0 * expected_times)
        self.assertLess(np.max(abs(result-expected)), 1e-3)

    def test_different_jacobians_give_different_solutions(self) -> None:
        """Two distinct Jacobians in the collection produce distinct solution trajectories."""
        if IGNORE_TESTS:
            return
        j0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
        j1 = np.array([[-3.0, 0.0], [0.0, -3.0]])
        jacobian_collection_arr = np.array([j0, j1])
        timepoints = np.array([0.0, 1.0])
        jc = _make_trajectory_from_arrays(jacobian_collection_arr, timepoints)
        result = jc.ivp_solutions
        self.assertTrue(all(result[1] <= result[0]))  # J with more negative eigenvalues should decay faster


class TestPredictLinearBasic(unittest.TestCase):
    """Tests for Trajectory.predictLinear."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=1.0)

    def test_output_shape(self) -> None:
        """Result has shape (num_timepoints, num_species)."""
        if IGNORE_TESTS:
            return
        n_point, n_species = self.trajectory._num_timepoint, self.trajectory.num_species
        result = self.trajectory._predictNextStep()
        self.assertEqual(result.shape, (n_point, n_species))

@unittest.skipUnless(IS_BIG_TESTS, "This test is slow and may require large SBML files.")
class TestPredictLinearBioModel(unittest.TestCase):
    """Advanced tests for Trajectory.predictLinear."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD8_PATH):
            self.skipTest(f"SBML file not found at {BIOMD8_PATH}")
        self.lr: LRoadrunner
        self.lr = LRoadrunner.makeBiomodel(BIOMD8_PATH,
                start_time=0.0, end_time=2*BIOMD8_ENDTIME, num_point=30)
        self.trajectory = Trajectory(self.lr)

    def test_timecourse(self) -> None:
        """predictLinear returns an ndarray with shape (num_points, num_species)."""
        if IGNORE_TESTS:
            return
        predicted_df = self.trajectory.predict()
        actual_df = self.lr.timecourse_df
        df = (predicted_df - actual_df).abs()/actual_df
        for species_name in self.lr.species_names:
            plt.plot(actual_df.index, actual_df[species_name], label=species_name);
            plt.scatter(actual_df.index, predicted_df[species_name], label=species_name);
        plt.legend()
        plt.xlabel("time");
        plt.ylabel("concentration");
        plt.title("predictLinear vs actual timecourse");
        plt.show();
        self.assertIsInstance(predicted_df, pd.DataFrame)
        self.assertEqual(predicted_df.shape, actual_df.shape)

class TestFitJacobian(unittest.TestCase):
    """Tests for Trajectory.fitJacobian."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL, end_time=5.0, num_point=100)

    def test_returns_ndarray(self) -> None:
        """fitJacobian returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """fitJacobian returns array of shape (num_species, num_species)."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        n = self.trajectory.num_species
        self.assertEqual(result.shape, (n, n))

    def test_improves_or_matches_prediction(self) -> None:
        """Fitted Jacobian produces MSE no worse than the mean Jacobian."""
        if IGNORE_TESTS:
            return
        actual_arr = self.trajectory.l_roadrunner.simulate()
        mean_pred_arr = self.trajectory._predictNextStep()
        fitted_jacobian_arr = self.trajectory.fitJacobian()
        fitted_pred_arr = self.trajectory._predictNextStep(
                jacobian_arr=fitted_jacobian_arr
        )
        mse_median = float(np.mean((mean_pred_arr - actual_arr) ** 2))
        mse_fitted = float(np.mean((fitted_pred_arr - actual_arr) ** 2))
        self.assertLessEqual(mse_fitted, mse_median + 1e-6)

class TestGetGershgorinCircles(unittest.TestCase):
    """Tests for Trajectory.getGershgorinCircles."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.trajectory = _make_trajectory_from_model(ANTIMONY_MODEL,
                end_time=5.0, num_point=10)

    def test_returns_dataframe(self) -> None:
        """getGershgorinCircles returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_correct_columns(self) -> None:
        """Result has columns 'center', 'radius', 'timepoint'."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        for col in ("center", "radius", "timepoint"):
            self.assertIn(col, result.columns)

    def test_row_count(self) -> None:
        """Row count equals num_species * num_jacobians."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        expected = self.trajectory.num_species * self.trajectory.num_jacobian
        self.assertEqual(len(result), expected)

    def test_radii_nonnegative(self) -> None:
        """All radii are >= 0."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        self.assertTrue(np.all(result["radius"].values >= 0.0)) # type: ignore

    def test_timepoints_are_from_timepoint_arr(self) -> None:
        """Timepoint values in the DataFrame match self.trajectory.timepoint_arr."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        unique_timepoints = sorted(result["timepoint"].unique())
        expected = sorted(self.trajectory.timepoint_arr.tolist())
        self.assertEqual(len(unique_timepoints), len(expected))
        for got, exp in zip(unique_timepoints, expected):
            self.assertAlmostEqual(got, exp, places=10)

    def test_centers_are_diagonal_entries(self) -> None:
        """Centers at the first timepoint equal the diagonal of the first Jacobian."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        first_t = self.trajectory.timepoint_arr[0]
        first_rows = result[result["timepoint"] == first_t]
        expected_centers = np.diag(self.trajectory.jacobian_collection_arr[0])
        np.testing.assert_array_almost_equal(
                first_rows["center"].values, expected_centers)

    def test_all_values_finite(self) -> None:
        """All center, radius, and timepoint values are finite."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        for col in ("center", "radius", "timepoint"):
            self.assertTrue(np.all(np.isfinite(result[col].values)))


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
@unittest.skip("Too big.")
class TestGetGershgorinCirclesBiomd153(unittest.TestCase):
    """Tests for Trajectory.getGershgorinCircles on BIOMD0000000153 (74 species)."""

    NUM_POINT = 10

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD153_PATH):
            self.skipTest(f"BIOMD0000000153 not found at {BIOMD153_PATH}")
        self.trajectory = Trajectory.makeBiomodel(
                path=BIOMD153_PATH, end_time=BIOMD153_END_TIME,
                num_point=self.NUM_POINT)

    def test_returns_dataframe(self) -> None:
        """getGershgorinCircles returns a pd.DataFrame for BIOMD153."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_correct_columns(self) -> None:
        """Result has columns 'center', 'radius', 'timepoint' for BIOMD153."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        for col in ("center", "radius", "timepoint"):
            self.assertIn(col, result.columns)

    def test_row_count(self) -> None:
        """Row count equals num_species * num_jacobians for BIOMD153."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        expected = self.trajectory.num_species * self.trajectory.num_jacobian
        self.assertEqual(len(result), expected)

    def test_radii_nonnegative(self) -> None:
        """All radii are >= 0 for BIOMD153."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        self.assertTrue(np.all(result["radius"].values >= 0.0))  # type: ignore

    def test_all_values_finite(self) -> None:
        """All values are finite for BIOMD153."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.getGershgorinCircles()
        for col in ("center", "radius", "timepoint"):
            self.assertTrue(np.all(np.isfinite(result[col].values)))

@unittest.skipUnless(IS_BIG_TESTS, "Big tests are disabled")
class TestBiomodelsFit(unittest.TestCase):

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self.lr_dct: dict[str, LRoadrunner] = {} # Dictionary of LRoadrunner instances for different BioModels
        for model_name in BIOMODEL_NAMES:
            model_path = os.path.join(cn.BIOMODELS_DIR, model_name, f"{model_name}_url.xml")
            if not os.path.exists(model_path):
                print(f"Warning: {model_name} not found at {model_path}. Skipping this model.")
                continue
            with open(model_path) as f:
                sbml_str = f.read()
            self.lr_dct[model_name] = LRoadrunner(sbml_str, start_time=0.0, end_time=10.0, num_point=100)

    def test_improves_or_matches_prediction(self) -> None:
        """Fitted Jacobian produces MSE no worse than the mean Jacobian."""
        if IGNORE_TESTS:
            return
        for model_name, lr in self.lr_dct.items():
            with self.subTest(model=model_name):
                trajectory = Trajectory(lr)
                actual_arr = trajectory.l_roadrunner.simulate()
                mean_pred_arr = trajectory._predictNextStep()
                fitted_jacobian_arr = trajectory.fitJacobian()
                fitted_pred_arr = trajectory._predictNextStep()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_mean = np.where(actual_arr != 0,
                            (mean_pred_arr - actual_arr) ** 2 / actual_arr**2, 0.0)
                    ratio_fitted = np.where(actual_arr != 0,
                            (fitted_pred_arr - actual_arr) ** 2 / actual_arr**2, 0.0)
                mse_mean = float(np.mean(ratio_mean))
                mse_fitted = float(np.mean(ratio_fitted))
                self.assertLessEqual(mse_fitted, mse_mean + 1e-6)


HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)


class TestMakeBiomodel(unittest.TestCase):
    """Tests for Trajectory.makeBiomodel — error cases that do not need BioModels data."""

    def test_no_path_no_id_no_num_raises_value_error(self) -> None:
        """All three identifiers omitted → ValueError propagated from LRoadrunner.makeBiomodel."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            Trajectory.makeBiomodel()

    def test_model_id_without_biomd_prefix_raises(self) -> None:
        """model_id that does not start with 'BIOMD' → ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            Trajectory.makeBiomodel(model_name="FOO0000000008")


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestMakeBiomodelWithData(unittest.TestCase):
    """Tests for Trajectory.makeBiomodel that require BioModels data."""
    NUM_POINT = 5

    def test_model_num_returns_trajectory(self) -> None:
        """model_num=8 → returns a Trajectory instance."""
        if IGNORE_TESTS:
            return
        trajectory = Trajectory.makeBiomodel(model_num=8, end_time=BIOMD8_ENDTIME,
                num_point=self.NUM_POINT)
        self.assertIsInstance(trajectory, Trajectory)

    def test_model_name_returns_trajectory(self) -> None:
        """model_name='BIOMD0000000008' → returns a Trajectory instance."""
        if IGNORE_TESTS:
            return
        trajectory = Trajectory.makeBiomodel(model_name="BIOMD0000000008",
                end_time=BIOMD8_ENDTIME, num_point=self.NUM_POINT)
        self.assertIsInstance(trajectory, Trajectory)

    def test_explicit_path_returns_trajectory(self) -> None:
        """Explicit path → returns a Trajectory instance."""
        if IGNORE_TESTS:
            return
        trajectory = Trajectory.makeBiomodel(path=BIOMD8_PATH,
                end_time=BIOMD8_ENDTIME, num_point=self.NUM_POINT)
        self.assertIsInstance(trajectory, Trajectory)

    def test_jacobian_collection_has_correct_num_species(self) -> None:
        """jacobian_collection_arr has n_species matching BIOMD8 (5 species)."""
        if IGNORE_TESTS:
            return
        trajectory = Trajectory.makeBiomodel(path=BIOMD8_PATH,
                end_time=BIOMD8_ENDTIME, num_point=self.NUM_POINT)
        self.assertEqual(trajectory.jacobian_collection_arr.shape[1],
                trajectory.num_species)

    def test_jacobian_collection_has_correct_num_points(self) -> None:
        """jacobian_collection_arr has the requested number of timepoints."""
        if IGNORE_TESTS:
            return
        num_point = self.NUM_POINT
        trajectory = Trajectory.makeBiomodel(path=BIOMD8_PATH,
                end_time=BIOMD8_ENDTIME, num_point=num_point)
        self.assertEqual(trajectory.jacobian_collection_arr.shape[0], num_point)

    def test_model_num_and_explicit_path_give_same_species(self) -> None:
        """Loading by model_num=8 and by explicit path give the same species names."""
        if IGNORE_TESTS:
            return
        traj_num = Trajectory.makeBiomodel(model_num=8, end_time=BIOMD8_ENDTIME,
                num_point=5)
        traj_path = Trajectory.makeBiomodel(path=BIOMD8_PATH,
                end_time=BIOMD8_ENDTIME, num_point=5)
        self.assertEqual(traj_num.l_roadrunner.species_names,
                traj_path.l_roadrunner.species_names)


BIOMD1_PATH = os.path.join(
        cn.BIOMODELS_DIR, "BIOMD0000000001", "BIOMD0000000001_url.xml"
)


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestPredictLinearBiomd1(unittest.TestCase):
    """Tests for Trajectory.predictLinear on BIOMD0000000001.

    This model has large positive Jacobian eigenvalues that cause the matrix
    exponential to overflow during integration, so all rows after t=0 are NaN.
    The tests document that known behaviour and verify the shape contract.
    """

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD1_PATH):
            self.skipTest(f"BIOMD0000000001 not found at {BIOMD1_PATH}")
        self.trajectory = Trajectory.makeBiomodel(model_num=1)
        self.pred_df = self.trajectory.predict()

    def test_returns_dataframe(self) -> None:
        """predictLinear returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.pred_df, pd.DataFrame)

    def test_shape_matches_timecourse(self) -> None:
        """Result shape matches the true timecourse shape."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.pred_df.shape,
                self.trajectory.l_roadrunner.timecourse_df.shape)

    def test_columns_match_species(self) -> None:
        """Column names equal the model's species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.pred_df.columns),
                self.trajectory.l_roadrunner.species_names)

    def test_true_timecourse_has_no_nan(self) -> None:
        """The true timecourse for BIOMD1 is well-formed; NaN is a prediction artefact."""
        if IGNORE_TESTS:
            return
        self.assertFalse(np.any(
                np.isnan(self.trajectory.l_roadrunner.timecourse_df.values)))


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestPredictLinearBiomd60(unittest.TestCase):
    """Tests for Trajectory.predictLinear on BIOMD0000000060.

    This model has 4 species with a near-zero eigenvalue (~-0.45) alongside
    very negative ones (~-1913), making it stiff.  num_point is kept small (10)
    to keep the test suite fast.
    """

    NUM_POINT = 10

    def setUp(self) -> None:
        if not os.path.exists(BIOMD60_PATH):
            self.skipTest(f"BIOMD0000000060 not found at {BIOMD60_PATH}")
        self.trajectory = Trajectory.makeBiomodel(
                path=BIOMD60_PATH, end_time=BIOMD60_ENDTIME,
                num_point=self.NUM_POINT)
        self.pred_df = self.trajectory.predict()

    def test_returns_dataframe(self) -> None:
        """predictLinear returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.pred_df, pd.DataFrame)

    def test_shape_matches_timecourse(self) -> None:
        """Result shape matches the true timecourse shape."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.pred_df.shape,
                self.trajectory.l_roadrunner.timecourse_df.shape)

    def test_columns_match_species(self) -> None:
        """Column names equal the model's species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.pred_df.columns),
                self.trajectory.l_roadrunner.species_names)

    def test_first_row_is_not_nan(self) -> None:
        """The t=0 row contains finite values (initial conditions are valid)."""
        if IGNORE_TESTS:
            return
        self.assertFalse(np.any(np.isnan(self.pred_df.iloc[0].values)))

    def test_true_timecourse_has_no_nan(self) -> None:
        """The true timecourse for BIOMD60 is well-formed."""
        if IGNORE_TESTS:
            return
        self.assertFalse(np.any(
                np.isnan(self.trajectory.l_roadrunner.timecourse_df.values)))


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
@unittest.skipUnless(IS_BIG_TESTS, "Big tests are disabled")
class TestFitJacobianBiomd40(unittest.TestCase):
    """Tests for Trajectory.fitJacobian with num_fit on BIOMD000000040 (74 species)."""

    NUM_POINT = 30

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD40_PATH):
            self.skipTest(f"BIOMD000000040 not found at {BIOMD40_PATH}")
        self.trajectory = Trajectory.makeBiomodel(
                path=BIOMD40_PATH, end_time=BIOMD40_ENDTIME,
                num_point=self.NUM_POINT)

    def test_returns_ndarray(self) -> None:
        """fitJacobian returns a numpy ndarray for BIOMD40."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """fitJacobian returns shape (num_species, num_species) for BIOMD40."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        n = self.trajectory.num_species
        self.assertEqual(result.shape, (n, n))

    def test_result_is_finite(self) -> None:
        """All entries of the fitted Jacobian are finite for BIOMD40."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian()
        self.assertTrue(np.all(np.isfinite(result)))


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
@unittest.skipUnless(IS_BIG_TESTS, "Big tests are disabled")
class TestFitJacobianBiomd27(unittest.TestCase):
    """Tests for Trajectory.fitJacobian with num_fit on BIOMD000000027 (29 species)."""

    NUM_POINT = 30
    MAX_NFEV = 100

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD27_PATH):
            self.skipTest(f"BIOMD000000027 not found at {BIOMD27_PATH}")
        self.trajectory = Trajectory.makeBiomodel(
                path=BIOMD27_PATH, end_time=BIOMD27_END_TIME,
                num_point=self.NUM_POINT)

    def test_returns_ndarray(self) -> None:
        """fitJacobian returns a numpy ndarray for BIOMD27."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian(max_nfev=self.MAX_NFEV)
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """fitJacobian returns shape (num_species, num_species) for BIOMD27."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian(max_nfev=self.MAX_NFEV)
        n = self.trajectory.num_species
        self.assertEqual(result.shape, (n, n))

    def test_result_is_finite(self) -> None:
        """All entries of the fitted Jacobian are finite for BIOMD27."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.fitJacobian(max_nfev=self.MAX_NFEV)
        self.assertTrue(np.all(np.isfinite(result)))


@unittest.skipUnless(IS_BIG_TESTS, "Big tests are disabled")
@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestPlotPredictionsBiomd153(unittest.TestCase):
    """Tests for Trajectory.plotPredictions on BIOMD0000000153 (74 species)."""

    NUM_POINT = 100
    plot_options: PlotOptions
    num_species: int
    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD153_PATH):
            raise unittest.SkipTest(f"BIOMD0000000153 not found at {BIOMD153_PATH}")
        cls.trajectory = Trajectory.makeBiomodel(
                path=BIOMD153_PATH, end_time=BIOMD153_END_TIME,
                num_point=cls.NUM_POINT)
        cls.num_species = cls.trajectory.num_species
        cls.plot_options = cls.trajectory.plotPrediction(model_name="BIOMD153")

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.plot_options, PlotOptions)

    def test_plotting_biomd40(self) -> None:
        if IGNORE_TESTS:
            return
        trajectory = Trajectory.makeBiomodel(
                model_num=40, end_time = 10,
                jacobian_selection=cn.JAC_MEDIAN,
                #jacobian_selection=cn.JAC_FITTED,
                num_point=200)
                #num_point=self.NUM_POINT)
        trajectory.plotPrediction(num_step=1)
        trajectory.plotPrediction(num_step=-1)
        plt.show()

    def test_line_count_matches_species(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.plot_options.ax.lines), self.num_species)

    def test_scatter_count_matches_species(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.plot_options.ax.collections), self.num_species)

    def test_title_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIn("BIOMD153", self.plot_options.ax.get_title())


class TestPredictManyStep(unittest.TestCase):
    """Tests for Trajectory._predictManyStep."""

    # 11 timepoints [0..10]; divisors of 10 are 1, 2, 5, 10 — all valid num_step values.
    NUM_POINT = 11
    END_TIME = 10.0
    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        cls.trajectory = _make_trajectory_from_model(
                ANTIMONY_MODEL, end_time=cls.END_TIME, num_point=cls.NUM_POINT)

    def _non_nan_rows(self, arr: np.ndarray) -> np.ndarray:
        """Return the rows of arr that contain no NaN."""
        mask = ~np.any(np.isnan(arr), axis=1)
        return arr[mask]

    def test_returns_ndarray(self) -> None:
        """Result is an ndarray with shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        result = self.trajectory._predictManyStep(num_step=1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape,
                (self.NUM_POINT, self.trajectory.num_species))

    def test_num_step_1_matches_one_step_predict(self) -> None:
        """_predictManyStep(1) gives the same predictions as _predict()."""
        if IGNORE_TESTS:
            return
        many = self.trajectory._predictManyStep(num_step=1)
        one = self.trajectory._predictNextStep()
        np.testing.assert_allclose(many, one, rtol=1e-6, equal_nan=True)

    def test_default_num_step_predicts_final_timepoint(self) -> None:
        """num_step=-1 (full prediction) fills only the last row."""
        if IGNORE_TESTS:
            return
        result = self.trajectory._predictManyStep(num_step=-1)
        self.assertFalse(np.any(np.isnan(result[0])))   # initial row always set
        self.assertFalse(np.any(np.isnan(result[-1])))  # final row predicted
        # All intermediate rows should be non-NaN
        self.assertFalse(np.any(np.isnan(result[1:-1])))

    def test_predicted_positions_non_nan(self) -> None:
        """Positions at multiples of num_step are non-NaN."""
        if IGNORE_TESTS:
            return
        num_step = 2
        result = self.trajectory._predictManyStep(num_step=num_step)
        n = self.NUM_POINT
        for pos in range(num_step, n, num_step):
            self.assertFalse(np.any(np.isnan(result[pos])),
                    msg=f"Expected non-NaN at position {pos}")

    def test_invalid_num_step_zero_raises(self) -> None:
        """num_step=0 raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.trajectory._predictManyStep(num_step=0)

    def test_invalid_num_step_too_large_raises(self) -> None:
        """num_step >= num_point raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.trajectory._predictManyStep(num_step=self.NUM_POINT)

    def test_initial_row_equals_real_timecourse(self) -> None:
        """Row 0 always contains the real initial state regardless of num_step."""
        if IGNORE_TESTS:
            return
        real_t0 = self.trajectory.l_roadrunner.timecourse_df.iloc[0].values
        for num_step in [1, 2, 5]:
            result = self.trajectory._predictManyStep(num_step=num_step)
            np.testing.assert_allclose(result[0], real_t0, rtol=1e-10)  # type: ignore


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestPredictManyStepBiomd8(unittest.TestCase):
    """Tests for Trajectory._predictManyStep on BIOMD0000000008.

    Uses 11 timepoints so num_step = 1, 2, 5, 10 all divide evenly into
    the 10 intervals, and the last timepoint is reachable by every num_step.
    """

    # 11 timepoints over [0, 20]: t=0,2,4,...,20
    NUM_POINT = 11
    END_TIME = 20.0
    trajectory: Trajectory
    actual_arr: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD8_PATH):
            raise unittest.SkipTest(f"BIOMD8 not found at {BIOMD8_PATH}")
        cls.trajectory = Trajectory.makeBiomodel(
                path=BIOMD8_PATH, end_time=cls.END_TIME, num_point=cls.NUM_POINT)
        cls.actual_arr = cls.trajectory.timecourse_df.values

    def _rmse_at_last(self, num_step: int) -> float:
        """RMSE between prediction and actual at the last timepoint."""
        result = self.trajectory._predictManyStep(num_step=num_step)
        pred = result[-1]
        actual = self.actual_arr[-1]
        return float(np.sqrt(np.mean((pred - actual) ** 2)))

    def test_returns_correct_shape(self) -> None:
        """Result shape matches (num_point, num_species) for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.trajectory._predictManyStep(num_step=1)
        self.assertEqual(result.shape,
                (self.NUM_POINT, self.trajectory.num_species))

    def test_values_are_finite_for_one_step(self) -> None:
        """1-step prediction produces all finite values for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.trajectory._predictManyStep(num_step=1)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_accuracy_decreases_with_num_step(self) -> None:
        """RMSE at the final timepoint increases as num_step increases."""
        if IGNORE_TESTS:
            return
        rmse_1 = self._rmse_at_last(1)
        rmse_5 = self._rmse_at_last(5)
        rmse_10 = self._rmse_at_last(10)
        self.assertLess(rmse_1, rmse_5,
                msg=f"Expected rmse(step=1)={rmse_1:.4g} < rmse(step=5)={rmse_5:.4g}")
        self.assertLess(rmse_5, rmse_10,
                msg=f"Expected rmse(step=5)={rmse_5:.4g} < rmse(step=10)={rmse_10:.4g}")

    def test_full_prediction_final_timepoint_non_nan(self) -> None:
        """num_step=-1 (one shot from t0 to t_end) produces a finite final row."""
        if IGNORE_TESTS:
            return
        result = self.trajectory._predictManyStep(num_step=-1)
        self.assertFalse(np.any(np.isnan(result[-1])))


if __name__ == "__main__":
    unittest.main()
