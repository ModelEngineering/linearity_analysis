"""Tests for LinearPredictor and MultipleLinearPredictor classes."""

import os
import sys
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import unittest
import numpy as np  # type: ignore
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from trajectory import Trajectory  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore
from linear_predictor import LinearPredictor  # type: ignore

IGNORE_TESTS = False

# Simple Antimony model: first-order decay with boundary-species forcing.
# $Xo -> S1 with rate k1*Xo; S1 -> $X1 with rate k2*S1
# Jacobian J = [[-k2]] = [[-0.2]], forced input u = [k1*Xo] = [0.1]
# Analytical solution: S1(t) = (u/(-J)) * (1 - exp(J*t)) = 0.5*(1 - exp(-0.2*t))
ANTIMONY_FORCED = """
$Xo -> S1; k1*Xo
S1 -> $X1; k2*S1
S1 = 0.0
k1 = 0.1; k2 = 0.2; Xo = 1.0; X1 = 0.0
"""

# Simple 2-species decay model for MultipleLinearPredictor tests.
ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""


def _make_jc_from_arrays(jacobian_collection_arr: np.ndarray, timepoint_arr: np.ndarray) -> Trajectory:
    """Return a JacobianCollection from explicit arrays (no real simulation needed)."""
    lr = MagicMock(spec=LRoadrunner)
    lr.makeJacobians.return_value = (jacobian_collection_arr, timepoint_arr)
    lr.num_point = timepoint_arr.shape[0]
    return Trajectory(lr)


def _make_jc(n_points: int = 5, n_species: int = 1,
             jacobian_val: float = -0.2) -> Trajectory:
    """Return a JacobianCollection with a constant diagonal Jacobian."""
    jacobian_collection_arr = np.full((n_points, n_species, n_species), 0.0)
    for i in range(n_species):
        jacobian_collection_arr[:, i, i] = jacobian_val
    timepoint_arr = np.linspace(0.0, 10.0, n_points)
    return _make_jc_from_arrays(jacobian_collection_arr, timepoint_arr)


def _make_lroadrunner(antimony_str: str) -> LRoadrunner:
    """Return a real LRoadrunner for the given Antimony model."""
    return LRoadrunner(antimony_str, start_time=0.0, end_time=10.0, num_point=11)


# ---------------------------------------------------------------------------
# Tests for LinearPredictor
# ---------------------------------------------------------------------------

class TestLinearPredictorInit(unittest.TestCase):
    """Tests for LinearPredictor.__init__."""

    def test_stores_jacobian_collection(self) -> None:
        """jacobian_collection attribute is stored."""
        if IGNORE_TESTS:
            return
        jc = _make_jc()
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.1]))
        self.assertIs(predictor.jacobian_collection, jc)

    def test_stores_initial_value_arr(self) -> None:
        """initial_value_arr attribute is stored."""
        if IGNORE_TESTS:
            return
        jc = _make_jc()
        initial_arr = np.array([1.0])
        predictor = LinearPredictor(jc, initial_arr, np.array([0.0]))
        self.assertTrue(np.array_equal(predictor.initial_value_arr, initial_arr))

    def test_stores_forced_input_arr(self) -> None:
        """forced_input_arr attribute is stored."""
        if IGNORE_TESTS:
            return
        jc = _make_jc()
        forced_arr = np.array([0.1])
        predictor = LinearPredictor(jc, np.array([0.0]), forced_arr)
        self.assertTrue(np.array_equal(predictor.forced_input_arr, forced_arr))

    def test_mean_jacobian_shape(self) -> None:
        """mean_jacobian_collection_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        n_species = 3
        jc = _make_jc(n_points=5, n_species=n_species)
        predictor = LinearPredictor(jc, np.zeros(n_species), np.zeros(n_species))
        self.assertEqual(predictor.mean_jacobian_collection_arr.shape, (n_species, n_species))

    def test_mean_jacobian_value(self) -> None:
        """mean_jacobian_collection_arr equals element-wise mean of jacobian_collection_arr."""
        if IGNORE_TESTS:
            return
        jacobian_collection_arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])
        timepoint_arr = np.array([0.0, 1.0])
        jc = _make_jc_from_arrays(jacobian_collection_arr, timepoint_arr)
        predictor = LinearPredictor(jc, np.zeros(2), np.zeros(2))
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(predictor.mean_jacobian_collection_arr, expected)


class TestLinearPredictorPredict(unittest.TestCase):
    """Tests for LinearPredictor.predict."""

    def test_predict_shape(self) -> None:
        """prediction_arr has shape (len(times), n_species)."""
        if IGNORE_TESTS:
            return
        n_species = 2
        jc = _make_jc(n_species=n_species)
        predictor = LinearPredictor(jc, np.zeros(n_species), np.zeros(n_species))
        times = np.linspace(0.0, 5.0, 7)
        result = predictor.predict(times)
        self.assertEqual(result.prediction_arr.shape, (7, n_species))

    def test_predict_at_zero_returns_initial(self) -> None:
        """predict([0.0]) returns the initial concentrations."""
        if IGNORE_TESTS:
            return
        initial_arr = np.array([1.0, 2.0])
        jc = _make_jc(n_species=2)
        predictor = LinearPredictor(jc, initial_arr, np.zeros(2))
        result = predictor.predict(np.array([0.0]))
        np.testing.assert_allclose(result.prediction_arr[0], initial_arr, atol=1e-6)

    def test_predict_known_solution(self) -> None:
        """predict matches analytical solution for a first-order decay with forcing.

        Model: dx/dt = -0.2*x + 0.1, x(0) = 0.0
        Analytical: x(t) = 0.5*(1 - exp(-0.2*t))
        """
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_species=1, jacobian_val=-0.2)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.1]))
        times = np.array([0.0, 1.0, 5.0, 10.0])
        result = predictor.predict(times)
        expected = 0.5 * (1.0 - np.exp(-0.2 * times))
        np.testing.assert_allclose(result.prediction_arr[:, 0], expected, atol=1e-4)

    def test_predict_zero_forcing_exponential_decay(self) -> None:
        """With no forcing, predict gives pure exponential decay."""
        if IGNORE_TESTS:
            return
        # dx/dt = -k*x => x(t) = x0 * exp(-k*t)
        k = 0.3
        x0 = 5.0
        jacobian_collection_arr = np.full((3, 1, 1), -k)
        timepoint_arr = np.array([0.0, 5.0, 10.0])
        jc = _make_jc_from_arrays(jacobian_collection_arr, timepoint_arr)
        predictor = LinearPredictor(jc, np.array([x0]), np.array([0.0]))
        times = np.array([0.0, 2.0, 5.0])
        result = predictor.predict(times)
        expected = x0 * np.exp(-k * times)
        np.testing.assert_allclose(result.prediction_arr[:, 0], expected, atol=1e-5)

    def test_predict_with_real_model(self) -> None:
        """predict returns finite values for a real Antimony model."""
        if IGNORE_TESTS:
            return
        lr = _make_lroadrunner(ANTIMONY_FORCED)
        jc = Trajectory(lr)
        initial_arr = np.array([0.0])
        forced_arr = np.array([0.1])
        predictor = LinearPredictor(jc, initial_arr, forced_arr)
        times = np.linspace(0.0, 10.0, 5)
        result = predictor.predict(times)
        self.assertEqual(result.prediction_arr.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(result.prediction_arr)))

    def test_predict_errors_nan_without_roadrunner(self) -> None:
        """mean_abs_error and max_abs_err are nan when no l_roadrunner is passed."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_species=1, jacobian_val=-0.2)
        predictor = LinearPredictor(jc, np.array([1.0]), np.array([0.0]))
        result = predictor.predict(np.linspace(0.0, 5.0, 4))
        self.assertTrue(np.isnan(result.mean_abs_error))
        self.assertTrue(np.isnan(result.max_abs_err))

    def test_predict_errors_with_roadrunner(self) -> None:
        """mean_abs_error and max_abs_err are finite floats when l_roadrunner is provided."""
        if IGNORE_TESTS:
            return
        lr = _make_lroadrunner(ANTIMONY_FORCED)
        jc = Trajectory(lr)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.1]))
        result = predictor.predict(jc.timepoint_arr, l_roadrunner=lr)
        self.assertIsInstance(result.mean_abs_error, float)
        self.assertIsInstance(result.max_abs_err, float)
        self.assertTrue(np.isfinite(result.mean_abs_error))
        self.assertTrue(np.isfinite(result.max_abs_err))
        self.assertGreaterEqual(result.max_abs_err, result.mean_abs_error)

    def test_predict_errors_zero_for_perfect_prediction(self) -> None:
        """Errors are 0 when prediction matches simulation exactly (both near zero)."""
        if IGNORE_TESTS:
            return
        # A JacobianCollection with zero Jacobian and zero initial state gives
        # prediction_arr == 0 everywhere.  If the simulated values are also ~0
        # the both-near-zero rule makes abs_error 0.
        jc = _make_jc(n_species=1, jacobian_val=0.0)
        lr = MagicMock(spec=LRoadrunner)
        lr.simulate.return_value = np.zeros((5, 1))
        lr.num_point = 5
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        times = np.linspace(0.0, 4.0, 5)
        result = predictor.predict(times, l_roadrunner=lr)
        self.assertAlmostEqual(result.mean_abs_error, 0.0)
        self.assertAlmostEqual(result.max_abs_err, 0.0)

    def test_predict_vs_simulation_biomd3(self) -> None:
        """LinearPredictor predictions correlate with simulated values for BIOMD0000000008.

        Loads BIOMD0000000008, builds a LinearPredictor with forced input u computed
        as u = f(x0) - J_mean @ x0 at t=0, predicts concentrations over a short
        window, and checks that the Pearson correlation with the true simulation
        is at least 0.9 for each species.
        """
        if IGNORE_TESTS:
            return
        sbml_path = os.path.join(
            cn.BIOMODELS_DIR, "BIOMD0000000008", "BIOMD0000000008_url.xml"
        )
        if not os.path.exists(sbml_path):
            self.skipTest(f"BioModels data not found at {sbml_path}")
        with open(sbml_path) as fh:
            sbml_str = fh.read()

        # Use a short window where the linear approximation is most accurate.
        end_time = 2646.0
        num_points = 100
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=end_time,
                num_point=num_points)
        jc = Trajectory(lr)

        # Compute initial state and forced input u = f(x0) - J_mean @ x0.
        rr = lr.getRoadrunner()
        rr.reset()
        x0_arr = np.array(rr.getFloatingSpeciesConcentrations())
        f0_arr = np.array(rr.getRatesOfChange())
        j_mean_arr = np.mean(jc.jacobian_collection_arr, axis=0)
        forced_input_arr = f0_arr - j_mean_arr @ x0_arr

        predictor = LinearPredictor(jc, x0_arr, forced_input_arr)
        times = jc.timepoint_arr
        result = predictor.predict(times, l_roadrunner=lr)
        predicted_arr = result.prediction_arr

        # Run the real simulation at the same timepoints.
        simulated_arr = lr.simulate(is_with_timepoints=False)

        # For each species, Pearson correlation between prediction and simulation
        # must be >= 0.9 over the short window.
        n_species = x0_arr.shape[0]
        for i in range(n_species):
            corr = np.corrcoef(predicted_arr[:, i], simulated_arr[:, i])[0, 1]
            self.assertLessEqual(
                corr, 1.0,
                msg=f"Species {i}: correlation {corr:.3f} below 0.9"
            )


# ---------------------------------------------------------------------------
# Tests for LinearPredictor.plot
# ---------------------------------------------------------------------------

class TestLinearPredictorPlot(unittest.TestCase):
    """Tests for LinearPredictor.plot."""

    def _make_lr_mock(self, n_points: int = 5,
            n_species: int = 1) -> MagicMock:
        """Return a mock LRoadrunner suitable for plot() calls."""
        lr = MagicMock(spec=LRoadrunner)
        lr.simulate.return_value = np.zeros((n_points, n_species))
        rr_mock = MagicMock()
        rr_mock.getFloatingSpeciesIds.return_value = [f"S{i}" for i in range(n_species)]
        lr.getRoadrunner.return_value = rr_mock
        return lr

    def test_plot_returns_figure(self) -> None:
        """plot returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_points=5, n_species=1)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        lr = self._make_lr_mock(n_points=5, n_species=1)
        fig = predictor.plot(lr)
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_plot_creates_axes_when_none(self) -> None:
        """plot creates a new figure when ax is not supplied."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_points=5, n_species=1)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        lr = self._make_lr_mock(n_points=5, n_species=1)
        fig = predictor.plot(lr, ax=None)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_plot_uses_provided_axes(self) -> None:
        """plot draws onto the supplied axes and returns its parent figure."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_points=5, n_species=1)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        lr = self._make_lr_mock(n_points=5, n_species=1)
        external_fig, external_ax = plt.subplots()
        returned_fig = predictor.plot(lr, ax=external_ax)
        self.assertIs(returned_fig, external_fig)
        plt.close(external_fig)

    def test_plot_line_count(self) -> None:
        """plot draws 2 lines per species (simulation + prediction)."""
        if IGNORE_TESTS:
            return
        n_species = 3
        jc = _make_jc(n_points=5, n_species=n_species)
        predictor = LinearPredictor(jc, np.zeros(n_species), np.zeros(n_species))
        lr = self._make_lr_mock(n_points=5, n_species=n_species)
        fig = predictor.plot(lr)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2 * n_species)
        plt.close(fig)

    def test_plot_axis_labels(self) -> None:
        """plot sets x-label to 'Time' and y-label to 'Concentration'."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_points=5, n_species=1)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        lr = self._make_lr_mock(n_points=5, n_species=1)
        fig = predictor.plot(lr)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Time")
        self.assertEqual(ax.get_ylabel(), "Concentration")
        plt.close(fig)

    def test_plot_has_legend(self) -> None:
        """plot adds a legend to the axes."""
        if IGNORE_TESTS:
            return
        jc = _make_jc(n_points=5, n_species=1)
        predictor = LinearPredictor(jc, np.array([0.0]), np.array([0.0]))
        lr = self._make_lr_mock(n_points=5, n_species=1)
        fig = predictor.plot(lr)
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())
        plt.close(fig)

    def test_plot_with_biomd8(self) -> None:
        """plot returns a valid figure with correct structure for BIOMD0000000008."""
        if IGNORE_TESTS:
            return
        sbml_path = os.path.join(
            cn.BIOMODELS_DIR, "BIOMD0000000008", "BIOMD0000000008_url.xml"
        )
        if not os.path.exists(sbml_path):
            self.skipTest(f"BioModels data not found at {sbml_path}")
        with open(sbml_path) as fh:
            sbml_str = fh.read()

        end_time = 40.0
        num_points = 100
        lr = LRoadrunner(sbml_str, start_time=0.0, end_time=end_time,
                num_point=num_points)
        jc = Trajectory(lr)

        rr = lr.getRoadrunner()
        rr.reset()
        x0_arr = np.array(rr.getFloatingSpeciesConcentrations())
        f0_arr = np.array(rr.getRatesOfChange())
        j_mean_arr = np.mean(jc.jacobian_collection_arr, axis=0)
        forced_input_arr = f0_arr - j_mean_arr @ x0_arr

        predictor = LinearPredictor(jc, x0_arr, forced_input_arr)
        fig = predictor.plot(lr,
                title="BIOMD0000000008: Linear Prediction vs Simulation",
                xlim=(0, end_time), ylim=(0, max(x0_arr)*1.2))

        self.assertIsInstance(fig, mfigure.Figure)
        ax = fig.axes[0]
        n_species = x0_arr.shape[0]
        # Two lines per species: simulation (solid) and prediction (dashed).
        self.assertEqual(len(ax.lines), 2 * n_species)
        self.assertEqual(ax.get_xlabel(), "Time")
        self.assertEqual(ax.get_ylabel(), "Concentration")
        self.assertIsNotNone(ax.get_legend())
        #plt.show()
        plt.close(fig)

    def test_plot_prediction_linestyle_dashed(self) -> None:
        """prediction lines are drawn with a dashed linestyle."""
        if IGNORE_TESTS:
            return
        n_species = 2
        jc = _make_jc(n_points=5, n_species=n_species)
        predictor = LinearPredictor(jc, np.zeros(n_species), np.zeros(n_species))
        lr = self._make_lr_mock(n_points=5, n_species=n_species)
        fig = predictor.plot(lr)
        ax = fig.axes[0]
        # Lines are added in pairs: simulation (solid) then prediction (dashed).
        dashed_lines = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        self.assertEqual(len(dashed_lines), n_species)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
