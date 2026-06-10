"""Tests for LinearPredictor."""

import os
import unittest
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from trajectory import Trajectory  # type: ignore
from linear_predictor import LinearPredictor  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

# pylint: disable=invalid-name
IGNORE_TESTS = True
if not IGNORE_TESTS:
    matplotlib.use("Agg")

HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

NUM_POINT = 11
NUM_SPECIES = 2

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

# Single-species decay with a known analytical solution.
# dx/dt = -K_DECAY*x, x(0) = X0_DECAY => x(t) = X0_DECAY * exp(-K_DECAY * t)
ANTIMONY_DECAY = """
S1 -> ; k*S1
k = 0.2; S1 = 5.0
"""
K_DECAY = 0.2
X0_DECAY = 5.0

# Two-species model with a bimolecular nonlinear term (k3*S1*S2) so the
# Jacobian is concentration-dependent and the linear ODE is imperfect.
ANTIMONY_NONLINEAR = """
S1 -> S2; k1*S1
S2 ->   ; k2*S2
S1 ->   ; k3*S1*S2
k1 = 0.1; k2 = 0.2; k3 = 0.01; S1 = 10; S2 = 2
"""


def _makeModel(model_str: str = ANTIMONY_MODEL) -> Model:
    return Model(model_str)


def _makeAnalyticalTrajectory() -> Trajectory:
    """Trajectory for single-species decay with a known analytical solution."""
    timepoints = np.linspace(0.0, 10.0, NUM_POINT)
    jacobian_collection_arr = np.array([[[-K_DECAY]]] * NUM_POINT)
    forcing_input_collection_arr = np.zeros((NUM_POINT, 1))
    conc_arr = X0_DECAY * np.exp(-K_DECAY * timepoints)
    timecourse_df = pd.DataFrame({"S1": conc_arr}, index=timepoints)
    timecourse_df.index.name = "time"
    return Trajectory(
            model=_makeModel(ANTIMONY_DECAY),
            jacobian_collection_arr=jacobian_collection_arr,
            timepoint_arr=timepoints,
            forcing_input_collection_arr=forcing_input_collection_arr,
            timecourse_df=timecourse_df,
    )


def _makeTrajectory() -> Trajectory:
    return Trajectory.makeFromSimulation(
            _makeModel(), num_point=NUM_POINT)


def _makeNonlinearTrajectory() -> Trajectory:
    return Trajectory.makeFromSimulation(
            _makeModel(ANTIMONY_NONLINEAR), num_point=NUM_POINT)


class TestLinearPredictorInit(unittest.TestCase):
    """Tests for LinearPredictor.__init__."""

    def setUp(self) -> None:
        self.trajectory = _makeTrajectory()

    def test_stores_trajectory(self) -> None:
        """trajectory is stored on the instance."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory)
        self.assertIs(predictor.trajectory, self.trajectory)

    def test_default_jacobian_selection(self) -> None:
        """Default jacobian_selection is cn.JAC_MEDIAN."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory)
        self.assertEqual(predictor.jacobian_selection, cn.JAC_MEDIAN)

    def test_stores_jacobian_selection(self) -> None:
        """Explicit jacobian_selection is stored."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.trajectory, jacobian_selection=cn.JAC_FIRST)
        self.assertEqual(predictor.jacobian_selection, cn.JAC_FIRST)

    def test_num_equal_1_step(self) -> None:
        """Default num_step is 1."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory, num_step=1)
        self.assertEqual(predictor.num_step, 1)

    def test_stores_num_step(self) -> None:
        """Explicit num_step is stored."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory, num_step=5)
        self.assertEqual(predictor.num_step, 5)

    def test_zero_num_step_raises(self) -> None:
        """num_step=0 raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LinearPredictor(self.trajectory, num_step=0)

    def test_negative_one_num_step_resolves_to_num_point_minus_one(self) -> None:
        """num_step=-1 is stored as trajectory.num_point - 1."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory, num_step=-1)
        self.assertEqual(predictor.num_step, self.trajectory.num_point - 1)


class TestLinearPredictorGetJacobian(unittest.TestCase):
    """Tests for LinearPredictor._getJacobian."""
    # pylint: disable=protected-access

    def setUp(self) -> None:
        self.trajectory = _makeTrajectory()

    def test_median_returns_jacobian_median_arr(self) -> None:
        """JAC_MEDIAN returns the trajectory's jacobian_median_arr."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.trajectory, jacobian_selection=cn.JAC_MEDIAN)
        np.testing.assert_array_equal(
                predictor._getJacobian(), self.trajectory.jacobian_median_arr)

    def test_first_returns_first_jacobian(self) -> None:
        """JAC_FIRST returns jacobian_collection_arr[0]."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.trajectory, jacobian_selection=cn.JAC_FIRST)
        np.testing.assert_array_equal(
                predictor._getJacobian(),
                self.trajectory.jacobian_collection_arr[0])

    def test_fitted_returns_ndarray_of_correct_shape(self) -> None:
        """JAC_FITTED returns an ndarray of shape (num_species, num_species)."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        predictor = LinearPredictor(traj, jacobian_selection=cn.JAC_FITTED)
        jac = predictor._getJacobian()
        self.assertIsInstance(jac, np.ndarray)
        n = traj.model.num_species
        self.assertEqual(jac.shape, (n, n))

    def test_fitted_stores_fitted_jacobian_arr(self) -> None:
        """JAC_FITTED sets _fitted_jacobian_arr on the predictor."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        predictor = LinearPredictor(traj, jacobian_selection=cn.JAC_FITTED)
        predictor._getJacobian()
        self.assertTrue(hasattr(predictor, "_fitted_jacobian_arr"))

    def test_unknown_selection_raises(self) -> None:
        """An unknown jacobian_selection raises ValueError."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.trajectory, jacobian_selection="bad_selection")
        with self.assertRaises(ValueError):
            predictor._getJacobian()


class TestLinearPredictorPredict(unittest.TestCase):
    """Tests for LinearPredictor.predict."""

    trajectory: Trajectory
    predictor: LinearPredictor
    prediction_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.predictor = LinearPredictor(cls.trajectory)
        cls.prediction_df = cls.predictor.predict()

    def test_returns_dataframe(self) -> None:
        """predict returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.prediction_df, pd.DataFrame)

    def test_shape(self) -> None:
        """predict returns shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        expected = (self.trajectory.num_point,
                self.trajectory.model.num_species)
        self.assertEqual(self.prediction_df.shape, expected)

    def test_columns_match_species_names(self) -> None:
        """Columns equal the model's species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.prediction_df.columns),
                self.trajectory.model.species_names)

    def test_index_matches_timepoints(self) -> None:
        """Index matches the Trajectory's timepoint_arr."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.index.values,
                self.trajectory.timepoint_arr)

    def test_initial_value_matches(self) -> None:
        """First predicted row matches the initial state in timecourse_df."""
        if IGNORE_TESTS:
            return
        expected = self.trajectory.timecourse_df.iloc[0].values
        np.testing.assert_allclose(             # type: ignore
                self.prediction_df.iloc[0].values, expected, atol=1e-5)  # type: ignore

    def test_analytical_solution(self) -> None:
        """Prediction matches the analytical solution for a 1-species decay."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        predictor = LinearPredictor(traj)
        pred_df = predictor.predict()
        expected = X0_DECAY * np.exp(-K_DECAY * traj.timepoint_arr)
        np.testing.assert_allclose(pred_df["S1"].values, expected, atol=1e-4)  # type: ignore

    def test_jac_first_returns_finite_values(self) -> None:
        """predict with JAC_FIRST returns finite values."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.trajectory, jacobian_selection=cn.JAC_FIRST)
        pred_df = predictor.predict()
        self.assertTrue(np.all(np.isfinite(pred_df.values)))

    def test_fitted_jacobian_returns_finite_values(self) -> None:
        """predict with JAC_FITTED returns finite values."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        predictor = LinearPredictor(traj, jacobian_selection=cn.JAC_FITTED)
        pred_df = predictor.predict()
        self.assertTrue(np.all(np.isfinite(pred_df.values)))

    def test_values_all_finite(self) -> None:
        """All predicted values are finite."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.prediction_df.values)))

    def test_num_step_large_gives_correct_shape(self) -> None:
        """A num_step larger than num_point gives shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory, num_step=1000)
        pred_df = predictor.predict()
        expected = (self.trajectory.num_point,
                self.trajectory.model.num_species)
        self.assertEqual(pred_df.shape, expected)

    def test_num_step_equals_one_gives_correct_shape(self) -> None:
        """num_step=1 gives shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.trajectory, num_step=1)
        pred_df = predictor.predict()
        expected = (self.trajectory.num_point,
                self.trajectory.model.num_species)
        self.assertEqual(pred_df.shape, expected)

    def test_different_num_step_same_result_for_linear_model(self) -> None:
        """For an exactly linear model, num_step does not affect predictions."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        pred1 = LinearPredictor(traj, num_step=1).predict()
        pred5 = LinearPredictor(traj, num_step=5).predict()
        np.testing.assert_allclose(pred1.values, pred5.values, atol=1e-4)


class TestLinearPredictorScore(unittest.TestCase):
    """Tests for LinearPredictor.score."""
    trajectory: Trajectory
    predictor: LinearPredictor

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.predictor = LinearPredictor(cls.trajectory)

    def test_returns_dataframe(self) -> None:
        """score returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_aggregation_type_column(self) -> None:
        """Returned DataFrame has an 'aggregation_type' column."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertIn("aggregation_type", result.columns)

    def test_has_mean_column(self) -> None:
        """Returned DataFrame has a 'mean' column."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertIn("mean", result.columns)

    def test_contains_model_row(self) -> None:
        """Returned DataFrame contains a row with aggregation_type == 'model'."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertTrue(result["aggregation_type"].eq("model").any())

    def test_description_stored(self) -> None:
        """Provided description appears in the returned DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score(description="mytest")
        self.assertTrue(result["description"].eq("mytest").any())

    def test_mean_finite_for_linear_model(self) -> None:
        """Model-level mean is finite for an exactly linear model."""
        if IGNORE_TESTS:
            return
        traj = _makeAnalyticalTrajectory()
        predictor = LinearPredictor(traj)
        result = predictor.score()
        model_mean = result.loc[result["aggregation_type"] == "model", "mean"].iloc[-1]
        self.assertTrue(np.isfinite(model_mean))


class TestLinearPredictorCost(unittest.TestCase):
    """Tests for LinearPredictor.cost."""

    def test_returns_float(self) -> None:
        """cost returns a float."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(_makeAnalyticalTrajectory())
        self.assertIsInstance(predictor.cost, float)

    def test_non_negative(self) -> None:
        """cost is non-negative."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(_makeAnalyticalTrajectory())
        self.assertGreaterEqual(predictor.cost, 0.0)

    def test_near_zero_for_exact_model(self) -> None:
        """cost is near zero for the analytical decay model with its exact Jacobian."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(_makeAnalyticalTrajectory(),
                jacobian_selection=cn.JAC_MEDIAN)
        self.assertAlmostEqual(predictor.cost, 0.0, places=4)

    def test_larger_num_step_increases_cost(self) -> None:
        """cost increases as num_step grows for a non-trivial model."""
        if IGNORE_TESTS:
            return
        traj = _makeTrajectory()
        cost_1 = LinearPredictor(traj, num_step=1).cost
        cost_large = LinearPredictor(traj, num_step=traj.num_point - 1).cost
        self.assertLessEqual(cost_1, cost_large)

    def test_multi_species_returns_scalar(self) -> None:
        """cost returns a single float for a multi-species model."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(_makeTrajectory())
        self.assertIsInstance(predictor.cost, float)


class TestLinearPredictorPlotPrediction(unittest.TestCase):
    """Tests for LinearPredictor.plotPrediction."""
    trajectory: Trajectory
    predictor: LinearPredictor

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.predictor = LinearPredictor(cls.trajectory)

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertIsInstance(plot_options, PlotOptions)

    def test_creates_axes_when_none(self) -> None:
        """plotPrediction creates a new figure and axes when none are supplied."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertEqual(len(plot_options.fig.axes), 1)  # type: ignore

    def test_uses_provided_axes(self) -> None:
        """plotPrediction draws onto the supplied axes."""
        if IGNORE_TESTS:
            return
        ext_fig, ext_ax = plt.subplots()
        plot_options = self.predictor.plotPrediction(ax=ext_ax, fig=ext_fig)
        self.assertIs(plot_options.ax, ext_ax)

    def test_line_count_two_per_species(self) -> None:
        """plotPrediction draws 2 lines per species (actual + predicted)."""
        if IGNORE_TESTS:
            return
        n_species = self.trajectory.model.num_species
        plot_options = self.predictor.plotPrediction()
        self.assertEqual(len(plot_options.ax.lines), 2 * n_species)  # type: ignore

    def test_axis_labels(self) -> None:
        """Default xlabel and ylabel are applied to the axes."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertEqual(plot_options.ax.get_xlabel(), plot_options.xlabel) # type: ignore
        self.assertEqual(plot_options.ax.get_ylabel(), plot_options.ylabel) # type: ignore

    def test_has_legend(self) -> None:
        """plotPrediction adds a legend by default."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertIsNotNone(plot_options.ax.get_legend())  # type: ignore

    def test_prediction_lines_dashed(self) -> None:
        """Prediction lines use dashed linestyle."""
        if IGNORE_TESTS:
            return
        n_species = self.trajectory.model.num_species
        plot_options = self.predictor.plotPrediction()
        dashed = [ln for ln in plot_options.ax.lines if ln.get_linestyle() == "--"] # type: ignore
        self.assertEqual(len(dashed), n_species)

    def test_custom_title(self) -> None:
        """Title kwarg is applied to the axes."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction(title="My Title")
        self.assertEqual(plot_options.ax.get_title(), "My Title")  # type: ignore

    def test_xlim_applied(self) -> None:
        """xlim kwarg is applied to the axes."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction(xlim=(1.0, 5.0))
        lo, hi = plot_options.ax.get_xlim() # type: ignore
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_ylim_applied(self) -> None:
        """ylim kwarg is applied to the axes."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction(ylim=(0.0, 20.0))
        lo, hi = plot_options.ax.get_ylim() # type: ignore
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 20.0)

    def test_model_name_prepended_to_title(self) -> None:
        """model_name is prepended to the title when both are provided."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction(
                title="Prediction", model_name="BIOMD8")
        self.assertIn("BIOMD8", plot_options.ax.get_title())  # type: ignore
        self.assertIn("Prediction", plot_options.ax.get_title())  # type: ignore


class TestLinearPredictorPredictWithJacobian(unittest.TestCase):
    """Tests for LinearPredictor._predictWithJacobian."""
    # pylint: disable=protected-access

    def setUp(self) -> None:
        self.traj = _makeAnalyticalTrajectory()
        self.predictor = LinearPredictor(self.traj)
        self.jac = self.traj.jacobian_median_arr

    def test_returns_ndarray(self) -> None:
        """_predictWithJacobian returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.predictor._predictWithJacobian(self.jac)
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self) -> None:
        """_predictWithJacobian returns shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        result = self.predictor._predictWithJacobian(self.jac)
        expected = (self.traj.num_point, self.traj.model.num_species)
        self.assertEqual(result.shape, expected)

    def test_first_row_matches_initial_condition(self) -> None:
        """First row of result equals the initial observed state."""
        if IGNORE_TESTS:
            return
        result = self.predictor._predictWithJacobian(self.jac)
        np.testing.assert_array_equal(result[0], self.traj.timecourse_df.iloc[0].values)

    def test_values_finite(self) -> None:
        """All returned values are finite."""
        if IGNORE_TESTS:
            return
        result = self.predictor._predictWithJacobian(self.jac)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_analytical_solution_accuracy(self) -> None:
        """Predictions match the analytical solution for a 1-species decay."""
        if IGNORE_TESTS:
            return
        result = self.predictor._predictWithJacobian(self.jac)
        expected = X0_DECAY * np.exp(-K_DECAY * self.traj.timepoint_arr)
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-4)  # type: ignore

    def test_num_step_minus_one_gives_correct_shape(self) -> None:
        """num_step=-1 (full trajectory) gives shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj, num_step=-1)
        result = predictor._predictWithJacobian(self.jac)
        expected = (self.traj.num_point, self.traj.model.num_species)
        self.assertEqual(result.shape, expected)

    def test_num_step_minus_one_same_as_full_trajectory(self) -> None:
        """num_step=-1 gives the same result as num_step=num_point-1."""
        if IGNORE_TESTS:
            return
        pred_minus_one = LinearPredictor(
                self.traj, num_step=-1)._predictWithJacobian(self.jac)
        pred_full = LinearPredictor(
                self.traj, num_step=self.traj.num_point - 1)._predictWithJacobian(self.jac)
        np.testing.assert_allclose(pred_minus_one, pred_full, atol=1e-10)  # type: ignore


class TestLinearPredictorIsSpeciesPrediction(unittest.TestCase):
    """Tests for LinearPredictor with is_species_prediction=True."""
    # pylint: disable=protected-access
    traj_nonlinear: Trajectory
    traj_1sp = _makeAnalyticalTrajectory()
    traj_2sp = _makeTrajectory()

    @classmethod
    def setUpClass(cls) -> None:
        cls.traj_1sp = _makeAnalyticalTrajectory()
        cls.traj_2sp = _makeTrajectory()
        cls.traj_nonlinear = _makeNonlinearTrajectory()

    def test_default_is_false(self) -> None:
        """Default is_species_prediction is False."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp)
        self.assertFalse(predictor.is_species_prediction)

    def test_stores_true(self) -> None:
        """is_species_prediction=True is stored on the instance."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp, is_species_prediction=True)
        self.assertTrue(predictor.is_species_prediction)

    def test_returns_correct_shape(self) -> None:
        """_predictWithJacobian returns (num_point, num_species) with is_species_prediction=True."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp, is_species_prediction=True)
        result = predictor._predictWithJacobian(self.traj_2sp.jacobian_median_arr)
        self.assertEqual(result.shape,
                (self.traj_2sp.num_point, self.traj_2sp.model.num_species))

    def test_first_row_matches_initial_condition(self) -> None:
        """First row of the result equals the actual initial condition."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp, is_species_prediction=True)
        result = predictor._predictWithJacobian(self.traj_2sp.jacobian_median_arr)
        np.testing.assert_array_equal(
                result[0], self.traj_2sp.timecourse_df.iloc[0].values)

    def test_values_are_finite(self) -> None:
        """All returned values are finite."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp, is_species_prediction=True)
        result = predictor._predictWithJacobian(self.traj_2sp.jacobian_median_arr)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_single_species_matches_standard(self) -> None:
        """For a 1-species model, is_species_prediction=True gives identical predictions.

        With one species there are no other species to condition on.
        x_i = actual_arr.copy(); x_i[0] = x[0] merely overwrites the copy with
        x[0] itself, leaving the ODE identical to the standard case.
        """
        if IGNORE_TESTS:
            return
        jac = self.traj_1sp.jacobian_median_arr
        pred_standard = LinearPredictor(
                self.traj_1sp,
                is_species_prediction=False)._predictWithJacobian(jac)
        pred_species = LinearPredictor(
                self.traj_1sp,
                is_species_prediction=True)._predictWithJacobian(jac)
        np.testing.assert_allclose(pred_species, pred_standard, atol=1e-10)  # type: ignore

    def test_nonlinear_model_differs_from_standard(self) -> None:
        """For a nonlinear model with a long window, is_species_prediction=True
        differs from is_species_prediction=False.

        The Jacobian is concentration-dependent in ANTIMONY_NONLINEAR, so the
        linear ODE approximation drifts from the actual trajectory.  With
        num_step=-1 (single window), is_species_prediction=True corrects the
        cross-species terms with actual trajectory values, yielding predictions
        that differ measurably from the standard coupled ODE.
        """
        if IGNORE_TESTS:
            return
        jac = self.traj_nonlinear.jacobian_median_arr
        pred_standard = LinearPredictor(
                self.traj_nonlinear, num_step=-1,
                is_species_prediction=False)._predictWithJacobian(jac)
        pred_species = LinearPredictor(
                self.traj_nonlinear, num_step=-1,
                is_species_prediction=True)._predictWithJacobian(jac)
        self.assertFalse(np.allclose(pred_standard, pred_species, atol=1e-6))

    def test_predict_returns_correct_dataframe(self) -> None:
        """predict() returns a correctly shaped DataFrame with is_species_prediction=True."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(self.traj_2sp, is_species_prediction=True)
        pred_df = predictor.predict()
        self.assertIsInstance(pred_df, pd.DataFrame)
        self.assertEqual(pred_df.shape,
                (self.traj_2sp.num_point, self.traj_2sp.model.num_species))
        self.assertEqual(list(pred_df.columns),
                self.traj_2sp.model.species_names)

    def test_cost_is_finite_and_nonnegative(self) -> None:
        """cost is a finite non-negative float with is_species_prediction=True."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor(
                self.traj_nonlinear, is_species_prediction=True)
        cost = predictor.cost
        self.assertIsInstance(cost, float)
        self.assertTrue(np.isfinite(cost))
        self.assertGreaterEqual(cost, 0.0)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestLinearPredictorBiomodel8(unittest.TestCase):
    """Integration tests for LinearPredictor with BIOMD0000000008."""
    model: Model
    trajectory: Trajectory
    predictor: LinearPredictor
    prediction_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.makeBiomodel("BIOMD0000000008")
        cls.trajectory = Trajectory.makeFromSimulation(
                cls.model, num_point=NUM_POINT)
        cls.predictor = LinearPredictor(cls.trajectory)
        cls.prediction_df = cls.predictor.predict()

    def tearDown(self) -> None:
        plt.close("all")

    def test_predict_shape(self) -> None:
        """predict returns shape (num_point, num_species) for BIOMD8."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.prediction_df.shape,
                (self.trajectory.num_point, self.model.num_species))

    def test_predict_values_finite(self) -> None:
        """All predicted values are finite for BIOMD8."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.prediction_df.values)))

    def test_score_returns_dataframe(self) -> None:
        """score returns a pd.DataFrame for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertIsInstance(result, pd.DataFrame)

    def test_score_model_mean_finite(self) -> None:
        """Model-level mean is finite for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        model_mean = result.loc[result["aggregation_type"] == "model", "mean"].iloc[-1]
        self.assertTrue(np.isfinite(model_mean))

    def test_plot_prediction_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions for BIOMD8."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertIsInstance(plot_options, PlotOptions)

    def test_plot_prediction_line_count(self) -> None:
        """plotPrediction draws 2 lines per species for BIOMD8."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertEqual(len(plot_options.ax.lines), 2 * self.model.num_species)  # type: ignore


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestLinearPredictorBiomodel40(unittest.TestCase):
    """Integration tests for LinearPredictor with BIOMD0000000040."""
    model: Model
    trajectory: Trajectory
    predictor: LinearPredictor
    prediction_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.makeBiomodel("BIOMD0000000040")
        cls.trajectory = Trajectory.makeFromSimulation(
                cls.model, num_point=NUM_POINT)
        cls.predictor = LinearPredictor(cls.trajectory)
        cls.prediction_df = cls.predictor.predict()

    def tearDown(self) -> None:
        plt.close("all")

    def test_predict_shape(self) -> None:
        """predict returns shape (num_point, num_species) for BIOMD40."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.prediction_df.shape,
                (self.trajectory.num_point, self.model.num_species))

    def test_predict_values_finite(self) -> None:
        """All predicted values are finite for BIOMD40."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.prediction_df.values)))

    def test_score_returns_dataframe(self) -> None:
        """score returns a pd.DataFrame for BIOMD40."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        self.assertIsInstance(result, pd.DataFrame)

    def test_score_model_mean_finite(self) -> None:
        """Model-level mean is finite for BIOMD40."""
        if IGNORE_TESTS:
            return
        result = self.predictor.score()
        model_mean = result.loc[result["aggregation_type"] == "model", "mean"].iloc[-1]
        self.assertTrue(np.isfinite(model_mean))

    def test_plot_prediction_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions for BIOMD40."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertIsInstance(plot_options, PlotOptions)

    def test_plot_prediction_line_count(self) -> None:
        """plotPrediction draws 2 lines per species for BIOMD40."""
        if IGNORE_TESTS:
            return
        plot_options = self.predictor.plotPrediction()
        self.assertEqual(len(plot_options.ax.lines), 2 * self.model.num_species)  # type: ignore


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestLinearPredictorMakeFromBiomodels(unittest.TestCase):
    """Tests for LinearPredictor.makeFromBiomodels."""

    def test_returns_linear_predictor_from_name(self) -> None:
        """makeFromBiomodels with model_name returns a LinearPredictor."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor.makeFromBiomodels(
                model_name="BIOMD0000000008")
        self.assertIsInstance(predictor, LinearPredictor)

    def test_returns_linear_predictor_from_num(self) -> None:
        """makeFromBiomodels with model_num returns a LinearPredictor."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor.makeFromBiomodels(model_num=8)
        self.assertIsInstance(predictor, LinearPredictor)

    def test_name_and_num_equivalent(self) -> None:
        """model_name and model_num resolve to the same model."""
        if IGNORE_TESTS:
            return
        p_name = LinearPredictor.makeFromBiomodels(
                model_name="BIOMD0000000008")
        p_num = LinearPredictor.makeFromBiomodels(model_num=8)
        self.assertEqual(
                p_name.trajectory.model.model_name,
                p_num.trajectory.model.model_name)

    def test_both_supplied_raises(self) -> None:
        """Supplying both model_name and model_num raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LinearPredictor.makeFromBiomodels(
                    model_name="BIOMD0000000008", model_num=8)

    def test_neither_supplied_raises(self) -> None:
        """Supplying neither model_name nor model_num raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LinearPredictor.makeFromBiomodels()

    def test_stores_jacobian_selection(self) -> None:
        """jacobian_selection is forwarded to the predictor."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor.makeFromBiomodels(
                model_num=8, jacobian_selection=cn.JAC_FIRST)
        self.assertEqual(predictor.jacobian_selection, cn.JAC_FIRST)

    def test_stores_num_step(self) -> None:
        """num_step is forwarded to the predictor."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor.makeFromBiomodels(
                model_num=8, num_step=3)
        self.assertEqual(predictor.num_step, 3)

    def test_predict_returns_finite_values(self) -> None:
        """predict on the returned predictor returns finite values."""
        if IGNORE_TESTS:
            return
        predictor = LinearPredictor.makeFromBiomodels(model_num=40)
        pred_df = predictor.predict()
        self.assertTrue(np.all(np.isfinite(pred_df.values)))

@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestLinearPredictorBug1(unittest.TestCase):
    """Test for bug in score. Seems to be calculated wrong."""

    def test_bug(self):
        #if IGNORE_TESTS:
        #    return
        traj = Trajectory.makeBiomodel(36)
        lp = LinearPredictor(traj)
        lp.plotPrediction()
        plt.show()


if __name__ == "__main__":
    unittest.main()
