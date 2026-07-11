"""Tests for MultipleSlowSubspacePredictor."""
import unittest

import matplotlib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore
from src.multiple_slow_subspace_predictor import MultipleSlowSubspacePredictor  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_SPECIES = 2
NUM_POINT = 21
START_TIME = 0.0
END_TIME = 10.0
SPLIT_TIME = 5.0


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL)


def _makeTrajectory() -> Trajectory:
    return Trajectory.makeFromSimulation(
            _makeModel(),
            start_time=START_TIME,
            end_time=END_TIME,
            num_point=NUM_POINT,
    )


def _makeCollection(trajectory: Trajectory) -> TrajectoryCollection:
    return TrajectoryCollection.split(trajectory, [SPLIT_TIME])


def _makeMssp(trajectory_collection: TrajectoryCollection,
        **kwargs) -> MultipleSlowSubspacePredictor:
    return MultipleSlowSubspacePredictor(trajectory_collection, num_step=1, **kwargs)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.mssp = _makeMssp(cls.trajectory_collection)

    def test_stores_trajectory_collection(self) -> None:
        """Constructor stores trajectory_collection."""
        if IGNORE_TESTS:
            return
        self.assertIs(self.mssp.trajectory_collection, self.trajectory_collection)

    def test_stores_num_step(self) -> None:
        """Constructor stores num_step."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.mssp.num_step, 1)

    def test_stores_eigenvalue_threshold(self) -> None:
        """Constructor stores eigenvalue_threshold (None by default)."""
        if IGNORE_TESTS:
            return
        self.assertIsNone(self.mssp.eigenvalue_threshold)

    def test_stores_explicit_threshold(self) -> None:
        """Explicit eigenvalue_threshold is stored correctly."""
        if IGNORE_TESTS:
            return
        mssp = MultipleSlowSubspacePredictor(
                self.trajectory_collection, eigenvalue_threshold=0.5)
        self.assertAlmostEqual(mssp.eigenvalue_threshold, 0.5)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.mssp = _makeMssp(cls.trajectory_collection)
        cls.prediction_df = cls.mssp.predict()
        cls.timecourse_df = cls.trajectory_collection.makeTimecourse()

    def test_returns_dataframe(self) -> None:
        """predict returns a DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.prediction_df, pd.DataFrame)

    def test_shape_matches_timecourse(self) -> None:
        """Prediction has same shape as makeTimecourse."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.prediction_df.shape, self.timecourse_df.shape)

    def test_columns_are_species_names(self) -> None:
        """Columns match the model species names."""
        if IGNORE_TESTS:
            return
        expected = self.trajectory_collection.model.species_names
        self.assertListEqual(list(self.prediction_df.columns), expected)

    def test_index_matches_timecourse(self) -> None:
        """Prediction index matches makeTimecourse index."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.index.values,
                self.timecourse_df.index.values)

    def test_first_row_equals_actual(self) -> None:
        """First row of prediction equals the actual initial condition."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.iloc[0].values,
                self.timecourse_df.iloc[0].values)

    def test_no_duplicate_boundary_rows(self) -> None:
        """Index has no duplicate values (boundary timepoints appear once)."""
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.prediction_df.index),
                len(self.prediction_df.index.unique()))


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.mssp = _makeMssp(cls.trajectory_collection)
        cls.score_df = cls.mssp.score("test")

    def test_returns_dataframe(self) -> None:
        """score returns a DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.score_df, pd.DataFrame)

    def test_has_model_row(self) -> None:
        """First row has aggregation_type='model'."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.score_df.iloc[0]["aggregation_type"], "model")

    def test_num_rows(self) -> None:
        """Number of rows is 1 (model) + num_species."""
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.score_df), 1 + NUM_SPECIES)

    def test_description_stored(self) -> None:
        """Description argument is stored in every row."""
        if IGNORE_TESTS:
            return
        for desc in self.score_df["description"]:
            self.assertEqual(desc, "test")

    def test_p95_is_finite(self) -> None:
        """p95 column contains finite values."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.score_df["p95"].values)))


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorCost(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.mssp = _makeMssp(cls.trajectory_collection)

    def test_is_float(self) -> None:
        """cost returns a float."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.mssp.cost, float)

    def test_nonnegative(self) -> None:
        """cost is non-negative."""
        if IGNORE_TESTS:
            return
        self.assertGreaterEqual(self.mssp.cost, 0.0)

    def test_split_lowers_cost_vs_single_segment(self) -> None:
        """Two-segment predictor has cost <= single-segment predictor."""
        if IGNORE_TESTS:
            return
        from src.slow_subspace_predictor import SlowSubspacePredictor  # type: ignore
        ssp_single = SlowSubspacePredictor(self.trajectory, num_step=1)
        self.assertLessEqual(self.mssp.cost, ssp_single.cost + 1e-9)


# ---------------------------------------------------------------------------
# plotPrediction
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorPlotPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.mssp = _makeMssp(cls.trajectory_collection)

    def test_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        result = self.mssp.plotPrediction()
        self.assertIsInstance(result, PlotOptions)

    def test_boundary_line_count(self) -> None:
        """One vertical separator line per interior boundary."""
        if IGNORE_TESTS:
            return
        plot_options = self.mssp.plotPrediction()
        ax = plot_options.ax
        vlines = [ln for ln in ax.lines  # type: ignore
                if len(set(ln.get_xdata())) == 1]  # type: ignore
        n_segments = len(self.trajectory_collection.trajectories)
        self.assertEqual(len(vlines), n_segments - 1)

    def test_data_line_count(self) -> None:
        """Two lines per species (actual and predicted)."""
        if IGNORE_TESTS:
            return
        plot_options = self.mssp.plotPrediction()
        ax = plot_options.ax
        non_vlines = [ln for ln in ax.lines  # type: ignore
                if len(set(ln.get_xdata())) > 1]  # type: ignore
        self.assertEqual(len(non_vlines), 2 * NUM_SPECIES)

    def test_custom_title_respected(self) -> None:
        """Title kwarg is used verbatim."""
        if IGNORE_TESTS:
            return
        plot_options = self.mssp.plotPrediction(title="custom")
        self.assertIn("custom", plot_options.ax.get_title())  # type: ignore


# ---------------------------------------------------------------------------
# Stiff model comparison (BIOMD0000000599)
# ---------------------------------------------------------------------------

class TestMultipleSlowSubspacePredictorStiffModel(unittest.TestCase):
    """Verify that MSSP outperforms MultipleLinearPredictor on BIOMD0000000599.

    A 5-way autoSplit of the stiff model still leaves LinearPredictor struggling
    (cost > 1.0) because stiffness dominates every segment.  MSSP removes the
    fast Schur modes per segment and corrects for QSS, achieving much lower cost.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from src.multiple_linear_predictor import MultipleLinearPredictor  # type: ignore
        model = Model.makeBiomodel("BIOMD0000000599")
        trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=None, num_point=101)
        tc = TrajectoryCollection.autoSplit(trajectory, num_split=2)
        cls.mssp = MultipleSlowSubspacePredictor(tc, num_step=1)
        cls.mlp = MultipleLinearPredictor(tc,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)

    def test_mssp_cost_lower_than_mlp(self) -> None:
        """MSSP cost is substantially lower than MultipleLinearPredictor cost."""
        if IGNORE_TESTS:
            return
        self.assertLess(self.mssp.cost, self.mlp.cost / 5.0)

    def test_predict_correct_shape(self) -> None:
        """predict returns DataFrame with correct number of rows and 30 columns."""
        if IGNORE_TESTS:
            return
        pred_df = self.mssp.predict()
        self.assertEqual(pred_df.shape[1], 30)
        self.assertGreater(pred_df.shape[0], 0)

    def test_cost_is_finite(self) -> None:
        """MSSP cost is finite (no runaway predictions)."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.isfinite(self.mssp.cost))


if __name__ == "__main__":
    unittest.main()
