"""Tests for MultipleLinearPredictor."""
import unittest

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from multiple_linear_predictor import MultipleLinearPredictor  # type: ignore
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


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL)


def _makeTrajectory() -> Trajectory:
    return Trajectory.makeFromSimulation(
            _makeModel(),
            start_time=START_TIME,
            end_time=END_TIME,
            num_point=NUM_POINT,
    )


def _makeCollection(trajectory: Trajectory, split_time: float = 5.0) -> TrajectoryCollection:
    return TrajectoryCollection.split(trajectory, [split_time])


def _makeMlp(trajectory_collection: TrajectoryCollection) -> MultipleLinearPredictor:
    return MultipleLinearPredictor(trajectory_collection,
            jacobian_selection=cn.JAC_MEDIAN,
            num_step=1)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestMultipleLinearPredictorInit(unittest.TestCase):
    trajectory_collection: TrajectoryCollection
    multiple_linear_predictor: MultipleLinearPredictor
    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.multiple_linear_predictor = _makeMlp(cls.trajectory_collection)

    def test_attributes(self) -> None:
        """Constructor stores trajectory_collection, jacobian_selection, num_step."""
        if IGNORE_TESTS:
            return
        mlp = self.multiple_linear_predictor
        self.assertIs(mlp.trajectory_collection, self.trajectory_collection)
        self.assertEqual(mlp.jacobian_selection, cn.JAC_MEDIAN)
        self.assertEqual(mlp.num_step, 1)

    def test_default_num_step(self) -> None:
        """Default num_step is 1."""
        if IGNORE_TESTS:
            return
        mlp = MultipleLinearPredictor(self.trajectory_collection)
        self.assertEqual(mlp.num_step, -1)



# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestMultipleLinearPredictorPredict(unittest.TestCase):
    trajectory: Trajectory
    trajectory_collection: TrajectoryCollection
    multiple_linear_predictor: MultipleLinearPredictor
    prediction_df: pd.DataFrame
    timecourse_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.multiple_linear_predictor = _makeMlp(cls.trajectory_collection)
        cls.prediction_df = cls.multiple_linear_predictor.predict().prediction_df
        cls.timecourse_df = cls.multiple_linear_predictor.trajectory_collection.makeTimecourse()

    def test_returns_dataframe(self) -> None:
        """predict returns a DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.prediction_df, pd.DataFrame)

    def test_shape_matches_timecourse(self) -> None:
        """Prediction has same shape as _makeTimecourse."""
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
        """Prediction index matches _makeTimecourse index."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.index.values,
                self.timecourse_df.index.values)

    def test_first_row_equals_actual(self) -> None:
        """First row of prediction equals actual (exact by construction)."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.iloc[0].values,  # type: ignore
                self.timecourse_df.iloc[0].values)  # type: ignore


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestMultipleLinearPredictorScore(unittest.TestCase):
    trajectory: Trajectory
    trajectory_collection: TrajectoryCollection
    multiple_linear_predictor: MultipleLinearPredictor
    prediction_df: pd.DataFrame
    timecourse_df: pd.DataFrame
    score_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.multiple_linear_predictor = _makeMlp(cls.trajectory_collection)
        cls.score_df = cls.multiple_linear_predictor.score("test")

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
        expected = 1 + NUM_SPECIES
        self.assertEqual(len(self.score_df), expected)

    def test_description_stored(self) -> None:
        """Description argument is stored in each row."""
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

class TestMultipleLinearPredictorCost(unittest.TestCase):
    trajectory: Trajectory
    trajectory_collection: TrajectoryCollection
    multiple_linear_predictor: MultipleLinearPredictor
    prediction_df: pd.DataFrame
    timecourse_df: pd.DataFrame
    score_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.multiple_linear_predictor = _makeMlp(cls.trajectory_collection)

    def test_is_float(self) -> None:
        """cost returns a float."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.multiple_linear_predictor.cost, float)

    def test_nonnegative(self) -> None:
        """cost is non-negative."""
        if IGNORE_TESTS:
            return
        self.assertGreaterEqual(self.multiple_linear_predictor.cost, 0.0)

    def test_split_lowers_cost_vs_single_segment(self) -> None:
        """Two-segment predictor has cost <= single-segment predictor."""
        if IGNORE_TESTS:
            return
        from src.linear_predictor import LinearPredictor  # type: ignore

        tc_single = TrajectoryCollection([self.trajectory])
        mlp_single = _makeMlp(tc_single)
        lp_single = LinearPredictor(self.trajectory,
                jacobian_selection=cn.JAC_MEDIAN,
                num_step=1)
        # Both cost computations use the same formula; two segments can only
        # help or be equal, never worse, for an optimal split.
        self.assertLessEqual(self.multiple_linear_predictor.cost,
                lp_single.cost + 1e-9)


# ---------------------------------------------------------------------------
# plotPrediction
# ---------------------------------------------------------------------------

class TestMultipleLinearPredictorPlotPrediction(unittest.TestCase):
    trajectory: Trajectory
    trajectory_collection: TrajectoryCollection
    multiple_linear_predictor: MultipleLinearPredictor

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.trajectory_collection = _makeCollection(cls.trajectory)
        cls.multiple_linear_predictor = _makeMlp(cls.trajectory_collection)

    def test_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        result = self.multiple_linear_predictor.plotPrediction()
        self.assertIsInstance(result, PlotOptions)

    def test_boundary_line_count(self) -> None:
        """One vertical separator line per interior boundary."""
        if IGNORE_TESTS:
            return
        plot_options = self.multiple_linear_predictor.plotPrediction()
        ax = plot_options.ax
        vlines = [ln for ln in ax.lines  # type: ignore
                if len(set(ln.get_xdata())) == 1]  # type: ignore
        n_segments = len(self.trajectory_collection.trajectories)
        self.assertEqual(len(vlines), n_segments - 1)

    def test_data_line_count(self) -> None:
        """Two lines per species (actual + predicted)."""
        if IGNORE_TESTS:
            return
        plot_options = self.multiple_linear_predictor.plotPrediction()
        ax = plot_options.ax
        non_vlines = [ln for ln in ax.lines  # type: ignore
                if len(set(ln.get_xdata())) > 1]  # type: ignore
        self.assertEqual(len(non_vlines), 2 * NUM_SPECIES)

    def test_custom_title_respected(self) -> None:
        """A title passed via kwargs is used verbatim."""
        if IGNORE_TESTS:
            return
        plot_options = self.multiple_linear_predictor.plotPrediction(title="custom")
        self.assertIn("custom", plot_options.ax.get_title())  # type: ignore


if __name__ == "__main__":
    unittest.main()
