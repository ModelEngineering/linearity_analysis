"""Tests for SlowSubspacePredictor."""
import unittest

import matplotlib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from trajectory import Trajectory  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore
from slow_subspace_predictor import SlowSubspacePredictor  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

# Simple two-species linear model; Jacobian eigenvalues ≈ -0.1 and -0.2.
ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""
NUM_SPECIES = 2
NUM_POINT = 21
START_TIME = 0.0
END_TIME = 10.0
# step_size = 10/20 = 0.5 → default threshold = 2.0
# Both eigenvalues (-0.1, -0.2) are slow (< 2.0)
# threshold=0.15 → only -0.1 is slow (1 slow mode)
# threshold=0.05 → no modes are slow (0 slow modes)
THRESHOLD_ALL_SLOW = 2.0
THRESHOLD_ONE_SLOW = 0.15
THRESHOLD_NONE_SLOW = 0.05


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL)


def _makeTrajectory() -> Trajectory:
    return Trajectory.makeFromSimulation(
            _makeModel(),
            start_time=START_TIME,
            end_time=END_TIME,
            num_point=NUM_POINT,
    )


def _makeSsp(trajectory: Trajectory,
        threshold: float = THRESHOLD_ALL_SLOW) -> SlowSubspacePredictor:
    return SlowSubspacePredictor(trajectory,
            eigenvalue_threshold=threshold,
            num_step=1)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.ssp = _makeSsp(cls.trajectory)

    def test_stores_trajectory(self) -> None:
        """trajectory is stored on the instance."""
        if IGNORE_TESTS:
            return
        self.assertIs(self.ssp.trajectory, self.trajectory)

    def test_stores_threshold(self) -> None:
        """eigenvalue_threshold is stored correctly."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.ssp.eigenvalue_threshold, THRESHOLD_ALL_SLOW)

    def test_stores_num_step(self) -> None:
        """num_step is stored correctly."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.ssp.num_step, 1)

    def test_default_threshold_equals_one_over_step_size(self) -> None:
        """Default threshold is 1 / step_size."""
        if IGNORE_TESTS:
            return
        ssp = SlowSubspacePredictor(self.trajectory, num_step=1)
        step_size = (END_TIME - START_TIME) / (NUM_POINT - 1)
        self.assertAlmostEqual(ssp.eigenvalue_threshold, 1.0 / step_size)

    def test_num_step_minus_one_uses_full_trajectory(self) -> None:
        """num_step=-1 is converted to num_point - 1."""
        if IGNORE_TESTS:
            return
        ssp = SlowSubspacePredictor(self.trajectory, num_step=-1)
        self.assertEqual(ssp.num_step, NUM_POINT - 1)

    def test_invalid_num_step_raises(self) -> None:
        """num_step=0 raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            SlowSubspacePredictor(self.trajectory, num_step=0)


# ---------------------------------------------------------------------------
# num_slow_modes
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorSlowModes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()

    def test_all_slow_gives_num_species_modes(self) -> None:
        """High threshold retains all modes (num_slow_modes == num_species)."""
        if IGNORE_TESTS:
            return
        ssp = _makeSsp(self.trajectory, threshold=THRESHOLD_ALL_SLOW)
        self.assertEqual(ssp.num_slow_modes, NUM_SPECIES)

    def test_low_threshold_gives_fewer_modes(self) -> None:
        """Low threshold removes fast modes."""
        if IGNORE_TESTS:
            return
        ssp = _makeSsp(self.trajectory, threshold=THRESHOLD_ONE_SLOW)
        self.assertLess(ssp.num_slow_modes, NUM_SPECIES)

    def test_zero_slow_modes_when_threshold_below_all_eigenvalues(self) -> None:
        """Threshold below all |Re(λ)| gives zero slow modes."""
        if IGNORE_TESTS:
            return
        ssp = _makeSsp(self.trajectory, threshold=THRESHOLD_NONE_SLOW)
        self.assertEqual(ssp.num_slow_modes, 0)

    def test_num_slow_modes_nonnegative(self) -> None:
        """num_slow_modes is always non-negative."""
        if IGNORE_TESTS:
            return
        for thresh in [THRESHOLD_NONE_SLOW, THRESHOLD_ONE_SLOW, THRESHOLD_ALL_SLOW]:
            ssp = _makeSsp(self.trajectory, threshold=thresh)
            self.assertGreaterEqual(ssp.num_slow_modes, 0)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.ssp = _makeSsp(cls.trajectory)
        cls.prediction_df = cls.ssp.predict()

    def test_returns_dataframe(self) -> None:
        """predict returns a DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.prediction_df, pd.DataFrame)

    def test_shape(self) -> None:
        """Prediction has shape (num_point, num_species)."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.prediction_df.shape, (NUM_POINT, NUM_SPECIES))

    def test_columns_are_species_names(self) -> None:
        """Columns match the model species names."""
        if IGNORE_TESTS:
            return
        self.assertListEqual(list(self.prediction_df.columns),
                self.trajectory.model.species_names)

    def test_index_matches_timepoints(self) -> None:
        """Index matches the trajectory timepoints."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.index.values,
                self.trajectory.timepoint_arr)

    def test_first_row_equals_actual(self) -> None:
        """First row equals the actual initial condition (exact by construction)."""
        if IGNORE_TESTS:
            return
        np.testing.assert_array_almost_equal(
                self.prediction_df.iloc[0].values,
                self.trajectory.timecourse_df.iloc[0].values)

    def test_zero_slow_modes_predicts_constant(self) -> None:
        """With zero slow modes, rows 1+ are all equal (QSS, not initial condition)."""
        if IGNORE_TESTS:
            return
        ssp_none = _makeSsp(self.trajectory, threshold=THRESHOLD_NONE_SLOW)
        pred_df = ssp_none.predict()
        for i in range(2, len(pred_df)):
            np.testing.assert_array_almost_equal(
                    pred_df.iloc[i].values, pred_df.iloc[1].values)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorCost(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.ssp_all = _makeSsp(cls.trajectory, threshold=THRESHOLD_ALL_SLOW)
        cls.lp = LinearPredictor(cls.trajectory,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)

    def test_is_float(self) -> None:
        """cost returns a float."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.ssp_all.cost, float)

    def test_nonnegative(self) -> None:
        """cost is non-negative."""
        if IGNORE_TESTS:
            return
        self.assertGreaterEqual(self.ssp_all.cost, 0.0)

    def test_all_slow_matches_linear_predictor(self) -> None:
        """When all modes are retained, SSP cost equals LinearPredictor cost.

        With all Schur vectors included the predictor is equivalent to
        LinearPredictor(JAC_MEDIAN): the reduced ODE is just a rotation of
        the full ODE.
        """
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.ssp_all.cost, self.lp.cost, places=5)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.ssp = _makeSsp(cls.trajectory)
        cls.score_df = cls.ssp.score("test")

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


# ---------------------------------------------------------------------------
# plotPrediction
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorPlotPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _makeTrajectory()
        cls.ssp = _makeSsp(cls.trajectory)

    def test_returns_plot_options(self) -> None:
        """plotPrediction returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        result = self.ssp.plotPrediction()
        self.assertIsInstance(result, PlotOptions)

    def test_data_line_count(self) -> None:
        """Two lines per species (actual and predicted)."""
        if IGNORE_TESTS:
            return
        plot_options = self.ssp.plotPrediction()
        ax = plot_options.ax
        self.assertEqual(len(ax.lines), 2 * NUM_SPECIES)  # type: ignore

    def test_custom_title_respected(self) -> None:
        """Title kwarg is used verbatim."""
        if IGNORE_TESTS:
            return
        plot_options = self.ssp.plotPrediction(title="custom")
        self.assertIn("custom", plot_options.ax.get_title())  # type: ignore


# ---------------------------------------------------------------------------
# Stiff model comparison (BIOMD0000000599)
# ---------------------------------------------------------------------------

class TestSlowSubspacePredictorStiffModel(unittest.TestCase):
    """Verify that SSP outperforms LinearPredictor on a stiff BioModel.

    BIOMD0000000599 has a max Jacobian eigenvalue of ~47.7 and a step size
    of 24 time units (stiffness ratio ~1143).  LinearPredictor cost ≈ 1.78;
    SSP removes the 8 fast Schur modes and corrects the slow forcing for the
    fast-mode QSS, achieving cost < 0.5.
    """

    @classmethod
    def setUpClass(cls) -> None:
        model = Model.makeBiomodel("BIOMD0000000599")
        cls.trajectory = Trajectory.makeFromSimulation(
                model, start_time=0.0, end_time=None, num_point=101)
        cls.ssp = SlowSubspacePredictor(cls.trajectory, num_step=1)
        cls.lp = LinearPredictor(cls.trajectory,
                jacobian_selection=cn.JAC_MEDIAN, num_step=1)

    def test_has_slow_and_fast_modes(self) -> None:
        """Some modes are slow and some are fast."""
        if IGNORE_TESTS:
            return
        n = self.trajectory.model.num_species
        self.assertGreater(self.ssp.num_slow_modes, 0)
        self.assertLess(self.ssp.num_slow_modes, n)

    def test_ssp_cost_lower_than_lp(self) -> None:
        """SSP cost is substantially lower than LinearPredictor cost."""
        if IGNORE_TESTS:
            return
        self.assertLess(self.ssp.cost, self.lp.cost / 5.0)

    def test_predict_correct_shape(self) -> None:
        """predict returns (101, 30) DataFrame."""
        if IGNORE_TESTS:
            return
        pred_df = self.ssp.predict()
        self.assertEqual(pred_df.shape, (101, 30))


if __name__ == "__main__":
    unittest.main()
