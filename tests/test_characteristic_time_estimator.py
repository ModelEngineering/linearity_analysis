"""Tests for CharacteristicTimeEstimator."""
import os
import unittest

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from unittest.mock import patch, MagicMock

import src.constants as cn  # type: ignore
from biomodels_iterator import getBiomodelsEndtimes  # type: ignore
from model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from simulator import Simulator, SimulationResult  # type: ignore
from characteristic_time_estimator import (  # type: ignore
    CharacteristicTimeEstimator,
    CharacteristicTimeResult,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> S3; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0; S3 = 0
"""

MODEL_NAME = "test_model"

HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL, model_name=MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    """Tests for CharacteristicTimeEstimator.__init__."""

    def test_stores_model(self) -> None:
        estimator = CharacteristicTimeEstimator(model=_makeModel())
        self.assertIsInstance(estimator.model, Model)

    def test_default_start_time(self) -> None:
        estimator = CharacteristicTimeEstimator(model=_makeModel())
        self.assertEqual(estimator.start_time, cn.START_TIME)

    def test_custom_start_time(self) -> None:
        estimator = CharacteristicTimeEstimator(
            model=_makeModel(), start_time=5.0)
        self.assertEqual(estimator.start_time, 5.0)

    def test_default_num_point(self) -> None:
        estimator = CharacteristicTimeEstimator(model=_makeModel())
        self.assertEqual(estimator.num_point, cn.NUM_POINT)

    def test_custom_num_point(self) -> None:
        estimator = CharacteristicTimeEstimator(
            model=_makeModel(), num_point=500)
        self.assertEqual(estimator.num_point, 500)


# ---------------------------------------------------------------------------
# Tests for estimate classmethod
# ---------------------------------------------------------------------------

class TestEstimate(unittest.TestCase):
    """Tests for CharacteristicTimeEstimator.estimate."""

    def test_user_specified_end_time_returns_immediately(self) -> None:
        model = _makeModel()
        end_time, source = CharacteristicTimeEstimator.estimate(
            model, end_time=42.0)
        self.assertEqual(end_time, 42.0)
        self.assertEqual(source, cn.ENDTIME_SOURCE_USER_SPECIFIED)

    def test_user_specified_end_time_float(self) -> None:
        model = _makeModel()
        end_time, source = CharacteristicTimeEstimator.estimate(
            model, end_time=100.5)
        self.assertEqual(end_time, 100.5)
        self.assertEqual(source, cn.ENDTIME_SOURCE_USER_SPECIFIED)

    def test_user_specified_end_time_int(self) -> None:
        model = _makeModel()
        end_time, source = CharacteristicTimeEstimator.estimate(
            model, end_time=200)
        self.assertEqual(end_time, 200.0)
        self.assertEqual(source, cn.ENDTIME_SOURCE_USER_SPECIFIED)

    def test_none_end_time_triggers_detection(self) -> None:
        """When end_time is None, detection methods are invoked."""
        model = _makeModel()
        with patch.object(CharacteristicTimeEstimator, 'detect_steadystate', return_value=15.0):
            end_time, source = CharacteristicTimeEstimator.estimate(model)
        self.assertEqual(end_time, 15.0)
        self.assertEqual(source, cn.ENDTIME_SOURCE_STEADYSTATE)

    def test_no_detection_raises(self) -> None:
        """If all detection methods return None, ValueError is raised."""
        model = _makeModel()
        with patch.object(CharacteristicTimeEstimator, 'detect_steadystate', return_value=None), \
             patch.object(CharacteristicTimeEstimator, 'detect_cv_maximized', return_value=None):
            with self.assertRaises(ValueError) as ctx:
                CharacteristicTimeEstimator.estimate(model)
        self.assertIn("Could not determine", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for detect_steadystate (unit tests without real simulation)
# ---------------------------------------------------------------------------

class TestDetectSteadyState(unittest.TestCase):
    """Tests for steady-state detection logic."""

    def test_returns_none_when_no_floating_species(self) -> None:
        """Models with no floating species should return None."""
        antimony = "a := 1; b := 2;"
        model = Model(antimony, model_name="no_float")
        estimator = CharacteristicTimeEstimator(model=model)

        # Real Simulator.getSteadyState() returns None for empty concentrations.
        result = estimator.detect_steadystate()
        self.assertIsNone(result)

    def test_returns_none_on_runtime_error(self) -> None:
        """If Simulator.getSteadyState() fails to find steady state, return None."""
        model = _makeModel()
        estimator = CharacteristicTimeEstimator(model=model)

        mock_sim_instance = MagicMock()
        mock_sim_instance.getSteadyState.return_value = None
        with patch('characteristic_time_estimator.Simulator', return_value=mock_sim_instance):
            result = estimator.detect_steadystate()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests for detect_cv_maximized (unit tests without real simulation)
# ---------------------------------------------------------------------------

class TestDetectCVMaximized(unittest.TestCase):
    """Tests for CV maximization detection logic."""

    def test_returns_none_when_no_floating_species(self) -> None:
        """Models with no floating species should return None."""
        antimony = "a := 1; b := 2;"
        model = Model(antimony, model_name="no_float")
        estimator = CharacteristicTimeEstimator(model=model)

        result = estimator.detect_cv_maximized()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests for _run_simulation helper
# ---------------------------------------------------------------------------

class TestRunSimulation(unittest.TestCase):
    """Tests for the internal _run_simulation method."""

    def test_returns_simulation_result(self) -> None:
        model = _makeModel()
        estimator = CharacteristicTimeEstimator(model=model, num_point=10)

        mock_sim_instance = MagicMock()
        mock_result = SimulationResult(
            timecourse_df=pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=[0.0]),
            jacobian_collection_arr=np.array([]),
        )
        mock_sim_instance.simulate.return_value = mock_result

        with patch('characteristic_time_estimator.Simulator', return_value=mock_sim_instance):
            result = estimator._run_simulation(end_time=5.0)
        self.assertIs(result, mock_result)

    def test_creates_simulator_with_correct_params(self) -> None:
        model = _makeModel()
        estimator = CharacteristicTimeEstimator(
            model=model, start_time=1.0, num_point=20)

        mock_sim_instance = MagicMock()
        mock_result = SimulationResult(
            timecourse_df=pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=[0.0]),
            jacobian_collection_arr=np.array([]),
        )
        mock_sim_instance.simulate.return_value = mock_result

        with patch('characteristic_time_estimator.Simulator', return_value=mock_sim_instance) as MockSim:
            estimator._run_simulation(end_time=15.0)

        MockSim.assert_called_once_with(
            model=model,
            start_time=1.0,
            end_time=15.0,
            num_point=20,
        )


# ---------------------------------------------------------------------------
# Tests for constants / type aliases
# ---------------------------------------------------------------------------

class TestTypeAlias(unittest.TestCase):
    """Tests that the type alias is defined correctly."""

    def test_characteristic_time_result_is_tuple(self) -> None:
        result: CharacteristicTimeResult = (10.0, "test_source")
        self.assertEqual(result[0], 10.0)
        self.assertEqual(result[1], "test_source")


# ---------------------------------------------------------------------------
# Integration-style test with a simple model that reaches steady state quickly
# ---------------------------------------------------------------------------

class TestIntegrationSimpleDecay(unittest.TestCase):
    """Integration test: first-order decay should reach steady state."""

    DECAY_ANTIMONY = """
    S1 -> X0; k*S1
    k = 1.0
    S1 = 10
    """

    def test_detect_steadystate_for_decay(self) -> None:
        """S1 -> X0 should converge to S1=0 at steady state."""
        model = Model(self.DECAY_ANTIMONY, model_name="decay_test")
        estimator = CharacteristicTimeEstimator(model=model, num_point=50)

        end_time = estimator.detect_steadystate()
        # Should find some finite end time (not None) since decay reaches steady state quickly
        self.assertIsNotNone(end_time)
        self.assertGreater(end_time, 0)


# ---------------------------------------------------------------------------
# Tests for plotStdNrml
# ---------------------------------------------------------------------------

class TestPlotStdNrml(unittest.TestCase):
    """Tests for CharacteristicTimeEstimator.plotStdNrml (three-panel plot)."""

    def setUp(self) -> None:
        self.model = _makeModel()
        self.estimator = CharacteristicTimeEstimator(model=self.model, num_point=20)
        self.timecourse_df = pd.DataFrame(
            {"S1": [10.0, 5.0, 0.0], "S2": [0.0, 3.0, 4.0], "S3": [0.0, 2.0, 6.0]},
            index=[0.0, 1.0, 2.0],
        )
        self.mock_result = SimulationResult(
            timecourse_df=self.timecourse_df, jacobian_collection_arr=np.array([]))

    def tearDown(self) -> None:
        plt.close("all")

    def _plot(self, **kwargs) -> list:
        with patch.object(self.estimator, '_run_simulation', return_value=self.mock_result):
            return self.estimator.plotStdNrml(end_time=2.0, **kwargs)

    def test_returns_three_plot_options(self) -> None:
        result = self._plot()
        self.assertEqual(len(result), 3)
        for po in result:
            self.assertIsInstance(po, PlotOptions)

    def test_panels_share_one_figure_with_distinct_axes(self) -> None:
        top_po, mid_po, bot_po = self._plot()
        self.assertIs(top_po.fig, mid_po.fig)
        self.assertIs(mid_po.fig, bot_po.fig)
        self.assertNotEqual(id(top_po.ax), id(mid_po.ax))
        self.assertNotEqual(id(mid_po.ax), id(bot_po.ax))

    def test_panel_ylabels(self) -> None:
        top_po, mid_po, bot_po = self._plot()
        self.assertEqual(top_po.ylabel, "Concentration")
        self.assertEqual(mid_po.ylabel, "Normalized Value")
        self.assertEqual(bot_po.ylabel, "Standard Deviation of Normalized Values")

    def test_panel_titles(self) -> None:
        top_po, mid_po, bot_po = self._plot()
        self.assertEqual(top_po.ax.get_title(), "Timecourse")
        self.assertEqual(mid_po.ax.get_title(), "Standardized Timecourse")

    def test_top_and_mid_legends_list_species(self) -> None:
        top_po, mid_po, _ = self._plot()
        for po in (top_po, mid_po):
            legend = po.ax.get_legend()
            self.assertIsNotNone(legend)
            labels = {t.get_text() for t in legend.get_texts()}
            self.assertEqual(labels, {"S1", "S2", "S3"})

    def test_suptitle_uses_title_and_model_name(self) -> None:
        top_po, _, _ = self._plot(title="My Title", model_name="MyModel")
        self.assertEqual(top_po.fig.get_suptitle(), "MyModel: My Title")  # type: ignore

    def test_no_suptitle_when_title_omitted(self) -> None:
        top_po, _, _ = self._plot()
        self.assertEqual(top_po.fig.get_suptitle(), "")  # type: ignore

    def test_calls_run_simulation_with_end_time(self) -> None:
        with patch.object(self.estimator, '_run_simulation',
                return_value=self.mock_result) as mock_run:
            self.estimator.plotStdNrml(end_time=7.5)
        mock_run.assert_called_once_with(end_time=7.5)

    def test_top_panel_matches_raw_values(self) -> None:
        top_po, _, _ = self._plot()
        for idx, name in enumerate(["S1", "S2", "S3"]):
            line = top_po.ax.lines[idx]
            np.testing.assert_allclose(line.get_ydata(), self.timecourse_df[name].values)  # type: ignore

    def test_mid_panel_matches_normalized_values(self) -> None:
        _, mid_po, _ = self._plot()
        normalized_df = (self.timecourse_df - self.timecourse_df.mean()) / self.timecourse_df.std()
        for idx, name in enumerate(["S1", "S2", "S3"]):
            line = mid_po.ax.lines[idx]
            np.testing.assert_allclose(line.get_ydata(), normalized_df[name].values)  # type: ignore

    def test_bottom_panel_matches_computed_metric(self) -> None:
        """The bottom line is the cross-species std of z-score normalized columns."""
        _, _, bot_po = self._plot()
        normalized_df = (self.timecourse_df - self.timecourse_df.mean()) / self.timecourse_df.std()
        expected_metric = normalized_df.std(axis=1).values
        line = bot_po.ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), expected_metric)  # type: ignore
        np.testing.assert_allclose(line.get_xdata(), self.timecourse_df.index.values)

    def test_handles_zero_variance_column(self) -> None:
        """A constant species column would divide by zero std; NaNs must not leak through."""
        timecourse_df = pd.DataFrame(
            {"S1": [10.0, 5.0, 0.0], "S2": [1.0, 1.0, 1.0]},
            index=[0.0, 1.0, 2.0],
        )
        mock_result = SimulationResult(
            timecourse_df=timecourse_df, jacobian_collection_arr=np.array([]))
        with patch.object(self.estimator, '_run_simulation', return_value=mock_result):
            _, mid_po, bot_po = self.estimator.plotStdNrml(end_time=2.0)
        for line in mid_po.ax.lines:
            self.assertFalse(np.any(np.isnan(line.get_ydata())))
        self.assertFalse(np.any(np.isnan(bot_po.ax.lines[0].get_ydata())))


# ---------------------------------------------------------------------------
# Integration test for plotStdNrml against a real BioModel
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestPlotStdNrmlBiomodel45(unittest.TestCase):
    """Integration tests exercising plotStdNrml against a real BioModel
    (BIOMD0000000045), simulated live via Simulator (no mocking)."""

    MODEL_NUM = 1
    end_time: float
    model: Model

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.makeBiomodel(model_num=cls.MODEL_NUM)
        end_time_dct = getBiomodelsEndtimes()
        cls.end_time = end_time_dct[cls.model.model_name]

    def setUp(self) -> None:
        self.estimator = CharacteristicTimeEstimator(model=self.model,
                num_point=1000)

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_three_plot_options(self) -> None:
        result = self.estimator.plotStdNrml(end_time=self.end_time)
        self.assertEqual(len(result), 3)
        for po in result:
            self.assertIsInstance(po, PlotOptions)

    def test_line_count_matches_species_count(self) -> None:
        top_po, mid_po, bot_po = self.estimator.plotStdNrml(end_time=self.end_time)
        self.assertEqual(len(top_po.ax.lines), self.model.num_species)
        self.assertEqual(len(mid_po.ax.lines), self.model.num_species)
        self.assertEqual(len(bot_po.ax.lines), 1)

    def test_metric_is_finite(self) -> None:
        _, _, bot_po = self.estimator.plotStdNrml(end_time=self.end_time)
        metric = bot_po.ax.lines[0].get_ydata()
        self.assertTrue(np.all(np.isfinite(metric)))

    def test_panel_titles(self) -> None:
        top_po, mid_po, bot_po = self.estimator.plotStdNrml(end_time=self.end_time)
        self.assertEqual(top_po.ax.get_title(), "Timecourse")
        self.assertEqual(mid_po.ax.get_title(), "Standardized Timecourse")


if __name__ == "__main__":
    unittest.main()