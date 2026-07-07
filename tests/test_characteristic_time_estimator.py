"""Tests for CharacteristicTimeEstimator."""
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from unittest.mock import patch, MagicMock

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
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

        # Mock the RR object to return empty floating species
        with patch('characteristic_time_estimator.te') as mock_te:
            mock_rr = MagicMock()
            mock_rr.getFloatingSpeciesIds.return_value = []
            mock_te.loadSBMLModel.return_value = mock_rr
            result = estimator.detect_steadystate()
        self.assertIsNone(result)

    def test_returns_none_on_runtime_error(self) -> None:
        """If steady-state solver raises RuntimeError, return None."""
        model = _makeModel()
        estimator = CharacteristicTimeEstimator(model=model)

        with patch('characteristic_time_estimator.te') as mock_te:
            mock_rr = MagicMock()
            mock_rr.getSteadyStateSolver.return_value.setValue.side_effect = RuntimeError("no ss")
            mock_te.loadSBMLModel.return_value = mock_rr
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

        with patch('characteristic_time_estimator.te') as mock_te:
            mock_rr = MagicMock()
            mock_rr.getFloatingSpeciesIds.return_value = []
            mock_te.loadSBMLModel.return_value = mock_rr
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


if __name__ == "__main__":
    unittest.main()