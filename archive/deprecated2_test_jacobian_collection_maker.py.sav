"""
Tests for JacobianCollectionMaker class.
"""

import os
import sys
import unittest

import numpy as np  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from trajectory import Trajectory  # type: ignore
from jacobian_collection_maker import JacobianCollectionMaker  # type: ignore
from roadrunner_maker import RoadRunnerMaker  # type: ignore

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


class TestJacobianCollectionMakerInit(unittest.TestCase):
    """Tests for JacobianCollectionMaker.__init__."""

    def test_roadrunner_maker_is_roadrunner_maker_instance(self) -> None:
        """roadrunner_maker attribute is a RoadRunnerMaker instance."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL)
        self.assertIsInstance(maker.roadrunner_maker, RoadRunnerMaker)

    def test_defaults_from_constants(self) -> None:
        """Default timing parameters are forwarded to roadrunner_maker."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL)
        self.assertEqual(maker.roadrunner_maker._start_time, cn.START_TIME)
        self.assertEqual(maker.roadrunner_maker._end_time, cn.END_TIME)
        self.assertEqual(maker.roadrunner_maker._num_points, cn.NUM_POINTS)

    def test_custom_params_forwarded(self) -> None:
        """Custom start_time, end_time, and num_points are forwarded to roadrunner_maker."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, start_time=1.0, end_time=5.0, num_points=20)
        self.assertEqual(maker.roadrunner_maker._start_time, 1.0)
        self.assertEqual(maker.roadrunner_maker._end_time, 5.0)
        self.assertEqual(maker.roadrunner_maker._num_points, 20)

    def test_roadrunner_kept_by_default(self) -> None:
        """RoadRunner instance is retained when is_keep_roadrunner=True (default)."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL)
        self.assertTrue(hasattr(maker.roadrunner_maker, "_roadrunner"))

    def test_roadrunner_deleted_when_not_kept(self) -> None:
        """RoadRunner instance is deleted when is_keep_roadrunner=False."""
        maker = JacobianCollectionMaker(ANTIMONY_MODEL, is_keep_roadrunner=False)
        self.assertFalse(hasattr(maker.roadrunner_maker, "_roadrunner"))

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            JacobianCollectionMaker(12345)

    def test_init_from_rr_instance(self) -> None:
        """Initialization from a RoadRunner instance succeeds."""
        rr = te.loada(ANTIMONY_MODEL)
        maker = JacobianCollectionMaker(rr)
        self.assertIsInstance(maker.roadrunner_maker, RoadRunnerMaker)


class TestJacobianCollectionMakerMakeCollection(unittest.TestCase):
    """Tests for JacobianCollectionMaker.makeCollection."""

    def setUp(self) -> None:
        self.maker = JacobianCollectionMaker(
            ANTIMONY_MODEL,
            start_time=0.0,
            end_time=5.0,
            num_points=11,
        )

    def test_returns_jacobian_collection(self) -> None:
        """makeCollection returns a JacobianCollection instance."""
        collection = self.maker.makeCollection()
        self.assertIsInstance(collection, Trajectory)

    def test_jacobian_ndim(self) -> None:
        """Returned jacobian_arr is 3-dimensional."""
        collection = self.maker.makeCollection()
        self.assertEqual(collection.jacobian_arr.ndim, 3)

    def test_jacobian_num_points(self) -> None:
        """First dimension of jacobian_arr matches num_points."""
        collection = self.maker.makeCollection()
        self.assertEqual(collection.jacobian_arr.shape[0], 11)

    def test_jacobian_square_species_axes(self) -> None:
        """Species axes (1 and 2) are equal in size."""
        collection = self.maker.makeCollection()
        self.assertEqual(collection.jacobian_arr.shape[1], collection.jacobian_arr.shape[2])

    def test_jacobian_num_species(self) -> None:
        """Species axis size matches the number of floating species (S1, S2)."""
        collection = self.maker.makeCollection()
        self.assertEqual(collection.jacobian_arr.shape[1], 2)

    def test_timepoints_length(self) -> None:
        """Timepoints array length matches num_points."""
        collection = self.maker.makeCollection()
        self.assertEqual(len(collection.timepoint_arr), 11)

    def test_timepoints_start(self) -> None:
        """First timepoint matches start_time."""
        collection = self.maker.makeCollection()
        self.assertAlmostEqual(collection.timepoint_arr[0], 0.0, places=5)

    def test_timepoints_end(self) -> None:
        """Last timepoint matches end_time."""
        collection = self.maker.makeCollection()
        self.assertAlmostEqual(collection.timepoint_arr[-1], 5.0, places=5)

    def test_jacobian_values_are_finite(self) -> None:
        """All Jacobian entries are finite (no NaN or Inf)."""
        collection = self.maker.makeCollection()
        self.assertTrue(np.all(np.isfinite(collection.jacobian_arr)))

    def test_make_collection_from_rr_instance(self) -> None:
        """makeCollection works when initialized with a RoadRunner instance."""
        rr = te.loada(ANTIMONY_MODEL)
        maker = JacobianCollectionMaker(rr, start_time=0.0, end_time=5.0, num_points=11)
        collection = maker.makeCollection()
        self.assertIsInstance(collection, Trajectory)
        self.assertEqual(collection.jacobian_arr.shape[0], 11)


if __name__ == "__main__":
    unittest.main()
