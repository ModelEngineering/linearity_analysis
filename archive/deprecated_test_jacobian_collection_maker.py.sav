"""
Tests for the Jacobian class.
"""
from jacobian_collection_maker import JacobianCollectionMaker  # type: ignore
from trajectory import Trajectory  # type: ignore

import os
import sys
import unittest

import numpy as np  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


def _make_rr():
    return te.loada(ANTIMONY_MODEL)


class TestJacobianInit(unittest.TestCase):
    """Tests for Jacobian.__init__."""

    def test_stores_array(self) -> None:
        """Constructor stores the provided array."""
        arr = np.zeros((5, 2, 2))
        jac = JacobianCollectionMaker(arr)
        np.testing.assert_array_equal(jac.jacobian_arr, arr)


class TestJacobianCollect(unittest.TestCase):
    """Tests for Jacobian.collect."""

    def test_returns_jacobian_instance(self) -> None:
        """collect returns a Jacobian instance."""
        rr = _make_rr()
        jac = JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=5)
        self.assertIsInstance(jac, JacobianCollectionMaker)

    def test_array_shape(self) -> None:
        """Collected array has shape (num_point, n_species, n_species)."""
        rr = _make_rr()
        n_species = len(rr.getFloatingSpeciesIds())
        num_point = 8
        jac = JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=num_point)
        self.assertEqual(jac.jacobian_arr.shape, (num_point, n_species, n_species))

    def test_array_contains_finite_values(self) -> None:
        """Collected array contains at least some finite values."""
        rr = _make_rr()
        jac = JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=5)
        self.assertTrue(np.any(np.isfinite(jac.jacobian_arr)))

    def test_raises_on_no_floating_species(self) -> None:
        """ValueError is raised when the model has no floating species."""
        rr = te.loada("k1 = 0.1")
        with self.assertRaises((ValueError, Exception)):
            JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=5)


class TestJacobianMaxCV(unittest.TestCase):
    """Tests for Jacobian.maxCV."""

    def test_returns_float(self) -> None:
        """maxCV returns a float."""
        rr = _make_rr()
        jac = JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=10)
        self.assertIsInstance(jac.maxCV(), float)

    def test_non_negative(self) -> None:
        """maxCV is non-negative."""
        rr = _make_rr()
        jac = JacobianCollectionMaker.collect(rr, start=0, end=10, num_point=10)
        self.assertGreaterEqual(jac.maxCV(), 0.0)

    def test_single_timepoint_returns_zero(self) -> None:
        """A single-timepoint array has no variation, so maxCV is 0."""
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
        jac = JacobianCollectionMaker(arr)
        self.assertEqual(jac.maxCV(), 0.0)

    def test_constant_entries_return_zero(self) -> None:
        """Identical Jacobians across timepoints yield maxCV == 0."""
        single = np.array([[1.0, -2.0], [0.5, 3.0]])
        arr = np.stack([single] * 5)  # shape (5, 2, 2)
        jac = JacobianCollectionMaker(arr)
        self.assertAlmostEqual(jac.maxCV(), 0.0)

    def test_zero_mean_entries_do_not_cause_division_error(self) -> None:
        """Entries with zero mean are handled without NaN or exception."""
        arr = np.zeros((4, 2, 2))
        jac = JacobianCollectionMaker(arr)
        result = jac.maxCV()
        self.assertFalse(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
