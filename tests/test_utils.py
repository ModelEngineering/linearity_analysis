"""Tests for utility functions in src/utils.py."""

import os
import sys
import unittest
import numpy as np  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import calculateGershgorinCircles, adjustJacobian  # type: ignore

IGNORE_TESTS = False


class TestCalculateGershgorinCircles(unittest.TestCase):
    """Tests for calculateGershgorinCircles."""

    def test_shape(self) -> None:
        """Returns array of shape (n, 2) for an n×n input."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.eye(3)
        result = calculateGershgorinCircles(jacobian_arr)
        self.assertEqual(result.shape, (3, 2))

    def test_centers_are_diagonal(self) -> None:
        """First column (centers) equals the diagonal of the Jacobian."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]])
        result = calculateGershgorinCircles(jacobian_arr)
        np.testing.assert_array_equal(result[:, 0], np.diag(jacobian_arr))

    def test_radii_are_nonnegative(self) -> None:
        """Second column (radii) is always non-negative."""
        if IGNORE_TESTS:
            return
        rng = np.random.default_rng(0)
        jacobian_arr = rng.standard_normal((4, 4))
        result = calculateGershgorinCircles(jacobian_arr)
        self.assertTrue(np.all(result[:, 1] >= 0.0))

    def test_diagonal_matrix_has_zero_radii(self) -> None:
        """Diagonal matrix has zero radii because all off-diagonal entries are zero."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.diag([1.0, -2.0, 3.0])
        result = calculateGershgorinCircles(jacobian_arr)
        np.testing.assert_array_equal(result[:, 1], np.zeros(3))

    def test_known_2x2_values(self) -> None:
        """Hand-computed Gershgorin circles for a 2×2 matrix."""
        if IGNORE_TESTS:
            return
        # Row 0: center=1, radius=|3|=3; Row 1: center=4, radius=|2|=2
        jacobian_arr = np.array([[1.0, 3.0], [2.0, 4.0]])
        result = calculateGershgorinCircles(jacobian_arr)
        np.testing.assert_array_equal(result[:, 0], [1.0, 4.0])
        np.testing.assert_array_equal(result[:, 1], [3.0, 2.0])

    def test_known_3x3_values(self) -> None:
        """Hand-computed Gershgorin circles for a 3×3 matrix."""
        if IGNORE_TESTS:
            return
        # Row 0: center=5, radius=|1|+|2|=3
        # Row 1: center=6, radius=|3|+|4|=7
        # Row 2: center=7, radius=|0|+|8|=8
        jacobian_arr = np.array([
                [5.0, 1.0, 2.0],
                [3.0, 6.0, 4.0],
                [0.0, 8.0, 7.0]])
        result = calculateGershgorinCircles(jacobian_arr)
        np.testing.assert_array_equal(result[:, 0], [5.0, 6.0, 7.0])
        np.testing.assert_array_equal(result[:, 1], [3.0, 7.0, 8.0])


class TestAdjustJacobian(unittest.TestCase):
    """Tests for adjustJacobian."""

    def test_returns_copy(self) -> None:
        """Original Jacobian is not modified."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[100.0, 0.0], [0.0, 100.0]])
        original = jacobian_arr.copy()
        adjustJacobian(jacobian_arr, time=1.0)
        np.testing.assert_array_equal(jacobian_arr, original)

    def test_shape_preserved(self) -> None:
        """Output shape equals input shape."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.eye(3) * 5.0
        result = adjustJacobian(jacobian_arr, time=1.0)
        self.assertEqual(result.shape, jacobian_arr.shape)

    def test_off_diagonals_unchanged(self) -> None:
        """Off-diagonal entries are not modified."""
        if IGNORE_TESTS:
            return
        jacobian_arr = np.array([[100.0, 3.0, -2.0],
                [1.0, 50.0, 4.0],
                [-5.0, 2.0, 80.0]])
        result = adjustJacobian(jacobian_arr, time=1.0)
        n = jacobian_arr.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertAlmostEqual(result[i, j], jacobian_arr[i, j], places=12)

    def test_eigenvalues_within_max_value_diagonal(self) -> None:
        """Diagonal Jacobian: exp(max_real_eigval * time) <= max_value."""
        if IGNORE_TESTS:
            return
        time = 1.0
        max_value = 1e10
        jacobian_arr = np.diag([100.0, 200.0, 50.0])
        result = adjustJacobian(jacobian_arr, time=time, max_value=max_value)
        max_real_eigval = float(np.max(np.real(np.linalg.eigvals(result))))
        self.assertLessEqual(np.exp(max_real_eigval * time), max_value)

    def test_eigenvalues_within_max_value_nondiagonal(self) -> None:
        """Non-diagonal Jacobian: exp(max_real_eigval * time) <= max_value."""
        if IGNORE_TESTS:
            return
        time = 2.0
        max_value = 1e6
        jacobian_arr = np.array([[50.0, 1.0], [1.0, 80.0]])
        result = adjustJacobian(jacobian_arr, time=time, max_value=max_value)
        max_real_eigval = float(np.max(np.real(np.linalg.eigvals(result))))
        self.assertLessEqual(np.exp(max_real_eigval * time), max_value)

    def test_eigenvalues_within_max_value_varying_time(self) -> None:
        """Eigenvalue bound holds across a range of time values."""
        if IGNORE_TESTS:
            return
        max_value = 1e4
        jacobian_arr = np.diag([30.0, 30.0])
        for time in [0.5, 1.0, 5.0, 10.0]:
            result = adjustJacobian(jacobian_arr, time=time, max_value=max_value)
            max_real_eigval = float(np.max(np.real(np.linalg.eigvals(result))))
            self.assertLessEqual(
                    np.exp(max_real_eigval * time),
                    max_value,
                    msg=f"Eigenvalue bound violated for time={time}")

    def test_eigenvalues_default_max_value(self) -> None:
        """Default max_value (1e10): exp(max_real_eigval * time) <= 1e10."""
        if IGNORE_TESTS:
            return
        time = 1.0
        jacobian_arr = np.diag([500.0, 1000.0])
        result = adjustJacobian(jacobian_arr, time=time)
        max_real_eigval = float(np.max(np.real(np.linalg.eigvals(result))))
        self.assertLessEqual(np.exp(max_real_eigval * time), 1e10)

    def test_diagonal_below_threshold_unchanged(self) -> None:
        """Diagonal already below r_i - max_circle_edge is not modified."""
        if IGNORE_TESTS:
            return
        time = 1.0
        max_value = 1e10
        # max_circle_edge = log(1e10) ≈ 23.03; r_i = 0 → threshold = -23.03.
        # diagonal = -100 < -23.03, so it must not be clamped.
        jacobian_arr = np.diag([-100.0, -100.0])
        result = adjustJacobian(jacobian_arr, time=time, max_value=max_value)
        np.testing.assert_array_almost_equal(np.diag(result), np.diag(jacobian_arr))

    def test_large_positive_diagonal_gets_clamped(self) -> None:
        """Large positive diagonal is clamped and the result obeys the eigenvalue bound."""
        if IGNORE_TESTS:
            return
        time = 2.0
        max_value = 1e10
        jacobian_arr = np.array([[1e6, 0.0], [0.0, 1e6]])
        result = adjustJacobian(jacobian_arr, time=time, max_value=max_value)
        max_real_eigval = float(np.max(np.real(np.linalg.eigvals(result))))
        self.assertLessEqual(np.exp(max_real_eigval * time), max_value)
        self.assertFalse(np.allclose(np.diag(result), np.diag(jacobian_arr)))


if __name__ == "__main__":
    unittest.main()
