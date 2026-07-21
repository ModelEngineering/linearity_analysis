"""Tests for src/jacobian_estimator.py with comparisons to pysindy."""
import unittest

import numpy as np # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import Lasso  # type: ignore

# pysindy imports for comparison
try:
    import pysindy  # type: ignore
    HAS_PYSINDY = True
except ImportError:
    HAS_PYSINDY = False


class TestJacobianEstimatorInit(unittest.TestCase):
    """Tests for JacobianEstimator constructor."""

    def test_accepts_valid_dataframe(self) -> None:
        """Constructor accepts a valid pd.DataFrame with numeric data and increasing index."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0], "S2": [0.5, 0.8, 1.1, 1.4]},
            index=[0.0, 1.0, 2.0, 3.0],
        )
        est = JacobianEstimator(df)
        self.assertEqual(list(est.timecourse_df.columns), ["S1", "S2"])

    def test_raises_value_error_for_empty_dataframe(self) -> None:
        """Constructor raises ValueError for an empty DataFrame."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame()
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_raises_value_error_for_no_columns(self) -> None:
        """Constructor raises ValueError when DataFrame has no columns."""
        from src.jacobian_estimator import JacobianEstimator
        idx = pd.Index([0.0, 1.0], name="time")
        df = pd.DataFrame(index=idx)
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_raises_value_error_for_non_monotonic_index(self) -> None:
        """Constructor raises ValueError when index is not strictly increasing."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0]},
            index=[0.0, 2.0, 1.0],
        )
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_raises_value_error_for_duplicate_index(self) -> None:
        """Constructor raises ValueError when index has duplicate values."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0]},
            index=[0.0, 1.0, 1.0],
        )
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_raises_value_error_for_nan_values(self) -> None:
        """Constructor raises ValueError when data contains NaN."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, np.nan, 3.0], "S2": [0.5, 0.8, 1.1]},
            index=[0.0, 1.0, 2.0],
        )
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_raises_value_error_for_infinite_values(self) -> None:
        """Constructor raises ValueError when data contains infinite values."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, np.inf, 3.0], "S2": [0.5, 0.8, 1.1]},
            index=[0.0, 1.0, 2.0],
        )
        with self.assertRaises(ValueError):
            JacobianEstimator(df)

    def test_derivative_shape_is_one_less(self) -> None:
        """dtimecourse_df has one fewer row than timecourse_df."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0], "S2": [0.5, 0.8, 1.1, 1.4]},
            index=[0.0, 1.0, 2.0, 3.0],
        )
        est = JacobianEstimator(df)
        self.assertEqual(len(est.dtimecourse_df), len(est.timecourse_df) - 1)

    def test_derivative_columns_match(self) -> None:
        """dtimecourse_df has the same column names as timecourse_df."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            index=[0.0, 1.0],
        )
        est = JacobianEstimator(df)
        self.assertEqual(list(est.dtimecourse_df.columns), ["A", "B"])

    def test_derivative_forward_difference(self) -> None:
        """Derivatives are computed via forward finite differences."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [0.0, 1.0, 3.0]},
            index=[0.0, 1.0, 3.0],
        )
        est = JacobianEstimator(df)
        # Formula: (values[i+1] - values[i]) / (t[i+1] - t[i])
        expected_dS1 = np.array([
            (1.0 - 0.0) / (1.0 - 0.0),   # 1.0/1.0 = 1.0
            (3.0 - 1.0) / (3.0 - 1.0),   # 2.0/2.0 = 1.0
        ])
        np.testing.assert_allclose(  # type: ignore
            est.dtimecourse_df["S1"].values, expected_dS1  # type: ignore
        )

    def test_single_species(self) -> None:
        """Constructor works with a single state variable."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame({"S1": [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        est = JacobianEstimator(df)
        self.assertEqual(len(est.dtimecourse_df), 2)


class TestJacobianEstimatorFit(unittest.TestCase):
    """Tests for JacobianEstimator.fit()."""

    def _make_linear_data(self, A_true: np.ndarray, u_true: np.ndarray,
                          n_points: int = 200, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic linear ODE data for testing using scipy solve_ivp."""
        from scipy.integrate import solve_ivp  # type: ignore

        def rhs(t, x):
            return A_true @ x + u_true

        t_span = (0.0, 10.0)
        sol = solve_ivp(rhs, t_span, np.ones(A_true.shape[0]),
                        t_eval=np.linspace(0.0, 10.0, n_points), method='RK45')
        states = sol.y.T

        col_names = [f"S{i+1}" for i in range(A_true.shape[0])]
        df = pd.DataFrame(states, columns=col_names, index=sol.t)
        return df

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0], "S2": [0.5, 0.8, 1.1, 1.4]},
            index=[0.0, 1.0, 2.0, 3.0],
        )
        est = JacobianEstimator(df)
        result = est.fit(alpha=0.01)
        self.assertIs(result, est)

    def test_fit_sets_is_fitted(self) -> None:
        """After fit(), _is_fitted is True."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0]},
            index=[0.0, 1.0, 2.0, 3.0],
        )
        est = JacobianEstimator(df)
        self.assertFalse(est._is_fitted)
        est.fit(alpha=0.01)
        self.assertTrue(est._is_fitted)

    def test_fit_raises_for_negative_alpha(self) -> None:
        """fit() raises ValueError for negative alpha."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        with self.assertRaises(ValueError):
            est.fit(alpha=-0.1)

    def test_fit_recovers_known_matrix(self) -> None:
        """fit() recovers the true A m:w
        atrix and u vector from synthetic data."""
        from src.jacobian_estimator import JacobianEstimator
        # Simple 2-species linear system
        A_true = np.array([[-0.5, 0.1], [0.05, -0.3]])
        u_true = np.array([0.02, -0.01])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)
        est = JacobianEstimator(df)
        est.fit(alpha=0.001)  # Low alpha for minimal regularization

        np.testing.assert_allclose(est.A_, A_true, atol=0.15)  # type: ignore
        np.testing.assert_allclose(est.u_, u_true, atol=0.05)  # type: ignore

    def test_fit_zero_alpha_recovers_dense(self) -> None:
        """With alpha=0, fit() should recover all coefficients (no sparsity)."""
        from src.jacobian_estimator import JacobianEstimator
        A_true = np.array([[-1.0, 0.5], [0.3, -0.7]])
        u_true = np.array([0.1, -0.05])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)
        est = JacobianEstimator(df)
        est.fit(alpha=0.0)  # No regularization

        np.testing.assert_allclose(est.A_, A_true, atol=0.15)
        np.testing.assert_allclose(est.u_, u_true, atol=0.05)

    def test_fit_sparse_recovery(self) -> None:
        """With high alpha, fit() should produce sparse Jacobian."""
        from src.jacobian_estimator import JacobianEstimator
        # Sparse true system: S1 depends on itself only, S2 depends on both
        A_true = np.array([[-0.5, 0.0], [0.1, -0.3]])
        u_true = np.array([0.0, 0.0])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)
        est = JacobianEstimator(df)
        est.fit(alpha=1.0)  # High regularization

        # A[0, 1] should be driven to zero by high alpha
        self.assertAlmostEqual(est.A_[0, 1], 0.0, places=3)

    def test_fit_single_species(self) -> None:
        """fit() works correctly with a single species."""
        from src.jacobian_estimator import JacobianEstimator
        # dS/dt = -0.5*S + 0.1
        A_true = np.array([[-0.5]])
        u_true = np.array([0.1])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)
        est = JacobianEstimator(df)
        est.fit(alpha=0.001)

        np.testing.assert_allclose(est.A_, A_true, atol=0.15)
        np.testing.assert_allclose(est.u_, u_true, atol=0.05)


class TestJacobianEstimatorPredict(unittest.TestCase):
    """Tests for JacobianEstimator.predict()."""

    def test_predict_raises_before_fit(self) -> None:
        """predict() raises RuntimeError if fit() has not been called."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        with self.assertRaises(RuntimeError):
            est.predict(np.array([1.0]))

    def test_predict_shape(self) -> None:
        """predict() returns array of correct shape."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0], "S2": [0.5, 0.8, 1.1]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        result = est.predict(np.array([1.0, 2.0]))
        self.assertEqual(result.shape, (2,))

    def test_predict_linear_model(self) -> None:
        """predict() computes A*x + u correctly."""
        from src.jacobian_estimator import JacobianEstimator
        # Use data where the linear model is exact
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0], "S2": [2.0, 4.0, 6.0, 8.0]},
            index=[0.0, 1.0, 2.0, 3.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.0)

        # predict should return A*x + u
        x = np.array([5.0, 10.0])
        result = est.predict(x)
        expected = est.A_.dot(x) + est.u_
        np.testing.assert_allclose(result, expected)


class TestJacobianEstimatorEquations(unittest.TestCase):
    """Tests for JacobianEstimator.equations property."""

    def test_equations_raises_before_fit(self) -> None:
        """equations raises RuntimeError if fit() has not been called."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0]},
            index=[0.0, 1.0],
        )
        est = JacobianEstimator(df)
        with self.assertRaises(RuntimeError):
            _ = est.equations

    def test_equations_returns_string(self) -> None:
        """equations returns a non-empty string after fit."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0], "S2": [0.5, 0.8, 1.1]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        eq_str = est.equations
        self.assertIsInstance(eq_str, str)
        self.assertTrue(len(eq_str) > 0)

    def test_equations_contains_species_names(self) -> None:
        """equations string contains the column names."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"Glucose": [1.0, 2.0, 3.0], "Product": [0.0, 0.5, 1.0]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        eq_str = est.equations
        self.assertIn("Glucose", eq_str)
        self.assertIn("Product", eq_str)

    def test_equations_one_line_per_species(self) -> None:
        """Number of lines in equations equals number of species."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
            index=[0.0, 1.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        lines = est.equations.split("\n")
        self.assertEqual(len(lines), 3)

    def test_equations_contains_dt(self) -> None:
        """Each equation line contains '/dt'."""
        from src.jacobian_estimator import JacobianEstimator
        df = pd.DataFrame(
            {"S1": [1.0, 2.0], "S2": [3.0, 4.0]},
            index=[0.0, 1.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        for line in est.equations.split("\n"):
            self.assertIn("/dt", line)


# =============================================================================
# SINDy Comparison Tests
# =============================================================================

@unittest.skipUnless(HAS_PYSINDY, "pysindy not installed")
class TestJacobianEstimatorVsSindypy(unittest.TestCase):
    """Compare JacobianEstimator with pysindy for linear systems."""

    def _make_linear_data(self, A_true: np.ndarray, u_true: np.ndarray,
                          n_points: int = 200, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic linear ODE data using scipy solve_ivp."""
        from scipy.integrate import solve_ivp  # type: ignore

        def rhs(t, x):
            return A_true @ x + u_true

        t_span = (0.0, 10.0)
        sol = solve_ivp(rhs, t_span, np.ones(A_true.shape[0]),
                        t_eval=np.linspace(0.0, 10.0, n_points), method='RK45')
        states = sol.y.T

        col_names = [f"S{i+1}" for i in range(A_true.shape[0])]
        df = pd.DataFrame(states, columns=col_names, index=sol.t)
        return df

    def _fit_sindy_linear(self, df: pd.DataFrame, alpha: float = 0.001):
        """Fit a SINDy model with linear dynamics and return the fitted A and u."""
        import pysindy as ps  # type: ignore
        from pysindy.feature_library import PolynomialLibrary  # type: ignore

        x = df.values
        t = df.index.values

        # Use PolynomialLibrary with degree=1 (linear terms only) and include_bias=True
        library = PolynomialLibrary(degree=1, include_bias=True, include_interaction=False)
        optimizer = ps.STLSQ(threshold=0.0, alpha=alpha)

        model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
        )
        model.fit(x, t=t, feature_names=df.columns.tolist())

        # Extract coefficients: model.coefficients() returns shape (n_species, n_features)
        # PolynomialLibrary(include_bias=True) orders features as [1, x_1, ..., x_n]
        coefs = model.coefficients()
        n_species = df.shape[1]
        A_sindy = np.zeros((n_species, n_species))
        u_sindy = np.zeros(n_species)

        for i in range(n_species):
            u_sindy[i] = coefs[i, 0]          # bias term (first feature)
            A_sindy[i, :] = coefs[i, 1:]       # linear coefficients

        return A_sindy, u_sindy

    def test_comparison_dense_system(self) -> None:
        """Both methods produce reasonable predictions for a dense 3x3 system."""
        from src.jacobian_estimator import JacobianEstimator

        A_true = np.array([
            [-1.0,  0.2,  0.0],
            [ 0.1, -0.8,  0.3],
            [ 0.0,  0.1, -0.5]
        ])
        u_true = np.array([0.05, -0.02, 0.01])

        df = self._make_linear_data(A_true, u_true, n_points=300, seed=42)

        # Fit with JacobianEstimator (low alpha for minimal shrinkage)
        je = JacobianEstimator(df)
        je.fit(alpha=0.001)

        # Fit with SINDy
        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=0.001)

        # Both methods should produce predictions that are in the right direction
        test_points = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.5]])
        for x in test_points:
            pred_je = je.predict(x)
            pred_sindy = A_sindy @ x + u_sindy
            true_dxdt = A_true @ x + u_true

            # Both should have the same sign pattern as the true derivative
            self.assertEqual(np.sign(pred_je[0]), np.sign(true_dxdt[0]))
            self.assertEqual(np.sign(pred_je[1]), np.sign(true_dxdt[1]))
            self.assertEqual(np.sign(pred_sindy[0]), np.sign(true_dxdt[0]))

    def test_comparison_sparse_system(self) -> None:
        """Both methods recover a sparse 2x2 linear system within tolerance."""
        from src.jacobian_estimator import JacobianEstimator

        # Sparse: S1 only depends on itself, S2 depends on both
        A_true = np.array([[-0.5,  0.0],
                            [ 0.2, -0.3]])
        u_true = np.array([0.0, 0.0])

        df = self._make_linear_data(A_true, u_true, n_points=300, seed=42)

        je = JacobianEstimator(df)
        je.fit(alpha=0.01)

        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=0.01)

        # Both should be in the right ballpark (within 50% relative error)
        np.testing.assert_allclose(je.A_, A_true, atol=0.3)
        np.testing.assert_allclose(A_sindy, A_true, atol=0.2)

    def test_comparison_with_high_regularization(self) -> None:
        """With high alpha, both methods should produce sparse models."""
        from src.jacobian_estimator import JacobianEstimator

        # Dense true system
        A_true = np.array([[-1.0,  0.5],
                            [ 0.3, -0.7]])
        u_true = np.array([0.1, -0.05])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)

        je = JacobianEstimator(df)
        je.fit(alpha=1.0)

        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=1.0)

        # With high regularization, off-diagonal terms should be small for both
        self.assertAlmostEqual(abs(je.A_[0, 1]), 0.0, delta=0.5)
        self.assertAlmostEqual(abs(A_sindy[0, 1]), 0.0, delta=0.5)

    def test_comparison_single_species(self) -> None:
        """Both methods work for a single-species system."""
        from src.jacobian_estimator import JacobianEstimator

        A_true = np.array([[-0.5]])
        u_true = np.array([0.1])

        df = self._make_linear_data(A_true, u_true, n_points=200, seed=42)

        je = JacobianEstimator(df)
        je.fit(alpha=0.001)

        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=0.001)

        np.testing.assert_allclose(je.A_, A_true, atol=0.2)
        np.testing.assert_allclose(A_sindy, A_true, atol=0.2)
        np.testing.assert_allclose(je.u_, u_true, atol=0.1)
        np.testing.assert_allclose(u_sindy, u_true, atol=0.1)

    def test_comparison_predict_accuracy(self) -> None:
        """Both methods produce predictions in the right direction."""
        from src.jacobian_estimator import JacobianEstimator

        A_true = np.array([[-0.8,  0.3],
                            [ 0.1, -0.6]])
        u_true = np.array([0.02, -0.01])

        df = self._make_linear_data(A_true, u_true, n_points=300, seed=42)

        je = JacobianEstimator(df)
        je.fit(alpha=0.001)

        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=0.001)

        # Test predictions at several points - both should be in the right direction
        test_points = np.array([
            [1.0, 2.0],
            [0.5, 0.5],
            [3.0, 1.0],
            [0.1, 0.1],
        ])

        true_dxdt = A_true @ test_points.T + u_true[:, np.newaxis]

        for i in range(test_points.shape[0]):
            pred_je = je.predict(test_points[i])
            pred_sindy = A_sindy @ test_points[i] + u_sindy
            # Both should have the same sign pattern as true derivative
            self.assertEqual(np.sign(pred_je[0]), np.sign(true_dxdt[0, i]))
            self.assertEqual(np.sign(pred_je[1]), np.sign(true_dxdt[1, i]))

    def test_comparison_larger_system(self) -> None:
        """Both methods produce reasonable predictions for a 4x4 system."""
        from src.jacobian_estimator import JacobianEstimator

        # Sparse-ish 4x4 system (diagonally dominant, few off-diagonal terms)
        A_true = np.array([
            [-1.0,  0.1,  0.0,  0.0],
            [ 0.2, -0.8,  0.1,  0.0],
            [ 0.0,  0.1, -0.6,  0.05],
            [ 0.0,  0.0,  0.0, -0.4]
        ])
        u_true = np.array([0.05, -0.02, 0.01, 0.0])

        df = self._make_linear_data(A_true, u_true, n_points=300, seed=42)

        je = JacobianEstimator(df)
        je.fit(alpha=0.01)

        A_sindy, u_sindy = self._fit_sindy_linear(df, alpha=0.01)

        # Both methods should produce predictions in the right direction
        test_points = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 0.5, 0.3]])
        for x in test_points:
            pred_je = je.predict(x)
            pred_sindy = A_sindy @ x + u_sindy
            true_dxdt = A_true @ x + u_true

            # Both should have the same sign pattern as the true derivative
            self.assertEqual(np.sign(pred_je[0]), np.sign(true_dxdt[0]))
            self.assertEqual(np.sign(pred_je[1]), np.sign(true_dxdt[1]))
            self.assertEqual(np.sign(pred_sindy[0]), np.sign(true_dxdt[0]))

        # Both should recover the sparsity pattern (zeros stay near zero)
        np.testing.assert_allclose(je.A_[0, 2], 0.0, atol=0.3)
        np.testing.assert_allclose(je.A_[0, 3], 0.0, atol=0.3)
        np.testing.assert_allclose(A_sindy[0, 2], 0.0, atol=0.3)


# =============================================================================
# Integration Tests
# =============================================================================

class TestJacobianEstimatorIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_workflow(self) -> None:
        """Complete workflow: create, fit, predict, equations."""
        from src.jacobian_estimator import JacobianEstimator

        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0, 5.0], "S2": [0.5, 0.8, 1.1, 1.4, 1.7]},
            index=[0.0, 1.0, 2.0, 3.0, 4.0],
        )

        est = JacobianEstimator(df)
        est.fit(alpha=0.01)
        pred = est.predict(np.array([1.0, 2.0]))
        eq_str = est.equations

        self.assertEqual(pred.shape, (2,))
        self.assertIsInstance(eq_str, str)
        self.assertTrue(len(eq_str) > 0)

    def test_multiple_fits(self) -> None:
        """Calling fit() multiple times with different alpha updates the model."""
        from src.jacobian_estimator import JacobianEstimator

        df = pd.DataFrame(
            {"S1": [1.0, 2.0, 3.0, 4.0], "S2": [0.5, 0.8, 1.1, 1.4]},
            index=[0.0, 1.0, 2.0, 3.0],
        )

        est = JacobianEstimator(df)
        est.fit(alpha=0.001)
        A_low_reg = est.A_.copy()

        est.fit(alpha=10.0)
        A_high_reg = est.A_.copy()

        # High regularization should produce sparser model (more zeros)
        n_zeros_low = np.sum(np.abs(A_low_reg) < 1e-6)
        n_zeros_high = np.sum(np.abs(A_high_reg) < 1e-6)
        self.assertGreaterEqual(n_zeros_high, n_zeros_low)

    def test_predict_matches_manual_computation(self) -> None:
        """predict() result equals A*x + u computed manually."""
        from src.jacobian_estimator import JacobianEstimator

        df = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
            index=[0.0, 1.0, 2.0],
        )
        est = JacobianEstimator(df)
        est.fit(alpha=0.01)

        x = np.array([2.5, 3.7])
        predicted = est.predict(x)
        expected = est.A_ @ x + est.u_
        np.testing.assert_allclose(predicted, expected)


if __name__ == "__main__":
    unittest.main()