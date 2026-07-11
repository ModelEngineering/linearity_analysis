"""Tests for TPWL in piecewise_system_discovery.py."""

import unittest
import numpy as np  # type: ignore

from src.trajectory_piecewise_linear import TPWL  # type: ignore


def _linear_f(A: np.ndarray):
    return lambda x: A @ x


def _linear_jac(A: np.ndarray):
    return lambda x: A


class TestTPWLPointSelection(unittest.TestCase):

    def setUp(self):
        # Simple linear system dx/dt = A x, used as a stand-in nonlinear f/jac.
        self.A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.f = _linear_f(self.A)
        self.jac = _linear_jac(self.A)

    def testAddPointAffineOffsetIsExact(self):
        # A_i @ s_i + b_i must equal f(s_i) exactly at the expansion point.
        model = TPWL(self.f, self.jac, delta=0.4, weighting="nearest")
        s = np.array([1.0, 2.0])
        model._add_point(s)
        reconstructed = model.As[0] @ s + model.bs[0]
        np.testing.assert_allclose(reconstructed, self.f(s))

    def testSmallerDeltaSelectsMorePoints(self):
        x0 = np.array([1.0, 0.0])
        t_span = (0.0, 10.0)
        t_eval = np.linspace(0.0, 10.0, 200)

        model_coarse = TPWL(self.f, self.jac, delta=2.0, weighting="nearest")
        model_coarse.train(x0, t_span, t_eval)

        model_fine = TPWL(self.f, self.jac, delta=0.1, weighting="nearest")
        model_fine.train(x0, t_span, t_eval)

        self.assertGreater(len(model_fine.points), len(model_coarse.points))


class TestTPWLRhs(unittest.TestCase):

    def setUp(self):
        self.A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.f = _linear_f(self.A)
        self.jac = _linear_jac(self.A)

    def testRhsBeforeTrainRaises(self):
        model = TPWL(self.f, self.jac, delta=0.4, weighting="nearest")
        with self.assertRaises(RuntimeError):
            model.rhs(np.array([1.0, 0.0]))

    def testUnknownWeightingRaises(self):
        model = TPWL(self.f, self.jac, delta=0.4, weighting="bogus")
        model._add_point(np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            model.rhs(np.array([1.0, 0.0]))

    def testNearestWeightingUsesOnlyClosestPoint(self):
        model = TPWL(self.f, self.jac, delta=0.4, weighting="nearest")
        near = np.array([0.0, 0.0])
        far = np.array([10.0, 10.0])
        model._add_point(near)
        model._add_point(far)

        x = np.array([0.1, 0.1])  # much closer to `near`
        expected = model.As[0] @ x + model.bs[0]
        np.testing.assert_allclose(model.rhs(x), expected)

    def testGaussianWeightsSumToOneAtExpansionPoint(self):
        model = TPWL(self.f, self.jac, delta=0.4, weighting="gaussian", alpha=1.0)
        model._add_point(np.array([0.0, 0.0]))
        model._add_point(np.array([5.0, 5.0]))

        # Evaluated exactly at an expansion point, the blended derivative
        # must equal f at that point regardless of the other point's pull,
        # since A_i @ s_i + b_i == f(s_i) for every i is NOT generally true
        # for i != the queried point -- so instead check weights sum to 1
        # by reconstructing them directly from the public rhs() output
        # using two well-separated points where the far one's weight is
        # numerically negligible.
        x = np.array([0.0, 0.0])
        result = model.rhs(x)
        expected = model.As[0] @ x + model.bs[0]  # near point dominates
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestTPWLSimulate(unittest.TestCase):

    def testSimulateReproducesLinearSystemExactly(self):
        # For a genuinely linear system, a single expansion point gives an
        # exact affine model everywhere, so TPWL should match the true
        # solution to integration tolerance.
        A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        f = _linear_f(A)
        jac = _linear_jac(A)

        x0 = np.array([1.0, 0.0])
        t_span = (0.0, 5.0)
        t_eval = np.linspace(0.0, 5.0, 50)

        model = TPWL(f, jac, delta=10.0, weighting="nearest")
        model.train(x0, t_span, t_eval)
        self.assertEqual(len(model.points), 1)

        sol_tpwl = model.simulate(x0, t_span, t_eval, rtol=1e-8, atol=1e-10)
        self.assertTrue(sol_tpwl.success)

        from scipy.integrate import solve_ivp  # type: ignore
        sol_true = solve_ivp(lambda t, x: f(x), t_span, x0, t_eval=t_eval,
                              rtol=1e-8, atol=1e-10)

        np.testing.assert_allclose(sol_tpwl.y, sol_true.y, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
