"""Trajectory piecewise-linear (TPWL) approximation of a nonlinear ODE.

Reference: Rewienski & White, "A trajectory piecewise-linear approach to
model order reduction and fast simulation of nonlinear circuits and
micromachined devices," IEEE Trans. CAD, 22(2):155-170, 2003.

Approximates ``dx/dt = f(x)`` by linearizing around a set of "expansion
points" collected along a representative trajectory, then blending the
local affine models at evaluation time.
"""

import numpy as np  # type: ignore
from typing import Callable

"""
Changes
1. Distance calculation must be normalized so that delta is in the same magnitudes.
"""

class TPWL:
    """Trajectory piecewise-linear surrogate for a nonlinear ODE.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE. ``f(x) -> array of shape (n,)``.
    jac : callable
        Jacobian of f. ``jac(x) -> array of shape (n, n)``.
    delta : float
        Minimum Euclidean distance between expansion points. Smaller
        delta selects more points along the training trajectory.
    weighting : {'nearest', 'gaussian'}
        ``'nearest'`` uses only the closest expansion point (hard switch).
        ``'gaussian'`` blends all expansion points with Gaussian weights.
    alpha : float
        Bandwidth parameter for Gaussian weighting (ignored for 'nearest').
    """

    def __init__(
        self,
        f: Callable,
        jac: Callable,
        delta: float = 0.5,
        weighting: str = "nearest",
        alpha: float = 2.0,
    ):
        self.f = f
        self.jac = jac
        self.delta = delta
        self.weighting = weighting
        self.alpha = alpha

        self.points: list[np.ndarray] = []
        self.As: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        self.fs: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, x0: np.ndarray, t_span: tuple, t_eval: np.ndarray,
              **ivp_kwargs) -> dict:
        """Simulate the full nonlinear ODE and collect expansion points.

        Parameters
        ----------
        x0 : array_like, shape (n,)
            Initial condition for the training trajectory.
        t_span : (t0, tf)
            Integration time interval.
        t_eval : array_like
            Times at which to evaluate the training solution.
        **ivp_kwargs
            Extra keyword arguments forwarded to ``scipy.integrate.solve_ivp``.

        Returns
        -------
        dict with keys 'sol' (the full ODE solution) and 'n_points'
        (number of expansion points selected).
        """
        from scipy.integrate import solve_ivp  # type: ignore

        sol = solve_ivp(
            lambda t, x: self.f(x),
            t_span,
            x0,
            t_eval=t_eval,
            dense_output=False,
            **ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"Training ODE failed: {sol.message}")

        X_train = sol.y.T  # shape (T, n)
        self._collect_expansion_points(X_train)
        return {"sol": sol, "n_points": len(self.points)}

    def _collect_expansion_points(self, X_train: np.ndarray) -> None:
        """Greedy selection: add a point whenever the current state is
        farther than ``self.delta`` from every existing point."""
        self.points.clear()
        self.As.clear()
        self.bs.clear()
        self.fs.clear()

        for x in X_train:
            if self._is_new_point(x):
                self._add_point(x)

    def _is_new_point(self, x: np.ndarray) -> bool:
        if not self.points:
            return True
        dists = [np.linalg.norm(x - s) for s in self.points]
        return bool(np.min(dists) > self.delta)

    def _add_point(self, x: np.ndarray) -> None:
        A = self.jac(x)
        fx = self.f(x)
        b = fx - A @ x  # so that A @ x + b == f(x) exactly at the point
        self.points.append(x.copy())
        self.As.append(A)
        self.bs.append(b)
        self.fs.append(fx.copy())

    # ------------------------------------------------------------------
    # Evaluation (online phase)
    # ------------------------------------------------------------------

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the TPWL approximation of f at state x.

        Returns dx/dt ≈ sum_i w_i(x) * (A_i x + b_i).
        """
        if not self.points:
            raise RuntimeError("Model not trained yet — call train() first.")

        dists_sq = np.array([np.dot(x - s, x - s) for s in self.points])

        if self.weighting == "nearest":
            i_star = int(np.argmin(dists_sq))
            return self.As[i_star] @ x + self.bs[i_star]

        elif self.weighting == "gaussian":
            log_w = -self.alpha * dists_sq
            log_w -= log_w.max()  # numerically stable normalization
            w = np.exp(log_w)
            w /= w.sum()
            result = np.zeros_like(x, dtype=float)
            for i, (A, b) in enumerate(zip(self.As, self.bs)):
                result += w[i] * (A @ x + b)
            return result

        else:
            raise ValueError(f"Unknown weighting: {self.weighting!r}")

    def simulate(self, x0: np.ndarray, t_span: tuple, t_eval: np.ndarray,
                 **ivp_kwargs):
        """Simulate the TPWL surrogate model from initial condition x0.

        Returns a scipy OdeResult object (same interface as solve_ivp).
        """
        from scipy.integrate import solve_ivp  # type: ignore

        return solve_ivp(
            lambda t, x: self.rhs(x),
            t_span,
            x0,
            t_eval=t_eval,
            **ivp_kwargs,
        )

    def nearest_point_index(self, x: np.ndarray) -> int:
        """Return the index of the expansion point nearest to x."""
        dists = [np.linalg.norm(x - s) for s in self.points]
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        print("\nTPWL Model Summary")
        print(f"  Weighting     : {self.weighting}")
        print(f"  delta (min dist): {self.delta}")
        print(f"  Expansion pts : {len(self.points)}")
        if self.points:
            n = len(self.points[0])
            print(f"  State dim     : {n}")
            eig_info = [np.linalg.eigvals(A).real.max() for A in self.As]
            print(f"  Max Re(eig) A_k: min={min(eig_info):.3f}, "
                  f"max={max(eig_info):.3f}")
