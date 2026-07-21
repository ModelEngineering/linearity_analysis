"""
Jacobian Estimator: estimates the Jacobian matrix and forcing input vector from time-series data.

Given a vector x(t) of state variables, this estimator finds the best-fit linear model:
    dx/dt = A * x + u
using Lasso regression.

Dependencies
------------
    pip install pandas numpy scikit-learn

Usage
-----
    from src.jacobian_estimator import JacobianEstimator

    estimator = JacobianEstimator(timecourse_df)
    estimator.fit(alpha=0.1)
    print(estimator.equations)
    pred = estimator.predict(np.array([1.0, 2.0]))
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import Lasso  # type: ignore


class JacobianEstimator:
    """Estimate the Jacobian matrix and forcing input vector from time-series data.

    Given a system of linear differential equations:
        dx/dt = A * x + u
    this class estimates the matrix ``A`` and the forcing input vector ``u`` using Lasso regression.

    Parameters
    ----------
    timecourse_df : pd.DataFrame
        Time series data with time as the index and state variables as columns.

    Raises
    ------
    TypeError
        If ``timecourse_df`` is not a ``pd.DataFrame``.
    ValueError
        If ``timecourse_df`` is empty, has no columns, has an index that is not strictly
        monotonically increasing, or contains any NaN or infinite values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'S1': [1.0, 2.0, 3.0, 4.0],
    ...     'S2': [0.5, 0.8, 1.1, 1.4]
    ... }, index=[0.0, 1.0, 2.0, 3.0])
    >>> estimator = JacobianEstimator(df)
    >>> estimator.fit(alpha=0.01)
    >>> print(estimator.equations)
    """

    def __init__(self, timecourse_df: pd.DataFrame) -> None:
        # Validate type
        if not isinstance(timecourse_df, pd.DataFrame):
            raise TypeError(
                f"timecourse_df must be a pd.DataFrame, got {type(timecourse_df).__name__}"
            )

        # Validate non-empty and has columns
        if timecourse_df.empty:
            raise ValueError("timecourse_df is empty.")
        if len(timecourse_df.columns) == 0:
            raise ValueError("timecourse_df has no columns (no state variables).")

        # Validate index is strictly monotonically increasing
        idx = timecourse_df.index
        if not (np.diff(idx.values.astype(float)) > 0).all():
            raise ValueError(
                "timecourse_df index must be strictly monotonically increasing."
            )

        # Validate no NaN or infinite values
        if timecourse_df.isnull().any().any():
            raise ValueError("timecourse_df contains NaN values.")
        if np.isinf(timecourse_df.values).any():
            raise ValueError("timecourse_df contains infinite values.")

        self.timecourse_df = timecourse_df

        # Compute derivatives using forward finite differences.
        # dtimecourse_df has one fewer row than timecourse_df (derivative at index i uses rows i and i+1).
        raw_values = timecourse_df.values.astype(float)
        idx_float = timecourse_df.index.to_numpy(dtype=float)
        dt_arr = np.diff(idx_float)
        self.dtimecourse_df = pd.DataFrame(
            data=(raw_values[1:, :] - raw_values[:-1, :]) / dt_arr[:, np.newaxis],
            index=idx_float[:-1],
            columns=timecourse_df.columns,
        )

        # State for fitted model
        self._is_fitted: bool = False
        self.A_: np.ndarray  # denormalized Jacobian matrix (n_species x n_species)
        self.u_: np.ndarray  # denormalized forcing input vector (n_species,)

    def _require_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this method.")

    def fit(self, alpha: float = 0.01) -> "JacobianEstimator":
        """Fit the Jacobian matrix and forcing input vector using Lasso regression.

        Parameters
        ----------
        alpha : float, optional
            L1 regularization strength for Lasso. Must be non-negative. Larger values
            produce sparser models. Default ``0.01``.

        Returns
        -------
        self
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        # Build the design matrix X from state variables (with intercept column)
        # Each row is [x_1, x_2, ..., x_n, 1]
        # Use only the first N-1 rows to match dtimecourse_df (derivatives at t_0..t_{N-2})
        X = self.timecourse_df.values[:-1].copy()
        n_rows = X.shape[0]
        intercept_col = np.ones((n_rows, 1))
        X_design = np.hstack([X, intercept_col])

        # Fit one regression model per species (column of dtimecourse)
        n_species = X_design.shape[1] - 1  # number of state variables
        self.A_ = np.zeros((n_species, n_species))
        self.u_ = np.zeros(n_species)

        for i in range(n_species):
            y = np.asarray(self.dtimecourse_df.iloc[:, i], dtype=float)
            if alpha == 0.0:
                # Use OLS when no regularization to avoid Lasso bias
                from sklearn.linear_model import LinearRegression  # type: ignore
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X_design, y)
                coefs: np.ndarray = np.atleast_1d(np.asarray(lr.coef_, dtype=float))
            else:
                lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
                lasso.fit(X_design, y)
                coefs = np.atleast_1d(np.asarray(lasso.coef_, dtype=float))
            self.A_[i, :] = coefs[0:n_species].copy()
            self.u_[i] = float(coefs[n_species])  # intercept term

        self._is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the derivative given a state vector.

        Parameters
        ----------
        x : np.ndarray
            State vector in physical units, shape ``(n_species,)``.

        Returns
        -------
        np.ndarray
            Predicted derivative ``dx/dt``, shape ``(n_species,)``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._require_fitted()

        # Compute derivative: dx/dt = A * x + u
        result = self.A_.dot(x) + self.u_

        return np.asarray(result, dtype=float)

    @property
    def equations(self) -> str:
        """Return a string representation of the estimated linear ODEs.

        Each line shows the equation for one species in the form::

            dS_i/dt = c_1*S_1 + c_2*S_2 + ... + u_i

        where ``c_j`` are the Jacobian entries and ``u_i`` is the forcing input.

        Returns
        -------
        str
            Multi-line string with one equation per species.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._require_fitted()

        col_names = list(self.timecourse_df.columns)
        n_species = len(col_names)
        lines = []
        for i in range(n_species):
            terms = []
            for j in range(n_species):
                coef = self.A_[i, j]
                if abs(coef) > 1e-12:
                    if j == 0:
                        terms.append(f"{coef:.6g}*{col_names[j]}")
                    else:
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f" {sign} {abs(coef):.6g}*{col_names[j]}")

            # Add forcing input term
            u_val = self.u_[i]
            if abs(u_val) > 1e-12:
                sign = "+" if u_val >= 0 else "-"
                terms.append(f" {sign} {abs(u_val):.6g}")

            if not terms:
                terms.append("0")

            lines.append(f"d{col_names[i]}/dt = {' '.join(terms)}")

        return "\n".join(lines)