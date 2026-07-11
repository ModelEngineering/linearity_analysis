"""Slow-subspace linear prediction by projecting out fast eigenmodes."""

import src.constants as cn  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.score import Score  # type: ignore
from src.trajectory import Trajectory  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from scipy.linalg import schur  # type: ignore
from typing import Optional, Tuple


class SlowSubspacePredictor(object):
    """Predicts species concentrations using only the slow eigenmodes of J.

    Uses the real Schur decomposition J = Q T Qᵀ with slow modes
    (|Re(λ)| < eigenvalue_threshold) sorted into the leading block.

    Fast modes are not simply zero at quasi-steady state; the forcing term
    drives them to a nonzero QSS value c_fast_qss = -T_ff⁻¹ f_fast.  The
    predictor accounts for this in two ways:

    1. QSS offset: x_qss = Q_fast @ c_fast_qss is added to every predicted
       timepoint, capturing the constant fast-mode contribution.
    2. Corrected slow forcing: f_corrected = f_slow − T_sf @ c_fast_qss
       so the slow ODE sees the effective forcing after fast modes have settled.

    The slow ODE dc/dt = T_ss c + f_corrected is then integrated with Radau,
    restarting from the projected actual state every num_step timepoints.
    Reconstruction: x_pred = Q_slow @ c + x_qss.

    When all modes are slow (n_fast = 0) the predictor is equivalent to
    LinearPredictor with JAC_MEDIAN.
    """

    def __init__(self,
            trajectory: Trajectory,
            eigenvalue_threshold: Optional[float] = None,
            num_step: int = -1) -> None:
        """
        Parameters
        ----------
        trajectory : Trajectory
        eigenvalue_threshold : Optional[float]
            Modes with |Re(λ)| >= threshold are fast and removed.
            Default: 1 / step_size (modes unresolvable within one step).
        num_step : int
            Window size in timepoints.  -1 uses the full trajectory length.
        """
        self.trajectory = trajectory
        if eigenvalue_threshold is None:
            step_size = ((trajectory.end_time - trajectory.start_time)
                    / (trajectory.num_point - 1))
            eigenvalue_threshold = 1.0 / step_size
        self.eigenvalue_threshold = eigenvalue_threshold
        if num_step < 0:
            num_step = trajectory.num_point - 1
        if num_step < 1:
            raise ValueError(f"num_step must be >= 1 or -1, got {num_step}.")
        self.num_step = num_step
        self._T, self._Q, self._n_slow = self._computeSchur()

    @property
    def num_slow_modes(self) -> int:
        """Number of slow eigenmodes retained for prediction."""
        return self._n_slow

    def _computeSchur(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Real Schur decomposition sorting slow modes into the leading block.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]
            (T, Q, n_slow): quasi-upper-triangular T, orthogonal Q, and the
            count of slow modes.  T[:n, :n] contains only slow eigenvalues;
            T[n:, n:] contains only fast eigenvalues; T[:n, n:] is the
            slow-to-fast coupling block.
        """
        J = self.trajectory.jacobian_median_arr
        schur_result: tuple = schur(J, output='real',  # type: ignore
                sort=lambda lam: abs(lam.real) < self.eigenvalue_threshold)
        return schur_result[0], schur_result[1], int(schur_result[2])

    def _computeQssOffset(self,
            forcing_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the fast-mode QSS offset and corrected slow forcing.

        At quasi-steady state the fast Schur coordinates satisfy
            T_ff c_fast + f_fast = 0  →  c_fast_qss = -T_ff⁻¹ f_fast
        The constant QSS contribution to full-space concentrations is
            x_qss = Q_fast @ c_fast_qss
        and the corrected slow-subspace forcing is
            f_corrected = f_slow − T_sf @ c_fast_qss

        Parameters
        ----------
        forcing_arr : np.ndarray
            Median forcing vector, shape (n_species,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x_qss, f_corrected): constant offset shape (n_species,) and
            corrected slow forcing shape (n_slow,).
        """
        T, Q, n_slow = self._T, self._Q, self._n_slow
        n_fast = T.shape[0] - n_slow
        f_schur = Q.T @ forcing_arr
        f_slow = f_schur[:n_slow]

        if n_fast == 0:
            return np.zeros(T.shape[0]), f_slow

        T_sf = T[:n_slow, n_slow:]
        T_ff = T[n_slow:, n_slow:]
        f_fast = f_schur[n_slow:]

        try:
            c_fast_qss = np.linalg.solve(T_ff, -f_fast)
        except np.linalg.LinAlgError:
            c_fast_qss = np.linalg.lstsq(T_ff, -f_fast, rcond=None)[0]

        x_qss = Q[:, n_slow:] @ c_fast_qss
        f_corrected = f_slow + T_sf @ c_fast_qss
        return x_qss, f_corrected

    def _predict(self) -> np.ndarray:
        """Integrate the reduced ODE in the slow subspace with QSS correction.

        Returns
        -------
        np.ndarray
            Shape (num_point, num_species).
        """
        timecourse_arr = self.trajectory.timecourse_df.values
        timepoints = self.trajectory.timepoint_arr
        num_point = len(timepoints)
        num_species = self.trajectory.model.num_species

        forcing_arr = np.median(
                self.trajectory.forcing_input_collection_arr, axis=0)
        x_qss, f_corrected = self._computeQssOffset(forcing_arr)

        prediction_arr = np.empty((num_point, num_species))
        prediction_arr[0] = timecourse_arr[0]

        n_slow = self._n_slow
        if n_slow == 0:
            for k in range(1, num_point):
                prediction_arr[k] = x_qss
            return prediction_arr

        T_ss = self._T[:n_slow, :n_slow]
        Q_slow = self._Q[:, :n_slow]

        def _ode(_: float, c: np.ndarray) -> np.ndarray:
            return T_ss @ c + f_corrected

        window_start = 0
        while window_start < num_point - 1:
            window_end = min(window_start + self.num_step, num_point - 1)
            c0 = Q_slow.T @ timecourse_arr[window_start].astype(float)
            t_eval = timepoints[window_start + 1: window_end + 1]
            sol = solve_ivp(
                    _ode,
                    (float(timepoints[window_start]),
                            float(timepoints[window_end])),
                    c0,
                    t_eval=t_eval,
                    method="Radau",
                    rtol=1e-8,
                    atol=1e-10,
            )
            prediction_arr[window_start + 1: window_end + 1] = (
                    Q_slow @ sol.y + x_qss[:, np.newaxis]).T
            window_start = window_end

        return prediction_arr

    def predict(self) -> pd.DataFrame:
        """Predict species concentrations using the slow subspace.

        Returns
        -------
        pd.DataFrame
            Time-indexed with columns matching the model's species names.
        """
        return pd.DataFrame(
                self._predict(),
                index=self.trajectory.timepoint_arr,
                columns=self.trajectory.model.species_names,
        )

    def score(self, description: str = "") -> pd.DataFrame:
        """Score the prediction against the actual timecourse.

        Parameters
        ----------
        description : str

        Returns
        -------
        pd.DataFrame
            One row per aggregation level (model + one per species).
        """
        prediction_df = self.predict()
        scorer = Score()
        score_infos = scorer.makeScoreInfo(
                description, self.trajectory.timecourse_df, prediction_df)
        return pd.DataFrame([info.__dict__ for info in score_infos])

    def plotPrediction(self, **kwargs) -> PlotOptions:
        """Plot actual and predicted timecourses.

        Actual values are solid lines; predictions are dashed.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions.

        Returns
        -------
        PlotOptions
        """
        prediction_df = self.predict()
        actual_df = self.trajectory.timecourse_df
        if "title" not in kwargs:
            scorer = Score()
            score_infos = scorer.makeScoreInfo("", actual_df, prediction_df)
            p95 = score_infos[0].p95
            model_name = self.trajectory.model.model_name
            kwargs["title"] = (f"{model_name} slow_modes={self.num_slow_modes},"
                    f" p95={p95:.2f}")
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        for i, name in enumerate(prediction_df.columns):
            color = f"C{i}"
            ax.plot(actual_df.index, actual_df[name],  # type: ignore
                    color=color, label=f"{name} (actual)")
            ax.plot(prediction_df.index, prediction_df[name],  # type: ignore
                    color=color, linestyle="--", label=f"{name} (predicted)")
        plot_options.apply()
        return plot_options

    @property
    def cost(self) -> float:
        """Mean squared relative prediction error, median over species.

        First timepoint is skipped (exact by construction).

        Returns
        -------
        float
        """
        actual_arr = self.trajectory.timecourse_df.values[1:]
        predicted_arr = self._predict()[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_arr = np.where(
                    actual_arr == 0,
                    np.nan,
                    (predicted_arr - actual_arr) / actual_arr,
            )
        species_costs = np.nanmean(rel_arr ** 2, axis=0)
        return float(np.nanmedian(species_costs))
