'''Linear predictor for chemical reaction network trajectories.'''

from trajectory import Trajectory  # type: ignore
from src.l_roadrunner import LRoadrunner  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from typing import Optional


ZERO_TOL = 1e-10  # Threshold below which concentrations are treated as zero.

PredictionResult = collections.namedtuple(
    "PredictionResult",
    ["prediction_arr", "mean_abs_error", "max_abs_err"],
)


class LinearPredictor(object):
    """Predicts floating species concentrations using a linear ODE with a mean Jacobian.

    The linear model is: dx/dt = mean_jacobian_collection_arr @ x + forced_input_arr,
    where mean_jacobian_collection_arr is the element-wise mean of all Jacobians in the
    collection and forced_input_arr is a constant forcing vector (e.g., from
    boundary species or 0th-order rate laws).
    """

    def __init__(self,
            jacobian_collection: Trajectory,
            initial_value_arr: np.ndarray,
            forced_input_arr: np.ndarray,
            ) -> None:
        """
        Parameters
        ----------
        jacobian_collection : JacobianCollection
            Collection of Jacobian matrices over one or more timepoints. The
            mean of these Jacobians is used as the linear model.
        initial_value_arr : np.ndarray
            1-D array of shape (n_species,) containing the initial floating
            species concentrations at t=0.
        forced_input_arr : np.ndarray
            1-D array of shape (n_species,) representing the constant forcing
            vector u in dx/dt = J_mean @ x + u.
        """
        self.jacobian_collection = jacobian_collection
        self._l_roadrunner = jacobian_collection.l_roadrunner
        self.initial_value_arr = initial_value_arr
        self.forced_input_arr = forced_input_arr
        self.mean_jacobian_collection_arr = np.mean(jacobian_collection.jacobian_collection_arr, axis=0)

    def predict(self, times: np.ndarray, l_roadrunner: Optional[LRoadrunner] = None) -> "PredictionResult":
        """Predict floating species concentrations at the given times.

        Solves the linear ODE dx/dt = mean_jacobian_collection_arr @ x + forced_input_arr
        with initial condition initial_value_arr at t=times[0], using
        scipy.integrate.solve_ivp for numerical robustness.

        When ``l_roadrunner`` is provided, the method also computes error metrics
        by comparing the prediction against the reference simulation.  The
        relative absolute error per entry is ``|1 - predicted / simulated|``; if
        both values are within ``ZERO_TOL`` the entry contributes 0 error.

        Parameters
        ----------
        times : np.ndarray
            1-D array of times relative to t=0 (i.e., relative to the state
            described by initial_value_arr). Must be non-decreasing.
        l_roadrunner : LRoadrunner, optional
            When supplied, its ``simulate()`` output is used as the reference
            to compute ``mean_abs_error`` and ``max_abs_err``.  The simulation
            must have the same number of timepoints as ``times``.  If omitted
            or if shapes do not match, error fields are set to ``np.nan``.

        Returns
        -------
        PredictionResult
            Named tuple with fields:
            - ``prediction_arr`` : ndarray of shape (len(times), n_species)
            - ``mean_abs_error`` : float — mean of |1 - predicted/simulated|
            - ``max_abs_err``    : float — max  of |1 - predicted/simulated|
        """
        # Degenerate case: all times equal — no integration needed.
        if float(times[0]) == float(times[-1]):
            prediction_arr = np.tile(self.initial_value_arr, (len(times), 1))
        else:
            def _ode(t: float, x: np.ndarray) -> np.ndarray:
                return self.mean_jacobian_collection_arr @ x + self.forced_input_arr

            sol = solve_ivp(
                _ode,
                (float(times[0]), float(times[-1])),
                self.initial_value_arr,
                t_eval=times,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )
            prediction_arr = sol.y.T  # shape: (len(times), n_species)

        mean_abs_error: float = np.nan
        max_abs_err: float = np.nan
        effective_lr = l_roadrunner if l_roadrunner is not None else self._l_roadrunner
        if effective_lr is not None:
            simulated_arr = effective_lr.simulate(is_with_timepoints=False)
            if simulated_arr.shape == prediction_arr.shape:
                with np.errstate(divide="ignore", invalid="ignore"):
                    abs_error_arr = np.abs(1.0 - prediction_arr / simulated_arr)
                both_near_zero = (
                    (np.abs(prediction_arr) < ZERO_TOL)
                    & (np.abs(simulated_arr) < ZERO_TOL)
                )
                abs_error_arr[both_near_zero] = 0.0
                mean_abs_error = float(np.nanmean(abs_error_arr))
                max_abs_err = float(np.nanmax(abs_error_arr))

        return PredictionResult(
            prediction_arr=prediction_arr,
            mean_abs_error=mean_abs_error,
            max_abs_err=max_abs_err,
        )

    def plot(self, l_roadrunner: LRoadrunner,
            ax: Optional[maxes.Axes] = None,
            title: str = "Linear Prediction vs Simulation",
            ylim: Optional[tuple[float, float]] = None,
            xlim: Optional[tuple[float, float]] = None) -> mfigure.Figure:
        """Plot the linear prediction and the simulation on the same axes.

        The simulation is drawn as solid lines; the linear prediction as dashed
        lines.  One line per floating species is drawn, using matplotlib's
        default colour cycle so that prediction and simulation for the same
        species share a colour.

        Parameters
        ----------
        l_roadrunner : LRoadrunner
            LRoadrunner instance used to run the reference simulation.  Its
            ``start_time``, ``end_time``, and ``num_points`` determine the
            simulated time grid.
        ax : plt.Axes, optional
            Axes to draw on.  A new figure and axes are created when omitted.
        title : str, optional
            Title for the plot.  Default is "Linear Prediction vs Simulation".
        ylim : tuple[float, float], optional
            y-axis limits as (ymin, ymax).  If omitted, determined automatically.
        xlim : tuple[float, float], optional
            x-axis limits as (xmin, xmax).  If omitted, determined automatically.

        Returns
        -------
        plt.Figure
            The figure containing the plot.
        """
        times = self.jacobian_collection.timepoint_arr
        predicted_arr = self.predict(times).prediction_arr
        simulated_arr = l_roadrunner.simulate(is_with_timepoints=False)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        rr = l_roadrunner.getRoadrunner()
        species_ids = rr.getFloatingSpeciesIds()
        n_species = predicted_arr.shape[1]

        for i in range(n_species):
            label = species_ids[i] if i < len(species_ids) else f"species_{i}"
            color = f"C{i}"
            ax.plot(times, simulated_arr[:, i], color=color, label=f"{label} (simulation)")
            ax.plot(times, predicted_arr[:, i], color=color, linestyle="--",
                    label=f"{label} (prediction)")

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.legend()
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        return fig  # type: ignore