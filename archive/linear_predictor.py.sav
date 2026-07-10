"""Predicts model timecourses using a linear ODE approximation."""
import src.constants as cn  # type: ignore
from src.trajectory import Trajectory  # type: ignore
from src.score import Score, ScoreInfo  # type: ignore
from src.model import Model  # type: ignore

from lmfit import Parameters, minimize as lmfit_minimize  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from scipy.linalg import expm  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from typing import List, Optional


class LinearPredictor(object):
    """Predicts floating species concentrations using a linear ODE.

    The ODE is dx/dt = J @ x + f, where J is the selected Jacobian and
    f is the median forcing input from the Trajectory.

    Prediction is windowed: starting from the actual observed state at
    t[i], the ODE is integrated forward for num_step steps to t[i+num_step],
    then restarted from the actual state at t[i+num_step].  With num_step=1
    each step is predicted independently from the true preceding state; with
    a large num_step the prediction approaches a single continuous ODE
    integration from t[0].
    """

    def __init__(self,
            trajectory: Trajectory,
            jacobian_selection: str = cn.JAC_MEDIAN,
            num_step: int = -1,
            is_species_prediction: bool = False) -> None:
        """
        Parameters
        ----------
        trajectory : Trajectory
        jacobian_selection : str
            Which Jacobian to use: cn.JAC_MEDIAN, cn.JAC_FIRST, or cn.JAC_FITTED.
        num_step : int
            Window size in timepoints. The ODE is restarted from the actual
            observed state every num_step points. Must be >= 1, or -1 to use
            num_point - 1 (single window covering the full trajectory).
        is_species_prediction : bool
            If True, predict each species in isolation, using the actual observed values of the other species at each timepoint.
            If False, predict all species concentrations without using other actual values
        """
        self.trajectory = trajectory
        self.jacobian_selection = jacobian_selection
        if num_step < 0:
            num_step = self.trajectory.num_point - 1
        if num_step < 1:
            raise ValueError(f"num_step must be >= 1 or -1, got {num_step}.")
        self.num_step = num_step
        self.is_species_prediction = is_species_prediction
        self.forcing_input_arr = np.median(self.trajectory.forcing_input_collection_arr, axis=0)
        self._fitted_jacobian_arr = np.array([])

    @classmethod
    def makeFromBiomodels(cls,
            model_name: str = "",
            model_num: int = 0,
            jacobian_selection: str = cn.JAC_MEDIAN,
            num_step: int = 1,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_point: int = cn.NUM_POINT) -> "LinearPredictor":
        """Construct a LinearPredictor from a BioModels entry.

        Exactly one of model_name or model_num must be supplied.

        Parameters
        ----------
        model_name : str
            BioModel identifier, e.g. 'BIOMD0000000008'. Must start with 'BIOMD'.
        model_num : int
            BioModel number, e.g. 8. Formatted to 'BIOMD0000000008'.
        jacobian_selection : str
            Passed to LinearPredictor constructor.
        num_step : int
            Passed to LinearPredictor constructor.
        start_time : float
            Passed to Trajectory.makeFromSimulation.
        end_time : Optional[float]
            Passed to Trajectory.makeFromSimulation. None triggers auto-detection.
        num_point : int
            Passed to Trajectory.makeFromSimulation.

        Returns
        -------
        LinearPredictor
        """
        if model_name and model_num:
            raise ValueError("Provide model_name or model_num, not both.")
        if not model_name and not model_num:
            raise ValueError("Provide either model_name or model_num.")
        if model_num:
            model_name = f"BIOMD{model_num:010d}"
        model = Model.makeBiomodel(model_name)
        trajectory = Trajectory.makeFromSimulation(
                model,
                start_time=start_time,
                end_time=end_time,
                num_point=num_point,
        )
        return cls(trajectory,
                jacobian_selection=jacobian_selection,
                num_step=num_step)

    def _getJacobian(self) -> np.ndarray:
        """Return the Jacobian matrix selected by jacobian_selection."""
        if self.jacobian_selection == cn.JAC_FITTED:
            if self.trajectory.model.num_species > self.trajectory.num_point:
                print(
                        f"Cannot use JAC_FITTED: num_species "
                        f"({self.trajectory.model.num_species}) > num_point "
                        f"({self.trajectory.num_point}).")
                return self.trajectory.jacobian_median_arr
            else:
                try:
                    jacobian = self._fitJacobian()
                    return jacobian
                except Exception as e:
                    print(f"Error fitting Jacobian: {e}")
                    print("Falling back to JAC_MEDIAN.")
                    return self.trajectory.jacobian_median_arr
        if self.jacobian_selection == cn.JAC_MEDIAN:
            return self.trajectory.jacobian_median_arr
        if self.jacobian_selection == cn.JAC_FIRST:
            return self.trajectory.jacobian_collection_arr[0]
        raise ValueError(
                f"Unknown jacobian_selection: {self.jacobian_selection!r}. "
                f"Must be {cn.JAC_MEDIAN!r}, {cn.JAC_FIRST!r}, or "
                f"{cn.JAC_FITTED!r}.")

    def _predictWithJacobian(self, jacobian_arr: np.ndarray,
            num_step: Optional[int] = None,
            ) -> np.ndarray:
        """Predict the trajectory using num_step-sized windows.

        For each window [t[i], t[i+num_step]], integrates the linear ODE
        forward starting from the actual observed state in timecourse_df at
        t[i].  Restarts from the next actual state at the end of each window.

        Parameters
        ----------
        jacobian_arr : np.ndarray
            Shape (num_species, num_species). The Jacobian to use.
        num_step : Optional[int]
            Window size in timepoints. Overrides self.num_step when provided.

        Returns
        -------
        np.ndarray
            Shape (num_point, num_species). Predicted concentrations at each
            timepoint.
        """
        if num_step is None:
            num_step = self.num_step
        #
        timecourse_arr = self.trajectory.timecourse_df.values
        timepoints = self.trajectory.timepoint_arr
        num_point = len(timepoints)
        num_species = self.trajectory.model.num_species

        prediction_arr = np.empty((num_point, num_species))
        prediction_arr[0] = timecourse_arr[0]

        ##
        def _ode(t: float, x: np.ndarray) -> np.ndarray:
            if self.is_species_prediction:
                matches = np.where(timepoints <= t)[0]
                idx = matches[-1] if len(matches) > 0 else 0
                actual_arr = timecourse_arr[idx]
                results = []
                for i in range(num_species):
                    x_i = actual_arr.copy()
                    x_i[i] = x[i]
                    results.append(jacobian_arr[i] @ x_i + self.forcing_input_arr[i])
                return np.array(results)
            else:
                return jacobian_arr @ x + self.forcing_input_arr
        ##

        window_start = 0
        while window_start < num_point - 1:
            window_end = min(window_start + num_step, num_point - 1)
            x0 = timecourse_arr[window_start].astype(float)
            t_eval = timepoints[window_start + 1: window_end + 1]
            sol = solve_ivp(
                    _ode,
                    (float(timepoints[window_start]), float(timepoints[window_end])),
                    x0,
                    t_eval=t_eval,
                    method="Radau",
                    rtol=1e-8,
                    atol=1e-10,
            )
            prediction_arr[window_start + 1: window_end + 1] = sol.y.T
            window_start = window_end

        return prediction_arr

    def _predictWithExpm(self, jacobian_arr: np.ndarray) -> np.ndarray:
        """Step-by-step prediction using the matrix exponential.

        Exact solution of dx/dt = J @ x + f for each one-step window,
        restarting from the actual observed state at each timepoint.
        Uses an augmented system [[J, f], [0, 0]] so the forcing term
        is handled exactly without inverting J.  When timepoints are
        uniformly spaced the matrix exponential is computed once and
        reused for all steps.

        Parameters
        ----------
        jacobian_arr : np.ndarray
            Shape (num_species, num_species). The Jacobian to use.

        Returns
        -------
        np.ndarray
            Shape (num_point, num_species). Predicted concentrations at each
            timepoint.
        """
        forcing_arr = np.median(self.trajectory.forcing_input_collection_arr, axis=0)
        timecourse_arr = self.trajectory.timecourse_df.values
        timepoints = self.trajectory.timepoint_arr
        num_point = len(timepoints)
        n = jacobian_arr.shape[0]

        # Augmented system: d/dt [x; 1] = [[J, f], [0, 0]] @ [x; 1]
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = jacobian_arr
        A_aug[:n, n] = forcing_arr

        prediction_arr = np.empty((num_point, n))
        prediction_arr[0] = timecourse_arr[0]

        dts = np.diff(timepoints)
        if np.allclose(dts, dts[0]):
            M = expm(A_aug * float(dts[0]))
            for k in range(num_point - 1):
                x_aug = np.append(timecourse_arr[k].astype(float), 1.0)
                prediction_arr[k + 1] = M[:n] @ x_aug
        else:
            for k in range(num_point - 1):
                dt = float(timepoints[k + 1] - timepoints[k])
                M = expm(A_aug * dt)
                x_aug = np.append(timecourse_arr[k].astype(float), 1.0)
                prediction_arr[k + 1] = M[:n] @ x_aug

        return prediction_arr

    def _fitJacobian(self, max_nfev: int = 1000) -> np.ndarray:
        """Fit each row of the Jacobian to minimise the L2 prediction error.

        Fits the Jacobian row by row (one species at a time).  For each row,
        lmfit least-squares adjusts the Jacobian coefficients so that the
        windowed prediction for that species best matches timecourse_df.
        Previously fitted rows are kept fixed during each row's optimisation.

        Parameters
        ----------
        max_nfev : int
            Maximum function evaluations per row optimisation.

        Returns
        -------
        np.ndarray
            Fitted Jacobian of shape (num_species, num_species).
        """
        # Check if computed previously
        if self._fitted_jacobian_arr.size > 0:
            return self._fitted_jacobian_arr
        #
        num_species = self.trajectory.model.num_species
        if self.trajectory.num_point < num_species:
            raise ValueError(
                    f"Cannot fit Jacobian: num_point "
                    f"({self.trajectory.num_point}) < num_species "
                    f"({num_species}).")

        jacobian_arr = self.trajectory.jacobian_median_arr.copy()

        def _calculateResiduals(params: Parameters, ispecies: int) -> np.ndarray:
            for i in range(num_species):
                jacobian_arr[ispecies, i] = params[f"d{i}"].value
            # prediction_arr = self._predictWithJacobian(
            #         jacobian_arr, num_step=1)[:, ispecies]
            prediction_arr = self._predictWithExpm(jacobian_arr)[:, ispecies]
            residual_arr = (self.trajectory.timecourse_df.iloc[:, ispecies].values
                    - prediction_arr)
            return residual_arr[1:]

        for ispecies, _ in enumerate(self.trajectory.model.species_names):
            params = Parameters()
            for idx, coeff in enumerate(jacobian_arr[ispecies]):
                params.add(f"d{idx}", value=coeff)
            result = lmfit_minimize(_calculateResiduals, params,
                    method="leastsq", args=(ispecies,), max_nfev=max_nfev)
            jacobian_arr[ispecies] = [result.params[f"d{idx}"].value  # type: ignore
                    for idx in range(jacobian_arr.shape[1])]

        self._fitted_jacobian_arr = jacobian_arr
        return self._fitted_jacobian_arr

    def predict(self) -> pd.DataFrame:
        """Predict species concentrations over the Trajectory's timecourse.

        Uses windowed ODE integration: the ODE is restarted from the actual
        observed state in timecourse_df every num_step timepoints.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with columns matching the model's species names.
        """
        prediction_arr = self._predictWithJacobian(self._getJacobian())
        return pd.DataFrame(
                prediction_arr,
                index=self.trajectory.timepoint_arr,
                columns=self.trajectory.model.species_names,
        )

    def score(self, description: str = "") -> pd.DataFrame:
        """Score the prediction against the actual timecourse.

        Parameters
        ----------
        description : str
            Label stored in the 'description' column of the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per aggregation level: one model-level row
            (aggregation_type='model') and one per species.
        """
        prediction_df = self.predict()
        scorer = Score(is_initialize=True)
        score_infos = scorer.makeScoreInfo(
                description, self.trajectory.timecourse_df, prediction_df)
        df = pd.DataFrame([info.__dict__ for info in score_infos])
        return df

    @property
    def cost(self) -> float:
        """Mean squared relative prediction error, aggregated across species.

        For each species: mean over timepoints of ((predicted - actual) / actual)²,
        skipping the first timepoint (exact by construction) and any timepoints
        where actual == 0.  Returns the median of the per-species costs.

        Returns
        -------
        float
        """
        actual_arr = self.trajectory.timecourse_df.values[1:]
        predicted_arr = self.predict().values[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_arr = np.where(
                    actual_arr == 0,
                    np.nan,
                    (predicted_arr - actual_arr) / actual_arr,
            )
        species_costs = np.nanmean(rel_arr ** 2, axis=0)
        return float(np.nanmedian(species_costs))

    def plotPrediction(self, **kwargs) -> PlotOptions:
        """Plot actual and predicted timecourses.

        Actual values are drawn as solid lines; predictions as dashed lines.
        One line per species; actual and predicted for the same species share a
        colour.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions. Supported keys: ax, fig, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.

        Returns
        -------
        PlotOptions
        """
        prediction_df = self.predict()
        actual_df = self.trajectory.timecourse_df
        score = Score(is_initialize=True)
        score.addTestResult(prediction_df, actual_df)
        p95 = score.score_df.loc[0, "p95"]

        if not "title" in kwargs:
            kwargs["title"] = (f"{self.trajectory.model.model_name} "
                    f"num_step={self.num_step}, p95={p95:.2f}")    
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax

        for i, name in enumerate(prediction_df.columns):
            color = f"C{i}"
            ax.plot(actual_df.index, actual_df[name], color=color,  # type: ignore
                    label=f"{name} (actual)")
            ax.plot(prediction_df.index, prediction_df[name], color=color,  # type: ignore
                    linestyle="--", label=f"{name} (predicted)")

        plot_options.apply()
        return plot_options
