"""Estimates characteristic times for models using multiple strategies.

Uses :class:`Simulator` for all simulation work and follows a priority-based
strategy selection:

1. User-specified end time (source: ``user_specified``)
2. BioModels SEDML lookup (source: ``sedml``)
3. Steady-state detection (source: ``steadystate``)
4. Maximum median coefficient of variation (source: ``max_median_cv``)

This module is the simulation-driven counterpart to the analysis logic in
:class:`Trajectory <src.trajectory.Trajectory>`.  Where Trajectory uses a raw
RoadRunner instance for step-by-step control, this estimator delegates all
simulation work to :class:`Simulator`.
"""

import src.constants as cn  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes  # type: ignore
from src.model import Model  # type: ignore
from src.simulator import Simulator, SimulationResult  # type: ignore

import numpy as np  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
import tellurium as te  # type: ignore
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

CharacteristicTimeResult = Tuple[float, str]
"""A pair of ``(end_time, end_time_source)``."""


# ---------------------------------------------------------------------------
# CharacteristicTimeEstimator
# ---------------------------------------------------------------------------

class CharacteristicTimeEstimator:
    """Estimates the characteristic time of a model using multiple strategies.

    All simulation work is delegated to :class:`Simulator`.  Detection methods
    are tried in priority order; the first successful method determines the
    result.

    Parameters
    ----------
    model : Model
        The model to estimate for.
    start_time : float
        Start time of the simulation.
    num_point : int
        Number of time points for simulations.
    perturbation_value_fraction : float
        Fraction by which to perturb each initial value (used for CV maximisation).
    perturbation_species_fraction : float
        Fraction of species whose initial values are perturbed.
    """

    # Thresholds and bounds
    STEADY_STATE_THRESHOLD = 0.01
    LOG_LOWER = -5.0
    LOG_UPPER = 6.0
    MAX_ITERATOR_STEP = 50 * int(1e6)

    def __init__(self,
            model: Model,
            start_time: float = cn.START_TIME,
            num_point: int = cn.NUM_POINT,
    ) -> None:
        self.model = model
        self.start_time = start_time
        self.num_point = num_point

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def estimate(cls,
            model: Model,
            end_time: Optional[float] = None,
            start_time: float = cn.START_TIME,
            num_point: int = cn.NUM_POINT) -> CharacteristicTimeResult:
        """Estimate the characteristic time using priority-based strategy selection.

        Parameters
        ----------
        model : Model
            The model to estimate for.
        end_time : float, optional
            If provided, returned immediately with source ``user_specified``.
        start_time : float
            Start time of simulation.
        num_point : int
            Number of time points for simulations.

        Returns
        -------
        CharacteristicTimeResult
            A ``(end_time, end_time_source)`` tuple.

        Raises
        ------
        ValueError
            If no strategy can determine a characteristic time.
        """
        # 1. User-specified
        if end_time is not None:
            return float(end_time), cn.ENDTIME_SOURCE_USER_SPECIFIED

        # 2. BioModels SEDML lookup
        if model.model_name.startswith("BIOMD"):
            csv_end_time = getBiomodelsEndtimes().get(model.model_name, None)
            if csv_end_time is not None:
                return float(csv_end_time), cn.ENDTIME_SOURCE_SEDML

        # 3. Steady-state detection
        estimator = cls(
            model=model,
            start_time=start_time,
            num_point=num_point,
        )
        end_time = estimator.detect_steadystate()
        if end_time is not None:
            return float(end_time), cn.ENDTIME_SOURCE_STEADYSTATE

        # 4. Maximum median CV optimisation
        estimator = cls(
            model=model,
            start_time=start_time,
            num_point=num_point,
        )
        end_time = estimator.detect_cv_maximized()
        if end_time is not None:
            return float(end_time), cn.ENDTIME_SOURCE_MAX_MEDIAN_CV

        raise ValueError(
            "Could not determine an appropriate characteristic time. "
            "The model may be invalid or unbounded."
        )

    # ------------------------------------------------------------------
    # Detection methods (public for testing)
    # ------------------------------------------------------------------

    def detect_steadystate(self) -> Optional[float]:
        """Find the shortest end time at which the model reaches steady state.

        Uses RoadRunner's built-in steady-state solver to obtain reference
        values, then uses :class:`Simulator` in a binary-search loop to find
        the earliest simulation end time that converges to those values.

        Returns
        -------
        float or None
            The detected end time, or ``None`` if steady state cannot be found.
        """
        # --- Step 1: obtain reference steady-state values via RoadRunner ---
        rr = te.loadSBMLModel(self.model.sbml_str)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', self.MAX_ITERATOR_STEP)

        try:
            solver = rr.getSteadyStateSolver()
            for key, value in {
                "allow_approx": True,
                "approx_tolerance": 1e-3,
                "relative_tolerance": 1e-3,
                "maximum_iterations": 1000,
            }.items():
                solver.setValue(key, value)
            rr.steadyState()
        except RuntimeError:
            return None

        raw_ss = np.array(rr.getFloatingSpeciesConcentrations())
        if len(raw_ss) == 0 or np.any(np.isnan(raw_ss)) or np.any(np.isinf(raw_ss)):
            return None

        # Use a small floor to avoid division by zero, but track which species
        # are genuinely near-zero (from the raw value) so we can handle them
        # specially -- the floored value itself can no longer be used for
        # this check since it is always >= the floor.
        near_zero_mask = raw_ss <= 1e-8
        ss_arr = np.where(near_zero_mask, 1e-8, raw_ss)

        # --- Step 2: binary search with Simulator to find earliest convergence ---
        threshold = self.STEADY_STATE_THRESHOLD

        def _is_at_steady_state(end_t: float) -> bool:
            try:
                sim_result = self._run_simulation(end_time=end_t)
            except Exception:
                return False
            final_values = sim_result.timecourse_df.iloc[-1].values

            # Check each species individually.  Species that are near-zero in
            # steady state should also be near-zero at the candidate end time.
            for i, (ss_val, final_val) in enumerate(zip(ss_arr, final_values)):
                if near_zero_mask[i]:
                    # This species converged to ~0; check that it's also ~0 now.
                    if not np.isclose(final_val, 0.0, atol=threshold):
                        return False
                else:
                    # Normalized comparison for non-negligible species.
                    normalized = final_val / ss_val
                    if abs(normalized - 1) >= threshold:
                        return False
            return True

        # Exponential search to bracket a valid end time
        candidate = 1.0
        while not _is_at_steady_state(candidate):
            candidate *= 2
            if candidate > 1e9:
                return None

        # Binary search for the shortest valid end time
        floor = candidate / 2
        ceiling = candidate
        while ceiling - floor > 1e-8 * floor and ceiling < 1e8:
            test_time = floor + (ceiling - floor) / 2
            if _is_at_steady_state(test_time):
                floor = test_time
                candidate = test_time
            else:
                ceiling = test_time

        return float(candidate)

    def detect_cv_maximized(self) -> Optional[float]:
        """Find the end time that maximises the median coefficient of variation.

        Uses :class:`Simulator` to run each candidate simulation and computes
        the median CV from the returned timecourse data.

        Returns
        -------
        float or None
            The detected end time, or ``None`` if optimisation fails.
        """
        rr = te.loadSBMLModel(self.model.sbml_str)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', self.MAX_ITERATOR_STEP)

        if len(rr.getFloatingSpeciesIds()) == 0:
            return None

        species_names = list(rr.getFloatingSpeciesIds())

        def _negative_median_cv(log_end_time: float) -> float:
            end_t = 10.0 ** log_end_time
            try:
                sim_result = self._run_simulation(end_time=end_t)
            except Exception:
                return 0.0

            timecourse_df = sim_result.timecourse_df
            cvs = []
            for name in species_names:
                if name not in timecourse_df.columns:
                    continue
                col_data = timecourse_df[name].values
                mean_val = float(np.mean(np.abs(col_data)))
                if mean_val < 1e-10:
                    continue
                cvs.append(float(np.std(col_data) / mean_val))

            return -float(np.median(cvs)) if cvs else 0.0

        opt = minimize_scalar(
            _negative_median_cv,
            bounds=(self.LOG_LOWER, self.LOG_UPPER),
            method="bounded",
        )
        return float(10.0 ** opt.x) if opt.fun < 0 else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_simulation(self, end_time: float) -> "SimulationResult":  # type: ignore 
        """Run a single simulation via :class:`Simulator`.

        Parameters
        ----------
        end_time : float
            End time for the simulation.
        perturbation : bool
            If ``True``, use default perturbation settings.  If ``False``, run
            with zero perturbation (useful for steady-state detection).

        Returns
        -------
        SimulationResult
        """
        simulator = Simulator(
            model=self.model,
            start_time=self.start_time,
            end_time=end_time,
            num_point=self.num_point,
        )
        return simulator.simulate()