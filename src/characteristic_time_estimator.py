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
from src.plot_options import PlotOptions  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from typing import Optional, Tuple, Any, List


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

        Uses :class:`Simulator` to obtain reference steady-state values, then
        uses :class:`Simulator` again in a binary-search loop to find the
        earliest simulation end time that converges to those values.

        Returns
        -------
        float or None
            The detected end time, or ``None`` if steady state cannot be found.
        """
        # --- Step 1: obtain reference steady-state values via Simulator ---
        steadystate_simulator = Simulator(
            model=self.model,
            start_time=self.start_time,
            num_point=self.num_point,
        )
        raw_ss = steadystate_simulator.getSteadyState()
        if raw_ss is None:
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
        if self.model.num_species == 0:
            return None

        species_names = self.model.species_names

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
        return float(10.0 ** opt.x) if opt.fun < 0 else None  # type: ignore
    
    def plotStdNrml(self, end_time: float, **plt_kwargs: Any) -> List[PlotOptions]:
        """Three-panel plot of the timecourse, its per-species normalization,
        and the cross-species standard deviation of the normalized values.

        Top panel: `timecourse_df` (one line per species, original units).
        Middle panel: the same timecourse with each species column
            normalized to zero mean and unit standard deviation (columns
            with zero variance become all-zero after normalization).
        Bottom panel: the standard deviation, across species, of the
            normalized values at each time point -- a high value indicates
            the species trajectories have diverged from one another at
            that time.

        Parameters
        ----------
        end_time : float
            End time for the simulation.
        **plt_kwargs
            Forwarded to PlotOptions for every panel (xlabel, legend, xlim,
            ylim, model_name). ``ylabel`` is set per panel and is not
            forwarded. ``title`` (if given) becomes the overall figure
            title, prefixed with ``model_name`` as in
            :meth:`PlotOptions.apply`. ``figsize`` is consumed here
            (default ``(10, 12)``). ``fig``/``ax`` are managed internally
            (one figure with three panels) and must not be passed.

        Returns
        -------
        List[PlotOptions]
            One PlotOptions per panel, in order [timecourse, standardized,
            metric].
        """
        timecourse_df = self._run_simulation(end_time=end_time).timecourse_df
        normalized_df = (timecourse_df - timecourse_df.mean()) / timecourse_df.std()
        normalized_df = normalized_df.fillna(0)  # Handle any NaN values resulting from std=0
        metric_arr = normalized_df.std(axis=1).values
        species_names = list(timecourse_df.columns)

        figsize = plt_kwargs.pop("figsize", (10, 12))
        suptitle = plt_kwargs.pop("title", None)
        plt_kwargs.pop("ylabel", None)
        plt_kwargs.setdefault("xlabel", "Time")
        model_name = plt_kwargs.get("model_name", "")

        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Top panel: raw timecourse.
        top_po = PlotOptions(fig=fig, ax=ax_top, **plt_kwargs)
        top_po.ylabel = "Concentration"
        for idx, name in enumerate(species_names):
            ax_top.plot(timecourse_df.index, timecourse_df[name],  # type: ignore
                    color=f"C{idx}", label=name)
        ax_top.set_title("Timecourse")  # type: ignore
        top_po.apply()

        # Middle panel: per-species normalized timecourse.
        mid_po = PlotOptions(fig=fig, ax=ax_mid, **plt_kwargs)
        mid_po.ylabel = "Normalized Value"
        for idx, name in enumerate(species_names):
            ax_mid.plot(normalized_df.index, normalized_df[name],  # type: ignore
                    color=f"C{idx}", label=name)
        ax_mid.set_title("Standardized Timecourse")  # type: ignore
        mid_po.apply()

        # Bottom panel: cross-species std of normalized values (single, unlabeled line).
        plt_kwargs.setdefault("legend", False)
        bot_po = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        bot_po.ylabel = "Standard Deviation of Normalized Values"
        ax_bot.plot(timecourse_df.index, metric_arr)  # type: ignore
        ax_bot.set_title("Standard Deviation Across Species")  # type: ignore
        bot_po.apply()

        if suptitle is not None:
            full_title = f"{model_name}: {suptitle}" if model_name else suptitle
            fig.suptitle(full_title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        return [top_po, mid_po, bot_po]

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