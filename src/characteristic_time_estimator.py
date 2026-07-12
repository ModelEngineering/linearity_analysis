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
import multiprocessing as mp
from multiprocessing.connection import Connection
import numpy as np  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from typing import Optional, Tuple, Any, List

# Timeout for running steady state estimation
TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

CharacteristicTimeResult = Tuple[float, str]
"""A pair of ``(end_time, end_time_source)``."""


# ---------------------------------------------------------------------------
# SteadyStateEstimator
# ---------------------------------------------------------------------------

class SteadystateEstimator:

    def __init__(self, model: Model, target, start_time: float,
            num_point: int, timeout: float = TIMEOUT) -> None:
        """
        Args:
            model (Model): Model being analyzed.
            target: Module-level callable run in the child process, with
                signature (model, start_time, num_point, conn) -> None. It
                must send its result via conn.send(...) and then conn.close().
            start_time (float)
            num_point (int)
            timeout (float): Timeout for the steady-state estimation.
        """
        self.model = model
        self.start_time = start_time
        self.num_point = num_point
        self.timeout = timeout
        self.target = target

    def estimate(self, timeout: float = -1.0) -> float | None:
        """Runs `self.target` in a subprocess so it can be killed at the OS
        level if it hangs.

        `target` (here, `_run_calculate_steadystate`) calls into RoadRunner's
        compiled integrator; a single call can run for an unbounded time for
        models that never settle (e.g. oscillators), and CPython cannot
        preempt time spent inside a C extension call via `signal.alarm` --
        pending signals are only delivered once control returns to the
        bytecode loop. Running the work in a separate OS process sidesteps
        this: `terminate()`/`kill()` act on the process itself, independent
        of what code it is currently executing.

        Returns
        -------
        float or None
            Whatever `target` sent back over the connection, or -1 if it did
            not finish within `timeout` seconds.
        """
        if timeout < 0:
            timeout = self.timeout
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=self.target,
            args=(self.model, self.start_time, self.num_point, child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()  # parent only reads; drop its copy of the write end

        if parent_conn.poll(timeout):
            result = parent_conn.recv()
        else:
            result = -1

        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join()
        parent_conn.close()
        return result


# ---------------------------------------------------------------------------
# Subprocess worker for steady-state detection
# ---------------------------------------------------------------------------

def _run_calculate_steadystate(
        model: Model, start_time: float, num_point: int, conn: Connection) -> None:
    """Runs in a child process: computes the steady-state end time and sends
    it back over `conn`.

    Defined at module level (not as a method) so it can be pickled and
    re-imported by the 'spawn' multiprocessing start method -- a bound method
    would instead require pickling the enclosing instance.
    """
    try:
        estimator = CharacteristicTimeEstimator(
            model=model, start_time=start_time, num_point=num_point)
        result = estimator._calculate_steadystate()
    except Exception:
        result = None
    conn.send(result)
    conn.close()


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
            timeout: float = TIMEOUT,
    ) -> None:
        self.model = model
        self.start_time = start_time
        self.num_point = num_point
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def estimate(cls,
            model: Model,
            end_time: Optional[float] = None,
            start_time: float = cn.START_TIME,
            num_point: int = cn.NUM_POINT,
            timeout: float = TIMEOUT) -> CharacteristicTimeResult:
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
            timeout=timeout,
        )
        end_time = estimator.detect_steadystate()
        if end_time is not None:
            return float(end_time), cn.ENDTIME_SOURCE_STEADYSTATE

        # 4. Maximum median CV optimisation
        estimator = cls(
            model=model,
            start_time=start_time,
            num_point=num_point,
            timeout=timeout,
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

    def detect_steadystate(self, timeout: float = -1.0) -> float | None:
        """Runs steady-state detection in a subprocess so it can be killed at
        the OS level if it hangs.

        `_calculate_steadystate` calls into RoadRunner's compiled integrator;
        a single call can run for an unbounded time for models that never
        settle (e.g. oscillators), and CPython cannot preempt time spent
        inside a C extension call via `signal.alarm` -- pending signals are
        only delivered once control returns to the bytecode loop. Running
        the work in a separate OS process sidesteps this: `terminate()`/
        `kill()` act on the process itself, independent of what code it is
        currently executing.

        The timeout should be set to a value that is long enough for 
        the model to reach steady state. Note that there is an overhead
        of ~1.2s that is unavoidable becausethe child process re-imports
        the module's dependencies (numpy, scipy, matplotlib, roadrunner)
        under the 'spawn' start method, on top of actual computation time.

        Returns
        -------
        float or None
            The detected end time, `None` if steady state could not be
            determined, or -1 if detection did not finish within `timeout`
            seconds.
        """
        if timeout < 0:
            timeout = self.timeout
        ss_estimator = SteadystateEstimator(
            self.model, _run_calculate_steadystate,
            start_time=self.start_time, num_point=self.num_point,
        )
        result = ss_estimator.estimate(timeout=timeout)
        return result

    def _calculate_steadystate(self) -> Optional[float]:
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
    
    def plotStdNrml(self, end_time: float, 
            ax_top=None, ax_mid=None, ax_bot=None,
            is_label: bool = True,
            **plt_kwargs: Any
            ) -> List[PlotOptions | None]:
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
        ax_top : matplotlib.axes.Axes, optional
            Axes object for the top panel.
        ax_mid : matplotlib.axes.Axes, optional
            Axes object for the middle panel.
        ax_bot : matplotlib.axes.Axes, optional
            Axes object for the bottom panel.
        is_label : bool, default=True
            Whether to include labels on the plots.
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
        plt_kwargs = dict(plt_kwargs)  # copy so we can pop without side effects
        timecourse_df = self._run_simulation(end_time=end_time).timecourse_df
        normalized_df = (timecourse_df - timecourse_df.mean()) / timecourse_df.std()
        normalized_df = normalized_df.fillna(0)  # Handle any NaN values resulting from std=0
        metric_arr = normalized_df.std(axis=1).values
        species_names = list(timecourse_df.columns)

        figsize = plt_kwargs.pop("figsize", (10, 12))
        suptitle = plt_kwargs.pop("title", None)
        plt_kwargs.pop("ylabel", None)
        model_name = plt_kwargs.get("model_name", "")

        fig = None
        if ax_top is None and ax_mid is None and ax_bot is None:
            fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        if is_label:
            plt_kwargs.setdefault("legend", True)
            plt_kwargs.setdefault("xlabel", "Time")
        else:
            plt_kwargs.setdefault("legend", False)
            plt_kwargs.setdefault("xlabel", "")
            plt_kwargs.setdefault("ylabel", "")
            plt_kwargs.setdefault("legend", False)

        # Top panel: raw timecourse.
        top_po = None
        if ax_top is not None:
            top_po = PlotOptions(fig=fig, ax=ax_top, title="Timecourse", **plt_kwargs)
            if is_label:
                top_po.ylabel = "Concentration"
            for idx, name in enumerate(species_names):
                ax_top.plot(timecourse_df.index, timecourse_df[name],  # type: ignore
                        color=f"C{idx}", label=name)
            top_po.apply()
            if not is_label:
                ax_top.set_xticks([])
                ax_top.set_yticks([])

        # Middle panel: per-species normalized timecourse.
        mid_po = None
        if ax_mid is not None:
            mid_po = PlotOptions(fig=fig, ax=ax_mid, title="Standardized Timecourse", **plt_kwargs)
            if is_label:
                mid_po.ylabel = "Normalized Value"
            for idx, name in enumerate(species_names):
                ax_mid.plot(normalized_df.index, normalized_df[name],  # type: ignore
                        color=f"C{idx}", label=name)
            mid_po.apply()
            if not is_label:
                ax_mid.set_xticks([])
                ax_mid.set_yticks([])

        # Bottom panel: cross-species std of normalized values (single, unlabeled line).
        bot_po = None
        if ax_bot is not None:
            if is_label:
                title = "Cross-Species Std of Normalized Values"
            else:
                title = ""
            plt_kwargs.update(title=title)
            bot_po = PlotOptions(
                fig=fig, ax=ax_bot, **plt_kwargs)
            ax_bot.plot(timecourse_df.index, metric_arr)  # type: ignore
            bot_po.apply()
            if not is_label:
                ax_bot.set_xticks([])
                ax_bot.set_yticks([])

        if suptitle is not None and fig is not None:
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
    
    def plotComparison(self,
            ax_sb=None,
            ax_ss=None,
            ax_mc=None,
            timeout: float = TIMEOUT,
            **plt_kwargs: Any
            ) -> List[PlotOptions]:
        """Compare the three ways an end time can be calculated: SEDML lookup,
        steady-state detection, and maximum-median-CV optimisation.

        Each panel plots the normalized timecourse (one line per species) on
        its own end time -- computed by the corresponding method in this
        class -- against the left vertical axis, and the cross-species
        normalized standard deviation ("the metric", see
        :meth:`plotStdNrml`) against a twin right vertical axis. If a method
        fails to produce an end time, its panel is left blank with the text
        "None" centered in it.

        Parameters
        ----------
        ax_sb : matplotlib.axes.Axes, optional
            Axes object for the SEDML subplot panel.
        ax_ss : matplotlib.axes.Axes, optional
            Axes object for the steady-state panel.
        ax_mc : matplotlib.axes.Axes, optional
            Axes object for the model maximum cv panel.
        timeout : float
            Timeout for the steady-state estimation.
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
            One PlotOptions per panel, in order [SEDML, steady-state,
            model maximum cv].
        """
        plt_kwargs = dict(plt_kwargs)  # copy so we can pop without side effects
        figsize = plt_kwargs.pop("figsize", (10, 12))
        suptitle = plt_kwargs.pop("title", None)
        plt_kwargs.pop("ylabel", None)
        plt_kwargs.setdefault("legend", False)
        plt_kwargs.setdefault("xlabel", "")
        plt_kwargs.setdefault("ylabel", "")
        model_name = plt_kwargs.get("model_name", "")

        fig = None
        if ax_sb is None and ax_ss is None and ax_mc is None:
            fig, (ax_sb, ax_ss, ax_mc) = plt.subplots(3, 1, figsize=figsize)

        sedml_end_time = getBiomodelsEndtimes().get(self.model.model_name, None)
        if sedml_end_time is not None:
            sedml_end_time = float(sedml_end_time)

        steadystate_end_time = self.detect_steadystate(timeout=timeout)
        if steadystate_end_time is not None and steadystate_end_time < 0:
            steadystate_end_time = None  # -1 sentinel means detection timed out

        max_cv_end_time = self.detect_cv_maximized()

        panels = [
            (ax_sb, sedml_end_time, "SEDML"),
            (ax_ss, steadystate_end_time, "Steady State"),
            (ax_mc, max_cv_end_time, "Maximum CV"),
        ]

        plot_options_list: List[PlotOptions] = []
        for ax, end_time, title in panels:
            po = PlotOptions(fig=fig, ax=ax, title=title, **plt_kwargs)
            if end_time is None:
                po.apply()
                ax.set_xticks([])  # type: ignore
                ax.set_yticks([])  # type: ignore
                ax.text(0.5, 0.5, "None", ha="center", va="center",  # type: ignore
                        transform=ax.transAxes)  # type: ignore
            else:
                timecourse_df = self._run_simulation(end_time=end_time).timecourse_df
                normalized_df = (timecourse_df - timecourse_df.mean()) / timecourse_df.std()
                normalized_df = normalized_df.fillna(0)
                metric_arr = normalized_df.std(axis=1).values
                for idx, name in enumerate(timecourse_df.columns):
                    ax.plot(normalized_df.index, normalized_df[name],  # type: ignore
                            color=f"C{idx}", label=name)
                po.apply()
                ax_twin = ax.twinx() # type: ignore
                ax_twin.plot(timecourse_df.index, metric_arr, color="black", linestyle="--")  # type: ignore
                ax.set_xticks([end_time])  # type: ignore
                ax.set_yticks([])  # type: ignore
                metric_max = float(np.max(metric_arr)) if len(metric_arr) else 0.0
                ax_twin.set_yticks([metric_max])
            plot_options_list.append(po)

        if suptitle is not None and fig is not None:
            full_title = f"{model_name}: {suptitle}" if model_name else suptitle
            fig.suptitle(full_title, fontsize=13, fontweight="bold")
            fig.tight_layout()

        return plot_options_list