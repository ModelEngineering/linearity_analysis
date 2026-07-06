"""Represents the dynamic (simulation-derived) properties of a model."""

import src.constants as cn  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes  # type: ignore
from src.model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
import tellurium as te  # type: ignore
from typing import List, Optional, Tuple

MAX_ITERATOR_STEP = 50 * int(1e6)


class Trajectory(object):
    """Simulation-derived properties of a model.

    Stores Jacobians, forcing inputs, and the timecourse obtained from a
    single simulation run.  Use makeFromSimulation() to create an instance
    by running a simulation.
    """

    def __init__(self,
            model: Model,
            jacobian_collection_arr: np.ndarray,
            timepoint_arr: np.ndarray,
            forcing_input_collection_arr: np.ndarray,
            timecourse_df: pd.DataFrame,
            end_time_source: str = "") -> None:
        """
        Parameters
        ----------
        model : Model
        jacobian_collection_arr : np.ndarray
            Shape (num_point, n_species, n_species).
        timepoint_arr : np.ndarray
            Shape (num_point,), sorted ascending.
        forcing_input_collection_arr : np.ndarray
            Shape (num_point, n_species).
        timecourse_df : pd.DataFrame
            Time-indexed, columns are species names.
        """
        if len(timepoint_arr) == 0:
            raise ValueError("timepoint_arr must not be empty.")
        self.model = model
        self.jacobian_collection_arr = jacobian_collection_arr
        self.timepoint_arr = timepoint_arr
        self.forcing_input_collection_arr = forcing_input_collection_arr
        self.timecourse_df = timecourse_df
        self.end_time_source = end_time_source
        self._jacobian_median_arr: np.ndarray = np.array([])
        self._jacobian_std_arr: np.ndarray = np.array([])

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.end_time <= other.start_time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (self.model == other.model
                and np.array_equal(self.jacobian_collection_arr, other.jacobian_collection_arr)
                and np.array_equal(self.timepoint_arr, other.timepoint_arr)
                and np.array_equal(self.forcing_input_collection_arr, other.forcing_input_collection_arr)
                and self.timecourse_df.equals(other.timecourse_df)
                and self.end_time_source == other.end_time_source)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def start_time(self) -> float:
        """First timepoint."""
        return float(self.timepoint_arr[0])

    @property
    def end_time(self) -> float:
        """Last timepoint."""
        return float(self.timepoint_arr[-1])

    @property
    def num_point(self) -> int:
        """Number of timepoints."""
        return len(self.timepoint_arr)

    @property
    def jacobian_median_arr(self) -> np.ndarray:
        """Element-wise median Jacobian across all timepoints."""
        if self._jacobian_median_arr.size == 0:
            self._jacobian_median_arr = np.median(self.jacobian_collection_arr, axis=0)
        return self._jacobian_median_arr

    @property
    def jacobian_std_arr(self) -> np.ndarray:
        """Element-wise standard deviation of Jacobians across all timepoints."""
        if self._jacobian_std_arr.size == 0:
            self._jacobian_std_arr = np.std(self.jacobian_collection_arr, axis=0)
        return self._jacobian_std_arr

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def makeSubmodel(self, start_time: float, end_time: float) -> "Trajectory":
        """Construct a Trajectory for the interval [start_time, end_time] by slicing.

        Parameters
        ----------
        start_time : float
        end_time : float

        Returns
        -------
        Trajectory
        """
        mask = (self.timepoint_arr >= start_time) & (self.timepoint_arr <= end_time)
        tc_mask = (self.timecourse_df.index >= start_time) & (
                self.timecourse_df.index <= end_time)
        return Trajectory(
                model=self.model,
                jacobian_collection_arr=self.jacobian_collection_arr[mask],
                timepoint_arr=self.timepoint_arr[mask],
                forcing_input_collection_arr=self.forcing_input_collection_arr[mask],
                timecourse_df=self.timecourse_df.loc[tc_mask],
                end_time_source=self.end_time_source,
        )

    def plotTimecourse(self, **kwargs) -> PlotOptions:
        """Plot the simulated timecourse for all species.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions. Supported keys: ax, fig, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.

        Returns
        -------
        PlotOptions
        """
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        for i, name in enumerate(self.model.species_names):
            ax.plot(  # type: ignore
                    self.timecourse_df.index,
                    self.timecourse_df[name],
                    color=f"C{i}",
                    label=name,
            )
        plot_options.apply()
        return plot_options

    @classmethod
    def makeBiomodel(cls,
            model_num: int,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_point: int = cn.NUM_POINT) -> "Trajectory":
        """Create a Trajectory from a BioModels entry by number.

        Parameters
        ----------
        model_num : int
            BioModel number, e.g. 8. Formatted to 'BIOMD0000000008'.
        start_time : float
            Passed to makeFromSimulation.
        end_time : Optional[float]
            Passed to makeFromSimulation. None triggers auto-detection.
        num_point : int
            Passed to makeFromSimulation.

        Returns
        -------
        Trajectory
        """
        model_name = f"BIOMD{model_num:010d}"
        model = Model.makeBiomodel(model_name)
        return cls.makeFromSimulation(
                model,
                start_time=start_time,
                end_time=end_time,
                num_point=num_point,
        )
    
    @classmethod
    def makeTimecourse(cls,
            model: Model,
            perturbation: float = 0,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = 10.0,
            num_point: int = cn.NUM_POINT) -> pd.DataFrame:

        """Create a timecourse DataFrame by running a simulation with a perturbation
        in non-zero initial values.

        Parameters
        ----------
        model : Model
        perturbation : float
            fraction by which to perturb each initial value, e.g. 0.1 for +10%
        start_time : float
        end_time : Optional[float]
            None triggers auto-detection.
        num_point : int

        Returns
        -------
        pd.DataFrame
            index: time, 
            columns: species names, 
            values: concentrations.
    """
        end_time, _ = cls._getEndtime(end_time, model.model_name)
        rr = te.loadSBMLModel(model.sbml_str) 
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', MAX_ITERATOR_STEP)
        for species_name in model.species_names:
            init_val = rr.getValue(f'init({species_name})')
            perturbed_val = init_val * (1 + perturbation)
            rr.setValue(f'init({species_name})', perturbed_val)
        rr.reset()
        # Do the simulation
        if start_time > 0:
            # Warm up needed
            rr.simulate(0, start_time, 2)
        data = rr.simulate(start_time, end_time, num_point)
        return pd.DataFrame(
                data[:, 1:],
                index=data[:, 0],
                columns=model.species_names,
        )
    
    @staticmethod
    def _getEndtime(end_time: Optional[float], model_name: str) \
            -> Tuple[Optional[float], str]:
        """Determine the end time and its source."""
        if end_time is not None:
            return end_time, cn.ENDTIME_SOURCE_USER_SPECIFIED
        if model_name.startswith("BIOMD"):
            endtime_dct = getBiomodelsEndtimes()
            csv_end_time = endtime_dct.get(model_name, None)
            if csv_end_time is not None:
                return csv_end_time, cn.ENDTIME_SOURCE_SEDML
        return None, ""
    
    @classmethod
    def makeFromSimulation(cls,
            model: Model,
            start_time: float = cn.START_TIME,
            end_time: Optional[float] = None,
            num_point: int = cn.NUM_POINT) -> "Trajectory":
        """Create a Trajectory by running a simulation.

        This is the only method that uses RoadRunner.

        end_time resolution order:
            1. Caller-supplied value (source: user_specified).
            2. BioModels CSV lookup (source: sedml).
            3. Auto-detection via _makeEndtime (source: set by that method).

        Parameters
        ----------
        model : Model
        start_time : float
        end_time : Optional[float]
            None triggers auto-detection.
        num_point : int

        Returns
        -------
        Trajectory
        """
        rr = te.loadSBMLModel(model.sbml_str)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', MAX_ITERATOR_STEP)

        # Resolve end_time and record its source
        end_time, end_time_source = cls._getEndtime(end_time, model.model_name)

        # Timecourse simulation
        rr.reset()
        if start_time > 0:
            rr.simulate(0, start_time, 2)
        try:
            result_arr = np.array(rr.simulate(start_time, end_time, num_point))
        except Exception as e:
            raise ValueError(f"Simulation failed: {e}")
        timepoint_arr = result_arr[:, 0]
        timecourse_df = pd.DataFrame(
                result_arr[:, 1:],
                index=timepoint_arr,
                columns=model.species_names,
        )
        timecourse_df.index.name = "time"

        # Initial values used for forcing input computation
        initial_values = timecourse_df.iloc[0].values

        # Step-by-step simulation to collect Jacobians and forcing inputs
        rr.reset()
        if start_time > 0:
            rr.simulate(0, start_time, 2)
        jacobian_collection: List[np.ndarray] = []
        forcing_input_collection: List[np.ndarray] = []
        for i, t in enumerate(timepoint_arr):
            if i == 0:
                rr.simulate(start_time, start_time + 1e-10, 2)
            else:
                rr.simulate(timepoint_arr[i - 1], t, 2)
            jacobian_arr = np.array(rr.getFullJacobian()).copy()
            if np.all(np.isclose(jacobian_arr, 0.0)):
                raise ValueError(
                        f"Jacobian at t={t} is all zeros; model may be degenerate.")
            f_arr = np.array(rr.getRatesOfChange())
            jacobian_collection.append(jacobian_arr)
            forcing_input_collection.append(f_arr - jacobian_arr @ initial_values)

        return cls(
                model=model,
                jacobian_collection_arr=np.array(jacobian_collection),
                timepoint_arr=timepoint_arr,
                forcing_input_collection_arr=np.array(forcing_input_collection),
                timecourse_df=timecourse_df,
                end_time_source=end_time_source
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @classmethod
    def _makeEndtime(cls,
            rr: "te.roadrunner.ExtendedRoadRunner",  # type: ignore
            start_time: float,
            num_point: int) -> Tuple[float, str]:
        """Auto-detect a suitable simulation end time.

        Tries steady-state search first, then median-CV maximisation.
        Raises ValueError if neither succeeds.
        """
        end_time_source = ""
        end_time = cls._makeEndtimeSteadystate(rr, start_time)
        if not np.isnan(end_time):
            end_time_source = cn.ENDTIME_SOURCE_STEADYSTATE
            return end_time, end_time_source
        end_time = cls._makeEndtimeCV(rr, start_time, num_point)
        if not np.isnan(end_time):
            end_time_source = cn.ENDTIME_SOURCE_MAX_MEDIAN_CV
            return end_time, end_time_source
        raise ValueError(
                "Could not determine an appropriate end time. "
                "The model may be invalid or unbounded.")

    @classmethod
    def _makeEndtimeSteadystate(cls,
            rr: "te.roadrunner.ExtendedRoadRunner",  # type: ignore
            start_time: float) -> float:
        """Find the shortest end time at which the model reaches steady state."""
        THRESHOLD = 0.01
        rr.reset()
        try:
            solver = rr.getSteadyStateSolver()
            for k, v in {"allow_approx": True, "approx_tolerance": 1e-3,
                    "relative_tolerance": 1e-3, "maximum_iterations": 1000}.items():
                solver.setValue(k, v)
            rr.steadyState()
        except RuntimeError:
            return np.nan
        ss_arr = np.array([max(v, 1e-8) for v in rr.getFloatingSpeciesConcentrations()])
        if len(ss_arr) == 0 or np.any(np.isnan(ss_arr)) or np.any(np.isinf(ss_arr)):
            return np.nan

        rr.reset()

        def _isAtSteadyState(end_t: float) -> bool:
            rr.reset()
            try:
                final_arr = rr.simulate(start_time, end_t, 2)[:, 1:]
            except Exception:
                return False
            if all(np.isclose(ss_arr, 0.0)) and all(np.isclose(final_arr[1], 0.0)):
                return True
            try:
                normalized_arr = final_arr[1] / ss_arr
            except Exception:
                return False
            return bool(np.max(np.abs(normalized_arr - 1)) < THRESHOLD)

        end_time: float = 1.0
        while not _isAtSteadyState(end_time):
            end_time *= 2
            if end_time > 1e9:
                return np.nan

        floor = end_time / 2
        ceiling = end_time
        while ceiling > 1e-8 and ceiling < 1e8:
            test_time = floor + (ceiling - floor) / 2
            if _isAtSteadyState(test_time):
                floor = test_time
                end_time = test_time
            else:
                ceiling = test_time
            if 1 - floor / ceiling < THRESHOLD:
                break
        return end_time

    @classmethod
    def _makeEndtimeCV(cls,
            rr: "te.roadrunner.ExtendedRoadRunner",  # type: ignore
            start_time: float,
            num_point: int) -> float:
        """Find the end time that maximises the median CV of the timecourse."""
        LOG_LOWER = -5.0
        LOG_UPPER = 6.0
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', MAX_ITERATOR_STEP)
        if len(rr.getFloatingSpeciesIds()) == 0:
            return np.nan

        def _negativeMedianCV(log_end_time: float) -> float:
            end_t = 10.0 ** log_end_time
            rr.reset()
            try:
                result_arr = np.array(rr.simulate(start_time, end_t, num_point))
            except Exception:
                return 0.0
            data_arr = result_arr[:, 1:]
            cvs = []
            for col in range(data_arr.shape[1]):
                col_data = data_arr[:, col]
                mean_val = float(np.mean(np.abs(col_data)))
                if mean_val < 1e-10:
                    continue
                cvs.append(float(np.std(col_data) / mean_val))
            return -float(np.median(cvs)) if cvs else 0.0

        opt = minimize_scalar(_negativeMedianCV, bounds=(LOG_LOWER, LOG_UPPER),
                method="bounded")
        return float(10.0 ** opt.x) if opt.fun < 0 else np.nan
