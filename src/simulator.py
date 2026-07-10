"""Simulates a model over a time course and collects Jacobians."""

import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes

from collections import namedtuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
from typing import Dict, List, Optional

MAX_ITERATOR_STEP = 50 * int(1e6)

SimulationResult = namedtuple('SimulationResult',
        ['timecourse_df', 'jacobian_collection_arr'])


class Simulator(object):
    """Simulates a model over a fixed time range.

    Encapsulates RoadRunner simulation, initial value setting, species name
    validation, and perturbation of initial values.
    """

    MAX_ITERATOR_STEP = 50 * int(1e6)

    def __init__(self,
        model: Model,
        start_time: float,
        end_time: Optional[float] = None,
        num_point: int = cn.NUM_POINT,
        perturbation_value_fraction: float = cn.PERTURBATION_VALUE_FRACTION,
        perturbation_species_fraction: float = cn.PERTURBATION_SPECIES_FRACTION
        ) -> None:
        """
        Parameters
        ----------
        model : Model
            The model to simulate.
        start_time : float
            Start time of the simulation.
        end_time : float, optional
            End time of the simulation. Required by :meth:`simulate`; may be
            omitted when only calling :meth:`getSteadyState`.
        num_point : int
            Number of time points to simulate.
        perturbation_value_fraction : float
            Amount of perturbation of initial values as a fraction of the original value.
            May be positive or negative.
        perturbation_species_fraction : float
            Fraction of non-zero initial values that are perturbed.
        """
        self.model = model
        self.start_time = start_time
        self.end_time = end_time
        self.num_point = num_point
        self.perturbation_value_fraction = perturbation_value_fraction
        self.perturbation_species_fraction = perturbation_species_fraction

    @classmethod
    def simulateBiomodel(cls, model_num: int, end_time: float=-1.0,
            start_time: float = cn.START_TIME,
            num_point: int = cn.NUM_POINT,
            is_jacobian_collection: bool = True) -> SimulationResult:
        """
        Simulates the model.

        Args:
            model_num (int): number of the BioModel to simulate.
            end_time (float): end time of the simulation. If -1, use the model's default.

        Returns:
            SimulationResult: _description_
        """
        model = Model.makeBiomodel(model_num=model_num)
        if end_time < 0:
            end_time_dct = getBiomodelsEndtimes()
            if end_time_dct.get(model.model_name, None) is None:
                raise ValueError(f"End time for model {model.model_name} not found.")
            end_time = end_time_dct[model.model_name]
        simulator = cls(
            model=model,
            start_time=start_time,
            end_time=end_time,
            num_point=num_point,
        )
        return simulator.simulate(is_jacobian_collection=is_jacobian_collection)

    def simulate(self, is_jacobian_collection: bool = False) -> SimulationResult:
        """Run a simulation and optionally collect Jacobians.

        This is the only method that uses RoadRunner for time-course integration.

        Parameters
        ----------
        is_jacobian_collection : bool
            Whether to collect Jacobians at each time point.

        Returns
        -------
        SimulationResult
        """
        if not isinstance(self.end_time, (int, float)):
            raise ValueError("end_time must be a number (int or float) to simulate.")
        rr, initial_dct = self._makeRoadRunner()

        # Pre-start simulation if start_time > 0
        if self.start_time > 0:
            rr.simulate(0, self.start_time, 2)

        # Batch timecourse simulation
        try:
            rr_result = rr.simulate(self.start_time, self.end_time, self.num_point)
        except Exception as e:
            raise ValueError(f"Simulation failed: {e}")

        # Check column order before converting to ndarray (colnames lost after np.array).
        # Skip the leading 'time' column and strip brackets from species names.
        result_species = [
                c[1:-1] if c.startswith("[") and c.endswith("]") else c
                for c in rr_result.colnames[1:]  # type: ignore
        ]
        self._checkSpeciesNames(result_species)
        result_arr = np.array(rr_result)
        timepoint_arr = result_arr[:, 0]
        timecourse_df = pd.DataFrame(
                result_arr[:, 1:],
                index=timepoint_arr,
                columns=self.model.species_names,
        )
        timecourse_df.index.name = "time"

        # Step-by-step simulation to collect Jacobians and forcing inputs
        if is_jacobian_collection:
            rr.reset()
            self._setInitialValues(rr, initial_dct)
            if self.start_time > 0:
                rr.simulate(0, self.start_time, 2)
            jacobian_collection: List[np.ndarray] = []
            for i, t in enumerate(timepoint_arr):
                if i == 0:
                    rr.simulate(self.start_time, self.start_time + 1e-10, 2)
                else:
                    rr.simulate(timepoint_arr[i - 1], t, 2)
                jacobian_arr = rr.getFullJacobian()
                self._checkSpeciesNames(jacobian_arr.rownames)
                self._checkSpeciesNames(jacobian_arr.colnames)
                jacobian_arr = np.array(jacobian_arr).copy()
                if np.all(np.isclose(jacobian_arr, 0.0)):
                    raise ValueError(
                            f"Jacobian at t={t} is all zeros; model may be degenerate.")
                jacobian_collection.append(jacobian_arr)
        else:
            jacobian_collection = []

        return SimulationResult(
                jacobian_collection_arr=np.array(jacobian_collection),
                timecourse_df=timecourse_df,
        )

    def getSteadyState(self) -> Optional[np.ndarray]:
        """Compute floating species concentrations at steady state.

        Uses RoadRunner's built-in steady-state solver with tolerant settings
        suitable for approximate convergence. Does not require `end_time`.

        Returns
        -------
        np.ndarray or None
            Concentrations in `self.model.species_names` order, or ``None``
            if steady state could not be found, or is degenerate (empty,
            NaN, or infinite).
        """
        rr, _ = self._makeRoadRunner()
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
        return raw_ss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _makeRoadRunner(self):
        """Load a RoadRunner instance configured with perturbed initial values.

        Returns
        -------
        tuple
            ``(rr, initial_dct)`` — the configured RoadRunner instance and the
            perturbed initial values that were applied.
        """
        rr = te.loadSBMLModel(self.model.sbml_str)
        rr.reset()
        rr.integrator.setValue('maximum_num_steps', self.MAX_ITERATOR_STEP)
        initial_dct = self._getPerturbedInitialValues()
        self._setInitialValues(rr, initial_dct)
        return rr, initial_dct

    def _checkSpeciesNames(self, names: List[str]) -> None:
        """Check that the species names in the simulation result match the model."""
        result_species = list(names)
        if result_species != self.model.species_names:
            raise ValueError(
                    f"Simulation species {result_species} do not match "
                    f"model species {self.model.species_names}.")

    def _setInitialValues(self, rr, initial_dct: Dict[str, float]) -> None:
        """Set initial values of floating species in the RoadRunner model."""
        for idx, name in enumerate(self.model.species_names):
            if name in initial_dct.keys():
                rr[name] = initial_dct[name]

    def _getPerturbedInitialValues(self) -> Dict[str, float]:
        """Perturb initial values of floating species by randomly selecting a fraction."""
        dct: Dict[str, float] = {}
        num_species = self.model.num_species
        num_perturb = int(self.perturbation_species_fraction * num_species)
        if num_perturb <= 0:
            return dct
        perturb_indices = np.random.choice(num_species, size=num_perturb, replace=False)
        for idx in perturb_indices:
            species_name = self.model.species_names[idx]
            try:
                original_value = self.model.initial_value_dct[species_name]
                perturbation = self.perturbation_value_fraction * original_value
                dct[species_name] = max(0.0, original_value + perturbation)
            except Exception:
                continue
        return dct