"""
Creates a JacobianCollection. Can be used to reinstatiate a RoadRunner instance if it was deleted after initialization.

Usage::

    maker = JacobianCollectionMaker(model_string)
    jacobian_collection = maker.makeCollection()
"""
import src.constants as cn
from trajectory import Trajectory  # type: ignore
from src.deprecated_roadrunner_maker import RoadRunnerMaker  # type: ignore
from src.deprecated_roadrunner_maker import RoadRunnerMaker  # type: ignore

import numpy as np  # type: ignore


class JacobianCollectionMaker:

    def __init__(self, roadrunner_specification,
            start_time: float=cn.START_TIME,
            end_time: float=cn.END_TIME,
            num_points: int=cn.NUM_POINTS,
            is_keep_roadrunner: bool=True,
            ) -> None:
        """
        Parameters
        ----------
        roadrunner_specification : str or roadrunner.ExtendedRoadRunner
            SBML/Antimony model string or an existing RoadRunner instance.
        start_time : float
            Simulation start time.
        end_time : float
            Simulation end time.
        num_points : int
            Number of output timepoints.
        is_keep_roadrunner : bool
            If False, the RoadRunner instance is deleted after initialization to
            free memory. Can be restored later with restoreRoadRunner() only if
            initialized from a string.
        """
        self.roadrunner_maker = RoadRunnerMaker(
            roadrunner_specification,
            start_time=start_time,
            end_time=end_time,
            num_points=num_points,
            is_keep_roadrunner=is_keep_roadrunner,
        )
        if not is_keep_roadrunner:
            self.roadrunner_maker.deleteRoadRunner()

    def makeCollection(self) -> Trajectory:
        """
        Run simulations and collect the full Jacobian at each timepoint.

        The model is reset, simulated once to obtain the output timepoints, reset
        again, then stepped forward point-by-point so a Jacobian can be captured
        at each output time.

        Returns
        -------
        JacobianCollection
            Collection of Jacobian matrices and their corresponding timepoints.

        Raises
        ------
        ValueError
            If the model has no floating species after reset.
        """
        rr = self.roadrunner_maker.roadrunner
        rr.reset()
        if len(rr.getFloatingSpeciesIds()) == 0:
            raise ValueError("Model has no floating species; cannot compute Jacobian.")
        rm = self.roadrunner_maker
        result_arr = rr.simulate(rm.start_time, rm._end_time, rm.num_points)
        times_arr = np.array(result_arr["time"])  # copy before reset invalidates buffer

        rr.reset()
        jacobians = []
        for i, t in enumerate(times_arr):
            if i == 0:
                rr.simulate(rm.start_time, t + 1e-10, 2)
            else:
                rr.simulate(times_arr[i - 1], t, 2)
            jacobians.append(np.array(rr.getFullJacobian()).copy())

        return Trajectory(np.array(jacobians), times_arr)