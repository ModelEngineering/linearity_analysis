'''Multiple linear predictor for chemical reaction network trajectories.'''

from trajectory_collection import TrajectoryCollection  # type: ignore
from src.l_roadrunner import LRoadrunner  # type: ignore
from src.linear_predictor import LinearPredictor, ZERO_TOL  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from typing import List, Optional

ScoreResult = collections.namedtuple("ScoreResult", ["mean_rae", "max_rae"])


class MultipleLinearPredictor(object):
    """Predicts species concentrations across a sequence of linear models.

    Uses a ClusteredJacobianCollection to build one LinearPredictor per cluster,
    propagating the predicted state from the end of each cluster as the initial
    condition for the next.

    Construction mirrors LinearPredictor: callers supply initial_value_arr and
    forcing_input_arr directly.  Use makeFromLRoadrunner to derive these arrays
    automatically from a simulation.
    """

    def __init__(self,
            clustered_jacobian_collection: TrajectoryCollection,
            initial_value_arr: np.ndarray,
            forcing_input_arr: np.ndarray,
            l_roadrunner: Optional[LRoadrunner] = None,
            ) -> None:
        """
        Parameters
        ----------
        clustered_jacobian_collection : ClusteredJacobianCollection
            Clustered Jacobian collections defining the sequence of linear models.
        initial_value_arr : np.ndarray
            Either:
            - 1-D array of shape (n_species,): the initial floating species
              concentrations at the start of the first cluster; subsequent
              clusters use the predicted state propagated from the previous one.
            - 2-D array of shape (n_clusters, n_species): a distinct initial
              concentration vector for each cluster.  n_clusters must equal the
              number of JacobianCollections in clustered_jacobian_collection.
        forcing_input_arr : np.ndarray
            Either:
            - 1-D array of shape (n_species,): the same forcing vector is
              reused for every cluster.
            - 2-D array of shape (n_clusters, n_species): a distinct forcing
              vector is used for each cluster.  n_clusters must equal the
              number of JacobianCollections in clustered_jacobian_collection.
        l_roadrunner : LRoadrunner, optional
            LRoadrunner instance used by score() and plot() to obtain the
            reference simulation.  Not required for predict().

        Raises
        ------
        ValueError
            If initial_value_arr is 2-D and its first dimension does not match
            the number of JacobianCollections.
        ValueError
            If forcing_input_arr is 2-D and its first dimension does not match
            the number of JacobianCollections.
        """
        n_clusters = len(clustered_jacobian_collection.trajectories)
        if initial_value_arr.ndim == 2 and initial_value_arr.shape[0] != n_clusters:
            raise ValueError(
                f"initial_value_arr has {initial_value_arr.shape[0]} rows but "
                f"clustered_jacobian_collection has {n_clusters} clusters."
            )
        if forcing_input_arr.ndim == 2 and forcing_input_arr.shape[0] != n_clusters:
            raise ValueError(
                f"forcing_input_arr has {forcing_input_arr.shape[0]} rows but "
                f"clustered_jacobian_collection has {n_clusters} clusters."
            )
        self.clustered_jacobian_collection = clustered_jacobian_collection
        self.initial_value_arr = initial_value_arr
        self.forcing_input_arr = forcing_input_arr
        self.l_roadrunner = l_roadrunner
        if self.l_roadrunner is not None:
            self.species_names = self.l_roadrunner.getRoadrunner().getFloatingSpeciesIds()
        else:
            self.species_names = []

    def _getForcedInput(self, cluster_idx: int) -> np.ndarray:
        """Return the forcing vector for the given cluster index.

        Parameters
        ----------
        cluster_idx : int
            Zero-based index of the cluster.

        Returns
        -------
        np.ndarray
            1-D array of shape (n_species,).
        """
        if self.forcing_input_arr.ndim == 1:
            return self.forcing_input_arr
        return self.forcing_input_arr[cluster_idx]

    def _getClusterStart(self, cluster_idx: int, current_x: np.ndarray) -> np.ndarray:
        """Return the starting state for the given cluster.

        When initial_value_arr is 2-D, each cluster has its own fixed starting
        state and current_x is ignored.  When initial_value_arr is 1-D, the
        caller-supplied current_x is returned unchanged (cluster 0 passes in
        initial_value_arr; subsequent clusters pass in the propagated state).

        Parameters
        ----------
        cluster_idx : int
            Zero-based index of the cluster.
        current_x : np.ndarray
            Propagated state from the previous cluster (used only when
            initial_value_arr is 1-D).

        Returns
        -------
        np.ndarray
            1-D array of shape (n_species,).
        """
        if self.initial_value_arr.ndim == 2:
            return self.initial_value_arr[cluster_idx]
        return current_x

    @classmethod
    def makeFromLRoadrunner(cls,
            clustered_jacobian_collection: TrajectoryCollection,
            l_roadrunner: LRoadrunner,
            is_per_cluster_forcing_input: bool = False,
            ) -> "MultipleLinearPredictor":
        """Construct a MultipleLinearPredictor by deriving inputs from a simulation.

        Resets the model to obtain initial_value_arr, then computes forced input(s).

        When is_per_cluster_forcing_input is False (default), a single forcing
        vector is computed at the initial state using the first cluster's mean
        Jacobian: u = f(x0) - J_mean_0 @ x0.

        When is_per_cluster_forcing_input is True, a per-cluster forcing vector is
        computed by propagating predicted states forward:
            u_i = f(x_i) - J_mean_i @ x_i
        where x_i is the predicted state at the start of cluster i.  The result
        is a 2-D forcing_input_arr of shape (n_clusters, n_species).

        Parameters
        ----------
        clustered_jacobian_collection : ClusteredJacobianCollection
            Clustered Jacobian collections defining the sequence of linear models.
        l_roadrunner : LRoadrunner
            LRoadrunner instance used to obtain initial concentrations and
            instantaneous rates of change.
        is_per_cluster_forcing_input : bool
            If True, compute a distinct forcing vector for each cluster.
            If False (default), compute one forcing vector at the initial state.

        Returns
        -------
        MultipleLinearPredictor
        """
        rr = l_roadrunner.getRoadrunner()
        rr.reset()
        initial_value_arr = np.array(rr.getFloatingSpeciesConcentrations())
        species_ids = rr.getFloatingSpeciesIds()
        if not is_per_cluster_forcing_input:
            f_arr = np.array(rr.getRatesOfChange())
            first_jc = clustered_jacobian_collection.trajectories[0]
            forcing_input_arr = f_arr - first_jc.jacobian_median_arr @ initial_value_arr
        else:
            current_x = initial_value_arr.copy()
            forcing_inputs: List[np.ndarray] = []
            for jc in clustered_jacobian_collection.trajectories:
                rr.reset()
                for idx, sp_id in enumerate(species_ids):
                    rr[sp_id] = float(current_x[idx])
                f_arr = np.array(rr.getRatesOfChange())
                forcing_inputs.append(f_arr - jc.jacobian_median_arr @ current_x)
                duration = float(jc.timepoint_arr[-1] - jc.timepoint_arr[0])
                lp = LinearPredictor(jc, current_x, forcing_inputs[-1])
                current_x = lp.predict(np.array([0.0, duration])).prediction_arr[-1]
            forcing_input_arr = np.array(forcing_inputs)

        return cls(
                clustered_jacobian_collection,
                initial_value_arr,
                forcing_input_arr,
                l_roadrunner)

    def predict(self) -> np.ndarray:
        """Predict floating species concentrations at the end of each cluster.

        Uses the stored initial_value_arr and forcing_input_arr.  For each cluster:
        1. Build a LinearPredictor with the current state and the pre-computed
            forced input for that cluster.
        2. Predict concentrations at the last timepoint of the cluster.
        3. Use the predicted concentrations as the initial state for the next cluster.

        Returns
        -------
        np.ndarray
            Array of shape (n_clusters, n_species) containing the predicted
            floating species concentrations at the last timepoint of each cluster.
        """
        current_x = self.initial_value_arr.copy()
        predictions: List[np.ndarray] = []
        for i, jc in enumerate(self.clustered_jacobian_collection.trajectories):
            duration = float(jc.timepoint_arr[-1] - jc.timepoint_arr[0])
            linear_predictor = LinearPredictor(jc, current_x, self._getForcedInput(i))
            predicted_arr = linear_predictor.predict(np.array([0.0, duration])).prediction_arr
            current_x = predicted_arr[-1]
            predictions.append(current_x)

        return np.array(predictions)  # shape: (n_clusters, n_species)

    def score(self) -> ScoreResult:
        """Compute the mean and max relative absolute error (RAE) of the prediction.

        RAE is defined as |1 - prediction / simulation| computed element-wise
        across all timepoints and all floating species.  When both the predicted
        and simulated values are within ZERO_TOL of zero, the entry contributes
        zero error rather than NaN.

        Returns
        -------
        ScoreResult
            Named tuple with fields:
            - ``mean_rae`` : float — mean of |1 - prediction / simulation|
            - ``max_rae``  : float — max  of |1 - prediction / simulation|
        """
        if self.l_roadrunner is None:
            raise ValueError("score() requires l_roadrunner to be set.")
        sim_arr = self.l_roadrunner.simulate(is_with_timepoints=False)

        current_x = self.initial_value_arr.copy()
        predicted_rows: List[np.ndarray] = []
        for i, jc in enumerate(self.clustered_jacobian_collection.trajectories):
            abs_times = jc.timepoint_arr
            if len(abs_times) == 0:
                import pdb; pdb.set_trace()  # FIXME: Why is this cluster empty?
            rel_times = abs_times - abs_times[0]
            linear_predictor = LinearPredictor(jc, current_x, self._getForcedInput(i))
            predicted_arr = linear_predictor.predict(rel_times).prediction_arr
            predicted_rows.append(predicted_arr)
            current_x = predicted_arr[-1]

        prediction_arr = np.concatenate(predicted_rows, axis=0)

        if prediction_arr.shape != sim_arr.shape:
            return ScoreResult(mean_rae=np.nan, max_rae=np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            rae_arr = np.abs(1.0 - prediction_arr / sim_arr)
        both_near_zero = (
            (np.abs(prediction_arr) < ZERO_TOL) & (np.abs(sim_arr) < ZERO_TOL)
        )
        rae_arr[both_near_zero] = 0.0

        return ScoreResult(
            mean_rae=float(np.nanmean(rae_arr)),
            max_rae=float(np.nanmax(rae_arr)),
        )

    def plot(self,
            ax: Optional[maxes.Axes] = None,
            title: str = "Multiple Linear Prediction vs Simulation",
            ylim: Optional[tuple] = None,
            xlim: Optional[tuple] = None,
            ) -> mfigure.Figure:
        """Plot predicted and simulated species concentrations over time.

        Simulation is drawn as solid lines; the piecewise linear prediction as
        dashed lines.  One colour per floating species is used so that prediction
        and simulation for the same species share a colour.  A vertical dotted
        line marks the start of each cluster after the first, indicating where a
        new Jacobian takes over.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to draw on.  A new figure and axes are created when omitted.
        title : str, optional
            Title for the axes.
        ylim : tuple[float, float], optional
            y-axis limits as (ymin, ymax).
        xlim : tuple[float, float], optional
            x-axis limits as (xmin, xmax).

        Returns
        -------
        plt.Figure
            The figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()  # type: ignore

        # --- simulated trajectory ---
        if self.l_roadrunner is None:
            raise ValueError("plot() requires l_roadrunner to be set.")
        sim_arr = self.l_roadrunner.simulate(is_with_timepoints=True)
        sim_times = sim_arr[:, 0]
        sim_conc_arr = sim_arr[:, 1:]
        n_species = sim_conc_arr.shape[1]

        for i in range(n_species):
            label = self.species_names[i] if i < len(self.species_names) else f"species_{i}"
            ax.plot(sim_times, sim_conc_arr[:, i], color=f"C{i}",
                    label=f"{label} (simulation)")

        # --- piecewise linear prediction ---
        current_x = self.initial_value_arr.copy()

        for cluster_idx, jc in enumerate(self.clustered_jacobian_collection.trajectories):
            # Draw vertical boundary line at the start of each new cluster.
            if cluster_idx > 0:
                ax.axvline(x=float(jc.timepoint_arr[0]), color="gray",
                linestyle=":", linewidth=1.0)

            abs_times = jc.timepoint_arr
            rel_times = abs_times - abs_times[0]
            linear_predictor = LinearPredictor(jc, current_x, self._getForcedInput(cluster_idx))
            predicted_arr = linear_predictor.predict(rel_times).prediction_arr

            for i in range(n_species):
                label = (self.species_names[i] if i < len(self.species_names)
                        else f"species_{i}")
                if cluster_idx == 0:
                    ax.plot(abs_times, predicted_arr[:, i], color=f"C{i}",
                            linestyle="--", label=f"{label} (prediction)")
                else:
                    ax.plot(abs_times, predicted_arr[:, i], color=f"C{i}",
                            linestyle="--")

            current_x = predicted_arr[-1]

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        return fig  # type: ignore

    def heatmaps(self) -> mfigure.Figure:
        """Plot the CV heatmap for each JacobianCollection in a single row.

        Delegates to ClusteredJacobianCollection.heatmaps().

        Returns
        -------
        plt.Figure
            The figure containing one heatmap per cluster and a shared colorbar.
        """
        return self.clustered_jacobian_collection.heatmaps()
