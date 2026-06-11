'''Represents a trajectory, the results of a model simulation.'''

import src.constants as cn
from src.l_roadrunner import LRoadrunner, NULL_L_ROADRUNNER  # type: ignore
import src.utils as utils
from src.plot_options import PlotOptions  # type: ignore
from src.score import Score  # type: ignore

import collections
import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from lmfit import Parameters, minimize as lmfit_minimize  # type: ignore
from typing import List, Optional, Tuple, cast
import warnings

COL_SPECIES_IDX = "species_idx"
COL_TIMEPOINT = "timepoint"
COL_RIGHT = "right"
COL_CENTER = "center"
COL_RADIUS = "radius"

NUM_FIT = 30
LARGE_RESIDUAL_VALUE = 1e10



# Times relative to the characteristic time at which to evaluate the weighted eigenvectors
IVP_RELATIVE_TIMES = np.array([0, 0.1, 0.3])
NULL_COST = -1


PlotInfo = collections.namedtuple("PlotInfo",
        ["top_ax", "bottom_ax", "fig"])

class Trajectory(object):
    """A collection of Jacobian matrices over one or more simulation timepoints."""

    def __init__(self, l_roadrunner: LRoadrunner=NULL_L_ROADRUNNER,
            diameter_metric: str = cn.DIAMETER_IVP,
            eigenvalues_collection_arr: np.ndarray = np.array([]),
            eigenvector_collection_arr: np.ndarray = np.array([]),
            num_fit: int = NUM_FIT,
            jacobian_selection: str = cn.JAC_MEDIAN) -> None:
            #jacobian_selection: str = cn.JAC_FIT_GERSHGORIN) -> None:
        """
        Parameters
        ----------
        l_roadrunner : LRoadrunner
            An LRoadrunner instance used to simulate the model and collect Jacobians.
            Jacobians and timepoints are obtained by calling makeJacobians() on this object.
        diameter_metric : str
            The metric to use for calculating the diameter of the Jacobian collection.
            Options are "weighted_eigenvectors" (default) or "max_cv".
        eigenvalues_collection : List[np.ndarray]
            An optional list of eigenvalue arrays corresponding to the Jacobians, used for diameter calculations.
        eigenvector_collection : List[np.ndarray]
            An optional list of eigenvector arrays corresponding to the Jacobians, used for diameter calculations.
        num_fit : int
            The number of entries to fit in fitJacobian() (default: 30).
        jacobian_selection : str
            The method to use for selecting Jacobians (default: "fit_gershgorin").
                cn.JAC_FIT_GERSHGORIN = "fit_gershgorin" # Fit a Jacobian, selecting diagonal elements using Gershgorin circles
                cn.JAC_MEDIAN = "median"  # Use the median Jacobian
                cn.JAC_FIRST = "first" # Use the first Jacobian
        """
        # Declarations
        self._forcing_input_arr: np.ndarray
        # Implementation
        result = l_roadrunner.makeJacobians()
        self.jacobian_collection_arr, self.timepoint_arr, self.forcing_input_collection_arr  \
                =   result.jacobians, result.timepoints, result.forcing_inputs
        self._initialize(l_roadrunner, eigenvalues_collection_arr, eigenvector_collection_arr,
                num_fit=num_fit, jacobian_selection=jacobian_selection)
        self._sortArrays()
        self._diameter_metric = diameter_metric

    # ------------------------------------------------------------------
    # Properties (alphabetical)
    # ------------------------------------------------------------------

    @property
    def diameter(self) -> float:
        if self._diameter_metric == cn.DIAMETER_MAX_CV:
            return self.max_cv
        return self.diameter_ivp

    @property
    def diameter_ivp(self) -> float:
        """
        Represents each Jacobian by a vector of IVP solutions, and calculates the diameter
        of the collection as the maximum distance of any solution vector to the centroid of the solution vectors.
        """
        if not np.isnan(self._diameter):
            return cast(float, self._diameter)
        # Exclude trivial case of empty collection to avoid NaN issues with mean and distance calculations
        if len(self.jacobian_collection_arr) <= 1:
            return 0.0
        # Calculate the solution to the IVP for each Jacobian in the collection
        solution_collection_arr = self.ivp_solutions
        solution_median_arr = np.median(solution_collection_arr, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_solution_collection_arr = np.where(
                    solution_median_arr == 0,
                    0.0,
                    solution_collection_arr / solution_median_arr)
        diameter: float = 0.0
        middle_arr = np.ones(solution_median_arr.shape)
        for solution_arr in normalized_solution_collection_arr:
            distance = np.linalg.norm(solution_arr - middle_arr)
            diameter = max(diameter, distance) # type: ignore
        self._diameter: float = diameter
        return cast(float, self._diameter)

    @property
    def eigenvalues_collection_arr(self) -> np.ndarray:
        """Get the of eigenvalues for all Jacobians."""
        if len(self._eigenvalues_collection_arr) == 0 and self.jacobian_collection_arr.size > 0:
            self._makeEigenvaluesAndVectors()
        return self._eigenvalues_collection_arr

    @property
    def eigenvectors_collection_arr(self) -> np.ndarray:
        """Get the list of eigenvectors for all Jacobians."""
        if len(self._eigenvectors_collection_arr) == 0 and self.jacobian_collection_arr.size > 0:
            self._makeEigenvaluesAndVectors()
        return self._eigenvectors_collection_arr

    @property
    def ivp_solutions(self) -> np.ndarray:
        """
        Calculate the solution to the linear IVP problem at multiple time points for each Jacobian
            in the collection.
        For each Jacobian, a vector is constructed that is
            the for multiple time points for each state variable
        Return an array of these vectors.
        """
        if self.jacobian_collection_arr.size == 0:
            return np.array([])
        timepoint_arr = self.l_roadrunner.end_time * IVP_RELATIVE_TIMES
        solutions: list = []
        for jacobian_arr in self.jacobian_collection_arr:
            sol = solve_ivp(self._ivp, (timepoint_arr[0], timepoint_arr[-1]),
                    np.ones(jacobian_arr.shape[0]), t_eval=timepoint_arr,
                    args=(jacobian_arr,))
            solutions.append(np.concatenate(sol.y))
        return np.array(solutions)

    @property
    def jacobian_median_arr(self) -> np.ndarray:
        """Compute the element-wise median Jacobian across all timepoints."""
        if self._jacobian_median_arr.size == 0:
            self._jacobian_median_arr: np.ndarray = np.median(self.jacobian_collection_arr, axis=0)
        return self._jacobian_median_arr

    @property
    def jacobian_std_arr(self) -> np.ndarray:
        """Compute the element-wise standard deviation of Jacobians across all timepoints."""
        if self._jacobian_std_arr.size == 0:
            self._jacobian_std_arr: np.ndarray = np.std(self.jacobian_collection_arr, axis=0)
        return self._jacobian_std_arr

    @property
    def max_cv(self) -> float:
        """Compute the maximum coefficient of variation (CV = |std/mean|) across all Jacobian entries."""
        if self.jacobian_collection_arr.size == 0:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_arr = np.abs(self.jacobian_std_arr / self.jacobian_median_arr)
            cv_arr[~np.isfinite(cv_arr)] = 0.0
        return np.max(cv_arr)

    # ------------------------------------------------------------------
    # Public methods (alphabetical)
    # ------------------------------------------------------------------

#    def deprecatedsequentialPartition(self, n_cluster: int) -> List["Trajectory"]:
#        """Partition the Jacobians into n_cluster contiguous time segments.
#
#        Dynamic programming finds the partition into exactly n_cluster
#        contiguous segments that minimises the maximum within-segment cost.
#
#        Parameters
#        ----------
#        n_cluster : int
#            Number of contiguous segments.
#        cost_criteria : str
#            Cost metric per segment: ``"expo_eigen"`` (default) or ``"max_cv"``.
#
#        Returns
#        -------
#        List[JacobianCollection]
#            One JacobianCollection per segment, in time order.
#
#        Raises
#        ------
#        ValueError
#            If n_cluster exceeds the number of timepoints.
#        """
#        # Check for trivial case of 1 cluster to avoid unnecessary cost calculations
#        if n_cluster == 1:
#            return [self]
#     
#        n_point = self.jacobian_collection_arr.shape[0]
#        if n_cluster > n_point:
#            raise ValueError(
#                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_point})."
#            )
#        cost = np.zeros((n_point, n_point))
#        for i in range(n_point):
#            for j in range(i, n_point):
#                jc = Trajectory.fromArrays(
#                        self.jacobian_collection_arr[i:j + 1],
#                        self.timepoint_arr[i:j + 1],
#                        self.l_roadrunner,
#                        self.eigenvalues_collection_arr[i:j],
#                        self.eigenvectors_collection_arr[i:j],
#                        forcing_input_collection_arr=self.forcing_input_collection_arr[i:j + 1],
#                        )
#                cost[i][j] = jc.diameter
#        INF = float("inf")
#        dp = [[INF] * (n_point + 1) for _ in range(n_cluster + 1)]
#        split = [[0] * (n_point + 1) for _ in range(n_cluster + 1)]
#        dp[0][0] = 0.0
#        for k in range(1, n_cluster + 1):
#            for i in range(k, n_point + 1):
#                for j in range(k - 1, i):
#                    val = max(dp[k - 1][j], cost[j][i - 1])
#                    if val < dp[k][i]:
#                        dp[k][i] = val
#                        split[k][i] = j
#        boundaries = []
#        i = n_point
#        for k in range(n_cluster, 0, -1):
#            j = split[k][i]
#            boundaries.append((j, i))
#            i = j
#        boundaries.reverse()
#        return [
#            Trajectory.fromArrays(
#                self.jacobian_collection_arr[start:end],
#                self.timepoint_arr[start:end],
#                self.l_roadrunner,
#                self.eigenvalues_collection_arr[start:end],
#                self.eigenvectors_collection_arr[start:end],
#                forcing_input_collection_arr=self.forcing_input_collection_arr[start:end],
#                )
#            for start, end in boundaries
#        ]

    def fitJacobian(self, max_nfev: int = 1000) -> np.ndarray:
        """
        Fit each row of the Jacobian by choosing elements that minimize the
        L2 error of the variable for that row.

        Parameters
        ----------
        max_nfev : int, optional
            Maximum number of function evaluations for the optimization (default is 1000).

        Returns
        -------
        np.ndarray: fitted Jacobian array of shape (num_species, num_species)
        """
        ##
        if not np.array_equal(self._fitted_jacobian_arr, cn.NULL_ARRAY):
            return self._fitted_jacobian_arr
        if self._num_timepoint < self.num_species:
            raise ValueError(f"Cannot fit Jacobian with num_timepoint ({self._num_timepoint}) < num_species ({self.num_species}).") 
        # Iteratively process all rows (species) of the Jacobian
        jacobian_arr = self.jacobian_median_arr.copy()
        ##
        def _calculateResiduals(params: Parameters, ispecies:int) -> np.ndarray:
            # Calculates the residuals when the i-th row is replaced by the values
            # in params, and the other rows are unchanged
            for i in range(self.num_species):
                jacobian_arr[ispecies, i] = params[f'd{i}'].value
            prediction_arr = self._predictNextStep(jacobian_arr=jacobian_arr)[:, ispecies]
            residual_arr = self.timecourse_df.iloc[:, ispecies].values - prediction_arr
            return residual_arr[1:]  # Exclude timepoint 0 to avoid issues with initial state
        ##
        for ispecies, _ in enumerate(self.l_roadrunner.species_names):
            # Set the parameters to fit this row
            params = Parameters()
            for idx, coeff in enumerate(jacobian_arr[ispecies]):
                params.add(f'd{idx}', value=coeff)
            result = lmfit_minimize(_calculateResiduals, params,
                    method='leastsq', args=(ispecies,), max_nfev=max_nfev)
            jacobian_arr[ispecies] = [result.params[f'd{idx}'].value  # type: ignore
                    for idx in range(jacobian_arr.shape[1])]  # type: ignore
        self._fitted_jacobian_arr: np.ndarray = jacobian_arr
        return self._fitted_jacobian_arr

    @classmethod
    def fromArrays(cls,
            jacobian_arr: np.ndarray,
            timepoint_arr: np.ndarray,
            l_roadrunner: LRoadrunner = NULL_L_ROADRUNNER,
            eigenvalues_collection_arr: np.ndarray = np.array([]),
            eigenvector_collection_arr: np.ndarray = np.array([]),
            forcing_input_collection_arr: np.ndarray = np.array([]),
            **kwargs) -> 'Trajectory':
        """Create a JacobianCollection from explicit arrays."""
        jc = object.__new__(cls)
        jc.jacobian_collection_arr = jacobian_arr.copy()
        jc.timepoint_arr = timepoint_arr.copy()
        # Ensure forcing_input_collection_arr is always 2-D so it can be sliced
        # by partition methods even when no forcing inputs were provided.
        if forcing_input_collection_arr.size == 0:
            jc.forcing_input_collection_arr = np.zeros((jacobian_arr.shape[0], 0))
        else:
            jc.forcing_input_collection_arr = forcing_input_collection_arr.copy()
        jc._initialize(l_roadrunner, eigenvalues_collection_arr,
                eigenvector_collection_arr, **kwargs)
        jc._sortArrays()
        if l_roadrunner is not NULL_L_ROADRUNNER:
            try:
                jc.timecourse_df = l_roadrunner.timecourse_df.loc[jc.timepoint_arr]
            except KeyError:
                pass  # timepoint_arr not a subset of timecourse_df; keep full timecourse
        return jc

    def getCost(self, istart: int, iend: int) -> float:
        """Get the cost of a segment of Jacobians from index i to j (inclusive).

        The cost is calculated as the diameter of the segment, which is determined
        by the specified diameter_metric. Costs are cached in a cost matrix to
        avoid redundant calculations.

        Parameters
        ----------
        istart : int
            Starting index of the segment
        iend : int
            Ending index of the segment (inclusive).

        Returns
        -------
        float
            The cost of the segment from index i to j.
        """
        iendplus = iend + 1
        if self._cost_mat[istart][iend] == NULL_COST:
            jc = Trajectory.fromArrays(
                    self.jacobian_collection_arr[istart:iendplus],
                    self.timepoint_arr[istart:iendplus],
                    self.l_roadrunner,
                    self.eigenvalues_collection_arr[istart:iendplus],
                    self.eigenvectors_collection_arr[istart:iendplus],
                    forcing_input_collection_arr=self.forcing_input_collection_arr[istart:iendplus],
                    )
            self._cost_mat[istart][iend] = jc.diameter
        return self._cost_mat[istart][iend]

    @property 
    def forcing_input_arr(self) -> np.ndarray:
        """Return the array of forcing inputs corresponding to the Jacobian collection."""
        if self._forcing_input_arr.size == 0:
            self._forcing_input_arr = np.median(self.forcing_input_collection_arr, axis=0)
            self._forcing_input_std_arr = np.std(self.forcing_input_collection_arr, axis=0)
        return self._forcing_input_arr

    def getGershgorinCircles(self) -> pd.DataFrame:
        """Finds the center and radius of the Gershgorin circle for each Jacobian in the
        collection, and returns them in a DataFrame.

        Returns:
            pd.DataFrame: _description_
        """
        ars = []
        for timepoint, jacobian_arr in enumerate(self.jacobian_collection_arr):
            circles = utils.calculateGershgorinCircles(jacobian_arr)
            centers, radii = circles[:, 0], circles[:, 1]
            timepoints = np.full(centers.shape, self.timepoint_arr[timepoint])
            species_idx = np.arange(centers.shape[0])
            ars.append(pd.DataFrame(
                {COL_CENTER: centers, COL_RADIUS: radii, COL_TIMEPOINT: timepoints,
                    COL_SPECIES_IDX: species_idx}))
        return pd.concat(ars, ignore_index=True)

    def getTimes(self) -> np.ndarray:
        """Return the sorted unique timepoints in this collection."""
        return np.unique(self.timepoint_arr)

    def heatmap(self, ax: Optional[maxes.Axes] = None) -> mfigure.Figure:
        """Construct a heatmap of the Jacobian where cells are colored by their
        coefficient of variation (CV = |std/mean|) and labelled with their mean value.

        Zero-mean entries (where CV is undefined) are shown with CV=0 and labelled
        with their mean (0.0).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created when omitted.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the heatmap.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_arr = np.abs(self.jacobian_std_arr / self.jacobian_median_arr)
        cv_arr = np.where(np.isfinite(cv_arr), cv_arr, 0.0)

        mean_arr = self.jacobian_median_arr
        annot_arr = np.vectorize(lambda v: f"{v:.2g}")(mean_arr)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()  # type: ignore
            assert isinstance(fig, mfigure.Figure)

        sns.heatmap(
            cv_arr,
            annot=annot_arr,
            fmt="",
            cmap="coolwarm",
            cbar_kws={"label": "CV (|std/mean|)"},
            xticklabels=self.species_names if self.species_names is not None else "auto",
            yticklabels=self.species_names if self.species_names is not None else "auto",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_xlabel("Species")
        ax.set_ylabel("Species")
        ax.set_title("Jacobian: CV (color) and mean (label)")
        assert isinstance(fig, mfigure.Figure)
        fig.tight_layout()
        return fig

    @classmethod
    def makeBiomodel(cls, path:str = "", model_name: str = "",
            model_num: int = 0, 
            jacobian_selection: str = cn.JAC_MEDIAN,
            **kwargs) -> 'Trajectory':
        """
        Create a Trajectory instance from a BioModels SBML file.

        Parameters
        ----------
        path : str
            Path to the SBML file.
        model_name : str
            Name of the BioModel to load.
        model_num : int
            Number of the model to load.
        kwargs: Additional keyword arguments to pass to LRoadrunner.makeBiomodel() to specify the trajectory

        Returns
        -------
        Trajectory
            An instance of Trajectory initialized with the model from the specified SBML file.
        """
        num_fit = kwargs.pop("num_fit", NUM_FIT)
        l_roadrunner = LRoadrunner.makeBiomodel(path=path, model_name=model_name,
                model_num=model_num, **kwargs)
        trajectory = Trajectory(l_roadrunner, num_fit=num_fit,
                jacobian_selection=jacobian_selection)
        return trajectory

    def nonsequentialPartition(self, n_cluster: int, max_iter: int = 300) -> List["Trajectory"]:
        """Partition the Jacobians into n_cluster clusters using KMeans.

        Clusters need not consist of contiguous timepoints. Each Jacobian
        matrix is flattened to a feature vector and clustered with KMeans
        (k-means++ init).

        Parameters
        ----------
        n_cluster : int
            Number of clusters to partition the Jacobians into.
        max_iter : int
            Maximum number of k-means iterations (default: 300).

        Returns
        -------
        List[JacobianCollection]
            One JacobianCollection per cluster.

        Raises
        ------
        ValueError
            If n_cluster exceeds the number of timepoints.
        """
        n_points = self.jacobian_collection_arr.shape[0]
        if n_cluster > n_points:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({n_points})."
            )
        flat_arr = self.jacobian_collection_arr.reshape(n_points, -1).astype(float)
        kmeans = KMeans(
            n_clusters=n_cluster, init="k-means++", max_iter=max_iter,
            n_init=1, random_state=0,
        )
        labels_arr = kmeans.fit_predict(flat_arr)
        cluster_indices = [np.where(labels_arr == c)[0] for c in range(n_cluster)]
        return [
            Trajectory.fromArrays(
                self.jacobian_collection_arr[idx], self.timepoint_arr[idx], self.l_roadrunner,
                forcing_input_collection_arr=self.forcing_input_collection_arr[idx])
            for idx in cluster_indices
        ]

    def plot(self,
            top_ax: Optional[plt.Axes] = None,   # type: ignore
            bottom_ax: Optional[plt.Axes] = None, # type: ignore
            fig: Optional[plt.Figure] = None,  # type: ignore
            is_legend: bool = True,
            ylim: Tuple[float, float] = (0.0, 1.0),
            xlim: Optional[Tuple[float, float]] = None,
            ) -> PlotInfo:
        """
        Constructs a figure with two plots with time on the x-axis: (1) the Frobenius-norm distance of each Jacobian from the centroid, and
        (2) the timecourse of simulation species concentrations.
        The first plot shows how the Jacobian changes over time relative to the centroid.
        The second plot shows the dynamics of the model's species concentrations
        over time.

        Parameters
        ----------
        top_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the top plot. If None, a new figure and axes will be created.
        bottom_ax : Optional[plt.Axes]
            An optional matplotlib Axes object to use for the bottom plot. If None, a new figure and axes will be created.
        fig : Optional[plt.Figure]
            An optional matplotlib Figure object to use. If None, a new figure will be created.
        is_legend : bool
            Whether to include a legend in the species timecourse plot (default: True).
        ylim : Tuple[float, float]
            The y-axis limits for the Jacobian deviation plot (default: (0.0, 1.0)).
        xlim: Tuple[float, float]
            The x-axis limits for both plots (default: None, which means automatic limits).
        """
        species_data = self.timecourse_df.values
        species_times = self.timecourse_df.index.values
        jacobian_times = self.getTimes()
        deviation_arr = self._calculateDeviation()

        if top_ax is None or bottom_ax is None or fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
        else:
            ax1 = top_ax
            ax2 = bottom_ax

        ax1.plot(jacobian_times, deviation_arr, marker="o")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Normalized distance")
        ax1.set_title(f"{self.model_name}: Normalized Distance of Jacobian to Centroid")
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        # Timecourse plot
        prediction_df = self.predict()
        colors = [sns.color_palette("tab10")[i % 10] for i in range(len(self.species_names))]
        for i, species_id in enumerate(self.species_names):
            ax2.plot(species_times, species_data[:, i], label=species_id, color=colors[i], alpha=0.7)
            ax2.scatter(species_times, prediction_df[species_id], s=8, alpha=0.7, color=colors[i])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Concentration")
        ax2.set_title(f"{self.model_name}: Species Timecourse")
        ax2.set_xlim(xlim)
        if is_legend:
            ax2.legend()

        fig.tight_layout()
        plt.show()
        return PlotInfo(top_ax=ax1, bottom_ax=ax2, fig=fig)
    
    def plotPrediction(self, num_step: int = 1, **kwargs) -> PlotOptions:
        """
        Plot the predicted timecourse of simulation species concentrations.
        The first plot shows how the Jacobian changes over time relative to the centroid.
        The second plot shows the dynamics of the model's species concentrations
        over time. Does not plot the first value since this is the initial state and not a prediction.

        Parameters
        ----------
        num_step : int
            The number of time steps to predict ahead.
        kwargs:
            ax : Optional[plt.Axes]
                An optional matplotlib Axes
            title: str
                The title for the plot
            fig : Optional[plt.Figure]
                An optional matplotlib Figure object to use. If None, a new figure will be created.
            is_legend : bool
                Whether to include a legend in the species timecourse plot (default: True).
            ylim : Tuple[float, float]
                The y-axis limits for the Jacobian deviation plot (default: (0.0, 1.0)).
            xlim: Tuple[float, float]
                The x-axis limits for both plots (default: None, which means automatic limits).
            model_name: str
                The model name
        """
        kwargs = dict(kwargs)  # Make a copy to avoid modifying the original
        #
        species_data = self.timecourse_df.values
        species_times = self.timecourse_df.index.values
        if num_step == -1:
            num_step = len(self.timepoint_arr) - 1
        pred_df = self.predict(num_step=num_step)
        # Extract model_name before passing kwargs to PlotOptions (not a PlotOptions param)
        model_name = kwargs.pop("model_name", "")
        if model_name and "title" not in kwargs:
            kwargs["title"] = f"{model_name}: Species Timecourse"
        # Timecourse plot
        if "title" not in kwargs:
            kwargs["title"] = f"Number steps: {num_step}"
        plt_opt = PlotOptions(**kwargs)
        ax = plt_opt.ax
        colors = [sns.color_palette("tab10")[i % 10] for i in range(len(self.species_names))]
        # Do separate loops so that legend works out correctly
        for i, species_id in enumerate(self.species_names):
            ax.plot(species_times, species_data[:, i], # type: ignore
                    label=species_id, color=colors[i], alpha=0.7)
        for i, species_id in enumerate(self.species_names):
            ax.scatter(species_times[1:], pred_df[species_id].values[1:],  # type: ignore
                    s=8, alpha=0.7, color=colors[i])
        plt_opt.apply()
        return plt_opt

    def predict(self, num_step: int = 1, **kwargs) -> pd.DataFrame:
        """
        Do 1 step predictions of the species concentrations at each timepoint in the trajectory,

        Parameters
        ----------
        num_step : int
            The number of time steps to predict ahead.
        forcing_input_arr : np.ndarray
            1-D array of shape (num_species,) containing constant forcing inputs for each species.
        jacobian_arr : np.ndarray
            2-D array of shape (num_species, num_species) containing the Jacobian to use for predictions

        Returns
        -------
        np.ndarray
            2-D array of shape (num_timepoints, num_species) containing the predicted concentrations.
        """
        arr = self._predictManyStep(num_step=num_step, **kwargs)
        df = pd.DataFrame(arr, columns=self.l_roadrunner.species_names)
        df = df.set_index(self.timepoint_arr)
        return df

    def sequentialPartition(self, n_cluster: int) -> List["Trajectory"]:
        """Partition the Jacobians into n_cluster contiguous time segments.
            Uses a greedy heuristic to find a partition into exactly n_cluster contiguous segments
            that reduces the maximum within-segment diameter.

        Parameters
        ----------
        n_cluster : int
            Number of contiguous segments.

        Returns
        -------
        List[JacobianCollection]
            One JacobianCollection per segment, in time order.

        Raises
        ------
        ValueError
            If n_cluster exceeds the number of timepoints.
        """
        # Check for trivial case of 1 cluster to avoid unnecessary cost calculations
        if n_cluster == 1:
            return [self]
        #
        if n_cluster > self.num_jacobian:
            raise ValueError(
                f"n_cluster ({n_cluster}) exceeds number of timepoints ({self.num_jacobian})."
            )
        ##
        def split(istart: int, iend: int) -> int:
            """
            The minimum cost split is the maximum cost of the two resulting segments
            The upper end of the interval (RHS) is exclusive, so the last index is iend-1

            Parameters
            ----------
            istart : int
                Starting index of the segment to split (inclusive).
            iend : int
                Ending index of the segment to split (exclusive).

            Returns
            -------
            int: The index at which to split the segment,
                    where the left segment is [istart, idx) and the right segment is [idx, iend].
            """
            costs = [max(self.getCost(istart, i),  self.getCost(i, iend)) for i in range(istart, iend)]
            idx = np.argmin(costs) + istart
            return cast(int, idx)
        ##
        first_interval = (0, self.num_jacobian-1)  # All indices are inclusive, so the last index is num_jacobian-1
        intervals: List[Tuple[int, int]] = [first_interval]
        interval_costs: List[float] = [self.getCost(*first_interval)]
        for _ in range(n_cluster - 1):
            highest_cost_idx = np.argmax(interval_costs)
            lower_idx = intervals[highest_cost_idx][0]
            upper_idx = intervals[highest_cost_idx][1]
            split_idx = split(lower_idx, upper_idx)
            sub_interval1 = (lower_idx, split_idx)
            sub_interval2 = (split_idx, upper_idx)
            # Update the intervals
            intervals.remove(intervals[highest_cost_idx])
            intervals.append(sub_interval1)
            intervals.append(sub_interval2)
            # Update the costs
            interval_costs.remove(interval_costs[highest_cost_idx])
            interval_costs.append(self.getCost(*sub_interval1))
            interval_costs.append(self.getCost(*sub_interval2))
        # Construct the JacobianCollections for each cluster
        jcs: List[Trajectory] = []
        for istart, iend in intervals:
            if iend == self.num_jacobian - 1:
                iend = self.num_jacobian # Get all of the Jacobians in the last segment
            jcs.append(
                Trajectory.fromArrays(
                self.jacobian_collection_arr[istart:iend],
                self.timepoint_arr[istart:iend],
                self.l_roadrunner,
                self.eigenvalues_collection_arr[istart:iend],
                self.eigenvectors_collection_arr[istart:iend],
                forcing_input_collection_arr=self.forcing_input_collection_arr[istart:iend],
                )
            )
        return jcs

    # ------------------------------------------------------------------
    # Private methods (alphabetical)
    # ------------------------------------------------------------------

    def _calculateDeviation(self) -> np.ndarray:
        """
        Calculate the Frobenius-norm distance of each Jacobian from the centroid.

        The centroid is the element-wise mean of all Jacobians in jacobian_arr.
        For each timepoint the deviation is ||J(t) - centroid||_F.

        Returns
        -------
        np.ndarray
            1-D array of shape (num_points,) containing the deviation at each timepoint.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            diff_arr = np.abs(self.jacobian_collection_arr - self.jacobian_median_arr)/np.abs(self.jacobian_median_arr)
        diff_arr[:, np.abs(self.jacobian_median_arr) == 0] = 0.0
        result = np.sqrt(np.sum(diff_arr**2, axis=(1,2)))
        return result

    def _initialize(self, l_roadrunner: LRoadrunner,
                eigenvalues_collection_arr: np.ndarray = np.array([]),
                eigenvector_collection_arr: np.ndarray = np.array([]),
                num_fit: int = NUM_FIT,
                jacobian_selection: str = cn.JAC_MEDIAN) -> None:
        """Initialize the JacobianCollection with a new LRoadrunner instance."""
        self.l_roadrunner = l_roadrunner
        self.model_name = self.l_roadrunner.model_name if hasattr(self.l_roadrunner, "model_name") else ""  
        self.num_species = self.l_roadrunner.num_species
        self.species_names = self.l_roadrunner.species_names
        self._num_timepoint = self.l_roadrunner.num_point
        self._jacobian_median_arr = np.array([])
        self._jacobian_std_arr = np.array([])
        self._diameter = np.nan
        self._eigenvalues_collection_arr: np.ndarray = eigenvalues_collection_arr
        self._eigenvectors_collection_arr: np.ndarray = eigenvector_collection_arr
        self._diameter_metric = cn.DIAMETER_IVP
        self.num_jacobian = self.jacobian_collection_arr.shape[0] if hasattr(self, "jacobian_collection_arr") else 0
        if self.num_jacobian == 0:
            raise ValueError("JacobianCollection initialized with no Jacobians.")
        self._cost_mat = NULL_COST*np.ones((self.num_jacobian, self.num_jacobian))  # Placeholder value indicating uninitialized cost matrix
        self._fitted_jacobian_arr = cn.NULL_ARRAY
        self._num_fit = num_fit
        self._jacobian_selection = jacobian_selection
        self.timecourse_df = self.l_roadrunner.timecourse_df.copy()
        self._forcing_input_arr = np.array([])

    @staticmethod
    def _ivp(_: float, x: np.ndarray, jacobian_arr: np.ndarray) -> np.ndarray:
        # Calculate the derivative of x with respect to time given the Jacobian and current state x
        return jacobian_arr @ x

    def _makeEigenvaluesAndVectors(self) -> None:
        """Calculate the eigenvalues and eigenvectors for each Jacobian in the collection."""
        eigenvalues_collection:list = []
        eigenvectors_collection: list = []
        for jacobian_arr in self.jacobian_collection_arr:
            eigval, eigvec = np.linalg.eig(jacobian_arr)
            eigenvalues_collection.append(eigval)
            eigenvectors_collection.append(eigvec)
        self._eigenvalues_collection_arr = np.array(eigenvalues_collection)
        self._eigenvectors_collection_arr = np.array(eigenvectors_collection)

    def _predictNextStep(self,
            forcing_input_arr: np.ndarray = cn.NULL_ARRAY,
            jacobian_arr: np.ndarray = cn.NULL_ARRAY,
            ) -> np.ndarray:
        """
        Does one-step prediction across the entire timecourse.
        Predict the timecourse of species concentrations
        using linear prediction. If no values are given for forced input,
        these are obtained from LRoadrunner.

        Parameters
        ----------
        forcing_input_arr : np.ndarray
            1-D array of shape (num_species,) containing constant forcing inputs for each species.
        jacobian_arr : np.ndarray
            2-D array of shape (num_species, num_species) to use as the Jacobian. If omitted,
            self.jacobian_mean_arr is used.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_timepoints, num_species) containing the predicted concentrations.
        """
        if np.array_equal(forcing_input_arr, cn.NULL_ARRAY):
            forcing_input_arr = self.forcing_input_arr
        if np.array_equal(jacobian_arr, cn.NULL_ARRAY):
            if self._jacobian_selection == cn.JAC_FITTED:
                jacobian_arr = self.fitJacobian()
            elif self._jacobian_selection == cn.JAC_MEDIAN:
                jacobian_arr = self.jacobian_median_arr
            elif self._jacobian_selection == cn.JAC_FIRST:
                jacobian_arr = self.jacobian_collection_arr[0]
            else:
                raise ValueError(f"Invalid jacobian_selection: {self._jacobian_selection}")
        ##
        def ode(t: float, x: np.ndarray) -> np.ndarray:
            return jacobian_arr @ x + forcing_input_arr
        ##
        n_time = len(self.timepoint_arr)
        n_species = len(forcing_input_arr)
        result_arr = np.full((n_time, n_species), np.nan)
        result_arr[0] = self.l_roadrunner.timecourse_df.loc[self.timepoint_arr[0]].values # type: ignore
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for itime, timepoint in enumerate(self.timepoint_arr[:-1]):
                    initial_state_arr = self.l_roadrunner.timecourse_df.loc[timepoint].values # type: ignore
                    sol = solve_ivp(ode,
                            (timepoint, self.timepoint_arr[itime+1]),
                            initial_state_arr,
                            t_eval=[self.timepoint_arr[itime+1]],
                            method='Radau')
                    if sol.success and sol.y.shape == (n_species, 1):
                        result_arr[itime+1] = sol.y.T
            return result_arr
        except Exception as e:
            return np.full((n_time, n_species), np.nan)
    
    def _predictManyStep(self,
            num_step: int = -1,
            forcing_input_arr: np.ndarray = cn.NULL_ARRAY,
            jacobian_arr: np.ndarray = cn.NULL_ARRAY,
            ) -> np.ndarray:
        """
        Does many-step prediction across the entire timecourse.
        num_step is the number of steps to predict, and for each prediction,
        the initial values are taken from the simulated ("real") value
        of the timecourse.

        Parameters
        ----------
        num_step : int
            The number of steps to predict. If -1, predict all steps.
            Must divide the number of timepoints in the trajectory.
        forcing_input_arr : np.ndarray
            1-D array of shape (num_species,) containing constant forcing inputs for each species.
        jacobian_arr : np.ndarray
            2-D array of shape (num_species, num_species) to use as the Jacobian. If omitted,
            self.jacobian_mean_arr is used.

        Returns
        -------
        np.ndarray
            2-D array of shape (num_timepoints, num_species) containing the predicted concentrations.
        """
        if num_step == -1:
            num_step = len(self.timepoint_arr) - 1
        if num_step <= 0 or num_step >= len(self.timepoint_arr):
            raise ValueError(f"Argument 'num_step' ({num_step}) must divide the number of timepoints ({len(self.timepoint_arr) - 1}).")
        #
        if np.array_equal(forcing_input_arr, cn.NULL_ARRAY):
            forcing_input_arr = self.forcing_input_arr
        if np.array_equal(jacobian_arr, cn.NULL_ARRAY):
            if self._jacobian_selection == cn.JAC_FITTED:
                jacobian_arr = self.fitJacobian()
            elif self._jacobian_selection == cn.JAC_MEDIAN:
                jacobian_arr = self.jacobian_median_arr
            elif self._jacobian_selection == cn.JAC_FIRST:
                jacobian_arr = self.jacobian_collection_arr[0]
            else:
                raise ValueError(f"Invalid jacobian_selection: {self._jacobian_selection}")
        ##
        def ode(t: float, x: np.ndarray) -> np.ndarray:
            return jacobian_arr @ x + forcing_input_arr
        ##
        n_time = len(self.timepoint_arr)
        n_species = len(forcing_input_arr)
        result_arr = np.full((n_time, n_species), np.nan)
        result_arr[0] = self.l_roadrunner.timecourse_df.loc[self.timepoint_arr[0]].values # type: ignore
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                index_arr = np.arange(0, len(self.timepoint_arr) - num_step, num_step)
                for itime in index_arr:
                    cur_time = self.timepoint_arr[itime]
                    pred_time = self.timepoint_arr[itime + num_step]
                    t_evals = [t for t in self.timepoint_arr if cur_time < t <= pred_time]
                    initial_state_arr = self.l_roadrunner.timecourse_df.loc[cur_time].values # type: ignore
                    sol = solve_ivp(ode,
                            (cur_time, pred_time),
                            initial_state_arr,
                            t_eval=t_evals,
                            method='Radau')
                    if sol.success and sol.y.shape == (n_species, num_step):
                        result_arr[itime+1:itime + num_step + 1] = sol.y.T
                    else:
                        raise ValueError(f"ODE solver failed at timepoint {cur_time} with message: {sol.message}")
            return result_arr
        except Exception as e:
            return result_arr

    def _sortArrays(self) -> None:
        """Sort the jacobian_arr, timepoint_arr, and forcing_input_collection_arr by timepoint."""
        sort_indices = np.argsort(self.timepoint_arr)
        self.timepoint_arr = self.timepoint_arr[sort_indices]
        self.jacobian_collection_arr = self.jacobian_collection_arr[sort_indices]
        if hasattr(self, "forcing_input_collection_arr") and self.forcing_input_collection_arr.size > 0:
            self.forcing_input_collection_arr = self.forcing_input_collection_arr[sort_indices]