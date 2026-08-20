"""
Discovery of a system of differential equations from data using PySINDy, tailored for chemical reaction networks.
====================================================================================
Discovers a system of ODEs from time-series concentration data using PySINDy
that estimate the derivatives of each species as a sparse linear combination of polynomial features of the species concentrations.
Assumes rate laws are at most quadratic in the species concentrations (i.e.,
the library includes constant, linear, and pairwise-product terms).

Supports up to ``MAX_SPECIES`` (200) chemical species.

Dependencies
------------
    pip install pysindy pandas numpy scipy matplotlib

Input
-----
A pandas DataFrame with:
  - Index is time
  - One column per species   (up to 200)

Usage
-----
    from src.system_discovery import SystemDiscovery, discoverNetwork

    disc = SystemDiscovery(
        df,
        threshold=0.05,          # STLSQ sparsity threshold
        alpha=0.05,              # L2 regularisation
        differentiation="smooth" # "smooth" | "finite" | "spectral"
    )
    disc.fit()
    disc.print_equations()
    disc.plot_results()
    summary = disc.summary()
"""

import constants as cn # type: ignore
from src.model import Model  # type: ignore
from src.scaler import Scaler  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.score import Score  # type: ignore

from collections import namedtuple
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pysindy as ps # type: ignore
from scipy.linalg import expm  # type: ignore
from pysindy.feature_library import PolynomialLibrary # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from typing import Literal, Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NULL_DF = pd.DataFrame()
# max allowed deviation in time step size for spectral derivative
MAX_TIME_FRACTIONAL_DEVIATION = 0.001

# ODE integration tolerances (used by solve_ivp)
ODE_RTOL: float = 1e-6
ODE_ATOL: float = 1e-8

# Scatter-plot density step for perturbation analysis
DEFAULT_FRAC_KEEP: float = 0.2

# Default number of true points to plot in trajectory comparison figures
DEFAULT_NUM_TRUE_POINT: int = 20

# Columns in dataframes
COL_FRACTION_SPECIES_PERTURBABLE: str = "fraction_species_perturbable"


MAX_SPECIES: int = 200
DifferentiationMethod = Literal["smooth", "finite", "spectral"]



# ---------------------------------------------------------------------------
# Named tuples
# ---------------------------------------------------------------------------
AnalyzePerturbationsResult = namedtuple("AnalyzePerturbationsResult", ["df", "fig"])
DiscoverNetworkResult = namedtuple("DiscoverNetworkResult",["sd", "fig"])

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SystemDiscovery:
    """Discover a chemical reaction network from concentration time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series data.  Must contain a time column and one column per
        chemical species (concentrations must be non-negative).
        Index is time
    threshold : float
        STLSQ sparsity threshold.  Terms whose coefficient magnitude falls
        below this value are pruned.  Tune this to trade sparsity for fit.
        Default ``0.05``.
    alpha : float
        L2 (ridge) regularisation coefficient for STLSQ.  Default ``0.05``.
    differentiation : str
        Numerical differentiation strategy:
        - ``"smooth"``   – SmoothedFiniteDifference (recommended for noisy data)
        - ``"finite"``   – standard finite differences
        - ``"spectral"`` – spectral derivative (requires uniform sampling)
        Default ``"smooth"``.
    poly_degree : int
        Maximum polynomial degree of the feature library.  Must be 1 or 2
        (linear or quadratic rate laws).  Default ``2``.
    include_bias : bool
        Whether to include a constant (zeroth-order / production) term in the
        library.  Default ``True``.
    species_names : list[str] | None
        Override species labels used in printed equations and plots.
        If ``None``, column names from *df* are used.
    bias_species : list[str] | None
        Names of species whose ODE is permitted to have a constant term.
        All other species have their constant coefficient forced to zero
        after fitting.  Names must match ``species_names`` (or the
        DataFrame column names when ``species_names`` is ``None``).
        When provided, ``include_bias`` is forced to ``True`` so that the
        constant feature exists in the library.  Default ``None`` (no
        per-species restriction).
    is_normalize : bool
        Whether to normalize the data before fitting.  Default ``True``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01,
        alpha: float = 0.05,
        differentiation: DifferentiationMethod = "smooth",
        poly_degree: int = 1,
        include_bias: bool = True,
        species_names: list[str] | None = None,
        bias_species: list[str] | None = None,
        is_normalize: bool = True,
    ) -> None:
        self.df = df
        self.threshold = threshold
        self.alpha = alpha
        self.differentiation = differentiation
        self.poly_degree = poly_degree
        self.include_bias = include_bias

        # Extract time and concentration arrays
        species_cols = self.df.columns.to_list()
        if len(species_cols) > MAX_SPECIES:
            raise ValueError(
                f"DataFrame contains {len(species_cols)} species columns; "
                f"maximum supported is {MAX_SPECIES}."
            )

        self.species_cols = species_cols
        self.num_species = len(species_cols)
        self._time_arr = self.df.index.to_numpy(dtype=float)
        self._X_arr: np.ndarray = self.df[species_cols].to_numpy(dtype=float)
        self._Xdot_arr: np.ndarray = np.diff(self._X_arr, axis=0) / np.diff(self._time_arr).reshape(-1,1)
        self.Xdot_df = pd.DataFrame(self._Xdot_arr, index=self._time_arr[1:], columns=species_cols)
        #
        if species_names is not None:
            if len(species_names) != len(species_cols):
                raise ValueError(
                    "`species_names` length must match the number of species columns."
                )
            self.species_names = species_names
        else:
            self.species_names = species_cols
        self.species_names = [n[1:-1] if n.startswith("[") else n for n in self.species_names]
        # Build Scaler with species_names as column labels so Scaler keys match
        # the feature names PySINDy generates from species_names.
        self._scaler = Scaler(self.df, is_null_scaler=not is_normalize)

        if bias_species is not None:
            invalid = set(bias_species) - set(self.species_names)
            if invalid:
                raise ValueError(
                    f"`bias_species` contains names not in species_names: {sorted(invalid)}"
                )
            self.include_bias = True
        self.bias_species: list[str] | None = bias_species

        self._differentiator = self._build_differentiator()

        library = PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=True,
        )
        optimizer = ps.STLSQ(threshold=0, alpha=self.alpha)

        diff_method = self._differentiator

        self.model: ps.SINDy = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
            differentiation_method=diff_method,
        )
        self.is_fitted: bool = False

    def __str__(self) -> str:
        if self.is_fitted:
            result = "\n".join([f"d{n}/dt = {e}" for n, e in self.getEquations().items()])
        else:
            result = "Model not fitted yet."
        return result

    def _applyThreshold(self) -> None:
        """
        Zero out normalized coefficients whose physical value is below
        self.threshold.
        Updates self.model.optimizer.coef_ in-place.
        """
        feature_names = self.model.get_feature_names()
        coefs = self.model.optimizer.coef_  # shape (n_species, n_features), modified in-place
        for i, sp_name in enumerate(self.species_names):
            for j, feat_name in enumerate(feature_names):
                if not np.isclose(coefs[i, j],  0.0):
                    norm_thresh = self._scaler.normalizeThreshold(
                        sp_name, feat_name, self.threshold)
                    if abs(coefs[i, j]) < norm_thresh:
                        coefs[i, j] = 0.0

    def _build_differentiator(self):
        if self.differentiation == "smooth":
            return ps.SmoothedFiniteDifference()
        elif self.differentiation == "finite":
            return ps.FiniteDifference()
        elif self.differentiation == "spectral":
            return ps.SpectralDerivative()
        else:
            raise ValueError(
                f"Unknown differentiation method '{self.differentiation}'. "
                "Choose from: 'smooth', 'finite', 'spectral'."
            )

    @staticmethod
    def _normalize_rsq(rsq: float) -> float:
        """Normalize R² to [0, 1] and handle NaN."""
        if np.isnan(rsq):
            return 0.0
        return max(0.0, min(1.0, rsq))

    @staticmethod
    def _perturbation_col(p: float) -> str:
        """Map a perturbation fraction to its CSV column name.

        Examples: 0.0 → 'r2_0', 0.05 → 'r2_+05', -0.20 → 'r2_-20'.
        """
        pct = round(p * 100)
        if pct == 0:
            return "r2_0"
        sign = "+" if pct > 0 else "-"
        return f"r2_{sign}{abs(pct):02d}"

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call `.fit()` before using this method.")

    def _simulate(self,
            x0: np.ndarray | None = None,
            time_arr: np.ndarray | None = None) -> np.ndarray:
        """
        Chooses the simulation method based on the model's configuration
        (matrix exponential for linear systems, otherwise general ODE integration).
        Does parameter checks and calls the appropriate simulation method.

        Args
        ----
        x0 : np.ndarray, optional
            Initial state vector in physical units.  If None, uses the first row of training data
        time_arr : np.ndarray, optional
            Time points at which to evaluate the solution.
            If None, uses the training time grid

        Returns
        -------
        np.ndarray
        """
        if x0 is None:
            x0 = self._X_arr[0, :]
        if time_arr is None:
            time_arr = self._time_arr
        # Check time step uniformity for matrix exponential simulation
        diff_arr = np.diff(time_arr)
        diff_min = np.min(diff_arr)
        diff_max = np.max(diff_arr)
        max_deviation = (diff_max - diff_min) / np.mean(diff_arr)
        # Determine if we can use matrix exponential
        #   simulation (linear system with uniform time steps) 
        if self.poly_degree != 1 or not self.include_bias  \
                or max_deviation > MAX_TIME_FRACTIONAL_DEVIATION:
            return self._simulateGeneral(x0=x0, time_arr=time_arr)
        else:
            return self._simulateSimple(x0=x0, time_arr=time_arr)
    
    def _simulateGeneral(self,
            x0: np.ndarray,
            time_arr: np.ndarray) -> np.ndarray:
        """
        Implements a general ODE simulation using solve_ivp. This method
        is used for integrating the discovered ODE forward from *x0* over *time_arr*.
        Integrate the discovered ODE forward from *x0* over *time_arr*.
        Assumes that parameter checks have been done.

        Args
        ----
        x0 : np.ndarray, optional
            Initial state vector in physical units.  If None, uses the first row of training data
        time_arr : np.ndarray, optional
            Time points at which to evaluate the solution.
            If None, uses the training time grid

        Returns
        -------
        np.ndarray
        """
        ##
        def rhs(_t, x):
            z = self._scaler.normalize(x)
            dz_dt = self.model.predict(z.reshape(1, -1))[0]
            dx_dt = self._scaler.denormalize(dz_dt)
            return np.array(dx_dt, dtype=float)
        ##

        try:
            sol = solve_ivp(
                rhs,
                t_span=(time_arr[0], time_arr[-1]),
                y0=x0,
                t_eval=time_arr,
                method="Radau",
                rtol=ODE_RTOL,
                atol=ODE_ATOL,
            )
        except Exception as exc:
            raise RuntimeError(f"ODE integration failed: {exc}") from exc
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y.T   # shape (n_timepoints, n_species)

    def _simulateSimple(self,
            x0: np.ndarray, 
            time_arr: np.ndarray) -> np.ndarray:
        """
        Integrate the discovered ODE forward from *x0* over *time_arr*.

        For a linear system (poly_degree=1, include_bias=True), the ODE is:
            dz/dt = A @ z + b
        where ``A`` is the state-coefficient matrix and ``b`` is the constant
        (bias) vector.  We solve this exactly using an augmented matrix
        exponential on the extended state ``[z; 1]``, which correctly handles
        the affine term without resorting to a first-order Euler approximation.

        Assumes uniform time steps and that parameter checks have been done
        in :meth:`_simulate`.

        Args
        ----
        x0 : np.ndarray
            Initial state vector in physical units.
        time_arr : np.ndarray
            Time points at which to evaluate the solution.

        Returns
        -------
        np.ndarray
            Predicted concentrations, shape ``(n_timepoints, n_species)``.
        """
        # Convert the initial value to the standardized inputs
        z0 = self._scaler.normalize(x0)
        # Extract the A matrix (state coefficients) and b vector (constant/bias).
        coef_arr = self.model.coefficients()   # shape (n_species, n_features)
        A = coef_arr[:, 1:]        # state terms (columns after bias)
        b = coef_arr[:, 0]         # constant / bias term

        # Build the augmented matrix for the affine system:
        #   d/dt [z; 1] = [[A, b], [0, 0]] @ [z; 1]
        aug_size = self.num_species + 1
        M_aug = np.zeros((aug_size, aug_size))
        M_aug[:self.num_species, :self.num_species] = A
        M_aug[:self.num_species, self.num_species] = b

        # Compute the discrete-time transition matrix via matrix exponential.
        dt = float(np.mean(np.diff(time_arr)))  # assume uniform time steps
        Md = expm(M_aug * dt)
        Ad = Md[:self.num_species, :self.num_species]
        Bd = Md[:self.num_species, self.num_species]

        # Iteratively propagate the standardized state.
        zpreds: list[np.ndarray] = [z0.copy()]
        for _ in time_arr[1:]:
            z_next = Ad @ zpreds[-1] + Bd
            zpreds.append(z_next)
        zpred_arr = np.array(zpreds)

        # Denormalize back to physical units.
        return self._scaler.denormalize(zpred_arr)

    @classmethod
    def analyzePerturbations(
        cls,
        model: Model | int,
        training_df: pd.DataFrame = NULL_DF,
        threshold: float = 0.001,
        perturbations: list[float] = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5],
        col_percentile: str = cn.COL_P10,
        perturbation_species_fraction: float = 1.0,
        figsize: tuple[float, float] | None = None,
        poly_degree: int = 1,
        frac_scatter_skip: float = 0.2,
        plot_species_names: list[str] | None = None,
        subtitle: str = "Perturbation Analysis",
        is_analyze_model: bool = True,
        is_analyze_species: bool = True,
        is_plot: bool = True,
    ) -> AnalyzePerturbationsResult:
        '''Fit on training_df; evaluate timecourse accuracy on perturbed timecourses.

        For each value in *perturbations* a fresh Timecourse is simulated from
        *model* with that ``perturbation_value_fraction``.  The SINDy model
        (fitted on the unperturbed *training_df*) is evaluated against each
        perturbed timecourse using absolute relative error (ARE) accuracy.

        When *is_analyze_model* and/or *is_analyze_species* are True, the returned DataFrame
        includes rows at that level of aggregation:
            - ``COL_AGGREGATION_TYPE == 'model'``  -- one model-level row per
              perturbation value with aggregated statistics across species.
            - Species-level rows (aggregation type = species name) -- one row
              per species per perturbation with per-species ARE values.

        By default both levels are included. Set ``is_analyze_model=False`` or
        ``is_analyze_species=False`` to drop the corresponding level from the output.

        When *is_plot* is True, a trajectory comparison figure is also shown
        (the chosen percentile accuracy shown in the legend for visual context).

        Parameters
        ----------
        model : Model
            Used to simulate ground-truth timecourses for each perturbation.
        training_df : pd.DataFrame
            Unperturbed timecourse used to fit the SINDy model.
        threshold : float
            STLSQ sparsity threshold.
        perturbations : list[float]
            Signed fractional perturbation values (e.g. [-0.05, 0.0, 0.05]).
        col_percentile : str
            The column name in the accuracy dataframe
        perturbation_species_fraction : float
            Fraction of species whose initial values are perturbed if their initial value > 0.
        figsize : tuple, optional
            Figure size in inches.  Auto-sized when None.
        poly_degree : int
            Degree of the polynomial library.
        frac_scatter_skip : float
            Scatter-plot density: num_skip = max(1, int(n_points * frac_scatter_skip)).
        is_plot : bool
            Show a trajectory comparison figure when True.
        is_analyze_model : bool
            Include model-level rows (aggregation_type='model') in the returned DataFrame.
        is_analyze_species : bool
            Include per-species rows (aggregation_type=species name) in the returned DataFrame.
        plot_species_names : list[str] | None
            List of species to plot. If None, all species are plotted.

        Returns
        -------
        AnalyzePerturbationsResult
            A named tuple containing the accuracy metrics and the plot figure.
                pd.DataFrame: Accuracy metrics (one row per perturbation at each requested aggregation level)
                fig
        '''
        if isinstance(model, int):
            model = Model.makeBiomodel(model_num=model)
        if training_df is NULL_DF:
            training_df = TimecourseIterator().getTimecourse(model.model_name).timecourse_df
        # Initializations
        fig = None
        modifable_species_names = model.getModifableSpecies()
        if model.num_species == 0:
            fraction_species_perturbable = 0.0
        else:
            fraction_species_perturbable = len(modifable_species_names) / model.num_species
        start_time = float(training_df.index[0])
        end_time = float(training_df.index[-1])
        num_point = len(training_df)

        disc = cls(training_df, threshold=threshold, poly_degree=poly_degree)
        if plot_species_names is None:
            plot_species_names = disc.species_names
        disc.fit()

        plot_records: list[
            tuple[float, pd.DataFrame, pd.DataFrame | None, pd.Series]
        ] = []
        score = Score(serialization_path="", is_persist=False)
        for p in perturbations:
            tc = Timecourse(
                model=model,
                start_time=start_time,
                end_time=end_time,
                num_point=num_point,
                perturbation_value_fraction=p,
                perturbation_species_fraction=perturbation_species_fraction,
            )
            test_df = tc.timecourse_df
            disc._checkColumns(test_df.columns.tolist())
            pred_df = None
            try:
                pred_df = disc.predict(test_df)
                score.add(test_df, pred_df, system_id=str(p))
            except Exception:
                pass
            if is_plot and pred_df is not None:
                new_df = score.score_df[score.score_df[cn.COL_SYSTEM_ID] == str(p)]
                new_ser = new_df[col_percentile]
                new_ser.index = new_df[cn.COL_AGGREGATION_TYPE]
                plot_records.append((p, test_df, pred_df, new_ser))
        # Build per-row records so each row carries its own perturbation value.
        _META_COLS = [cn.COL_PERTURBATION, COL_FRACTION_SPECIES_PERTURBABLE, cn.COL_SYSTEM_ID]

        rows: list[dict] = []
        if not score.score_df.empty:
            for p in perturbations:
                if is_analyze_model:
                    model_row_df = (
                        score.score_df[
                            (score.score_df[cn.COL_SYSTEM_ID] == str(p)) &
                            (score.score_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL)
                        ]
                    )
                    if not model_row_df.empty:
                        for _, r in model_row_df.iterrows():
                            rec = {**r.to_dict(), **{cn.COL_PERTURBATION: p,
                                    COL_FRACTION_SPECIES_PERTURBABLE: fraction_species_perturbable,
                                    cn.COL_SYSTEM_ID: model.model_name}}
                            rows.append(rec)
                if is_analyze_species:
                    species_row_df = (
                        score.score_df[
                            (score.score_df[cn.COL_SYSTEM_ID] == str(p)) &
                            (score.score_df[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL)
                        ]
                    )
                    if not species_row_df.empty:
                        for _, r in species_row_df.iterrows():
                            rec = {**r.to_dict(), **{cn.COL_PERTURBATION: p,
                                    COL_FRACTION_SPECIES_PERTURBABLE: fraction_species_perturbable,
                                    cn.COL_SYSTEM_ID: model.model_name}}
                            rows.append(rec)
        accuracy_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(dict.fromkeys(_META_COLS + list(score.score_df.columns))))
        # Construct plots
        if is_plot and plot_records:
            n = len(plot_species_names)
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            if figsize is None:
                figsize = (5 * ncols, 3.5 * nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            fig.suptitle(subtitle, fontsize=14)
            num_skip = max(1, int(num_point * frac_scatter_skip))
            for pos_idx, sp_name in enumerate(plot_species_names):
                sp_idx = disc.species_names.index(sp_name)
                ax_row, ax_col = divmod(pos_idx, ncols)
                ax = axes[ax_row][ax_col]
                sp_col = disc.species_cols[sp_idx]
                for p_idx, (p, test_df, pred_df, accuracy_ser) in enumerate(plot_records):
                    acc = f"{accuracy_ser.loc[sp_name]:.3f}" if np.isfinite(accuracy_ser.loc[sp_name]) else "0"
                    color = f"C{p_idx}"
                    ax.scatter(
                        test_df.index[::num_skip],
                        test_df[sp_col][::num_skip],
                        s=10, color=color, alpha=0.6,
                        label=f"vfrac={p:.2f}, Accuracy={acc}",
                    )
                    if pred_df is not None:
                        ax.plot(
                            pred_df.index, pred_df[sp_name],
                            linestyle="--", color=color, lw=1.5,
                        )
                ax.set_title(sp_name)
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            for idx in range(n, nrows * ncols):
                ax_row, ax_col = divmod(idx, ncols)
                axes[ax_row][ax_col].set_visible(False)
            fig.tight_layout()
            plt.show()
            plt.close(fig)
        return AnalyzePerturbationsResult(df=accuracy_df, fig=fig)

    def calculateSpeciesScores(self, score_type: str = "timecourse",
            test_df: pd.DataFrame = NULL_DF,
            col_percentile: str = cn.COL_P10) -> dict[str, float]:
        """Compute accuracies for each species in the fitted model.

        Parameters
        ----------
        score_type : str
            The type of score to compute.  The default is "timecourse".
            ``"timecourse"`` (default) – computes R² on the timecourses.
            ``"derivative"`` (default) – computes R² on the numerical time
            derivatives, which is fast and always works.
            ``"simulation"`` – integrates the ODE forward and compares
            trajectories; more informative but may fail for stiff systems or
            poorly-identified models.
        test_df : pd.DataFrame, optional
            If provided, R² is computed against this DataFrame instead of the
            training data.  Must have the same column structure as the training
            DataFrame.
        percentile : str
            The column name in the score DataFrame to use for R².  Default is
            ``"p10"`` (10th percentile).  Other options include ``"mean"``,
            ``"p50"``, ``"p90"``, etc.

        Returns
        -------
        dict mapping species name → R² (clamped to [0, 1])
        """
        if not col_percentile in cn.STATISTICS:
            raise ValueError(f"Invalid percentile '{col_percentile}'. Must be one of {cn.STATISTICS}.")  
        #
        self._require_fitted()
        detail_df = self.getScoreDetails(test_df=test_df, score_type=score_type)
        species_ser = detail_df[detail_df[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL].copy()
        result: dict[str, float] = {}
        for i, sp_name in enumerate(self.species_names):
            if i < len(species_ser):
                raw = float(species_ser.iloc[i][col_percentile])
                result[sp_name] = self._normalize_rsq(raw)
            else:
                result[sp_name] = 0.0
        return result
    
    def fit(self) -> "SystemDiscovery":
        """Fit the SINDy model to the data.

        Returns
        -------
        self
        """
        Z = self._scaler.normalize(self._X_arr)
        # Fit the normalized value
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            self.model.fit(Z, t=self._time_arr, feature_names=self.species_names)
        if _caught:
            print("Warnings from model.fit():")
            for w in _caught:
                print(f"  {w.category.__name__}: {w.message}")
        if self.bias_species is not None:
            allowed = set(self.bias_species)
            for i, name in enumerate(self.species_names):
                if name not in allowed:
                    self.model.optimizer.coef_[i, 0] = 0.0
        self._applyThreshold()
        # Check that the features align with the species names.
        if not all([n1 == n2 for n1, n2 in zip(self.species_names, self.model.feature_names)]):  # type: ignore
            raise RuntimeError(
                "Mismatch between species names and model feature names after fitting."
            )
        self.is_fitted = True
        return self

    def getEquations(self) -> Dict[str, str]:
        """Return a dict mapping species name → string representation of its ODE."""
        self._require_fitted()
        equation_dct = {self.species_names[n]: eq for n, eq in enumerate(self.model.equations())}
        return equation_dct

    def getNonzeroTerms(self) -> dict[str, int]:
        """Return a dict mapping species name → number of non-zero terms in its ODE."""
        self._require_fitted()
        coefs = self.model.coefficients()  # shape (n_species, n_features)
        return {
            sp_name: np.sum(np.abs(coefs[i]) > 1e-10)  # type: ignore
            for i, sp_name in enumerate(self.species_names)
        }

    def getScoreDetails(self, test_df: pd.DataFrame = NULL_DF, score_type: str = "timecourse") -> pd.DataFrame:
        """
        Calculates evaluation metrics for the fitted model return information
        about the model as a whole and the individual species.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, evaluation is performed against this DataFrame instead of the training data.
        score_type : str
            The type of score to calculate.  Must be one of:
            - ``"derivative"``: R² on predicted vs numerical derivatives of concentrations.
            - ``"timecourse"``: R² for the species timecourses

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the score information for the model and each species.
        """
        score = Score()
        score_df = pd.DataFrame()
        if score_type == "derivative":
            if test_df is NULL_DF:
                test_df = self.df
            pred_arr = self.predictAllDerivatives(test_df.to_numpy(dtype=float))
            pred_df = pd.DataFrame(pred_arr[:-1], index=test_df.index[1:],
                    columns=self.species_names)
            score_df = score.add(self.Xdot_df, pred_df)
        elif score_type == "timecourse":
            pred_df = self.predict()
            score_df =score.add(self.df, pred_df)
        else:
            raise ValueError(f"Invalid score_type '{score_type}'. Must be 'derivative' or 'timecourse'.")
        #
        return score_df
    
    def getScoreAggregatedBySpecies(self, test_df: pd.DataFrame = NULL_DF, score_type: str = "timecourse",
                statistic_column: str = "p95") -> Dict[str, float]:
        """Aggregate species-level scores into a model-level statistics

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, evaluation is performed against this DataFrame instead of the training data.
        score_type : str
            The type of score to calculate.  Must be one of:
            - ``"derivative"``: R² on predicted vs numerical derivatives of concentrations.
            - ``"timecourse"``: R² for the species timecourses
        statistic_column : str
            The column name in the score DataFrame to aggregate

        Returns
        -------
        dict[str, float]
            Dictionary containing aggregated model-level scores for mean, min, max, median
        """
        self._require_fitted()
        df = self.getScoreDetails(test_df=test_df, score_type=score_type)
        species_df = df[df[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
        if species_df.empty:
            raise ValueError("No model-level score found in the provided DataFrame.")
        return {
            "mean": float(species_df[statistic_column].mean()),
            "min": float(species_df[statistic_column].min()),
            "max": float(species_df[statistic_column].max()),
            "median": float(species_df[statistic_column].median()),
        }

    @classmethod
    def makeBiomodel(
        cls,
        model_name: str,
        *,
        threshold: float = 0.01,
        poly_degree: int = 1,
        timecourse: Timecourse | None = None,
    ) -> "SystemDiscovery":
        """Create a SystemDiscovery from a BioModel timecourse.

        Parameters
        ----------
        model_name : str
            BioModel identifier (e.g. ``'BIOMD0000000003'``).
        threshold : float
            STLSQ sparsity threshold passed to ``SystemDiscovery``.
        poly_degree : int
            Degree of the polynomial library.
        timecourse : Timecourse | None
            Pre-loaded timecourse.  When ``None``, the timecourse is loaded
            from the default zip archive via ``TimecourseIterator``.
        """
        if timecourse is None:
            timecourse = TimecourseIterator().getTimecourse(model_name)
        return cls(timecourse.timecourse_df, threshold=threshold, poly_degree=poly_degree)

    def plotResult(
        self,
        test_df: pd.DataFrame = NULL_DF,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        is_plot: bool = True,
        num_true_point: int = 30,
        plot_species_names: list[str] | None = None,
        subtitle: str = "",
    ) -> plt.Figure:  # type: ignore
        """Plot observed vs. model-simulated trajectories for each species.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.  Auto-sized if *None*.
        xlim : tuple, optional
            X-axis limits ``(left, right)``.  Auto-sized if *None*.
        is_plot: bool
            Show the figure when True.  Set to False when embedding in a larger
            figure or saving manually.
        num_true_point : int
            Number of true points to plot.
        plot_species_names : list[str] | None
            List of species names to plot.  If *None*, all species are plotted.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if plot_species_names is None:
            plot_species_names = self.species_names
        if test_df is not NULL_DF:
            X = test_df.values
            time_arr = test_df.index.values
        else:
            X = self._X_arr
            time_arr = self._time_arr
            test_df = pd.DataFrame(X, index=time_arr, columns=self.species_names)
        #
        self._require_fitted()

        #n = len(self.species_names)
        n = len(plot_species_names)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(subtitle, fontsize=14, fontweight="bold", y=1.01)

        try:
            pred_df = self.predict(test_df)
            prediction_ok = True
        except Exception as exc:
            warnings.warn(f"Prediction failed for plotting: {exc}")
            pred_df = None
            prediction_ok = False

        PERCENTILE = "p10"
        score_dct = self.calculateSpeciesScores(score_type="timecourse", test_df=test_df,
                col_percentile=PERCENTILE)

        if num_true_point is None:
            num_true_point = DEFAULT_NUM_TRUE_POINT
        num_skip_point = max(1, len(time_arr) // num_true_point)
        ymax = max(X.max().max(), pred_df.max().max() if pred_df is not None else 0)
        ymin = min(X.min().min(), pred_df.min().min() if pred_df is not None else 0)
        ymax = ymax if np.isfinite(ymax) else None
        ymin = ymin if np.isfinite(ymin) else None
        plot_idx = 0
        for idx, name in enumerate(self.species_names):
            if not name in plot_species_names:
                continue
            row, col = divmod(plot_idx - 1, ncols)
            plot_idx += 1
            ax = axes[row][col]
            color = f"C{idx}"
            ax.scatter(time_arr[::num_skip_point], X[::num_skip_point, idx], s=20, color=color, label=f"{name} (observed)")
            if prediction_ok and pred_df is not None:
                ax.plot(pred_df.index.to_numpy(), pred_df[name].to_numpy(), "-", lw=2, color=color, label=f"{name} (predicted)")
            score = score_dct.get(name, float("nan"))
            if len(name) > 20:
                name = name[:7] + "..." + name[-10:]
            title = f"{name}"
            if not np.isnan(score):
                title += f"   {PERCENTILE} accuracy={score:.4f}"
            low_y = X[:, idx].min()
            high_y = X[:, idx].max()
            if np.isclose(low_y, high_y):
                low_y -= 0.1*low_y
                high_y += 0.1*high_y
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_ylim(ymin, ymax)
            #ax.set_ylim(low_y - 0.1 * abs(high_y - low_y), high_y + 0.1 * abs(high_y - low_y))
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if is_plot:
            plt.show()
        return fig

    def plot_coefficient_heatmap(
        self,
        figsize: tuple[float, float] | None = None,
        is_plot: bool = True,
    ) -> plt.Figure:  # type: ignore
        """Visualise the coefficient matrix as a heatmap.

        Each row is a library feature; each column is a species.
        Non-zero entries (active terms) are highlighted.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.  Auto-sized if *None*.
        is_plot: bool
            Show the figure when True.  Set to False when embedding in a larger

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_fitted()

        df_coef = self.summary().T
        if df_coef.empty:
            print("No non-zero coefficients found; heatmap skipped.")
            return plt.figure()

        if figsize is None:
            figsize = (max(6, len(df_coef.columns) * 1.5), max(4, len(df_coef) * 0.5))

        fig, ax = plt.subplots(figsize=figsize)
        max_coef = np.abs(df_coef.values).max()
        cax = ax.imshow(df_coef.values, aspect="auto", cmap="RdBu_r",
                vmin=-max_coef, vmax=max_coef)
        fig.colorbar(cax, ax=ax, label="Coefficient value")

        ax.set_xticks(range(len(df_coef.columns)))
        ax.set_xticklabels(df_coef.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df_coef.index)))
        ax.set_yticklabels(df_coef.index)
        ax.set_title("SINDy Coefficient Matrix (non-zero terms)", fontweight="bold")

        # Annotate cells
        for i in range(len(df_coef.index)):
            for j in range(len(df_coef.columns)):
                val = df_coef.iloc[i, j]
                if abs(val) > 1e-10:  # type: ignore
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > df_coef.values.max() * 0.5 else "black",  # type: ignore
                    )

        fig.tight_layout()
        if is_plot:
            plt.show()
            plt.close(fig)
        return fig

    def _checkColumns(self, candidate_columns: list[str]) -> None:
        """Check that the candidate columns are present in the DataFrame."""
        differences = set(candidate_columns).symmetric_difference(set(self.df.columns))
        if differences:
            raise ValueError(f"Mismatched columns in DataFrame: {differences}")    

    def predict(self, test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame:
        """Integrate the discovered ODE and return predicted concentrations.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, integration starts from ``test_df.values[0]`` and
            evaluates at ``test_df.index`` time points.  When omitted, the
            training initial condition and time grid are used.

        Returns
        -------
        pd.DataFrame
            Predicted concentrations with time as the index and one column per
            species.  Raises ``RuntimeError`` if the ODE integrator fails.
            columns: species names; index: time points
        """
        if not test_df.empty:
            self._checkColumns(test_df.columns.tolist())
        self._require_fitted()
        if test_df is not NULL_DF:
            x0 = test_df.to_numpy(dtype=float)[0, :]
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            x0 = None
            time_arr = None
        X_sim = self._simulate(x0=x0, time_arr=time_arr)
        t_idx = time_arr if time_arr is not None else self._time_arr
        return pd.DataFrame(X_sim, index=t_idx, columns=self.species_names)
    
    def predictAllDerivatives(self, X: np.ndarray = cn.NULL_ARRAY) -> np.ndarray:
        """Evaluate the fitted ODE's right-hand side at all states (no integration).

        Parameters
        ----------
        X : np.ndarray
            State vector in physical units, shape (n_species,), in the same
            species order as `self.species_names`.

        Returns
        -------
        np.ndarray
            Derivative dx/dt at all time points, in physical units, shape (n_timepoints, n_species).
        """
        if X is cn.NULL_ARRAY:
            X = self._X_arr
        else:
            if X.ndim == 1:
                X = X.reshape(1, -1) 
        self._require_fitted()
        Z = self._scaler.normalize(X)
        dZ_dt = self.model.predict(Z)
        return np.array(self._scaler.denormalize(dZ_dt), dtype=float)  # type: ignore

    def predictOneStepDerivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted ODE's right-hand side at a single state (no integration).

        Parameters
        ----------
        x : np.ndarray
            State vector in physical units, shape (n_species,), in the same
            species order as `self.species_names`.

        Returns
        -------
        np.ndarray
            Derivative dx/dt at `x`, in physical units, shape (n_species,).
        """
        self._require_fitted()
        z = self._scaler.normalize(x)
        dz_dt = self.model.predict(z.reshape(1, -1))[0]
        return np.array(self._scaler.denormalize(dz_dt), dtype=float)

    def get_derivatives(self, test_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the differentiated values computed by PySINDy's differentiation method.

        After fitting, this returns the numerical time derivatives of each species
        as computed by the configured differentiation strategy (``"smooth"``,
        ``"finite"``, or ``"spectral"``).  These are *not* the model-predicted
        right-hand-side values — they are the raw differentiated data used during
        fitting.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, derivatives are computed for this DataFrame using a
            simple finite-difference approximation on the normalized data and
            denormalized back to physical units.  When ``None``, returns the
            derivatives from the original training data (which were computed
            during ``fit``).

        Returns
        -------
        pd.DataFrame
            Derivatives with time as the index and one column per species.
            Shape is ``(n_samples, n_species)`` when *test_df* is given, or
            ``(n_samples - 1, n_species)`` for training data (since PySINDy's
            differentiation drops the first sample).

        Raises
        ------
        RuntimeError
            If ``.fit()`` has not been called yet.

        Example
        -------
        >>> disc.fit()
        >>> X_dot = disc.get_derivatives()   # training data derivatives
        >>> X_dot_test = disc.get_derivatives(test_df)  # for new data
        """
        self._require_fitted()
        if test_df is None:
            return self.Xdot_df.copy()
        if not test_df.empty:
            self._checkColumns(test_df.columns.tolist())

        # For test data, compute finite-difference derivatives on normalized
        # values and denormalize back to physical units.
        Z = self._scaler.normalize(test_df.to_numpy(dtype=float))
        t_test = test_df.index.to_numpy(dtype=float)
        dZ_dt = np.diff(Z, axis=0) / np.diff(t_test).reshape(-1, 1)
        X_dot_arr = np.array(self._scaler.denormalize(dZ_dt), dtype=float)
        return pd.DataFrame(
            X_dot_arr,
            index=t_test[1:],
            columns=self.species_names,
        )

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE equations."""
        print(self.__str__())

    def score(self, score_type: str = "derivative", score_column: str = "p50") -> float:
        """
        Calculates a single measure of model performance.
            derivative: minimum value of R² across all species
            timecourse: maximum value of ARE across all species
        

        Parameters
        ----------
        score_type : str
            The type of score to calculate.  Must be one of:
            - ``"derivative"``
            - ``"timecourse"``

        Returns
        -------
        float
            The calculated score.
            - ``"derivative"``: R² on predicted vs numerical derivatives of concentrations.
            - ``"timecourse"``: R² for the species timecourses
        """
        score_detail_df = self.getScoreDetails(score_type=score_type)
        model_sel = score_detail_df[cn.COL_AGGREGATION_TYPE] == "model"
        if score_type == "derivative":
            result = float(score_detail_df[model_sel][score_column].iloc[0])
            return result
        elif score_type == "timecourse":
            species_sel = score_detail_df[cn.COL_AGGREGATION_TYPE] != "model"
            vals = score_detail_df[species_sel][score_column].to_numpy(dtype=float)
            result = float(np.max(vals))
            return result
        else:
            raise ValueError(f"Invalid score_type '{score_type}'. Must be 'derivative' or 'timecourse'.")

    def summary(self, entry_threshold: float = 0) -> pd.DataFrame:
        """Return a DataFrame of denormalized non-zero coefficients for all species.

        Coefficients are adjusted from the normalized fit back to original-space
        units: each raw coefficient c' is multiplied by σ_i / Π_j σ_j^{p_j},
        where σ_i is the std of the output species (column) and σ_j^{p_j} are
        the stds of the input species in the polynomial term (row) raised to
        their powers.

        Rows are candidate library terms; columns are species.

        Parameters
        ----------
        entry_threshold : float
            Rows are kept only if the maximum absolute normalized coefficient
            |c_norm| = |c_physical| * Π(σ_j^{p_j}) / σ_i exceeds this value.
            Since |c_norm| is dimensionless (contribution relative to one
            standard-deviation of the derivative), ``entry_threshold=1`` retains
            terms whose effect is at least one standard-deviation-equivalent.
            Default ``0`` (show all nonzero rows; sparsity is controlled by the
            constructor ``threshold`` argument via :meth:`fit`).

        Returns
        -------
        pd.DataFrame
        """
        self._require_fitted()
        feature_names = self.model.get_feature_names()
        coefs = self.model.coefficients()          # shape (n_species, n_features)
        col_names = [f"d{n}/dt" for n in self.species_names]
        df_norm = pd.DataFrame(coefs.T, index=feature_names, columns=col_names)
        # Filter on normalized coefficients — exclude constant species whose fallback
        # scaling makes c_norm values meaningless for the retention decision.
        constant_cols = self._scaler._constant_cols
        variable_cols = [col for sp, col in zip(self.species_names, col_names)
                if sp not in constant_cols]
        eval_cols = variable_cols if variable_cols else col_names
        keep_mask = df_norm[eval_cols].abs().T.max() > entry_threshold
        df_norm = df_norm[keep_mask].copy()        # type: ignore
        # Denormalize surviving rows
        df_coef = df_norm.copy()
        for factor_str, row in df_norm.iterrows():
            for sp_name, col in zip(self.species_names, col_names):
                df_coef.loc[factor_str, col] = self._scaler.denormalizeCoordinate(
                    sp_name, factor_str, row[col])
        return df_coef  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def discoverNetwork(
    df: pd.DataFrame | list[pd.DataFrame],
    test_df: pd.DataFrame = NULL_DF,
    threshold: float = 0.01,
    alpha: float = 0.05,
    differentiation: DifferentiationMethod = "smooth",
    poly_degree: int = 1,
    include_bias: bool = True,
    species_names: list[str] | None = None,
    is_plot_comparisons: bool = True,
    is_plot_heatmap: bool = True,
    xlim: tuple[float, float] | None = None,
    plot_species_names: list[str] | None = None,
    subtitle: str = "",
    is_plot: bool = True,
    is_print_equations: bool = True,
    is_print_accuracy: bool = True,
) -> SystemDiscovery:
    """One-shot helper: construct, fit, print, and optionally plot.

    Parameters
    ----------
    df : pd.DataFrame or list[pd.DataFrame]
        One trajectory or a list of trajectories (see :class:`SystemDiscovery`).
    threshold : float
        STLSQ sparsity threshold.
    alpha : float
        Ridge regularisation.
    differentiation : str
        ``"smooth"`` | ``"finite"`` | ``"spectral"``.
    poly_degree : int
        1 (linear) or 2 (quadratic).
    include_bias : bool
        Include a constant term in the library.
    species_names : list[str] | None
        Human-readable species labels.
    plot_species_names : list[str] | None
        List of species names to plot.  If *None*, all species are plotted.
    xlim : tuple[float, float] | None
    is_plot : bool
        Show plots when True.  Set to False when embedding in a larger figure or saving manually
    is_plot_heatmap : bool
        Show coefficient heatmap.
    is_print_equations : bool
        Print the discovered equations.
    is_print_accuracy : bool
        Print accuracy values for each species.
    subtitle : str
        Optional subtitle for the plots.

    Returns
    -------
    SystemDiscovery
        Fitted discovery object.

    Example
    -------
    >>> disc = discoverNetwork(df, threshold=0.02)
    >>> disc.print_equations()
    >>> summary = disc.summary()
    """
    disc = SystemDiscovery(
        df,   # type: ignore
        threshold=threshold,
        alpha=alpha,
        differentiation=differentiation,
        poly_degree=poly_degree,
        include_bias=include_bias,
        species_names=species_names,
        is_normalize=True,
    )
    if not test_df.empty:
            disc._checkColumns(test_df.columns.tolist())
    disc.fit()
    if is_print_equations:
        print("Discovered equations:")
        disc.printEquations()

    if is_print_accuracy:
        accuracy_dct = disc.calculateSpeciesScores(score_type="timecourse", test_df=test_df)
        print("Accuracy for species timecourses:")
        for name, val in accuracy_dct.items():
            print(f"  {name}: {val:.6f}")
        print()
        # Print accuracy for time derivatives
        accuracy_dct = disc.calculateSpeciesScores(score_type="derivative", test_df=test_df)
        print("Accuracy for species time derivatives:")
        for name, val in accuracy_dct.items():
            print(f"  {name}: {val:.6f}")
        print()

    fig = None
    if is_plot_comparisons:
        fig = disc.plotResult(test_df, xlim=xlim, plot_species_names=plot_species_names, is_plot=is_plot
                , subtitle=subtitle)
    if is_plot_heatmap:
        disc.plot_coefficient_heatmap(is_plot=is_plot)

    return DiscoverNetworkResult(sd=disc, fig=fig)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Demo  (Brusselator – a classic chemical oscillator)
# ---------------------------------------------------------------------------


def _generate_brusselator(
    t_end: float = 20.0,
    n_points: int = 4000,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Brusselator data for testing.

    The Brusselator is a simple two-species chemical oscillator:
        dX/dt = A + X²Y - (B+1)X
        dY/dt = BX - X²Y
    with A=1, B=3 → limit cycle.
    """
    rng = np.random.default_rng(seed)
    A, B = 1.0, 3.0

    def brusselator(t, z):
        X, Y = z
        return [A + X**2 * Y - (B + 1) * X, B * X - X**2 * Y]

    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(brusselator, [0, t_end], [0.5, 2.0], t_eval=t_eval, rtol=1e-8)

    X_data = sol.y[0] + rng.normal(0, noise_std, n_points)
    Y_data = sol.y[1] + rng.normal(0, noise_std, n_points)

    df = pd.DataFrame({"time": t_eval, "X": X_data, "Y": Y_data})
    df = df.set_index("time")
    return df


if __name__ == "__main__":
    print("Brusselator demo\n" + "-" * 40)
    print("True equations:")
    print("  dX/dt =  1  +  X²Y  -  4X")
    print("  dY/dt = 3X  -  X²Y\n")

    df_demo = _generate_brusselator(noise_std=0.01)

    disc = discoverNetwork(
        df_demo,
        threshold=0.01,
        alpha=0.01,
        differentiation="smooth",
        poly_degree=3,
        include_bias=True,
        is_plot_comparisons=True,
        is_plot_heatmap=True,
    )

    print("\nCoefficient summary:")
    print(disc.summary().to_string())