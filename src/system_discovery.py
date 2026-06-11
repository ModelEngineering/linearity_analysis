"""
Discovery of a system of differential equations from data using PySINDy, tailored for chemical reaction networks.
====================================================================================
Discovers a system of ODEs from time-series concentration data using PySINDy
that estimate the derivatives of each species as a sparse linear combination of polynomial features of the species concentrations.
Assumes rate laws are at most quadratic in the species concentrations (i.e.,
the library includes constant, linear, and pairwise-product terms).

Supports up to 10 chemical species.

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
    from chemical_network_sindy import NetworkRateDiscovery

    discovery = NetworkRateDiscovery(
        df,
        threshold=0.05,          # STLSQ sparsity threshold
        alpha=0.05,              # L2 regularisation
        differentiation="smooth" # "smooth" | "finite" | "spectral"
    )
    discovery.fit()
    discovery.print_equations()
    discovery.plot_results()
    summary = discovery.summary()

To Do:
1. Integrate normalizer
"""
from dataclasses import dataclass
from src.model import Model  # type: ignore
from src.scaler import Scaler  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pysindy as ps # type: ignore
from pysindy.feature_library import PolynomialLibrary # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from typing import Literal, cast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

NULL_DF = pd.DataFrame()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SPECIES = 200
DifferentiationMethod = Literal["smooth", "finite", "spectral"]


# ---------------------------------------------------------------------------
# ScoreInfo
# ---------------------------------------------------------------------------

@dataclass
class ScoreInfo:
    min: float
    median: float
    max: float
    values: list[float]
    num_nonzero_term: int


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
        df: pd.DataFrame | list[pd.DataFrame],
        threshold: float = 0.01,
        alpha: float = 0.05,
        differentiation: DifferentiationMethod = "smooth",
        poly_degree: int = 2,
        include_bias: bool = True,
        species_names: list[str] | None = None,
        bias_species: list[str] | None = None,
        is_normalize: bool = True,
    ) -> None:
        self._is_normalize = is_normalize

        dfs: list[pd.DataFrame] = [df] if isinstance(df, pd.DataFrame) else list(df)
        if not dfs:
            raise ValueError("`df` must be a non-empty DataFrame or list of DataFrames.")
        for d in dfs:
            self._validate_dataframe(d)

        ref_cols = list(dfs[0].columns)
        for i, d in enumerate(dfs[1:], start=1):
            if list(d.columns) != ref_cols:
                raise ValueError(
                    f"All DataFrames must have identical columns. "
                    f"DataFrame 0: {ref_cols}, DataFrame {i}: {list(d.columns)}."
                )

        self.df = dfs[0].copy()
        self.threshold = threshold
        self.alpha = alpha
        self.differentiation = differentiation
        self.poly_degree = poly_degree
        self.include_bias = include_bias

        # Extract time and concentration arrays
        species_cols = ref_cols
        if len(species_cols) > MAX_SPECIES:
            raise ValueError(
                f"DataFrame contains {len(species_cols)} species columns; "
                f"maximum supported is {MAX_SPECIES}."
            )

        self.species_cols = species_cols
        self._X_list: list[np.ndarray] = [d[species_cols].to_numpy(dtype=float) for d in dfs]
        self._time_list: list[np.ndarray] = [d.index.to_numpy(dtype=float) for d in dfs]
        # First trajectory used for simulation and plotting
        self.time_arr: np.ndarray = self._time_list[0]
        self.X: np.ndarray = self._X_list[0]
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
        scaler_df = pd.concat(dfs, ignore_index=True)
        scaler_df.columns = pd.Index(self.species_names)
        self._normalizer = Scaler(scaler_df, is_null_scaler=not is_normalize)

        if bias_species is not None:
            invalid = set(bias_species) - set(self.species_names)
            if invalid:
                raise ValueError(
                    f"`bias_species` contains names not in species_names: {sorted(invalid)}"
                )
            self.include_bias = True
        self.bias_species: list[str] | None = bias_species

        self.model: ps.SINDy

        library = PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=True,
        )

        optimizer = ps.STLSQ(threshold=0, alpha=self.alpha)

        diff_method = self._build_differentiator()

        self.model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
            differentiation_method=diff_method,
        )
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> "SystemDiscovery":
        """Fit the SINDy model to the data.

        Returns
        -------
        self
        """
        Z_list: list[np.ndarray] = [self._normalizer.normalize(X) for X in self._X_list]
        """ if self._is_normalize:
            Z_list = [X / self._species_std for X in self._X_list]
        else:
            Z_list = self._X_list """

        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            self.model.fit(Z_list, t=self._time_list, feature_names=self.species_names)
        if _caught:
            print("Warnings from model.fit():")
            for w in _caught:
                print(f"  {w.category.__name__}: {w.message}")
        if self.bias_species is not None:
            allowed = set(self.bias_species)
            for i, name in enumerate(self.species_names):
                if name not in allowed:
                    self.model.optimizer.coef_[i, 0] = 0.0
        self._apply_threshold()
        self._is_fitted = True
        return self

    @classmethod
    def makeBiomodel(
        cls,
        model_name: str,
        *,
        threshold: float = 0.01,
        poly_degree: int = 2,
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

    @classmethod
    def perturbationAnalysis(
        cls,
        model: Model,
        training_df: pd.DataFrame,
        threshold: float,
        perturbations: list[float],
        perturbation_species_fraction: float = 1.0,
        figsize: tuple[float, float] | None = None,
        poly_degree: int = 1,
        frac_keep: float = 0.2,
        show: bool = True,
    ) -> plt.Figure:  # type: ignore
        """Fit on training_df then plot observed trajectories for each perturbation.

        For each value in *perturbations*, a ground-truth timecourse is simulated
        from *model* using that ``perturbation_value_fraction``.  The discovered
        ODE is evaluated against each timecourse and R² is reported in the legend.
        The unperturbed trajectory (fraction=0) is drawn as a solid line; all
        others are dashed.

        Parameters
        ----------
        model : Model
            Used to simulate ground-truth timecourses for each perturbation.
        training_df : pd.DataFrame
            Unperturbed timecourse used to fit the SystemDiscovery.  Index is
            time; columns are species.
        threshold : float
            STLSQ sparsity threshold.
        perturbations : list[float]
            Values of ``perturbation_value_fraction`` to evaluate.
        poly_degree : int
            Degree of the polynomial library.
        perturbation_species_fraction : float
            Fraction of species whose initial values are perturbed.  Default 1.0.
        figsize : tuple, optional
            Figure size in inches.  Auto-sized when *None*.
        frac_keep : float, optional
            Fraction of data points to keep for plotting.  Default 0.2.

        show : bool
            Call ``plt.show()`` at the end.

        Returns
        -------
        matplotlib.figure.Figure
        """
        start_time = float(training_df.index[0])
        end_time = float(training_df.index[-1])
        num_point = len(training_df)

        disc = cls(training_df, threshold=threshold, poly_degree=poly_degree)
        disc.fit()

        # Generate (perturbation_value_frac, test_df, pred_df, r2_per_species) for each value
        records: list[tuple[float, pd.DataFrame, pd.DataFrame | None, dict[str, float]]] = []
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
            try:
                pred_df: pd.DataFrame | None = disc.predict(test_df)
            except Exception:
                pred_df = None
            r2_dct = disc.calculateRsq(method="simulation", test_df=test_df)
            records.append((p, test_df, pred_df, r2_dct))

        n = len(disc.species_names)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle("Perturbation Analysis", fontsize=14, fontweight="bold")

        num_skip = int(num_point * frac_keep) 
        for sp_idx, sp_name in enumerate(disc.species_names):
            row, col = divmod(sp_idx, ncols)
            ax = axes[row][col]
            sp_col = disc.species_cols[sp_idx]
            for p_idx, (p, test_df, pred_df, r2_dct) in enumerate(records):
                r2 = r2_dct.get(sp_name, float("nan"))
                r2_str = f"{r2:.3f}" if np.isfinite(r2) else "nan"
                label = f"vfrac={p:.2f}, R²={r2_str}"
                color = f"C{p_idx}"
                ax.scatter(test_df.index[::num_skip], test_df[sp_col][::num_skip],
                        s=10, color=color,
                        alpha=0.6, label=label)
                if pred_df is not None:
                    ax.plot(pred_df.index, pred_df[sp_name], linestyle="--",
                            color=color, lw=1.5)
            ax.set_title(sp_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_coefficient_heatmap(
        self,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> plt.Figure:  # type: ignore
        """Visualise the coefficient matrix as a heatmap.

        Each row is a library feature; each column is a species.
        Non-zero entries (active terms) are highlighted.

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
        if show:
            plt.show()
        return fig

    def plotResult(
        self,
        test_df: pd.DataFrame = NULL_DF,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        num_true_point: int = 20,
    ) -> plt.Figure:  # type: ignore
        """Plot observed vs. model-simulated trajectories for each species.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.  Auto-sized if *None*.
        show : bool
            Call ``plt.show()`` at the end.  Set to ``False`` when embedding
            in a larger figure or saving manually.
        num_true_point : int
            Number of true points to plot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if test_df is not NULL_DF:
            X = test_df.values
            time_arr = test_df.index.values
        else:
            X = self.X
            time_arr = self.time_arr
        #
        self._require_fitted()

        n = len(self.species_names)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(
            "Chemical Network Discovery – Observed vs Predicted",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        try:
            pred_df = self.predict(test_df)
            prediction_ok = True
        except Exception as exc:
            warnings.warn(f"Prediction failed for plotting: {exc}")
            pred_df = None
            prediction_ok = False

        r2_vals = self.calculateRsq(method="simulation", test_df=test_df)

        num_skip_point = max(1, len(time_arr) // num_true_point)
        for idx, name in enumerate(self.species_names):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            color = f"C{idx}"
            ax.scatter(time_arr[::num_skip_point], X[::num_skip_point, idx], s=20, color=color, label=f"{name} (observed)")
            if prediction_ok and pred_df is not None:
                ax.plot(pred_df.index, pred_df[name], "-", lw=2, color=color, label=f"{name} (predicted)")
            r2 = r2_vals.get(name, float("nan"))
            title = f"{name}"
            if not np.isnan(r2):
                title += f"   R²={r2:.4f}"  # may be negative; clamping removed
            low_y = X[:, idx].min()
            high_y = X[:, idx].max()
            if np.isclose(low_y, high_y):
                low_y -= 0.1*low_y
                high_y += 0.1*high_y
            ax.set_ylim(low_y - 0.1 * abs(high_y - low_y), high_y + 0.1 * abs(high_y - low_y))
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
        if show:
            plt.show()
        return fig

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
        self._require_fitted()
        if test_df is not NULL_DF:
            x0 = test_df.to_numpy(dtype=float)[0, :]
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            x0 = None
            time_arr = None
        X_sim = self._simulate(x0=x0, time_arr=time_arr)
        t_idx = time_arr if time_arr is not None else self.time_arr
        return pd.DataFrame(X_sim, index=t_idx, columns=self.species_names)

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE equations."""
        print(self.__str__())

    def __str__(self) -> str:
        equation_str = ""
        summary_df = self.summary()
        for idx, col in enumerate(summary_df.columns):
            if idx > 0:
                equation_str = equation_str[0:-3]
            equation_str += f"\n{col} = "
            for idx in summary_df.index:
                val = cast(float, summary_df.at[idx, col])
                if np.isclose(val, 0):
                    continue
                term = f"{val:.3f} * {idx}" if idx != "1" else f"{val:.3f}"
                equation_str += f"{term} + "
        return equation_str.rstrip(" + ")

    def getNonzeroTerms(self) -> dict[str, int]:
        """Return a dict mapping species name → number of non-zero terms in its ODE."""
        self._require_fitted()
        coefs = self.model.coefficients()  # shape (n_species, n_features)
        return {
            sp_name: np.sum(np.abs(coefs[i]) > 1e-10)  # type: ignore
            for i, sp_name in enumerate(self.species_names)
        }

    def calculateRsq(self, method: str = "derivative",
            test_df: pd.DataFrame = NULL_DF) -> dict[str, float]:
        """Compute R² for each species.

        Parameters
        ----------
        method : str
            ``"derivative"`` (default) – computes R² on the numerical time
            derivatives, which is fast and always works.
            ``"simulation"`` – integrates the ODE forward and compares
            trajectories; more informative but may fail for stiff systems or
            poorly-identified models.
        test_df : pd.DataFrame, optional
            If provided, R² is computed against this DataFrame instead of the
            training data.  Must have the same column structure as the training
            DataFrame.

        Returns
        -------
        dict mapping species name → R²
        """
        self._require_fitted()
        if test_df is not NULL_DF:
            X = test_df.to_numpy(dtype=float)
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            X = None
            time_arr = None
        try:
            if method == "simulation":
                result = self._r_squared_simulation(X=X, time_arr=time_arr)
            else:
                X_list = [X] if X is not None else None
                time_list = [time_arr] if time_arr is not None else None
                result = self._r_squared_derivative(X_list=X_list, time_list=time_list)
        except Exception as exc:
            warnings.warn(f"R² computation failed: {exc}")
            result = self._r_squared_derivative()
        return result
    
    @staticmethod
    def _normalize_rsq(rsq: float) -> float:
        """Normalize R² to [0, 1] and handle NaN."""
        if np.isnan(rsq):
            return 0.0
        return max(0.0, min(1.0, rsq))

    def score(self) -> ScoreInfo:
        """Return a ScoreInfo with the min, median, and max of r_squared values."""
        values = list(self.calculateRsq().values())
        num_nonzero_term = sum(self.getNonzeroTerms().values())
        return ScoreInfo(
            min=self._normalize_rsq(float(np.min(values))),
            median=self._normalize_rsq(float(np.median(values))),
            max=self._normalize_rsq(float(np.max(values))),
            values=[self._normalize_rsq(x) for x in values],
            num_nonzero_term=num_nonzero_term,
        )

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
        constant_cols = self._normalizer._constant_cols
        variable_cols = [col for sp, col in zip(self.species_names, col_names)
                         if sp not in constant_cols]
        eval_cols = variable_cols if variable_cols else col_names
        keep_mask = df_norm[eval_cols].abs().T.max() > entry_threshold
        df_norm = df_norm[keep_mask].copy()        # type: ignore
        # Denormalize surviving rows
        df_coef = df_norm.copy()
        for factor_str, row in df_norm.iterrows():
            for sp_name, col in zip(self.species_names, col_names):
                df_coef.loc[factor_str, col] = self._normalizer.denormalizeCoordinate(
                    sp_name, factor_str, row[col])
        return df_coef  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_threshold(self) -> None:
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
                    norm_thresh = self._normalizer.normalizeThreshold(
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

    def _parse_feature_powers(self, feature_name: str) -> dict[str, int]:
        """Parse a PySINDy feature name into {species_name: power}.

        Handles: '1' → {}, 'X' → {'X': 1}, 'X^2' → {'X': 2}, 'X Y' → {'X': 1, 'Y': 1}.
        """
        if feature_name == "1":
            return {}
        powers: dict[str, int] = {}
        for factor in feature_name.split(" "):
            if not factor:
                continue
            if "^" in factor:
                name, exp = factor.split("^", 1)
                powers[name] = int(exp)
            else:
                powers[factor] = 1
        return powers

    def _r_squared_derivative(self,
            X_list: list[np.ndarray] | None = None,
            time_list: list[np.ndarray] | None = None) -> dict[str, float]:
        """R² on predicted vs numerical derivatives. Returns raw (possibly negative) values."""
        if X_list is None:
            X_list = self._X_list
        if time_list is None:
            time_list = self._time_list
        zdot_pred_parts = []
        zdot_num_parts = []
        for X, t in zip(X_list, time_list):
            Z = self._normalizer.normalize(X)
            zdot_pred_parts.append(np.array(self.model.predict(Z)))
            zdot_num_parts.append(self.model.differentiation_method(Z, t))  # type: ignore
        Zdot_pred = np.vstack(zdot_pred_parts)
        Zdot_num  = np.vstack(zdot_num_parts)
        r2 = {}
        for i, name in enumerate(self.species_names):
            y_true = Zdot_num[:, i]
            y_pred = Zdot_pred[:, i]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return r2

    def _r_squared_simulation(self,
                               X: np.ndarray | None = None,
                               time_arr: np.ndarray | None = None) -> dict[str, float]:
        """R² on simulated concentration trajectories. Returns raw (possibly negative) values."""
        if X is None:
            X = self.X
        if time_arr is None:
            time_arr = self.time_arr
        try:
            X_sim = self._simulate(x0=X[0, :], time_arr=time_arr)
            r2 = {}
            for i, name in enumerate(self.species_names):
                ss_res = np.sum((X[:, i] - X_sim[:, i]) ** 2)
                ss_tot = np.sum((X[:, i] - X[:, i].mean()) ** 2)
                r2[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return r2
        except Exception as exc:
            warnings.warn(f"Simulation R² failed: {exc}. Use method='derivative'.")
            return {name: float("nan") for name in self.species_names}

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call `.fit()` before using this method.")

    def _simulate(self,
                  x0: np.ndarray | None = None,
                  time_arr: np.ndarray | None = None) -> np.ndarray:
        """Integrate the discovered ODE forward from *x0* over *time_arr*."""
        if x0 is None:
            x0 = self.X[0, :]
        if time_arr is None:
            time_arr = self.time_arr

        def rhs(_t, x):
            z = self._normalizer.normalize(x)
            dz_dt = self.model.predict(z.reshape(1, -1))[0]
            dx_dt = self._normalizer.denormalize(dz_dt)
            return np.array(dx_dt, dtype=float)

        try:
            sol = solve_ivp(
                rhs,
                t_span=(time_arr[0], time_arr[-1]),
                y0=x0,
                t_eval=time_arr,
                method="Radau",
                rtol=1e-6,
                atol=1e-8,
            )
        except Exception as exc:
            raise RuntimeError(f"ODE integration failed: {exc}") from exc
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y.T   # shape (n_timepoints, n_species)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")
        if len(df.columns) == 0:
            raise ValueError("DataFrame must contain at least one species column.")
        if len(df.columns) > MAX_SPECIES:
            raise ValueError(
                f"DataFrame has {len(df.columns)} species columns; "
                f"maximum is {MAX_SPECIES}."
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index (time) must be strictly increasing.")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def discoverNetwork(
    df: pd.DataFrame | list[pd.DataFrame],
    test_df: pd.DataFrame = NULL_DF,
    threshold: float = 0.01,
    alpha: float = 0.05,
    differentiation: DifferentiationMethod = "smooth",
    poly_degree: int = 2,
    include_bias: bool = True,
    species_names: list[str] | None = None,
    plot: bool = True,
    heatmap: bool = True,
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
    plot : bool
        Show trajectory comparison plots.
    heatmap : bool
        Show coefficient heatmap.

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
        df,
        threshold=threshold,
        alpha=alpha,
        differentiation=differentiation,
        poly_degree=poly_degree,
        include_bias=include_bias,
        species_names=species_names,
        is_normalize=True,
    )
    disc.fit()
    disc.printEquations()

    r2 = disc.calculateRsq()
    print("R² on time derivatives per species:")
    for name, val in r2.items():
        print(f"  {name}: {val:.6f}")
    print()

    try:
        r2_sim = disc.calculateRsq(method="simulation", test_df=test_df)
        print("R² on simulated trajectories per species:")
        for name, val in r2_sim.items():
            print(f"  {name}: {val:.6f}")
        print()
    except Exception:
        pass

    if plot:
        disc.plotResult(test_df)
    if heatmap:
        disc.plot_coefficient_heatmap()

    return disc


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
        plot=True,
        heatmap=True,
    )

    print("\nCoefficient summary:")
    print(disc.summary().to_string())
