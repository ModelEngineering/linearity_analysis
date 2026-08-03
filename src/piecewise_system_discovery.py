"""Piecewise system discovery based on detecting change points in Jacobians."""

from typing import Any, List, Tuple

import collections
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from typing import cast

from src.change_point_detector import ChangePointDetector  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.jacobian_signal import JacobianSignal  # type: ignore
import src.constants as cn  # type: ignore
from src.system_discovery import SystemDiscovery  # type: ignore

NULL_DF = pd.DataFrame()

PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-points."""

    def __init__(
        self,
        timecourse: Timecourse,
        max_change_point: int = 2,
        min_fractional_reduction: float = 0.1,  
        min_subsequence_length: int = 100,
        predict_kernel_bandwidth: float = 0.5,
        **sd_kwargs: Any,
    ) -> None:
        """_summary_

        Args:
            timecourse (Timecourse): _description_
            max_change_point (int, optional): _description_. Defaults to 2.
            min_fractional_reduction (float, optional): _description_. Defaults to 0.1.
            min_subsequence_length (int, optional): _description_. Defaults to 100.
            predict_kernel_bandwidth (float, optional): _description_. Defaults to 0.5.
            **kwargs: Arguments for SystemDiscovery constructor (e.g. fit_kernel_bandwidth, model_name).
        """
        self.timecourse = timecourse
        self.max_change_point = max_change_point
        self.min_fractional_reduction = min_fractional_reduction
        self.min_subsequence_length = min_subsequence_length
        self.predict_kernel_bandwidth = predict_kernel_bandwidth
        sd_kwargs["poly_degree"] = sd_kwargs.get("poly_degree", 1)
        self._sd_kwargs = sd_kwargs

        self._subsequence_models: List[SystemDiscovery] = []
        self._subsequence_boundaries: List[Tuple[float, float]] = []
        self._subsequence_lengths: List[int] = []
        self._is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")
    
    def fit(self) -> "PiecewiseSystemDiscovery":
        """fit() steps 1-4: detect change points, fit per-subsequence models."""
        jacobian_signal = JacobianSignal(self.timecourse)
        detector = jacobian_signal.fit(max_change_point=self.max_change_point,
                min_fractional_reduction=self.min_fractional_reduction,
                min_subsequence_length=self.min_subsequence_length)
        time_arr = self.timecourse.timecourse_df.index.to_numpy(dtype=float)
        num_point = time_arr.shape[0]
        boundary_index_arr = [i.splice_start for i in detector.subsequences] + [num_point]
        # Construct the subsequence information
        self._subsequence_models = []
        self._subsequence_boundaries = []
        self._subsequence_lengths = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            subsequence_df = self.timecourse.timecourse_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < num_point else time_arr[-1]
            self._subsequence_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._subsequence_lengths.append(hi - lo)
            model = SystemDiscovery(subsequence_df, **self._sd_kwargs).fit()
            self._subsequence_models.append(model)

        self._is_fitted = True
        return self

    def _predict_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """Blend per-subsequence derivative predictions at (t, x) with a Gaussian
        kernel over each subsequence's midpoint. See docs/piecewise_system_discovery.md.
        """
        # Eliminated smoothing across model subsequences
        self._require_fitted()
        # Find the model for this subsequence
        subsequence_idx = int(np.sum([1 for _, end in self._subsequence_boundaries if t > end]))
        model = self._subsequence_models[subsequence_idx]
        return model.predictOneStepDerivative(x)

    def predict(self, test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame:
        """Integrate the blended ODE forward and return predicted concentrations."""
        self._require_fitted()
        if test_df is not NULL_DF:
            x0 = test_df.to_numpy(dtype=float)[0, :]
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            raw_df = self.timecourse.timecourse_df
            x0 = raw_df.to_numpy(dtype=float)[0, :]
            time_arr = raw_df.index.to_numpy(dtype=float)

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self._predict_derivative(t, x)

        sol = solve_ivp(
                rhs,
                t_span=(time_arr[0], time_arr[-1]),
                y0=x0,
                t_eval=time_arr,
                method="Radau",
                rtol=1e-6,
                atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        species_names = self._subsequence_models[0].species_names
        return pd.DataFrame(sol.y.T, index=time_arr, columns=species_names)

    # FIXME: Must handle each column separately
    def getWeightedScores(self) ->  pd.DataFrame:
        """Length-weighted aggregation of per-subsequence ScoreInfo. See
        docs/piecewise_system_discovery.md `score()` section."""
        self._require_fitted()
        weighted_values: List[float] = []
        num_nonzero_term = 0
        dfs : List[pd.DataFrame] = []
        for model, length in zip(self._subsequence_models, self._subsequence_lengths):
            df = model.getScoreDetails()
            weighted_values.extend(df.values * length)
            num_nonzero_term += df.num_nonzero_term
            dfs.append(df)
        final_df = pd.concat(dfs, axis=0) / len(dfs)
        return final_df
    
    def score(self, column_name: str = 'p95') -> float:
        """Length-weighted aggregation of per-subsequence ScoreInfo. See
        docs/piecewise_system_discovery.md `score()` section."""
        weighted_score_df = self.getWeightedScores()
        sel = weighted_score_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
        result = weighted_score_df[sel][column_name].values
        if len(result) != 1:
            raise RuntimeError(f"Expected 1 row for {cn.COL_AGGREGATION_TYPE_MODEL} but got {len(result)}")
        return cast(float, result[0])

    def __str__(self) -> str:
        block_list: List[str] = []
        for idx, (model, (start, end)) in enumerate(
                zip(self._subsequence_models, self._subsequence_boundaries), start=1):
            header = f"[subsequence {idx}: t in [{start:.1f}, {end:.1f})]"
            equation_line_list = [f"  {line}" for line in str(model).strip().split("\n")]
            block_list.append("\n".join([header] + equation_line_list))
        return "\n\n".join(block_list)

    def plotPiecewise(self, num_true_point: int = -1, **plt_kwargs: Any) -> PlotOptions:
        """Two-panel comparison: 0 change points (top) vs max_change_point (bottom).

        Both panels show actual (scatter) vs predicted (line) species concentrations.
        The bottom panel marks each detected change point with a vertical dashed line.

        Parameters
        ----------
        num_true_point : int
            Number of actual-data scatter points to show per panel.
            -1 means show all points.  If the timecourse has more than this many points,
        **plt_kwargs
            Forwarded to PlotOptions. Supported keys: fig, ax, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.  ``figsize`` is also
            accepted and consumed here (not passed to PlotOptions).

        Returns
        -------
        PlotOptions
            Wraps the figure and the bottom axes.  Call ``plt.show()`` or
            ``po.fig.savefig(...)`` on the returned object as needed.
        """
        self._require_fitted()
        figsize: tuple[float, float] = plt_kwargs.pop("figsize", (10, 8))

        timecourse_df = self.timecourse.timecourse_df
        time_arr = timecourse_df.index.to_numpy(dtype=float)
        actual_arr = timecourse_df.to_numpy(dtype=float)
        species_names = self._subsequence_models[0].species_names
        if num_true_point < 0 or num_true_point >= len(time_arr):
            num_true_point = len(time_arr)
        num_skip = max(1, len(time_arr) // num_true_point)

        baseline = SystemDiscovery(timecourse_df, **self._sd_kwargs).fit()
        try:
            baseline_pred_df = baseline.predict()
        except Exception:
            baseline_pred_df = None

        try:
            psd_pred_df = self.predict()
        except Exception:
            psd_pred_df = None
        baseline_score = baseline.score()
        psd_score = self.score()

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        change_point_times = [start for start, _ in self._subsequence_boundaries[1:]]

        plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)

        def _draw(po: PlotOptions, pred_df: pd.DataFrame | None, title: str,
                vlines: List[float] | None = None) -> None:
            ax = po.ax
            for idx, name in enumerate(species_names):
                color = f"C{idx}"
                ax.plot(  # type: ignore
                    time_arr[::num_skip], actual_arr[::num_skip, idx],
                    linestyle="-", color=color, label=f"{name} actual", zorder=3,
                )
                if pred_df is not None and name in pred_df.columns:
                    ax.plot(  # type: ignore
                        pred_df.index, pred_df[name],
                        "--", lw=1.5, color=color, label=f"{name} predicted",
                    )
            if vlines:
                for t in vlines:
                    ax.axvline(t, color="black", linestyle="--", lw=1.0, alpha=0.6)  # type: ignore
            ax.set_title(title, fontsize=11)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
            po.apply()

        _draw(PlotOptions(fig=fig, ax=ax_top, **plt_kwargs), baseline_pred_df,
                #f"0 change points, r2: {baseline_score.median:.3f}")
                f"0 change points, r2: {baseline_score:.3f}")
        _draw(plot_options, psd_pred_df,
                #f"{self.max_change_point} change point(s), r2: {psd_score.median:.3f}",
                f"{self.max_change_point} change point(s), r2: {psd_score:.3f}",
                vlines=change_point_times)
        fig.suptitle("Actual vs Predicted", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return plot_options

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each subsequence."""
        self._require_fitted()
        print(str(self))
