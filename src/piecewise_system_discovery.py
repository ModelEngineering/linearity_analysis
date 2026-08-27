"""Piecewise system discovery based on detecting change points in 1-step prediction of derivatives."""

"""
The approach is:
1. Find change points in the timecourse based on changes in the accuracy of one-step prediction of derivatives.
2. Fit a SystemDiscovery model to each segment of the timecourse between change points.
3. Return a PiecewiseSystemDiscovery object that contains the fitted SystemDiscovery models for each segment.
"""

import src.constants as cn
from src.model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.system_discovery import SystemDiscovery, NULL_DF  # type: ignore
from src.system_discovery_changepoint_detector import SystemDiscoveryChangepointDetector as _SCPD

import collections
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Any, List, Tuple
from typing import cast, Optional


PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-points."""

    def __init__(
        self,
        training_df:  pd.DataFrame,
        max_changepoint: int = 2,
        min_fractional_reduction: float = 0.1,  
        min_subsequence_length: int = 100,
        predict_kernel_bandwidth: float = 0.5,
        model_name: str = "",
        **sd_kwargs: Any,
    ) -> None:
        """Construct a piecewise-linear ODE discovery pipeline.

        Args:
            training_df (pd.DataFrame): Time-series data with one column per species.
            max_changepoint (int, optional): Maximum number of change points to detect. Defaults to 2.
            min_fractional_reduction (float, optional): Minimum fractional reduction in ASS required. Defaults to 0.1.
            min_subsequence_length (int, optional): Minimum length of segments for splitting. Defaults to 100.
            predict_kernel_bandwidth (float, optional): Gaussian kernel width used by :meth:`predict_derivative`. Defaults to 0.5.
            **sd_kwargs: Arguments forwarded to each per-segment ``SystemDiscovery`` constructor.
        """
        self.training_df = training_df
        self.species_names = list(training_df.columns)
        self.num_species = len(self.species_names)
        self.num_point = training_df.shape[0]
        self.model_name = model_name
        self.max_changepoint = max_changepoint
        self.min_fractional_reduction = min_fractional_reduction
        self.min_subsequence_length = min_subsequence_length
        self.predict_kernel_bandwidth = predict_kernel_bandwidth
        sd_kwargs["poly_degree"] = sd_kwargs.get("poly_degree", 1)
        self._sd_kwargs = sd_kwargs

        self._subsequence_models: List[SystemDiscovery] = []
        self._subsequence_boundaries: List[Tuple[float, float]] = []
        self._subsequence_lengths: List[int] = []
        self._is_fitted: bool = False

        # Diagnostic global model and changepoint detector — initialized lazily in fit() so
        # their hyper-parameters are not fixed at construction time.
        self.sys_disc = None  # type: ignore[assignment]
        self.detector = None  # type: ignore[assignment]

    def _requireFitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")
    
    def fit(self) -> 'PiecewiseSystemDiscovery':
        """fit() steps 1-4: detect change points, fit per-subsequence models.

        Lazy-initializes the diagnostic global SystemDiscovery and changepoint detector on
        first call so construction remains cheap when fit() is never invoked."""
        if self.detector is None:
            self.sys_disc = SystemDiscovery(
                self.training_df, threshold=0.001, alpha=0.001, is_normalize=True)
            self.sys_disc.fit()
            self.detector = _SCPD(self.sys_disc)
        self.detector.fit(max_changepoint=self.max_changepoint,
                min_segment_length=self.min_subsequence_length,
                min_fractional_reduction=self.min_fractional_reduction)
        time_arr = self.training_df.index.to_numpy(dtype=float)
        boundary_index_arr = [0] + self.detector.changepoints + [self.num_point]
        # Construct the subsequence information
        self._subsequence_models = []
        self._subsequence_boundaries = []
        self._subsequence_lengths = []
        # If there are no change points, treat the entire time range as one segment.
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            subsequence_df = self.training_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < self.num_point else time_arr[-1]
            self._subsequence_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._subsequence_lengths.append(hi - lo)
            sys_disc = SystemDiscovery(subsequence_df, **self._sd_kwargs).fit()
            self._subsequence_models.append(sys_disc)
        #
        self._is_fitted = True
        return self

    def predict(self, test_df: Optional[pd.DataFrame] = NULL_DF) -> pd.DataFrame:
        """Predict concentrations by using SystemDiscovery models for
            each segment of the timecourse based on the initial condition in
            test_df. Only timepoint 0 of test_df is used for initial conditions;
            the rest of the rows are ignored.
            Verifies that the time grid of test_df matches the training data's time grid.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, provides initial conditions (first row) and time grid
            (index).  When omitted, the training data's first row and index are used.

        Returns
        -------
        pd.DataFrame
            Predicted concentrations with one column per species; time as the index.
        """
        self._requireFitted()

        # Build a full time grid for this prediction run.
        if test_df is None:
            test_df = NULL_DF
        if test_df.empty:
            test_df = self.training_df.copy()
        else:
            self._checkColumns(list(test_df.columns))
            self._checkTimegGrid(test_df.index.to_numpy(dtype=float))
        test_df = cast(pd.DataFrame, test_df)

        # Integrate each segment with its own fitted SystemDiscovery model.
        pred_frames: List[pd.DataFrame] = []
        for seg_model, (t_start, t_end) in zip(
                    self._subsequence_models, self._subsequence_boundaries):
            test_seg_df = test_df[(test_df.index >= t_start) & (test_df.index <= t_end)]
            pred_df = seg_model.predict(test_df=test_seg_df)
            if t_start > 0:
                # Do not include the first row of each segment except for the first segment,
                # since it is already included in the previous segment's prediction.
                pred_df = pred_df[pred_df.index > t_start]
            pred_frames.append(pred_df)
        full_pred_df = pd.concat(pred_frames)
        # Check for duplicate rows
        if full_pred_df.index.duplicated().any():
            raise RuntimeError(
                "Duplicate rows in prediction result.  This should not happen.")
        return full_pred_df

    def _checkColumns(self, col_list: List[str]) -> None:
        """Validate that *col_list* matches the species names of the first segment model."""
        if sorted(col_list) != sorted(self.species_names):
            raise ValueError(
                f"Column mismatch: test_df columns {sorted(col_list)} do not match "
                f"expected species names {self.species_names}."
            )

    def _checkTimegGrid(self, time_arr: np.ndarray) -> None:
        """Validate that *time_arr* matches the time grid of the first segment model."""
        if not np.allclose(time_arr, self.training_df.index.to_numpy(dtype=float)):
            raise ValueError(
                f"Time grid mismatch: test_df index {time_arr} does not match "
                f"expected time grid {self.training_df.index.to_numpy(dtype=float)}."
            )
        if not time_arr[0] == 0:
            raise ValueError(
                f"Time grid mismatch: test_df index {time_arr} does not start at 0."
            )

    def getScoreDetails(self, test_df: Optional[pd.DataFrame] = None,
                score_type="timecourse") -> pd.DataFrame:
        """Return a DataFrame of per-subsequence ScoreInfo."""
        self._requireFitted()
        if test_df is None:
            test_df = NULL_DF
        score_dfs = []
        for sys_disc, (start, end) in zip(self._subsequence_models, self._subsequence_boundaries):
            test_seg_df = test_df.iloc[(test_df.index >= start) & (test_df.index <= end)]
            score_info = sys_disc.getScoreDetails(test_df=test_seg_df, score_type=score_type)
            score_info[cn.COL_START_TIME] = start
            score_info[cn.COL_END_TIME] = end
            score_dfs.append(score_info)
        result_df = pd.concat(score_dfs, ignore_index=True)
        return result_df

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
            -1 means show all points.  If the training_df has more than this many points,
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
        self._requireFitted()
        figsize: tuple[float, float] = plt_kwargs.pop("figsize", (10, 8))

        time_arr = self.training_df.index.to_numpy(dtype=float)
        actual_arr = self.training_df.to_numpy(dtype=float)
        if num_true_point < 0 or num_true_point >= len(time_arr):
            num_true_point = len(time_arr)
        num_skip = max(1, len(time_arr) // num_true_point)

        # Get the data
        self.sys_disc = cast(SystemDiscovery, self.sys_disc)
        baseline_score = self.sys_disc.score()
        baseline_pred_df = self.sys_disc.predict()
        psd_score = self.score()
        psd_pred_df = self.predict()
        # Construct the plot
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        change_point_times = [start for start, _ in self._subsequence_boundaries[1:]]
        plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        ##
        def _draw(pred_df: pd.DataFrame, score: float, 
                vlines: Optional[List[float]] = None, **plt_options) -> None:
            po = PlotOptions(**plt_options)
            ax = po.ax
            ymax = actual_arr.min().min()
            for idx, name in enumerate(self.species_names):
                color = f"C{idx}"
                ax.scatter(  # type: ignore
                    time_arr[::num_skip], actual_arr[::num_skip, idx],
                    marker="o", s=30, linestyle="-", color=color, label=f"{name} actual", zorder=3,
                )
                if pred_df is not None and name in pred_df.columns:
                    ax.plot(  # type: ignore
                        pred_df.index, pred_df[name],
                        "--", lw=1.5, color=color, label=f"{name} predicted",
                    )
            if vlines:
                for t in vlines:
                    ax.axvline(t, color="black", linestyle="--", lw=1.0, alpha=0.6)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
            if self.model_name.startswith("BIOMD"):
                model_num_str = str(int(self.model_name[6:]))
            else:
                model_num_str = self.model_name
            po.title = model_num_str + ": " + plt_options.get("title", "") + f" (Mean accuracy={score:.3f})"
            if ymax > 0.0:
                po.ylim = (0.0, ymax)
            po.apply()
        ##
        _draw(fig=fig, ax=ax_top, pred_df=baseline_pred_df, score=baseline_score,
                title="0 change points", **plt_kwargs)
        _draw(fig=fig, ax=ax_bot, pred_df=psd_pred_df, score=psd_score,
                title=f"{self.max_changepoint} change points",
                vlines=change_point_times, **plt_kwargs)
        fig.suptitle("Actual vs Predicted", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return plot_options

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each subsequence."""
        self._requireFitted()
        print(str(self))

    def score(self, test_df: Optional[pd.DataFrame] = None, score_type="timecourse") -> float:
        """Return the average score across all subsequences."""
        self._requireFitted()
        score_df = self.getScoreDetails(test_df=test_df, score_type=score_type)
        return float(score_df[cn.COL_MEAN].mean())
