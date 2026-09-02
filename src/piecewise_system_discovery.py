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

import collections
from dataclasses import dataclass  # noqa: E402 (dataclass used by PiecewiseSystemDiscovery._ScoreSummary)
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Any, List, Tuple
from typing import cast, Optional


PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-points."""

    @dataclass
    class _ScoreSummary:
        """Lightweight summary of scores across all subsequences.

        Attributes mirror the column names produced by ``src.score.Score`` so they line up with CSV output.
        """
        min: float  # minimum species-level score (accuracy)
        median: float  # median species-level accuracy
        max: float  # maximum species-level accuracy
        num_nonzero_term: int  # total number of non-zero ODE terms across all segments and species

    def __init__(
        self,
        training_df:  pd.DataFrame,
        max_changepoint: int = 2,
        max_fractional_reduction: float = 0.01,  
        min_segment_length: int = 100,
        model_name: str = "",
        num_trail: int = 1,
        changepoints: Optional[List[int]] = None,
        **sd_kwargs: Any,
    ) -> None:
        """Construct a piecewise-linear ODE discovery pipeline.

        Args:
            training_df (pd.DataFrame): Time-series data with one column per species.
            max_changepoint (int, optional): Maximum number of change points to detect. Defaults to 2.
            max_fractional_reduction (float, optional): Minimum fractional reduction in ASS required. Defaults to 0.1.
            min_segment_length (int, optional): Minimum length of segments for splitting. Defaults to 100.
            model_name (str, optional): Optional name tag used in plots and error messages. Defaults to "".
            num_trail (int, optional): Number of random changepoint trials
            changepoints (List[int], optional): List of pre-determined change points. Defaults to None.
            **sd_kwargs: Arguments forwarded to each per-segment ``SystemDiscovery`` constructor.
        """
        self.training_df = training_df
        self.species_names = list(training_df.columns)
        self.num_species = len(self.species_names)
        self.num_point = training_df.shape[0]
        self.model_name = model_name
        self.max_changepoint = max_changepoint
        self.max_fractional_reduction = max_fractional_reduction
        self.min_segment_length = min_segment_length
        self.is_random_changepoints = sd_kwargs.pop("is_random_changepoints", False)
        self.num_trail = num_trail
        sd_kwargs["poly_degree"] = sd_kwargs.get("poly_degree", 1)
        self._sd_kwargs = sd_kwargs
        self.changepoints = changepoints

        self._subsequence_models: List[SystemDiscovery] = []
        self._subsequence_boundaries: List[Tuple[float, float]] = []
        self._subsequence_lengths: List[int] = []
        self._is_fitted: bool = False
        # Baseline (whole-timecourse) SystemDiscovery model, fit lazily so construction stays cheap
        # when ``fit()`` is never invoked.  Accessed via :meth:`_getBaselineSystemDiscovery`.
        self._sys_disc: Optional[SystemDiscovery] = None

    @property
    def num_changepoint(self) -> int:
        """Return the number of detected change points."""
        return len(self._subsequence_models) - 1 if self._is_fitted else 0

    def _getBaselineSystemDiscovery(self) -> SystemDiscovery:
        """Lazily build and cache the whole-timecourse baseline ``SystemDiscovery`` model."""
        if self._sys_disc is None:
            self._sys_disc = SystemDiscovery(
                self.training_df, is_normalize=True, **self._sd_kwargs).fit()
        return cast(SystemDiscovery, self._sys_disc)

    def _requireFitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")

    def _makeRandomChangepoints(self, seed: Optional[int] = None) -> List[int]:
        """Generate random change points respecting ``max_changepoint`` and
        ``min_segment_length``.

        Indices are drawn uniformly from ``[1, num_point - 1)`; each new point is
        rejected if it lies within ``min_segment_length`` of a previously chosen one.
        If the constraint set cannot accommodate all ``max_changepoint`` placements,
        fewer points are returned rather than raising — callers should treat the
        returned length as an upper bound.

        Parameters
        ----------
        seed : int or None
            RNG seed for deterministic generation.  Defaults to a fresh random state.

        Returns
        -------
        list[int]
            Sorted, unique indices in ``[1, num_point - 1)`` with pairwise distance
            at least ``min_segment_length`` (or fewer elements if constraints make
            that impossible).
        """
        if self.max_changepoint <= 0:
            return []
        if self.max_changepoint >= self.num_point:
            raise ValueError(
                f"max_changepoint {self.max_changepoint} exceeds number of points "
                f"{self.num_point}.")

        rng = np.random.default_rng(seed)
        changepoints: List[int] = []
        candidates = list(range(1, self.num_point - 1))
        for _ in range(self.max_changepoint):
            if not candidates:
                break
            idx = int(rng.integers(0, len(candidates)))
            changepoint = candidates[idx]
            changepoints.append(changepoint)
            # Reject future candidates within min_segment_length of the chosen point.
            cutoff = self.min_segment_length
            candidates = [c for c in candidates if abs(changepoint - c) >= cutoff]
        return sorted(changepoints)

    def _makeBestRandomChangepoints(self) -> List[int]:
        """Try several random changepoint sets and keep the one whose piecewise fit scores best.

        For each trial a fresh PiecewiseSystemDiscovery is built with that candidate set, fit() against
        training data, and scored via accuracy across all species/segments. The candidate producing
        the highest score wins; ties go to the first (lowest seed) trial.

        Returns
        -------
        list[int]
            Sorted changepoint indices in ``[1, num_point - 1)`` for the best trial (or the only one when
            ``num_trail <= 1``).  May be shorter than ``max_changepoint`` if segment constraints prevent it.
        """
        best_cp: Optional[List[int]] = None
        best_score = float("-inf")
        rng = np.random.default_rng()
        for trial_idx in range(self.num_trail):
            seed = int(rng.integers(0, 2 ** 31)) if self.num_trail > 1 else None
            cp = self._makeRandomChangepoints(seed=seed)
            if self.num_trail <= 1:
                best_cp = cp
                break
            # Score this candidate by fitting a full PiecewiseSystemDiscovery against training data.
            try:
                trial_psd = PiecewiseSystemDiscovery(
                    self.training_df,
                    max_changepoint=self.max_changepoint,
                    max_fractional_reduction=self.max_fractional_reduction,
                    min_segment_length=self.min_segment_length,
                    model_name=f"{self.model_name}_trial_{trial_idx}",
                    **self._sd_kwargs,
                )
                trial_psd._is_fitted = True  # bypass recursion into _getChangepoints()
                trial_psd._subsequence_models, trial_psd._subsequence_boundaries, trial_psd._subsequence_lengths = \
                    self._fitSegments(cp)
                score = trial_psd.score()
            except Exception:
                # Treat fit failures as infinitely bad so they don't win.
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_cp = cp
        return list(best_cp or [])

    def _fitSegments(self, changepoints: List[int]) -> Tuple[List[SystemDiscovery],
            List[Tuple[float, float]], List[int]]:
        """Build (models, boundaries, lengths) from a given set of indices."""
        time_arr = self.training_df.index.to_numpy(dtype=float)
        boundary_index_arr = [0] + changepoints + [self.num_point]
        models: List[SystemDiscovery] = []
        boundaries: List[Tuple[float, float]] = []
        lengths: List[int] = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            subsequence_df = self.training_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < self.num_point else time_arr[-1]
            boundaries.append((float(time_arr[lo]), float(end_time)))
            lengths.append(hi - lo)
            try:
                sys_disc = SystemDiscovery(subsequence_df, **self._sd_kwargs).fit()
                models.append(sys_disc)
            except Exception as e:
                raise RuntimeError(f"Error fitting SystemDiscovery for segment {lo}:{hi}: {e}")
        return models, boundaries, lengths

    def _makeChangepointsIteratively(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints and then iteratively
        remove those whose elimination does not degrade accuracy by more than
        ``max_fractional_reduction``.

        The initial configuration divides the time series into ``max_changepoint + 1``
        segments of roughly equal length (respecting :attr:`min_segment_length` so that no
        segment shrinks below that threshold). Each candidate changepoint is then tested
        for removal: a new piecewise model is fit on the remaining changepoints, and if
        the resulting accuracy reduction stays within ``max_fractional_reduction``,
        that changepoint is dropped. The process repeats until no further removal is
        cheap enough.

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        max_changepoint = self.max_changepoint
        num_point = self.num_point
        min_seg = self.min_segment_length

        if max_changepoint <= 0:
            return []
        if max_changepoint >= num_point:
            raise ValueError(
                f"max_changepoint {max_changepoint} exceeds number of points {num_point}.")

        # (a) Evenly spaced initial changepoints.
        step = num_point / (max_changepoint + 1)
        candidate_indices = [int(round((i + 1) * step)) for i in range(max_changepoint)]
        changepoints: List[int] = []
        last_kept = -min_seg  # sentinel so the first kept point has room before index 0
        for cp in sorted(candidate_indices):
            if cp < 1 or cp >= num_point:
                continue
            if cp - last_kept < min_seg:
                continue
            changepoints.append(cp)
            last_kept = cp

        # Sanity cap: never return more than max_changepoint.
        changepoints = changepoints[:max_changepoint]
        if not changepoints:
            return []

        def _score_for(cps: List[int]) -> float:
            """Fit piecewise models for the given changepoints and return their score,
            leaving self unchanged on any failure."""
            saved_m = self._subsequence_models
            saved_b = self._subsequence_boundaries
            saved_l = self._subsequence_lengths
            saved_fitted = self._is_fitted
            try:
                _models, _bounds, _lens = self._fitSegments(cps)
                self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                    _models, _bounds, _lens)
                self._is_fitted = True
                return float(self.score(test_df=self.training_df))
            except Exception:
                raise
            finally:
                self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                    saved_m, saved_b, saved_l)
                self._is_fitted = saved_fitted

        # (b) Iteratively remove changepoints whose removal is cheap enough.
        while True:
            try:
                baseline_score = _score_for(changepoints)
            except Exception:
                # If we can't even fit the full set, bail out with what we have.
                return changepoints

            best_rm_idx: Optional[int] = None
            best_reduction = float('inf')
            for idx in range(len(changepoints)):
                trial_cp = [c for i, c in enumerate(changepoints) if i != idx]
                try:
                    trial_score = _score_for(trial_cp)
                except Exception:
                    # Fit failure on this candidate -- treat as too costly.
                    continue

                reduction = baseline_score - trial_score
                if reduction <= self.max_fractional_reduction and reduction < best_reduction:
                    best_reduction = reduction
                    best_rm_idx = idx

            if best_rm_idx is None:
                break
            changepoints.pop(best_rm_idx)

        return changepoints
    
    def fit(self) -> 'PiecewiseSystemDiscovery':
        """Detect change points and fit a ``SystemDiscovery`` model to each segment.

        After this call, :attr:`_subsequence_models`, :attr:`_subsequence_boundaries`,
        and :attr:`_subsequence_lengths` are populated; :meth:`predict` is available.
        The baseline whole-timecourse model is built lazily on first access.
        """
        if self.changepoints is None:
            changepoints = self._makeChangepointsIteratively()
        else:
            changepoints = self.changepoints
        (self._subsequence_models, self._subsequence_boundaries,
        self._subsequence_lengths) = self._fitSegments(changepoints)
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
            score_info[cn.COL_ENDTIME] = end
            score_dfs.append(score_info)
        result_df = pd.concat(score_dfs, ignore_index=True)
        return result_df

    def getScoreSummary(self) -> 'PiecewiseSystemDiscovery._ScoreSummary':
        """Return a lightweight summary object with ``.min``/``.median``/``.max``/``.num_nonzero_term``.

        Aggregates per-segment scores and term counts across all subsequences so callers can compare
        trial runs without dealing directly with the raw DataFrame columns.
        """
        self._requireFitted()
        score_dfs = []
        total_nonzero = 0
        for sys_disc, (start, end) in zip(self._subsequence_models, self._subsequence_boundaries):
            score_info = sys_disc.getScoreDetails()
            species_rows = score_info[score_info[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
            if not species_rows.empty:
                score_dfs.append(species_rows)
            total_nonzero += sum(sys_disc.getNonzeroTerms().values())
        combined = pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame()
        return self._ScoreSummary(
            min=float(combined[cn.COL_MIN].min()) if not combined.empty else float("nan"),
            median=float(combined[cn.COL_P50].median()) if not combined.empty else float("nan"),
            max=float(combined[cn.COL_MAX].max()) if not combined.empty else float("nan"),
            num_nonzero_term=total_nonzero,
        )

    def __str__(self) -> str:
        block_list: List[str] = []
        for idx, (model, (start, end)) in enumerate(
                zip(self._subsequence_models, self._subsequence_boundaries), start=1):
            header = f"[subsequence {idx}: t in [{start:.1f}, {end:.1f})]"
            equation_line_list = [f"  {line}" for line in str(model).strip().split("\n")]
            block_list.append("\n".join([header] + equation_line_list))
        return "\n\n".join(block_list)

    def plotPiecewise(self, num_true_point: int = -1, 
                suptitle="Actual vs. Predicted", **plt_kwargs: Any) -> PlotOptions:
        """Two-panel comparison: 0 change points (top) vs max_changepoint (bottom).

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

        # Get baseline vs. piecewise scores and predictions; lazily build the whole-timecourse model.
        sys_disc = self._getBaselineSystemDiscovery()
        baseline_score = sys_disc.score()
        baseline_pred_df = sys_disc.predict()
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
                title=f"{self.num_changepoint} change points",
                vlines=change_point_times, **plt_kwargs)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
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
