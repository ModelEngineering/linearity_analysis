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
import concurrent.futures  # noqa: E402 (used by parallel helpers below)
from dataclasses import dataclass  # noqa: E402 (dataclass used by PiecewiseSystemDiscovery._ScoreSummary)
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os  # noqa: E402 (used to cap parallel workers at cpu_count())
import pandas as pd  # type: ignore
from typing import Any, Dict, List, Optional, Tuple
from typing import cast


PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])


# ---------------------------------------------------------------------------
# Module-level helpers used by the parallel changepoint elimination pipeline.
# They live at module scope so they are picklable and can be dispatched across
# process boundaries by ``concurrent.futures.ProcessPoolExecutor``.
# ---------------------------------------------------------------------------

def _compute_median_score(
        models: List[SystemDiscovery],
        boundaries: List[Tuple[float, float]],
        training_df: pd.DataFrame) -> float:
    """Return the median p10 accuracy across all fitted segment models."""
    score_dfs = []
    for sys_disc, (start, end) in zip(models, boundaries):
        test_seg_df = training_df.iloc[(training_df.index >= start) & (training_df.index <= end)]
        score_info = sys_disc.getScoreDetails(test_df=test_seg_df, score_type="timecourse")
        score_dfs.append(score_info)
    combined = pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame()
    return float(combined[cn.COL_P10].median())


def _fit_segments_parallel(
        training_df: pd.DataFrame,
        boundary_index_arr: List[int],
        time_arr: np.ndarray,
        num_point: int,
        sd_kwargs: Dict[str, Any]) -> Tuple[List[SystemDiscovery],
                                            List[Tuple[float, float]],
                                            List[int]]:
    """Fit one ``SystemDiscovery`` per segment in parallel threads.

    Each segment's fit is dispatched to its own thread so that GIL-releasing
    scipy/numpy work (PySINDy STLSQ sparse regression) can run concurrently
    across segments.  Within a single call no mutable state is shared between
    threads -- each gets its own ``SystemDiscovery`` instance and its own data slice.

    Parameters
    ----------
    training_df : pd.DataFrame
        Full timecourse used to build per-segment slices.
    boundary_index_arr : list[int]
        Boundary indices including the leading 0 and trailing num_point, e.g.
        ``[0, 150, 300, 500]`` for three segments on a 500-row timecourse.
    time_arr : np.ndarray
        Full index as a float numpy array (used to compute boundary end-times).
    num_point : int
        Number of rows in ``training_df``; used to clamp the last segment's end-time.
    sd_kwargs : dict
        Keyword arguments forwarded to every ``SystemDiscovery(...)`` call.

    Returns
    -------
    tuple[list[SystemDiscovery], list[tuple[float, float]], list[int]]
        ``(models, boundaries, lengths)`` -- identical shape to
        :meth:`PiecewiseSystemDiscovery._fitSegments`.
    """
    models: List[SystemDiscovery] = []
    boundaries: List[Tuple[float, float]] = []
    lengths: List[int] = []

    def _fit_one_segment(lo: int, hi: int) -> Tuple[SystemDiscovery, float, float, int]:
        subsequence_df = training_df.iloc[lo:hi].copy()  # defensive copy for thread safety
        end_time_idx = hi if hi < num_point else len(time_arr) - 1
        return (SystemDiscovery(subsequence_df, **sd_kwargs).fit(),
                float(time_arr[lo]), float(time_arr[end_time_idx]), hi - lo)

    n_segments = max(len(boundary_index_arr) - 1, 1)
    max_workers = min(4, n_segments) if n_segments > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx: Dict[int, concurrent.futures.Future] = {}
        for i, (lo, hi) in enumerate(zip(boundary_index_arr[:-1], boundary_index_arr[1:])):
            future_to_idx[i] = pool.submit(_fit_one_segment, lo, hi)
        # Collect results in order so the returned list matches ``boundary_index_arr``.
        for idx in sorted(future_to_idx.keys()):
            sys_disc, bnd_lo, bnd_hi, length = future_to_idx[idx].result()  # type: ignore[index]
            models.append(sys_disc)
            boundaries.append((bnd_lo, bnd_hi))
            lengths.append(length)

    return models, boundaries, lengths


def _worker_evaluate_removal(
        training_df: pd.DataFrame,
        time_arr_full: np.ndarray,
        num_point: int,
        species_names: List[str],
        sd_kwargs: Dict[str, Any],
        trial_changepoints: List[int]) -> float:
    """Worker function (runs in its own process).

    Fits a piecewise model for *trial_changepoints* using parallel segment fits
    and returns the median p10 accuracy score.  Returns ``float('inf')`` on any
    fit failure so the caller can treat it as infinitely bad and skip that trial.
    """
    try:
        boundary_index_arr = [0] + list(trial_changepoints) + [num_point]
        models, boundaries, lengths = _fit_segments_parallel(
            training_df=training_df,
            boundary_index_arr=boundary_index_arr,
            time_arr=time_arr_full,
            num_point=num_point,
            sd_kwargs=dict(sd_kwargs),  # copy to avoid pickle surprises with nested defaults
        )
        return _compute_median_score(models, boundaries, training_df)
    except Exception:
        return float("inf")


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
        is_changepoint_removal: bool = True,
        is_random_changepoints: bool = False,
        **sd_kwargs: Any,
    ) -> None:
        """Construct a piecewise-linear ODE discovery pipeline.

        Args:
            training_df (pd.DataFrame): Time-series data with one column per species.
            max_changepoint (int, optional): Maximum number of change points to detect. Defaults to 2.
                Can be adjusted so that there is enough data per segment.
            max_fractional_reduction (float, optional): Maximum fractional reduction in the sum of squared errors required to accept a new change point. Defaults to 0.01.
            min_segment_length (int, optional): Minimum length of segments for splitting. Defaults to 100.
            model_name (str, optional): Optional name tag used in plots and error messages. Defaults to "".
            num_trail (int, optional): Number of random changepoint trials.
                Only used if is_random_changepoints is True.
            changepoints (List[int], optional): List of pre-determined change points. Defaults to None.
            is_changepoint_removal (bool, optional): Whether to allow removal of detected change points. Defaults to True.
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
        self.changepoints = changepoints  # if None, will be determined during fit()a
        self._is_random_changepoints = is_random_changepoints
        self._is_changepoint_removal = is_changepoint_removal

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
        max_changepoint = self._adjustMaxChangepoints()
        cutoff = self.num_species # Minimum distance between changepoints that allows estimation of the system parameters

        rng = np.random.default_rng(seed)
        changepoints: List[int] = []
        candidates = list(range(1, self.num_point - 1))
        for _ in range(max_changepoint):
            if not candidates:
                break
            idx = int(rng.integers(0, len(candidates)))
            changepoint = candidates[idx]
            changepoints.append(changepoint)
            # Reject future candidates within min_segment_length of the chosen point.
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
                    max_changepoint=0,
                    changepoints=cp,
                    max_fractional_reduction=self.max_fractional_reduction,
                    min_segment_length=self.min_segment_length,
                    model_name=f"{self.model_name}_trial_{trial_idx}",
                    **self._sd_kwargs,
                )
                trial_psd.changepoints = cp
                trial_psd._is_fitted = True  # bypass recursion into _getChangepoints()
                trial_psd._subsequence_models, trial_psd._subsequence_boundaries, trial_psd._subsequence_lengths = \
                    self._fitSegments(cp)
                score = trial_psd.score(col=cn.COL_P50, statistic="median")
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

    def _makeChangepointsWithoutElimination(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints and then repeatedly 
        eliminates changepoints that do not degrade accuracy by more than
        ``max_fractional_reduction``.

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        num_point = self.num_point
        max_changepoint = self._adjustMaxChangepoints()

        if max_changepoint <= 0:
            return []
        if max_changepoint >= num_point:
            raise ValueError(
                f"max_changepoint {max_changepoint} exceeds number of points {num_point}.")

        # (a) Evenly spaced initial changepoints -- identical to the original.
        step = num_point / (max_changepoint + 1)
        candidate_indices = [int(round((i + 1) * step)) for i in range(max_changepoint)]
        changepoints: List[int] = []
        for cp in sorted(candidate_indices):
            if cp < 1 or cp >= num_point:
                continue
            changepoints.append(cp)

        return changepoints[:max_changepoint]

    def _makeChangepointsWithElimination(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints and then repeatedly 
        eliminates changepoints that do not degrade accuracy by more than
        ``max_fractional_reduction``.

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        threshold = self.max_fractional_reduction
        changepoints = self._makeChangepointsWithoutElimination()
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
                return float(self.score(test_df=self.training_df, col=cn.COL_MEAN, statistic="mean")
                        / self.num_species)
            except Exception:
                raise
            finally:
                self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                    saved_m, saved_b, saved_l)
                self._is_fitted = saved_fitted

        # (b) Fit the baseline once before entering the loop. If we can't even fit the full set,
        # bail out with what we have -- same semantics as the original's first iteration.
        try:
            baseline_score = _score_for(changepoints)
        except Exception:
            return changepoints

        # (c) Iteratively remove changepoints whose removal is cheap enough. The key difference
        # from ``_makeChangepointsIteratively`` is that ``baseline_score`` is hoisted outside the
        # while loop, and after every successful pop we recover the new baseline from the trial
        # score already computed for that exact removal -- no extra fit needed.
        while True:
            best_rm_idx: Optional[int] = None
            best_reduction = float('inf')
            trial_scores: dict[int, float] = {}

            for idx in range(len(changepoints)):
                trial_cp = changepoints[:idx] + changepoints[idx + 1:]   # slice concat
                try:
                    ts = _score_for(trial_cp)
                except Exception:
                    trial_scores[idx] = float('inf')
                    continue

                reduction = baseline_score - ts
                if reduction <= threshold and reduction < best_reduction:
                    best_reduction = reduction
                    best_rm_idx = idx

                trial_scores[idx] = ts

            if best_rm_idx is None:
                break
            changepoints.pop(best_rm_idx)

            """ # Recover new baseline from the just-computed trial score of the removal we just made.
            winning_ts = trial_scores.get(best_rm_idx, float('inf'))
            if winning_ts == float('inf'):
                # Fit failed on the selected candidate in the previous pass (shouldn't happen --
                # only finite-reduction candidates are selected); refit baseline as a safety net.
                try:
                    baseline_score = _score_for(changepoints)
                except Exception:
                    return changepoints
            else:
                baseline_score = winning_ts """

        return changepoints


    def _parallel_evaluate_removals(
            self,
            changepoints: List[int],
            baseline_score: float,
            time_arr_full: np.ndarray) -> Tuple[Optional[int], Dict[int, float]]:
        """Evaluate all single-changepoint removals in parallel and pick the best cheap one.

        Dispatches ``len(changepoints)`` independent trials to a
        ``ProcessPoolExecutor`` -- each trial calls :func:`_worker_evaluate_removal` with its
        own copy of the training data, so there is no shared mutable state across workers.

        Parameters
        ----------
        changepoints : list[int]
            Current set of candidate change-point indices (indices into ``training_df``).
        baseline_score : float
            Score of the current piecewise fit before any removal -- used as reference for
            computing each trial's reduction and comparing it against ``max_fractional_reduction``.
        time_arr_full : np.ndarray
            Cached full time array (precomputed once outside the while loop to avoid repeated
            ``index.to_numpy()`` calls in workers).

        Returns
        -------
        tuple[Optional[int], dict[int, float]]
            ``(best_rm_idx, trial_scores)`` where ``best_rm_idx`` is the index of the changepoint
            whose removal yields the smallest reduction that is still within threshold (or None if
            no such removal exists), and ``trial_scores`` maps each candidate index to its computed score.
        """
        n = len(changepoints)
        max_workers = min(n, os.cpu_count() or 4)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx: Dict[int, concurrent.futures.Future] = {}
            for idx in range(n):
                trial_cp = changepoints[:idx] + changepoints[idx + 1:]
                # Submit each removal evaluation. The full training_df is pickled per worker --
                # amortized against the CPU-heavy PySINDy fit that follows inside each worker.
                future_to_idx[idx] = pool.submit(
                    _worker_evaluate_removal,
                    training_df=self.training_df.copy(),
                    time_arr_full=time_arr_full,
                    num_point=self.num_point,
                    species_names=list(self.species_names),
                    sd_kwargs=dict(self._sd_kwargs),
                    trial_changepoints=trial_cp,
                )

            trial_scores: Dict[int, float] = {}
            for idx in range(n):
                try:
                    trial_scores[idx] = future_to_idx[idx].result()
                except Exception:  # pragma: no cover - defensive fallback
                    trial_scores[idx] = float("inf")

        best_rm_idx: Optional[int] = None
        best_reduction = float('inf')
        for idx, ts in trial_scores.items():
            if not np.isfinite(ts):
                continue
            reduction = baseline_score - ts
            if reduction <= self.max_fractional_reduction and reduction < best_reduction:
                best_rm_idx = idx
                best_reduction = reduction

        return best_rm_idx, trial_scores

    def _makeChangepointsWithEliminationParallel(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints then repeatedly eliminate those
        whose removal is cheap enough -- same algorithm as ``_makeChangepointsWithElimination`` but
        with two layers of parallelism:

        1. **Outer loop (inner for-loop)**: all single-changepoint removal trials in a given
           iteration are dispatched to a ``ProcessPoolExecutor``, so they run concurrently across
           processes (each worker gets its own GIL and its own memory).
        2. **Inner fit**: each trial's segment fits use :func:`_fit_segments_parallel` which runs
           one ``SystemDiscovery.fit()`` per segment in its own thread, letting the GIL-releasing
           scipy/numpy work inside PySINDy STLSQ run concurrently across segments.

        The initial baseline fit is done serially (it's a one-time cost and avoids touching the
        instance state from multiple threads simultaneously).

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        num_point = self.num_point
        max_changepoint = self._adjustMaxChangepoints()

        if max_changepoint <= 0:
            return []
        if max_changepoint >= num_point:
            raise ValueError(
                f"max_changepoint {max_changepoint} exceeds number of points {num_point}.")

        # (a) Evenly spaced initial changepoints -- identical to the original.
        step = num_point / (max_changepoint + 1)
        candidate_indices = [int(round((i + 1) * step)) for i in range(max_changepoint)]
        changepoints: List[int] = []
        for cp in sorted(candidate_indices):
            if cp < 1 or cp >= num_point:
                continue
            changepoints.append(cp)

        changepoints = changepoints[:max_changepoint]
        if not changepoints:
            return []

        # (b) Fit the baseline once serially before entering the loop.  Mirrors semantics of the
        # original method but avoids touching ``self`` from multiple threads at once.
        saved_m, saved_b, saved_l, saved_fitted = (
            self._subsequence_models, self._subsequence_boundaries,
            self._subsequence_lengths, self._is_fitted)
        try:
            baseline_models, baseline_bounds, baseline_lens = self._fitSegments(changepoints)
            self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                baseline_models, baseline_bounds, baseline_lens)
            self._is_fitted = True
            baseline_score = float(self.score(test_df=self.training_df))
        except Exception:
            return changepoints
        finally:
            # Restore prior state -- the caller of ``fit()`` is responsible for the final assignment.
            self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                saved_m, saved_b, saved_l)
            self._is_fitted = saved_fitted

        # Pre-compute the full time array once outside the while loop to avoid repeated .to_numpy()
        # calls and to pass a stable reference into worker processes.
        time_arr_full = self.training_df.index.to_numpy(dtype=float)

        # (c) Iteratively remove changepoints whose removal is cheap enough, parallelizing the
        # inner evaluation over candidates via ProcessPoolExecutor.
        while True:
            best_rm_idx, trial_scores = self._parallel_evaluate_removals(
                changepoints, baseline_score, time_arr_full)

            if best_rm_idx is None:
                break
            changepoints.pop(best_rm_idx)

            winning_ts = trial_scores.get(best_rm_idx, float('inf'))
            if np.isfinite(winning_ts):
                # The just-computed trial score of the removal we just made IS the new baseline.
                baseline_score = winning_ts
            else:  # pragma: no cover - fit failed on a candidate that shouldn't have been selected
                try:
                    new_models, new_bounds, new_lens = self._fitSegments(changepoints)
                    saved_m2, saved_b2, saved_l2, saved_fitted2 = (
                        self._subsequence_models, self._subsequence_boundaries,
                        self._subsequence_lengths, self._is_fitted)
                    try:
                        self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                            new_models, new_bounds, new_lens)
                        self._is_fitted = True
                        baseline_score = float(self.score(test_df=self.training_df))
                    finally:
                        self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (
                            saved_m2, saved_b2, saved_l2)
                        self._is_fitted = saved_fitted2
                except Exception:
                    return changepoints

        return changepoints

    def _adjustMaxChangepoints(self) -> int:
        """Adjust the maximum number of changepoints based on data points and species."""
        if (self.max_changepoint > 0) and (self.num_species > self.num_point / self.max_changepoint):
            # Few data points per species relative to changepoints -- cap so each
            # segment retains enough rows for a reliable PySINDy estimate.
            return self.num_point // self.num_species - 1
        else:
            return self.max_changepoint

    # NOTE: Deprecated because much lower scores than using _makeChangepointsWithRecursiveElimination
    def _makeChangepointsDivideAndconquor(self) -> List[int]:
        """Recursively split root group into left/right halves; prune children that don't beat threshold."""
        num_point = self.num_point
        max_changepoint = self._adjustMaxChangepoints()
        threshold_frac = self.max_fractional_reduction

        if not (self.max_changepoint > 0) or num_point <= 1:
            return []

        step = num_point / (max_changepoint + 1)
        candidate_indices = [int(round((i + 1) * step)) for i in range(max_changepoint)]
        changepoints: List[int] = []
        for cp in sorted(candidate_indices):
            if cp < 1 or cp >= num_point:
                continue
            changepoints.append(cp)
            last_kept = cp
        changepoints = changepoints[:max_changepoint]
        if not changepoints:
            return []

        def _score_for(cps):
            saved_m, saved_b, saved_l, saved_fitted = (
                self._subsequence_models, self._subsequence_boundaries,
                self._subsequence_lengths, self._is_fitted)
            try:
                _models, _bounds, _lens = self._fitSegments(cps)
                self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (_models, _bounds, _lens)
                self._is_fitted = True
                return float(self.score(test_df=self.training_df))
            except Exception:
                raise
            finally:
                self._subsequence_models, self._subsequence_boundaries, self._subsequence_lengths = (saved_m, saved_b, saved_l)
                self._is_fitted = saved_fitted

        def _dnc_node(cps, parent_score):
            if len(cps) <= 1:
                return list(cps)
            mid = len(cps) // 2
            left_half, right_half = cps[:mid], cps[mid:]
            try: score_left = _score_for(left_half)
            except Exception: score_left = float('-inf')
            try: score_right = _score_for(right_half)
            except Exception: score_right = float('-inf')
            keep_left = (score_left - parent_score) > threshold_frac * parent_score
            keep_right = (score_right - parent_score) > threshold_frac * parent_score
            survivors = []
            if keep_left:  survivors.extend(_dnc_node(left_half, score_left))
            if keep_right: survivors.extend(_dnc_node(right_half, score_right))
            return survivors

        try: root_score = _score_for(changepoints)
        except Exception: return changepoints
        dnc_survivors = _dnc_node(changepoints, root_score)

        def _greedy_remove(cps):
            if not cps: return []
            try: baseline = _score_for(cps)
            except Exception: return cps
            changed = True
            while changed:
                changed = False
                for idx in range(len(cps)):
                    trial = cps[:idx] + cps[idx + 1:]
                    try: ts = _score_for(trial)
                    except Exception: continue
                    if baseline - ts <= threshold_frac * baseline:
                        cps, baseline = trial, ts; changed = True; break
            return cps

        return _greedy_remove(dnc_survivors)

    def fit(self) -> 'PiecewiseSystemDiscovery':
        """Detect change points and fit a ``SystemDiscovery`` model to each segment.

        After this call, :attr:`_subsequence_models`, :attr:`_subsequence_boundaries`,
        and :attr:`_subsequence_lengths` are populated; :meth:`predict` is available.
        The baseline whole-timecourse model is built lazily on first access.
        """
        if self.changepoints is None:
            if self._is_random_changepoints:
                self.changepoints = self._makeBestRandomChangepoints()
            else:
                if self._is_changepoint_removal:
                    self.changepoints = self._makeChangepointsWithElimination()
                else:
                    self.changepoints = self._makeChangepointsWithoutElimination()
                #self.changepoints = self._makeChangepointsWithEliminationParallel()
        (self._subsequence_models, self._subsequence_boundaries,
        self._subsequence_lengths) = self._fitSegments(self.changepoints)
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
                suptitle="Actual vs. Predicted",
                species_names: Optional[List[str]] = None,
                is_nochangepoint_plot: bool = True,
                is_changepoint_plot: bool = True,
                **plt_kwargs: Any) -> PlotOptions:
        """Two-panel comparison: 0 change points (top) vs max_changepoint (bottom).

        Both panels show actual (scatter) vs predicted (line) species concentrations.
        The bottom panel marks each detected change point with a vertical dashed line.

        Parameters
        ----------
        num_true_point : int
            Number of actual-data scatter points to show per panel.
            -1 means show all points.  If the training_df has more than this many points,
        species_names : Optional[List[str]]
            List of species names to plot. If None, all species are plotted.
        suptitle : str
            Title for the entire figure.  Defaults to "Actual vs. Predicted".
        is_nochangepoint_plot: bool
            Plot with no changepoints
        is_changepoint_plot: bool
            Plot with changepoints

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
        if species_names is None:
            species_names = self.species_names
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
        if is_nochangepoint_plot and is_changepoint_plot:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        elif is_nochangepoint_plot:
            fig, ax_top = plt.subplots(1, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_top, **plt_kwargs)
        elif is_changepoint_plot:
            fig, ax_bot = plt.subplots(1, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        else:
            raise ValueError("At least one of is_nochangepoint_plot or is_changepoint_plot must be True.")
        change_point_times = [start for start, _ in self._subsequence_boundaries[1:]]
        ##
        def _draw(pred_df: pd.DataFrame, score: float, 
                vlines: Optional[List[float]] = None, **plt_options) -> None:
            po = PlotOptions(**plt_options)
            ax = po.ax
            ymax = actual_arr.max().max()
            for idx, name in enumerate(species_names):
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
                    ax.axvline(t, color="black", linestyle=":", lw=2.5, alpha=0.6)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
            if self.model_name.startswith("BIOMD"):
                model_num_str = str(int(self.model_name[6:]))
            else:
                model_num_str = self.model_name
            po.title = model_num_str + ": " + plt_options.get("title", "") + f" (Min p10 accuracy={score:.3f})"
            if ymax > 0.0:
                po.ylim = (0.0, ymax)
            po.apply()
        ##
        if is_nochangepoint_plot:
            _draw(fig=fig, ax=ax_top, pred_df=baseline_pred_df, score=baseline_score,  # type: ignore
                    title="0 change points", **plt_kwargs)
        if is_changepoint_plot:
            _draw(fig=fig, ax=ax_bot, pred_df=psd_pred_df, score=psd_score,  # type: ignore
                    title=f"{self.num_changepoint} change points",
                    vlines=change_point_times, **plt_kwargs)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return plot_options

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each subsequence."""
        self._requireFitted()
        print(str(self))

    def score(self, test_df: Optional[pd.DataFrame] = None, score_type="timecourse",
            col: str = cn.COL_P10, statistic: str = "min") -> float:
        """Return the average score across all subsequences.
        statistic:
            min, median, mean
        """
        self._requireFitted()
        score_df = self.getScoreDetails(test_df=test_df, score_type=score_type)
        if statistic == "min":
            return float(score_df[col].min())
        elif statistic == "median":
            return float(score_df[col].median())
        elif statistic == "mean":
            return float(score_df[col].mean())
        else:
            raise ValueError("statistic must be 'min' or 'median' or 'mean'")