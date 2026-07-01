"""Piecewise system discovery: detects Jacobian-based change points in a
Timecourse and fits a separate SystemDiscovery model to each segment,
blending predictions across segment boundaries with a Gaussian kernel.

See docs/piecewise_system_discovery.md for the full design.
"""

import bisect
from typing import Any, List, Tuple

import collections
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from src.change_point_detector import ChangePointDetector  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.system_discovery import ScoreInfo, SystemDiscovery  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

NULL_DF = pd.DataFrame()

PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-point segments."""

    def __init__(
        self,
        timecourse: Timecourse,
        max_change_point: int = 2,
        min_fractional_reduction: float = 0.1,  
        min_segment_length: int = 100,
        predict_kernel_bandwidth: float = 0.5,
        **sd_kwargs: Any,
    ) -> None:
        """_summary_

        Args:
            timecourse (Timecourse): _description_
            max_change_point (int, optional): _description_. Defaults to 2.
            min_fractional_reduction (float, optional): _description_. Defaults to 0.1.
            min_segment_length (int, optional): _description_. Defaults to 100.
            predict_kernel_bandwidth (float, optional): _description_. Defaults to 0.5.
            **kwargs: Arguments for SystemDiscovery constructor (e.g. fit_kernel_bandwidth, model_name).
        """
        self.timecourse = timecourse
        self.max_change_point = max_change_point
        self.min_fractional_reduction = min_fractional_reduction
        self.min_segment_length = min_segment_length
        self.predict_kernel_bandwidth = predict_kernel_bandwidth
        sd_kwargs["poly_degree"] = sd_kwargs.get("poly_degree", 1)
        self._sd_kwargs = sd_kwargs

        self._segment_models: List[SystemDiscovery] = []
        self._segment_boundaries: List[Tuple[float, float]] = []
        self._segment_lengths: List[int] = []
        self._is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")

    @staticmethod
    def _gaussianSmooth(times: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray:
        """Nadaraya-Watson Gaussian kernel smoothing of `values` over `times`."""
        delta_arr = times[:, np.newaxis] - times[np.newaxis, :]
        weight_arr = np.exp(-0.5 * (delta_arr / bandwidth) ** 2)
        return (weight_arr @ values) / weight_arr.sum(axis=1)
    
    def _denan(self, jacobian_arr: np.ndarray) -> np.ndarray:
        """Replace NaN entries in a Jacobian with zeros."""
        return np.where(np.isnan(jacobian_arr), 0.0, jacobian_arr)
    
    def _normalizeJacobian(self, jacobian_arr: np.ndarray,
            timecourse_df: pd.DataFrame) -> np.ndarray:
        """Normalize a single Jacobian by the species std deviations."""
        jacobian_arr = self._denan(jacobian_arr)
        std_arr = timecourse_df.to_numpy(dtype=float).std(axis=0, ddof=1)
        std_arr = self._denan(std_arr)
        safe_std_arr = np.where(np.isclose(std_arr, 0.0), 1.0, std_arr)
        return jacobian_arr * (safe_std_arr[np.newaxis, np.newaxis, :]
                / safe_std_arr[np.newaxis, :, np.newaxis])

    def _computeChangePointSignalDifference(self, timecourse_df: pd.DataFrame,
            jacobian_collection_arr: np.ndarray) -> np.ndarray:
        """fit() steps 2-3: normalized-Jacobian Frobenius distance, smoothed.

        Returns one value per interior candidate split index 1..num_point-1
        (signal[k] corresponds to split index k+1: the point where a new
        segment would start if a change point were placed there).
        """
        num_species = self.timecourse.model.num_species
        norm_jacobian_arr = self._normalizeJacobian(jacobian_collection_arr,
                timecourse_df)
        diff_arr = norm_jacobian_arr[1:] - norm_jacobian_arr[:-1]
        raw_signal_arr = np.linalg.norm(
                diff_arr.reshape(diff_arr.shape[0], -1), axis=1) / (num_species ** 2)
        return np.sqrt(raw_signal_arr)
        #split_time_arr = timecourse_df.index.to_numpy(dtype=float)[1:]

    # TODO: Plot and evaluate, comparing with the difference approach 
    def _computeChangePointSignalMedian(self, timecourse_df: pd.DataFrame,
            jacobian_collection_arr: np.ndarray) -> np.ndarray:
        """
        Calculate the distance from the median Jacobian at each time point for the
        normalized Jacobian matrices, and smooth the resulting signal.
        The Frobenius norm of the difference between each normalized Jacobian
        and the median Jacobian is computed, and then smoothed using a Gaussian kernel.
        """
        num_species = self.timecourse.model.num_species
        norm_jacobian_arr = self._normalizeJacobian(jacobian_collection_arr,
                timecourse_df)
        median_jacobian_arr = np.median(norm_jacobian_arr, axis=0)
        raw_signal_arr = np.array([ np.sum((j - median_jacobian_arr) ** 2)
                for j in norm_jacobian_arr])/(num_species ** 2)
        #split_time_arr = timecourse_df.index.to_numpy(dtype=float)
        return np.sqrt(raw_signal_arr[1:])


    def _detectChangePoints(self, signal_arr: np.ndarray, num_point: int) -> List[int]:
        """fit() step 4. signal_arr[k] is the signal for split index k+1
        (the time-grid index at which a new segment would begin).

        Returns a sorted (by time) list of accepted interior split indices.
        """
        detector = ChangePointDetector(signal_arr)
        # self.min_fraction_reduction gates the SSE reduction a split must achieve
        # (as a fraction of the total adjusted sum of squares), not the raw
        # signal value at the split point: the split index itself often falls
        # on a low-signal point right after a spike, since the detector
        # chooses the boundary that best separates two segments, not the
        # point with the largest signal value.
        detector.fit(max_change_point=self.max_change_point,
                min_fractional_reduction=self.min_fractional_reduction,
                min_segment_length=self.min_segment_length)

        # Map signal index k to timecourse split index k + 1
        split_indices = [info.splice_start + 1 for info in detector.subsequences
                if info.splice_start > 0]

        return sorted(split_indices)

    def fit(self) -> "PiecewiseSystemDiscovery":
        """fit() steps 1-4: detect change points, fit per-segment models."""
        raw_df = self.timecourse.timecourse_df
        jacobian_collection_arr = self.timecourse.jacobian_collection_arr
        num_point = raw_df.shape[0]
        time_arr = raw_df.index.to_numpy(dtype=float)

        signal_arr = self._computeChangePointSignalMedian(raw_df, jacobian_collection_arr)
        split_index_list = self._detectChangePoints(signal_arr, num_point)
        boundary_index_arr = [0] + split_index_list + [num_point]

        self._segment_models = []
        self._segment_boundaries = []
        self._segment_lengths = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            segment_df = raw_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < num_point else time_arr[-1]
            self._segment_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._segment_lengths.append(hi - lo)
            model = SystemDiscovery(segment_df, **self._sd_kwargs).fit()
            self._segment_models.append(model)

        self._is_fitted = True
        return self

    def dynamicFit(self) -> "PiecewiseSystemDiscovery":
        """Find optimal change points via dynamic programming on the median signal.

        Partitions the signal from ``_computeChangePointSignalMedian`` into
        ``max_change_point + 1`` segments minimising the total within-segment
        sum of squared deviations from each segment mean.  Respects
        ``min_segment_length``.

        After finding the optimal split indices, fits a ``SystemDiscovery``
        model on each segment — identical post-processing to ``fit()``.

        Raises
        ------
        RuntimeError
            If the series is too short to accommodate the requested number of
            segments at the current ``min_segment_length``.
        """
        raw_df = self.timecourse.timecourse_df
        jacobian_collection_arr = self.timecourse.jacobian_collection_arr
        num_point = raw_df.shape[0]
        time_arr = raw_df.index.to_numpy(dtype=float)
        num_seg = self.max_change_point + 1
        mseg = self.min_segment_length

        signal_arr = np.nan_to_num(
            self._computeChangePointSignalMedian(raw_df, jacobian_collection_arr),
            nan=0.0,
        )

        # Prefix sums for O(1) within-segment SSE: cost(lo, hi) = Σ(x-μ)²
        prefix_sum = np.concatenate([[0.0], np.cumsum(signal_arr)])
        prefix_sum_sq = np.concatenate([[0.0], np.cumsum(signal_arr ** 2)])

        INF = float("inf")
        # dp[i][s] = min total SSE to partition signal[0:i] into s segments
        dp = np.full((num_point + 1, num_seg + 1), INF)
        backtrack = np.full((num_point + 1, num_seg + 1), -1, dtype=int)

        for i in range(mseg, num_point + 1):
            s = prefix_sum[i]
            dp[i][1] = float(prefix_sum_sq[i] - s * s / i)

        for s in range(2, num_seg + 1):
            for i in range(s * mseg, num_point + 1):
                j_lo = (s - 1) * mseg
                j_hi = i - mseg
                if j_lo > j_hi:
                    continue
                j_arr = np.arange(j_lo, j_hi + 1)
                prev = dp[j_arr, s - 1]
                lengths = i - j_arr
                seg_sum = prefix_sum[i] - prefix_sum[j_arr]
                seg_sq = prefix_sum_sq[i] - prefix_sum_sq[j_arr]
                seg_cost = seg_sq - seg_sum ** 2 / lengths
                total = prev + seg_cost
                total[prev >= INF] = INF
                best = int(np.argmin(total))
                if total[best] < INF:
                    dp[i][s] = total[best]
                    backtrack[i][s] = int(j_arr[best])

        if dp[num_point][num_seg] >= INF:
            raise RuntimeError(
                f"Cannot partition {num_point} points into {num_seg} segments "
                f"each with at least {mseg} points."
            )

        split_index_list: List[int] = []
        i, s = num_point, num_seg
        while s > 1:
            j = int(backtrack[i][s])
            split_index_list.append(j)
            i, s = j, s - 1
        split_index_list.reverse()

        boundary_index_arr = [0] + split_index_list + [num_point]
        self._segment_models = []
        self._segment_boundaries = []
        self._segment_lengths = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            segment_df = raw_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < num_point else time_arr[-1]
            self._segment_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._segment_lengths.append(hi - lo)
            model = SystemDiscovery(segment_df, **self._sd_kwargs).fit()
            self._segment_models.append(model)

        self._is_fitted = True
        return self

    def predict_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """Blend per-segment derivative predictions at (t, x) with a Gaussian
        kernel over each segment's midpoint. See docs/piecewise_system_discovery.md.
        """
        # Eliminated smoothing across model segments
        self._require_fitted()
        # Find the model for this segment
        segment_idx = int(np.sum([1 for _, end in self._segment_boundaries if t > end]))
        model = self._segment_models[segment_idx]
        return model.predictOneStepDerivative(x)
#        midpoint_arr = np.array(
#                [0.5 * (start + end) for start, end in self._segment_boundaries])
#        weight_arr = np.exp(-0.5 * ((t - midpoint_arr) / self.predict_kernel_bandwidth) ** 2)
#        derivative_arr = np.array([
#                model.predictOneStepDerivative(x) for model in self._segment_models])
#        result = (weight_arr[:, np.newaxis] * derivative_arr).sum(axis=0) / weight_arr.sum()
#        return result

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
            return self.predict_derivative(t, x)

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
        species_names = self._segment_models[0].species_names
        return pd.DataFrame(sol.y.T, index=time_arr, columns=species_names)

    def score(self) -> ScoreInfo:
        """Length-weighted aggregation of per-segment ScoreInfo. See
        docs/piecewise_system_discovery.md `score()` section."""
        self._require_fitted()
        weighted_values: List[float] = []
        num_nonzero_term = 0
        for model, length in zip(self._segment_models, self._segment_lengths):
            info = model.score()
            weighted_values.extend(info.values * length)
            num_nonzero_term += info.num_nonzero_term
        return ScoreInfo(
                min=float(np.min(weighted_values)),
                median=float(np.median(weighted_values)),
                max=float(np.max(weighted_values)),
                values=weighted_values,
                num_nonzero_term=num_nonzero_term,
        )

    def __str__(self) -> str:
        block_list: List[str] = []
        for idx, (model, (start, end)) in enumerate(
                zip(self._segment_models, self._segment_boundaries), start=1):
            header = f"[Segment {idx}: t in [{start:.1f}, {end:.1f})]"
            equation_line_list = [f"  {line}" for line in str(model).strip().split("\n")]
            block_list.append("\n".join([header] + equation_line_list))
        return "\n\n".join(block_list)

    def plotPiecewise(self, num_true_point: int = 20, **plt_kwargs: Any) -> PlotOptions:
        """Two-panel comparison: 0 change points (top) vs max_change_point (bottom).

        Both panels show actual (scatter) vs predicted (line) species concentrations.
        The bottom panel marks each detected change point with a vertical dashed line.

        Parameters
        ----------
        num_true_point : int
            Number of actual-data scatter points to show per panel.
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
        species_names = self._segment_models[0].species_names
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

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        change_point_times = [start for start, _ in self._segment_boundaries[1:]]

        plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)

        def _draw(po: PlotOptions, pred_df: pd.DataFrame | None, title: str,
                vlines: List[float] | None = None) -> None:
            ax = po.ax
            for idx, name in enumerate(species_names):
                color = f"C{idx}"
                ax.scatter(  # type: ignore
                    time_arr[::num_skip], actual_arr[::num_skip, idx],
                    s=15, color=color, label=f"{name} actual", zorder=3,
                )
                if pred_df is not None and name in pred_df.columns:
                    ax.plot(  # type: ignore
                        pred_df.index, pred_df[name],
                        "-", lw=1.5, color=color, label=f"{name} predicted",
                    )
            if vlines:
                for t in vlines:
                    ax.axvline(t, color="black", linestyle="--", lw=1.0, alpha=0.6)  # type: ignore
            ax.set_title(title, fontsize=11)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
            po.apply()

        _draw(PlotOptions(fig=fig, ax=ax_top, **plt_kwargs), baseline_pred_df, "0 change points")
        _draw(plot_options, psd_pred_df, f"{self.max_change_point} change point(s)",
                vlines=change_point_times)
        fig.suptitle("Actual vs Predicted", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return plot_options

    @classmethod
    def plotBiomodelsSignal(
        cls,
        model_number: int = -1,
        timecourse: Timecourse | None = None,
        signal_kind: str = "median",
        max_change_point: int = 2,
        min_fractional_reduction: float = 0.1,
        min_segment_length: int = 100,
        is_standarized: bool = True,
        **kwargs: Any,
    ) -> PlotBiomodelsSignalResult:
        """Plot the change point signal and timecourses for a BioModel.

        Parameters
        ----------
        model_number : int
            BioModel number (e.g. 331 for BIOMD0000000331).
        signal_kind : str
            ``"difference"`` — frame-to-frame normalized Jacobian distance.
            ``"median"``     — distance of each Jacobian from the median Jacobian.
        is_standarized : bool
            Whether to standardize the signal.
        **kwargs
            Forwarded to PlotOptions (bottom/signal axes).  ``figsize`` is
            consumed here and not passed to PlotOptions.

        Returns
        -------
        PlotOptions
            Wraps the figure and the bottom (signal) axes.
        """
        if signal_kind not in ("difference", "median"):
            raise ValueError(
                f"signal_kind must be 'difference' or 'median', got {signal_kind!r}"
            )

        model_name = f"BIOMD{model_number:010d}"
        if timecourse is None:
            timecourse = TimecourseIterator().getTimecourse(model_name)
        else:
            # Get the correct model name
            model_name = timecourse.model.model_name
        timecourse_df = timecourse.timecourse_df
        if is_standarized:
            timecourse_df = (timecourse_df - timecourse_df.mean()) / timecourse_df.std()
        jacobian_collection_arr = timecourse.jacobian_collection_arr
        time_arr = timecourse_df.index.to_numpy(dtype=float)
        num_point = timecourse_df.shape[0]

        psd = cls(timecourse, max_change_point=max_change_point,
                min_fractional_reduction=min_fractional_reduction,
                min_segment_length=min_segment_length)

        if signal_kind == "difference":
            signal_arr = psd._computeChangePointSignalDifference(  # pylint: disable=protected-access
                timecourse_df, jacobian_collection_arr)
            signal_times = time_arr[1:]
            detection_signal = signal_arr
        else:
            signal_arr = psd._computeChangePointSignalMedian(  # pylint: disable=protected-access
                timecourse_df, jacobian_collection_arr)
            signal_times = time_arr[1:]
            detection_signal = signal_arr[:-1]

        change_point_indices = psd._detectChangePoints(detection_signal, num_point)  # pylint: disable=protected-access
        change_point_times = [time_arr[idx] for idx in change_point_indices]

        figsize: tuple[float, float] = kwargs.pop("figsize", (10, 8))
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        plot_options = PlotOptions(fig=fig, ax=ax_bot, **kwargs)

        top_po = PlotOptions(fig=fig, ax=ax_top, title=model_name)
        for i, name in enumerate(timecourse_df.columns):
            ax_top.plot(timecourse_df.index, timecourse_df[name],  # type: ignore
                        color=f"C{i}", label=name)
        top_po.apply()
        ax_top.set_ylabel("normalized concentration", fontsize=10)  # type: ignore

        ax_bot.plot(signal_times, signal_arr, color="black", lw=1.5)  # type: ignore
        for t in change_point_times:
            ax_bot.axvline(t, color="red", linestyle="--", lw=1.0, alpha=0.8)  # type: ignore
        ax_bot.set_title(f"Change point signal ({signal_kind})", fontsize=11)  # type: ignore
        plot_options.apply()
        ax_bot.set_ylabel("Signal", fontsize=10)  # type: ignore

        fig.suptitle(f"{model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        #
        result = PlotBiomodelsSignalResult(
            plot_options=plot_options,
            piecewise_system_discovery=psd,
            change_point_times=change_point_times,
        )
        return result

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each segment."""
        self._require_fitted()
        print(str(self))
