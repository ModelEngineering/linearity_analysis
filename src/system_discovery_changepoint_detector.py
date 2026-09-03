'''Detects changepoints based on changes in the accuracy of one-step prediction of derivatives of a timecourse.'''


import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List, Union, Optional, Tuple, cast  # type: ignore

from src.change_point_detector import ChangePointDetector  # type: ignore
import src.constants as cn  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.score import Score  # type: ignore
from src.system_discovery import NULL_DF, SystemDiscovery  # type: ignore

MIN_SIGNIFICANT_VALUE = 1e-8
LARGE_VALUE = 1e8


class SystemDiscoveryChangepointDetector(object):
    # Detects changepoints in the timecourse of the training data for the SystemDiscovery object.

    def __init__(self, system_discovery: SystemDiscovery, max_changepoint: int = 0,
                min_segment_length: int = 5, min_fractional_reduction: float = 0.01):
        """Initialize the detector with parameters set during construction."""
        self.system_discovery = system_discovery
        self.training_df = system_discovery.training_df
        self._scaler = system_discovery._scaler
        # Change-point detection parameters set during construction.
        self.max_changepoint = max_changepoint
        self.min_segment_length = min_segment_length
        self.min_fractional_reduction = min_fractional_reduction

        self.changepoints: List[int] = []  # List of indices representing detected changepoints
        self.change_point_detector: Union[ChangePointDetector, None] = None  # Instance of ChangePointDetector for detecting changepoints

    def fit(self) -> List[int]:
        """Detect changepoints using parameters set during construction.

        Args:
            (none — uses self.max_changepoint, self.min_segment_length,
            and self.min_fractional_reduction).

        Returns:
            List[int]: A list of detected changepoint indices into the training DataFrame.
        """
        true_df, pred_df = self._calculateNormalizedOneStepPredictions()
        signal_arr = self._calculateSignal(true_df, pred_df)
        self.change_point_detector = ChangePointDetector(signal_arr, max_changepoint=self.max_changepoint,
                min_segment_length=self.min_segment_length,
                min_fractional_reduction=self.min_fractional_reduction)
        self.change_point_detector.fit()
        # First partition starts at index 0 (not a changepoint); subsequent partitions
        # start at the detected change-point indices.
        self._is_detected = True
        self.changepoints = [cp.splice_start for cp in self.change_point_detector.subsequences[1:]]
        return self.changepoints

    def _calculateNormalizedOneStepPredictions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the normalized one step predicted and true values of derivatives for each time point in the
        training data, and return them as a tuple.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the normalized
            true values and normalized predicted values of derivatives
            for each time point.
        """
        # Calculate one step prediction of derivatives using the fitted ODE model,
        # then normalize them so they are on the same scale as the normalized true
        # derivatives below. predictOneStepDerivative returns physical units (denormalized).
        predictions = []
        for x in self.training_df.values:
            predicted_derivative = self.system_discovery.predictOneStepDerivative(x)
            predictions.append(self._scaler.normalize(np.asarray(predicted_derivative, dtype=float)))
        pred_df = pd.DataFrame(predictions, index=self.training_df.index,
                columns=self.training_df.columns)
        true_df = self.system_discovery.Xdot_df
        true_derivative_arr = self._scaler.normalize(true_df.to_numpy())
        true_df = pd.DataFrame(true_derivative_arr, index=true_df.index,
                columns=true_df.columns)
        # Reindex true_df to match pred_df's index: Xdot is naturally one row shorter
        # (first training point has no derivative estimate), so reindex fills the missing
        # leading row with NaN rather than dropping it. This keeps _calculateSignal's
        # per-timepoint computation well-defined for all training points.
        true_df = true_df.reindex(pred_df.index)
        return true_df, pred_df

    def is_detected(self) -> bool:
        """Check if changepoints have been detected.

        Returns:
            bool: True if changepoints have been detected, False otherwise.
        """
        return self.change_point_detector is not None

    def plotTimecourseWithChangepoints(self, changepoints: Optional[List[int]] = None,
                **plot_kwargs) -> PlotOptions:
        """Plot the timecourse with detected changepoints and log10(ASS) per segment.

        For each subsequence delimited by consecutive change points (or start/end of
        the data), display ``log10(ASS)`` centered near the top of the plot at the
        midpoint between adjacent boundaries, where ASS is the adjusted sum of squares
        of the one-step-prediction-accuracy signal used to detect changepoints.

        Args:
            changepoints (List[int], optional): Indices representing detected change points.
                If ``None``, uses stored ``self.changepoints`` from a prior ``detect()`` call.
            plot_kwargs: Additional keyword arguments to customize the plot
                (e.g., title, xlabel, ylabel, legend, xlim, ylim).

        Returns:
            PlotOptions: The options for the plot.
        """
        plot_kwargs = dict(plot_kwargs)
        if self.change_point_detector is None:
            raise RuntimeError(
                "ChangepointDetector must be fit before plotting; call detect() first.")
        if not 'figsize' in plot_kwargs:
            plot_kwargs['figsize'] = (10, 6)
        if not 'xlabel' in plot_kwargs:
            plot_kwargs['xlabel'] = 'Time'
        if not 'ylabel' in plot_kwargs:
            plot_kwargs['ylabel'] = 'Value'
        if not 'title' in plot_kwargs:
            plot_kwargs['title'] = 'Timecourse with Detected Changepoints'
        if not 'xlim' in plot_kwargs:
            plot_kwargs['xlim'] = (self.training_df.index[0], self.training_df.index[-1])
        if not 'ylim' in plot_kwargs:
            plot_kwargs['ylim'] = (self.training_df.min().min(), self.training_df.max().max())
        plot_options = PlotOptions(**plot_kwargs)
        ax = cast(plt.Axes, plot_options.ax)  # type: ignore
        fig = cast(plt.Figure, plot_options.fig)  # type: ignore
        time_arr = self.training_df.index.to_numpy()
        ax.plot(time_arr, self.training_df.values, label='Timecourse')

        if changepoints is None:
            cp_indices = [cp.splice_start for cp in self.change_point_detector.subsequences[1:]]
        else:
            cp_indices = changepoints
        for i, cp in enumerate(cp_indices):
            lbl = 'Changepoint' if i == 0 else None
            ax.axvline(x=time_arr[cp], color='r', linestyle=':', lw=2.5, alpha=0.6, label=lbl)

        # Annotate each segment with log10(ASS). ASS is taken from the ChangePointDetector's
        # subsequence records, which were computed over the one-step-prediction-accuracy signal.
        segments = self.change_point_detector.subsequences
        ylim_top = float(self.training_df.max().max())
        ylim_bot = float(self.training_df.min().min())
        for i, segment in enumerate(segments):
            seg_start, seg_end, seg_ass = segment.splice_start, segment.splice_end, segment.adj_sum_sq
            mid_time = time_arr[(seg_start + seg_end) // 2 ] if (seg_start + seg_end) // 2 < len(time_arr) else time_arr[0]
            if seg_ass > 0:
                label = f"{np.log10(seg_ass):.2f}"
            else:
                label = "log10(ASS)=--"
            increment = (ylim_top - ylim_bot) / 10
            y_offset = ylim_bot - 1.7 * increment
            ax.text(mid_time, y_offset, label, rotation=90,    # type: ignore
                    ha="center", va="bottom", fontsize=10, color="blue")
        plot_options.apply()
        legend = self.training_df.columns.tolist() + ['Changepoint']
        ax.legend(legend, loc='upper right', fontsize=10)
        fig.tight_layout()
        return plot_options

    @staticmethod 
    def _calculateSignal(true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> np.ndarray:
        """
        Calculates a one-dimensional signal related to the accuracy of one-step prediction of derivatives.

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.

        Returns
        -------
        np.ndarray
            A one-dimensional array representing the signal of accuracy of one-step prediction of derivatives.
        """
        ape_df = prediction_df - true_df
        ape_df = ape_df.fillna(0)
        ape_df = ape_df.abs()
        result_arr = ape_df.max(axis=1).to_numpy()
        return result_arr