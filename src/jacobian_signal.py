'''Creates and analyzes changes in Jacobian by creating a signal from the normalized Jacobians.'''


from src.plot_options import PlotOptions  # type: ignore
from src.change_point_detector import ChangePointDetector  # type: ignore
from src.timecourse import Timecourse  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Any, Tuple, List


class JacobianSignal(object):

    def __init__(self, timecourse: Timecourse) -> None:
        """ 
        Parameters
        ----------
        timecourse : Timecourse
            The timecourse object containing the time series data and Jacobian matrices.
        """
        self.timecourse = timecourse
        self._timecourse_df = timecourse.timecourse_df
        self._jacobian_collection_arr = timecourse.jacobian_collection_arr
        self._num_species = self.timecourse.model.num_species
        self._species_names = self.timecourse.model.species_names
        #
        self.signal_arr, self.species_signal_sq_arr = self._makeSignal()
    
    def _makeSignal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distance from the median Jacobian at each time point for the
        normalized Jacobian matrices, and smooth the resulting signal.
        The Frobenius norm of the difference between each normalized Jacobian
        and the median Jacobian is computed, and then smoothed using a Gaussian kernel.

        Returns
        -------
        signal_arr : np.ndarray
            The smoothed signal array representing the distance from the median Jacobian.
        species_signal_sq_arr : np.ndarray
            An array containing the squared differences for each species, normalized by the total signal.
        """
        norm_jacobian_arr = self._normalizeJacobian(self._jacobian_collection_arr,
                self._timecourse_df)
        median_jacobian_arr = np.median(norm_jacobian_arr, axis=0)
        signal_sq_arr = np.array([ np.sum((j - median_jacobian_arr) ** 2)
                for j in norm_jacobian_arr])/(self._num_species ** 2)
        species_signal_sq_arr = np.array([ (j - median_jacobian_arr) ** 2
                for j in norm_jacobian_arr])
        # Calculate the fractional contribution of each species to the total signal
        for t in range(species_signal_sq_arr.shape[0]):
            species_signal_sq_arr[t] /= (self._num_species ** 2) * signal_sq_arr[t]
        species_signal_sq_arr = np.nan_to_num(species_signal_sq_arr, nan=0.0)
        species_signal_sq_arr = species_signal_sq_arr**0.5
        #
        return np.sqrt(signal_sq_arr[1:]), species_signal_sq_arr
    
    def _normalizeJacobian(self, jacobian_arr: np.ndarray,
            timecourse_df: pd.DataFrame) -> np.ndarray:
        """Normalize a single Jacobian by the species std deviations."""
        jacobian_arr = self._denan(jacobian_arr)
        std_arr = timecourse_df.to_numpy(dtype=float).std(axis=0, ddof=1)
        std_arr = self._denan(std_arr)
        safe_std_arr = np.where(np.isclose(std_arr, 0.0), 1.0, std_arr)
        return jacobian_arr * (safe_std_arr[np.newaxis, np.newaxis, :]
                / safe_std_arr[np.newaxis, :, np.newaxis])

    def _denan(self, jacobian_arr: np.ndarray) -> np.ndarray:
        """Replace NaN entries in a Jacobian with zeros."""
        return np.where(np.isnan(jacobian_arr), 0.0, jacobian_arr)
    
    def makeDetector(self, max_change_point: int,
            min_fractional_reduction: float,
            min_segment_length: int) -> ChangePointDetector:
        """fit() step 4. signal_arr[k] is the signal for split index k+1
        (the time-grid index at which a new segment would begin).

        Returns a sorted (by time) list of accepted interior split indices.
        """
        detector = ChangePointDetector(self.signal_arr)
        detector.fit(max_change_point=max_change_point,
                min_fractional_reduction=min_fractional_reduction,
                min_segment_length=min_segment_length)
        return detector

    def plot(self,
            max_change_point: int,
            min_fractional_reduction: float,
            min_segment_length: int,
            **plt_kwargs: Any) -> List[PlotOptions]:
        """
        Two-panel plot.
        The top panel shows actual species concentrations over time, with detected change points marked.
        The bottom panel shows the signal derived from the Jacobian, with detected change points marked.

        Parameters
        ----------
        **plt_kwargs
            Forwarded to PlotOptions. Supported keys: fig, ax, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.  ``figsize`` is also
            accepted and consumed here (not passed to PlotOptions).

        Returns
        -------
        List[PlotOptions]
            Wraps the figure and the bottom axes.  Call ``plt.show()`` or
            ``po.fig.savefig(...)`` on the returned object as needed.
        """
        detector = self.makeDetector(max_change_point=max_change_point,
                min_fractional_reduction=min_fractional_reduction,
                min_segment_length=min_segment_length)
        change_point_idxs = [i.splice_start for i in detector.subsequences[1:]]
        change_point_times = [self._timecourse_df.index[i+1] 
                for i in change_point_idxs]  
        time_arr = self._timecourse_df.index.to_numpy(dtype=float)
        actual_arr = self._timecourse_df.to_numpy(dtype=float)
        # Do the plots
        figsize: tuple[float, float] = plt_kwargs.pop("figsize", (10, 8))
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # Top plot
        ##
        def _draw(po: PlotOptions):
            ax = po.ax
            for idx, name in enumerate(self._species_names):
                color = f"C{idx}"
                ax.plot(time_arr, actual_arr[:,idx],  # type: ignore
                    linestyle="-", color=color, label=f"{name} actual", zorder=3,
                )
            for t in change_point_times:
                ax.axvline(t, color="black", linestyle="--", lw=1.0, alpha=0.6)  # type: ignore
            ax.set_title(po.title, fontsize=11)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
        ##
        top_plot_options = PlotOptions(fig=fig, ax=ax_top, **plt_kwargs)
        _draw(top_plot_options)
        # Bottom plot
        bot_plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        ax_bot.plot(time_arr[:-1], self.signal_arr, color="steelblue", lw=1.0)  # type: ignore
        for t in change_point_times:
            ax_bot.axvline(t, color="black", linestyle="--", lw=1.0, alpha=0.6)  # type: ignore
        fig.suptitle("Change points over timecourse", fontsize=13, fontweight="bold")
        bot_plot_options.ylabel = "Signal (distance from median Jacobian)"
        fig.tight_layout()
        top_plot_options.apply()
        bot_plot_options.apply()
        #
        return [top_plot_options, bot_plot_options]