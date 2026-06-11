"""Collection of Trajectory objects sharing the same Model."""

import src.constants as cn  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.trajectory import Trajectory  # type: ignore

from collections import namedtuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List, Optional, Union

Segment = namedtuple("Segment", ["trajectory", "start_time", "end_time"])


class TrajectoryCollection(object):
    """An ordered, non-overlapping collection of Trajectory sharing one Model.

    Use split() to partition a single Trajectory into a TrajectoryCollection.
    """

    def __init__(self, trajectories: List[Trajectory]) -> None:
        """
        Parameters
        ----------
        trajectories : List[Trajectory]
            All must share the same Model. Sorted by Trajectory.__lt__;
            overlapping time ranges raise ValueError.
        """
        # Error checking
        if not trajectories:
            raise ValueError("trajectories must not be empty.")
        self.trajectories = sorted(trajectories)
        if not self.isConsecutive():
            raise ValueError("Trajectories must be consecutive (no gaps or overlaps). "
                    "continuous time range.")
        #
        self.model = self.trajectories[0].model
        self.segments = [Segment(traj, traj.start_time, traj.end_time)
                for traj in self.trajectories]
        for traj in trajectories[1:]:
            if traj.model != self.model:
                raise ValueError(
                        "All trajectories must share the same Model.")
        self.start_time = self.segments[0].start_time
        self.end_time = self.segments[-1].end_time

    def isConsecutive(self) -> bool:
        """True iff trajectories are consecutive (no gaps or overlaps)."""
        for t1, t2 in zip(self.trajectories, self.trajectories[1:]):
            if not np.isclose(t1.end_time, t2.start_time):
                return False
        return True 

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryCollection):
            raise ValueError("Can only compare TrajectoryCollection to another ")
        if len(self.trajectories) != len(other.trajectories):
            return False
        return all(t1 == t2 for t1, t2 in zip(self.trajectories, other.trajectories))

    def makeTimecourse(self) -> pd.DataFrame:
        """Concatenate actual timecourses, dropping duplicate boundary rows.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with one row per unique timepoint.
        """
        dfs = []
        for i, traj in enumerate(self.trajectories):
            tc_df = traj.timecourse_df
            if i > 0:
                tc_df = tc_df.iloc[1:]
            dfs.append(tc_df)
        return pd.concat(dfs)

    def plotTimecourse(self, **kwargs) -> PlotOptions:
        """Plot the pieced-together timecourse with dashed vertical separators.

        Each trajectory's timecourse is drawn continuously; a vertical dashed
        line marks the boundary between adjacent trajectories.  One legend
        entry per species.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions. Supported keys: ax, fig, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.

        Returns
        -------
        PlotOptions
        """
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        for i, name in enumerate(self.model.species_names):
            color = f"C{i}"
            for j, traj in enumerate(self.trajectories):
                label = name if j == 0 else None
                ax.plot(                                           # type: ignore 
                        traj.timecourse_df.index,
                        traj.timecourse_df[name],
                        color=color,
                        label=label,
                )
        for traj in self.trajectories[:-1]:
            ax.axvline(x=traj.end_time, color="black", linestyle="--",   # type: ignore
                    linewidth=0.8)
        plot_options.apply()
        return plot_options

    @classmethod
    def split(cls,
            trajectory: Trajectory,
            timepoints: List[float]) -> "TrajectoryCollection":
        """Partition a Trajectory at the given split times.

        Each split time becomes the shared end/start boundary between adjacent
        sub-trajectories.  Times not exactly in timepoint_arr snap to the
        nearest available timepoint.  Split times that snap to start_time or
        end_time are ignored.

        Parameters
        ----------
        trajectory : Trajectory
        timepoints : List[float]

        Returns
        -------
        TrajectoryCollection
        """
        tp_arr = trajectory.timepoint_arr
        snapped = []
        for t in timepoints:
            idx = int(np.argmin(np.abs(tp_arr - t)))
            snapped.append(float(tp_arr[idx]))
        snapped = sorted({
                t for t in snapped
                if trajectory.start_time < t < trajectory.end_time
        })
        boundaries = [trajectory.start_time] + snapped + [trajectory.end_time]
        sub_trajectories = [
                trajectory.makeSubmodel(boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)
        ]
        return cls(sub_trajectories)

    @classmethod
    def autoSplit(cls,
            trajectory: Trajectory,
            num_split: int,
            is_slow_subspace: bool = False,
            is_species_prediction: bool = False,
            min_num_point_per_segment: int = 3,
            ) -> "TrajectoryCollection":
        """Find split points greedily, each step picking the globally best available split.

        At each iteration every existing segment is scanned for its best internal
        split point. The split that gives the greatest cost reduction across all
        segments is applied. Segments with only two timepoints (no internal point)
        are skipped. Stops early if no splittable segment remains.

        Parameters
        ----------
        trajectory : Trajectory
        num_split : int
            Number of splits (num_split+1 sub-trajectories). Clamped to
            [0, len(timepoint_arr) - 2].
        min_num_point_per_segment : int
            Minimum number of timepoints required for a segment to be considered splittable.

        Returns
        -------
        TrajectoryCollection
        """
        from src.linear_predictor import LinearPredictor  # local import avoids circularity
        from src.slow_subspace_predictor import SlowSubspacePredictor  # type: ignore

        jacobian_selection = cn.JAC_MEDIAN
        tp_arr = trajectory.timepoint_arr
        num_point = trajectory.num_point
        num_split = max(0, min(num_split, num_point - 2))

        if num_split == 0:
            return cls([trajectory])

        INF = float("inf")
        cost_cache: dict = {}

        def _segmentCost(i: int, j: int) -> float:
            # llambda is a regularization parameter, lowering cost for longer segments
            if i == j:
                return 0.0
            if (i, j) in cost_cache:
                return cost_cache[(i, j)]
            sub_traj = trajectory.makeSubmodel(float(tp_arr[i]), float(tp_arr[j]))
            if is_slow_subspace:
                lp: Union[SlowSubspacePredictor, LinearPredictor] = SlowSubspacePredictor(sub_traj,
                        num_step=-1)
            else:
                lp = LinearPredictor(sub_traj,
                        jacobian_selection=jacobian_selection, num_step=-1,
                        is_species_prediction=is_species_prediction)  # Changed from -1 to 1
            c = lp.cost
            cost_cache[(i, j)] = c if np.isfinite(c) else INF
            return cost_cache[(i, j)]

        def _bestSplitForSegment(istart: int, iend: int):
            """Return (best_k, combined_cost) for the best split of [istart, iend].

            Returns (None, INF) when there are no internal timepoints to split on.
            """
            best_k = None
            best_cost = INF
            for k in range(istart + 1, iend):
                c = 1/np.max([_segmentCost(istart, k), _segmentCost(k, iend)])
                #c = _segmentCost(istart, k) +  _segmentCost(k, iend)
                if c < best_cost:
                    best_cost = c
                    best_k = k
            return best_k, best_cost

        # Maintain active segments as a dict: (istart, iend) -> segment cost.
        segments_dct: dict = {(0, num_point - 1): _segmentCost(0, num_point - 1)}

        for _ in range(num_split):
            # Among all segments' best splits, pick the one with the greatest gain.
            best_gain = -INF
            best_segment = None
            best_split_k = None

            for (istart, iend), seg_cost in segments_dct.items():
                if iend - istart < min_num_point_per_segment:
                    continue
                split_k, split_cost = _bestSplitForSegment(istart, iend)
                if split_k is None:
                    continue
                gain = seg_cost - split_cost
                if gain > best_gain:
                    best_gain = gain
                    best_segment = (istart, iend)
                    best_split_k = split_k

            if best_segment is None:
                break  # every remaining segment has only 2 points; cannot split

            assert best_split_k is not None
            istart, iend = best_segment
            del segments_dct[best_segment]
            segments_dct[(istart, best_split_k)] = _segmentCost(istart, best_split_k)
            segments_dct[(best_split_k, iend)] = _segmentCost(best_split_k, iend)

        # Recover split times from the interior boundaries between segments.
        sorted_segs = sorted(segments_dct.keys())
        split_indices = [iend for (_, iend) in sorted_segs[:-1]]
        split_times = [float(tp_arr[idx]) for idx in split_indices]
        return cls.split(trajectory, split_times)
