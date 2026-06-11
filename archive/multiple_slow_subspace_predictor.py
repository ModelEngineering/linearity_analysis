"""Piece-wise slow-subspace prediction using a TrajectoryCollection."""

from src.plot_options import PlotOptions  # type: ignore
from src.score import Score  # type: ignore
from src.slow_subspace_predictor import SlowSubspacePredictor  # type: ignore
from src.trajectory_collection import TrajectoryCollection  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Optional


class MultipleSlowSubspacePredictor(object):
    """Piece-wise slow-subspace predictor across a TrajectoryCollection.

    Each Trajectory is predicted independently with SlowSubspacePredictor;
    results are concatenated into a single timecourse.  Boundary timepoints
    shared between adjacent segments appear once in the output (the duplicate
    first row of each interior segment is dropped).
    """

    def __init__(self,
            trajectory_collection: TrajectoryCollection,
            eigenvalue_threshold: Optional[float] = None,
            num_step: int = 1) -> None:
        """
        Parameters
        ----------
        trajectory_collection : TrajectoryCollection
        eigenvalue_threshold : Optional[float]
            Passed to each SlowSubspacePredictor.  None uses the per-segment
            default (1 / step_size of that segment).
        num_step : int
            Window size in timepoints passed to each SlowSubspacePredictor.
        """
        self.trajectory_collection = trajectory_collection
        self.eigenvalue_threshold = eigenvalue_threshold
        self.num_step = num_step

    def predict(self) -> pd.DataFrame:
        """Predict concentrations across all segments.

        Returns
        -------
        pd.DataFrame
            Time-indexed with columns matching the model's species names.
        """
        pred_dfs = []
        for i, traj in enumerate(self.trajectory_collection.trajectories):
            ssp = SlowSubspacePredictor(traj,
                    eigenvalue_threshold=self.eigenvalue_threshold,
                    num_step=self.num_step)
            prediction_df = ssp.predict()
            if i > 0:
                prediction_df = prediction_df.iloc[1:]
            pred_dfs.append(prediction_df)
        return pd.concat(pred_dfs)

    def score(self, description: str = "") -> pd.DataFrame:
        """Score the piece-wise prediction against the actual timecourse.

        Parameters
        ----------
        description : str

        Returns
        -------
        pd.DataFrame
            One row per aggregation level (model + one per species).
        """
        prediction_df = self.predict()
        actual_df = self.trajectory_collection.makeTimecourse()
        scorer = Score()
        score_infos = scorer.makeScoreInfo(description, actual_df, prediction_df)
        return pd.DataFrame([info.__dict__ for info in score_infos])

    def plotPrediction(self, **kwargs) -> PlotOptions:
        """Plot actual and predicted timecourses with segment boundary lines.

        Actual values are solid lines; predictions are dashed.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions.

        Returns
        -------
        PlotOptions
        """
        prediction_df = self.predict()
        actual_df = self.trajectory_collection.makeTimecourse()
        if "title" not in kwargs:
            scorer = Score()
            score_infos = scorer.makeScoreInfo("", actual_df, prediction_df)
            p95 = score_infos[0].p95
            model_name = self.trajectory_collection.model.model_name
            n_seg = len(self.trajectory_collection.trajectories)
            kwargs["title"] = f"{model_name} n_seg={n_seg}, p95={p95:.2f}"
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        for i, name in enumerate(self.trajectory_collection.model.species_names):
            color = f"C{i}"
            ax.plot(actual_df.index, actual_df[name],  # type: ignore
                    color=color, label=f"{name} (actual)")
            ax.plot(prediction_df.index, prediction_df[name],  # type: ignore
                    color=color, linestyle="--", label=f"{name} (predicted)")
        for traj in self.trajectory_collection.trajectories[:-1]:
            ax.axvline(x=traj.end_time, color="black",  # type: ignore
                    linestyle="--", linewidth=0.8)
        plot_options.apply()
        return plot_options

    @property
    def cost(self) -> float:
        """Mean squared relative prediction error, median over species.

        First timepoint is skipped (exact by construction).

        Returns
        -------
        float
        """
        actual_arr = self.trajectory_collection.makeTimecourse().values[1:]
        prediction_arr = self.predict().values[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_arr = np.where(
                    actual_arr == 0,
                    np.nan,
                    (prediction_arr - actual_arr) / actual_arr,
            )
        species_costs = np.nanmean(rel_arr ** 2, axis=0)
        return float(np.nanmedian(species_costs))
