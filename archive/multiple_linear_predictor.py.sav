"""Piece-wise linear prediction using a TrajectoryCollection."""

import src.constants as cn  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.score import Score  # type: ignore
from src.trajectory_collection import TrajectoryCollection  # type: ignore
from src.slow_subspace_predictor import SlowSubspacePredictor  # type: ignore

from collections import namedtuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List, Union


SegmentResult = namedtuple("SegmentResult", ["actual_df", "prediction_df"])
PredictionResult = namedtuple("PredictionResult", ["actual_df", "prediction_df", "segment_results"])


class MultipleLinearPredictor(object):
    """Piece-wise linear predictor across a TrajectoryCollection.

    Each Trajectory is predicted independently with LinearPredictor; results
    are concatenated into a single timecourse.  Boundary timepoints shared
    between adjacent segments appear once in the output (the duplicate first
    row of each interior segment is dropped).
    """

    def __init__(self,
            trajectory_collection: TrajectoryCollection,
            jacobian_selection: str = cn.JAC_FITTED,
            num_step: int = -1,
            is_slow_subspace: bool = False,
            is_species_prediction: bool = False) -> None:
        """
        Parameters
        ----------
        trajectory_collection : TrajectoryCollection
        jacobian_selection : str
            Passed to each LinearPredictor.
        num_step : int
            Number of steps ahead for windowed prediction.
        is_species_prediction : bool
            Whether to predict species concentrations.
        """
        self.trajectory_collection = trajectory_collection
        self.jacobian_selection = jacobian_selection
        self.num_step = num_step
        self.is_slow_subspace = is_slow_subspace
        self.is_species_prediction = is_species_prediction

    def _makeTimecourse(self) -> pd.DataFrame:
        """Concatenate actual timecourses, dropping duplicate boundary rows.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with one row per unique timepoint.
        """
        dfs = []
        for i, traj in enumerate(self.trajectory_collection.trajectories):
            tc_df = traj.timecourse_df
            if i > 0:
                tc_df = tc_df.iloc[1:]
            dfs.append(tc_df)
        return pd.concat(dfs)

    def predict(self) -> PredictionResult:
        """Predict concentrations across all segments.

        Each segment is predicted by an independent LinearPredictor.  Duplicate
        boundary rows are dropped to produce a single time-indexed DataFrame.

        Returns
        -------
        PredictionResult
            Result containing actual and predicted timecourses along with segment results.
        """
        segment_results: List[SegmentResult] = []
        pred_dfs: List[pd.DataFrame] = []
        actual_dfs: List[pd.DataFrame] = []
        for i, traj in enumerate(self.trajectory_collection.trajectories):
            if self.is_slow_subspace:
                lp: Union[SlowSubspacePredictor, LinearPredictor] = SlowSubspacePredictor(traj,
                        num_step=self.num_step)
            else:
                lp = LinearPredictor(traj,
                        jacobian_selection=self.jacobian_selection,
                        num_step=self.num_step,
                        is_species_prediction=self.is_species_prediction)
            prediction_df = lp.predict()
            if i > 0:
                prediction_df = prediction_df.iloc[1:]
                segment_result = SegmentResult(prediction_df=prediction_df,
                        actual_df=traj.timecourse_df.iloc[1:])
            else:
                segment_result = SegmentResult(prediction_df=prediction_df,
                        actual_df=traj.timecourse_df)
            segment_results.append(segment_result)
            pred_dfs.append(segment_result.prediction_df)
            actual_dfs.append(segment_result.actual_df)
        pred_all_df = pd.concat(pred_dfs)
        actual_all_df = pd.concat(actual_dfs)
        # FIXME: get the actual timecourse from the TrajectoryCollection instead of re-constructing it here
        prediction_result = PredictionResult(prediction_df=pred_all_df,
                actual_df=actual_all_df, segment_results=segment_results)
        return prediction_result

    def score(self, description: str = "") -> pd.DataFrame:
        """Score the piece-wise prediction against the actual timecourse.

        Parameters
        ----------
        description : str
            Label stored in the 'description' column of the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per aggregation level (model + one per species).
        """
        prediction_df = self.predict().prediction_df
        actual_df = self.trajectory_collection.makeTimecourse()
        scorer = Score()
        score_infos = scorer.makeScoreInfo(description, actual_df, prediction_df)
        return pd.DataFrame([info.__dict__ for info in score_infos])

    def plotPrediction(self, **kwargs) -> PlotOptions:
        """Plot actual and predicted timecourses with segment boundary lines.

        Actual values are solid lines; predictions are dashed.  Vertical dashed
        lines mark boundaries between adjacent segments.

        Parameters
        ----------
        **kwargs
            Passed to PlotOptions.

        Returns
        -------
        PlotOptions
        """
        prediction_result = self.predict()
        prediction_df = prediction_result.prediction_df
        actual_df = self.trajectory_collection.makeTimecourse()
        if "title" not in kwargs:
            scorer = Score()
            score_infos = scorer.makeScoreInfo("", actual_df, prediction_df)
            p95 = score_infos[0].p95
            model_name = self.trajectory_collection.model.model_name
            n_seg = len(self.trajectory_collection.trajectories)
            kwargs["title"] = f"{model_name} n_seg={n_seg}, p95={p95:.2f}"
        # Get the score information for each segement
        segment_score_p95s:list = []
        scorer = Score()
        for i, segment_result in enumerate(prediction_result.segment_results):
            try:
                score_info = scorer.makeScoreInfo(f"Segment {i}",
                        segment_result.actual_df, segment_result.prediction_df)
                p95 = score_info[0].p95
            except Exception as e:
                print(f"Error occurred while processing segment {i}: {e}")
                p95 = np.nan
            segment_score_p95s.append(p95)
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
        for i, traj in enumerate(self.trajectory_collection.trajectories):
            xpos = traj.start_time + 0.4 * (traj.end_time - traj.start_time)
            ax.text(xpos, ax.get_ylim()[1], f"{segment_score_p95s[i]:.2f}",  # type: ignore
                    ha="center", va="top", fontsize=8)
        plot_options.apply()
        return plot_options

    @property
    def cost(self) -> float:
        """Mean squared relative prediction error, aggregated across species.

        Computed on the concatenated timecourse with the first row dropped.
        Returns the median of per-species mean squared relative errors.

        Returns
        -------
        float
        """
        actual_arr = self.trajectory_collection.makeTimecourse().values[1:]
        prediction_arr = self.predict().prediction_df.values[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_arr = np.where(
                    actual_arr == 0,
                    np.nan,
                    (prediction_arr - actual_arr) / actual_arr,
            )
        species_costs = np.nanmean(rel_arr ** 2, axis=0)
        return float(np.nanmedian(species_costs))
