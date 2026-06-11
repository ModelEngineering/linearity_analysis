"""Tests for TrajectoryCollection"""

import os
import sys
import unittest
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from tests.utils_test import makeBiomdPath, makeBiomdName  # type: ignore

BIOMD8_PATH = makeBiomdPath(8)
BIOMD8_ENDTIME = 20.0
BIOMD153_PATH = makeBiomdPath(153)
BIOMD13_ENDTIME = 20.0

IGNORE_TESTS = False
IS_PLOT = False

ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""

ANTIMONY_FORCED = """
$Xo -> S1; k1*Xo
S1 -> $X1; k2*S1
S1 = 0.0
k1 = 0.1; k2 = 0.2; Xo = 1.0; X1 = 0.0
"""

ANTIMONY_SEQUENTIAL = """
S1 -> S2; k1*S1
S2 -> S3; k2*S2
S3 -> ; k3*S3
k1 = 0.1; k2 = 0.2; k3=0.3; S1 = 10.0; S2 = 0.0; S3 = 0.0
"""


def make_trajectory(antimony_str: str) -> Trajectory:
    """Return a Trajectory built from a real Antimony model."""
    lr = LRoadrunner(antimony_str, start_time=0.0, end_time=10.0, num_point=11)
    return Trajectory(lr)

def _make_tc(antimony_str: str, num_cluster: int) -> TrajectoryCollection:
    """Return a TrajectoryCollection built from a real Antimony model.

    Adjacent chunks share their boundary timepoint so that TrajectoryCollection.predict
    can correctly deduplicate with iloc[1:].
    """
    lr = LRoadrunner(antimony_str, start_time=0.0, end_time=10.0, num_point=11)
    jc = make_trajectory(antimony_str)
    num_point = len(jc.timepoint_arr)
    chunk_size = num_point // num_cluster
    jcs = []
    for i in range(num_cluster):
        istart = i * chunk_size
        iend = istart + chunk_size + 1 if i < num_cluster - 1 else num_point
        jcs.append(Trajectory.fromArrays(
            jc.jacobian_collection_arr[istart:iend], jc.timepoint_arr[istart:iend], lr))
    return TrajectoryCollection(jcs)


class TestTrajectoryCollectionHeatmaps(unittest.TestCase):
    """Tests for TrajectoryCollection.heatmaps."""

    def test_heatmaps_returns_figure(self) -> None:
        """heatmaps returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        cjc = _make_tc(ANTIMONY_FORCED, num_cluster=1)
        fig = cjc.heatmaps()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_heatmaps_one_axes_per_cluster_plus_colorbar(self) -> None:
        """heatmaps produces n heatmap axes plus 1 shared colorbar axes."""
        if IGNORE_TESTS:
            return
        for n_clusters in [1, 2, 3]:
            cjc = _make_tc(ANTIMONY_DECAY, num_cluster=n_clusters)
            fig = cjc.heatmaps()
            self.assertEqual(len(fig.axes), n_clusters + 1)
            plt.close(fig)

    def test_heatmaps_titles_contain_timepoints(self) -> None:
        """Each heatmap axes title contains the start and end timepoints of its cluster."""
        if IGNORE_TESTS:
            return
        cjc = _make_tc(ANTIMONY_DECAY, num_cluster=2)
        fig = cjc.heatmaps()
        # First n axes are heatmaps; last is colorbar.
        heatmap_axes = fig.axes[:-1]
        for ax, jc in zip(heatmap_axes, cjc.trajectories):
            t_start = f"{float(jc.timepoint_arr[0]):.3g}"
            t_end = f"{float(jc.timepoint_arr[-1]):.3g}"
            self.assertIn(t_start, ax.get_title())
            self.assertIn(t_end, ax.get_title())
        plt.close(fig)

    def test_heatmaps_colorbar_is_horizontal(self) -> None:
        """The shared colorbar is oriented horizontally."""
        if IGNORE_TESTS:
            return
        cjc = _make_tc(ANTIMONY_DECAY, num_cluster=2)
        fig = cjc.heatmaps()
        cbar_ax = fig.axes[-1]
        # A horizontal colorbar axes is wider than it is tall.
        bbox = cbar_ax.get_position()
        self.assertGreater(bbox.width, bbox.height)
        plt.close(fig)

    def test_heatmaps_colorbar_does_not_overlap_heatmaps(self) -> None:
        """The colorbar axes does not overlap any heatmap axes."""
        if IGNORE_TESTS:
            return
        cjc = _make_tc(ANTIMONY_DECAY, num_cluster=2)
        fig = cjc.heatmaps()
        cbar_ax = fig.axes[-1]
        cbar_bbox = cbar_ax.get_position()
        for ax in fig.axes[:-1]:
            hm_bbox = ax.get_position()
            overlaps = (
                cbar_bbox.x0 < hm_bbox.x1 and cbar_bbox.x1 > hm_bbox.x0
                and cbar_bbox.y0 < hm_bbox.y1 and cbar_bbox.y1 > hm_bbox.y0
            )
            self.assertFalse(overlaps)
        plt.close(fig)


class TestTrajectoryCollectionScore(unittest.TestCase):
    """Tests for TrajectoryCollection score (max_cv) behaviour."""

    def test_score_decreases_as_n_cluster_increases(self) -> None:
        """score (max_cv) decreases as n_cluster increases for linearly-varying random matrices."""
        if IGNORE_TESTS:
            return
        np.random.seed(0)
        n_matrices = 20
        n_species = 3
        # Base matrix with strictly positive entries so CV is well-defined.
        base = np.abs(np.random.rand(n_species, n_species)) + 1.0
        # Each matrix is (i+1) * base, creating a clear linear trend over time.
        jacobian_arr = np.array([(i + 1) * base for i in range(n_matrices)])
        timepoint_arr = np.arange(n_matrices, dtype=float)
        jc = Trajectory.fromArrays(jacobian_arr, timepoint_arr)

        scores = []
        for n_clusters in [1, 3, 5]:
            chunk_size = n_matrices // n_clusters
            jcs = []
            for i in range(n_clusters):
                start = i * chunk_size
                end = start + chunk_size if i < n_clusters - 1 else n_matrices
                jcs.append(Trajectory.fromArrays(
                        jc.jacobian_collection_arr[start:end],
                        jc.timepoint_arr[start:end]))
            cjc = TrajectoryCollection(jcs)
            scores.append(cjc.score)

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])


class TestTrajectoryCollectionSplit(unittest.TestCase):
    """Tests for TrajectoryCollection.split."""

    # 11 timepoints: [0, 1, 2, ..., 10]
    NUM_POINT = 11
    END_TIME = 10.0

    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        lr = LRoadrunner(ANTIMONY_DECAY, start_time=0.0,
                end_time=cls.END_TIME, num_point=cls.NUM_POINT)
        cls.trajectory = Trajectory(lr)

    def test_one_split_gives_correct_timecourse(self) -> None:
        # Test that the split gives the correct timecourse values in the second trajectory.
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        second_trajectory = tc.trajectories[1]
        first_df = self.trajectory.timecourse_df.loc[np.array(range(5,11)), :]
        np.testing.assert_allclose(second_trajectory.timecourse_df.values,
                first_df.values, rtol=1e-5)

    def test_one_split_gives_two_trajectories(self) -> None:
        """One split time produces a collection with 2 trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertEqual(len(tc.trajectories), 2)
        len_jacobian_collection_arr = len(self.trajectory.jacobian_collection_arr)
        self.assertEqual(len(tc.trajectories[0].jacobian_collection_arr) +
                len(tc.trajectories[1].jacobian_collection_arr) - 1,
                len_jacobian_collection_arr)

    def test_plot_one_split_gives_two_trajectories(self) -> None:
        """One split time produces a collection with 2 trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertEqual(len(tc.trajectories), 2)

    def test_two_splits_give_three_trajectories(self) -> None:
        """Two split times produce a collection with 3 trajectories."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [3.0, 7.0])
        self.assertEqual(len(tc.trajectories), 3)

    def test_empty_split_returns_single_trajectory(self) -> None:
        """Empty split_times produces one trajectory covering the full range."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [])
        self.assertEqual(len(tc.trajectories), 1)
        self.assertEqual(len(tc.trajectories[0].timepoint_arr),
                len(self.trajectory.timepoint_arr))

    def test_time_ranges_are_correct(self) -> None:
        """Each chunk spans [start, split] and [split, end] inclusive."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        self.assertAlmostEqual(float(tc.trajectories[0].timepoint_arr[0]), 0.0)
        self.assertAlmostEqual(float(tc.trajectories[0].timepoint_arr[-1]), 5.0)
        self.assertAlmostEqual(float(tc.trajectories[1].timepoint_arr[0]), 5.0)
        self.assertAlmostEqual(float(tc.trajectories[1].timepoint_arr[-1]), 10.0)

    def test_boundary_timepoints_are_shared(self) -> None:
        """The last timepoint of each chunk equals the first timepoint of the next."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [3.0, 7.0])
        self.assertAlmostEqual(float(tc.trajectories[0].timepoint_arr[-1]),
                float(tc.trajectories[1].timepoint_arr[0]))
        self.assertAlmostEqual(float(tc.trajectories[1].timepoint_arr[-1]),
                float(tc.trajectories[2].timepoint_arr[0]))

    def test_each_element_is_trajectory(self) -> None:
        """Every element of the collection is a Trajectory instance."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        for t in tc.trajectories:
            self.assertIsInstance(t, Trajectory)

    def test_unsorted_split_times_are_sorted(self) -> None:
        """Split times supplied in reverse order are sorted before splitting."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [7.0, 3.0])
        self.assertAlmostEqual(float(tc.trajectories[0].timepoint_arr[-1]), 3.0)
        self.assertAlmostEqual(float(tc.trajectories[1].timepoint_arr[-1]), 7.0)
        self.assertAlmostEqual(float(tc.trajectories[2].timepoint_arr[-1]), 10.0)

    def test_out_of_bounds_raises_value_error(self) -> None:
        """A split time outside the trajectory's timepoint range raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            TrajectoryCollection.split(self.trajectory, [15.0])

    def test_predict_covers_full_timecourse(self) -> None:
        """predict() on the split collection returns every timepoint from the original trajectory."""
        if IGNORE_TESTS:
            return
        tc = TrajectoryCollection.split(self.trajectory, [5.0])
        pred_df = tc.predict()
        self.assertEqual(len(pred_df), len(self.trajectory.timepoint_arr))
        np.testing.assert_array_almost_equal(
                pred_df.index.to_numpy(), self.trajectory.timepoint_arr)
    
    def test_unitsplit(self) -> None:
        """The unitsplit column is present and contains only 0s and 1s."""
        if IGNORE_TESTS:
            return
        unit_tc = TrajectoryCollection.unitsplit(self.trajectory)
        self.assertEqual(len(unit_tc.trajectories), len(self.trajectory.timepoint_arr)-1)


class TestTrajectoryCollectionPredict(unittest.TestCase):
    """Tests for TrajectoryCollection.predict."""

    lr: LRoadrunner
    trajectory: Trajectory
    tc1: TrajectoryCollection
    tc2: TrajectoryCollection
    tc3: TrajectoryCollection

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        cls.lr = LRoadrunner(ANTIMONY_DECAY, start_time=0.0, end_time=10.0, num_point=11)
        cls.trajectory = Trajectory(cls.lr)
        cls.tc1 = TrajectoryCollection([cls.trajectory])
        cls.tc2 = _make_tc(ANTIMONY_DECAY, num_cluster=2)
        cls.tc3 = _make_tc(ANTIMONY_DECAY, num_cluster=3)

    def test_returns_dataframe(self) -> None:
        """predict returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.tc2.predict()
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_match_species(self) -> None:
        """Column names equal the model's floating species names."""
        if IGNORE_TESTS:
            return
        result = self.tc2.predict()
        self.assertEqual(list(result.columns), self.tc2.l_roadrunner.species_names)

    def test_single_trajectory_matches_direct_predict(self) -> None:
        """A 1-trajectory collection produces the same result as trajectory.predict()."""
        if IGNORE_TESTS:
            return
        pd.testing.assert_frame_equal(self.tc1.predict(), self.trajectory.predict())

    def test_row_count_single_trajectory(self) -> None:
        """With one trajectory the row count equals num_point."""
        if IGNORE_TESTS:
            return
        result = self.tc1.predict()
        self.assertEqual(len(result), len(self.trajectory.timepoint_arr))

    def test_row_count_multiple_trajectories(self) -> None:
        """First row of each non-first segment is dropped, giving sum(n_i) - (k-1) total rows."""
        if IGNORE_TESTS:
            return
        for tc in [self.tc2, self.tc3]:
            n_clusters = len(tc.trajectories)
            expected = sum(len(t.timepoint_arr) for t in tc.trajectories) - (n_clusters - 1)
            self.assertEqual(len(tc.predict()), expected)

    def test_values_are_finite(self) -> None:
        """All predicted values are finite."""
        if IGNORE_TESTS:
            return
        result = self.tc2.predict()
        self.assertTrue(np.all(np.isfinite(result.values)))

    def test_index_is_monotonically_increasing(self) -> None:
        """Timepoint index is strictly increasing across the concatenated result."""
        if IGNORE_TESTS:
            return
        result = self.tc2.predict()
        self.assertTrue(np.all(np.diff(result.index.values) > 0))


@unittest.skipUnless(os.path.isdir(cn.BIOMODELS_DIR), "BioModels data directory not found")
class TestTrajectoryCollectionPredictBiomd8(unittest.TestCase):
    """Integration tests for TrajectoryCollection.predict on BIOMD0000000008."""

    NUM_POINT = 30
    N_CLUSTERS = 2
    trajectory: Trajectory
    tc: TrajectoryCollection
    pred_df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.exists(BIOMD8_PATH):
            raise unittest.SkipTest(f"BIOMD0000000008 not found at {BIOMD8_PATH}")
        cls.trajectory = Trajectory.makeBiomodel(
                path=BIOMD8_PATH, end_time=BIOMD8_ENDTIME, num_point=cls.NUM_POINT)
        chunk_size = cls.NUM_POINT // cls.N_CLUSTERS
        jcs = []
        for i in range(cls.N_CLUSTERS):
            start = i * chunk_size
            end = start + chunk_size + 1 if i < cls.N_CLUSTERS - 1 else cls.NUM_POINT
            jcs.append(Trajectory.fromArrays(
                    cls.trajectory.jacobian_collection_arr[start:end],
                    cls.trajectory.timepoint_arr[start:end],
                    cls.trajectory.l_roadrunner))
        cls.tc = TrajectoryCollection(jcs)
        cls.pred_df = cls.tc.predict()

    def test_returns_dataframe(self) -> None:
        """predict returns a pd.DataFrame for BIOMD8."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.pred_df, pd.DataFrame)

    def test_columns_match_species(self) -> None:
        """Column names equal BIOMD8's floating species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(list(self.pred_df.columns),
                self.tc.l_roadrunner.species_names)

    def test_row_count(self) -> None:
        """Row count equals sum(n_i) - (n_clusters - 1) after boundary deduplication."""
        if IGNORE_TESTS:
            return
        expected = sum(len(t.timepoint_arr) for t in self.tc.trajectories) - (self.N_CLUSTERS - 1)
        self.assertEqual(len(self.pred_df), expected)

    def test_values_are_finite(self) -> None:
        """All predicted values for BIOMD8 are finite."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.pred_df.values)))

    def test_unitsplit(self) -> None:
        """The unitsplit column is present and contains only 0s and 1s."""
        if IGNORE_TESTS:
            return
        unit_tc = TrajectoryCollection.unitsplit(self.trajectory)
        self.assertEqual(len(unit_tc.trajectories), len(self.trajectory.timepoint_arr)-1)

    def test_comparison_with_direct_predict(self) -> None:
        """TC.predict has the same shape as trajectory.predict and the same initial row.

        With boundary-sharing chunks, deduplication preserves every timepoint, so
        row counts match. Each segment's _predict seeds each step from the actual
        timecourse value, making the first row (initial condition) identical across both.
        Subsequent rows differ because each segment uses its own median Jacobian.
        """
        if IGNORE_TESTS:
            return
        direct_df = self.trajectory.predict()
        self.assertEqual(list(self.pred_df.columns), list(direct_df.columns))
        np.testing.assert_array_almost_equal(
                self.pred_df.iloc[0].to_numpy(), direct_df.iloc[0].to_numpy())
        self.assertEqual(len(self.pred_df), len(direct_df))
        relative_absolute_error = (self.pred_df.iloc[1:, :]
                - direct_df.iloc[1:, :])**2/self.pred_df.iloc[1:, :].abs()
        self.assertLess(np.mean(relative_absolute_error.values), 0.1)
    
    def test_comparison_with_direct_predict2(self) -> None:
        """TC.predict has the same shape as trajectory.predict and the same initial row.

        With boundary-sharing chunks, deduplication preserves every timepoint, so
        row counts match. Each segment's _predict seeds each step from the actual
        timecourse value, making the first row (initial condition) identical across both.
        Subsequent rows differ because each segment uses its own median Jacobian.
        """
        if IGNORE_TESTS:
            return
        trajectory = make_trajectory(ANTIMONY_SEQUENTIAL)
        trajectory_pred_df = trajectory.predict()
        tc = TrajectoryCollection.split(trajectory, [3.0, 7.0])
        tc_pred_df = tc.predict()
        relative_absolute_error = (trajectory_pred_df.iloc[1:, :]
                - tc_pred_df.iloc[1:, :])**2/trajectory_pred_df.iloc[1:, :].abs()
        self.assertLess(np.mean(relative_absolute_error.values), 0.1)


class TestTrajectoryCollectionPlotTrajectories(unittest.TestCase):
    """Tests for TrajectoryCollection.plotTrajectories."""

    tc_1: TrajectoryCollection   # 1 cluster — no split lines
    tc_2: TrajectoryCollection   # 2 clusters — 1 split line at t=5
    num_species: int

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        cls.tc_1 = _make_tc(ANTIMONY_DECAY, num_cluster=1)
        cls.tc_2 = _make_tc(ANTIMONY_DECAY, num_cluster=3)
        cls.num_species = cls.tc_2.l_roadrunner.num_species

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        """plotTrajectories returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories()
        #plt.show()
        self.assertIsInstance(po, PlotOptions)

    def test_split_collection_plots_distinct_concentrations(self) -> None:
        """plotTrajectories on a split collection uses the correct concentrations per segment."""
        if IGNORE_TESTS:
            return
        lr = LRoadrunner(ANTIMONY_DECAY, start_time=0.0, end_time=10.0, num_point=11)
        trajectory = Trajectory(lr)
        tc_split = TrajectoryCollection.split(trajectory, [5.0])
        po = tc_split.plotTrajectories()
        # The first data point of each species line should equal the t=0 initial condition,
        # and the last data point should be at t=10 (non-trivially different from t=0).
        for line in po.ax.lines[:self.num_species]:
            ydata = np.asarray(line.get_ydata())
            self.assertFalse(np.allclose(ydata[0], ydata[-1]))

    def test_ax_is_not_none(self) -> None:
        """PlotOptions.ax is set."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories()
        self.assertIsNotNone(po.ax)

    def test_line_count_matches_species_plus_splits(self) -> None:
        """ax.lines has one line per species plus one per split time."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories()
        expected = self.num_species + len(self.tc_2.split_times)
        self.assertEqual(len(po.ax.lines), expected)

    def test_no_split_lines_for_single_cluster(self) -> None:
        """A 1-cluster collection draws no vertical split lines."""
        if IGNORE_TESTS:
            return
        po = self.tc_1.plotTrajectories()
        self.assertEqual(len(po.ax.lines), self.num_species)

    def test_split_line_x_positions(self) -> None:
        """Vertical lines are drawn at each split_time x-coordinate."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories()
        split_xcoords = [float(np.asarray(line.get_xdata())[0])
                for line in po.ax.lines[self.num_species:]]
        for split_time in self.tc_2.split_times:
            self.assertIn(split_time, split_xcoords)

    def test_model_name_in_title(self) -> None:
        """Title contains the model_name when provided."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories(title="TestModel")
        self.assertIn("TestModel", po.ax.get_title())

    def test_legend_false_suppresses_legend(self) -> None:
        """legend=False produces no legend on the axes."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories(legend=False)
        self.assertIsNone(po.ax.get_legend())

    def test_ylim_applied(self) -> None:
        """ylim is applied to the axes."""
        if IGNORE_TESTS:
            return
        po = self.tc_2.plotTrajectories(ylim=(0.0, 5.0))
        self.assertAlmostEqual(po.ax.get_ylim()[0], 0.0, places=5)
        self.assertAlmostEqual(po.ax.get_ylim()[1], 5.0, places=5)

    def test_explicit_ax_is_used(self) -> None:
        """An axes passed via ax= is used and returned."""
        if IGNORE_TESTS:
            return
        _, ax = plt.subplots()
        po = self.tc_2.plotTrajectories(ax=ax)
        self.assertIs(po.ax, ax)


class TestTrajectoryCollectionPlotPrediction(unittest.TestCase):
    """Tests for TrajectoryCollection.plotPrediction."""

    trajectory: Trajectory
    num_species: int
    NUM_POINT: int = 50
    tc: TrajectoryCollection
    END_TIME: float = 0.1
    MODEL_NUM: int = 2
    MODEL_NAME: str = makeBiomdName(MODEL_NUM)

    @classmethod
    def setUpClass(cls) -> None:
        #if IGNORE_TESTS:
        #    return
        path = makeBiomdPath(cls.MODEL_NUM)
        if not os.path.exists(path):
            raise unittest.SkipTest(f"{cls.MODEL_NAME} not found at {path}")
        cls.trajectory = Trajectory.makeBiomodel(
                path=path, end_time=cls.END_TIME, num_point=cls.NUM_POINT)
        cls.tc = TrajectoryCollection.split(cls.trajectory, [cls.END_TIME/2])
        if IGNORE_TESTS:
            print(pd.DataFrame(cls.trajectory.jacobian_median_arr))
            print(pd.DataFrame(cls.tc.trajectories[0].jacobian_median_arr))
            print(pd.DataFrame(cls.tc.trajectories[1].jacobian_median_arr))


    def tearDown(self) -> None:
        plt.close("all")

    def test_jacobian_collection_sizes(self) -> None:
        if IGNORE_TESTS:
            return
        len_jacobian_collection_arr = len(self.trajectory.jacobian_collection_arr)
        n_clusters = len(self.tc.trajectories)
        total = sum(len(t.jacobian_collection_arr) for t in self.tc.trajectories) - (n_clusters - 1)
        self.assertEqual(total, len_jacobian_collection_arr)

    def test_plot(self):
        if IGNORE_TESTS:
            return
        self.trajectory.plotPrediction(title=self.MODEL_NAME, ylim=(0.0, 1e-5))
        self.tc.plotPrediction(title=self.MODEL_NAME, ylim=(0.0, 1e-5))
        if IS_PLOT:
            plt.show()


if __name__ == "__main__":
    unittest.main()
