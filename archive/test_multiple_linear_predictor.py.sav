"""Tests for MultipleLinearPredictor class."""

import os
import sys
import unittest
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tellurium as te  # type: ignore
from unittest.mock import MagicMock
from typing import Tuple

import src.constants as cn  # type: ignore
from trajectory import Trajectory  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from src.l_roadrunner import LRoadrunner  # type: ignore
from src.multiple_linear_predictor import MultipleLinearPredictor, ScoreResult  # type: ignore
from src.biomodels_cluster import BiomodelsCluster  # type: ignore

IGNORE_TESTS = True

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)

# Simple Antimony model: first-order decay with boundary-species forcing.
ANTIMONY_FORCED = """
$Xo -> S1; k1*Xo
S1 -> $X1; k2*S1
S1 = 0.0
k1 = 0.1; k2 = 0.2; Xo = 1.0; X1 = 0.0
"""

# Simple 2-species decay model.
ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""

# Nonlinear model: Michaelis-Menten kinetics gives a time-varying Jacobian.
ANTIMONY_MM = """
$Xo -> S1; k1*Xo
S1 -> $X1; Vmax*S1/(Km + S1)
S1 = 0.1
k1 = 1.0; Vmax = 2.0; Km = 0.5; Xo = 1.0; X1 = 0.0
"""


# Simple 2-species decay model.
ANTIMONY_DECAY_FORCED = """
 -> S1; k0
S1 -> S2; k1*S1
S2 -> S1; k2*S2*S1
S2 -> ; k3*S2
k0=1; k1 = 0.1; k2 = 0.002; k3 = 0.05; S1 = 0; S2 = 0.0
"""
""" rr = te.loada(ANTIMONY_DECAY_FORCED)
data = rr.simulate(0, 5, 100)
rr.plot() """


def _make_lroadrunner(antimony_str: str, end_time:float = 10.0) -> LRoadrunner:
    """Return a real LRoadrunner for the given Antimony model."""
    return LRoadrunner(antimony_str, start_time=0.0, end_time=end_time, num_point=11)


def _make_predictor_from_model(antimony_str: str,
        n_cluster: int, end_time:float = 10.0) -> MultipleLinearPredictor:
    """Return a MultipleLinearPredictor built from a real Antimony model."""
    lr = _make_lroadrunner(antimony_str)
    jc = Trajectory(lr)
    n_points = len(jc.timepoint_arr)
    chunk_size = n_points // n_cluster
    jcs = []
    for i in range(n_cluster):
        start = i * chunk_size
        end = start + chunk_size if i < n_cluster - 1 else n_points
        jcs.append(Trajectory.fromArrays(
            jc.jacobian_collection_arr[start:end], jc.timepoint_arr[start:end], lr))
    cjc = TrajectoryCollection(jcs)
    return MultipleLinearPredictor.makeFromLRoadrunner(cjc, lr)


class TestMultipleLinearPredictorInit(unittest.TestCase):
    """Tests for MultipleLinearPredictor.__init__."""

    def _make_cjc_and_arrays(self):
        """Return (cjc, lr, initial_value_arr, forced_input_arr) with mock LRoadrunner."""
        lr = MagicMock(spec=LRoadrunner)
        lr.makeJacobians.return_value = (
            np.full((3, 1, 1), -0.2), np.linspace(0.0, 10.0, 3)
        )
        jc = Trajectory(lr)
        cjc = TrajectoryCollection([jc])
        initial_value_arr = np.array([1.0])
        forced_input_arr = np.array([0.5])
        return cjc, lr, initial_value_arr, forced_input_arr

    def test_stores_clustered_jacobian_collection(self) -> None:
        """clustered_jacobian_collection attribute is stored."""
        if IGNORE_TESTS:
            return
        cjc, _, initial_value_arr, forced_input_arr = self._make_cjc_and_arrays()
        predictor = MultipleLinearPredictor(cjc, initial_value_arr, forced_input_arr)
        self.assertIs(predictor.clustered_jacobian_collection, cjc)

    def test_stores_initial_value_arr(self) -> None:
        """initial_value_arr attribute is stored."""
        if IGNORE_TESTS:
            return
        cjc, _, initial_value_arr, forced_input_arr = self._make_cjc_and_arrays()
        predictor = MultipleLinearPredictor(cjc, initial_value_arr, forced_input_arr)
        np.testing.assert_array_equal(predictor.initial_value_arr, initial_value_arr)

    def test_stores_forced_input_arr(self) -> None:
        """forced_input_arr is a 1-D array of shape (n_species,) and is stored."""
        if IGNORE_TESTS:
            return
        cjc, _, initial_value_arr, forced_input_arr = self._make_cjc_and_arrays()
        predictor = MultipleLinearPredictor(cjc, initial_value_arr, forced_input_arr)
        np.testing.assert_array_equal(predictor.forced_input_arr, forced_input_arr)
        self.assertEqual(predictor.forced_input_arr.ndim, 1)

    def test_stores_l_roadrunner(self) -> None:
        """l_roadrunner attribute is stored when provided."""
        if IGNORE_TESTS:
            return
        cjc, lr, initial_value_arr, forced_input_arr = self._make_cjc_and_arrays()
        predictor = MultipleLinearPredictor(
                cjc, initial_value_arr, forced_input_arr, l_roadrunner=lr)
        self.assertIs(predictor.l_roadrunner, lr)


class TestMultipleLinearPredictorPredict(unittest.TestCase):
    """Tests for MultipleLinearPredictor.predict."""

    def test_predict_shape_single_cluster(self) -> None:
        """predict returns shape (1, n_species) for a single-cluster collection."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        result = predictor.predict()
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result.shape[1], 0)

    def test_predict_shape_multiple_clusters(self) -> None:
        """predict returns shape (n_clusters, n_species)."""
        if IGNORE_TESTS:
            return
        n_clusters = 3
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_cluster=n_clusters)
        result = predictor.predict()
        self.assertEqual(result.shape[0], n_clusters)

    def test_predict_finite_values(self) -> None:
        """predict returns finite concentration values."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=2)
        result = predictor.predict()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_predict_non_negative_concentrations(self) -> None:
        """predict returns non-negative concentrations for a decay model starting at 0."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=2)
        result = predictor.predict()
        self.assertTrue(np.all(result >= -1e-6))

    def test_predict_approaches_steady_state(self) -> None:
        """For ANTIMONY_FORCED, predictions should increase toward steady state 0.5."""
        if IGNORE_TESTS:
            return
        n_clusters = 5
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=n_clusters)
        result = predictor.predict()
        self.assertTrue(result[-1, 0] > result[0, 0])
        self.assertLess(result[-1, 0], 0.6)


class TestMultipleLinearPredictorPlot(unittest.TestCase):
    """Tests for MultipleLinearPredictor.plot."""

    def test_plot_returns_figure(self) -> None:
        """plot returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_plot_creates_axes_when_none(self) -> None:
        """plot creates a new figure with one axes when ax is not supplied."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        fig = predictor.plot(ax=None)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_plot_uses_provided_axes(self) -> None:
        """plot draws on the supplied axes and returns its parent figure."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        external_fig, external_ax = plt.subplots()
        returned_fig = predictor.plot(ax=external_ax)
        self.assertIs(returned_fig, external_fig)
        plt.close(external_fig)

    def test_plot_no_vertical_lines_single_cluster(self) -> None:
        """With one cluster there are no vertical boundary lines."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        fig = predictor.plot()
        ax = fig.axes[0]
        def _is_vline(l) -> bool:  # type: ignore[no-untyped-def]
            xd: np.ndarray = np.asarray(l.get_xdata())
            return len(xd) >= 2 and bool(np.isclose(xd[0], xd[-1]))
        vlines = [l for l in ax.lines if _is_vline(l)]
        self.assertEqual(len(vlines), 0)
        plt.close(fig)

    def test_plot_vertical_lines_for_multiple_clusters(self) -> None:
        """With n clusters there are n-1 vertical boundary lines."""
        if IGNORE_TESTS:
            return
        n_clusters = 3
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_cluster=n_clusters)
        fig = predictor.plot()
        ax = fig.axes[0]
        def _is_vline(l) -> bool:  # type: ignore[no-untyped-def]
            xd: np.ndarray = np.asarray(l.get_xdata())
            return len(xd) >= 2 and bool(np.isclose(xd[0], xd[-1]))
        vlines = [l for l in ax.lines if _is_vline(l)]
        self.assertEqual(len(vlines), n_clusters - 1)
        plt.close(fig)



class TestMultipleLinearPredictorScore(unittest.TestCase):
    """Tests for MultipleLinearPredictor.score."""

    def test_score_returns_score_result(self) -> None:
        """score returns a ScoreResult namedtuple."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        result = predictor.score()
        self.assertIsInstance(result, ScoreResult)

    def test_score_mean_rae_is_non_negative(self) -> None:
        """mean_rae is a non-negative float."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        result = predictor.score()
        self.assertGreaterEqual(result.mean_rae, 0.0)

    def test_score_max_rae_is_non_negative(self) -> None:
        """max_rae is a non-negative float."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_FORCED, n_cluster=1)
        result = predictor.score()
        self.assertGreaterEqual(result.max_rae, 0.0)

    def test_score_max_rae_ge_mean_rae(self) -> None:
        """max_rae is greater than or equal to mean_rae."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_cluster=2)
        result = predictor.score()
        self.assertGreaterEqual(result.max_rae, result.mean_rae)

    def test_score_multiple_clusters_finite(self) -> None:
        """score returns finite values for a multi-cluster predictor."""
        if IGNORE_TESTS:
            return
        predictor = _make_predictor_from_model(ANTIMONY_DECAY, n_cluster=3)
        result_3 = predictor.score()
        self.assertTrue(np.isfinite(result_3.mean_rae))
        self.assertTrue(np.isfinite(result_3.max_rae))

    def test_score_finite_for_multiple_cluster_counts(self) -> None:
        """score returns finite mean_rae for n_cluster in [1, 2, 3] on a nonlinear model."""
        if IGNORE_TESTS:
            return
        lr = LRoadrunner(ANTIMONY_MM, start_time=0.0, end_time=5.0, num_point=60)
        jc_full = Trajectory(lr)
        n_points = len(jc_full.timepoint_arr)

        for n_clusters in [1, 2, 3]:
            chunk_size = n_points // n_clusters
            jcs = []
            for i in range(n_clusters):
                start = i * chunk_size
                end = start + chunk_size if i < n_clusters - 1 else n_points
                jcs.append(Trajectory.fromArrays(
                        jc_full.jacobian_collection_arr[start:end],
                        jc_full.timepoint_arr[start:end],
                        lr))
            cjc = TrajectoryCollection(jcs)
            predictor = MultipleLinearPredictor.makeFromLRoadrunner(cjc, lr)
            self.assertTrue(np.isfinite(predictor.score().mean_rae))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestMultipleLinearPredictorScoreWithBioModels(unittest.TestCase):
    """Tests for MultipleLinearPredictor.score using real BioModels data (BIOMD8)."""

    MODEL_NAME = "BIOMD0000000008"
    BIOMD8_DIR = os.path.join(BIOMODELS_DIR, MODEL_NAME)
    BIOMD8_ENDTIME = 20.0

    def _skip_if_no_endtime(self) -> None:
        """Skip if BIOMD8 is not in the precomputed end-time table."""
        if self.MODEL_NAME not in LRoadrunner.endtime_dct:
            self.skipTest(f"{self.MODEL_NAME} not in endtime_dct")

    def _make_predictor(self, n_cluster: int) -> MultipleLinearPredictor:
        """Return a MultipleLinearPredictor for BIOMD8 with the given cluster count."""
        b_cluster = BiomodelsCluster(
                model_name=self.MODEL_NAME,
                start_time=0.0,
                end_time=self.BIOMD8_ENDTIME,
                num_point=100,
                diameter_metric="weighted_eigenvectors")
        cjc = b_cluster.cluster(n_cluster=n_cluster, is_sequential_partition=True)
        return MultipleLinearPredictor.makeFromLRoadrunner(cjc, b_cluster.l_roadrunner)

    def test_score_biomd8_returns_score_result(self) -> None:
        """score returns a ScoreResult for BIOMD8."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=1)
        result = predictor.score()
        self.assertIsInstance(result, ScoreResult)

    def test_score_biomd8_finite_values(self) -> None:
        """score returns finite mean_rae and max_rae for {self.MODEL_NAME}."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=1)
        result = predictor.score()
        self.assertTrue(np.isfinite(result.mean_rae))
        self.assertTrue(np.isfinite(result.max_rae))

    # FIXME: Score for 3 clusters is worse than score for 1 cluster
    def test_score_biomd8_three_clusters_finite(self) -> None:
        """score returns finite values for BIOMD8 with 3 clusters."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor_1 = self._make_predictor(n_cluster=1)
        result_1 = predictor_1.score()
        self.assertTrue(np.isfinite(result_1.mean_rae))
        self.assertTrue(np.isfinite(result_1.max_rae))
        predictor_3 = self._make_predictor(n_cluster=3)
        result_3 = predictor_3.score()
        self.assertTrue(np.isfinite(result_3.mean_rae))
        self.assertTrue(np.isfinite(result_3.max_rae))


class TestMultipleLinearPredictorScoreSimpleModel(unittest.TestCase):
    """Tests for MultipleLinearPredictor.score using a simple model."""

    MODEL_NAME = ANTIMONY_DECAY
    ENDTIME = 50


    def _make_jc(self) -> Tuple[LRoadrunner, Trajectory]:
        """Return a MultipleLinearPredictor built from a real Antimony model."""
        lr = _make_lroadrunner(self.MODEL_NAME, end_time=self.ENDTIME)
        return lr, Trajectory(lr)

    def _make_predictor_from_model(self, n_cluster: int, end_time:float = 10.0) -> MultipleLinearPredictor:
        """Return a MultipleLinearPredictor built from a real Antimony model."""
        lr, jc = self._make_jc()
        jcs = jc.sequentialPartition(n_cluster=n_cluster)
        cjc = TrajectoryCollection(jcs)
        return MultipleLinearPredictor.makeFromLRoadrunner(cjc, lr)

    def test_score_decay_forced_three_clusters_finite(self) -> None:
        """score returns finite values for BIOMD8 with 3 clusters."""
        #if IGNORE_TESTS:
        #    return
        predictor_1 = self._make_predictor_from_model(n_cluster=1, end_time=self.ENDTIME)
        result_1 = predictor_1.score()
        self.assertTrue(np.isfinite(result_1.mean_rae))
        self.assertTrue(np.isfinite(result_1.max_rae))
        predictor_3 = self._make_predictor_from_model(n_cluster=3, end_time=self.ENDTIME)
        result_3 = predictor_3.score()
        self.assertTrue(np.isfinite(result_3.mean_rae))
        self.assertTrue(np.isfinite(result_3.max_rae))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestMultipleLinearPredictorPlotWithBioModels(unittest.TestCase):
    """Tests for MultipleLinearPredictor.plot using real BioModels data (BIOMD8)."""

    BIOMD8_DIR = os.path.join(BIOMODELS_DIR, "BIOMD0000000008")

    def _skip_if_no_endtime(self) -> None:
        """Skip if BIOMD8 is not in the precomputed end-time table."""
        if "BIOMD0000000008" not in LRoadrunner.endtime_dct:
            self.skipTest("BIOMD0000000008 not in endtime_dct")

    def _make_predictor(self, n_cluster: int) -> MultipleLinearPredictor:
        """Return a MultipleLinearPredictor for BIOMD8 with the given cluster count."""
        b_cluster = BiomodelsCluster(
                model_name="BIOMD0000000008",
                start_time=0.0,
                end_time=20,
                num_point=100,
                diameter_metric="weighted_eigenvectors")
        cjc = b_cluster.cluster(n_cluster=n_cluster, is_sequential_partition=True)
        return MultipleLinearPredictor.makeFromLRoadrunner(cjc,
                b_cluster.l_roadrunner)

    def test_plot_biomd8_one_cluster(self) -> None:
        """plot returns a Figure for BIOMD8 with 1 cluster."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=1)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        plt.close(fig)

    def test_plot_biomd8_three_clusters(self) -> None:
        """plot for BIOMD8 with 3 clusters has exactly 2 vertical boundary lines."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        predictor = self._make_predictor(n_cluster=3)
        fig = predictor.plot()
        self.assertIsInstance(fig, mfigure.Figure)
        ax = fig.axes[0]
        def _is_vline(l) -> bool:  # type: ignore[no-untyped-def]
            xd: np.ndarray = np.asarray(l.get_xdata())
            return len(xd) >= 2 and bool(np.isclose(xd[0], xd[-1]))
        vlines = [l for l in ax.lines if _is_vline(l)]
        self.assertEqual(len(vlines), 2)
        plt.close(fig)



if __name__ == "__main__":
    unittest.main()
