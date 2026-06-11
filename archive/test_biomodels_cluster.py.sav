"""Tests for BiomodelsCluster."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn
from biomodels_cluster import BiomodelsCluster  # type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from trajectory import Trajectory  # type: ignore
from l_roadrunner import LRoadrunner  # type: ignore

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)
TEST_MODEL = "BIOMD0000000001"
FIRST_MODEL_NUM = 1
LAST_MODEL_NUM = 10

# Minimal SBML for a two-species decay used in unit tests that don't need
# real BioModels data.
ANTIMONY_DECAY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10.0; S2 = 0.0
"""


def _make_fake_biomodels_dir(model_name: str, sbml_content: str) -> str:
    """
    Create a temporary BioModels-style directory tree and return its root path.

    The directory layout is  <tmpdir>/<model_name>/<model_name>_url.xml  so that
    BiomodelsCluster._getSbml() can locate the file.
    """
    tmpdir = tempfile.mkdtemp()
    model_dir = os.path.join(tmpdir, model_name)
    os.makedirs(model_dir)
    xml_path = os.path.join(model_dir, f"{model_name}_url.xml")
    with open(xml_path, "w") as f:
        f.write(sbml_content)
    return tmpdir


# ---------------------------------------------------------------------------
# _getSbml
# ---------------------------------------------------------------------------

class TestGetSbml(unittest.TestCase):
    """Tests for BiomodelsCluster._getSbml."""

    def setUp(self) -> None:
        import tellurium as te  # type: ignore
        rr = te.loada(ANTIMONY_DECAY)
        self._sbml_content = rr.getSBML()
        self._model_name = "BIOMD9999999999"
        self._tmpdir = _make_fake_biomodels_dir(self._model_name, self._sbml_content)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_cluster(self) -> BiomodelsCluster:
        """Return a BiomodelsCluster with BIOMODELS_DIR patched to the temp dir."""
        with patch("biomodels_cluster.cn.BIOMODELS_DIR", self._tmpdir):
            return BiomodelsCluster(self._model_name,
                end_time=10.0, num_point=11)

    def test_returns_string(self) -> None:
        """_getSbml returns a string."""
        if IGNORE_TESTS:
            return
        bc = self._make_cluster()
        self.assertIsInstance(bc.sbml_str, str)

    def test_content_is_nonempty(self) -> None:
        """Returned SBML string is non-empty."""
        if IGNORE_TESTS:
            return
        bc = self._make_cluster()
        self.assertGreater(len(bc.sbml_str), 0)

    def test_raises_when_no_xml_file(self) -> None:
        """FileNotFoundError is raised when the model directory has no XML files."""
        if IGNORE_TESTS:
            return
        import shutil
        empty_model = "BIOMD0000000000"
        model_dir = os.path.join(self._tmpdir, empty_model)
        os.makedirs(model_dir)
        with patch("biomodels_cluster.cn.BIOMODELS_DIR", self._tmpdir):
            with self.assertRaises(FileNotFoundError):
                BiomodelsCluster(empty_model, end_time=10.0,
                        num_point=11)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestBiomodelsClusterInit(unittest.TestCase):
    """Tests for BiomodelsCluster.__init__ using a fake BioModels directory."""

    def setUp(self) -> None:
        import tellurium as te  # type: ignore
        rr = te.loada(ANTIMONY_DECAY)
        self._sbml_content = rr.getSBML()
        self._model_name = "BIOMD9999999999"
        self._tmpdir = _make_fake_biomodels_dir(self._model_name, self._sbml_content)
        with patch("biomodels_cluster.cn.BIOMODELS_DIR", self._tmpdir):
            self._biomodels_cluster = BiomodelsCluster(
                    self._model_name, start_time=0.0, end_time=10.0, num_point=11,
                    diameter_metric=cn.DIAMETER_MAX_CV)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_model_name_stored(self) -> None:
        """model_name attribute matches the constructor argument."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self._biomodels_cluster.model_name, self._model_name)

    def test_start_time_stored(self) -> None:
        """start_time attribute matches the constructor argument."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self._biomodels_cluster.start_time, 0.0)

    def test_end_time_stored(self) -> None:
        """end_time attribute matches the constructor argument."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self._biomodels_cluster.end_time, 10.0)

    def test_num_point_stored(self) -> None:
        """num_point attribute matches the constructor argument."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self._biomodels_cluster.num_point, 11)

    def test_sbml_str_is_string(self) -> None:
        """sbml_str attribute is a non-empty string."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._biomodels_cluster.sbml_str, str)
        self.assertGreater(len(self._biomodels_cluster.sbml_str), 0)

    def test_l_roadrunner_created(self) -> None:
        """l_roadrunner attribute is an LRoadrunner instance."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                type(self._biomodels_cluster.l_roadrunner).__name__, "LRoadrunner")

    def test_jacobian_collection_created(self) -> None:
        """_jacobian_collection attribute is a JacobianCollection instance."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._biomodels_cluster._jacobian_collection, Trajectory)

    def test_end_time_default_from_lookup(self) -> None:
        """When end_time is NaN, it falls back to LRoadrunner.endtime_dct or cn.END_TIME."""
        if IGNORE_TESTS:
            return
        with patch("biomodels_cluster.cn.BIOMODELS_DIR", self._tmpdir):
            bc = BiomodelsCluster(self._model_name, end_time=np.nan, num_point=11)
        # end_time must be a finite number (from default or lookup)
        self.assertFalse(np.isnan(bc.end_time))


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

class TestBiomodelsClusterCluster(unittest.TestCase):
    """Tests for BiomodelsCluster.cluster."""

    def setUp(self) -> None:
        import tellurium as te  # type: ignore
        rr = te.loada(ANTIMONY_DECAY)
        self._sbml_content = rr.getSBML()
        self._model_name = "BIOMD9999999999"
        self._tmpdir = _make_fake_biomodels_dir(self._model_name, self._sbml_content)
        with patch("biomodels_cluster.cn.BIOMODELS_DIR", self._tmpdir):
            self._biomodels_cluster = BiomodelsCluster(
                    self._model_name, start_time=0.0, end_time=10.0, num_point=11,
                    diameter_metric=cn.DIAMETER_MAX_CV)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_trajectory_collection(self) -> None:
        """cluster returns a ClusteredJacobianCollection."""
        if IGNORE_TESTS:
            return
        cjc = self._biomodels_cluster.cluster(n_cluster=1)
        self.assertIsInstance(cjc, TrajectoryCollection)

    def test_sequential_partition_single_cluster(self) -> None:
        """Sequential partition with n_cluster=1 yields one JacobianCollection."""
        if IGNORE_TESTS:
            return
        cjc = self._biomodels_cluster.cluster(n_cluster=1, is_sequential_partition=True)
        self.assertEqual(len(cjc.trajectories), 1)

    def test_sequential_partition_multiple_clusters(self) -> None:
        """Sequential partition with n_cluster=2 yields two JacobianCollections."""
        if IGNORE_TESTS:
            return
        cjc = self._biomodels_cluster.cluster(n_cluster=2, is_sequential_partition=True)
        self.assertEqual(len(cjc.trajectories), 2)

    def test_nonsequential_partition(self) -> None:
        """Non-sequential partition returns a ClusteredJacobianCollection."""
        if IGNORE_TESTS:
            return
        cjc = self._biomodels_cluster.cluster(n_cluster=2, is_sequential_partition=False)
        self.assertIsInstance(cjc, TrajectoryCollection)

    def test_max_cv_is_finite(self) -> None:
        """max_cv of the returned ClusteredJacobianCollection is a finite number."""
        if IGNORE_TESTS:
            return
        cjc = self._biomodels_cluster.cluster(n_cluster=1)
        self.assertTrue(np.isfinite(cjc.max_cv))


# ---------------------------------------------------------------------------
# clusterAnalysis  (integration, requires real BioModels directory)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestClusterAnalysis(unittest.TestCase):
    """Integration tests for BiomodelsCluster.clusterAnalysis."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._output_csv = os.path.join(self._tmpdir, "results.csv")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_dataframe(self) -> None:
        """clusterAnalysis returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = BiomodelsCluster.clusterAnalysis(
                output_data_file=self._output_csv,
                excluded_models=[],
                n_cluster=1,
                is_report=False,
                is_sequential_partition=True,
                diameter_metric=cn.DIAMETER_MAX_CV,
                first_model_num=FIRST_MODEL_NUM, last_model_num=LAST_MODEL_NUM,
        )
        self.assertIsInstance(df, pd.DataFrame)

    def test_output_csv_created(self) -> None:
        """clusterAnalysis writes a CSV file to output_data_file."""
        if IGNORE_TESTS:
            return
        BiomodelsCluster.clusterAnalysis(
                output_data_file=self._output_csv,
                excluded_models=[],
                n_cluster=1,
                is_report=False,
                is_sequential_partition=True,
                diameter_metric=cn.DIAMETER_MAX_CV,
                first_model_num=FIRST_MODEL_NUM, last_model_num=LAST_MODEL_NUM,
        )
        self.assertTrue(os.path.isfile(self._output_csv))

    def test_dataframe_has_expected_columns(self) -> None:
        """Result DataFrame contains model_name and max_cv columns."""
        if IGNORE_TESTS:
            return
        df = BiomodelsCluster.clusterAnalysis(
                output_data_file=self._output_csv,
                n_cluster=1,
                is_report=IGNORE_TESTS,
                is_sequential_partition=True,
                diameter_metric=cn.DIAMETER_MAX_CV,
                first_model_num=FIRST_MODEL_NUM, last_model_num=LAST_MODEL_NUM,
        )
        for col in [cn.COL_MAXCV, cn.COL_ENDTIME]:
            self.assertIn(col, df.columns)

    def test_excluded_model_not_in_result(self) -> None:
        """A model listed in excluded_models does not appear in the result."""
        if IGNORE_TESTS:
            return
        df = BiomodelsCluster.clusterAnalysis(
                output_data_file=self._output_csv,
                excluded_models=[TEST_MODEL],
                n_cluster=1,
                is_report=IGNORE_TESTS,
                is_sequential_partition=True,
                diameter_metric=cn.DIAMETER_MAX_CV,
                first_model_num=FIRST_MODEL_NUM, last_model_num=LAST_MODEL_NUM,
        )
        if TEST_MODEL in df.index:
            self.fail(f"{TEST_MODEL} should have been excluded")


if __name__ == "__main__":
    unittest.main()
