"""
Tests for LinearAnalyzer class.
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import ClassVar

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn
from trajectory import Trajectory  # type: ignore
from linear_analyzer import LinearAnalyzer  #  type: ignore
from trajectory_collection import TrajectoryCollection  # type: ignore
from trajectory import Trajectory as JC  # type: ignore

IGNORE_TESTS = False
ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
BIOMD1_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000001", "BIOMD0000000001_url.xml")
BIOMD300_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000300", "BIOMD0000000300_url.xml")
BIOMD4_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000004", "BIOMD0000000004_url.xml")
BIOMD206_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000206", "BIOMD0000000206_url.xml")
BIOMD241_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000241", "BIOMD0000000241_url.xml")
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)


def _load_sbml(path: str) -> str:
    """Read an SBML file and return its contents as a string."""
    with open(path) as f:
        return f.read()


class TestLinearAnalyzerInit(unittest.TestCase):
    """Tests for LinearAnalyzer.__init__."""

    def test_defaults_stored(self) -> None:
        """Default simulation parameters are stored on the instance."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.start_time, 0)
        self.assertIsNone(analyzer.end_time)  # None triggers auto-detect of steady-state end time
        self.assertEqual(analyzer.num_point, 100)

    def test_custom_params_stored(self) -> None:
        """Custom start, end, and num_point are stored correctly."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, start_time=1, end_time=5, num_point=50)
        self.assertEqual(analyzer.start_time, 1)
        self.assertEqual(analyzer.end_time, 5)
        self.assertEqual(analyzer.num_point, 50)

    def test_model_stored(self) -> None:
        """The model string is stored on the instance."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL)
        self.assertEqual(analyzer.model, ANTIMONY_MODEL)

    def test_jacobian_collection_created(self) -> None:
        """A JacobianCollection is created during __init__."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        self.assertIsInstance(analyzer._jacobian_collection, Trajectory)

    def test_jacobian_collection_shape(self) -> None:
        """The JacobianCollection has the expected shape for the ANTIMONY_MODEL."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(ANTIMONY_MODEL, num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 2, 2))  # 2 floating species: S1, S2

    def test_init_sbml(self) -> None:
        """Initializing with an SBML string (from Antimony) loads correctly."""
        if IGNORE_TESTS:
            return
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        analyzer = LinearAnalyzer(sbml_str, num_point=5)
        self.assertIsInstance(analyzer._jacobian_collection, Trajectory)
        self.assertEqual(analyzer._jacobian_collection.jacobian_arr.ndim, 3)


FIRST_FIVE_MODEL_IDS = [
    "BIOMD0000000026",
    "BIOMD0000000027",
    "BIOMD0000000028",
    "BIOMD0000000029",
    "BIOMD0000000030",
]


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestPartitionBiomodelsJacobians(unittest.TestCase):
    """Tests for LinearAnalyzer.partitionBiomodelsJacobians.

    Uses the first 5 models from the BioModels final directory.  Some models
    may fail to load; tests that require at least one success will assert on
    ``len(df) > 0`` rather than a hard-coded model ID.
    """
    def setUp(self) -> None:
        """Copy the first 5 BioModels into a shared temporary directory."""
        self._tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir_obj.name
        for model_id in FIRST_FIVE_MODEL_IDS:
            src = os.path.join(BIOMODELS_DIR, model_id)
            dst = os.path.join(self.tmp_dir, model_id)
            shutil.copytree(src, dst)

    def tearDown(self) -> None:
        self._tmp_dir_obj.cleanup()

    def _data_file(self) -> str:
        """Return a per-test CSV path inside the shared tmp directory."""
        return os.path.join(self.tmp_dir, f"{self._testMethodName}.csv")

    def test_returns_dataframe(self) -> None:
        """partitionBiomodelsJacobians returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file(),
            is_report=IGNORE_TESTS,  # Suppress print statements during testing
        )
        self.assertIsInstance(df, pd.DataFrame)

    def test_empty_directory_returns_empty_dataframe(self) -> None:
        """An empty directory yields an empty DataFrame."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as empty_dir:
            data_file = os.path.join(empty_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=empty_dir, output_data_file=data_file,
                is_report=IGNORE_TESTS,
            )
        self.assertEqual(len(df), 0)

    def test_index_contains_model_ids(self) -> None:
        """DataFrame index contains at least one of the first-five model IDs."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file(),
            is_report=IGNORE_TESTS
        )
        self.assertTrue(
            any(mid in df.index for mid in FIRST_FIVE_MODEL_IDS),
            f"Expected at least one of {FIRST_FIVE_MODEL_IDS} in index {list(df.index)}",
        )

    def test_has_max_cv_and_end_time_columns(self) -> None:
        """DataFrame has max_cv and end_time columns but not end_time_source."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file(),
            is_report=IGNORE_TESTS
        )
        self.assertIn("max_cv", df.columns)
        self.assertIn("end_time", df.columns)
        self.assertNotIn("end_time_source", df.columns)

    def test_values_are_floats(self) -> None:
        """max_cv values for successfully loaded models are floats."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=self._data_file(),
            is_report=IGNORE_TESTS
        )
        self.assertGreater(len(df), 0, "No models succeeded; cannot check value types")
        first_model = df.index[0]
        self.assertIsInstance(df.loc[first_model, "max_cv"], (int, float))

    def test_csv_is_created(self) -> None:
        """The output CSV file is created."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file,
            is_report=IGNORE_TESTS
        )
        self.assertTrue(os.path.isfile(data_file))

    def test_csv_is_valid_and_readable(self) -> None:
        """The written CSV can be read back by pandas and contains model IDs."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file,
            is_report=IGNORE_TESTS
        )
        df_on_disk = pd.read_csv(data_file, header=None, index_col=0)
        self.assertTrue(
            any(mid in df_on_disk.index for mid in FIRST_FIVE_MODEL_IDS),
            "CSV does not contain any expected model ID",
        )

    def test_skips_invalid_model(self) -> None:
        """A directory with an invalid XML file is skipped without raising."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_dir = os.path.join(tmp_dir, "BADMODEL")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "BADMODEL_url.xml"), "w") as f:
                f.write("<?xml version='1.0'?><not_sbml/>")
            data_file = os.path.join(tmp_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file,
                is_report=IGNORE_TESTS,
            )
        self.assertEqual(len(df), 0)

    def test_excluded_models_are_skipped(self) -> None:
        """Models in excluded_models are not processed."""
        if IGNORE_TESTS:
            return
        excluded = FIRST_FIVE_MODEL_IDS[:3]
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir,
            output_data_file=self._data_file(),
            excluded_models=excluded,
            is_report=IGNORE_TESTS,
        )
        for model_id in excluded:
            self.assertNotIn(model_id, df.index)

    def test_sequential_partition_flag(self) -> None:
        """is_sequential_partition=True runs without error and returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir,
            output_data_file=self._data_file(),
            is_sequential_partition=True,
            is_report=IGNORE_TESTS,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(
            any(mid in df.index for mid in FIRST_FIVE_MODEL_IDS),
            "No first-five model IDs found in sequential-partition result",
        )

    def test_already_processed_model_is_skipped(self) -> None:
        """A model already in the CSV is not reprocessed (row count stays the same)."""
        if IGNORE_TESTS:
            return
        data_file = self._data_file()
        df1 = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file, is_report=IGNORE_TESTS,
        )
        df2 = LinearAnalyzer.partitionBiomodelsJacobians(
            directory=self.tmp_dir, output_data_file=data_file, is_report=IGNORE_TESTS,
        )
        self.assertEqual(len(df1), len(df2))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestMakeBiomodelCusteredJacobianCollection(unittest.TestCase):
    """Tests for LinearAnalyzer.makeBiomodelCusteredJacobianCollection."""

    MODEL_DIR = os.path.join(BIOMODELS_DIR, "BIOMD0000000300")

    def setUp(self) -> None:
        from l_roadrunner import LRoadrunner  # type: ignore
        self._endtime_dct = LRoadrunner.endtime_dct

    def _skip_if_no_endtime(self) -> None:
        """Skip the calling test if MODEL_DIR's model name is not in endtime_dct."""
        model_name = os.path.basename(self.MODEL_DIR)
        if model_name not in self._endtime_dct:
            self.skipTest(f"{model_name} not in endtime_dct")

    def test_returns_clustered_jacobian_collection(self) -> None:
        """Returns a ClusteredJacobianCollection for a valid model with a known end time."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(self.MODEL_DIR)
        self.assertIsInstance(result, TrajectoryCollection)

    def test_default_n_cluster_one(self) -> None:
        """Default n_cluster=1 yields a single JacobianCollection."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(self.MODEL_DIR)
        self.assertEqual(len(result.jacobian_collections), 1)

    def test_n_cluster_parameter(self) -> None:
        """n_cluster controls the number of jacobian_collections returned."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(
            self.MODEL_DIR, n_cluster=3
        )
        self.assertEqual(len(result.jacobian_collections), 3)

    def test_each_collection_is_jacobian_collection(self) -> None:
        """Each element in jacobian_collections is a JacobianCollection."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(
            self.MODEL_DIR, n_cluster=2
        )
        for jc in result.jacobian_collections:
            self.assertIsInstance(jc, JC)

    def test_max_cv_is_non_negative(self) -> None:
        """max_cv of the result is non-negative."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(self.MODEL_DIR)
        self.assertGreaterEqual(result.max_cv, 0.0)

    def test_sequential_partition_false(self) -> None:
        """is_sequential_partition=False (k-means) returns a valid result."""
        if IGNORE_TESTS:
            return
        self._skip_if_no_endtime()
        result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(
            self.MODEL_DIR, n_cluster=2, is_sequential_partition=False
        )
        self.assertIsInstance(result, TrajectoryCollection)
        self.assertEqual(len(result.jacobian_collections), 2)

    def test_missing_end_time_returns_empty(self) -> None:
        """A model whose directory name is absent from endtime_dct returns an empty collection."""
        if IGNORE_TESTS:
            return
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "FAKE_MODEL_NOT_IN_DICT")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "FAKE_MODEL_NOT_IN_DICT_url.xml"), "w") as f:
                f.write(sbml_str)
            result = LinearAnalyzer.makeBiomodelCusteredJacobianCollection(model_dir)
        self.assertIsInstance(result, TrajectoryCollection)
        self.assertEqual(len(result.jacobian_collections), 0)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels directory not available")
class TestWithBioModels(unittest.TestCase):
    """Integration tests using real SBML files from temp-biomodels."""

    def test_init_biomd3(self) -> None:
        """LinearAnalyzer initializes correctly for BIOMD300 (3 floating species)."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.shape, (10, 3, 3))

    def test_init_biomd1_timepoints(self) -> None:
        """JacobianCollection timepoints length equals num_point."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=10)
        self.assertEqual(len(analyzer._jacobian_collection.timepoint_arr), 10)

    def test_partition_jacobians_biomd3(self) -> None:
        """partitionJacobians works on a real SBML model."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=20)
        result = analyzer._jacobian_collection.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result), 3)
        total = sum(jc.jacobian_arr.shape[0] for jc in result)
        self.assertEqual(total, 20)

    def test_sequential_partition_biomd3(self) -> None:
        """partitionJacobiansSequentially produces contiguous segments on a real SBML model."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD300_SBML), num_point=20)
        result = analyzer._jacobian_collection.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result], axis=0
        )
        np.testing.assert_array_equal(
            reconstructed, analyzer._jacobian_collection.jacobian_arr
        )

    def test_partition_biomodels_with_real_models(self) -> None:
        """partitionBiomodelsJacobians returns valid max_cv values for real models."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_id, src_file in [
                ("BIOMD0000000300", BIOMD300_SBML),
                ("BIOMD0000000004", BIOMD4_SBML),
            ]:
                model_dir = os.path.join(tmp_dir, model_id)
                os.makedirs(model_dir)
                dst = os.path.join(model_dir, f"{model_id}_url.xml")
                with open(src_file) as src, open(dst, "w") as out:
                    out.write(src.read())
            data_file = os.path.join(tmp_dir, "out.csv")
            df = LinearAnalyzer.partitionBiomodelsJacobians(
                directory=tmp_dir, output_data_file=data_file,
                is_report=IGNORE_TESTS,
            )
        self.assertIn("BIOMD0000000300", df.index)
        self.assertIn("BIOMD0000000004", df.index)
        self.assertGreaterEqual(df.loc["BIOMD0000000300", "max_cv"], 0.0)  # type: ignore[arg-type]

    def test_init_biomd241(self) -> None:
        """LinearAnalyzer initializes correctly for BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=10)
        arr = analyzer._jacobian_collection.jacobian_arr
        self.assertEqual(arr.ndim, 3)
        self.assertEqual(arr.shape[0], 10)

    def test_partition_jacobians_biomd241(self) -> None:
        """partitionJacobians works on BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=20)
        result = analyzer._jacobian_collection.partitionJacobians(n_cluster=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(jc.jacobian_arr.shape[0] for jc in result), 20)

    def test_sequential_partition_biomd241(self) -> None:
        """partitionJacobiansSequentially produces contiguous segments on BIOMD241."""
        if IGNORE_TESTS:
            return
        analyzer = LinearAnalyzer(_load_sbml(BIOMD241_SBML), num_point=20)
        result = analyzer._jacobian_collection.partitionJacobiansSequentially(n_cluster=3)
        reconstructed = np.concatenate(
            [jc.jacobian_arr for jc in result], axis=0
        )
        np.testing.assert_array_equal(
            reconstructed, analyzer._jacobian_collection.jacobian_arr
        )


if __name__ == "__main__":
    unittest.main()
