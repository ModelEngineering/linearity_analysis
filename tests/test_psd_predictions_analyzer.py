"""Tests for src/psd_predictions_analyzer.py."""

import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.psd_predictions_analyzer import (  # type: ignore
    DEFAULT_MAX_CHANGEPOINT,
    PSDPredictionAnalyzer,
)


# ---------------------------------------------------------------------------
# Shared row factory
# ---------------------------------------------------------------------------


def _row(
    system_id: str = "SYS1",
    aggregation_type: str = cn.COL_AGGREGATION_TYPE_MODEL,
    max_changepoint: int = DEFAULT_MAX_CHANGEPOINT,
    max_fractional_reduction: float = 0.01,
    auc: float = 0.8,
    p50: float = 0.7,
    num_changepoint: int = 5,
) -> dict:
    """Return a dict with every column required by analyzeAUCConsistency."""
    return {
        cn.COL_SYSTEM_ID: system_id,
        cn.COL_AGGREGATION_TYPE: aggregation_type,
        cn.COL_MAX_CHANGEPOINT: max_changepoint,
        cn.COL_MAX_FRACTIONAL_REDUCTION: max_fractional_reduction,
        cn.COL_AUC: auc,
        cn.COL_P50: p50,
        cn.COL_NUM_CHANGEPOINT: num_changepoint,
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPSDPredictionAnalyzerInit(unittest.TestCase):

    def test_default_is_pkl_true(self) -> None:
        analyzer = PSDPredictionAnalyzer()
        self.assertTrue(analyzer.psd_iterator.is_pkl)

    def test_is_pkl_false_stored(self) -> None:
        analyzer = PSDPredictionAnalyzer(is_pkl=False)
        self.assertFalse(analyzer.psd_iterator.is_pkl)


# ---------------------------------------------------------------------------
# analyzeAUCConsistency
# ---------------------------------------------------------------------------


class TestAnalyzeAUCConsistency(unittest.TestCase):
    """End-to-end tests using a patched cn.DATA_DIR and real pickle fixtures."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _write_pkl(self, name: str, df: pd.DataFrame) -> None:
        """Write df as a pickle file named `name` inside the temp directory."""
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            pickle.dump(df, f)

    def _run(self, system_id: str = "SYS1") -> pd.DataFrame:
        """Run analyzeAUCConsistency against the temp directory."""
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            analyzer = PSDPredictionAnalyzer(is_pkl=True)
            return analyzer.analyzeAUCConsistency(system_id)

    def _write_default_pkl(self, **override) -> None:
        """Write a single-row pickle with default column values, overridden by kwargs."""
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([_row(**override)]),
        )

    # ------------------------------------------------------------------
    # Return type / structure
    # ------------------------------------------------------------------

    def test_returns_dataframe(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_name_is_max_fractional_reduction(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertEqual(result.index.name, cn.COL_MAX_FRACTIONAL_REDUCTION)

    def test_has_auc_mean_column(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertIn((cn.COL_AUC, "mean"), result.columns)

    def test_has_auc_std_column(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertIn((cn.COL_AUC, "std"), result.columns)

    # ------------------------------------------------------------------
    # Aggregation correctness
    # ------------------------------------------------------------------

    def test_auc_mean_single_row(self) -> None:
        self._write_default_pkl(auc=0.75, max_fractional_reduction=0.01)
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.75, places=10) # type: ignore

    def test_auc_mean_two_rows_same_group(self) -> None:
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([
                _row(auc=0.6, max_fractional_reduction=0.01),
                _row(auc=0.8, max_fractional_reduction=0.01),
            ]),
        )
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.7, places=10)  # type: ignore

    def test_two_reductions_produce_two_groups(self) -> None:
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([
                _row(max_fractional_reduction=0.01,  auc=0.8),
                _row(max_fractional_reduction=0.001, auc=0.9),
            ]),
        )
        result = self._run()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc[0.01,  (cn.COL_AUC, "mean")], 0.8, places=10)  # type: ignore
        self.assertAlmostEqual(result.loc[0.001, (cn.COL_AUC, "mean")], 0.9, places=10)  # type: ignore

    def test_rows_from_multiple_files_merged_into_same_group(self) -> None:
        """Rows from two separate files that share max_fractional_reduction are averaged."""
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([_row(auc=0.6, max_fractional_reduction=0.01)]),
        )
        self._write_pkl(
            "piecewise_predictions__numpoint_200__maxreduction_0.01.pkl",
            pd.DataFrame([_row(auc=0.8, max_fractional_reduction=0.01)]),
        )
        result = self._run()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.7, places=10) # type: ignore

    # ------------------------------------------------------------------
    # Filter: system_id
    # ------------------------------------------------------------------

    def test_wrong_system_id_excluded(self) -> None:
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([
                _row(system_id="KEEP", auc=0.9),
                _row(system_id="DROP", auc=0.1),
            ]),
        )
        result = self._run(system_id="KEEP")
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.9, places=10)  # type: ignore

    # ------------------------------------------------------------------
    # Filter: aggregation_type
    # ------------------------------------------------------------------

    def test_non_model_aggregation_type_excluded(self) -> None:
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([
                _row(aggregation_type=cn.COL_AGGREGATION_TYPE_MODEL, auc=0.9),
                _row(aggregation_type="species",                      auc=0.1),
            ]),
        )
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.9, places=10) # type: ignore

    # ------------------------------------------------------------------
    # Filter: max_changepoint
    # ------------------------------------------------------------------

    def test_wrong_max_changepoint_excluded(self) -> None:
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.01.pkl",
            pd.DataFrame([
                _row(max_changepoint=DEFAULT_MAX_CHANGEPOINT, auc=0.9),
                _row(max_changepoint=100,                     auc=0.1),
            ]),
        )
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, "mean")], 0.9, places=10)  # type: ignore

    # ------------------------------------------------------------------
    # Empty-result handling
    # ------------------------------------------------------------------

    def test_empty_result_when_no_rows_match_system_id(self) -> None:
        """No rows for the requested system_id → empty grouped DataFrame."""
        self._write_default_pkl(system_id="OTHER")
        result = self._run(system_id="MISSING")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# End-to-end tests using the real production data file
# ---------------------------------------------------------------------------

# The file with the validated full schema (system_id, auc, max_changepoint, …)
_E2E_PKL = (
    "piecewise_predictions__numpoint_100000"
    "__threshold_0.001__removal_0__manycp_1.pkl"
)
_E2E_DATA_PATH = os.path.join(cn.DATA_DIR, _E2E_PKL)
_HAS_E2E_DATA = os.path.isfile(_E2E_DATA_PATH)


@unittest.skipUnless(_HAS_E2E_DATA, f"real data file {_E2E_PKL!r} not present in data/")
class TestAnalyzeAUCConsistencyEndToEnd(unittest.TestCase):
    """End-to-end tests against the real production pkl from data/.

    The dummy pkl files in data/ have a different column structure and would
    cause a KeyError inside _filter_dataframe.  A symlink of just the valid
    file into a temp directory isolates the test from those files while still
    exercising the real data end-to-end.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory() # type: ignore
        os.symlink(_E2E_DATA_PATH, os.path.join(cls._tmpdir.name, _E2E_PKL)) # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()  # type: ignore

    def _run(self, system_id: str) -> pd.DataFrame:
        with patch.object(cn, "DATA_DIR", self._tmpdir.name): # type: ignore
            return PSDPredictionAnalyzer(is_pkl=True).analyzeAUCConsistency(system_id)

    # --- structural checks ---

    def test_returns_dataframe(self) -> None:
        self.assertIsInstance(self._run("BIOMD0000000005"), pd.DataFrame)

    def test_has_auc_mean_and_std_columns(self) -> None:
        result = self._run("BIOMD0000000005")
        self.assertIn((cn.COL_AUC, "mean"), result.columns)
        self.assertIn((cn.COL_AUC, "std"), result.columns)

    def test_index_name_is_max_fractional_reduction(self) -> None:
        self.assertEqual(
            self._run("BIOMD0000000005").index.name,
            cn.COL_MAX_FRACTIONAL_REDUCTION,
        )

    # --- group structure from the real file ---

    def test_one_group_because_file_has_one_reduction_value(self) -> None:
        """The file contains only max_fractional_reduction=0.05 → exactly 1 group."""
        self.assertEqual(len(self._run("BIOMD0000000005")), 1)

    def test_group_index_value_is_0_05(self) -> None:
        result = self._run("BIOMD0000000005")
        self.assertAlmostEqual(result.index[0], 0.05, places=10)

    # --- quantitative checks against known values ---

    def test_auc_mean_matches_known_value_for_biomd_5(self) -> None:
        """BIOMD0000000005 has a single qualifying row; known AUC is 0.89999598."""
        result = self._run("BIOMD0000000005")
        self.assertAlmostEqual(result.loc[0.05, (cn.COL_AUC, cn.COL_MEAN)],
            0.8999959790295811, places=6)  # type: ignore

    def test_auc_mean_matches_known_value_for_biomd_10(self) -> None:
        """BIOMD0000000010 has a single qualifying row; known AUC is 0.90000699."""
        result = self._run("BIOMD0000000010")
        self.assertAlmostEqual(
            result.loc[0.05, (cn.COL_AUC, "mean")],
            0.9000069937498743, # type: ignore
            places=6,
        )  # type: ignore

    # --- boundary / filter checks ---

    def test_unknown_system_id_returns_empty_dataframe(self) -> None:
        result = self._run("NONEXISTENT_BIOMD")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
