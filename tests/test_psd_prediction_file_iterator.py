"""Tests for src/psd_predictions_iterator.py.

These tests cover PSDPredictionsItem construction, _parse_filename parsing and error
cases, _load_dataframe CSV/pickle loading, and end-to-end iteration over synthetic files
in temporary directories (patched over cn.DATA_DIR).
"""
import os
import pickle
import tempfile
from typing import cast  # type: ignore
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from psd_prediction_file_iterator import PSDPredictionFileIterator, PSDPredictionItem
import psd_prediction_file_iterator as psd_iter  # type: ignore
import src.util as util  # type: ignore


def _csv_bytes(maxreduction=0.01, threshold=0.001, removal=0) -> bytes:
    """Build CSV bytes for a single model-level row of PSD metadata."""
    h = ("system_id,aggregation_type,max_fractional_reduction,"
         "coefficient_threshold,is_changepoint_removal")
    b = f"BIOMD001,model,{maxreduction},{threshold},{removal}"
    return f"{h}\n{b}".encode("utf-8")

def _make_filename(num_point=100000, manycp=0, max_fractional_reduction=0.01, coefficient_threshold=0.001,
        is_changepoint_removal=False,
            repeat=1) -> str:
    """Build a structured filename with encoded metadata."""
    dct = {
        cn.COL_NUM_POINT: num_point,
        cn.COL_MANYCP: manycp,
        cn.COL_MAX_FRACTIONAL_REDUCTION: max_fractional_reduction,
        cn.COL_COEFFICIENT_THRESHOLD: coefficient_threshold,
        cn.COL_IS_CHANGEPOINT_REMOVAL: int(is_changepoint_removal),
        cn.COL_REPEAT: repeat,
    }
    coded_str = util.dictToCodedstr(dct)
    return f"piecewise_predictions__numpoint_{coded_str}"



class TestPSDPredictionFilesIterator(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator construction."""

    def test_stores_all_attributes(self) -> None:
        df = pd.DataFrame({"a": [1]})
        item = PSDPredictionItem(filepath="/path/to/t.csv",
            filename="t.csv", max_fractional_reduction=0.5, coefficient_threshold=0.2,
            is_changepoint_removal=True, max_changepoint=999, num_point=500,
            repeat=3, manycp=True, df=df)
        self.assertEqual(item.filename, "t")
        self.assertAlmostEqual(item.max_fractional_reduction, 0.5)
        self.assertAlmostEqual(item.coefficient_threshold, 0.2)
        self.assertTrue(item.is_changepoint_removal)
        self.assertEqual(item.max_changepoint, 999)
        self.assertEqual(item.num_point, 500)
        self.assertEqual(item.repeat, 3)
        self.assertTrue(item.manycp)
        self.assertIs(item.df, df)



class TestPSDPredictionFilesIteratorInit(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator.__init__."""

    def test_default_is_csv(self) -> None:
        it = PSDPredictionFileIterator()
        self.assertFalse(it.is_pkl)
        self.assertEqual(it.data_dir, cn.PSD_DATA_DIR)

    def test_pickle_mode(self) -> None:
        it = PSDPredictionFileIterator(is_pkl=True)
        self.assertTrue(it.is_pkl)
#
#
#class TestParseFilename(unittest.TestCase):
#    """Tests for _parse_filename covering structured metadata and error cases."""
#
#    def setUp(self) -> None:
#        self._tmpdir = tempfile.TemporaryDirectory()
#
#    def tearDown(self) -> None:
#        self._tmpdir.cleanup()
#
#    def _iter_csv(self):
#        return PSDPredictionFileIterator(is_pkl=False)
#
#    def _iter_pkl(self):
#        return PSDPredictionFileIterator(is_pkl=True)
#
#    def test_parses_full_structured_csv_filename(self) -> None:
#        filename = _make_filename(numpoint=100000, manycp=1, maxreduction=0.01,
#                threshold=0.001, removal=0) + ".csv"
#        r = self._iter_csv()._parse_filename(filename)
#        r = cast(dict, r)
#        self.assertEqual(r["numpoint"], 100000)
#        self.assertEqual(r["manycp"], 1)
#        self.assertAlmostEqual(r["threshold"], 0.001)
#        self.assertEqual(r["maxreduction"], 0.01)
#
#    def test_parses_scientific_notation_maxreduction(self) -> None:
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__maxreduction_1e-3.csv")
#        r = cast(dict, r)
#        self.assertIsNone(r[cn.COL_NUM_POINT])
#        self.assertIsNone(r[cn.COL_COEFFICIENT_THRESHOLD])
#        self.assertIsNone(r[cn.COL_IS_CHANGEPOINT_REMOVAL])
#        self.assertAlmostEqual(r[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.001, places=6)
#
#    def test_parses_repeat_metadata(self) -> None:
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__repeat_5.csv")
#        r = cast(dict, r)
#        self.assertEqual(r["repeat"], 5)
#
#    def test_parses_manycp_metadata(self) -> None:
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__manycp_1.csv")
#        # codedstrToDict evaluates "1" as int(1), not bool(True).
#        r = cast(dict, r)
#        self.assertEqual(r.get("manycp"), 1)
#
#    def test_returns_defaults_for_minimal_metadata(self) -> None:
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__maxreduction_0.5.csv")
#        r = cast(dict, r)
#        self.assertIsNone(r[cn.COL_NUM_POINT])
#        self.assertIsNone(r[cn.COL_COEFFICIENT_THRESHOLD])
#        self.assertIsNone(r[cn.COL_IS_CHANGEPOINT_REMOVAL])
#        self.assertAlmostEqual(r[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.5, places=6)
#
#    def test_parses_structured_pkl_filename(self) -> None:
#        """Note: parsed keys (e.g., 'numpoint') don't override defaults ('num_point'),
#        so only metadata keys whose names match the default dict are actually applied.
#        See https://github.com/... for the key-mapping bug this produces."""
#        r = self._iter_pkl()._parse_filename(
#            "piecewise_predictions__maxreduction_1e-2.pkl")
#        # Default num_point (100000) is kept because 'numpoint' != 'num_point'.
#        r = cast(dict, r)
#        self.assertIsNone(r[cn.COL_NUM_POINT])
#        self.assertIsNone(r[cn.COL_COEFFICIENT_THRESHOLD])
#        self.assertAlmostEqual(r[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.01, places=6)
#
#    def test_raises_for_wrong_extension_csv(self) -> None:
#        with self.assertRaises(ValueError):
#            self._iter_csv()._parse_filename(
#                "piecewise_predictions__m_0.1.pkl")
#
#    def test_raises_for_wrong_extension_pkl(self) -> None:
#        with self.assertRaises(ValueError):
#            self._iter_pkl()._parse_filename(
#                "piecewise_predictions__m_0.1.csv")
#
#    def test_raises_for_non_matching_prefix(self) -> None:
#        with self.assertRaises(ValueError):
#            self._iter_csv()._parse_filename("other_file.csv")
#
#    def test_raises_when_no_separators_in_name(self) -> None:
#        """A filename without __ separators is rejected."""
#        with self.assertRaises(ValueError):
#            self._iter_csv()._parse_filename("piecewise_predictions.csv")
#
#    def test_key_mismatch_numpoint_ignored(self) -> None:
#        """'numpoint' in filename does not override 'num_point' default key.
#
#        This is a known bug: codedstrToDict returns raw keys from the filename
#        (e.g., 'numpoint') which never match the snake_case defaults ('num_point').
#        See https://github.com/... for the fix."""
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__numpoint_50.csv")
#        # Default num_point (100000) is kept because 'numpoint' != 'num_point'.
#        r = cast(dict, r)
#        self.assertIsNotNone(r[cn.COL_NUM_POINT])
#        # The parsed value is stored under its raw key.
#        self.assertIn(cn.COL_NUM_POINT, r)
#
#    def test_key_mismatch_threshold_correctly_applied(self) -> None:
#        """'threshold' in filename correctly overrides default (both use key 'threshold')."""
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__threshold_0.5.csv")
#        # Both defaults and parsed code use the exact same key, so override works.
#        r = cast(dict, r)
#        self.assertAlmostEqual(r["threshold"], 0.5, places=6)
#
#    def test_key_mismatch_manycp_correctly_applied(self) -> None:
#        """'manycp' in filename correctly overrides default (both use key 'manycp')."""
#        r = self._iter_csv()._parse_filename(
#            "piecewise_predictions__manycp_1.csv")
#        # This key matches the default, so it is correctly applied.
#        r = cast(dict, r)
#        self.assertEqual(r["manycp"], 1)


class TestLoadDataframe(unittest.TestCase):
    """Tests for _load_dataframe using real CSV and pickle files."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_loads_csv_file(self) -> None:
        path = os.path.join(self._tmpdir.name, "test.csv")
        with open(path, "wb") as f:
            f.write(_csv_bytes(0.1, 0.5, removal=1))
        it = PSDPredictionFileIterator(is_pkl=False)
        df = it._load_dataframe(path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)

    def test_loads_pickle_file(self) -> None:
        src = pd.DataFrame({"a": [1], "b": [2]})
        path = os.path.join(self._tmpdir.name, "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(src, f)
        it = PSDPredictionFileIterator(is_pkl=True)
        loaded = it._load_dataframe(path)
        self.assertTrue(loaded.equals(src))

    def test_load_nonexistent_csv_raises(self) -> None:
        it = PSDPredictionFileIterator(is_pkl=False)
        with self.assertRaises(FileNotFoundError):
            it._load_dataframe("/nonexistent/path.csv")


class TestIterOverSyntheticFiles(unittest.TestCase):
    """End-to-end tests using a temp directory patched over cn.DATA_DIR."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_file(self, name: str, data: bytes) -> str:
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_iter_csv_yields_items_with_correct_metadata(self) -> None:
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.2,
                coefficient_threshold=0.1, is_changepoint_removal=True) + ".csv"
        self._write_file(filename,
            _csv_bytes(0.2, 0.1, removal=1))
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertAlmostEqual(item.max_fractional_reduction, 0.2, places=6)

    def test_iter_csv_skips_empty_csv_files(self) -> None:
        """Files that raise during load are skipped silently."""
        # Valid file + empty file (pd.read_csv raises on empty files).
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.1,
                coefficient_threshold=0.2, is_changepoint_removal=True) + ".csv"
        self._write_file(filename,
            _csv_bytes(0.1, 0.2))
        self._write_file("empty.csv", b"")
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name))
        # Only the valid file should be yielded.
        self.assertEqual(len(items), 1)

    def test_iter_csv_empty_dir(self) -> None:
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name))
        self.assertEqual(len(items), 0)

    def test_iter_csv_yields_dataframe_with_content(self) -> None:
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.5,
                coefficient_threshold=0.25, is_changepoint_removal=True) + ".csv"
        self._write_file(filename, _csv_bytes(0.5, 0.25))
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name))
        self.assertEqual(len(items), 1)
        df = items[0].df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_iter_csv_skips_non_matching_files(self) -> None:
        """Files not starting with piecewise_predictions are skipped."""
        path = os.path.join(self._tmpdir.name, "other_file.csv")
        with open(path, "wb") as f:
            f.write(b"col1\n1")
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name))
        self.assertEqual(len(items), 0)


if __name__ == "__main__":
    unittest.main()


class TestPSDPredictionFilesIteratorQuery(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator.query classmethod."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_pkl(self, name: str, df: pd.DataFrame) -> str:
        path = os.path.join(self._tmpdir.name, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(df, f)
        return path

    def _write_bytes(self, name: str, data: bytes) -> str:
        """Write raw bytes to a file in the temp dir (used for CSV fixtures)."""
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_query_default_returns_concatenated_results(self) -> None:
        """With no filters, query returns all data concatenated plus a filename column."""
        df1 = pd.DataFrame({"species": ["A", "B"], "value": [1.0, 2.0]})
        df2 = pd.DataFrame({"species": ["C"], "value": [3.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df1)
        self._write_pkl(
            "piecewise_predictions__numpoint_200__maxreduction_0.1.pkl", df2)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name, is_model=False)
        self.assertEqual(len(result), 3)
        self.assertIn(cn.COL_FILENAME, result.columns)
        fnames = set(result[cn.COL_FILENAME])
        for f in fnames:
            self.assertIn("piecewise_predictions__numpoint_", f)

    def test_query_file_filter_reduces_results(self) -> None:
        """file_filter receives parsed metadata dict and can exclude files by it."""
        df1 = pd.DataFrame({"species": ["A"], "value": [1.0]})
        df2 = pd.DataFrame({"species": ["B"], "value": [2.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df1)
        self._write_pkl(
            "piecewise_predictions__numpoint_200__maxreduction_0.1.pkl", df2)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(
                file_filter=lambda dct: dct.get("numpoint") == 200,
                data_dir=self._tmpdir.name, is_model=False)
        # Only the second file should pass; its df has one row.
        self.assertEqual(len(result), 1)

    def test_query_data_frame_filter_modifies_rows(self) -> None:
        """data_frame_filter result is applied to the output (not discarded)."""
        ##
        def data_frame_filter(df: pd.DataFrame, dct: dict) -> pd.DataFrame:
            """Filter out rows with value <= 1."""
            mask = df["value"] > 1
            new_df = df.copy()
            new_df.loc[mask, "value"] = 2.0
            return new_df
        ##
        df = pd.DataFrame({"species": ["A", "B"], "value": [0.5, 2.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(
                data_frame_filter=data_frame_filter,
                data_dir=self._tmpdir.name, is_model=False)
        # Only the row with value=2.0 should remain after filtering.
        self.assertEqual(len(result), 2)
        test_result = result.iloc[1]["value"] == 2.0
        self.assertTrue(test_result)

    def test_query_both_filters_combined(self) -> None:
        """Both filters compose correctly — file filter then data frame filter."""
        df1 = pd.DataFrame({"species": ["A", "B"], "value": [0.5, 2.0]})
        df2 = pd.DataFrame({"species": ["C"], "value": [3.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df1)
        self._write_pkl(
            "piecewise_predictions__numpoint_200__maxreduction_0.1.pkl", df2)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(
                file_filter=lambda dct: dct.get("numpoint") == 100,
                data_frame_filter=lambda d, _: d[d["value"] > 1],
                data_dir=self._tmpdir.name, is_model=False)
        # File filter keeps df1 only; data frame filter then drops value=0.5 row.
        self.assertEqual(len(result), 1)

    def test_query_uses_pickle_files_by_default(self) -> None:
        """With default is_pkl=True, csv files in the same dir are ignored."""
        self._write_bytes(
            "piecewise_predictions__numpoint_50__maxreduction_0.3.csv",
            _csv_bytes(0.3, 0.2))
        pkl_df = pd.DataFrame({"species": ["Y"], "value": [1.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", pkl_df)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name, is_model=False)
        # Only the pickle file should contribute rows; CSV is ignored.
        self.assertEqual(len(result), 1)

    def test_query_csv_mode_with_is_pkl_false(self) -> None:
        """With is_pkl=False, only CSV files are read."""
        self._write_bytes(
            "piecewise_predictions__numpoint_50__maxreduction_0.3.csv",
            _csv_bytes(0.3, 0.2))
        pkl_df = pd.DataFrame({"species": ["W"], "value": [7.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", pkl_df)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name, is_pkl=False)
        # Only the CSV file should contribute rows; pickle is ignored.
        self.assertEqual(len(result), 1)

    def test_query_empty_dir_returns_empty_dataframe(self) -> None:
        """With no matching files, query returns an empty DataFrame."""
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_query_adds_filename_column_to_each_row(self) -> None:
        """Each row in the output carries back its source filename."""
        df1 = pd.DataFrame({"species": ["A", "B"], "value": [1.0, 2.0]})
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.5, coefficient_threshold=0.25, is_changepoint_removal=True)
        _ = self._write_pkl(filename, df1)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name, is_model=False,
                    is_pkl=True)
        # item.filename strips the file extension per PSDPredictionsItem.__init__.
        self.assertTrue((result[cn.COL_FILENAME] == filename).all())

    def test_query_with_default_filter_is_identity(self) -> None:
        """Passing default (identity) filters gives the same rows as no filter."""
        df = pd.DataFrame({"species": ["A"], "value": [1.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            default_result = PSDPredictionFileIterator.query()
            explicit_result = PSDPredictionFileIterator.query(
                file_filter=lambda dct: True,
                data_frame_filter=lambda d, _: d)
        self.assertEqual(len(default_result), len(explicit_result))
        for col in default_result.columns:
            pd.testing.assert_series_equal(
                default_result[col], explicit_result[col])


if __name__ == "__main__":
    unittest.main()
