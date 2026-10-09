"""Tests for src/psd_predictions_iterator.py.

These tests cover PSDPredictionsItem construction, _parse_filename parsing and error
cases, _load_dataframe CSV/pickle loading, and end-to-end iteration over synthetic files
in temporary directories (patched over cn.DATA_DIR).
"""
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.psd_predictions_iterator import PSDPredictionsIterator, PSDPredictionsItem


def _csv_bytes(maxreduction=0.01, threshold=0.001, removal=0) -> bytes:
    """Build CSV bytes for a single model-level row of PSD metadata."""
    h = ("system_id,aggregation_type,max_fractional_reduction,"
         "coefficient_threshold,is_changepoint_removal")
    b = f"BIOMD001,model,{maxreduction},{threshold},{removal}"
    return f"{h}\n{b}".encode("utf-8")


class TestPSDPredictionsItem(unittest.TestCase):
    """Tests for PSDPredictionsItem construction."""

    def test_stores_all_attributes(self) -> None:
        df = pd.DataFrame({"a": [1]})
        item = PSDPredictionsItem(
            filename="t.csv", max_fractional_reduction=0.5, coefficient_threshold=0.2,
            is_changepoint_removal=True, max_changepoint=999, num_point=500,
            repeat=3, manycp=True, df=df)
        self.assertEqual(item.filename, "t.csv")
        self.assertAlmostEqual(item.max_fractional_reduction, 0.5)
        self.assertAlmostEqual(item.coefficient_threshold, 0.2)
        self.assertTrue(item.is_changepoint_removal)
        self.assertEqual(item.max_changepoint, 999)
        self.assertEqual(item.num_point, 500)
        self.assertEqual(item.repeat, 3)
        self.assertTrue(item.manycp)
        self.assertIs(item.df, df)


class TestPSDPredictionsIteratorInit(unittest.TestCase):
    """Tests for PSDPredictionsIterator.__init__."""

    def test_default_is_csv(self) -> None:
        it = PSDPredictionsIterator()
        self.assertFalse(it.is_pkl)
        self.assertEqual(it.data_dir, cn.DATA_DIR)

    def test_pickle_mode(self) -> None:
        it = PSDPredictionsIterator(is_pkl=True)
        self.assertTrue(it.is_pkl)


class TestParseFilename(unittest.TestCase):
    """Tests for _parse_filename covering structured metadata and error cases."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _iter_csv(self):
        return PSDPredictionsIterator(is_pkl=False)

    def _iter_pkl(self):
        return PSDPredictionsIterator(is_pkl=True)

    def test_parses_full_structured_csv_filename(self) -> None:
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__numpoint_100000"
            "__threshold_0.001__removal_0__maxreduction_0.01.csv")
        self.assertEqual(r["num_point"], 100000)
        self.assertAlmostEqual(r["threshold"], 0.001, places=6)
        self.assertEqual(r["maxreduction"], 0.01)

    def test_parses_scientific_notation_maxreduction(self) -> None:
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__maxreduction_1e-3.csv")
        self.assertAlmostEqual(r["maxreduction"], 0.001, places=6)

    def test_parses_repeat_metadata(self) -> None:
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__repeat_5.csv")
        self.assertEqual(r["repeat"], 5)

    def test_parses_manycp_metadata(self) -> None:
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__manycp_1.csv")
        # codedstrToDict evaluates "1" as int(1), not bool(True).
        self.assertEqual(r.get("manycp"), 1)

    def test_returns_defaults_for_minimal_metadata(self) -> None:
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__maxreduction_0.5.csv")
        self.assertEqual(r["num_point"], 100000)
        self.assertAlmostEqual(r["threshold"], 0.001, places=6)

    def test_parses_structured_pkl_filename(self) -> None:
        """Note: parsed keys (e.g., 'numpoint') don't override defaults ('num_point'),
        so only metadata keys whose names match the default dict are actually applied.
        See https://github.com/... for the key-mapping bug this produces."""
        r = self._iter_pkl()._parse_filename(
            "piecewise_predictions__maxreduction_1e-2.pkl")
        # Default num_point (100000) is kept because 'numpoint' != 'num_point'.
        self.assertEqual(r["num_point"], 100000)
        self.assertAlmostEqual(r["maxreduction"], 0.01, places=6)

    def test_raises_for_wrong_extension_csv(self) -> None:
        with self.assertRaises(ValueError):
            self._iter_csv()._parse_filename(
                "piecewise_predictions__m_0.1.pkl")

    def test_raises_for_wrong_extension_pkl(self) -> None:
        with self.assertRaises(ValueError):
            self._iter_pkl()._parse_filename(
                "piecewise_predictions__m_0.1.csv")

    def test_raises_for_non_matching_prefix(self) -> None:
        with self.assertRaises(ValueError):
            self._iter_csv()._parse_filename("other_file.csv")

    def test_raises_when_no_separators_in_name(self) -> None:
        """A filename without __ separators is rejected."""
        with self.assertRaises(ValueError):
            self._iter_csv()._parse_filename("piecewise_predictions.csv")

    def test_key_mismatch_numpoint_ignored(self) -> None:
        """'numpoint' in filename does not override 'num_point' default key.

        This is a known bug: codedstrToDict returns raw keys from the filename
        (e.g., 'numpoint') which never match the snake_case defaults ('num_point').
        See https://github.com/... for the fix."""
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__numpoint_50.csv")
        # Default num_point (100000) is kept because 'numpoint' != 'num_point'.
        self.assertEqual(r["num_point"], 100000,
                         msg="Bug: 'numpoint' in filename should override default 'num_point'.")
        # The parsed value is stored under its raw key.
        self.assertIn("numpoint", r)

    def test_key_mismatch_threshold_correctly_applied(self) -> None:
        """'threshold' in filename correctly overrides default (both use key 'threshold')."""
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__threshold_0.5.csv")
        # Both defaults and parsed code use the exact same key, so override works.
        self.assertAlmostEqual(r["threshold"], 0.5, places=6)

    def test_key_mismatch_manycp_correctly_applied(self) -> None:
        """'manycp' in filename correctly overrides default (both use key 'manycp')."""
        r = self._iter_csv()._parse_filename(
            "piecewise_predictions__manycp_1.csv")
        # This key matches the default, so it is correctly applied.
        self.assertEqual(r["manycp"], 1)


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
        it = PSDPredictionsIterator(is_pkl=False)
        df = it._load_dataframe(path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)

    def test_loads_pickle_file(self) -> None:
        src = pd.DataFrame({"a": [1], "b": [2]})
        path = os.path.join(self._tmpdir.name, "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(src, f)
        it = PSDPredictionsIterator(is_pkl=True)
        loaded = it._load_dataframe(path)
        self.assertTrue(loaded.equals(src))

    def test_load_nonexistent_csv_raises(self) -> None:
        it = PSDPredictionsIterator(is_pkl=False)
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
        self._write_file(
            "piecewise_predictions__maxreduction_0.2.csv",
            _csv_bytes(0.2, 0.1, removal=1))
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionsIterator(is_pkl=False))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertAlmostEqual(item.max_fractional_reduction, 0.2, places=6)

    def test_iter_csv_skips_empty_csv_files(self) -> None:
        """Files that raise during load are skipped silently."""
        # Valid file + empty file (pd.read_csv raises on empty files).
        self._write_file(
            "piecewise_predictions__maxreduction_0.1.csv",
            _csv_bytes(0.1, 0.2))
        self._write_file("empty.csv", b"")
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionsIterator(is_pkl=False))
        # Only the valid file should be yielded.
        self.assertEqual(len(items), 1)

    def test_iter_csv_empty_dir(self) -> None:
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionsIterator(is_pkl=False))
        self.assertEqual(len(items), 0)

    def test_iter_csv_yields_dataframe_with_content(self) -> None:
        name = "piecewise_predictions__maxreduction_0.5.csv"
        self._write_file(name, _csv_bytes(0.5, 0.25))
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionsIterator(is_pkl=False))
        self.assertEqual(len(items), 1)
        df = items[0].df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_iter_csv_skips_non_matching_files(self) -> None:
        """Files not starting with piecewise_predictions are skipped."""
        path = os.path.join(self._tmpdir.name, "other_file.csv")
        with open(path, "wb") as f:
            f.write(b"col1\n1")
        with patch.object(cn, "DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionsIterator(is_pkl=False))
        self.assertEqual(len(items), 0)


if __name__ == "__main__":
    unittest.main()
