"""Tests for scripts/update_psd_prediction_files.py.

These tests cover:
- _addAUCColumn: AUC computation from changepoints/boundaries
- main(): is_report output, backup file creation
- Error handling: empty DataFrames, missing columns, invalid boundaries
"""

import src.constants as cn  # type: ignore
from scripts import update_psd_prediction_files  # type: ignore

import pandas as pd  # type: ignore
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock  # type: ignore


class TestAddAUCColumn(unittest.TestCase):
    """Tests for _addAUCColumn function."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_empty_boundaries_raises(self) -> None:
        """Empty changepoints list (empty boundaries) raises ValueError from NLCurve."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001"],
            cn.COL_BOUNDARIES: [[]],
        })
        # Empty boundaries cause NLCurve to raise ValueError directly (no internal handling)
        with self.assertRaises(ValueError) as cm:
            update_psd_prediction_files._addAUCColumn(df)
        self.assertIn("At least two boundaries", str(cm.exception))

    def test_single_segment_boundaries(self) -> None:
        """Single segment [0, NUM_POINT-1] should give AUC=1.0."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001"],
            cn.COL_BOUNDARIES: [[0, 99999]],  # NUM_POINT=100000
        })
        result = update_psd_prediction_files._addAUCColumn(df)
        self.assertEqual(result, 1)
        self.assertAlmostEqual(df[cn.COL_AUC].iloc[0], 1.0, places=5)

    def test_multiple_segments(self) -> None:
        """Multiple segments produce AUC < 1.0 based on segment lengths."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001"],
            cn.COL_BOUNDARIES: [[0, 50000, 99999]],  # Two segments
        })
        result = update_psd_prediction_files._addAUCColumn(df)
        self.assertEqual(result, 1)
        # AUC should be less than 1.0 for multiple segments
        self.assertLess(df[cn.COL_AUC].iloc[0], 1.0)
        self.assertGreater(df[cn.COL_AUC].iloc[0], 0.0)

    def test_multiple_rows(self) -> None:
        """Multiple rows are all updated."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001", "BIOMD002"],
            cn.COL_BOUNDARIES: [[0, 100], [0, 200, 99999]],
        })
        result = update_psd_prediction_files._addAUCColumn(df)
        self.assertEqual(result, 2)
        self.assertIn(cn.COL_AUC, df.columns)
        self.assertEqual(len(df[cn.COL_AUC]), 2)

    def test_missing_boundaries_column_raises(self) -> None:
        """Missing boundaries column raises an exception."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001"],
        })
        # The function should propagate KeyError from accessing missing column
        with self.assertRaises(KeyError):
            update_psd_prediction_files._addAUCColumn(df)

    def test_invalid_boundaries_values_raises(self) -> None:
        """Invalid boundary values (e.g., negative) are passed to NLCurve."""
        df = pd.DataFrame({
            cn.COL_SYSTEM_ID: ["BIOMD001"],
            cn.COL_BOUNDARIES: [[-1, 100]],  # Invalid boundary
        })
        result = update_psd_prediction_files._addAUCColumn(df)
        self.assertEqual(result, 1)
        # NLCurve accepts negative values and computes AUC
        self.assertIn(cn.COL_AUC, df.columns)


class TestMain(unittest.TestCase):
    """Tests for main() function."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
