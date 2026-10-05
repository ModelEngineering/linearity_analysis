"""Tests for src/util.getPSDPredictionDF."""

import os
import sys
import unittest

import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from util import getPSDPredictionDF  # type: ignore

IGNORE_TESTS = False


class TestGetPSDPredictionDFFileNotFound(unittest.TestCase):
    """getPSDPredictionDF raises FileNotFoundError when no CSV matches."""

    def test_unknown_maxreduction_raises(self) -> None:
        """A max_fractional_reduction that appears in no filename raises."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(FileNotFoundError):
            getPSDPredictionDF("0.9999")

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.003__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_unknown_repeat_raises(self) -> None:
        """Passing a repeat value that no filename contains raises."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(FileNotFoundError):
            getPSDPredictionDF("0.003", repeat=99999)


class TestGetPSDPredictionDFSingleFile(unittest.TestCase):
    """Single-file match returns a correctly-shaped, filtered DataFrame."""

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.001__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_single_match_returns_df(self) -> None:
        """A unique match returns a non-empty DataFrame."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.001")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.001__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_single_match_csv_file_column(self) -> None:
        """The csv_file column is populated with the matching filename."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.001")
        self.assertIn("csv_file", df.columns.tolist())
        expected_name = (
            "piecewise_predictions__numpoint_100000__threshold_0.001"
            "__removal_0__maxreduction_0.001__manycp_0.csv"
        )
        self.assertTrue((df["csv_file"] == expected_name).all())

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.001__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_single_match_aggregation_type_model(self) -> None:
        """All returned rows have aggregation_type == 'model'."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.001")
        self.assertTrue((df["aggregation_type"] == "model").all())


class TestGetPSDPredictionDFAggregation(unittest.TestCase):

    """Concatenation and filtering across multiple matching files."""

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.003__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_multiple_files_concatenated(self) -> None:
        """When two files match, the result has rows from both."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.003")
        self.assertIsInstance(df, pd.DataFrame)
        expected_rows = 430 + 2556
        self.assertEqual(len(df), expected_rows)

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.003__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_multiple_files_both_filenames_present(self) -> None:
        """All csv_file values are among the two matching filenames."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.003")
        expected_names = {
            "piecewise_predictions__numpoint_100000__threshold_0.001"
            "__removal_0__maxreduction_0.003__manycp_0.csv",
            "piecewise_predictions__numpoint_100000__threshold_0.001"
            "__removal_1__maxreduction_0.003__manycp_1.csv",
        }
        self.assertEqual(set(df["csv_file"].tolist()), expected_names)

    @unittest.skipUnless(
        os.path.isfile(
            "/Users/jlheller/home/Technical/repos/linearity_analysis/data/piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.003__manycp_0.csv"
        ),
        "Test fixture file missing",
    )
    def test_multiple_files_no_species_rows(self) -> None:
        """The result contains no species-level rows."""
        if IGNORE_TESTS:
            return
        df = getPSDPredictionDF("0.003")
        self.assertTrue((df["aggregation_type"] == "model").all())


if __name__ == "__main__":
    unittest.main()

