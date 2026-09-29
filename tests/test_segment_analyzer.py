"""Tests for src.segment_analyzer.

Exercises SegmentAnalyzer with synthetic changepoint arrays so tests run
in under a second without needing BioModels files on disk.
"""

import unittest

import numpy as np  # type: ignore
import pandas as pd # type: ignore

import io  # type: ignore
import tempfile  # type: ignore
import os  # type: ignore

from src.segment_analyzer import SegmentAnalyzer  # type: ignore
import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore


# ---------------------------------------------------------------------------
# Helpers shared by tests.
# ---------------------------------------------------------------------------


def _make_csv_df(changpoints_list, system_ids, agg_types):
    """Build a DataFrame suitable for writing to a changepoints CSV file.

    Note: pandas serializes Python lists in columns as their string
    representation (e.g., "[0.0, 5.0]"), so consumers must be prepared
    to parse those strings back into actual lists.
    """
    return pd.DataFrame({
        cn.COL_CHANGEPOINTS: changpoints_list,
        cn.COL_SYSTEM_ID: system_ids,
        cn.COL_AGGREGATION_TYPE: agg_types,
    })


def _write_csv(df):
    """Write a DataFrame to a temporary CSV and return the file path."""
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_changpoints_")
    with os.fdopen(fd, "w") as fh:
        df.to_csv(fh, index=False)
    return path


def _remove_if_exists(path):
    """Remove a temporary file if it exists."""
    try:
        os.remove(path)
    except OSError:
        pass


def _write_json(df):
    """Write a DataFrame to a temporary JSON-lines file preserving list columns.

    Uses JSON format so Python lists survive the round-trip (unlike CSV).
    """
    import json  # type: ignore
    fd, path = tempfile.mkstemp(suffix=".json", prefix="test_changpoints_")
    with os.fdopen(fd, "w") as fh:
        for _, row in df.iterrows():
            fh.write(json.dumps(row.to_dict()) + "\n")
    return path


def _read_json(path):
    """Read back a JSON-lines file written by _write_json."""
    import json  # type: ignore
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)




# ---------------------------------------------------------------------------
# Helpers shared by tests.
# ---------------------------------------------------------------------------

_FLOAT_CHANGEPOINTS = [0.0, 2.5, 5.0, 10.0]
_INT_CHANGEPOINTS = [0, 3, 7, 10]
_SINGLE_CHANGEPINT = [0.0]


def _make_analyzer(changepoints=None):
    """Return a SegmentAnalyzer with the given changepoints (or defaults)."""
    return SegmentAnalyzer(changepoints if changepoints is not None else _FLOAT_CHANGEPOINTS)


# ---------------------------------------------------------------------------
# Tests for SegmentAnalyzer.__init__.
# ---------------------------------------------------------------------------

class TestSegmentAnalyzerInit(unittest.TestCase):
    """Tests for SegmentAnalyzer construction."""

    def test_stores_changpoints_as_given(self) -> None:
        changepoints = [0.0, 1.5, 4.0]
        analyzer = _make_analyzer(changepoints)
        self.assertEqual(analyzer._changepoints, changepoints)

    def test_segment_length_arr_computed_for_float_changpoints(self) -> None:
        expected = np.array([2.5, 2.5, 5.0])
        analyzer = _make_analyzer(_FLOAT_CHANGEPOINTS)
        np.testing.assert_array_equal(analyzer._segment_length_arr, expected)

    def test_segment_length_arr_computed_for_int_changpoints(self) -> None:
        expected = np.array([3, 4, 3])
        analyzer = _make_analyzer(_INT_CHANGEPOINTS)
        np.testing.assert_array_equal(analyzer._segment_length_arr, expected)

    def test_empty_changpoints_yields_empty_segment_lengths(self) -> None:
        analyzer = _make_analyzer([])
        self.assertEqual(len(analyzer._segment_length_arr), 0)
        np.testing.assert_array_equal(analyzer._segment_length_arr, np.array([]))

    def test_single_changpoint_yields_no_segments(self) -> None:
        """A single changepoint defines zero intervals."""
        analyzer = _make_analyzer(_SINGLE_CHANGEPINT)
        self.assertEqual(len(analyzer._segment_length_arr), 0)
        np.testing.assert_array_equal(analyzer._segment_length_arr, np.array([]))


# ---------------------------------------------------------------------------
# Tests for SegmentAnalyzer.makeSegmentAreaCDF.
# ---------------------------------------------------------------------------

class TestMakeSegmentAreaCDFAgainstKnownValues(unittest.TestCase):
    """Tests that pin down the exact numeric output of makeSegmentAreaCDF."""

    def test_uniform_segments_yield_single_step_at_length(self) -> None:
        changepoints = [0.0, 5.0, 10.0, 15.0]  # lengths: 5, 5, 5
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertEqual(len(cdf), 1)
        self.assertIn(5.0, cdf.index)
        self.assertAlmostEqual(cdf[5.0], 5.0)

    def test_mixed_lengths_produce_weighted_cumsum(self) -> None:
        changepoints = [0.0, 2.0, 4.0, 10.0]  # lengths: 2, 2, 6
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertAlmostEqual(cdf[2.0], 4 / 3, places=10)
        self.assertAlmostEqual(cdf[6.0], 10 / 3, places=10)

    def test_all_distinct_lengths_cumsum_is_quadratic(self) -> None:
        changepoints = [0.0, 1.0, 3.0, 6.0]  # lengths: 1, 2, 3 (all unique)
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertAlmostEqual(cdf[1.0], 1 / 3, places=10)
        self.assertAlmostEqual(cdf[2.0], 1.0, places=10)
        self.assertAlmostEqual(cdf[3.0], 2.0, places=10)

    def test_area_normalization_ends_at_mean_length(self) -> None:
        """The final CDF value equals mean(segment_lengths) for any input.

        The weighted-CDF cumsum accumulates (count(l)/n)*l across sorted unique l,
        which sums to sum(lengths)/n = mean(segment_lengths).
        """
        rng = np.random.default_rng(7)
        changepoints = [0.0] + list(np.sort(rng.uniform(1, 50, size=20)))
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()

        expected = np.mean(analyzer._segment_length_arr)
        self.assertAlmostEqual(cdf.iloc[-1], expected, places=10)

class TestMakeSegmentAreaCDFProperties(unittest.TestCase):
    """Property-based tests that hold for any valid SegmentAnalyzer input."""

    def test_returns_pandas_series(self) -> None:
        analyzer = _make_analyzer()
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertIsInstance(cdf, pd.Series)

    def test_index_is_sorted_ascending(self) -> None:
        changepoints = [0.0, 1.0, 3.0, 2.5]
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertTrue(cdf.index.is_monotonic_increasing)

    def test_index_matches_sorted_segment_lengths(self) -> None:
        changepoints = [0.0, 4.0, 1.0, 3.0]  # lengths: 4, -3, 2
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        expected_index = pd.Index(np.sort(analyzer._segment_length_arr))
        pd.testing.assert_index_equal(cdf.index, expected_index)

    def test_cumsum_is_monotonic_non_decreasing(self) -> None:
        rng = np.random.default_rng(0)
        changepoints = [0.0] + list(np.sort(rng.uniform(1, 20, size=8)))
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        diffs = cdf.diff().dropna()
        self.assertTrue((diffs >= -1e-12).all())

    def test_uniform_segments_collapse_to_one_step(self) -> None:
        changepoints = [0.0, 3.0, 6.0, 9.0, 12.0]
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertEqual(len(cdf), 1)

    def test_duplicate_lengths_produce_one_entry_per_unique_value(self) -> None:
        changepoints = [0.0, 2.0, 4.0, 7.0]  # lengths: 2, 2, 3 (two unique)
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertEqual(len(cdf), len(set(analyzer._segment_length_arr)))

    def test_negative_segment_lengths_allowed(self) -> None:
        changepoints = [0.0, 4.0, 1.0, 3.0]  # lengths: 4, -3, 2
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertIsInstance(cdf, pd.Series)
        self.assertEqual(len(cdf), 3)

    def test_empty_changpoints_yields_empty_series(self) -> None:
        analyzer = _make_analyzer([])
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertIsInstance(cdf, pd.Series)
        self.assertTrue(cdf.empty)

    def test_single_changpoint_yields_empty_series(self) -> None:
        analyzer = _make_analyzer([0.0])
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertIsInstance(cdf, pd.Series)
        self.assertTrue(cdf.empty)

    def test_two_changpoints_single_segment(self) -> None:
        changepoints = [0.0, 7.5]
        analyzer = _make_analyzer(changepoints)
        cdf = analyzer.makeSegmentAreaCDF()
        self.assertEqual(len(cdf), 1)
        self.assertIn(7.5, cdf.index)
        self.assertAlmostEqual(cdf[7.5], 7.5)



class TestGetChangpoints(unittest.TestCase):
    """Tests for the static getChangpoints class method."""

    def test_raises_value_error_for_missing_path(self) -> None:
        with self.assertRaises(ValueError):
            SegmentAnalyzer.getChangpoints("/nonexistent/path/to/file.csv")

    def _make_changpoints_csv(self, rows):
        """Build a CSV file from changepoint rows and return its path."""
        df = pd.DataFrame(rows)
        fd, p = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        return p

    def test_returns_dataframe_with_expected_columns(self) -> None:
        rows = [{cn.COL_CHANGEPOINTS: "[0.0, 5.0]", cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL}]
        path = self._make_changpoints_csv(rows)
        try:
            result = SegmentAnalyzer.getChangpoints(path)
            self.assertIsInstance(result, pd.DataFrame)
            expected_cols = {cn.COL_CHANGEPOINTS, cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE}
            self.assertEqual(set(result.columns), expected_cols)
        finally:
            _remove_if_exists(path)

    def test_preserves_system_id_values(self) -> None:
        rows = [
            {cn.COL_CHANGEPOINTS: "[0.0, 5.0]", cn.COL_SYSTEM_ID: "BIOMD0000000001",
             cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL},
            {cn.COL_CHANGEPOINTS: "[0.0, 3.0]", cn.COL_SYSTEM_ID: "BIOMD0000000002",
             cn.COL_AGGREGATION_TYPE: "Species_S1"},
        ]
        path = self._make_changpoints_csv(rows)
        try:
            result = SegmentAnalyzer.getChangpoints(path)
            pd.testing.assert_series_equal(
                result[cn.COL_SYSTEM_ID].reset_index(drop=True),
                pd.Series(["BIOMD0000000001", "BIOMD0000000002"]),
                check_names=False,
            )
        finally:
            _remove_if_exists(path)

    def test_preserves_aggregation_type_values(self) -> None:
        rows = [
            {cn.COL_CHANGEPOINTS: "[0.0, 5.0]", cn.COL_SYSTEM_ID: "BIOMD0000000001",
             cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL},
            {cn.COL_CHANGEPOINTS: "[0.0, 3.0]", cn.COL_SYSTEM_ID: "BIOMD0000000001",
             cn.COL_AGGREGATION_TYPE: "S1"},
            {cn.COL_CHANGEPOINTS: "[0.0, 7.0]", cn.COL_SYSTEM_ID: "BIOMD0000000002",
             cn.COL_AGGREGATION_TYPE: cn.COL_SPECIES_NAME},
        ]
        path = self._make_changpoints_csv(rows)
        try:
            result = SegmentAnalyzer.getChangpoints(path)
            pd.testing.assert_series_equal(
                result[cn.COL_AGGREGATION_TYPE].reset_index(drop=True),
                pd.Series([cn.COL_AGGREGATION_TYPE_MODEL, "S1", cn.COL_SPECIES_NAME]),
                check_names=False,
            )
        finally:
            _remove_if_exists(path)


class TestFromChangpointsModelLevel(unittest.TestCase):
    """Tests for fromChangpoints when requesting model-level aggregation."""

    def test_creates_analyzer_with_expected_changpoints(self) -> None:
        changepoints = [0.0, 2.5, 10.0]
        # Write CSV; getChangpoints will return the string repr of the list.
        rows = [{cn.COL_CHANGEPOINTS: "[0.0, 2.5, 10.0]",
                 cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL}]
        df = pd.DataFrame(rows)
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        try:
            analyzer = SegmentAnalyzer.fromChangpoints(path, model_num=1)
            self.assertEqual(analyzer._changepoints, changepoints)
        finally:
            _remove_if_exists(path)

    def test_handles_integer_model_number(self) -> None:
        """model_num=42 should match system_id 'BIOMD0000000042'."""
        changepoints = [0.0, 5.0]
        rows = [{cn.COL_CHANGEPOINTS: "[0.0, 5.0]",
                 cn.COL_SYSTEM_ID: "BIOMD0000000042",
                 cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL}]
        df = pd.DataFrame(rows)
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        try:
            analyzer = SegmentAnalyzer.fromChangpoints(path, model_num=42)
            self.assertEqual(analyzer._changepoints, changepoints)
        finally:
            _remove_if_exists(path)

    def test_raises_value_error_when_no_matching_row(self) -> None:
        rows = [{cn.COL_CHANGEPOINTS: "[0.0, 5.0]",
                 cn.COL_SYSTEM_ID: "BIOMD0000000099",
                 cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL}]
        df = pd.DataFrame(rows)
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                SegmentAnalyzer.fromChangpoints(path, model_num=1)
            self.assertIn("model 1", str(ctx.exception))
        finally:
            _remove_if_exists(path)


class TestFromChangpointsSpeciesLevel(unittest.TestCase):
    """Tests for fromChangpoints when requesting species-specific aggregation."""

    def test_returns_species_specific_changpoints(self) -> None:
        model_cp_str = "[0.0, 5.0]"
        species_cp = [0.0, 3.0, 8.0]
        rows = [
            {cn.COL_CHANGEPOINTS: model_cp_str,
             cn.COL_SYSTEM_ID: "BIOMD0000000001",
             cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL},
            {cn.COL_CHANGEPOINTS: str(species_cp),
             cn.COL_SYSTEM_ID: "BIOMD0000000001",
             cn.COL_AGGREGATION_TYPE: "Species_S1"},
        ]
        df = pd.DataFrame(rows)
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        try:
            analyzer = SegmentAnalyzer.fromChangpoints(path, model_num=1, species_name="Species_S1")
            self.assertEqual(analyzer._changepoints, species_cp)
        finally:
            _remove_if_exists(path)

    def test_raises_value_error_for_unknown_species(self) -> None:
        rows = [{cn.COL_CHANGEPOINTS: "[0.0, 5.0]",
                 cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL}]
        df = pd.DataFrame(rows)
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_cp_")
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        try:
            with self.assertRaises(ValueError):
                SegmentAnalyzer.fromChangpoints(path, model_num=1, species_name="NONEXISTENT")
        finally:
            _remove_if_exists(path)


if __name__ == "__main__":
    unittest.main()


