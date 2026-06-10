"""Tests for DataframeSerializer."""

import os
import shutil  # type: ignore
import sys
import tempfile  # type: ignore
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataframe_serializer import DataframeSerializer  # type: ignore

IGNORE_TESTS = False

ROWS = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]


class TestInit(unittest.TestCase):
    """Tests for DataframeSerializer.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'data.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_path_stored(self) -> None:
        """Constructor stores the path."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        self.assertEqual(dataframe_serializer._path, self.tmp_path)

    def test_new_file_empty_df(self) -> None:
        """self._df is an empty DataFrame when the file does not exist."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        self.assertTrue(dataframe_serializer.dataframe.empty)

    def test_existing_file_loads_data(self) -> None:
        """self._df is populated from the CSV when the file already exists."""
        if IGNORE_TESTS:
            return
        pd.DataFrame(ROWS).to_csv(self.tmp_path, index=False)
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        self.assertEqual(len(dataframe_serializer.dataframe), 2)
        self.assertListEqual(list(dataframe_serializer.dataframe.columns), ['a', 'b'])

    def test_existing_file_values_correct(self) -> None:
        """Values loaded from an existing CSV match what was written."""
        if IGNORE_TESTS:
            return
        pd.DataFrame(ROWS).to_csv(self.tmp_path, index=False)
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        np.testing.assert_array_equal(
                dataframe_serializer.dataframe['a'].values, [1, 3])  # type: ignore


class TestSerialize(unittest.TestCase):
    """Tests for DataframeSerializer.serialize."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'data.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_file(self) -> None:
        """serialize() creates the CSV file."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        self.assertTrue(os.path.exists(self.tmp_path))

    def test_updates_df(self) -> None:
        """serialize() appends the new rows to self._df."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        self.assertEqual(len(dataframe_serializer.dataframe), 2)

    def test_multiple_calls_accumulate(self) -> None:
        """Successive serialize() calls accumulate all rows in self._df."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        dataframe_serializer.serialize(ROWS)
        self.assertEqual(len(dataframe_serializer.dataframe), 4)

    def test_csv_content_matches_df(self) -> None:
        """The written CSV round-trips back to the same DataFrame."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        reloaded_df = pd.read_csv(self.tmp_path)
        pd.testing.assert_frame_equal(dataframe_serializer.dataframe, reloaded_df)

    def test_column_mismatch_raises(self) -> None:
        """serialize() raises ValueError when dict keys differ from existing columns."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        with self.assertRaises(ValueError):
            dataframe_serializer.serialize([{'a': 5, 'c': 6}])

    def test_empty_df_accepts_any_keys(self) -> None:
        """serialize() imposes no column constraint when self._df is empty."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize([{'x': 1, 'y': 2}])
        self.assertListEqual(list(dataframe_serializer.dataframe.columns), ['x', 'y'])

    def test_extra_key_raises(self) -> None:
        """serialize() raises ValueError when new dicts have an extra key."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        with self.assertRaises(ValueError):
            dataframe_serializer.serialize([{'a': 5, 'b': 6, 'c': 7}])

    def test_missing_key_raises(self) -> None:
        """serialize() raises ValueError when new dicts are missing a key."""
        if IGNORE_TESTS:
            return
        dataframe_serializer = DataframeSerializer(self.tmp_path)
        dataframe_serializer.serialize(ROWS)
        with self.assertRaises(ValueError):
            dataframe_serializer.serialize([{'a': 5}])


class TestEq(unittest.TestCase):
    """Tests for DataframeSerializer.__eq__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.path_a = os.path.join(self.tmp_dir, 'a.csv')
        self.path_b = os.path.join(self.tmp_dir, 'b.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make(self, path: str,
            rows: list = None) -> DataframeSerializer:  # type: ignore
        """Returns a DataframeSerializer, optionally with rows serialized."""
        ds = DataframeSerializer(path)
        if rows:
            ds.serialize(rows)
        return ds

    def test_equal_empty_instances(self) -> None:
        """Two instances with no data are equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a)
        b = self._make(self.path_b)
        self.assertEqual(a, b)

    def test_equal_same_data(self) -> None:
        """Two instances with identical rows are equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = self._make(self.path_b, ROWS)
        self.assertEqual(a, b)

    def test_equal_to_self(self) -> None:
        """An instance is equal to itself."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        self.assertEqual(a, a)

    def test_not_equal_different_values(self) -> None:
        """Two instances with different row values are not equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = self._make(self.path_b, [{'a': 9, 'b': 9}])
        self.assertNotEqual(a, b)

    def test_not_equal_different_columns(self) -> None:
        """Two instances with different column names are not equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = self._make(self.path_b, [{'x': 1, 'y': 2}])
        self.assertNotEqual(a, b)

    def test_not_equal_different_row_count(self) -> None:
        """Two instances with different numbers of rows are not equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = self._make(self.path_b, ROWS[:1])
        self.assertNotEqual(a, b)

    def test_path_does_not_affect_equality(self) -> None:
        """Same data at different paths compares as equal."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = self._make(self.path_b, ROWS)
        self.assertEqual(a, b)

    def test_not_equal_to_non_instance(self) -> None:
        """Comparing with a non-DataframeSerializer returns False."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        self.assertFalse(a == "not a DataframeSerializer")
        self.assertFalse(a == 42)
        self.assertFalse(a == None)  # type: ignore

    def test_nan_values_treated_as_equal(self) -> None:
        """NaN values in the same positions compare as equal (DataFrame.equals semantics)."""
        if IGNORE_TESTS:
            return
        nan_rows = [{'a': float('nan'), 'b': 1.0}]
        a = self._make(self.path_a, nan_rows)
        b = self._make(self.path_b, nan_rows)
        self.assertEqual(a, b)

    def test_equal_after_round_trip(self) -> None:
        """An instance equals one reconstructed from the same CSV file."""
        if IGNORE_TESTS:
            return
        a = self._make(self.path_a, ROWS)
        b = DataframeSerializer(self.path_a)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
