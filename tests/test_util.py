import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import util  # noqa: E402
from util import getPSDPredictionDF, codedstrToDict, dictToCodedstr  # noqa: E402
import src.constants as cn  # noqa: E402


def _model_df(max_fractional_reduction: float = 0.001, n: int = 1) -> pd.DataFrame:
    """Return a minimal valid model-level DataFrame."""
    return pd.DataFrame(
        [
            {
                "aggregation_type": "model",
                "max_fractional_reduction": max_fractional_reduction,
                "value": v,
                cn.COL_CHANGEPOINTS: [],
                cn.COL_COUNT: 0,
            }
            for v in range(1, n + 1)
        ]
    )


def _mixed_df(max_fractional_reduction: float = 0.001) -> pd.DataFrame:
    """Return a DataFrame with both species and model rows."""
    return pd.DataFrame(
        [
            {"aggregation_type": "species", "max_fractional_reduction": max_fractional_reduction, "x": 1, cn.COL_CHANGEPOINTS: [], cn.COL_COUNT: 0},
            {"aggregation_type": "model", "max_fractional_reduction": max_fractional_reduction, "x": 2, cn.COL_CHANGEPOINTS: [], cn.COL_COUNT: 0},
        ]
    )


class TestGetPSDPredictionDFFileNotFound(unittest.TestCase):
    """getPSDPredictionDF raises FileNotFoundError when no CSV matches."""

    def test_unknown_maxreduction_raises(self) -> None:
        """A max_fractional_reduction that produces a file-selection string no filename contains raises."""
        with patch("util.os.listdir", return_value=["piecewise_predictions__numpoint_10.csv"]):
            with self.assertRaises(FileNotFoundError):
                getPSDPredictionDF(0.9999)

    def test_unknown_repeat_raises(self) -> None:
        """Passing a repeat value that no filename contains raises."""
        with patch("util.os.listdir", return_value=["piecewise_predictions__numpoint_10.csv"]):
            with self.assertRaises(FileNotFoundError):
                getPSDPredictionDF(0.001, repeat=99999)


class TestGetPSDPredictionDFSingleFile(unittest.TestCase):
    """Single-file match returns a correctly-shaped, filtered DataFrame."""

    def _run_with_fixture(
        self,
        csv_name: str,
        max_fractional_reduction: float = 0.001,
        payload=None,
    ) -> pd.DataFrame:
        if payload is None:
            payload = _model_df(max_fractional_reduction, n=3)
        with patch("util.os.listdir", return_value=[csv_name]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=payload):
            return getPSDPredictionDF(max_fractional_reduction)

    def test_single_match_returns_df(self) -> None:
        """A unique match returns a non-empty DataFrame."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        csv_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        df = self._run_with_fixture(csv_name, max_fractional_reduction=0.001)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_single_match_csv_file_column(self) -> None:
        """The csv_file column is populated with the matching filename."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        expected_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        df = self._run_with_fixture(expected_name, max_fractional_reduction=0.001)
        self.assertIn("csv_file", df.columns.tolist())
        self.assertTrue(all(expected_name in v for v in df["csv_file"].tolist()))

    def test_single_match_aggregation_type_model(self) -> None:
        """All returned rows have aggregation_type == 'model' (species rows are filtered out)."""
        #payload = _mixed_df(0.001)
        #with patch("util.os.listdir", return_value=[csv_name]), \
        #        patch("pandas.read_csv", return_value=payload):
        df = getPSDPredictionDF(0.01)
        self.assertTrue((df["aggregation_type"] == "model").all())


class TestGetPSDPredictionDFAggregation(unittest.TestCase):
    """Concatenation and filtering across multiple matching files."""

    def test_multiple_files_concatenated(self) -> None:
        """When two files match, the result has rows from both."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_100_a__{sel}.csv"
        name2 = f"piecewise_predictions__numpoint_100_b__{sel}.csv"
        df1, df2 = _model_df(0.003, n=5), _model_df(0.003, n=7)
        with patch("util.os.listdir", return_value=[name1, name2]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[df1, df2]):
            result = getPSDPredictionDF(0.003)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 12)  # 5 + 7

    def test_multiple_files_both_filenames_present(self) -> None:
        """All csv_file values are among the two matching filenames."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_100_a__{sel}.csv"
        name2 = f"piecewise_predictions__numpoint_100_b__{sel}.csv"
        with patch("util.os.listdir", return_value=[name1, name2]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[_model_df(0.003, n=1), _model_df(0.003, n=1)]):
            df = getPSDPredictionDF(0.003)
        self.assertEqual(set(df["csv_file"].tolist()), {str(util.PSD_DATA_DIR / name) for name in [name1, name2]})

    def test_multiple_files_no_species_rows(self) -> None:
        """The result contains no species-level rows even when inputs do."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_100_a__{sel}.csv"
        payload = _mixed_df(0.003)
        with patch("util.os.listdir", return_value=[name1]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=payload):
            df = getPSDPredictionDF(0.003)
        self.assertTrue((df["aggregation_type"] == "model").all())


class TestGetPSDPredictionDFRepeatFiltering(unittest.TestCase):
    """Behavior of the repeat= parameter in the revised function."""

    def test_no_repeat_matches_any_file_with_selection(self) -> None:
        """Without repeat, every file containing max_fractional_reduction is matched."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        name_no_repeat = f"piecewise_predictions__numpoint_50_a__{sel}.csv"
        name_with_repeat = f"piecewise_predictions__numpoint_50_b__{sel}__repeat_1.csv"
        with patch("util.os.listdir", return_value=[name_no_repeat, name_with_repeat]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[_model_df(0.001, n=1), _model_df(0.001, n=1)]):
            df = getPSDPredictionDF(0.001)
        self.assertEqual(len(df), 2)  # Both files matched (no repeat filter applied)

    def test_repeat_filters_to_matching_file_only(self) -> None:
        """With repeat=N, only the file containing 'repeat_N' in its name is kept."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        rep_sel = dictToCodedstr({"repeat": 3})
        name_no_repeat = f"piecewise_predictions__numpoint_50_a__{sel}.csv"
        name_with_repeat = f"piecewise_predictions__numpoint_50_b__{sel}__{rep_sel}.csv"
        with patch("util.os.listdir", return_value=[name_no_repeat, name_with_repeat]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=_model_df(0.001, n=2)):
            df = getPSDPredictionDF(0.001, repeat=3)
        self.assertEqual(len(df), 2)


class TestCodedStr2Dict(unittest.TestCase):
    """Tests for util.codedstr2Dict."""

    def test_multiple_key_value_pairs(self) -> None:
        """A string with multiple __-separated key=value pairs is parsed correctly."""
        result = codedstrToDict("key1_val1__key2_val2")
        self.assertEqual(result, {"key1": "val1", "key2": "val2"})

    def test_single_key_value(self) -> None:
        """A single key=value pair produces a one-entry dict."""
        result = codedstrToDict("foo_bar")
        self.assertEqual(result, {"foo": "bar"})

    def test_empty_string_returns_empty_dict(self) -> None:
        """An empty string yields an empty dict."""
        result = codedstrToDict("")
        self.assertEqual(result, {})

    def test_value_containing_underscore_preserved(self) -> None:
        """split('_', 1) keeps underscores in the value intact."""
        result = codedstrToDict("key_sub_val")
        self.assertEqual(result, {"key": "sub_val"})

    def test_part_without_underscore_skipped(self) -> None:
        """Segments without _ are silently ignored."""
        result = codedstrToDict("key1_val1__nounderscore__key2_val2")
        self.assertEqual(result, {"key1": "val1", "key2": "val2"})

    def test_trailing_separator_ignored(self) -> None:
        """Trailing __ produces an empty segment that is skipped."""
        result = codedstrToDict("key1_val1__")
        self.assertEqual(result, {"key1": "val1"})


class TestDictToCodedstr(unittest.TestCase):
    """Tests for the original dictToCodedstr (no convert_strs)."""

    def test_single_key_value(self) -> None:
        """A single key=value pair produces 'key_value'."""
        result = dictToCodedstr({"key": "value"})
        self.assertEqual(result, "key_value")

    def test_multiple_key_values(self) -> None:
        """Multiple pairs are joined by '__' in insertion order."""
        result = dictToCodedstr({"a": "1", "b": "2"})
        self.assertEqual(result, "a_1__b_2")

    def test_empty_dict_returns_empty_string(self) -> None:
        """An empty dict produces an empty string."""
        result = dictToCodedstr({})
        self.assertEqual(result, "")

    def test_numeric_values_use_repr(self) -> None:
        """Ints/floats are rendered via default repr (5, 1.23)."""
        result = dictToCodedstr({"repeat": 5, "threshold": 1.23})
        self.assertEqual(result, "repeat_5__threshold_1.23")

    def test_roundtrip_with_codedstrToDict(self) -> None:
        """Non-numeric string values survive the round-trip (keys without underscores)."""
        original = {"threshold": "alpha", "removal": "beta"}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded)
        self.assertEqual(decoded, original)

    def test_roundtrip_numeric_values_eval_to_int_float(self) -> None:
        """Documents that numeric strings are eval'd back to int/float (a real limitation)."""
        original = {"repeat": 3, "threshold": 0.1}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded)
        self.assertEqual(decoded["repeat"], 3)
        self.assertAlmostEqual(decoded["threshold"], 0.1, places=5)

    def test_roundtrip_succeeds_when_value_has_single_underscore(self) -> None:
        """Values with one _ survive the round-trip via split('_', 1)."""
        original = {"key": "sub_val"}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded)
        self.assertEqual(decoded, original)

    def test_roundtrip_fails_when_value_contains_separator(self) -> None:
        """Values containing '__' are ambiguous and break the round-trip."""
        problematic = {"key": "val__extra"}
        coded = dictToCodedstr(problematic)
        decoded = codedstrToDict(coded)
        self.assertNotEqual(decoded, problematic)


class TestDictToCodedstrConvertStrs(unittest.TestCase):
    """Tests for dictToCodedstr with the convert_strs parameter (revised implementation).

    The source uses f"{value:.1e}" which produces proper scientific notation, e.g. 0.001 → "1.0e-03".
    Passing a non-numeric value raises ValueError because str cannot be formatted with '.1e'.
    """

    def test_convert_strs_formats_value(self) -> None:
        """A key in convert_strs gets its numeric value converted to scientific notation."""
        result = dictToCodedstr(
            {"max_fractional_reduction": 0.001}, convert_strs=["max_fractional_reduction"]
        )
        self.assertEqual(result, "max_fractional_reduction_1e-3")

    def test_convert_strs_with_multiple_keys(self) -> None:
        """Multiple keys in convert_strs all get scientific notation applied."""
        result = dictToCodedstr(
            {"a": 1.5, "b": 2.5}, convert_strs=["a", "b"]
        )
        self.assertEqual(result, "a_1.5e+00__b_2.5e+00")

    def test_convert_strs_non_numeric_value_raises(self) -> None:
        """Non-numeric values raise ValueError because str cannot be formatted with '.1e'."""
        with self.assertRaises(ValueError):
            dictToCodedstr(
                {"key": "value"}, convert_strs=["key"]
            )

    def test_convert_strs_ignores_missing_key(self) -> None:
        """Keys in convert_strs absent from the dict are skipped (no error)."""
        result = dictToCodedstr(
            {"a": 1}, convert_strs=["nonexistent"]
        )
        self.assertEqual(result, "a_1")

    def test_convert_strs_does_not_mutate_input(self) -> None:
        """The defensive copy (dct = dict(dct)) prevents modification of the caller's dict."""
        original = {"key": 0.5}
        original_copy = dict(original)
        result = dictToCodedstr(
            original, convert_strs=["key"]
        )
        self.assertEqual(result, "key_5e-1")
        self.assertEqual(original, original_copy)

class TestGetCSVPaths(unittest.TestCase):
    """Tests for the makeCSVPaths function."""

    def test_makeCSVPaths_returns_list_of_paths(self) -> None:
        """makeCSVPaths returns a list of Path objects."""
        paths = util.makeCSVPaths(max_fractional_reduction=0.001, repeat=1)
        self.assertIsInstance(paths, list)
        self.assertTrue(all(isinstance(p, (os.PathLike, str)) for p in paths))

    def test_makeCSVPaths_empty_list_when_no_files(self) -> None:
        """If no files match the criteria, an empty list is returned."""
        with patch("util.os.listdir", return_value=[]):
            paths = util.makeCSVPaths(max_fractional_reduction=0.9999, repeat=99999)
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
