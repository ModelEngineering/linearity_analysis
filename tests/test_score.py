"""Tests for src/score.py: StatisticAccumulator and Score classes."""

import src.constants as cn  # type: ignore
from src.score import _StatisticAccumulator, Score, MEAN, MIN, MAX, COUNT  # type: ignore

import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import shutil  # type: ignore
import sys
import tempfile  # type: ignore
import unittest




#########################################
# StatisticAccumulator Tests
#########################################

class TestStatisticAccumulatorInit(unittest.TestCase):
    """Tests for _StatisticAccumulator.__init__."""

    def test_default_initialization(self) -> None:
        """Default initialization creates empty lists for all statistics."""
        acc = _StatisticAccumulator()
        for stat in _StatisticAccumulator.STATISTICS:
            self.assertIn(stat, acc.statistic_dct)
            self.assertEqual(acc.statistic_dct[stat], [])
        self.assertIn(cn.AGGREGATION_TYPE, acc.statistic_dct)
        self.assertIn(cn.DESCRIPTION, acc.statistic_dct)

    def test_statistics_list_contains_expected_keys(self) -> None:
        """STATISTICS list contains all expected percentile and summary keys."""
        expected = ["mean", "min", "max", "count", "invalid_count",
                "p25", "p30", "p50", "p80", "p95", "p99"]
        self.assertEqual(_StatisticAccumulator.STATISTICS, expected)


class TestStatisticAccumulatorAdd(unittest.TestCase):
    """Tests for _StatisticAccumulator.add()."""

    def setUp(self) -> None:
        self.acc = _StatisticAccumulator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_basic_mean_min_max(self) -> None:
        """Basic statistics are computed correctly for valid data."""
        arr = self._make_arr([1.0, 2.0, 3.0, 4.0, 5.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertEqual(len(dct[cn.DESCRIPTION]), 1)
        self.assertAlmostEqual(dct[MEAN][0], 3.0)
        self.assertAlmostEqual(dct[MIN][0], 1.0)
        self.assertAlmostEqual(dct[MAX][0], 5.0)

    def test_add_count_is_valid_values(self) -> None:
        """count reflects number of valid (non-sentinel) values."""
        arr = self._make_arr([1.0, -1.0, 3.0, -1.0, 5.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertEqual(dct[COUNT][0], 3)

    def test_add_invalid_count_tracked(self) -> None:
        """invalid_count reflects number of sentinel (-1) values excluded."""
        arr = self._make_arr([1.0, -1.0, 3.0, -1.0, 5.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertEqual(dct["invalid_count"][0], 2)

    def test_add_empty_array_produces_row(self) -> None:
        """Empty array still produces a row with zeroed values."""
        arr = self._make_arr([])
        self.acc.add(arr, label="empty", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertEqual(len(dct[cn.DESCRIPTION]), 1)
        self.assertAlmostEqual(dct[MEAN][0], 0.0)
        self.assertAlmostEqual(dct[MIN][0], 0.0)
        self.assertAlmostEqual(dct[MAX][0], 0.0)
        self.assertEqual(dct[COUNT][0], 0)

    def test_add_all_invalid_produces_row(self) -> None:
        """All sentinel values still produce a row with zeroed values."""
        arr = self._make_arr([-1.0, -1.0, -1.0])
        self.acc.add(arr, label="all_invalid", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertEqual(len(dct[cn.DESCRIPTION]), 1)
        self.assertAlmostEqual(dct[MEAN][0], 0.0)
        self.assertEqual(dct[COUNT][0], 0)
        self.assertEqual(dct["invalid_count"][0], 3)

    def test_add_nan_values_counted_as_invalid(self) -> None:
        """NaN values are < 0 in numpy comparison so counted as invalid sentinel."""
        arr = self._make_arr([1.0, float('nan'), 3.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        # NaN >= 0 returns False in numpy, so excluded from valid (count=2, invalid_count=1).
        self.assertEqual(dct[COUNT][0], 2)
        self.assertEqual(dct["invalid_count"][0], 1)

    def test_add_inf_values_counted_as_valid(self) -> None:
        """Inf values are >= 0 so counted as valid, then replaced with LARGE_VAL."""
        arr = self._make_arr([1.0, float('inf'), 3.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        # inf >= 0 so count includes it (3 valid values).
        self.assertEqual(dct[COUNT][0], 3)

    def test_add_large_values_counted_as_valid(self) -> None:
        """Values > 1e6 are counted as valid, then replaced with LARGE_VAL."""
        arr = self._make_arr([1.0, 1e9, 3.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        # 1e9 >= 0 so count includes it (3 valid values).
        self.assertEqual(dct[COUNT][0], 3)

    def test_add_percentiles(self) -> None:
        """Percentile values are computed correctly."""
        arr = self._make_arr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        # p25 should be around 3.25 (linear interpolation).
        self.assertAlmostEqual(dct["p25"][0], 3.25, places=1)
        # p95 should be around 9.55.
        self.assertAlmostEqual(dct["p95"][0], 9.55, places=1)

    def test_add_multiple_calls_accumulate(self) -> None:
        """Multiple add() calls accumulate rows."""
        self.acc.add(self._make_arr([1.0, 2.0]), label="a", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        self.acc.add(self._make_arr([3.0, 4.0]), label="b", aggregation_type="A")
        dct = self.acc.statistic_dct
        self.assertEqual(len(dct[cn.DESCRIPTION]), 2)
        self.assertEqual(dct[cn.DESCRIPTION][0], "a")
        self.assertEqual(dct[cn.DESCRIPTION][1], "b")

    def test_add_flattens_2d_array(self) -> None:
        """2D arrays are flattened before aggregation."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertAlmostEqual(dct[MEAN][0], 2.5)

    def test_add_label_and_aggregation_stored(self) -> None:
        """Label and aggregation_type are stored correctly."""
        species_type = "A"
        arr = self._make_arr([1.0, 2.0])
        self.acc.add(arr, label="my_label", aggregation_type=species_type)
        dct = self.acc.statistic_dct
        self.assertEqual(dct[cn.DESCRIPTION][0], "my_label")
        self.assertEqual(dct[cn.AGGREGATION_TYPE][0], species_type)

    def test_add_single_value(self) -> None:
        """Single value produces correct statistics."""
        arr = self._make_arr([42.0])
        self.acc.add(arr, label="single", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        self.assertAlmostEqual(dct[MEAN][0], 42.0)
        self.assertAlmostEqual(dct[MIN][0], 42.0)
        self.assertAlmostEqual(dct[MAX][0], 42.0)
        self.assertEqual(dct[COUNT][0], 1)

    def test_add_negative_inf_counted_as_invalid(self) -> None:
        """Negative inf values are < 0 so counted as invalid sentinel."""
        arr = self._make_arr([1.0, float('-inf'), 3.0])
        self.acc.add(arr, label="test", aggregation_type=cn.AGGREGATION_TYPE_MODEL)
        dct = self.acc.statistic_dct
        # -inf < 0 so excluded from valid (count=2, invalid_count=1).
        self.assertEqual(dct[COUNT][0], 2)
        self.assertEqual(dct["invalid_count"][0], 1)


#########################################
# Score Initialization Tests
#########################################

class TestScoreInit(unittest.TestCase):
    """Tests for Score.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_serialization_path(self) -> None:
        """Default serialization path is used when none provided."""
        score = Score()
        self.assertEqual(score.serialization_path, "score.csv")

    def test_custom_serialization_path(self) -> None:
        """Custom serialization path is stored correctly."""
        score = Score(serialization_path=self.tmp_path)
        self.assertEqual(score.serialization_path, self.tmp_path)

    def test_empty_path_uses_default(self) -> None:
        """Empty string serialization path falls back to default."""
        score = Score(serialization_path="")
        self.assertEqual(score.serialization_path, "score.csv")

    def test_statistic_accumulator_initialized(self) -> None:
        """StatisticAccumulator is created on init."""
        score = Score()
        self.assertIsInstance(score.statistic_accumulator, _StatisticAccumulator)


#########################################
# Score.calculateMAPE Tests
#########################################

class TestCalculateMAPE(unittest.TestCase):
    """Tests for Score.calculateMAPE static method."""

    def _make_df(self, data):
        return pd.DataFrame(data)

    def test_perfect_prediction_returns_one(self) -> None:
        """Identical predictions yield MAPE of 1.0 (perfect score)."""
        true_df = self._make_df({'A': [10.0, 20.0, 30.0]})
        pred_df = self._make_df({'A': [10.0, 20.0, 30.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        np.testing.assert_array_almost_equal(result['A'].values, [1.0, 1.0, 1.0])  # type: ignore

    def test_mape_formula_correct(self) -> None:
        """MAPE formula is max(0, 1 - |pred-true|/true)."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [15.0, 25.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # |15-10|/10 = 0.5 -> 1 - 0.5 = 0.5
        # |25-20|/20 = 0.25 -> 1 - 0.25 = 0.75
        np.testing.assert_array_almost_equal(result['A'].values, [0.5, 0.75])  # type: ignore

    def test_mape_clipped_to_zero_floor(self) -> None:
        """MAPE values are clipped to >= 0 (zero floor)."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [50.0, 60.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # |50-10|/10 = 4.0 -> clipped to 1.0 -> 1 - 1.0 = 0.0
        # |60-20|/20 = 2.0 -> clipped to 1.0 -> 1 - 1.0 = 0.0
        np.testing.assert_array_almost_equal(result['A'].values, [0.0, 0.0])  # type: ignore

    def test_zero_true_value_produces_mape_two(self) -> None:
        """Zero true values produce ARE=inf -> mask catches (clip preserves inf) -> MAPE = 2.0."""
        true_df = self._make_df({'A': [10.0, 0.0, 30.0]})
        pred_df = self._make_df({'A': [12.0, 5.0, 36.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # true=0 -> ARE=inf -> clip preserves inf -> mask catches (not finite) -> set to -1 -> MAPE=2
        self.assertEqual(result['A'].values[1], 2.0)
        # |36-30|/30 = 0.2 -> 1 - 0.2 = 0.8
        np.testing.assert_almost_equal(result['A'].values[2], 0.8)  # type: ignore

    def test_nan_true_value_produces_mape_two(self) -> None:
        """NaN true values produce ARE=NaN -> mask catches -> MAPE = 2.0."""
        true_df = self._make_df({'A': [float('nan'), 20.0]})
        pred_df = self._make_df({'A': [5.0, 25.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # NaN -> ARE=NaN -> clip keeps NaN -> mask catches -> set to -1 -> MAPE=1-(-1)=2
        np.testing.assert_almost_equal(result['A'].values[0], 2.0)  # type: ignore
        # Valid row: |25-20|/20=0.25 -> 1-0.25=0.75
        np.testing.assert_almost_equal(result['A'].values[1], 0.75)  # type: ignore

    def test_inf_true_value_produces_mape_two(self) -> None:
        """Inf true values produce ARE=NaN (inf/inf=nan in numpy) -> MAPE = 2.0."""
        true_df = self._make_df({'A': [float('inf'), 20.0]})
        pred_df = self._make_df({'A': [5.0, 25.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # (5-inf)/inf -> nan in numpy -> same path as NaN -> MAPE=2
        np.testing.assert_almost_equal(result['A'].values[0], 2.0)  # type: ignore
        np.testing.assert_almost_equal(result['A'].values[1], 0.75)  # type: ignore

    def test_returns_dataframe_with_same_columns(self) -> None:
        """Returns a DataFrame with same columns as input."""
        true_df = self._make_df({'A': [10.0, 20.0], 'B': [5.0, 15.0]})
        pred_df = self._make_df({'A': [12.0, 24.0], 'B': [6.0, 18.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['A', 'B'])

    def test_multiple_species(self) -> None:
        """MAPE is computed for each species independently."""
        true_df = self._make_df({'A': [10.0, 20.0], 'B': [5.0, 15.0]})
        pred_df = self._make_df({'A': [12.0, 24.0], 'B': [6.0, 18.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # A: |12-10|/10=0.2 -> 0.8, |24-20|/20=0.2 -> 0.8
        np.testing.assert_array_almost_equal(result['A'].values, [0.8, 0.8])  # type: ignore
        # B: |6-5|/5=0.2 -> 0.8, |18-15|/15=0.2 -> 0.8
        np.testing.assert_array_almost_equal(result['B'].values, [0.8, 0.8])  # type: ignore

    def test_all_zero_true_values_produce_mape_two(self) -> None:
        """All zero true values produce ARE=inf -> mask catches -> MAPE = 2.0."""
        true_df = self._make_df({'A': [0.0, 0.0]})
        pred_df = self._make_df({'A': [5.0, 5.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # true=0 -> ARE=inf -> clip preserves inf -> mask catches -> set to -1 -> MAPE=2
        np.testing.assert_array_almost_equal(result['A'].values, [2.0, 2.0])  # type: ignore

    def test_prediction_smaller_than_true(self) -> None:
        """When prediction is smaller than true, MAPE decreases."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [5.0, 10.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        # |5-10|/10=0.5 -> 0.5, |10-20|/20=0.5 -> 0.5
        np.testing.assert_array_almost_equal(result['A'].values, [0.5, 0.5])  # type: ignore

    def test_large_prediction_clipped(self) -> None:
        """Very large predictions are clipped to zero MAPE."""
        true_df = self._make_df({'A': [1.0]})
        pred_df = self._make_df({'A': [1e10]})
        result = Score.calculateMAPE(true_df, pred_df)
        # |1e10-1|/1 ≈ 1e10 -> clipped to 1 -> 1 - 1 = 0
        np.testing.assert_almost_equal(result['A'].values[0], 0.0)  # type: ignore

    def test_mape_between_zero_and_one(self) -> None:
        """All valid MAPE values are in [0, 1] range."""
        true_df = self._make_df({'A': [10.0, 20.0, 30.0, 40.0]})
        pred_df = self._make_df({'A': [5.0, 25.0, 35.0, 50.0]})
        result = Score.calculateMAPE(true_df, pred_df)
        for val in result['A'].values:
            if val >= 0:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


#########################################
# Score.add() Tests
#########################################

class TestScoreAdd(unittest.TestCase):
    """Tests for Score.add()."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'score.csv')
        self.score = Score(serialization_path=self.tmp_path, is_initialize=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_df(self, data):
        return pd.DataFrame(data)

    def test_add_model_aggregation_row(self) -> None:
        """Model-level aggregation row is added."""
        true_df = self._make_df({'A': [10.0, 20.0, 30.0]})
        pred_df = self._make_df({'A': [10.0, 20.0, 30.0]})
        self.score.add(true_df, pred_df, label="test")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        self.assertEqual(len(model_rows), 1)

    def test_add_species_aggregation_rows(self) -> None:
        """Species-level aggregation rows are added for each species."""
        true_df = self._make_df({'A': [10.0, 20.0], 'B': [5.0, 15.0]})
        pred_df = self._make_df({'A': [12.0, 24.0], 'B': [6.0, 18.0]})
        self.score.add(true_df, pred_df, label="test")
        df = self.score.score_df
        species_rows = df[df[cn.AGGREGATION_TYPE] == "A"]
        self.assertEqual(len(species_rows), 1)
        species_rows = df[df[cn.AGGREGATION_TYPE] == "B"]
        self.assertEqual(len(species_rows), 1)

    def test_add_description_stored(self) -> None:
        """Label is stored in all aggregation rows."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="my_label")
        df = self.score.score_df
        for _, row in df.iterrows():
            self.assertEqual(row[cn.DESCRIPTION], "my_label")

    def test_add_multiple_species(self) -> None:
        """One species aggregation per column."""
        true_df = self._make_df({'X': [10.0, 20.0], 'Y': [30.0, 40.0], 'Z': [50.0, 60.0]})
        pred_df = self._make_df({'X': [12.0, 22.0], 'Y': [33.0, 43.0], 'Z': [55.0, 65.0]})
        self.score.add(true_df, pred_df, label="multi")
        df = self.score.score_df
        species_rows = df[df[cn.AGGREGATION_TYPE] == "Y"]
        self.assertEqual(len(species_rows), 1)
        species_rows = df[df[cn.AGGREGATION_TYPE] == "Z"]
        self.assertEqual(len(species_rows), 1)

    def test_add_persists_to_csv(self) -> None:
        """Data is persisted to the CSV file."""
        csv_path = os.path.join(self.tmp_dir, 'persist.csv')
        score_persist = Score(
                serialization_path=csv_path, is_initialize=False, is_persist=True)
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        score_persist.add(true_df, pred_df, label="persist")
        self.assertTrue(os.path.exists(csv_path))

    def test_add_returns_dataframe(self) -> None:
        """add() returns the full score DataFrame."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        result = self.score.add(true_df, pred_df, label="test")
        self.assertIsInstance(result, pd.DataFrame)

    def test_add_aggregation_type_values(self) -> None:
        """aggregation_type values are 'model' and 'species'."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        df = self.score.score_df
        agg_types = set(df[cn.AGGREGATION_TYPE].values)
        self.assertEqual(agg_types, {cn.AGGREGATION_TYPE_MODEL, "A"})

    def test_add_multiple_runs_accumulate(self) -> None:
        """Multiple add() calls accumulate rows."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="run_1")
        self.score.add(true_df, pred_df, label="run_2")
        df = self.score.score_df
        # Each run adds 1 model + 1 species = 2 rows.
        self.assertEqual(len(df), 4)

    def test_add_with_invalid_true_values(self) -> None:
        """Zero true values produce ARE=inf -> mask catches -> MAPE=2.0, included in aggregation."""
        true_df = self._make_df({'A': [0.0, 20.0]})
        pred_df = self._make_df({'A': [5.0, 25.0]})
        self.score.add(true_df, pred_df, label="invalid")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        # true=0 -> MAPE=2.0; |25-20|/20=0.25 -> MAPE=0.75
        # mean=(2.0+0.75)/2=1.375
        np.testing.assert_almost_equal(model_rows['mean'].values[0], 1.375)  # type: ignore

    def test_add_with_all_zero_true_values(self) -> None:
        """All zero true values produce ARE=inf -> mask catches -> MAPE=2.0 for all rows."""
        true_df = self._make_df({'A': [0.0, 0.0]})
        pred_df = self._make_df({'A': [5.0, 5.0]})
        self.score.add(true_df, pred_df, label="all_zero")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        # true=0 -> MAPE=2.0 for both rows -> mean=2.0
        np.testing.assert_almost_equal(model_rows['mean'].values[0], 2.0)  # type: ignore

    def test_add_with_nan_true_values(self) -> None:
        """NaN true values produce ARE=NaN -> mask catches -> MAPE=2.0, included in aggregation."""
        true_df = self._make_df({'A': [float('nan'), 20.0]})
        pred_df = self._make_df({'A': [5.0, 25.0]})
        self.score.add(true_df, pred_df, label="nan")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        # NaN -> MAPE=2.0; |25-20|/20=0.25 -> MAPE=0.75
        # mean=(2.0+0.75)/2=1.375
        np.testing.assert_almost_equal(model_rows['mean'].values[0], 1.375)  # type: ignore

    def test_add_with_persist(self) -> None:
        """Data is written to CSV when is_persist=True."""
        csv_path = os.path.join(self.tmp_dir, 'persist_test.csv')
        score_persist = Score(
                serialization_path=csv_path, is_initialize=False, is_persist=True)
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        score_persist.add(true_df, pred_df, label="persist")
        loaded = pd.read_csv(csv_path)
        self.assertEqual(len(loaded), 2)

    def test_add_invalid_count_zero_for_valid_data(self) -> None:
        """No sentinel values means invalid_count is 0."""
        true_df = self._make_df({'A': [10.0, 20.0, 30.0]})
        pred_df = self._make_df({'A': [12.0, 25.0, 35.0]})
        self.score.add(true_df, pred_df, label="no_inv")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        # All values are valid (>=0) -> invalid_count=0.
        self.assertEqual(model_rows['invalid_count'].values[0], 0)

    def test_add_percentiles_computed(self) -> None:
        """Percentile statistics are computed and stored."""
        true_df = self._make_df({'A': [10.0, 20.0, 30.0, 40.0]})
        pred_df = self._make_df({'A': [11.0, 22.0, 28.0, 42.0]})
        self.score.add(true_df, pred_df, label="pctl")
        df = self.score.score_df
        model_rows = df[df[cn.AGGREGATION_TYPE] == cn.AGGREGATION_TYPE_MODEL]
        # Should have percentile columns populated (not NaN).
        self.assertFalse(np.isnan(model_rows['p25'].values[0]))   # type: ignore
        self.assertFalse(np.isnan(model_rows['p95'].values[0]))     # type: ignore

    def test_add_default_label(self) -> None:
        """Default empty label is stored when no label provided."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df)
        df = self.score.score_df
        for _, row in df.iterrows():
            self.assertEqual(row[cn.DESCRIPTION], "")


#########################################
# Score.plotCDF Tests
#########################################

class TestScorePlotCDF(unittest.TestCase):
    """Tests for Score.plotCDF()."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'score.csv')
        self.score = Score(serialization_path=self.tmp_path, is_initialize=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_df(self, data):
        return pd.DataFrame(data)

    def test_plot_cdf_returns_plot_options(self) -> None:
        """plotCDF returns a PlotOptions object."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        from src.plot_options import PlotOptions  # type: ignore
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_raises_on_no_plot_option(self) -> None:
        """plotCDF raises ValueError when both is_plot_model and is_plot_species are False."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        with self.assertRaises(ValueError):
            self.score.plotCDF("mean", is_plot_model=False, is_plot_species=False, is_plot=False)

    def test_plot_cdf_with_custom_title(self) -> None:
        """Custom title is applied to the plot."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        from src.plot_options import PlotOptions  # type: ignore
        result = self.score.plotCDF("mean", title="Custom Title", is_plot=False)
        self.assertEqual(result.title, "Custom Title")

    def test_plot_cdf_with_model_only(self) -> None:
        """plotCDF with only model plotting works."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        from src.plot_options import PlotOptions  # type: ignore
        result = self.score.plotCDF("mean", is_plot_species=False, is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_with_species_only(self) -> None:
        """plotCDF with only species plotting works."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        from src.plot_options import PlotOptions  # type: ignore
        result = self.score.plotCDF("mean", is_plot_model=False, is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_default_xlabel_ylabel(self) -> None:
        """Default xlabel and ylabel are set."""
        true_df = self._make_df({'A': [10.0, 20.0]})
        pred_df = self._make_df({'A': [12.0, 24.0]})
        self.score.add(true_df, pred_df, label="test")
        from src.plot_options import PlotOptions  # type: ignore
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertEqual(result.xlabel, "mean")
        self.assertEqual(result.ylabel, "fraction")


if __name__ == "__main__":
    unittest.main()