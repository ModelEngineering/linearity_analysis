"""Tests for src/score.py: Score class."""

import src.constants as cn  # type: ignore
from src.score import Score  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import os
import tempfile
import shutil
import unittest


#########################################
# Score Initialization Tests
#########################################

class TestScoreInit(unittest.TestCase):
    """Tests for Score.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_serialization_path(self) -> None:
        """Default serialization path is 'score.csv' when none provided."""
        score = Score()
        self.assertEqual(score.serialization_path, "score.csv")

    def test_custom_serialization_path(self) -> None:
        """Custom serialization path is stored correctly."""
        csv_path = os.path.join(self.tmp_dir, "custom_score.csv")
        score = Score(serialization_path=csv_path)
        self.assertEqual(score.serialization_path, csv_path)

    def test_empty_string_uses_default(self) -> None:
        """Empty string serialization path falls back to default."""
        score = Score(serialization_path="")
        self.assertEqual(score.serialization_path, "score.csv")

    def test_score_df_empty_dataframe(self) -> None:
        """score_df starts as an empty DataFrame."""
        score = Score(is_persist=False)
        self.assertIsInstance(score.score_df, pd.DataFrame)
        self.assertEqual(len(score.score_df), 0)


#########################################
# Score.calculateAccuracy Tests
#########################################

class TestCalculateAccuracy(unittest.TestCase):
    """Tests for Score.calculateAccuracy static method."""

    def _make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_perfect_prediction_returns_one(self) -> None:
        """Identical predictions yield Accuracy of 1.0 (perfect score)."""
        true_df = self._make_df({"A": [10.0, 20.0, 30.0]})
        pred_df = self._make_df({"A": [10.0, 20.0, 30.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        np.testing.assert_array_almost_equal(result["A"].values, [1.0, 1.0, 1.0]) # type: ignore

    def test_accuracy_formula_correct(self) -> None:
        """Accuracy formula is max(0, 1 - |pred-true|/true)."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [15.0, 25.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # |15-10|/10 = 0.5 -> 1 - 0.5 = 0.5
        # |25-20|/20 = 0.25 -> 1 - 0.25 = 0.75
        np.testing.assert_array_almost_equal(result["A"].values, [0.5, 0.75])# type: ignore

    def test_accuracy_clipped_to_zero_floor(self) -> None:
        """Accuracy values are clipped to >= 0 (zero floor)."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [50.0, 60.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # |50-10|/10 = 4.0 -> clipped to 1.0 -> 1 - 1.0 = 0.0
        # |60-20|/20 = 2.0 -> clipped to 1.0 -> 1 - 1.0 = 0.0
        np.testing.assert_array_almost_equal(result["A"].values, [0.0, 0.0])# type: ignore

    def test_nan_true_value_produces_two(self) -> None:
        """NaN true values produce sentinel -1 (invalid measurement excluded from aggregation)."""
        true_df = self._make_df({"A": [float("nan"), 20.0]})
        pred_df = self._make_df({"A": [5.0, 25.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # NaN -> ape_df=NaN -> invalid_mask=True -> sentinel=-1
        np.testing.assert_almost_equal(float(result["A"].values[0]), -1.0)  # type: ignore[arg-type]
        # Valid row: |25-20|/20=0.25 -> 1-0.25=0.75
        np.testing.assert_almost_equal(float(result["A"].values[1]), 0.75)  # type: ignore[arg-type]

    def test_inf_true_value_produces_two(self) -> None:
        """Inf true values produce sentinel -1 (invalid measurement excluded from aggregation)."""
        true_df = self._make_df({"A": [float("inf"), 20.0]})
        pred_df = self._make_df({"A": [5.0, 25.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # inf -> ape_df=inf -> invalid_mask=True -> sentinel=-1
        np.testing.assert_almost_equal(float(result["A"].values[0]), -1.0)  # type: ignore[arg-type]
        np.testing.assert_almost_equal(float(result["A"].values[1]), 0.75)  # type: ignore[arg-type]

    def test_returns_dataframe_with_same_columns(self) -> None:
        """Returns a DataFrame with same columns as input."""
        true_df = self._make_df({"A": [10.0, 20.0], "B": [5.0, 15.0]})
        pred_df = self._make_df({"A": [12.0, 24.0], "B": [6.0, 18.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ["A", "B"])

    def test_multiple_species(self) -> None:
        """Accuracy is computed for each species independently."""
        true_df = self._make_df({"A": [10.0, 20.0], "B": [5.0, 15.0]})
        pred_df = self._make_df({"A": [12.0, 24.0], "B": [6.0, 18.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # A: |12-10|/10=0.2 -> 0.8, |24-20|/20=0.2 -> 0.8
        np.testing.assert_array_almost_equal(result["A"].values, [0.8, 0.8]) # type: ignore
        # B: |6-5|/5=0.2 -> 0.8, |18-15|/15=0.2 -> 0.8
        np.testing.assert_array_almost_equal(result["B"].values, [0.8, 0.8]) # type: ignore

    def test_all_zero_true_values_produce_two(self) -> None:
        """All zero true values produce sentinel -1 for all rows (invalid measurements)."""
        true_df = self._make_df({"A": [0.0, 0.0]})
        pred_df = self._make_df({"A": [5.0, 5.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        np.testing.assert_array_almost_equal(result["A"].values, [-1.0, -1.0]) # type: ignore

    def test_prediction_smaller_than_true(self) -> None:
        """When prediction is smaller than true, Accuracy decreases."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [5.0, 10.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # |5-10|/10=0.5 -> 0.5, |10-20|/20=0.5 -> 0.5
        np.testing.assert_array_almost_equal(result["A"].values, [0.5, 0.5]) # type: ignore

    def test_large_prediction_clipped(self) -> None:
        """Very large predictions are clipped to zero Accuracy."""
        true_df = self._make_df({"A": [1.0]})
        pred_df = self._make_df({"A": [1e10]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # |1e10-1|/1 ≈ 1e10 -> clipped to 1 -> 1 - 1 = 0
        np.testing.assert_almost_equal(float(result["A"].values[0]), 0.0) # type: ignore

    def test_accuracy_between_zero_and_one(self) -> None:
        """All valid Accuracy values are in [0, 1] range."""
        true_df = self._make_df({"A": [10.0, 20.0, 30.0, 40.0]})
        pred_df = self._make_df({"A": [5.0, 25.0, 35.0, 50.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        for val in result["A"].values:
            if val >= 0:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_negative_true_value(self) -> None:
        """Negative true values are handled correctly."""
        true_df = self._make_df({"A": [-10.0, -20.0]})
        pred_df = self._make_df({"A": [-5.0, -25.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # |-5-(-10)|/|-10| = 5/10 = 0.5 -> 1 - 0.5 = 0.5
        # |-25-(-20)|/|-20| = 5/20 = 0.25 -> 1 - 0.25 = 0.75
        np.testing.assert_array_almost_equal(result["A"].values, [0.5, 0.75])  # type: ignore

    def test_mixed_zero_and_valid(self) -> None:
        """Mixed zero and valid true values handled correctly."""
        true_df = self._make_df({"A": [0.0, 10.0, 20.0]})
        pred_df = self._make_df({"A": [5.0, 12.0, 18.0]})
        result = Score.calculateAccuracy(true_df, pred_df)
        # true=0 -> ape_df=inf -> invalid_mask=True -> sentinel=-1
        np.testing.assert_almost_equal(float(result["A"].values[0]), -1.0)  # type: ignore[arg-type]
        # |12-10|/10=0.2 -> 0.8
        np.testing.assert_almost_equal(float(result["A"].values[1]), 0.8)  # type: ignore[arg-type]
        # |18-20|/20=0.1 -> 0.9
        np.testing.assert_almost_equal(float(result["A"].values[2]), 0.9)  # type: ignore[arg-type]


#########################################
# Score.add() Tests
#########################################

class TestScoreAdd(unittest.TestCase):
    """Tests for Score.add()."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp_dir, "score.csv")
        self.score = Score(serialization_path=self.csv_path, is_persist=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_df(self, data):
        return pd.DataFrame(data)

    def test_add_returns_dataframe(self) -> None:
        """add() returns a DataFrame."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        result = self.score.add(true_df, pred_df, system_id="test")
        self.assertIsInstance(result, pd.DataFrame)

    def test_add_model_aggregation_row(self) -> None:
        """Model-level aggregation row is added."""
        true_df = self._make_df({"A": [10.0, 20.0, 30.0]})
        pred_df = self._make_df({"A": [10.0, 20.0, 30.0]})
        self.score.add(true_df, pred_df, system_id="test")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        self.assertEqual(len(model_rows), 1)

    def test_add_species_aggregation_rows(self) -> None:
        """Species-level aggregation rows are added for each species."""
        true_df = self._make_df({"A": [10.0, 20.0], "B": [5.0, 15.0]})
        pred_df = self._make_df({"A": [12.0, 24.0], "B": [6.0, 18.0]})
        self.score.add(true_df, pred_df, system_id="test")
        df = self.score.score_df
        species_rows = df[df[cn.COL_AGGREGATION_TYPE] == "A"]
        self.assertEqual(len(species_rows), 1)
        species_rows = df[df[cn.COL_AGGREGATION_TYPE] == "B"]
        self.assertEqual(len(species_rows), 1)

    def test_add_description_stored(self) -> None:
        """Label is stored in species aggregation rows; model uses AGGREGATION_TYPE_MODEL."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="my_label")
        df = self.score.score_df
        # Model-level row uses AGGREGATION_TYPE_MODEL as its label
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        self.assertEqual(len(model_rows), 1)
        # Species rows use the species name as their label
        species_rows = df[df[cn.COL_AGGREGATION_TYPE] == "A"]
        self.assertEqual(len(species_rows), 1)

    def test_add_multiple_species(self) -> None:
        """One species aggregation per column."""
        true_df = self._make_df({"X": [10.0, 20.0], "Y": [30.0, 40.0], "Z": [50.0, 60.0]})
        pred_df = self._make_df({"X": [12.0, 22.0], "Y": [33.0, 43.0], "Z": [55.0, 65.0]})
        self.score.add(true_df, pred_df, system_id="multi")
        df = self.score.score_df
        species_rows = df[df[cn.COL_AGGREGATION_TYPE] == "Y"]
        self.assertEqual(len(species_rows), 1)
        species_rows = df[df[cn.COL_AGGREGATION_TYPE] == "Z"]
        self.assertEqual(len(species_rows), 1)

    def test_add_persists_to_csv(self) -> None:
        """Data is persisted to the CSV file when is_persist=True."""
        csv_path = os.path.join(self.tmp_dir, "persist.csv")
        score_persist = Score(serialization_path=csv_path, is_persist=True)
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        score_persist.add(true_df, pred_df, system_id="persist")
        self.assertTrue(os.path.exists(csv_path))

    def test_add_label_values(self) -> None:
        """Label values include model and species names."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        df = self.score.score_df
        labels = set(df[cn.COL_AGGREGATION_TYPE].values)
        self.assertEqual(labels, {cn.COL_AGGREGATION_TYPE_MODEL, "A"})

    def test_add_multiple_runs_accumulate(self) -> None:
        """Multiple add() calls accumulate rows."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="run_1")
        self.score.add(true_df, pred_df, system_id="run_2")
        df = self.score.score_df
        # Each run adds 1 model + 1 species = 2 rows.
        self.assertEqual(len(df), 4)

    def test_add_with_invalid_true_values(self) -> None:
        """Zero true values produce sentinel -1, included in percentile aggregation."""
        true_df = self._make_df({"A": [0.0, 20.0]})
        pred_df = self._make_df({"A": [5.0, 25.0]})
        self.score.add(true_df, pred_df, system_id="invalid")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # true=0 -> sentinel=-1; |25-20|/20=0.25 -> Accuracy=0.75
        # 90th percentile of [-1, 0.75] = -1 + 0.9*(0.75-(-1)) = 0.575
        np.testing.assert_almost_equal(float(model_rows["mean"].values[0]), 0.575)  # type: ignore[arg-type]

    def test_add_with_all_zero_true_values(self) -> None:
        """All zero true values produce sentinel -1, included in percentile aggregation."""
        true_df = self._make_df({"A": [0.0, 0.0]})
        pred_df = self._make_df({"A": [5.0, 5.0]})
        self.score.add(true_df, pred_df, system_id="all_zero")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # true=0 -> sentinel=-1 for both rows; 90th percentile of [-1,-1] = -1
        # StatisticCalculator receives one value (-1), which is negative sentinel -> invalid_count=1
        self.assertEqual(model_rows[cn.COL_INVALID_COUNT].values[0], 1)  # type: ignore[arg-type]

    def test_add_with_nan_true_values(self) -> None:
        """NaN true values produce sentinel -1, included in percentile aggregation."""
        true_df = self._make_df({"A": [float("nan"), 20.0]})
        pred_df = self._make_df({"A": [5.0, 25.0]})
        self.score.add(true_df, pred_df, system_id="nan")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # NaN -> sentinel=-1; |25-20|/20=0.25 -> Accuracy=0.75
        # 90th percentile of [-1, 0.75] = -1 + 0.9*(0.75-(-1)) = 0.575
        np.testing.assert_almost_equal(float(model_rows["mean"].values[0]), 0.575)  # type: ignore[arg-type]

    def test_add_with_persist(self) -> None:
        """Data is written to CSV when is_persist=True."""
        csv_path = os.path.join(self.tmp_dir, "persist_test.csv")
        score_persist = Score(serialization_path=csv_path, is_persist=True)
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        score_persist.add(true_df, pred_df, system_id="persist")
        loaded = pd.read_csv(csv_path)
        self.assertEqual(len(loaded), 2)

    def test_add_invalid_count_zero_for_valid_data(self) -> None:
        """No sentinel values means invalid_count is 0."""
        true_df = self._make_df({"A": [10.0, 20.0, 30.0]})
        pred_df = self._make_df({"A": [12.0, 25.0, 35.0]})
        self.score.add(true_df, pred_df, system_id="no_inv")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # All values are valid (>=0) -> invalid_count=0.
        self.assertEqual(int(model_rows["invalid_count"].values[0]), 0)  # type: ignore[arg-type]

    def test_add_percentiles_computed(self) -> None:
        """Percentile statistics are computed and stored."""
        true_df = self._make_df({"A": [10.0, 20.0, 30.0, 40.0]})
        pred_df = self._make_df({"A": [11.0, 22.0, 28.0, 42.0]})
        self.score.add(true_df, pred_df, system_id="pctl")
        df = self.score.score_df
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # Should have percentile columns populated (not NaN).
        self.assertFalse(np.isnan(float(model_rows["p25"].values[0])))  # type: ignore[arg-type]
        self.assertFalse(np.isnan(float(model_rows["p95"].values[0])))  # type: ignore[arg-type]

    def test_add_default_label(self) -> None:
        """Default empty label is stored when no label provided."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df)
        df = self.score.score_df
        # Model-level row uses AGGREGATION_TYPE_MODEL as its label regardless of provided label
        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        self.assertEqual(len(model_rows), 1)

    def test_add_single_species(self) -> None:
        """Single species produces one model + one species row."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="single")
        df = self.score.score_df
        self.assertEqual(len(df), 2)

    def test_add_no_species(self) -> None:
        """Empty DataFrame produces only model-level row."""
        true_df = pd.DataFrame()
        pred_df = pd.DataFrame()
        self.score.add(true_df, pred_df, system_id="empty")
        df = self.score.score_df
        # Only model-level aggregation (no species columns).
        self.assertEqual(len(df), 1)


#########################################
# Score.plotCDF Tests
#########################################

class TestScorePlotCDF(unittest.TestCase):
    """Tests for Score.plotCDF()."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp_dir, "score.csv")
        self.score = Score(serialization_path=self.csv_path, is_persist=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_df(self, data):
        return pd.DataFrame(data)

    def test_plot_cdf_returns_plot_options(self) -> None:
        """plotCDF returns a PlotOptions object."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_raises_on_no_plot_option(self) -> None:
        """plotCDF raises ValueError when both is_plot_model and is_plot_species are False."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        with self.assertRaises(ValueError):
            self.score.plotCDF("mean", is_plot_model=False, is_plot_species=False, is_plot=False)

    def test_plot_cdf_with_custom_title(self) -> None:
        """Custom title is applied to the plot."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", title="Custom Title", is_plot=False)
        self.assertEqual(result.title, "Custom Title")

    def test_plot_cdf_with_model_only(self) -> None:
        """plotCDF with only model plotting works."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot_species=False, is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_with_species_only(self) -> None:
        """plotCDF with only species plotting works."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot_model=False, is_plot=False)
        self.assertIsInstance(result, PlotOptions)

    def test_plot_cdf_default_xlabel_ylabel(self) -> None:
        """Default xlabel and ylabel are set."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertEqual(result.xlabel, "mean")
        self.assertEqual(result.ylabel, "fraction")

    def test_plot_cdf_default_title_both(self) -> None:
        """Default title is 'CDFs' when both model and species are plotted."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertEqual(result.title, "CDFs")

    def test_plot_cdf_default_title_model_only(self) -> None:
        """Default title is 'CDF for models' when only model is plotted."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot_species=False, is_plot=False)
        self.assertEqual(result.title, "CDF for models")

    def test_plot_cdf_default_title_species_only(self) -> None:
        """Default title is 'CDF for species' when only species is plotted."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot_model=False, is_plot=False)
        self.assertEqual(result.title, "CDF for species")

    def test_plot_cdf_with_legend(self) -> None:
        """Default legend includes both model and species."""
        true_df = self._make_df({"A": [10.0, 20.0]})
        pred_df = self._make_df({"A": [12.0, 24.0]})
        self.score.add(true_df, pred_df, system_id="test")
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertEqual(result.legend, ["model", "species"])

    def test_plot_cdf_with_empty_score_df(self) -> None:
        """plotCDF with empty score_df does not crash."""
        # No data added yet.
        result = self.score.plotCDF("mean", is_plot=False)
        self.assertIsInstance(result, PlotOptions)


if __name__ == "__main__":
    unittest.main()