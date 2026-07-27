"""Tests for R2Score and R2ScoreInfo."""

import math
import os
import shutil  # type: ignore
import sys
import tempfile  # type: ignore
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2_score import R2Score, R2ScoreInfo, R2_METRICS, SERIALIZATION_PATH  # type: ignore
from score import Score  # type: ignore


#########################################
def _make_true_df() -> pd.DataFrame:
    """Returns a simple true timecourse DataFrame (3 timepoints x 2 species)."""
    return pd.DataFrame(
        {"species_a": [10.0, 20.0, 30.0], "species_b": [5.0, 10.0, 15.0]},
        index=[0, 1, 2],
    )


def _make_prediction_df() -> pd.DataFrame:
    """Returns a prediction that exactly matches the true timecourse."""
    return pd.DataFrame(
        {"species_a": [10.0, 20.0, 30.0], "species_b": [5.0, 10.0, 15.0]},
        index=[0, 1, 2],
    )


def _make_prediction_df_offset() -> pd.DataFrame:
    """Returns a prediction that is 10% higher than the true timecourse."""
    return pd.DataFrame(
        {"species_a": [11.0, 22.0, 33.0], "species_b": [5.5, 11.0, 16.5]},
        index=[0, 1, 2],
    )


#########################################
class TestR2ScoreInfoInit(unittest.TestCase):
    """Tests for R2ScoreInfo.__init__."""

    def setUp(self) -> None:
        self.kwargs = {m: float(i) for i, m in enumerate(R2ScoreInfo.METRICS)}

    def test_all_metrics_stored_as_attributes(self) -> None:
        """All metrics are stored as attributes on the instance."""
        if IGNORE_TESTS:
            return
        info = R2ScoreInfo(**self.kwargs)
        for metric in R2ScoreInfo.METRICS:
            self.assertTrue(hasattr(info, metric))

    def test_description_stored(self) -> None:
        """The description is stored correctly."""
        if IGNORE_TESTS:
            return
        info = R2ScoreInfo(description="test_desc", **self.kwargs)
        self.assertEqual(info.description, "test_desc")

    def test_aggregation_type_stored(self) -> None:
        """The aggregation_type is stored correctly."""
        if IGNORE_TESTS:
            return
        info = R2ScoreInfo(aggregation_type="species_a", **self.kwargs)
        self.assertEqual(info.aggregation_type, "species_a")

    def test_default_nan_values(self) -> None:
        """When no metrics are provided, all default to NaN."""
        if IGNORE_TESTS:
            return
        info = R2ScoreInfo()
        for metric in R2ScoreInfo.METRICS:
            self.assertTrue(math.isnan(getattr(info, metric)),
                            f"{metric} should be NaN by default")

    def test_percentile_names_match_makePercentileName(self) -> None:
        """Percentile names in METRICS match Score.makePercentileName output."""
        if IGNORE_TESTS:
            return
        for p in R2ScoreInfo.PERCENTILES:
            expected = Score.makePercentileName(p)
            self.assertIn(expected, R2ScoreInfo.METRICS)

    def test_metrics_list_matches_percentiles(self) -> None:
        """METRICS contains mean/min/max/count plus one percentile per PERCENTILE."""
        if IGNORE_TESTS:
            return
        expected_count = 4 + len(R2ScoreInfo.PERCENTILES)
        self.assertEqual(len(R2ScoreInfo.METRICS), expected_count)


#########################################
class TestR2Metrics(unittest.TestCase):
    """Tests for module-level R2_METRICS."""

    def test_r2_metrics_match_class_metrics(self) -> None:
        """Module-level R2_METRICS equals R2ScoreInfo.METRICS."""
        if IGNORE_TESTS:
            return
        self.assertEqual(R2_METRICS, R2ScoreInfo.METRICS)

    def test_r2_metrics_uses_make_percentile_name(self) -> None:
        """R2_METRICS percentile names are generated via Score.makePercentileName."""
        if IGNORE_TESTS:
            return
        for p in R2ScoreInfo.PERCENTILES:
            expected = Score.makePercentileName(p)
            self.assertIn(expected, R2_METRICS)


#########################################
class TestSERIALIZATION_PATH(unittest.TestCase):
    """Tests for module-level SERIALIZATION_PATH."""

    def test_default_path(self) -> None:
        """Default serialization path is 'r2_score.csv'."""
        if IGNORE_TESTS:
            return
        self.assertEqual(SERIALIZATION_PATH, "r2_score.csv")


#########################################
class TestR2ScoreInit(unittest.TestCase):
    """Tests for R2Score.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_initialization(self) -> None:
        """Default constructor sets up with default path and is_ignore_first_prediction=True."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        self.assertTrue(score.dataframe.empty)

    def test_is_initialize_creates_empty_df(self) -> None:
        """is_initialize=True creates an empty DataFrame with columns."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path, is_initialize=True)
        self.assertTrue(score.dataframe.empty)

    def test_is_ignore_first_prediction_default_true(self) -> None:
        """By default, the first prediction row is ignored (but R² doesn't use this for per-timepoint)."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # R² is computed per species (not per timepoint), so result count doesn't change.
        self.assertEqual(len(infos), 3)  # model + 2 species

    def test_is_ignore_first_prediction_false(self) -> None:
        """When False, behavior should be the same since R² aggregates across all rows."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path,
                        is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 3)


#########################################
class TestComputeR2(unittest.TestCase):
    """Tests for R2Score._computeR2."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')
        self.score = R2Score(serialization_path=self.tmp_path,
                             is_ignore_first_prediction=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_perfect_prediction_yields_one_r2(self) -> None:
        """When prediction equals true, R² should be 1.0 for each species."""
        if IGNORE_TESTS:
            return
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        r2_ser = self.score._computeR2(true_df, pred_df)
        # Should have one value per species column.
        self.assertEqual(len(r2_ser), 2)
        for sp in r2_ser:
            self.assertAlmostEqual(sp, 1.0)

    def test_zero_variance_yields_negative_one(self) -> None:
        """When true values are constant (zero variance), R² should be -1."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [5.0, 5.0, 5.0]}, index=[0, 1, 2])
        pred_df = pd.DataFrame({"sp": [6.0, 6.0, 6.0]}, index=[0, 1, 2])
        r2_ser = self.score._computeR2(true_df, pred_df)
        # ss_tot = 0 → division by zero → R² set to -1.
        self.assertAlmostEqual(r2_ser.iloc[0], -1.0)

    def test_negative_r2_for_bad_predictions(self) -> None:
        """When prediction is worse than mean(true), R² should be negative (clipped to -1)."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [10.0, 20.0, 30.0]}, index=[0, 1, 2])
        pred_df = pd.DataFrame({"sp": [100.0, 200.0, 300.0]}, index=[0, 1, 2])
        r2_ser = self.score._computeR2(true_df, pred_df)
        # Prediction is much worse than mean → R² < 0 → clipped to -1.
        self.assertAlmostEqual(r2_ser.iloc[0], -1.0)

    def test_r2_less_than_one_for_imperfect(self) -> None:
        """For imperfect predictions, R² should be less than 1 but greater than -1."""
        if IGNORE_TESTS:
            return
        true_df = _make_true_df()
        pred_df = _make_prediction_df_offset()
        r2_ser = self.score._computeR2(true_df, pred_df)
        for sp in r2_ser:
            self.assertLess(sp, 1.0)
            self.assertGreater(sp, -1.0)

    def test_multiple_species(self) -> None:
        """R² is computed correctly for multiple species columns."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({
            "a": [10.0, 20.0, 30.0],
            "b": [5.0, 10.0, 15.0],
        }, index=[0, 1, 2])
        pred_df = pd.DataFrame({
            "a": [10.0, 20.0, 30.0],
            "b": [5.0, 10.0, 15.0],
        }, index=[0, 1, 2])
        r2_ser = self.score._computeR2(true_df, pred_df)
        self.assertEqual(len(r2_ser), 2)
        for sp in r2_ser:
            self.assertAlmostEqual(sp, 1.0)

    def test_output_indexed_by_species(self) -> None:
        """Output Series is indexed by species name."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp_a": [10.0], "sp_b": [20.0]}, index=[0])
        pred_df = pd.DataFrame({"sp_a": [10.0], "sp_b": [20.0]}, index=[0])
        r2_ser = self.score._computeR2(true_df, pred_df)
        self.assertEqual(list(r2_ser.index), ["sp_a", "sp_b"])

    def test_identical_true_values_yields_negative_one(self) -> None:
        """When true values are all the same constant, R² is -1."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [7.0, 7.0, 7.0]}, index=[0, 1, 2])
        pred_df = pd.DataFrame({"sp": [7.0, 7.0, 7.0]}, index=[0, 1, 2])
        r2_ser = self.score._computeR2(true_df, pred_df)
        # ss_tot = 0 → R² set to -1 even though prediction is perfect.
        self.assertAlmostEqual(r2_ser.iloc[0], -1.0)


#########################################
class TestMakeScoreInfo(unittest.TestCase):
    """Tests for R2Score.makeScoreInfo."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_model_plus_species_infos(self) -> None:
        """Returns one model-level info plus one per species."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 3)  # model + species_a + species_b

    def test_first_info_is_model(self) -> None:
        """First element has aggregation_type='model'."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        infos = score.makeScoreInfo("test", _make_true_df(), _make_prediction_df())
        self.assertEqual(infos[0].aggregation_type, "model")

    def test_species_info_aggregation_types(self) -> None:
        """Subsequent elements have aggregation_type matching species names."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        agg_types = [info.aggregation_type for info in infos]
        self.assertIn("species_a", agg_types)
        self.assertIn("species_b", agg_types)

    def test_description_propagated(self) -> None:
        """The description is stored in all ScoreInfo objects."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        infos = score.makeScoreInfo("my_test", _make_true_df(), _make_prediction_df())
        for info in infos:
            self.assertEqual(info.description, "my_test")

    def test_perfect_prediction_model_mean_one(self) -> None:
        """For perfect predictions, model-level mean R² should be 1.0."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertAlmostEqual(model_info.mean, 1.0)

    def test_perfect_prediction_species_mean_one(self) -> None:
        """For perfect predictions, species-level mean R² should be 1.0."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        for info in infos:
            if info.aggregation_type != "model":
                self.assertAlmostEqual(info.mean, 1.0)

    def test_single_species(self) -> None:
        """Works correctly with a single species column."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = pd.DataFrame({"only_sp": [10.0, 20.0, 30.0]}, index=[0, 1, 2])
        pred_df = pd.DataFrame({"only_sp": [10.0, 20.0, 30.0]}, index=[0, 1, 2])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 2)  # model + only_sp

    def test_model_count_equals_species(self) -> None:
        """Model-level count equals the number of species."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()  # 2 species
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertEqual(model_info.count, 2)

    def test_species_count_is_one(self) -> None:
        """Species-level count is always 1 (one R² value per species)."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        for info in infos:
            if info.aggregation_type != "model":
                self.assertEqual(info.count, 1)

    def test_percentiles_computed(self) -> None:
        """Percentile metrics are computed and stored."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df_offset()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        for p in R2ScoreInfo.PERCENTILES:
            attr_name = Score.makePercentileName(p)
            self.assertTrue(hasattr(model_info, attr_name))

    def test_model_aggregation_uses_all_species(self) -> None:
        """Model-level aggregation combines all species values."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [10.0, 20.0],   # perfect → R²=1
            "b": [5.0, 10.0],    # perfect → R²=1
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertAlmostEqual(model_info.mean, 1.0)

    def test_species_aggregation_is_column_specific(self) -> None:
        """Species-level aggregation uses only that species column."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [10.0, 40.0],   # worse prediction → R² < 0 → clipped to -1
            "b": [5.5, 11.0],    # slightly off → positive R²
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        species_a_info = next(i for i in infos if i.aggregation_type == "a")
        species_b_info = next(i for i in infos if i.aggregation_type == "b")
        # Species a: ss_res=400, ss_tot=50 → R²=-7→-1; Species b: positive R².
        self.assertNotEqual(species_a_info.mean, species_b_info.mean)


#########################################
class TestMakeR2ScoreInfo(unittest.TestCase):
    """Tests for R2Score._makeR2ScoreInfo."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')
        self.score = R2Score(serialization_path=self.tmp_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_array_returns_nan(self) -> None:
        """An empty array returns NaN for all metrics."""
        if IGNORE_TESTS:
            return
        info = self.score._makeR2ScoreInfo(np.array([], dtype=float))
        self.assertTrue(math.isnan(info.mean))
        self.assertTrue(math.isnan(info.min))
        self.assertTrue(math.isnan(info.max))

    def test_mean_correct(self) -> None:
        """Mean is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, 0.6, 0.7])
        info = self.score._makeR2ScoreInfo(arr.copy())
        self.assertAlmostEqual(info.mean, 0.6)

    def test_min_correct(self) -> None:
        """Min is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.3, 0.5, 0.7])
        info = self.score._makeR2ScoreInfo(arr.copy())
        self.assertAlmostEqual(info.min, 0.3)

    def test_max_correct(self) -> None:
        """Max is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.3, 0.5, 0.7])
        info = self.score._makeR2ScoreInfo(arr.copy())
        self.assertAlmostEqual(info.max, 0.7)

    def test_percentiles_correct(self) -> None:
        """Percentile values are computed correctly."""
        if IGNORE_TESTS:
            return
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        info = self.score._makeR2ScoreInfo(arr.copy())
        expected_p25 = float(np.nanpercentile(arr, 25.0))
        self.assertAlmostEqual(info.p25, expected_p25)

    def test_nan_values_replaced_before_aggregation(self) -> None:
        """NaN values are replaced with LARGE_VAL before aggregation and still counted."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, float('nan'), 0.3])
        info = self.score._makeR2ScoreInfo(arr.copy())
        # NaN is replaced with LARGE_VAL for aggregation but still counted.
        self.assertEqual(info.count, 3)

    def test_inf_values_replaced_before_aggregation(self) -> None:
        """Inf values are replaced with LARGE_VAL before aggregation and still counted."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, float('inf'), 0.3])
        info = self.score._makeR2ScoreInfo(arr.copy())
        # Inf is replaced with LARGE_VAL for aggregation but still counted.
        self.assertEqual(info.count, 3)

    def test_large_values_replaced_before_aggregation(self) -> None:
        """Values exceeding LARGE_VAL are replaced with LARGE_VAL before aggregation."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, 2e6])
        info = self.score._makeR2ScoreInfo(arr.copy())
        # The large value is counted but replaced with LARGE_VAL for mean/min/max computation.
        self.assertEqual(info.count, 2)
        # max should be LARGE_VAL since 2e6 was replaced.
        self.assertAlmostEqual(info.max, 1e6)

    def test_single_value(self) -> None:
        """A single valid value produces correct stats."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.7])
        info = self.score._makeR2ScoreInfo(arr.copy())
        self.assertAlmostEqual(info.mean, 0.7)
        self.assertAlmostEqual(info.min, 0.7)
        self.assertAlmostEqual(info.max, 0.7)


#########################################
class TestAddTestResult(unittest.TestCase):
    """Tests for R2Score.addTestResult."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_test_result_serializes_data(self) -> None:
        """addTestResult writes R2ScoreInfo dicts to the CSV."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path, is_initialize=True)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        score.addTestResult(true_df, pred_df, description="test_run")
        self.assertFalse(score.dataframe.empty)

    def test_add_test_result_multiple_calls(self) -> None:
        """Multiple addTestResult calls accumulate rows."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path, is_initialize=True)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        score.addTestResult(true_df, pred_df, description="run1")
        score.addTestResult(true_df, pred_df, description="run2")
        # Each call produces 3 rows (model + 2 species).
        self.assertEqual(len(score.dataframe), 6)

    def test_serialization_path_property(self) -> None:
        """serialization_path returns the correct path."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        self.assertEqual(score.serialization_path, self.tmp_path)


#########################################
class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_r2.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_many_species(self) -> None:
        """Works correctly with many species columns."""
        if IGNORE_TESTS:
            return
        n_species = 50
        true_data = {f"sp_{i}": [float(i + j * 10) for j in range(3)]
                     for i in range(n_species)}
        pred_data = {f"sp_{i}": [float((i + j * 10) * 1.1) for j in range(3)]
                     for i in range(n_species)}
        true_df = pd.DataFrame(true_data, index=[0, 1, 2])
        pred_df = pd.DataFrame(pred_data, index=[0, 1, 2])

        score = R2Score(serialization_path=self.tmp_path)
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # Should have model + n_species entries.
        self.assertEqual(len(infos), 1 + n_species)

    def test_identical_predictions_across_species(self) -> None:
        """When all species have identical R² values, model and species stats match."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [10.0, 20.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [15.0, 30.0],
            "b": [15.0, 30.0],
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        species_a_info = next(i for i in infos if i.aggregation_type == "a")
        species_b_info = next(i for i in infos if i.aggregation_type == "b")
        self.assertAlmostEqual(model_info.mean, species_a_info.mean)
        self.assertAlmostEqual(model_info.mean, species_b_info.mean)

    def test_single_timepoint(self) -> None:
        """Works correctly with a single timepoint."""
        if IGNORE_TESTS:
            return
        score = R2Score(serialization_path=self.tmp_path)
        true_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        pred_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # With single timepoint and zero variance → R²=-1 for species.
        self.assertEqual(len(infos), 2)


#########################################
IGNORE_TESTS = False


if __name__ == "__main__":
    unittest.main()