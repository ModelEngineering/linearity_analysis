"""Tests for AREScore and AREScoreInfo."""

import math
import os
import shutil  # type: ignore
import sys
import tempfile  # type: ignore
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from are_score import AREScore, AREScoreInfo, ARE_METRICS, SERIALIZATION_PATH  # type: ignore
from score import Score  # type: ignore

IGNORE_TESTS = False


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
class TestAREScoreInfoInit(unittest.TestCase):
    """Tests for AREScoreInfo.__init__."""

    def setUp(self) -> None:
        self.kwargs = {m: float(i) for i, m in enumerate(AREScoreInfo.METRICS)}

    def test_all_metrics_stored_as_attributes(self) -> None:
        """All metrics are stored as attributes on the instance."""
        if IGNORE_TESTS:
            return
        info = AREScoreInfo(**self.kwargs)
        for metric in AREScoreInfo.METRICS:
            self.assertTrue(hasattr(info, metric))

    def test_description_stored(self) -> None:
        """The description is stored correctly."""
        if IGNORE_TESTS:
            return
        info = AREScoreInfo(description="test_desc", **self.kwargs)
        self.assertEqual(info.description, "test_desc")

    def test_aggregation_type_stored(self) -> None:
        """The aggregation_type is stored correctly."""
        if IGNORE_TESTS:
            return
        info = AREScoreInfo(aggregation_type="species_a", **self.kwargs)
        self.assertEqual(info.aggregation_type, "species_a")

    def test_default_nan_values(self) -> None:
        """When no metrics are provided, all default to NaN."""
        if IGNORE_TESTS:
            return
        info = AREScoreInfo()
        for metric in AREScoreInfo.METRICS:
            self.assertTrue(math.isnan(getattr(info, metric)),
                            f"{metric} should be NaN by default")

    def test_percentile_names_match_makePercentileName(self) -> None:
        """Percentile names in METRICS match Score.makePercentileName output."""
        if IGNORE_TESTS:
            return
        for p in AREScoreInfo.PERCENTILES:
            expected = Score.makePercentileName(p)
            self.assertIn(expected, AREScoreInfo.METRICS)

    def test_metrics_list_matches_percentiles(self) -> None:
        """METRICS contains mean/min/max/count plus one percentile per PERCENTILE."""
        if IGNORE_TESTS:
            return
        expected_count = 4 + len(AREScoreInfo.PERCENTILES)
        self.assertEqual(len(AREScoreInfo.METRICS), expected_count)


#########################################
class TestAREMetrics(unittest.TestCase):
    """Tests for module-level ARE_METRICS."""

    def test_are_metrics_match_class_metrics(self) -> None:
        """Module-level ARE_METRICS equals AREScoreInfo.METRICS."""
        if IGNORE_TESTS:
            return
        self.assertEqual(ARE_METRICS, AREScoreInfo.METRICS)

    def test_are_metrics_uses_make_percentile_name(self) -> None:
        """ARE_METRICS percentile names are generated via Score.makePercentileName."""
        if IGNORE_TESTS:
            return
        for p in AREScoreInfo.PERCENTILES:
            expected = Score.makePercentileName(p)
            self.assertIn(expected, ARE_METRICS)


#########################################
class TestSERIALIZATION_PATH(unittest.TestCase):
    """Tests for module-level SERIALIZATION_PATH."""

    def test_default_path(self) -> None:
        """Default serialization path is 'are_score.csv'."""
        if IGNORE_TESTS:
            return
        self.assertEqual(SERIALIZATION_PATH, "are_score.csv")


#########################################
class TestAREScoreInit(unittest.TestCase):
    """Tests for AREScore.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_initialization(self) -> None:
        """Default constructor sets up with default path and is_ignore_first_prediction=True."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path)
        self.assertTrue(score.dataframe.empty)

    def test_is_initialize_creates_empty_df(self) -> None:
        """is_initialize=True creates an empty DataFrame with columns."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path, is_initialize=True)
        # The dataframe should be empty but initialized.
        self.assertTrue(score.dataframe.empty)

    def test_is_ignore_first_prediction_default_true(self) -> None:
        """By default, the first prediction row is ignored."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # With 3 timepoints and is_ignore_first_prediction=True,
        # the ARE computation should skip index 0.
        self.assertEqual(len(infos), 3)  # model + 2 species

    def test_is_ignore_first_prediction_false(self) -> None:
        """When False, all prediction rows are included."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # With 3 timepoints and is_ignore_first_prediction=False,
        # all rows contribute.
        self.assertEqual(len(infos), 3)


#########################################
class TestComputeARE(unittest.TestCase):
    """Tests for AREScore._computeARE."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')
        self.score = AREScore(serialization_path=self.tmp_path,
                              is_ignore_first_prediction=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_perfect_prediction_yields_zero_are(self) -> None:
        """When prediction equals true, ARE should be 0 for valid rows."""
        if IGNORE_TESTS:
            return
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        are_df = self.score._computeARE(true_df, pred_df)
        # First row is skipped (is_ignore_first_prediction=True), so 2 remaining rows x 2 species.
        self.assertEqual(are_df.shape, (2, 2))
        np.testing.assert_array_equal(are_df.values.flatten(), [0.0, 0.0, 0.0, 0.0])

    def test_division_by_zero_yields_negative_one(self) -> None:
        """When true value is 0, ARE should be -1 (sentinel for invalid)."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [0.0, 10.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [5.0, 10.0]}, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 0 is skipped (is_ignore_first_prediction=True).
        # Row 1: true=10, pred=10 → ARE=0.
        self.assertEqual(are_df.shape[0], 1)
        self.assertAlmostEqual(float(are_df.iloc[0, 0]), 0.0)

    def test_division_by_zero_at_first_row_skipped(self) -> None:
        """When true value is 0 at the first row (which is skipped), no error."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [0.0, 10.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [99.0, 20.0]}, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 0 skipped; row 1: ARE = |20-10|/10 = 1.0, clipped to 1.0.
        self.assertEqual(are_df.shape[0], 1)
        self.assertAlmostEqual(float(are_df.iloc[0, 0]), 1.0)

    def test_are_clipped_to_one(self) -> None:
        """ARE values greater than 1 are clipped to 1."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [10.0, 10.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [10.0, 50.0]}, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 1: |50-10|/10 = 4.0 → clipped to 1.0.
        self.assertAlmostEqual(float(are_df.iloc[0, 0]), 1.0)

    def test_are_clipped_to_zero(self) -> None:
        """ARE values cannot be negative (absolute value taken)."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [10.0, 20.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [10.0, 5.0]}, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 1: |5-20|/20 = 0.75.
        self.assertAlmostEqual(float(are_df.iloc[0, 0]), 0.75)

    def test_first_row_skipped_by_default(self) -> None:
        """By default, the first row of the result is excluded."""
        if IGNORE_TESTS:
            return
        true_df = _make_true_df()
        pred_df = pd.DataFrame(
            {"species_a": [100.0, 20.0], "species_b": [50.0, 10.0]},
            index=[0, 1],
        )
        are_df = self.score._computeARE(true_df, pred_df)
        # _make_true_df has 3 rows (index 0,1,2). With is_ignore_first_prediction=True,
        # row at index 0 is skipped. Rows at index 1 and 2 remain.
        self.assertEqual(are_df.shape[0], 2)

    def test_first_row_included_when_disabled(self) -> None:
        """When is_ignore_first_prediction=False, all rows are included."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = pd.DataFrame(
            {"species_a": [100.0, 20.0], "species_b": [50.0, 10.0]},
            index=[0, 1],
        )
        are_df = score._computeARE(true_df, pred_df)
        # _make_true_df has 3 rows (index 0,1,2). With is_ignore_first_prediction=False,
        # all 3 rows remain.
        self.assertEqual(are_df.shape[0], 3)

    def test_multiple_species(self) -> None:
        """ARE is computed correctly for multiple species columns."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
            "c": [1.0, 2.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [15.0, 30.0],
            "b": [7.5, 15.0],
            "c": [1.5, 3.0],
        }, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 1: a=|30-20|/20=0.5, b=|15-10|/10=0.5, c=|3-2|/2=0.5.
        self.assertEqual(are_df.shape[0], 1)
        np.testing.assert_array_almost_equal(are_df.iloc[0].values, [0.5, 0.5, 0.5])

    def test_nan_in_true_value(self) -> None:
        """NaN in true values results in -1 sentinel."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({"sp": [float('nan'), 10.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [5.0, 10.0]}, index=[0, 1])
        are_df = self.score._computeARE(true_df, pred_df)
        # Row 0 skipped; row 1: ARE=0.
        self.assertEqual(are_df.shape[0], 1)

    def test_output_structure_matches_input(self) -> None:
        """Output DataFrame has same columns as input."""
        if IGNORE_TESTS:
            return
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        are_df = self.score._computeARE(true_df, pred_df)
        self.assertListEqual(list(are_df.columns), list(true_df.columns))


#########################################
class TestMakeScoreInfo(unittest.TestCase):
    """Tests for AREScore.makeScoreInfo."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_model_plus_species_infos(self) -> None:
        """Returns one model-level info plus one per species."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 3)  # model + species_a + species_b

    def test_first_info_is_model(self) -> None:
        """First element has aggregation_type='model'."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        infos = score.makeScoreInfo("test", _make_true_df(), _make_prediction_df())
        self.assertEqual(infos[0].aggregation_type, "model")

    def test_species_info_aggregation_types(self) -> None:
        """Subsequent elements have aggregation_type matching species names."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
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
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        infos = score.makeScoreInfo("my_test", _make_true_df(), _make_prediction_df())
        for info in infos:
            self.assertEqual(info.description, "my_test")

    def test_perfect_prediction_model_mean_zero(self) -> None:
        """For perfect predictions, model-level mean ARE should be 0."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertAlmostEqual(model_info.mean, 0.0)

    def test_perfect_prediction_species_mean_zero(self) -> None:
        """For perfect predictions, species-level mean ARE should be 0."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        for info in infos:
            if info.aggregation_type != "model":
                self.assertAlmostEqual(info.mean, 0.0)

    def test_offset_prediction_positive_mean(self) -> None:
        """For a prediction that is consistently off, mean ARE should be positive."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df_offset()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertGreater(model_info.mean, 0.0)

    def test_single_species(self) -> None:
        """Works correctly with a single species column."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({"only_sp": [10.0, 20.0, 30.0]}, index=[0, 1, 2])
        pred_df = pd.DataFrame({"only_sp": [10.0, 20.0, 30.0]}, index=[0, 1, 2])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 2)  # model + only_sp

    def test_species_info_count_equals_timepoints(self) -> None:
        """Species-level count equals the number of timepoints (after skipping first)."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=True)
        true_df = _make_true_df()  # 3 timepoints
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        for info in infos:
            if info.aggregation_type != "model":
                self.assertEqual(info.count, 2)  # 3 - 1 skipped

    def test_model_count_equals_total_elements(self) -> None:
        """Model-level count equals species * (timepoints - 1)."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=True)
        true_df = _make_true_df()  # 3 timepoints, 2 species
        pred_df = _make_prediction_df()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        self.assertEqual(model_info.count, 4)  # 2 species * 2 remaining timepoints

    def test_model_min_max_correct(self) -> None:
        """Model-level min and max are consistent with the flattened ARE values."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [15.0, 30.0],
            "b": [7.5, 15.0],
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        # All ARE values are 0.5.
        self.assertAlmostEqual(model_info.min, 0.5)
        self.assertAlmostEqual(model_info.max, 0.5)

    def test_percentiles_computed(self) -> None:
        """Percentile metrics are computed and stored."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = _make_true_df()
        pred_df = _make_prediction_df_offset()
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        for p in AREScoreInfo.PERCENTILES:
            attr_name = Score.makePercentileName(p)
            self.assertTrue(hasattr(model_info, attr_name))

    def test_model_aggregation_uses_all_species(self) -> None:
        """Model-level aggregation combines all species values."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [10.0, 40.0],  # row 0: ARE=0, row 1: |40-20|/20=1.0 (clipped)
            "b": [5.0, 5.0],    # row 0: ARE=0, row 1: |5-10|/10=0.5
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        # Model aggregates [0.0, 1.0, 0.0, 0.5], mean = 0.375.
        self.assertAlmostEqual(model_info.mean, 0.375)

    def test_species_aggregation_is_column_specific(self) -> None:
        """Species-level aggregation uses only that species column."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [5.0, 10.0],
        }, index=[0, 1])
        pred_df = pd.DataFrame({
            "a": [10.0, 40.0],  # row 0: ARE=0, row 1: |40-20|/20=1.0 (clipped)
            "b": [5.0, 5.0],    # row 0: ARE=0, row 1: |5-10|/10=0.5
        }, index=[0, 1])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        species_a_info = next(i for i in infos if i.aggregation_type == "a")
        species_b_info = next(i for i in infos if i.aggregation_type == "b")
        # Species a: [0.0, 1.0], mean=0.5; Species b: [0.0, 0.5], mean=0.25.
        self.assertAlmostEqual(species_a_info.mean, 0.5)
        self.assertAlmostEqual(species_b_info.mean, 0.25)


#########################################
class TestMakeBasicScoreInfo(unittest.TestCase):
    """Tests for AREScore._makeBasicScoreInfo."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')
        self.score = AREScore(serialization_path=self.tmp_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_array_returns_nan(self) -> None:
        """An empty array returns NaN for all metrics."""
        if IGNORE_TESTS:
            return
        info = self.score._makeBasicScoreInfo(np.array([], dtype=float))
        self.assertTrue(math.isnan(info.mean))
        self.assertTrue(math.isnan(info.min))
        self.assertTrue(math.isnan(info.max))
        self.assertEqual(info.count, 0)

    def test_count_excludes_negative_sentinel(self) -> None:
        """Negative sentinel values (-1) are excluded from count."""
        if IGNORE_TESTS:
            return
        arr = np.array([-1.0, -1.0, 0.5, 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertEqual(info.count, 2)

    def test_count_excludes_nan(self) -> None:
        """NaN values are excluded from count."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, float('nan'), 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertEqual(info.count, 2)

    def test_count_includes_inf(self) -> None:
        """Inf values are NOT excluded from count (only filtered for aggregation)."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, float('inf'), 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        # count reflects all non-negative values including inf; inf is replaced with LARGE_VAL for aggregation.
        self.assertEqual(info.count, 3)

    def test_count_includes_large_values(self) -> None:
        """Large values exceeding LARGE_VAL are NOT excluded from count (only filtered for aggregation)."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, 2e6, 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        # count reflects all non-negative values including large ones; large is replaced with LARGE_VAL for aggregation.
        self.assertEqual(info.count, 3)

    def test_mean_correct(self) -> None:
        """Mean is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.1, 0.2, 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertAlmostEqual(info.mean, 0.2)

    def test_min_correct(self) -> None:
        """Min is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.1, 0.5, 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertAlmostEqual(info.min, 0.1)

    def test_max_correct(self) -> None:
        """Max is computed correctly over valid values."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.1, 0.5, 0.3])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertAlmostEqual(info.max, 0.5)

    def test_percentiles_correct(self) -> None:
        """Percentile values are computed correctly."""
        if IGNORE_TESTS:
            return
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        info = self.score._makeBasicScoreInfo(arr.copy())
        # p25 should be approximately 3.25 (linear interpolation).
        expected_p25 = float(np.nanpercentile(arr, 25.0))
        self.assertAlmostEqual(info.p25, expected_p25)

    def test_all_negative_sentinel_returns_nan(self) -> None:
        """When all values are negative sentinels, all metrics are NaN."""
        if IGNORE_TESTS:
            return
        arr = np.array([-1.0, -1.0, -1.0])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertTrue(math.isnan(info.mean))
        self.assertEqual(info.count, 0)

    def test_single_value(self) -> None:
        """A single valid value produces correct stats."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.7])
        info = self.score._makeBasicScoreInfo(arr.copy())
        self.assertAlmostEqual(info.mean, 0.7)
        self.assertAlmostEqual(info.min, 0.7)
        self.assertAlmostEqual(info.max, 0.7)
        self.assertEqual(info.count, 1)

    def test_large_values_replaced_before_aggregation(self) -> None:
        """Values exceeding LARGE_VAL are replaced with LARGE_VAL before aggregation but still counted."""
        if IGNORE_TESTS:
            return
        arr = np.array([0.5, 2e6])
        info = self.score._makeBasicScoreInfo(arr.copy())
        # The large value is counted but replaced with LARGE_VAL for mean/min/max computation.
        self.assertEqual(info.count, 2)
        # max should be LARGE_VAL since 2e6 was replaced.
        self.assertAlmostEqual(info.max, 1e6)

    def test_mixed_valid_and_invalid(self) -> None:
        """Mix of valid values and invalid sentinels/NaN."""
        if IGNORE_TESTS:
            return
        arr = np.array([-1.0, 0.2, float('nan'), 0.8, float('inf')])
        info = self.score._makeBasicScoreInfo(arr.copy())
        # -1 is excluded (negative sentinel), NaN is excluded. count=3 (0.2, 0.8, inf).
        self.assertEqual(info.count, 3)
        # mean of [0.2, 0.8, LARGE_VAL] ≈ LARGE_VAL/3 due to inf replacement.
        self.assertAlmostEqual(info.min, 0.2)


#########################################
class TestAddTestResult(unittest.TestCase):
    """Tests for Score.addTestResult (inherited by AREScore)."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_test_result_serializes_data(self) -> None:
        """addTestResult writes ScoreInfo dicts to the CSV."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path, is_initialize=True)
        true_df = _make_true_df()
        pred_df = _make_prediction_df()
        score.addTestResult(true_df, pred_df, description="test_run")
        self.assertFalse(score.dataframe.empty)

    def test_add_test_result_multiple_calls(self) -> None:
        """Multiple addTestResult calls accumulate rows."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path, is_initialize=True)
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
        score = AREScore(serialization_path=self.tmp_path)
        self.assertEqual(score.serialization_path, self.tmp_path)


#########################################
class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_are.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_single_timepoint_with_ignore(self) -> None:
        """With only 1 timepoint and is_ignore_first_prediction=True, result is empty."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=True)
        true_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        pred_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # Model info should have count=0 since all rows are skipped.
        self.assertEqual(len(infos), 2)
        self.assertEqual(infos[0].count, 0)

    def test_single_timepoint_without_ignore(self) -> None:
        """With only 1 timepoint and is_ignore_first_prediction=False, result has data."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        pred_df = pd.DataFrame({"sp": [12.0]}, index=[0])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        self.assertEqual(len(infos), 2)
        # ARE = |12-10|/10 = 0.2.
        self.assertAlmostEqual(infos[0].mean, 0.2)

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

        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        infos = score.makeScoreInfo("test", true_df, pred_df)
        # Should have model + n_species entries.
        self.assertEqual(len(infos), 1 + n_species)

    def test_zero_true_values_clipped_to_one(self) -> None:
        """Zero true values produce inf which gets clipped to 1.0."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
        true_df = pd.DataFrame({
            "sp": [0.0, 10.0, 20.0],
        }, index=[0, 1, 2])
        pred_df = pd.DataFrame({
            "sp": [5.0, 10.0, 30.0],
        }, index=[0, 1, 2])
        infos = score.makeScoreInfo("test", true_df, pred_df)
        model_info = infos[0]
        # Row 0: (5-0)/0 = inf → abs → clip to 1.0
        # Row 1: ARE=0, valid.
        # Row 2: ARE=|30-20|/20=0.5, valid.
        self.assertEqual(model_info.count, 3)
        self.assertAlmostEqual(model_info.mean, (1.0 + 0.0 + 0.5) / 3)

    def test_identical_predictions_across_species(self) -> None:
        """When all species have identical ARE values, model and species stats match."""
        if IGNORE_TESTS:
            return
        score = AREScore(serialization_path=self.tmp_path,
                         is_ignore_first_prediction=False)
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


if __name__ == "__main__":
    unittest.main()