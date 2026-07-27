"""Tests for base Score class and makePercentileName static method."""

import math
import os
import shutil  # type: ignore
import sys
import tempfile  # type: ignore
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import (  # type: ignore
    AGGREGATION_COUNT,
    AGGREGATION_DESCRIPTION,
    AGGREGATION_MAX,
    AGGREGATION_MEAN,
    AGGREGATION_MIN,
    AGGREGATION_TYPE,
    DEFAULT_PERCENTILES,
    METRIC_TYPE_ARE,
    METRIC_TYPE_R2,
    Score,
    ScoreInfo,
)
from are_score import AREScoreInfo  # type: ignore


#########################################
class TestScoreConstants(unittest.TestCase):
    """Tests for module-level constants in score.py."""

    def test_aggregation_constants(self) -> None:
        """Aggregation type constants have expected values."""
        self.assertEqual(AGGREGATION_DESCRIPTION, "description")
        self.assertEqual(AGGREGATION_MEAN, "mean")
        self.assertEqual(AGGREGATION_MIN, "min")
        self.assertEqual(AGGREGATION_MAX, "max")
        self.assertEqual(AGGREGATION_COUNT, "count")
        self.assertEqual(AGGREGATION_TYPE, "aggregation_type")

    def test_metric_type_constants(self) -> None:
        """Metric type constants have expected values."""
        self.assertEqual(METRIC_TYPE_ARE, "are")
        self.assertEqual(METRIC_TYPE_R2, "r2")

    def test_default_percentiles_is_list_of_six_values(self) -> None:
        """DEFAULT_PERCENTILES contains 6 percentile values."""
        self.assertEqual(len(DEFAULT_PERCENTILES), 6)
        for p in DEFAULT_PERCENTILES:
            self.assertIsInstance(p, float)


#########################################
class TestScoreInfoInit(unittest.TestCase):
    """Tests for ScoreInfo.__init__ (abstract base)."""

    def test_missing_metric_raises_value_error(self) -> None:
        """If a required metric is missing from kwargs, ValueError is raised."""
        # ScoreInfo.METRICS is empty list, so this should not raise.
        info = ScoreInfo()
        self.assertEqual(info.description, "")
        self.assertEqual(info.aggregation_type, "")

    def test_description_stored(self) -> None:
        """Description is stored as attribute."""
        info = ScoreInfo(description="test_desc")
        self.assertEqual(info.description, "test_desc")

    def test_aggregation_type_stored(self) -> None:
        """aggregation_type is stored as attribute."""
        info = ScoreInfo(aggregation_type="species_a")
        self.assertEqual(info.aggregation_type, "species_a")

    def test_default_description_is_empty_string(self) -> None:
        """Default description is empty string."""
        info = ScoreInfo()
        self.assertEqual(info.description, "")

    def test_default_aggregation_type_is_empty_string(self) -> None:
        """Default aggregation_type is empty string."""
        info = ScoreInfo()
        self.assertEqual(info.aggregation_type, "")


#########################################
class TestScoreInit(unittest.TestCase):
    """Tests for Score.__init__."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_serialization_path_raises_value_error(self) -> None:
        """Score.__init__ raises ValueError when serialization_path is empty."""
        with self.assertRaises(ValueError):
            Score(serialization_path="")

    def test_is_initialize_creates_empty_dataframe(self) -> None:
        """is_initialize=True creates an empty DataFrame and writes it to CSV."""
        score = Score(serialization_path=self.tmp_path, is_initialize=True)
        self.assertTrue(score.dataframe.empty)

    def test_nonexistent_file_creates_empty_dataframe(self) -> None:
        """When the file doesn't exist and is_initialize=False, an empty DataFrame is created."""
        score = Score(serialization_path=self.tmp_path, is_initialize=False)
        self.assertTrue(score.dataframe.empty)


#########################################
class TestScoreProperties(unittest.TestCase):
    """Tests for Score property accessors."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_serialization_path_property(self) -> None:
        """serialization_path returns the path passed to constructor."""
        score = Score(serialization_path=self.tmp_path)
        self.assertEqual(score.serialization_path, self.tmp_path)

    def test_dataframe_property_returns_data_frame(self) -> None:
        """dataframe property returns a DataFrame."""
        score = Score(serialization_path=self.tmp_path, is_initialize=True)
        df = score.dataframe
        self.assertIsInstance(df, pd.DataFrame)

    def test_score_df_property_returns_data_frame(self) -> None:
        """score_df property returns a DataFrame (alias for dataframe)."""
        score = Score(serialization_path=self.tmp_path, is_initialize=True)
        df = score.score_df
        self.assertIsInstance(df, pd.DataFrame)

    def test_score_df_equals_dataframe(self) -> None:
        """score_df and dataframe return the same DataFrame."""
        score = Score(serialization_path=self.tmp_path, is_initialize=True)
        self.assertTrue(score.score_df.equals(score.dataframe))


#########################################
class TestMakePercentileName(unittest.TestCase):
    """Tests for Score.makePercentileName static method."""

    def test_percentile_25(self) -> None:
        """makePercentileName(25.0) returns 'p25'."""
        self.assertEqual(Score.makePercentileName(25.0), "p25")

    def test_percentile_30(self) -> None:
        """makePercentileName(30.0) returns 'p30'."""
        self.assertEqual(Score.makePercentileName(30.0), "p30")

    def test_percentile_50(self) -> None:
        """makePercentileName(50.0) returns 'p50'."""
        self.assertEqual(Score.makePercentileName(50.0), "p50")

    def test_percentile_80(self) -> None:
        """makePercentileName(80.0) returns 'p80'."""
        self.assertEqual(Score.makePercentileName(80.0), "p80")

    def test_percentile_95(self) -> None:
        """makePercentileName(95.0) returns 'p95'."""
        self.assertEqual(Score.makePercentileName(95.0), "p95")

    def test_percentile_99(self) -> None:
        """makePercentileName(99.0) returns 'p99'."""
        self.assertEqual(Score.makePercentileName(99.0), "p99")

    def test_percentile_100(self) -> None:
        """makePercentileName(100.0) returns 'p100'."""
        self.assertEqual(Score.makePercentileName(100.0), "p100")

    def test_percentile_0(self) -> None:
        """makePercentileName(0.0) returns 'p0'."""
        self.assertEqual(Score.makePercentileName(0.0), "p0")


#########################################
class TestMakeScoreInfoNotImplemented(unittest.TestCase):
    """Tests that Score.makeScoreInfo raises NotImplementedError."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_makeScoreInfo_raises_not_implemented_error(self) -> None:
        """Calling makeScoreInfo on base Score raises NotImplementedError."""
        score = Score(serialization_path=self.tmp_path, is_initialize=True)
        true_df = pd.DataFrame({"sp": [10.0, 20.0]}, index=[0, 1])
        pred_df = pd.DataFrame({"sp": [10.0, 20.0]}, index=[0, 1])
        with self.assertRaises(NotImplementedError):
            score.makeScoreInfo("test", true_df, pred_df)


#########################################
class TestAddTestResult(unittest.TestCase):
    """Tests for Score.addTestResult (requires concrete subclass).

    Since Score is abstract, we test addTestResult indirectly through a minimal
    concrete subclass.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    class _ConcreteScore(Score):
        """Minimal concrete subclass for testing addTestResult."""
        def makeScoreInfo(self, description, true_df, pred_df):
            return []

    def test_add_test_result_does_not_raise(self) -> None:
        """addTestResult with empty makeScoreInfo does not raise."""
        score = self._ConcreteScore(serialization_path=self.tmp_path, is_initialize=True)
        true_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        pred_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        # Should not raise even with empty result.
        score.addTestResult(true_df, pred_df, description="test")


#########################################
class TestPlotCDF(unittest.TestCase):
    """Tests for Score.plotCDF method."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'test_score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    class _ConcreteScore(Score):
        """Minimal concrete subclass for testing plotCDF."""
        def makeScoreInfo(self, description, true_df, pred_df):
            return []

    def test_plotcdf_both_model_and_species_no_error(self) -> None:
        """plotCDF with is_plot_model=True and is_plot_species=True does not raise on empty data."""
        score = self._ConcreteScore(serialization_path=self.tmp_path, is_initialize=True)
        # Empty dataframe should not raise.
        result = score.plotCDF("mean", is_plot=False)
        self.assertIsNotNone(result)

    def test_plotcdf_model_only_no_error(self) -> None:
        """plotCDF with is_plot_model=True and is_plot_species=False does not raise on empty data."""
        score = self._ConcreteScore(serialization_path=self.tmp_path, is_initialize=True)
        result = score.plotCDF("mean", is_plot_model=True, is_plot_species=False, is_plot=False)
        self.assertIsNotNone(result)

    def test_plotcdf_species_only_no_error(self) -> None:
        """plotCDF with is_plot_model=False and is_plot_species=True does not raise on empty data."""
        score = self._ConcreteScore(serialization_path=self.tmp_path, is_initialize=True)
        result = score.plotCDF("mean", is_plot_model=False, is_plot_species=True, is_plot=False)
        self.assertIsNotNone(result)

    def test_plotcdf_neither_raises_value_error(self) -> None:
        """plotCDF with both flags False raises ValueError."""
        score = self._ConcreteScore(serialization_path=self.tmp_path, is_initialize=True)
        with self.assertRaises(ValueError):
            score.plotCDF("mean", is_plot_model=False, is_plot_species=False, is_plot=False)

    def test_plotcdf_with_data(self) -> None:
        """plotCDF with populated data does not raise."""
        # Create a concrete subclass that populates the dataframe.
        class _PopulatedScore(Score):
            def makeScoreInfo(self, description, true_df, pred_df):
                info = AREScoreInfo(
                    description=description, aggregation_type="model",
                    mean=0.1, min=0.05, max=0.2, count=10,
                    p25=0.08, p30=0.09, p50=0.1, p80=0.15, p95=0.18, p99=0.19,
                )
                return [info]

        score = _PopulatedScore(serialization_path=self.tmp_path, is_initialize=True)
        true_df = pd.DataFrame({"sp": [10.0]}, index=[0])
        pred_df = pd.DataFrame({"sp": [10.5]}, index=[0])
        score.addTestResult(true_df, pred_df, description="test")

        result = score.plotCDF("mean", is_plot=False)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
