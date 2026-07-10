"""Tests for the Score class."""

import os
import shutil  # type: ignore
import tempfile  # type: ignore
import unittest
from unittest.mock import patch
import matplotlib  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.model import Model  # type: ignore
from src.system_discovery import SystemDiscovery  # type: ignore
from src.score import (Score, ScoreInfo,  # type: ignore
        AGGREGATION_TYPE, AGGREGATION_MEAN, AGGREGATION_COUNT)

BIOMODELS_DIR = cn.BIOMODELS_DIR
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)

IGNORE_TESTS = False
if not IGNORE_TESTS:
    matplotlib.use("Agg")

TIMES = [0.0, 1.0, 2.0]
TRUE_DF = pd.DataFrame({'A': [1.0, 2.0, 4.0], 'B': [2.0, 4.0, 8.0]}, index=TIMES)
PRED_DF = 2.0 * TRUE_DF  # ARE = (2x - x) / x = 1.0 everywhere


def _temp_csv_path() -> str:
    """Returns a unique temp path for a CSV that does not yet exist."""
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    os.unlink(path)
    return path


def _make_score(description: str = "test") -> Score:
    """Returns a Score with one test result where ARE == 1.0 everywhere."""
    score = Score(serialization_path=_temp_csv_path())
    score.addTestResult(TRUE_DF, PRED_DF, description=description)
    return score


class TestScoreInit(unittest.TestCase):
    """Tests for Score.__init__."""

    def test_serialization_path_stored(self) -> None:
        """Constructor stores _serialization_path."""
        if IGNORE_TESTS:
            return
        path = _temp_csv_path()
        score = Score(serialization_path=path)
        self.assertEqual(score._serialization_path, path)

    def test_score_df_initially_empty(self) -> None:
        """score_df is empty when no pre-existing file exists."""
        if IGNORE_TESTS:
            return
        score = Score(serialization_path=_temp_csv_path())
        self.assertTrue(score.score_df.empty)

    def test_existing_file_loaded(self) -> None:
        """score_df is populated when the serialization file already exists."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        restored = Score(serialization_path=score._serialization_path)
        self.assertFalse(restored.score_df.empty)

    def test_is_initialize_true_clears_existing_data(self) -> None:
        """is_initialize=True gives empty score_df even when CSV has existing data."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        path = score._serialization_path
        reset = Score(serialization_path=path, is_initialize=True)
        self.assertTrue(reset.score_df.empty)

    def test_is_initialize_true_writes_empty_csv(self) -> None:
        """is_initialize=True creates the CSV file on disk."""
        if IGNORE_TESTS:
            return
        path = _temp_csv_path()
        Score(serialization_path=path, is_initialize=True)
        self.assertTrue(os.path.exists(path))

    def test_is_initialize_true_subsequent_add_starts_fresh(self) -> None:
        """After is_initialize=True, addTestResult produces the correct row count."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        path = score._serialization_path
        reset = Score(serialization_path=path, is_initialize=True)
        reset.addTestResult(TRUE_DF, PRED_DF)
        self.assertEqual(len(reset.score_df), 1 + len(TRUE_DF.columns))

    def test_is_initialize_false_is_default(self) -> None:
        """Default is_initialize=False preserves existing CSV data."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        path = score._serialization_path
        reloaded = Score(serialization_path=path)
        self.assertFalse(reloaded.score_df.empty)


class TestAddTestResult(unittest.TestCase):
    """Tests for Score.addTestResult."""

    def test_adds_model_and_species_rows(self) -> None:
        """addTestResult adds one model row plus one row per species."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        self.assertEqual(len(score.score_df), 1 + len(TRUE_DF.columns))

    def test_model_row_present(self) -> None:
        """score_df contains a row with aggregation_type == 'model'."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        model_rows = score.score_df[score.score_df[AGGREGATION_TYPE] == "model"]
        self.assertEqual(len(model_rows), 1)

    def test_species_rows_present(self) -> None:
        """score_df contains one row per species with the correct aggregation_type."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        species_rows = score.score_df[score.score_df[AGGREGATION_TYPE] != "model"]
        self.assertEqual(
                list(species_rows[AGGREGATION_TYPE]),
                list(TRUE_DF.columns))

    def test_model_row_mean_correct(self) -> None:
        """Model row mean equals 1.0 when ARE == 1.0 everywhere."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        model_row = score.score_df[score.score_df[AGGREGATION_TYPE] == "model"].iloc[0]
        self.assertAlmostEqual(float(model_row[AGGREGATION_MEAN]), 1.0)

    def test_model_row_count_excludes_zero_true(self) -> None:
        """Count excludes cells where true == 0 (ARE would be NaN)."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [0.0, 1.0]}, index=[0.0, 1.0])
        pred_df = pd.DataFrame({'A': [1.0, 2.0]}, index=[0.0, 1.0])
        score = Score(serialization_path=_temp_csv_path(), is_ignore_first_prediction=False)
        score.addTestResult(true_df, pred_df)
        model_row = score.score_df[score.score_df[AGGREGATION_TYPE] == "model"].iloc[0]
        self.assertEqual(int(model_row[AGGREGATION_COUNT]), 2)

    def test_multiple_calls_accumulate_rows(self) -> None:
        """Two addTestResult calls double the number of rows."""
        if IGNORE_TESTS:
            return
        score = _make_score()
        score.addTestResult(TRUE_DF, PRED_DF)
        self.assertEqual(len(score.score_df), 2 * (1 + len(TRUE_DF.columns)))

    def test_creates_csv_file(self) -> None:
        """addTestResult creates the CSV file at the serialization path."""
        if IGNORE_TESTS:
            return
        path = _temp_csv_path()
        score = Score(serialization_path=path)
        self.assertFalse(os.path.exists(path))
        score.addTestResult(TRUE_DF, PRED_DF)
        self.assertTrue(os.path.exists(path))


class TestDeserialize(unittest.TestCase):
    """Tests for Score.deserialize."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = os.path.join(self.tmp_dir, 'score.csv')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_score(self) -> None:
        """deserialize() returns a Score instance."""
        if IGNORE_TESTS:
            return
        score = Score(serialization_path=self.tmp_path)
        score.addTestResult(TRUE_DF, PRED_DF)
        restored = Score.deserialize(self.tmp_path)
        self.assertIsInstance(restored, Score)

    def test_recovers_serialization_path(self) -> None:
        """deserialize() sets _serialization_path to the source path."""
        if IGNORE_TESTS:
            return
        score = Score(serialization_path=self.tmp_path)
        score.addTestResult(TRUE_DF, PRED_DF)
        restored = Score.deserialize(self.tmp_path)
        self.assertEqual(restored._serialization_path, self.tmp_path)

    def test_recovers_row_count(self) -> None:
        """deserialize() restores the correct number of rows."""
        if IGNORE_TESTS:
            return
        score = Score(serialization_path=self.tmp_path)
        score.addTestResult(TRUE_DF, PRED_DF)
        score.addTestResult(TRUE_DF, PRED_DF)
        restored = Score.deserialize(self.tmp_path)
        self.assertEqual(len(restored.score_df), len(score.score_df))

    def test_recovers_model_mean(self) -> None:
        """deserialize() restores the model row mean correctly."""
        if IGNORE_TESTS:
            return
        score = Score(serialization_path=self.tmp_path)
        score.addTestResult(TRUE_DF, PRED_DF)
        restored = Score.deserialize(self.tmp_path)
        model_row = restored.score_df[
                restored.score_df[AGGREGATION_TYPE] == "model"].iloc[0]
        self.assertAlmostEqual(float(model_row[AGGREGATION_MEAN]), 1.0)


class TestPlot(unittest.TestCase):
    """Tests for Score.plot."""

    def setUp(self) -> None:
        self.score = _make_score()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_figure(self) -> None:
        """plot() returns a matplotlib Figure."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plot()
        self.assertIsInstance(fig, mfigure.Figure)

    def test_xlabel_is_column_name(self) -> None:
        """x-axis label equals the column_name argument."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plot(column_name="mean")
        self.assertTrue("mean" in fig.axes[0].get_xlabel())

    def test_ylabel_is_count(self) -> None:
        """y-axis is labelled 'count'."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plot()
        self.assertEqual(fig.axes[0].get_ylabel(), "count")

    def test_title_is_aggregation_type(self) -> None:
        """Plot title equals the aggregation_type argument."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plot(is_model_aggregation=True)
        self.assertEqual(fig.axes[0].get_title(), "model")

    def test_species_aggregation_type(self) -> None:
        """plot() works when given a species name as aggregation_type."""
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            fig = self.score.plot(is_model_aggregation=False)
        self.assertIsInstance(fig, mfigure.Figure)


class TestMakeScoreInfo(unittest.TestCase):
    """Tests for Score.makeScoreInfo."""

    def setUp(self) -> None:
        self.score = Score(serialization_path=_temp_csv_path(), is_ignore_first_prediction=False)

    def test_returns_list(self) -> None:
        """makeScoreInfo returns a list."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        self.assertIsInstance(result, list)

    def test_length_is_one_plus_num_species(self) -> None:
        """List has one model entry plus one entry per species."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        self.assertEqual(len(result), 1 + len(TRUE_DF.columns))

    def test_all_elements_are_score_info(self) -> None:
        """Every element in the list is a ScoreInfo instance."""
        if IGNORE_TESTS:
            return
        for item in self.score.makeScoreInfo("", TRUE_DF, PRED_DF):
            self.assertIsInstance(item, ScoreInfo)

    def test_first_element_aggregation_type_model(self) -> None:
        """First element has aggregation_type == 'model'."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        self.assertEqual(result[0].aggregation_type, "model")

    def test_description_stored(self) -> None:
        """Provided description is stored in every ScoreInfo."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("mymodel", TRUE_DF, PRED_DF)
        for info in result:
            self.assertEqual(info.description, "mymodel")

    def test_model_count_all_valid(self) -> None:
        """Model count equals total cells when no true values are zero."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        self.assertEqual(result[0].count, TRUE_DF.size)

    def test_model_count_excludes_zero_true(self) -> None:
        """Model count excludes cells where true == 0."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [0.0, 1.0], 'B': [1.0, 1.0]},
                index=[0.0, 1.0])
        pred_df = pd.DataFrame({'A': [1.0, 2.0], 'B': [2.0, 2.0]},
                index=[0.0, 1.0])
        result = self.score.makeScoreInfo("", true_df, pred_df)
        self.assertEqual(result[0].count, 4)

    def test_species_aggregation_types(self) -> None:
        """Species entries have aggregation_type equal to their column name."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        for i, col in enumerate(TRUE_DF.columns):
            self.assertEqual(result[i + 1].aggregation_type, col)

    def test_species_count_equals_num_timepoints(self) -> None:
        """Species count equals the number of timepoints."""
        if IGNORE_TESTS:
            return
        result = self.score.makeScoreInfo("", TRUE_DF, PRED_DF)
        for info in result[1:]:
            self.assertEqual(info.count, len(TRUE_DF))

    def test_percentile_ordering(self) -> None:
        """p25 <= p30 <= median <= p75 <= p95 <= p99 for any distribution."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'A': [1.0] * 10}, index=range(10))
        pred_df = pd.DataFrame(
                {'A': [float(i) for i in range(1, 11)]}, index=range(10))
        result = self.score.makeScoreInfo("", true_df, pred_df)
        info = result[0]
        self.assertLessEqual(info.p25, info.p30)
        self.assertLessEqual(info.p30, info.median)
        self.assertLessEqual(info.median, info.p75)
        self.assertLessEqual(info.p75, info.p95)
        self.assertLessEqual(info.p95, info.p99)

    def test_single_species(self) -> None:
        """One-species DataFrame produces a list of length 2."""
        if IGNORE_TESTS:
            return
        true_df = pd.DataFrame({'X': [1.0, 2.0]}, index=[0.0, 1.0])
        pred_df = pd.DataFrame({'X': [2.0, 4.0]}, index=[0.0, 1.0])
        result = self.score.makeScoreInfo("", true_df, pred_df)
        self.assertEqual(len(result), 2)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data not available")
class TestScoreBiomodels(unittest.TestCase):
    """Non-mocked tests using real BioModels and Trajectory.predictLinear."""

    def _checkBiomodel(self, model_num: int) -> None:
        """Run full pipeline for one BioModel and assert valid score statistics."""
        if IGNORE_TESTS:
            return
        iterator = TimecourseIterator()
        model_name = Model.getBiomodelName(model_num)
        timecourse = iterator.getTimecourse(model_name=model_name)
        discovery = SystemDiscovery.makeBiomodel(
                    model_name=model_name, timecourse=timecourse)
        discovery.fit()
        pred_df = discovery.predict()
        score = Score(serialization_path=_temp_csv_path())
        score.addTestResult(timecourse.timecourse_df, pred_df,
                description=f"BIOMD{model_num:010d}")
        # Correct row structure
        n_species = timecourse.model.num_species
        self.assertEqual(len(score.score_df), 1 + n_species)
        # Model row statistics are valid
        model_row = score.score_df[
                score.score_df[AGGREGATION_TYPE] == "model"].iloc[0]
        self.assertTrue(np.isfinite(float(model_row["mean"])))
        self.assertGreaterEqual(float(model_row["mean"]), 0.0)
        self.assertLessEqual(float(model_row["median"]), float(model_row["p75"]))
        self.assertLessEqual(float(model_row["p75"]), float(model_row["p95"]))
        self.assertLessEqual(float(model_row["p95"]), float(model_row["p99"]))
        # Species rows have correct aggregation types
        species_rows = score.score_df[
                score.score_df[AGGREGATION_TYPE] != "model"]
        self.assertEqual(
                list(species_rows[AGGREGATION_TYPE]),
                list(timecourse.timecourse_df.columns))

    def test_biomd8(self) -> None:
        """Full pipeline for BIOMD0000000008."""
        if IGNORE_TESTS:
            return
        self._checkBiomodel(8)

    def test_biomd53(self) -> None:
        """Full pipeline for BIOMD0000000053."""
        if IGNORE_TESTS:
            return
        self._checkBiomodel(53)

    def test_biomd206(self) -> None:
        """Full pipeline for BIOMD0000000206."""
        if IGNORE_TESTS:
            return
        self._checkBiomodel(206)


if __name__ == "__main__":
    unittest.main()
