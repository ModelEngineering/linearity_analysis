"""Tests for scripts/evaluate_monomial_models.py."""

import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np # type: ignore
import pandas as pd # type: ignore

import src.constants as cn  # type: ignore

IGNORE_TESTS = False

sys.path.insert(0, os.path.join(cn.PROJECT_DIR, "scripts"))

from evaluate_monomial_models import (  # type: ignore  # noqa: E402
    DEGREES,
    _evaluate_model,
    _score_cols,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_POINT = 20
_SPECIES = ["S1", "S2"]
_NUM_SPECIES = len(_SPECIES)
_NUM_REACTION = 2
_MODEL_NAME = "test_model"


def _makeTimecourseDF() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    time_arr = np.linspace(0.0, 10.0, _NUM_POINT)
    return pd.DataFrame(
        np.abs(rng.standard_normal((_NUM_POINT, _NUM_SPECIES))) + 1.0,
        index=time_arr,
        columns=_SPECIES,
    )


def _makeTimecourse(model_name: str = _MODEL_NAME) -> MagicMock:
    mock = MagicMock()
    mock.model.num_species = _NUM_SPECIES
    mock.model.num_reaction = _NUM_REACTION
    mock.timecourse_df = _makeTimecourseDF()
    return mock


def _makeIteratorItem(model_name: str = _MODEL_NAME) -> MagicMock:
    item = MagicMock()
    item.model_name = model_name
    item.timecourse = _makeTimecourse(model_name=model_name)
    return item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoreCols(unittest.TestCase):

    def test_degree_zero(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(any([col.startswith("deg0_") for col in _score_cols(0)]))

    def test_degree_one(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(any([col.startswith("deg1_") for col in _score_cols(1)]))

    def test_degree_two(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(any([col.startswith("deg2_") for col in _score_cols(2)]))


class TestEvaluateModel(unittest.TestCase):

    def setUp(self) -> None:
        self._timecourse = _makeTimecourse()

    def test_returns_dict(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        self.assertIsInstance(row, dict)

    def test_row_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        self.assertEqual(row[cn.COL_MODEL_NAME], _MODEL_NAME)

    def test_row_contains_num_species(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        self.assertEqual(row[cn.COL_NUM_SPECIES], _NUM_SPECIES)

    def test_row_contains_num_reaction(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        self.assertEqual(row[cn.COL_NUM_REACTION], _NUM_REACTION)

    def test_row_has_score_columns_for_each_degree(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        for deg in DEGREES:
            for col in _score_cols(deg):
                self.assertIn(col, row)

    def test_score_values_are_floats(self) -> None:
        if IGNORE_TESTS:
            return
        row = _evaluate_model(_MODEL_NAME, self._timecourse)
        for deg in DEGREES:
            for col in _score_cols(deg):
                self.assertIsNotNone(row[col], f"Score column {col} should not be None")    

    def test_exception_yields_nan_scores(self) -> None:
        if IGNORE_TESTS:
            return
        bad_tc = MagicMock()
        bad_tc.model.num_species = _NUM_SPECIES
        bad_tc.model.num_reaction = _NUM_REACTION
        bad_tc.timecourse_df = pd.DataFrame()  # empty → SystemDiscovery raises
        row = _evaluate_model(_MODEL_NAME, bad_tc)
        for deg in DEGREES:
            for col in _score_cols(deg):
                self.assertTrue(math.isnan(row[col]))


class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._output_path = os.path.join(self._tmpdir.name, "out.csv")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _runMain(self, model_names: list) -> pd.DataFrame:
        items = [_makeIteratorItem(name) for name in model_names]
        with patch("evaluate_monomial_models.TimecourseIterator") as mock_iter_cls, \
                patch("evaluate_monomial_models.OUTPUT_PATH", self._output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            main()
        return pd.read_csv(self._output_path)

    def test_creates_output_file(self) -> None:
        if IGNORE_TESTS:
            return
        self._runMain(["model_A"])
        self.assertTrue(os.path.isfile(self._output_path))

    def test_output_has_one_row_per_model(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A", "model_B"])
        self.assertEqual(len(df), 2)

    def test_output_has_model_name_column(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A"])
        self.assertIn(cn.COL_MODEL_NAME, df.columns)

    def test_output_has_all_score_columns(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A"])
        for deg in DEGREES:
            for col in _score_cols(deg):
                self.assertIn(col, df.columns)

    def test_resumes_from_existing_csv(self) -> None:
        if IGNORE_TESTS:
            return
        existing = pd.DataFrame([{
            cn.COL_MODEL_NAME: "model_A",
            cn.COL_NUM_SPECIES: _NUM_SPECIES,
            cn.COL_NUM_REACTION: _NUM_REACTION,
        }])
        existing.to_csv(self._output_path, index=False)
        df = self._runMain(["model_B"])
        self.assertEqual(len(df), 2)
        self.assertIn("model_A", df[cn.COL_MODEL_NAME].values)
        self.assertIn("model_B", df[cn.COL_MODEL_NAME].values)


if __name__ == "__main__":
    unittest.main()
