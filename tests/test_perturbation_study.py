"""Tests for scripts/perturbation_study.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd  # type: ignore

import src.constants as cn

IGNORE_TESTS = False

sys.path.insert(0, os.path.join(cn.PROJECT_DIR, "scripts"))

from perturbation_study import (  # type: ignore  # noqa: E402
    MIN_R2,
    PERTURBATIONS,
    THRESHOLD,
    main,
)
from system_discovery import SystemDiscovery  # type: ignore  # noqa: E402

_MODEL_NAME = "test_model"
_NUM_POINT = 20
_SPECIES = ["S1", "S2"]


def _makeTimecourseDF() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    time_arr = np.linspace(0.0, 10.0, _NUM_POINT)
    return pd.DataFrame(
        np.abs(rng.standard_normal((_NUM_POINT, len(_SPECIES)))) + 1.0,
        index=time_arr,
        columns=_SPECIES,
    )


def _makeTimecourse(model_name: str = _MODEL_NAME) -> MagicMock:
    mock = MagicMock()
    mock.model.model_name = model_name
    mock.timecourse_df = _makeTimecourseDF()
    return mock


def _makeIteratorItem(model_name: str = _MODEL_NAME) -> MagicMock:
    item = MagicMock()
    item.model_name = model_name
    item.timecourse = _makeTimecourse(model_name)
    return item


def _makeSourceDF(model_names: list, deg1_max: float = 0.95) -> pd.DataFrame:
    return pd.DataFrame({
        cn.COL_MODEL_NAME: model_names,
        "deg1_max": [deg1_max] * len(model_names),
    })


def _makeFakeSeries(model_name: str) -> pd.Series:
    data: dict = {cn.COL_MODEL_NAME: model_name, "threshold": THRESHOLD}
    for p in PERTURBATIONS:
        base = SystemDiscovery._perturbation_col(p)
        for suffix in ("_min", "_med", "_max"):
            data[f"{base}{suffix}"] = 0.8
    return pd.Series(data)


class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._output_path = os.path.join(self._tmpdir.name, "out.csv")
        self._source_path = os.path.join(self._tmpdir.name, "source.csv")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _runMain(
        self,
        iterator_model_names: list,
        qualifying_model_names: list,
        existing_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        source_df = _makeSourceDF(qualifying_model_names, deg1_max=0.95)
        source_df.to_csv(self._source_path, index=False)
        if existing_df is not None:
            existing_df.to_csv(self._output_path, index=False)

        items = [_makeIteratorItem(name) for name in iterator_model_names]

        def fake_analyze(model, training_df, threshold, perturbations,
                         perturbation_species_fraction, poly_degree, is_plot):
            return _makeFakeSeries(model.model_name)

        with patch("perturbation_study.TimecourseIterator") as mock_iter_cls, \
             patch("perturbation_study.SystemDiscovery.analyzePerturbations",
                   side_effect=fake_analyze), \
             patch("perturbation_study.SOURCE_PATH", self._source_path), \
             patch("perturbation_study.OUTPUT_PATH", self._output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(
                return_value=iter(items)
            )
            return main()

    def test_creates_output_file(self) -> None:
        if IGNORE_TESTS:
            return
        self._runMain(["model_A"], ["model_A"])
        self.assertTrue(os.path.isfile(self._output_path))

    def test_output_has_one_row_per_qualifying_model(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A", "model_B"], ["model_A", "model_B"])
        self.assertEqual(len(df), 2)

    def test_non_qualifying_models_excluded(self) -> None:
        """model_B is in the iterator but not in qualifying list → excluded."""
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A", "model_B"], ["model_A"])
        self.assertIn("model_A", df[cn.COL_MODEL_NAME].values)
        self.assertNotIn("model_B", df[cn.COL_MODEL_NAME].values)

    def test_output_has_model_name_column(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A"], ["model_A"])
        self.assertIn(cn.COL_MODEL_NAME, df.columns)

    def test_output_has_threshold_column(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A"], ["model_A"])
        self.assertIn("threshold", df.columns)

    def test_output_has_r2_columns_for_all_perturbations(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._runMain(["model_A"], ["model_A"])
        for p in PERTURBATIONS:
            base = SystemDiscovery._perturbation_col(p)
            for suffix in ("_min", "_med", "_max"):
                col = f"{base}{suffix}"
                self.assertIn(col, df.columns, msg=f"Missing column {col}")

    def test_resumes_from_existing_csv(self) -> None:
        if IGNORE_TESTS:
            return
        existing = pd.DataFrame([{cn.COL_MODEL_NAME: "model_A", "threshold": THRESHOLD}])
        df = self._runMain(
            ["model_A", "model_B"], ["model_A", "model_B"],
            existing_df=existing,
        )
        self.assertEqual(len(df), 2)
        self.assertIn("model_A", df[cn.COL_MODEL_NAME].values)
        self.assertIn("model_B", df[cn.COL_MODEL_NAME].values)

    def test_already_done_models_are_skipped(self) -> None:
        """model_A in existing CSV → analyzePerturbations not called for it."""
        if IGNORE_TESTS:
            return
        existing = _makeFakeSeries("model_A").to_frame().T
        call_log: list[str] = []

        def fake_analyze(model, training_df, threshold, perturbations,
                         perturbation_species_fraction, poly_degree, is_plot):
            call_log.append(model.model_name)
            return _makeFakeSeries(model.model_name)

        source_df = _makeSourceDF(["model_A", "model_B"], deg1_max=0.95)
        source_df.to_csv(self._source_path, index=False)
        existing.to_csv(self._output_path, index=False)
        items = [_makeIteratorItem("model_A"), _makeIteratorItem("model_B")]

        with patch("perturbation_study.TimecourseIterator") as mock_iter_cls, \
             patch("perturbation_study.SystemDiscovery.analyzePerturbations",
                   side_effect=fake_analyze), \
             patch("perturbation_study.SOURCE_PATH", self._source_path), \
             patch("perturbation_study.OUTPUT_PATH", self._output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            main()

        self.assertNotIn("model_A", call_log)
        self.assertIn("model_B", call_log)


if __name__ == "__main__":
    unittest.main()
