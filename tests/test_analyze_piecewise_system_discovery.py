"""Tests for scripts/analyze_piecewise_system_discovery.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import src.constants as cn

IGNORE_TESTS = False

sys.path.insert(0, os.path.join(cn.PROJECT_DIR, "scripts"))

from analyze_piecewise_system_discovery import (  # type: ignore  # noqa: E402
    COL_MODEL_NAME,
    COL_NUM_NONZERO_TERM,
    COL_SCORE_MAX,
    COL_SCORE_MEDIAN,
    COL_SCORE_MIN,
    _output_path,
    main,
)

_OUTPUT_COLUMNS = [COL_MODEL_NAME, COL_SCORE_MIN, COL_SCORE_MEDIAN,
                   COL_SCORE_MAX, COL_NUM_NONZERO_TERM]


def _makeScoreInfo(min_val=0.1, median_val=0.5, max_val=0.9, num_nonzero=3):
    info = MagicMock()
    info.min = min_val
    info.median = median_val
    info.max = max_val
    info.num_nonzero_term = num_nonzero
    return info


def _makeIteratorItem(model_name: str) -> MagicMock:
    item = MagicMock()
    item.model_name = model_name
    item.timecourse = MagicMock()
    return item


def _patchMain(tmpdir: str, iterator_model_names: list, num_change_point: int = 2,
               error_models: set | None = None, existing_df: pd.DataFrame | None = None,
               is_initialize: bool = False):
    """Patch TimecourseIterator and PiecewiseSystemDiscovery, run main(), return result df."""
    output_path = os.path.join(tmpdir, f"scores-{num_change_point}.csv")
    if existing_df is not None:
        existing_df.to_csv(output_path, index=False)

    error_models = error_models or set()
    items = [_makeIteratorItem(name) for name in iterator_model_names]

    def fake_psd_init(timecourse, num_change_point=2, **kwargs):
        mock = MagicMock()
        mock.fit.return_value = mock
        mock.score.return_value = _makeScoreInfo()
        return mock

    with patch("analyze_piecewise_system_discovery.TimecourseIterator") as mock_iter_cls, \
         patch("analyze_piecewise_system_discovery.PiecewiseSystemDiscovery",
               side_effect=fake_psd_init), \
         patch("analyze_piecewise_system_discovery._output_path",
               return_value=output_path):
        mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
        return main(num_change_point=num_change_point, is_initialize=is_initialize)


class TestOutputPath(unittest.TestCase):

    def test_encodes_num_change_point_in_filename(self) -> None:
        if IGNORE_TESTS:
            return
        path = _output_path(3)
        self.assertIn("3", os.path.basename(path))

    def test_different_num_change_points_produce_different_paths(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertNotEqual(_output_path(1), _output_path(2))

    def test_output_is_under_data_dir(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(_output_path(2).startswith(cn.DATA_DIR))


class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run(self, model_names, *, num_change_point=2, error_models=None,
             existing_df=None, is_initialize=False):
        return _patchMain(
            self._tmpdir.name, model_names,
            num_change_point=num_change_point,
            error_models=error_models,
            existing_df=existing_df,
            is_initialize=is_initialize,
        )

    def test_creates_output_csv_file(self) -> None:
        if IGNORE_TESTS:
            return
        output_path = os.path.join(self._tmpdir.name, "scores-2.csv")
        self._run(["model_A"])
        self.assertTrue(os.path.isfile(output_path))

    def test_output_has_one_row_per_model(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._run(["model_A", "model_B"])
        self.assertEqual(len(df), 2)

    def test_output_has_expected_columns(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._run(["model_A"])
        for col in _OUTPUT_COLUMNS:
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")

    def test_score_fields_are_written(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._run(["model_A"])
        row = df.iloc[0]
        self.assertAlmostEqual(row[COL_SCORE_MIN], 0.1)
        self.assertAlmostEqual(row[COL_SCORE_MEDIAN], 0.5)
        self.assertAlmostEqual(row[COL_SCORE_MAX], 0.9)
        self.assertEqual(row[COL_NUM_NONZERO_TERM], 3)

    def test_empty_iterator_returns_empty_dataframe(self) -> None:
        if IGNORE_TESTS:
            return
        df = self._run([])
        self.assertEqual(len(df), 0)

    def test_models_with_errors_are_skipped(self) -> None:
        if IGNORE_TESTS:
            return
        items = [_makeIteratorItem("model_A"), _makeIteratorItem("model_B")]

        def fake_psd_init(timecourse, num_change_point=2, **kwargs):
            mock = MagicMock()
            if timecourse is items[0].timecourse:
                mock.fit.side_effect = RuntimeError("boom")
            else:
                mock.fit.return_value = mock
                mock.score.return_value = _makeScoreInfo()
            return mock

        output_path = os.path.join(self._tmpdir.name, "scores-2.csv")
        with patch("analyze_piecewise_system_discovery.TimecourseIterator") as mock_iter_cls, \
             patch("analyze_piecewise_system_discovery.PiecewiseSystemDiscovery",
                   side_effect=fake_psd_init), \
             patch("analyze_piecewise_system_discovery._output_path",
                   return_value=output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            df = main(num_change_point=2)

        self.assertEqual(len(df), 1)
        self.assertIn("model_B", df[COL_MODEL_NAME].values)
        self.assertNotIn("model_A", df[COL_MODEL_NAME].values)

    def test_resumes_from_existing_csv(self) -> None:
        if IGNORE_TESTS:
            return
        existing = pd.DataFrame([{
            COL_MODEL_NAME: "model_A",
            COL_SCORE_MIN: 0.1, COL_SCORE_MEDIAN: 0.5,
            COL_SCORE_MAX: 0.9, COL_NUM_NONZERO_TERM: 3,
        }])
        df = self._run(["model_A", "model_B"], existing_df=existing)
        self.assertEqual(len(df), 2)
        self.assertIn("model_A", df[COL_MODEL_NAME].values)
        self.assertIn("model_B", df[COL_MODEL_NAME].values)

    def test_already_done_models_are_not_reprocessed(self) -> None:
        if IGNORE_TESTS:
            return
        existing = pd.DataFrame([{
            COL_MODEL_NAME: "model_A",
            COL_SCORE_MIN: 0.1, COL_SCORE_MEDIAN: 0.5,
            COL_SCORE_MAX: 0.9, COL_NUM_NONZERO_TERM: 3,
        }])
        call_log: list[str] = []
        items = [_makeIteratorItem("model_A"), _makeIteratorItem("model_B")]

        def fake_psd_init(timecourse, num_change_point=2, **kwargs):
            for item in items:
                if timecourse is item.timecourse:
                    call_log.append(item.model_name)
            mock = MagicMock()
            mock.fit.return_value = mock
            mock.score.return_value = _makeScoreInfo()
            return mock

        output_path = os.path.join(self._tmpdir.name, "scores-2.csv")
        existing.to_csv(output_path, index=False)
        with patch("analyze_piecewise_system_discovery.TimecourseIterator") as mock_iter_cls, \
             patch("analyze_piecewise_system_discovery.PiecewiseSystemDiscovery",
                   side_effect=fake_psd_init), \
             patch("analyze_piecewise_system_discovery._output_path",
                   return_value=output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            main(num_change_point=2)

        self.assertNotIn("model_A", call_log)
        self.assertIn("model_B", call_log)

    def test_initialize_ignores_existing_csv(self) -> None:
        if IGNORE_TESTS:
            return
        existing = pd.DataFrame([{
            COL_MODEL_NAME: "model_A",
            COL_SCORE_MIN: 0.1, COL_SCORE_MEDIAN: 0.5,
            COL_SCORE_MAX: 0.9, COL_NUM_NONZERO_TERM: 3,
        }])
        call_log: list[str] = []
        items = [_makeIteratorItem("model_A"), _makeIteratorItem("model_B")]

        def fake_psd_init(timecourse, num_change_point=2, **kwargs):
            for item in items:
                if timecourse is item.timecourse:
                    call_log.append(item.model_name)
            mock = MagicMock()
            mock.fit.return_value = mock
            mock.score.return_value = _makeScoreInfo()
            return mock

        output_path = os.path.join(self._tmpdir.name, "scores-2.csv")
        existing.to_csv(output_path, index=False)
        with patch("analyze_piecewise_system_discovery.TimecourseIterator") as mock_iter_cls, \
             patch("analyze_piecewise_system_discovery.PiecewiseSystemDiscovery",
                   side_effect=fake_psd_init), \
             patch("analyze_piecewise_system_discovery._output_path",
                   return_value=output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            main(num_change_point=2, is_initialize=True)

        self.assertIn("model_A", call_log)

    def test_num_change_point_passed_to_piecewise_system_discovery(self) -> None:
        if IGNORE_TESTS:
            return
        received: list[int] = []
        items = [_makeIteratorItem("model_A")]

        def fake_psd_init(timecourse, num_change_point=2, **kwargs):
            received.append(num_change_point)
            mock = MagicMock()
            mock.fit.return_value = mock
            mock.score.return_value = _makeScoreInfo()
            return mock

        output_path = os.path.join(self._tmpdir.name, "scores-5.csv")
        with patch("analyze_piecewise_system_discovery.TimecourseIterator") as mock_iter_cls, \
             patch("analyze_piecewise_system_discovery.PiecewiseSystemDiscovery",
                   side_effect=fake_psd_init), \
             patch("analyze_piecewise_system_discovery._output_path",
                   return_value=output_path):
            mock_iter_cls.return_value.__iter__ = MagicMock(return_value=iter(items))
            main(num_change_point=5)

        self.assertEqual(received, [5])


if __name__ == "__main__":
    unittest.main()
