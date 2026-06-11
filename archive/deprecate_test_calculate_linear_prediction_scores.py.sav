"""Tests for scripts/calculate_linear_prediction_scores.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.calculate_linear_prediction_scores import (  # type: ignore
        _getModelNums, _getChunk, processModels, EXCLUDED_MODELS)
import src.constants as cn  # type: ignore

IGNORE_TESTS = False

MODULE = "scripts.calculate_linear_prediction_scores"

TIMES = [0.0, 1.0, 2.0]
TRUE_DF = pd.DataFrame({'A': [1.0, 2.0, 4.0], 'B': [2.0, 4.0, 8.0]}, index=TIMES)
PRED_DF = 2.0 * TRUE_DF


def _makeMockTrajectory(pred_df: pd.DataFrame = PRED_DF) -> MagicMock:
    trajectory = MagicMock()
    trajectory.timecourse = TRUE_DF
    trajectory.predictLinear.return_value = pred_df
    return trajectory


def _makeMockScore(has_existing: bool = False) -> MagicMock:
    score = MagicMock()
    score._serialization_path = "/tmp/test_scores.csv"
    if has_existing:
        score.score_df = pd.DataFrame({"description": ["BIOMD0000000001"]})
    else:
        score.score_df = pd.DataFrame({"description": []})
    return score


def _mockIterator(items: list) -> MagicMock:
    instance = MagicMock()
    instance.__iter__ = MagicMock(return_value=iter(items))
    return instance


def _makeMockItem(model_name: str) -> MagicMock:
    item = MagicMock()
    item.model_name = model_name
    return item


class TestGetModelNums(unittest.TestCase):

    def test_includesBiomdDirs(self):
        if IGNORE_TESTS:
            return
        entries = ["BIOMD0000000001", "BIOMD0000000003"]
        with patch(f"{MODULE}.os.listdir", return_value=entries), \
                patch(f"{MODULE}.os.path.isdir", return_value=True):
            result = _getModelNums()
        self.assertIn(1, result)
        self.assertIn(3, result)

    def test_excludesNonBiomdNames(self):
        if IGNORE_TESTS:
            return
        entries = ["BIOMD0000000001", "not_a_biomd", "somefile.xml"]
        with patch(f"{MODULE}.os.listdir", return_value=entries), \
                patch(f"{MODULE}.os.path.isdir", return_value=True):
            result = _getModelNums()
        self.assertEqual(result, [1])

    def test_excludesNonDirs(self):
        if IGNORE_TESTS:
            return
        entries = ["BIOMD0000000001", "BIOMD0000000002"]
        def is_dir(path):
            return "0000000001" in path
        with patch(f"{MODULE}.os.listdir", return_value=entries), \
                patch(f"{MODULE}.os.path.isdir", side_effect=is_dir):
            result = _getModelNums()
        self.assertEqual(result, [1])

    def test_returnsSorted(self):
        if IGNORE_TESTS:
            return
        entries = ["BIOMD0000000005", "BIOMD0000000002", "BIOMD0000000010"]
        with patch(f"{MODULE}.os.listdir", return_value=entries), \
                patch(f"{MODULE}.os.path.isdir", return_value=True):
            result = _getModelNums()
        self.assertEqual(result, sorted(result))

    def test_emptyDirectory(self):
        if IGNORE_TESTS:
            return
        with patch(f"{MODULE}.os.listdir", return_value=[]), \
                patch(f"{MODULE}.os.path.isdir", return_value=True):
            result = _getModelNums()
        self.assertEqual(result, [])


class TestGetChunk(unittest.TestCase):

    def setUp(self):
        self.all_model_nums = list(range(1, 11))  # [1..10]

    def test_singleProcess(self):
        if IGNORE_TESTS:
            return
        first, last = _getChunk(self.all_model_nums, 1, 0)
        self.assertEqual(first, 1)
        self.assertEqual(last, 10)

    def test_twoProcessesFirstSlice(self):
        if IGNORE_TESTS:
            return
        first, last = _getChunk(self.all_model_nums, 2, 0)
        self.assertEqual(first, 1)
        self.assertEqual(last, 5)

    def test_twoProcessesSecondSlice(self):
        if IGNORE_TESTS:
            return
        first, last = _getChunk(self.all_model_nums, 2, 1)
        self.assertEqual(first, 6)
        self.assertEqual(last, 10)

    def test_processIndexBeyondModels(self):
        if IGNORE_TESTS:
            return
        first, last = _getChunk(self.all_model_nums, 20, 15)
        self.assertEqual(first, 0)
        self.assertEqual(last, -1)

    def test_unevenSplit(self):
        if IGNORE_TESTS:
            return
        nums = [1, 2, 3]
        first0, last0 = _getChunk(nums, 2, 0)
        first1, last1 = _getChunk(nums, 2, 1)
        self.assertEqual(first0, 1)
        self.assertEqual(last0, 2)
        self.assertEqual(first1, 3)
        self.assertEqual(last1, 3)

    def test_singleModel(self):
        if IGNORE_TESTS:
            return
        first, last = _getChunk([42], 1, 0)
        self.assertEqual(first, 42)
        self.assertEqual(last, 42)

    def test_slicesCoverAllModels(self):
        if IGNORE_TESTS:
            return
        nums = list(range(1, 8))
        num_processes = 3
        covered = set()
        for idx in range(num_processes):
            first, last = _getChunk(nums, num_processes, idx)
            if last >= first:
                covered.update(n for n in nums if first <= n <= last)
        self.assertEqual(covered, set(nums))


class TestProcessModels(unittest.TestCase):

    def test_callsAddTestResult(self):
        if IGNORE_TESTS:
            return
        mock_item = _makeMockItem("BIOMD0000000001")
        mock_score = _makeMockScore()
        mock_trajectory = _makeMockTrajectory()

        with patch(f"{MODULE}.Score", return_value=mock_score), \
                patch(f"{MODULE}.BiomodelsIterator",
                        return_value=_mockIterator([mock_item])), \
                patch(f"{MODULE}.Trajectory.makeBiomodel",
                        return_value=mock_trajectory), \
                patch(f"{MODULE}.os.path.exists", return_value=False):
            processModels(1, 1, 0, 1)

        mock_score.addTestResult.assert_called_once_with(
                TRUE_DF, PRED_DF, description="BIOMD0000000001")

    def test_usesAdjustedJacobianWhenNaNInPrediction(self):
        if IGNORE_TESTS:
            return
        pred_with_nan = PRED_DF.copy()
        pred_with_nan.iloc[0, 0] = np.nan
        pred_clean = PRED_DF.copy()

        mock_item = _makeMockItem("BIOMD0000000001")
        mock_score = _makeMockScore()
        trajectory1 = _makeMockTrajectory(pred_with_nan)
        trajectory2 = _makeMockTrajectory(pred_clean)

        with patch(f"{MODULE}.Score", return_value=mock_score), \
                patch(f"{MODULE}.BiomodelsIterator",
                        return_value=_mockIterator([mock_item])), \
                patch(f"{MODULE}.Trajectory.makeBiomodel",
                        side_effect=[trajectory1, trajectory2]), \
                patch(f"{MODULE}.os.path.exists", return_value=False):
            processModels(1, 1, 0, 1)

        # Second predictLinear call must use is_adjust_fitted_jacobian=True
        trajectory2.predictLinear.assert_called_once_with(
                is_adjust_fitted_jacobian=True)
        mock_score.addTestResult.assert_called_once_with(
                TRUE_DF, pred_clean, description="BIOMD0000000001")

    def test_continuesOnException(self):
        if IGNORE_TESTS:
            return
        item1 = _makeMockItem("BIOMD0000000001")
        item2 = _makeMockItem("BIOMD0000000002")
        mock_score = _makeMockScore()
        ok_trajectory = _makeMockTrajectory()

        def make_trajectory(model_name):
            if model_name == "BIOMD0000000001":
                raise RuntimeError("simulation error")
            return ok_trajectory

        with patch(f"{MODULE}.Score", return_value=mock_score), \
                patch(f"{MODULE}.BiomodelsIterator",
                        return_value=_mockIterator([item1, item2])), \
                patch(f"{MODULE}.Trajectory.makeBiomodel",
                        side_effect=make_trajectory), \
                patch(f"{MODULE}.os.path.exists", return_value=False):
            processModels(1, 2, 0, 1)

        mock_score.addTestResult.assert_called_once_with(
                TRUE_DF, PRED_DF, description="BIOMD0000000002")

    def test_skipsExistingModels(self):
        if IGNORE_TESTS:
            return
        mock_score = _makeMockScore(has_existing=True)  # BIOMD0000000001 already processed
        mock_item = _makeMockItem("BIOMD0000000002")
        mock_trajectory = _makeMockTrajectory()

        with patch(f"{MODULE}.Score", return_value=mock_score), \
                patch(f"{MODULE}.BiomodelsIterator") as mock_bi_class, \
                patch(f"{MODULE}.Trajectory.makeBiomodel",
                        return_value=mock_trajectory), \
                patch(f"{MODULE}.os.path.exists", return_value=True):
            mock_bi_class.return_value = _mockIterator([mock_item])
            processModels(1, 2, 0, 1)

        excluded = mock_bi_class.call_args.kwargs["excluded_models"]
        self.assertIn("BIOMD0000000001", excluded)

    def test_excludedModelsPassedToIterator(self):
        if IGNORE_TESTS:
            return
        mock_score = _makeMockScore()
        mock_trajectory = _makeMockTrajectory()

        with patch(f"{MODULE}.Score", return_value=mock_score), \
                patch(f"{MODULE}.BiomodelsIterator") as mock_bi_class, \
                patch(f"{MODULE}.Trajectory.makeBiomodel",
                        return_value=mock_trajectory), \
                patch(f"{MODULE}.os.path.exists", return_value=False):
            mock_bi_class.return_value = _mockIterator([])
            processModels(1, 1, 0, 1)

        excluded = mock_bi_class.call_args.kwargs["excluded_models"]
        for model in EXCLUDED_MODELS:
            self.assertIn(model, excluded)

    def test_serializationPathIncludesProcessIndex(self):
        if IGNORE_TESTS:
            return
        mock_score = _makeMockScore()

        with patch(f"{MODULE}.Score") as mock_score_class, \
                patch(f"{MODULE}.BiomodelsIterator",
                        return_value=_mockIterator([])), \
                patch(f"{MODULE}.os.path.exists", return_value=False):
            mock_score_class.return_value = mock_score
            processModels(1, 1, 7, 10)

        path_arg = mock_score_class.call_args.kwargs["serialization_path"]
        self.assertIn("7", path_arg)


if __name__ == "__main__":
    unittest.main()
