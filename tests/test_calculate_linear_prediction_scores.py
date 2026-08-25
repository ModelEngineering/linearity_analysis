"""Tests for scripts/calculate_linear_prediction_scores.py"""
import os
import sys
import unittest
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from calculate_linear_prediction_scores import (  # type: ignore
        _getChunk, _getModelNums, processModels)

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

_MODEL_NUMS = [1, 2, 3, 4, 5, 6]


class TestGetChunk(unittest.TestCase):
    """Tests for _getChunk."""

    def test_returns_tuple_of_two(self) -> None:
        """_getChunk returns a 2-tuple."""
        if IGNORE_TESTS:
            return
        result = _getChunk(_MODEL_NUMS, 2, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_single_process_covers_all(self) -> None:
        """One process is assigned every model."""
        if IGNORE_TESTS:
            return
        first, last = _getChunk(_MODEL_NUMS, 1, 1)
        self.assertEqual(first, _MODEL_NUMS[0])
        self.assertEqual(last, _MODEL_NUMS[-1])

    def test_first_process_starts_at_first_model(self) -> None:
        """Process 1 always starts at the first model number."""
        if IGNORE_TESTS:
            return
        first, _ = _getChunk(_MODEL_NUMS, 3, 1)
        self.assertEqual(first, _MODEL_NUMS[0])

    def test_last_process_ends_at_last_model(self) -> None:
        """The final process always ends at the last model number."""
        if IGNORE_TESTS:
            return
        _, last = _getChunk(_MODEL_NUMS, 3, 3)
        self.assertEqual(last, _MODEL_NUMS[-1])

    def test_even_split(self) -> None:
        """Two processes each get half the models when count is even."""
        if IGNORE_TESTS:
            return
        nums = [10, 20, 30, 40]
        first1, last1 = _getChunk(nums, 2, 1)
        first2, last2 = _getChunk(nums, 2, 2)
        self.assertEqual(first1, 10)
        self.assertEqual(last1, 20)
        self.assertEqual(first2, 30)
        self.assertEqual(last2, 40)

    def test_process_index_beyond_models_returns_invalid(self) -> None:
        """A process index with no assigned models returns (0, -1)."""
        if IGNORE_TESTS:
            return
        first, last = _getChunk(_MODEL_NUMS, 10, 10)
        self.assertEqual(first, 0)
        self.assertEqual(last, -1)

    def test_chunks_are_non_overlapping_and_complete(self) -> None:
        """All processes together cover every model exactly once."""
        if IGNORE_TESTS:
            return
        num_processes = 3
        all_assigned = []
        for i in range(1, num_processes + 1):
            first, last = _getChunk(_MODEL_NUMS, num_processes, i)
            if last >= first:
                all_assigned.extend(
                        [n for n in _MODEL_NUMS if first <= n <= last])
        self.assertEqual(all_assigned, _MODEL_NUMS)

    def test_first_less_than_or_equal_to_last_when_valid(self) -> None:
        """first <= last for any in-range process."""
        if IGNORE_TESTS:
            return
        for i in range(1, 4):
            first, last = _getChunk(_MODEL_NUMS, 3, i)
            self.assertLessEqual(first, last)

    def test_returned_values_are_in_model_nums(self) -> None:
        """first and last are elements of all_model_nums."""
        if IGNORE_TESTS:
            return
        for i in range(1, 4):
            first, last = _getChunk(_MODEL_NUMS, 3, i)
            self.assertIn(first, _MODEL_NUMS)
            self.assertIn(last, _MODEL_NUMS)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestGetModelNums(unittest.TestCase):
    """Tests for _getModelNums."""

    def test_returns_list(self) -> None:
        """_getModelNums returns a list."""
        if IGNORE_TESTS:
            return
        result = _getModelNums()
        self.assertIsInstance(result, list)

    def test_elements_are_ints(self) -> None:
        """Every element is an int."""
        if IGNORE_TESTS:
            return
        for num in _getModelNums():
            self.assertIsInstance(num, int)

    def test_all_positive(self) -> None:
        """Every element is a positive integer."""
        if IGNORE_TESTS:
            return
        for num in _getModelNums():
            self.assertGreater(num, 0)

    def test_is_sorted(self) -> None:
        """Result is in ascending order."""
        if IGNORE_TESTS:
            return
        result = _getModelNums()
        self.assertEqual(result, sorted(result))

    def test_nonempty(self) -> None:
        """At least one model is found."""
        if IGNORE_TESTS:
            return
        self.assertGreater(len(_getModelNums()), 0)


if __name__ == "__main__":
    unittest.main()
