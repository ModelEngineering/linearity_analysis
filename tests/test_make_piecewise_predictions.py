"""Tests for scripts/make_piecewise_predictions.py.

Exercises processModel (with mocked PiecewiseSystemDiscovery) and main()
(with a concrete mock TimecourseIterator that can be both instantiated
and iterated). Heavy ODE fitting is never actually run — the goal is
structural coverage of the orchestration logic: skip-lists, column
augmentation, CSV persistence, exception handling.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import src.constants as cn  # type: ignore
from make_piecewise_predictions import (  # type: ignore
    COEFFICIENT_THRESHOLD,
    MIN_FRACTIONAL_REDUCTION,
    NUM_RANDOM_CHANGPOINTS,
    MAX_CHANGPOINTS,
    processModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_IGNORE_TESTS = False
_NUM_POINTS = 100


def _make_true_df(seed: int = 42) -> pd.DataFrame:
    """Deterministic decay timecourse with two species (non-zero values)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, _NUM_POINTS)
    a = 5 * np.exp(-0.3 * t) + rng.normal(0, 0.01, _NUM_POINTS)
    b = 3 * np.exp(-0.2 * t) + rng.normal(0, 0.01, _NUM_POINTS)
    return pd.DataFrame({"A": a, "B": b}, index=t)


def _make_pred_df(seed: int = 42) -> pd.DataFrame:
    """Predicted DataFrame close to true (noise ~0.05)."""
    rng = np.random.default_rng(seed + 1)
    t = np.linspace(0, 10, _NUM_POINTS)
    a = 5 * np.exp(-0.3 * t) + rng.normal(0, 0.05, _NUM_POINTS)
    b = 3 * np.exp(-0.2 * t) + rng.normal(0, 0.05, _NUM_POINTS)
    return pd.DataFrame({"A": a, "B": b}, index=t)


def _make_mock_item(model_name: str = "BIOMD0000000001") -> mock.Mock:
    item = mock.Mock()
    item.model_name = model_name
    item.timecourse.timecourse_df = _make_true_df()
    return item


class MockPSD:
    """Minimal stand-in for PiecewiseSystemDiscovery that always succeeds."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.training_df = args[0] if args else kwargs.get("training_df")
        self.max_changepoint = kwargs.get("max_changepoint", 2)
        self.min_segment_length = kwargs.get("min_segment_length", 100)
        self.model_name = kwargs.get("model_name", "")

    def fit(self):
        pass

    def predict(self):
        return _make_pred_df()


class FailingPSD(MockPSD):
    """PiecewiseSystemDiscovery stand-in that raises on fit()."""

    def fit(self):
        raise RuntimeError("simulated fitting failure")


def _make_mock_iterator_class(model_names=None):
    """Return a class whose instances yield mock TimecourseIteratorItem objects.

    This replaces ``make_piecewise_predictions.TimecodeIterator`` so that the
    script can both instantiate it (TimecodeIterator(...)) and iterate over
    the instance — something a bare Mock cannot do.
    """
    names = model_names or ["BIOMD0000000001", "BIOMD0000000002"]

    class _MockTimecourseIterator:
        def __init__(self, **kwargs):
            self._items = [_make_mock_item(n) for n in names]

        def __iter__(self):
            return iter(self._items)

    return _MockTimecourseIterator


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------

class TestModuleConstants(unittest.TestCase):
    """Verify module-level constants are sensible defaults."""

    def test_max_changepoints_is_list_of_ints(self) -> None:
        self.assertIsInstance(MAX_CHANGPOINTS, list)
        for v in MAX_CHANGPOINTS:
            self.assertIsInstance(v, int)

    def test_num_random_changepoints(self) -> None:
        self.assertEqual(NUM_RANDOM_CHANGPOINTS, 5)

    def test_min_fractional_reduction_default_is_zero(self) -> None:
        # The script sets MIN_FRACTIONAL_REDUCTION = 0.0
        self.assertAlmostEqual(MIN_FRACTIONAL_REDUCTION, 0.0)

    def test_coefficient_threshold_positive(self) -> None:
        self.assertGreater(COEFFICIENT_THRESHOLD, 0.0)


# ---------------------------------------------------------------------------
# processModel tests
# ---------------------------------------------------------------------------

class TestProcessModel(unittest.TestCase):
    """Tests for the processModel function."""

    @mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery", MockPSD)
    def test_returns_dataframe_on_success(self) -> None:
        if _IGNORE_TESTS:
            return
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
            existing_model_names=set(),
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    @mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery", MockPSD)
    def test_augmented_columns_present(self) -> None:
        if _IGNORE_TESTS:
            return
        item = _make_mock_item(model_name="BIOMD0000000012")
        result = processModel(
            item=item,
            max_changepoint=3,
            min_segment_length=75,
            coefficient_threshold=0.005,
            existing_model_names=set(),
            is_random_changepoints=True,
            min_fractional_reduction=MIN_FRACTIONAL_REDUCTION,
        )
        required_cols = {
            cn.COL_SYSTEM_ID,
            cn.COL_NUM_CHANGEPOINT,
            cn.COL_MIN_SEGMENT_LENGTH,
            cn.COL_MIN_FRACTIONAL_REDUCTION,
            cn.COL_IS_RANDOM_CHANGPOINTS,
            cn.COL_COEFFICIENT_THRESHOLD,
        }
        self.assertTrue(
            required_cols.issubset(set(result.columns)),
            msg=f"Missing columns: {required_cols - set(result.columns)}",
        )

    @mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery", MockPSD)
    def test_augmented_columns_have_correct_values(self) -> None:
        if _IGNORE_TESTS:
            return
        item = _make_mock_item(model_name="BIOMD0000009999")
        result = processModel(
            item=item,
            max_changepoint=5,
            min_segment_length=200,
            coefficient_threshold=0.1,
            existing_model_names=set(),
            is_random_changepoints=False,
            min_fractional_reduction=0.25,
        )
        self.assertEqual(result[cn.COL_SYSTEM_ID].iloc[0], "BIOMD0000009999")
        self.assertTrue((result[cn.COL_NUM_CHANGEPOINT] == 5).all())
        self.assertTrue((result[cn.COL_MIN_SEGMENT_LENGTH] == 200).all())
        self.assertTrue((result[cn.COL_COEFFICIENT_THRESHOLD] == 0.1).all())
        self.assertTrue(result[cn.COL_IS_RANDOM_CHANGPOINTS].eq(False).all())

    @mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery", FailingPSD)
    def test_exception_returns_none(self) -> None:
        if _IGNORE_TESTS:
            return
        item = _make_mock_item()
        result = processModel(
            item=item, max_changepoint=0, min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
            existing_model_names=set(),
        )
        self.assertIsNone(result)

    @mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery")
    def test_returns_none_when_predict_returns_none(self, MockPSD_cls):
        if _IGNORE_TESTS:
            return
        instance = mock.Mock(spec=[])
        instance.fit = mock.Mock()
        instance.predict = mock.Mock(return_value=None)
        MockPSD_cls.return_value = instance

        item = _make_mock_item()
        result = processModel(
            item=item, max_changepoint=0, min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
            existing_model_names=set(),
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# main() tests — full integration with mocked iterators + real file I/O
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """Integration-style tests for the main orchestration function."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- helper to run main() with mocks in place -----------------------------

    def _run_main(
        self,
        model_names=None,
        output_path=None,
        is_initialize=True,
        **kwargs,
    ):
        """Invoke ``main()`` with heavy deps mocked, returning the CSV path."""
        out = output_path or os.path.join(self.tmp_dir, "out.csv")
        with mock.patch(
            "make_piecewise_predictions.TimecourseIterator",
            _make_mock_iterator_class(model_names),
        ), mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery", MockPSD):
            from make_piecewise_predictions import main as _main  # noqa: E402
            _main(
                first_model_num=0,
                last_model_num=1_000_000,
                is_initialize=is_initialize,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
                min_segment_length=50,
                min_fractional_reduction=MIN_FRACTIONAL_REDUCTION,
                output_path=out,
                **kwargs,
            )
        return out

    # -- individual test methods -----------------------------------------------

    def test_creates_csv_when_no_existing_output(self):
        """main() creates the output CSV even when no prior file exists."""
        if _IGNORE_TESTS:
            return
        out = self._run_main()
        self.assertTrue(os.path.isfile(out))
        df = pd.read_csv(out)
        self.assertFalse(df.empty)

    def test_output_contains_expected_columns(self):
        """CSV has the augmented metadata columns plus score columns."""
        if _IGNORE_TESTS:
            return
        out = self._run_main()
        df = pd.read_csv(out)
        expected_cols = {
            cn.COL_SYSTEM_ID,
            cn.COL_NUM_CHANGEPOINT,
            cn.COL_MIN_SEGMENT_LENGTH,
            cn.COL_MIN_FRACTIONAL_REDUCTION,
            cn.COL_IS_RANDOM_CHANGPOINTS,
            cn.COL_COEFFICIENT_THRESHOLD,
        }
        self.assertTrue(
            expected_cols.issubset(set(df.columns)),
            msg=f"Missing columns: {expected_cols - set(df.columns)}",
        )

    def test_output_sorted_by_num_changepoint(self):
        """Rows are ordered by (system_id, num_changepoint) ascending."""
        if _IGNORE_TESTS:
            return
        out = self._run_main()
        df = pd.read_csv(out)
        pairs = list(zip(df[cn.COL_SYSTEM_ID].tolist(), df[cn.COL_NUM_CHANGEPOINT].tolist()))
        self.assertEqual(pairs, sorted(pairs))

    def test_both_random_and_deterministic_rows_present(self):
        """For each changepoint/model pair, both is_random flag values exist."""
        if _IGNORE_TESTS:
            return
        out = self._run_main()
        df = pd.read_csv(out)
        for (cp, sys_id), grp in df.groupby([cn.COL_NUM_CHANGEPOINT, cn.COL_SYSTEM_ID]):
            flags = set(grp[cn.COL_IS_RANDOM_CHANGPOINTS].tolist())
            self.assertEqual(
                flags, {True, False},
                msg=f"Missing flag variety at cp={cp} model={sys_id}: got {flags}",
            )

    def test_skips_already_processed_model(self):
        """If a model appears in the existing CSV (model_name col), it is skipped."""
        if _IGNORE_TESTS:
            return
        out = os.path.join(self.tmp_dir, "out.csv")
        # Pre-populate with one row for BIOMD0000000001 (must include `model_name` col).
        init_df = pd.DataFrame([{cn.COL_MODEL_NAME: "BIOMD0000000001"}])
        init_df.to_csv(out, index=False)

        with mock.patch(
            "make_piecewise_predictions.TimecourseIterator",
            _make_mock_iterator_class(["BIOMD0000000001"]),
        ), mock.patch("make_piecewise_predictions.PiecewiseSystemDiscovery") as MockPSD_cls:
            from make_piecewise_predictions import main as _main
            # is_initialize=False so the existing CSV gets read.
            _main(first_model_num=0, last_model_num=1_000_000, output_path=out)

        self.assertEqual(MockPSD_cls.call_count, 0)


if __name__ == "__main__":
    unittest.main()
