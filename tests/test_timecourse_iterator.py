"""Tests for TimecourseIterator and TimecourseIteratorItem."""

import os
import pickle
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd  # type: ignore

import src.constants as cn
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator, TimecourseIteratorItem  # type: ignore

IGNORE_TESTS = False

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""
NUM_POINT = 11
NUM_SPECIES = 2
MODEL_NAME = "test_model"
FIRST_REAL_MODEL = "BIOMD0000000005"
HAS_REAL_ZIP = os.path.isfile(cn.TIMECOURSE_ZIP_PATH)


def _makeModel(model_name: str = MODEL_NAME) -> Model:
    return Model(ANTIMONY_MODEL, model_name=model_name)


def _makeTimecourse(model_name: str = MODEL_NAME) -> Timecourse:
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
        rng.standard_normal((NUM_POINT, NUM_SPECIES)),
        index=timepoint_arr,
        columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    return Timecourse(
        model=_makeModel(model_name=model_name),
        timecourse_df=timecourse_df,
    )


def _makeZipWithTimecourses(zip_path: str, model_names: list) -> None:
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for name in model_names:
            tc = _makeTimecourse(model_name=name)
            dct = {
                "model": tc.model,
                "start_time": tc.start_time,
                "end_time": tc.end_time,
                "num_point": tc.num_point,
                "timecourse_df": tc._timecourse_df,
            }
            zf.writestr(f"{name}_timecourse.pkl", pickle.dumps(dct))


class TestTimecourseIteratorItem(unittest.TestCase):
    """Tests for TimecourseIteratorItem."""

    def test_stores_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        item = TimecourseIteratorItem(model_name=MODEL_NAME, timecourse=tc)
        self.assertEqual(item.model_name, MODEL_NAME)

    def test_stores_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        item = TimecourseIteratorItem(model_name=MODEL_NAME, timecourse=tc)
        self.assertIs(item.timecourse, tc)


class TestTimecourseIteratorInit(unittest.TestCase):
    """Tests for TimecourseIterator.__init__."""

    def test_default_zip_path(self) -> None:
        if IGNORE_TESTS:
            return
        it = TimecourseIterator()
        self.assertEqual(it.zip_path, cn.TIMECOURSE_ZIP_PATH)

    def test_custom_zip_path(self) -> None:
        if IGNORE_TESTS:
            return
        it = TimecourseIterator(zip_path="/some/path.zip")
        self.assertEqual(it.zip_path, "/some/path.zip")


class TestTimecourseIteratorIter(unittest.TestCase):
    """Tests for TimecourseIterator.__iter__ using a synthetic zip."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._zip_path = os.path.join(self._tmpdir.name, "test.zip")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _makeIterator(self, model_names: list) -> TimecourseIterator:
        _makeZipWithTimecourses(self._zip_path, model_names)
        return TimecourseIterator(zip_path=self._zip_path)

    def test_yields_timecourse_iterator_items(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertTrue(all(isinstance(item, TimecourseIteratorItem) for item in items))

    def test_model_name_extracted_from_filename(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].model_name, "model_A")

    def test_yields_timecourse_instances(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertIsInstance(items[0].timecourse, Timecourse)

    def test_skips_non_timecourse_pkl_files(self) -> None:
        if IGNORE_TESTS:
            return
        _makeZipWithTimecourses(self._zip_path, ["model_A"])
        with zipfile.ZipFile(self._zip_path, 'a') as zf:
            zf.writestr("README.txt", "skip me")
            zf.writestr("model_B_other.pkl", b"skip me too")
        items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].model_name, "model_A")

    def test_yields_in_sorted_order(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_C", "model_A", "model_B"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, sorted(names))

    def test_multiple_models_all_yielded(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_B"]))
        self.assertEqual(len(items), 2)

    def test_empty_zip_yields_no_items(self) -> None:
        if IGNORE_TESTS:
            return
        with zipfile.ZipFile(self._zip_path, 'w') as _:
            pass
        items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(items, [])


class TestTimecourseIteratorGetTimecourse(unittest.TestCase):
    """Tests for TimecourseIterator.getTimecourse using a synthetic zip."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._zip_path = os.path.join(self._tmpdir.name, "test.zip")
        _makeZipWithTimecourses(self._zip_path, ["model_A", "model_B"])

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _iterator(self) -> TimecourseIterator:
        return TimecourseIterator(zip_path=self._zip_path)

    def test_returns_timecourse_for_valid_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = self._iterator().getTimecourse("model_A")
        self.assertIsInstance(tc, Timecourse)

    def test_raises_for_unknown_model(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(KeyError):
            self._iterator().getTimecourse("does_not_exist")

    def test_timecourse_df_is_nonempty(self) -> None:
        if IGNORE_TESTS:
            return
        tc = self._iterator().getTimecourse("model_A")
        self.assertFalse(tc._timecourse_df.empty)

    def test_can_get_each_model_independently(self) -> None:
        if IGNORE_TESTS:
            return
        it = self._iterator()
        tc_a = it.getTimecourse("model_A")
        tc_b = it.getTimecourse("model_B")
        self.assertEqual(tc_a.model.model_name, "model_A")
        self.assertEqual(tc_b.model.model_name, "model_B")


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestTimecourseIteratorRealZip(unittest.TestCase):
    """Integration tests using the real timecourse zip at cn.TIMECOURSE_ZIP_PATH."""

    def _first_item(self) -> TimecourseIteratorItem:
        return next(iter(TimecourseIterator()))

    def test_yields_timecourse_iterator_item(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._first_item(), TimecourseIteratorItem)

    def test_model_name_is_nonempty_string(self) -> None:
        if IGNORE_TESTS:
            return
        item = self._first_item()
        self.assertIsInstance(item.model_name, str)
        self.assertGreater(len(item.model_name), 0)

    def test_first_model_is_sorted_first(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(self._first_item().model_name, FIRST_REAL_MODEL)

    def test_timecourse_is_correct_type(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._first_item().timecourse, Timecourse)

    def test_zip_contains_many_timecourses(self) -> None:
        if IGNORE_TESTS:
            return
        with zipfile.ZipFile(cn.TIMECOURSE_ZIP_PATH) as zf:
            count = sum(1 for n in zf.namelist() if n.endswith('_timecourse.pkl'))
        self.assertGreater(count, 1)


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestTimecourseIteratorGetTimecourseRealZip(unittest.TestCase):
    """Integration tests for getTimecourse using the real zip."""

    def test_returns_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = TimecourseIterator().getTimecourse(FIRST_REAL_MODEL)
        self.assertIsInstance(tc, Timecourse)

    def test_timecourse_df_is_nonempty(self) -> None:
        if IGNORE_TESTS:
            return
        tc = TimecourseIterator().getTimecourse(FIRST_REAL_MODEL)
        self.assertFalse(tc._timecourse_df.empty)


if __name__ == "__main__":
    unittest.main()
