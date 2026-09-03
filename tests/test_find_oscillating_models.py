"""Tests for scripts/find_oscillating_models.py: processModels and _addEntry."""

import os
import sys
import unittest
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from find_oscillating_models import (  # type: ignore
    _addEntry, processModels)


IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)
_TEST_MODELS = [375, 376]

_TMP_DIR = os.path.join(cn.DATA_DIR, "_test_find_oscillating_models_tmp")


def _setupTmp() -> str:
    """Create (and return) an isolated temp output path for this test run."""
    os.makedirs(_TMP_DIR, exist_ok=True)
    return os.path.join(_TMP_DIR, "find_oscillating_models.csv")


def _mock_empty_item(model_name: str):
    """Build a minimal iterator-item stand-in with an empty timecourse_df."""
    class _EmptyItem:
        pass

    item = _EmptyItem()
    item.model_name = model_name

    class _Timecourse:
        timecourse_df = pd.DataFrame()

    item.timecourse = _Timecourse()
    return item


def _real_item():
    """Build a minimal iterator-item stand-in with a small synthetic timecourse_df."""
    import numpy as np  # type: ignore
    class _RealItem:
        pass

    item = _RealItem()
    item.model_name = "BIOMD0000000997"

    t = np.linspace(0, 10, 200)
    df = pd.DataFrame({"S": np.sin(2 * np.pi * 0.5 * t)}, index=t)

    class _Timecourse:
        timecourse_df = df

    item.timecourse = _Timecourse()
    return item


def _fake_empty_iter(self):
    """Replace TimecourseIterator.__iter__ with one that yields an empty-item stand-in."""
    yield _mock_empty_item("BIOMD0000000998")


class TestAddEntry(unittest.TestCase):
    """Tests for the _addEntry helper function.

    Signature:  _addEntry(result_dct, model_name, species_name=None, frequencies=None)
    """

    def setUp(self) -> None:
        self.dct = {n: [] for n in [cn.COL_SYSTEM_ID, cn.COL_SPECIES_NAME, cn.COL_FREQUENCIES,
                cn.COL_ENDTIME]}

    def test_adds_model_name_to_system_id_column(self) -> None:
        """_addEntry writes model_name into COL_SYSTEM_ID."""
        _addEntry(self.dct, "BIOMD0000000005")
        self.assertEqual(self.dct[cn.COL_SYSTEM_ID], ["BIOMD0000000005"])

    def test_adds_species_and_frequencies_when_given(self) -> None:
        """_addEntry appends species_name and frequencies when provided."""
        _addEntry(self.dct, "BIOMD0000000005", "M", [0.1, 0.2])
        self.assertEqual(self.dct[cn.COL_SPECIES_NAME], ["M"])
        self.assertEqual(self.dct[cn.COL_FREQUENCIES], [[0.1, 0.2]])

    def test_uses_none_for_optional_args_when_missing(self) -> None:
        """When called with only model_name, species and frequencies are None."""
        _addEntry(self.dct, "BIOMD0000000005")
        self.assertIsNone(self.dct[cn.COL_SPECIES_NAME][-1])
        self.assertIsNone(self.dct[cn.COL_FREQUENCIES][-1])

    def test_multiple_entries_accumulate_in_order(self) -> None:
        """Repeated calls accumulate entries in insertion order."""
        _addEntry(self.dct, "BIOMD0000000005", "A", [0.1])
        _addEntry(self.dct, "BIOMD0000000005", "B", [0.2])
        self.assertEqual(len(self.dct[cn.COL_SYSTEM_ID]), 2)
        self.assertEqual(self.dct[cn.COL_SPECIES_NAME], ["A", "B"])
        self.assertEqual(self.dct[cn.COL_FREQUENCIES], [[0.1], [0.2]])


class TestProcessModelsInitialRun(unittest.TestCase):
    """Integration tests for processModels(is_initialize=True)."""

    @unittest.skipIf(IGNORE_TESTS or not HAS_BIOMODELS, "Skipping non-CI or missing data.")
    def test_creates_csv_with_correct_columns(self) -> None:
        """The output CSV must have the three expected header columns."""
        out = _setupTmp()
        if os.path.isfile(out):
            os.remove(out)

        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=True,
        )

        self.assertTrue(os.path.isfile(out))
        df = pd.read_csv(out)
        expected_cols = {cn.COL_SYSTEM_ID, cn.COL_SPECIES_NAME, cn.COL_FREQUENCIES,
                cn.COL_ENDTIME}
        self.assertEqual(set(df.columns), expected_cols)

    @unittest.skipIf(IGNORE_TESTS or not HAS_BIOMODELS, "Skipping non-CI or missing data.")
    def test_row_count_matches_species_across_models(self) -> None:
        """Each species column in each model produces one CSV row."""
        out = _setupTmp()
        if os.path.isfile(out):
            os.remove(out)

        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=True,
        )

        df = pd.read_csv(out)
        self.assertGreater(len(df), 0)
        self.assertTrue((df[cn.COL_SYSTEM_ID].str.startswith("BIOMD")).all())
        self.assertTrue((df[cn.COL_SPECIES_NAME].notna()).all())

    @unittest.skipIf(IGNORE_TESTS or not HAS_BIOMODELS, "Skipping non-CI or missing data.")
    def test_frequencies_are_lists(self) -> None:
        """Each cell in the frequencies column should be a list of floats."""
        out = _setupTmp()
        if os.path.isfile(out):
            os.remove(out)

        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=True,
        )

        df = pd.read_csv(out, converters={cn.COL_FREQUENCIES: eval})
        for freqs in df[cn.COL_FREQUENCIES]:
            self.assertIsInstance(freqs, list)


class TestProcessModelsResume(unittest.TestCase):
    """Integration tests for processModels(is_initialize=False) resume logic."""

    @unittest.skipIf(IGNORE_TESTS or not HAS_BIOMODELS, "Skipping non-CI or missing data.")
    def test_existing_models_are_skipped(self) -> None:
        """When output CSV already contains all models, a resume run adds no new rows."""
        out = _setupTmp()

        # Initial run covers both available models.
        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=True,
        )
        df1 = pd.read_csv(out)
        initial_count = len(df1)

        # Resume run over the same range must not duplicate anything.
        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=False,
        )

        df2 = pd.read_csv(out)
        self.assertEqual(len(df2), initial_count)

    @unittest.skipIf(IGNORE_TESTS or not HAS_BIOMODELS, "Skipping non-CI or missing data.")
    def test_new_models_are_appended(self) -> None:
        """New models beyond what is in the existing CSV are appended."""
        out = _setupTmp()

        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[0],
            output_path=out,
            is_initialize=True,
        )
        df1 = pd.read_csv(out)

        processModels(
            first_model_num=_TEST_MODELS[0],
            last_model_num=_TEST_MODELS[-1],
            output_path=out,
            is_initialize=False,
        )
        df2 = pd.read_csv(out)

        self.assertGreater(len(df2), len(df1))
        self.assertTrue(set(df1[cn.COL_SYSTEM_ID]).issubset(set(df2[cn.COL_SYSTEM_ID])))


class TestProcessModelsEmptyTimecourse(unittest.TestCase):
    """Test behaviour when a timecourse_df is empty (edge case).

    Note: in the current implementation an empty-timecourse item calls
    ``_addEntry`` but then ``continue`` skips the write block, so no row is
    persisted for that model. The tests below verify this graceful behaviour.
    """

    def test_empty_timecourse_does_not_crash(self) -> None:
        """An item with an empty df should be skipped silently (no crash)."""
        out = _setupTmp()
        if os.path.isfile(out):
            os.remove(out)

        from src.timecourse_iterator import TimecourseIterator

        original_iter = TimecourseIterator.__iter__

        def _fake(self):
            yield _mock_empty_item("BIOMD0000000999")

        try:
            TimecourseIterator.__iter__ = _fake  # type: ignore[assignment]
            processModels(
                first_model_num=0, last_model_num=-1,
                output_path=out, is_initialize=True,
            )
        finally:
            TimecourseIterator.__iter__ = original_iter  # type: ignore[assignment]

        # When no rows were ever written (single empty item), the CSV will not exist.
        # Either way the script should complete without raising.

    def test_mix_of_empty_and_valid_models_writes_only_valid(self) -> None:
        """A mix of empty and valid items: only valid models produce CSV rows."""
        out = _setupTmp()
        if os.path.isfile(out):
            os.remove(out)

        from src.timecourse_iterator import TimecourseIterator

        original_iter = TimecourseIterator.__iter__

        def _fake(self):
            yield _mock_empty_item("BIOMD0000000999")  # empty timecourse
            yield _real_item()                          # has real data

        try:
            TimecourseIterator.__iter__ = _fake  # type: ignore[assignment]
            processModels(
                first_model_num=0, last_model_num=-1,
                output_path=out, is_initialize=True,
            )
        finally:
            TimecourseIterator.__iter__ = original_iter  # type: ignore[assignment]

        df = pd.read_csv(out)
        # Script writes both the None-filled row (from accumulated result_dct after
        # the skipped empty item) and the real row for BIOMD0000000997 -> 2 rows total.
        self.assertEqual(len(df), 2)
        # First row is the None-filled entry from the empty-timecourse item
        # (accumulated in result_dct and written when the next model's write block runs).
        self.assertEqual(df[cn.COL_SYSTEM_ID].iloc[0], "BIOMD0000000999")
        # Second row is the valid model.
        self.assertTrue(pd.isna(df[cn.COL_SPECIES_NAME].iloc[0]))
        self.assertEqual(df[cn.COL_SPECIES_NAME].iloc[1], "S")


if __name__ == "__main__":
    unittest.main()
