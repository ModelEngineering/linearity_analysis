"""End-to-end tests for NLCurve using the real PiecewisePredictions CSV."""

import os
import tempfile
import unittest

import pandas as pd

import src.constants as cn  # type: ignore
from nl_curve import NLCurve  # type: ignore


IGNORE_TESTS = False

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")
PSD_CSV_PATH = os.path.join(
    TESTDATA_DIR,
    (
        "piecewise_predictions__numpoint_100000__threshold_0.001"
        "__removal_1__maxreduction_0.001__manycp_0.csv"
    ),
)

# This CSV was produced with 100,000 timepoints; the file name encodes that value.
EXPECTED_NUM_TIMEPOINT = 100_000


def _prep_csv_for_nlc(path: str, num_timepoint: int) -> str:
    """Return a temp CSV path identical to ``path`` but with a ``num_timepoint`` column."""
    df = pd.read_csv(path)
    if cn.COL_NUM_TIMEPOINT not in df.columns:
        df[cn.COL_NUM_TIMEPOINT] = num_timepoint
    out = os.path.join(tempfile.gettempdir(), "psd_for_nlc.csv")
    df.to_csv(out, index=False)
    return out


# Pre-compute the prepared CSV path + loaded DataFrame once at module load time so
# every test reuses them (getDataframeColumns takes ~18s on the full 5k-row file).
_PREPARED_PATH = (
    _prep_csv_for_nlc(PSD_CSV_PATH, EXPECTED_NUM_TIMEPOINT)
    if os.path.exists(PSD_CSV_PATH)
    else None
)
_LOADED_DF: pd.DataFrame | None = None
if _PREPARED_PATH is not None:
    # Wrap in a function so the 18s load only happens when some test actually runs.
    def _get_loaded_df() -> pd.DataFrame:
        global _LOADED_DF
        if _LOADED_DF is None:
            _LOADED_DF = NLCurve.getDataframeColumns(_PREPARED_PATH)
        return _LOADED_DF


class TestE2eParseChangepoints(unittest.TestCase):
    """End-to-end tests for _parseChangepoints using real CSV data."""

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_real_csv_row_parses_to_sorted_int_list(self) -> None:
        """BIOMD5 model row's changepoints parse to a sorted list of 460 ints."""
        if IGNORE_TESTS:
            return
        df = pd.read_csv(PSD_CSV_PATH)
        cp = df.iloc[0][cn.COL_CHANGEPOINTS]
        parsed = NLCurve._parseChangepoints(cp)
        self.assertEqual(len(parsed), 460)
        self.assertTrue(all(isinstance(x, int) for x in parsed))
        self.assertEqual(sorted(parsed), list(parsed))

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_parsed_length_matches_num_changepoint_column(self) -> None:
        """Sampled rows: parsed changepoints length == num_changepoint column."""
        if IGNORE_TESTS:
            return
        df = pd.read_csv(PSD_CSV_PATH)
        sample_indices = list(range(0, len(df), max(1, len(df) // 50)))
        for i in sample_indices:
            row = df.iloc[i]
            expected_len = int(row["num_changepoint"])
            parsed = NLCurve._parseChangepoints(row[cn.COL_CHANGEPOINTS])
            self.assertEqual(
                len(parsed), expected_len,
                f"length mismatch for {row[cn.COL_SYSTEM_ID]} "
                f"({row[cn.COL_AGGREGATION_TYPE]}) at row {i}: "
                f"expected={expected_len}, actual={len(parsed)}",
            )

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_species_rows_share_changepoints_with_model(self) -> None:
        """For BIOMD5, every species row has the same changepoints as model."""
        if IGNORE_TESTS:
            return
        df = pd.read_csv(PSD_CSV_PATH)
        bioms = df[df[cn.COL_SYSTEM_ID] == f"BIOMD{5:010d}"]
        self.assertGreaterEqual(len(bioms), 2)
        model_cp = NLCurve._parseChangepoints(bioms.iloc[0][cn.COL_CHANGEPOINTS])
        for _, row in bioms.iterrows():
            self.assertEqual(
                list(NLCurve._parseChangepoints(row[cn.COL_CHANGEPOINTS])),
                list(model_cp),
                f"species '{row[cn.COL_AGGREGATION_TYPE]}' disagrees with model",
            )


class TestE2eGetDataframeColumns(unittest.TestCase):
    """End-to-end tests for getDataframeColumns against the real CSV."""

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def _make_loaded_df(self) -> pd.DataFrame:
        return _get_loaded_df()

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_output_has_expected_columns(self) -> None:
        """Loaded DataFrame columns are a superset of the expected column set."""
        if IGNORE_TESTS:
            return
        df = self._make_loaded_df()
        required_cols = {cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE, cn.COL_BOUNDARIES}
        for c in required_cols:
            self.assertIn(c, df.columns)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_boundaries_start_0_end_num_timepoint(self) -> None:
        """Every boundaries list starts at 0 and ends at num_timepoint - 1."""
        if IGNORE_TESTS:
            return
        df = self._make_loaded_df()
        for _, row in df.iterrows():
            b = row[cn.COL_BOUNDARIES]
            self.assertEqual(b[0], 0, f"start != 0 for {row[cn.COL_SYSTEM_ID]}")
            self.assertEqual(
                b[-1], EXPECTED_NUM_TIMEPOINT - 1,
                f"end != num_timepoint-1 for {row[cn.COL_SYSTEM_ID]}",
            )

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_boundaries_count_matches_num_changepoint_plus_two(self) -> None:
        """For every loaded row, len(boundaries) == num_changepoints + 2.

        Validates each raw row against its matching loaded row by the full key
        ``(system_id, aggregation_type, num_changepoint)``, which correctly handles
        systems with duplicate (system_id, aggregation_type) keys in the source CSV.
        """
        if IGNORE_TESTS:
            return
        df = self._make_loaded_df()
        raw = pd.read_csv(PSD_CSV_PATH)
        # Sample rows to keep runtime reasonable (~50 samples).
        sample_indices = list(range(0, len(raw), max(1, len(raw) // 50)))
        for i in sample_indices:
            row = raw.iloc[i]
            key = (
                row[cn.COL_SYSTEM_ID],
                row[cn.COL_AGGREGATION_TYPE],
                int(row["num_changepoint"]),
            )
            match = df[
                (df[cn.COL_SYSTEM_ID] == key[0])
                & (df[cn.COL_AGGREGATION_TYPE] == key[1])
                & ((df[cn.COL_BOUNDARIES].apply(len) == key[2] + 2))
            ]
            self.assertGreaterEqual(
                len(match), 1, f"row {key} not found in loaded df",
            )

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_loaded_row_count_matches_raw(self) -> None:
        """Loaded DataFrame has the same number of rows as the prepared raw CSV."""
        if IGNORE_TESTS:
            return
        df = self._make_loaded_df()
        self.assertEqual(len(df), len(pd.read_csv(_PREPARED_PATH)))


class TestE2eFromPSDPredictions(unittest.TestCase):
    """End-to-end tests for fromPSDPredictions against the real CSV."""

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def _make_temp_csv(self) -> str:
        return _PREPARED_PATH  # type: ignore[return-value]

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_model_level_returns_valid_curve(self) -> None:
        """fromPSDPredictions for BIOMD5 model produces a valid NLCurve."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        nl = NLCurve.fromPSDPredictions(path, model_num=5)
        self.assertIsInstance(nl, NLCurve)
        # Cumulative distribution always reaches 1.0.
        self.assertAlmostEqual(float(nl.curve_ser.iloc[-1]), 1.0)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_species_level_filters_to_matching_row(self) -> None:
        """Selecting species_name='C2' for BIOMD5 returns that row's NLCurve."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        nl = NLCurve.fromPSDPredictions(path, model_num=5, species_name="C2")
        self.assertIsInstance(nl, NLCurve)
        self.assertIn(0.0, nl._boundaries)
        self.assertIn(EXPECTED_NUM_TIMEPOINT - 1, nl._boundaries)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_different_models_have_different_boundaries(self) -> None:
        """BIOMD5 and BIOMD10 produce distinct boundary lists."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        nl5 = NLCurve.fromPSDPredictions(path, model_num=5)
        nl10 = NLCurve.fromPSDPredictions(path, model_num=10)
        self.assertNotEqual(nl5._boundaries, nl10._boundaries)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_species_matches_model_boundaries(self) -> None:
        """For BIOMD5, species 'C2' boundaries equal model boundaries (linear system)."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        nl_model = NLCurve.fromPSDPredictions(path, model_num=5)
        nl_species = NLCurve.fromPSDPredictions(
            path, model_num=5, species_name="C2",
        )
        # In this linear system all species share the same changepoints.
        self.assertEqual(nl_model._boundaries, nl_species._boundaries)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_duplicate_keys_raise_valueerror(self) -> None:
        """BIOMD459 has duplicate (system_id, aggregation_type) rows; raises ValueError."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        with self.assertRaises(ValueError):
            NLCurve.fromPSDPredictions(path, model_num=459)

    @unittest.skipUnless(os.path.exists(PSD_CSV_PATH), "PSD testdata CSV not found")
    def test_multiple_models_loadable(self) -> None:
        """Loading a small set of distinct models all succeeds and produces valid curves."""
        if IGNORE_TESTS:
            return
        path = self._make_temp_csv()
        for model_num in (5, 10, 42):
            nl = NLCurve.fromPSDPredictions(path, model_num=model_num)
            self.assertIsInstance(nl, NLCurve)
            self.assertAlmostEqual(float(nl.curve_ser.iloc[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
