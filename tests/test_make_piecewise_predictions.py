"""Tests for scripts/make_piecewise_predictions.py."""

import os
import sys
import tempfile
import unittest

import matplotlib  # noqa: F401 -- non-interactive backend needed before pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import src.constants as cn  # type: ignore
from make_piecewise_predictions import (  # type: ignore
    processModel,
    main,
    COEFFICIENT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_POINT = 200  # enough for piecewise fitting; small fixture uses this.


def _make_linear_df(
    n_points: int = NUM_POINT,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a simple linear ODE timecourse for testing."""
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


def _make_mock_item(model_name: str = "BIOMD0000000001") -> MagicMock:
    """Build a mock TimecourseIteratorItem with a synthetic timecourse."""
    item = MagicMock()
    item.model_name = model_name
    df = _make_linear_df(n_points=NUM_POINT)
    item.timecourse.timecourse_df = df
    return item


# ---------------------------------------------------------------------------
# processModel tests
# ---------------------------------------------------------------------------


class TestProcessModel(unittest.TestCase):

    def test_returns_dataframe_on_success(self) -> None:
        """processModel returns a DataFrame when the fit/predict pipeline succeeds."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """processModel augments the score DataFrame with all required metadata columns."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            min_segment_length=75,
            coefficient_threshold=0.002,
        )
        self.assertIn(cn.COL_SYSTEM_ID, result.columns)
        self.assertIn(cn.COL_MAX_CHANGEPOINT, result.columns)
        self.assertIn(cn.COL_MIN_SEGMENT_LENGTH, result.columns)
        self.assertIn(cn.COL_MAX_FRACTIONAL_REDUCTION, result.columns)
        self.assertIn(cn.COL_COEFFICIENT_THRESHOLD, result.columns)

    def test_metadata_values_match_parameters(self) -> None:
        """Metadata columns hold the exact parameter values passed to processModel."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=5,
            min_segment_length=100,
            coefficient_threshold=0.003,
            max_fractional_reduction=0.25,
        )
        self.assertEqual(result[cn.COL_MAX_CHANGEPOINT].iloc[0], 5)
        self.assertEqual(result[cn.COL_MIN_SEGMENT_LENGTH].iloc[0], 100)
        self.assertAlmostEqual(result[cn.COL_COEFFICIENT_THRESHOLD].iloc[0], 0.003)
        self.assertAlmostEqual(result[cn.COL_MAX_FRACTIONAL_REDUCTION].iloc[0], 0.25)

    def test_system_id_matches_model_name(self) -> None:
        """The COL_SYSTEM_ID column is set to the item model name."""
        item = _make_mock_item(model_name="BIOMD0000009999")
        result = processModel(
            item=item,
            max_changepoint=0,
            min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        self.assertEqual(result[cn.COL_SYSTEM_ID].iloc[0], "BIOMD0000009999")

    def test_aggregation_type_rows_present(self) -> None:
        """Score output contains both model and per-species aggregation rows."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            min_segment_length=50,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        self.assertIn(cn.COL_AGGREGATION_TYPE, result.columns)
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].unique())
        # At minimum: one model row and two species rows (A, B).
        self.assertIn("model", agg_types)
        self.assertGreaterEqual(len(agg_types), 3)

    def test_returns_none_on_exception(self) -> None:
        """processModel returns None when the underlying pipeline raises an exception."""
        item = _make_mock_item()
        with patch(
            "make_piecewise_predictions.PiecewiseSystemDiscovery",
            side_effect=ValueError("simulated failure"),
        ):
            result = processModel(
                item=item,
                max_changepoint=0,
                min_segment_length=50,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
            )
        self.assertIsNone(result)

    def test_returns_none_when_predict_returns_none(self) -> None:
        """processModel returns None when psd.predict() returns None (defensive check)."""
        item = _make_mock_item()
        mock_psd = MagicMock()
        mock_psd.fit.return_value = None
        mock_psd.predict.return_value = None
        with patch(
            "make_piecewise_predictions.PiecewiseSystemDiscovery",
            return_value=mock_psd,
        ):
            result = processModel(
                item=item,
                max_changepoint=0,
                min_segment_length=50,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
            )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# main tests
# ---------------------------------------------------------------------------


def _make_mock_timecourse_iterator(mock_items):
    """Build a mock TimecourseIterator that yields predefined items."""
    iterator = MagicMock()
    iterator.__iter__ = MagicMock(return_value=iter(mock_items))
    return iterator


class TestMain(unittest.TestCase):

    def _run_with_mocked_iterator(self, output_path, is_initialize=False,
                                  model_names=None):
        """Helper: run main() with a mocked TimecourseIterator."""
        if model_names is None:
            model_names = ["BIOMD0000000001"]

        mock_items = [_make_mock_item(name) for name in model_names]
        mock_iter = _make_mock_timecourse_iterator(mock_items)

        with patch("make_piecewise_predictions.TimecourseIterator",
                   return_value=mock_iter):
            main(
                first_model_num=0,
                last_model_num=len(model_names),
                is_initialize=is_initialize,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
                min_segment_length=50,
                output_path=output_path,
            )

    def test_creates_output_file(self) -> None:
        """main creates the output CSV when it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            self.assertFalse(os.path.isfile(output_path))
            self._run_with_mocked_iterator(output_path,
                                           is_initialize=True)
            self.assertTrue(os.path.isfile(output_path))

    def test_skips_already_processed_models(self) -> None:
        """main skips models already present in the existing output CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            # First run: process 1 model.
            self._run_with_mocked_iterator(output_path,
                                           is_initialize=True)
            df_first = pd.read_csv(output_path)
            # Second run: same range. Model should be skipped (no new rows added).
            self._run_with_mocked_iterator(
                output_path, is_initialize=False)
            df_second = pd.read_csv(output_path)
            self.assertEqual(len(df_first), len(df_second))

    def test_is_initialize_resets_output(self) -> None:
        """is_initialize=True resets the output file to empty before processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            # First run: process 1 model.
            self._run_with_mocked_iterator(output_path,
                                           is_initialize=True)
            df_first = pd.read_csv(output_path)
            self.assertGreater(len(df_first), 0)
            # Second run with is_initialize: should reprocess (no skip).
            self._run_with_mocked_iterator(
                output_path, is_initialize=True)
            df_second = pd.read_csv(output_path)
            self.assertGreater(len(df_second), 0)

    def test_persists_results_to_disk(self) -> None:
        """main persists the accumulated DataFrame to the output CSV at end of run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            self._run_with_mocked_iterator(output_path,
                                           is_initialize=True)
            self.assertTrue(os.path.isfile(output_path))
            df = pd.read_csv(output_path)
            self.assertIn(cn.COL_SYSTEM_ID, df.columns)
            self.assertIn(cn.COL_MAX_CHANGEPOINT, df.columns)

    def test_iterates_over_max_changepoints(self) -> None:
        """main iterates over all max_changepoint values for each model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            self._run_with_mocked_iterator(output_path,
                                           is_initialize=True)
            df = pd.read_csv(output_path)
            unique_models = df[cn.COL_SYSTEM_ID].unique()
            self.assertGreater(len(unique_models), 0)

    def test_skips_excluded_models(self) -> None:
        """main skips models listed in EXCLUDED_MODELS (e.g., badmodels.txt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            # Create a badmodels.txt file that excludes the first model.
            bad_models_file = os.path.join(tmpdir, "badmodels.txt")
            with open(bad_models_file, "w") as f:
                f.write("BIOMD0000000001" + chr(10))

            # Patch cn.DATA_DIR to point to tmpdir so badmodels.txt is found.
            mock_items = [_make_mock_item(model_name="BIOMD0000000002")]
            mock_iter = _make_mock_timecourse_iterator(mock_items)
            with patch.object(cn, 'DATA_DIR', tmpdir):
                with patch("make_piecewise_predictions.TimecourseIterator",
                           return_value=mock_iter):
                    main(
                        first_model_num=0,
                        last_model_num=1,
                        is_initialize=True,
                        coefficient_threshold=COEFFICIENT_THRESHOLD,
                        min_segment_length=50,
                        output_path=output_path,
                    )
            df = pd.read_csv(output_path)
            # The excluded model should not appear in results.
            self.assertNotIn("BIOMD0000000001", df[cn.COL_SYSTEM_ID].values)


if __name__ == "__main__":
    unittest.main()
