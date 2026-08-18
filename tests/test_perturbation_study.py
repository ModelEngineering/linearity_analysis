"""Tests for scripts/perturbation_study.py."""

import os
import sys
import unittest
import tempfile

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from collections import namedtuple  # type: ignore

import perturbation_study as ps  # type: ignore
import src.constants as cn  # type: ignore

FakeResult = namedtuple("FakeResult", ["df", "fig"])


class TestConstants(unittest.TestCase):
    """Tests for the module-level constants."""

    def test_threshold_is_positive_float(self) -> None:
        self.assertIsInstance(ps.THRESHOLD, (int, float))
        self.assertGreater(ps.THRESHOLD, 0.0)

    def test_poly_degree_is_one(self) -> None:
        self.assertEqual(ps.POLY_DEGREE, 1)

    def test_species_fraction_is_one(self) -> None:
        self.assertEqual(ps.SPECIES_FRACTION, 1.0)

    def test_perturbations_sorted_ascending(self) -> None:
        self.assertEqual(ps.PERTURBATIONS, sorted(ps.PERTURBATIONS))

    def test_perturbations_include_zero(self) -> None:
        self.assertIn(0.0, ps.PERTURBATIONS)

    def test_perturbations_symmetric_around_zero(self) -> None:
        """For every non-zero perturbation value, its negation is also present."""
        for p in ps.PERTURBATIONS:
            if p != 0.0:
                self.assertIn(-p, ps.PERTURBATIONS)

    def test_perturbations_include_expected_values(self) -> None:
        """Expected perturbation values are present."""
        expected = {-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.50}
        self.assertEqual(set(ps.PERTURBATIONS), expected)

    def test_output_path_ends_with_csv(self) -> None:
        self.assertTrue(
            os.path.basename(ps.OUTPUT_PATH).endswith('.csv'),
        )


class TestExcludes(unittest.TestCase):
    """Tests for the EXCLUDES list."""

    def test_excludes_is_list_of_strings(self) -> None:
        for item in ps.EXCLUDES:
            self.assertIsInstance(item, str)

    def test_excluded_models_have_biomd_prefix(self) -> None:
        for model_name in ps.EXCLUDES:
            self.assertTrue(
                model_name.startswith("BIOMD"),
                f"{model_name} should have BIOMD prefix.",
            )


class TestMainNoExistingResults(unittest.TestCase):
    """Test main() when OUTPUT_PATH does not yet exist."""

    def _make_timecourse_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 10.0, 50)
        df = pd.DataFrame(
            rng.standard_normal((len(t), 3)),
            index=t,
            columns=["A", "B", "C"],
        )
        df.index.name = "time"
        return df

    def _make_fake_timecourse(self) -> mock.Mock:
        tc_df = self._make_timecourse_df()
        fake_model = mock.Mock()
        fake_model.model_name = "BIOMD0000000001"
        tc = mock.Mock()
        tc.model = fake_model
        tc.timecourse_df = tc_df
        return tc

    def test_processes_models_when_no_existing_results(self) -> None:
        """main processes all models when no output CSV exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(model_name="BIOMD0000000001", timecourse=self._make_fake_timecourse()),
                mock.Mock(model_name="BIOMD0000000002", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000001", "BIOMD0000000002"],
                        'mean': [0.5, 0.6],
                        'min': [0.3, 0.4],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()
            self.assertTrue(os.path.isfile(fake_csv_path))

    def test_skips_already_processed_models(self) -> None:
        """main does not re-analyze models already in existing CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            initial_df = pd.DataFrame({
                cn.COL_SYSTEM_ID: ["BIOMD0000009999"],
                'mean': [0.7],
                'min': [0.5],
            })
            initial_df.to_csv(fake_csv_path, index=False)
            items_to_iterate = [
                mock.Mock(model_name="BIOMD0000009999", timecourse=self._make_fake_timecourse()),
                mock.Mock(model_name="BIOMD0000000001", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000001"],
                        'mean': [0.5],
                        'min': [0.3],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()

            # analyzePerturbations should be called only once (for the new model),
            # not for the already-processed BIOMD0000009999.
            self.assertEqual(mock_analyze.call_count, 1)


class TestMainExcludes(unittest.TestCase):
    """Test main() skips excluded models."""

    def _make_timecourse_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 10.0, 50)
        df = pd.DataFrame(
            rng.standard_normal((len(t), 3)),
            index=t,
            columns=["A", "B", "C"],
        )
        df.index.name = "time"
        return df

    def _make_fake_timecourse(self) -> mock.Mock:
        tc_df = self._make_timecourse_df()
        fake_model = mock.Mock()
        fake_model.model_name = "BIOMD0000000001"
        tc = mock.Mock()
        tc.model = fake_model
        tc.timecourse_df = tc_df
        return tc

    def test_skips_excluded_models(self) -> None:
        """main skips models listed in EXCLUDES."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(
                    model_name=ps.EXCLUDES[0],
                    timecourse=self._make_fake_timecourse(),
                ),
                mock.Mock(model_name="BIOMD0000000001", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000001"],
                        'mean': [0.5],
                        'min': [0.3],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()
            written_df = pd.read_csv(fake_csv_path)
            system_ids = set(written_df[cn.COL_SYSTEM_ID].values)
            self.assertNotIn(ps.EXCLUDES[0], system_ids)
            self.assertIn("BIOMD0000000001", system_ids)


class TestMainExceptionHandling(unittest.TestCase):
    """Test main() handles analyzePerturbations exceptions gracefully."""

    def _make_timecourse_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 10.0, 50)
        df = pd.DataFrame(
            rng.standard_normal((len(t), 3)),
            index=t,
            columns=["A", "B", "C"],
        )
        df.index.name = "time"
        return df

    def _make_fake_timecourse(self) -> mock.Mock:
        tc_df = self._make_timecourse_df()
        fake_model = mock.Mock()
        fake_model.model_name = "BIOMD0000000001"
        tc = mock.Mock()
        tc.model = fake_model
        tc.timecourse_df = tc_df
        return tc

    def test_continues_on_analyzePerturbations_exception(self) -> None:
        """main continues iterating if a model raises an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(model_name="BIOMD000000BAD", timecourse=self._make_fake_timecourse()),
                mock.Mock(model_name="BIOMD000000GOOD", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    calls = [0]

                    def _side_effect(*args, **kwargs):
                        calls[0] += 1
                        if calls[0] == 1:
                            raise ValueError("simulated failure")
                        return FakeResult(df=pd.DataFrame({
                            cn.COL_SYSTEM_ID: ["BIOMD000000GOOD"],
                            'mean': [0.5],
                            'min': [0.3],
                        }), fig=None)

                    mock_analyze.side_effect = _side_effect
                    result_df = ps.main()
            self.assertIsInstance(result_df, pd.DataFrame)
            system_ids = set(result_df[cn.COL_SYSTEM_ID].values)
            self.assertNotIn("BIOMD000000BAD", system_ids)
            self.assertIn("BIOMD000000GOOD", system_ids)


class TestMainOutputFormat(unittest.TestCase):
    """Verify the CSV output has the expected columns."""

    def _make_timecourse_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 10.0, 50)
        df = pd.DataFrame(
            rng.standard_normal((len(t), 3)),
            index=t,
            columns=["A", "B", "C"],
        )
        df.index.name = "time"
        return df

    def _make_fake_timecourse(self) -> mock.Mock:
        tc_df = self._make_timecourse_df()
        fake_model = mock.Mock()
        fake_model.model_name = "BIOMD0000000042"
        tc = mock.Mock()
        tc.model = fake_model
        tc.timecourse_df = tc_df
        return tc

    def test_output_csv_contains_threshold_column(self) -> None:
        """The output CSV has the threshold column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(model_name="BIOMD0000000042", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000042"],
                        'mean': [0.5],
                        'min': [0.3],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()
            self.assertTrue(os.path.isfile(fake_csv_path))
            written_df = pd.read_csv(fake_csv_path)
            self.assertIn(cn.COL_THRESHOLD, written_df.columns.tolist())

    def test_threshold_value_in_output(self) -> None:
        """The threshold value in output matches THRESHOLD constant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(model_name="BIOMD0000000042", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000042"],
                        'mean': [0.5],
                        'min': [0.3],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()
            written_df = pd.read_csv(fake_csv_path)
            threshold_col = written_df[cn.COL_THRESHOLD].unique()
            self.assertEqual(len(threshold_col), 1)
            self.assertAlmostEqual(float(threshold_col[0]), float(ps.THRESHOLD))

    def test_output_csv_contains_system_id_column(self) -> None:
        """The output CSV has the system_id column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv_path = os.path.join(tmpdir, "fake_perturbation_study.csv")
            items_to_iterate = [
                mock.Mock(model_name="BIOMD0000000042", timecourse=self._make_fake_timecourse()),
            ]
            with mock.patch.object(ps, 'OUTPUT_PATH', fake_csv_path), \
                 mock.patch('perturbation_study.TimecourseIterator') as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(
                    return_value=iter(items_to_iterate),
                )
                with mock.patch('perturbation_study.SystemDiscovery.analyzePerturbations') as mock_analyze:
                    fake_result_df = pd.DataFrame({
                        cn.COL_SYSTEM_ID: ["BIOMD0000000042"],
                        'mean': [0.5],
                        'min': [0.3],
                    })
                    mock_analyze.return_value = FakeResult(df=fake_result_df, fig=None)
                    ps.main()
            written_df = pd.read_csv(fake_csv_path)
            self.assertIn(cn.COL_SYSTEM_ID, written_df.columns.tolist())


if __name__ == '__main__':
    unittest.main()
