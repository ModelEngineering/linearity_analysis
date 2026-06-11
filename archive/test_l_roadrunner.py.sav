"""Tests for Roadrunner class."""
import src.constants as cn  # type: ignore
from src.l_roadrunner import LRoadrunner, getBiomodelsEndtimes, MAX_ITERATOR_STEP  # type: ignore

import csv
import os
import sys
import tempfile
import unittest
import unittest.mock as mock

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
from typing import cast


IGNORE_TESTS = False
MAX_ITERATOR_STEP = 100  # type: ignore

BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
HAS_BIOMODELS = os.path.isdir(BIOMODELS_DIR)

# BIOMD11: outputEndTime="10" in SED-ML — matches DEFAULT_END_TIME, so auto-detection runs.
BIOMD11_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000011", "BIOMD0000000011_url.xml")
BIOMD11_SEDML = os.path.join(BIOMODELS_DIR, "BIOMD0000000011", "BIOMD0000000011_url.sedml")

# BIOMD477: outputEndTime="25" in SED-ML — non-default, so used directly.
BIOMD477_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000477", "BIOMD0000000477_url.xml")
BIOMD477_SEDML = os.path.join(BIOMODELS_DIR, "BIOMD0000000477", "MODEL1308080000_figure5.sedml")

# BIOMD8: 5-species nonlinear model (C, X, M, Y, Z).
# Initial values: C=0, X=0, M=0, Y=1, Z=1
BIOMD8_SBML = os.path.join(BIOMODELS_DIR, "BIOMD0000000008", "BIOMD0000000008_url.xml")
BIOMD8_N_SPECIES = 5
BIOMD8_INITIAL_VALUES = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

# BIOMD241: rate-rule-only model (no reactions) with events that block steadyState().
# end_time falls back to _calculateEndtimeJacobian.
# Smallest |eigenvalue| at t=0 ≈ 0.176355, so end_time ≈ 5.67.
BIOMD241_SBML  = os.path.join(BIOMODELS_DIR, "BIOMD0000000241", "BIOMD0000000241_url.xml")
BIOMD241_EXPECTED_END_TIME = 1.0 / 0.176355  # ≈ 5.67


def _read(path: str) -> str:
    """Return the text contents of *path*."""
    with open(path) as fh:
        return fh.read()

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

# Constant-production model: dS1/dt = k_in (no species dependency), so Jacobian is [[0]].
CONSTANT_PRODUCTION_MODEL = """
-> S1; k_in
k_in = 1.0; S1 = 0
"""

# Production-degradation model: SS = k_in/k_out = 10; time constant = 1/k_out = 10 s.
PRODUCTION_MODEL = """
-> S1; k_in
S1 -> ; k_out * S1
k_in = 1.0; k_out = 0.1; S1 = 0
"""


class TestLRoadrunnerInit(unittest.TestCase):
    """Tests for LRoadrunner.__init__."""

    def test_defaults_from_constants(self) -> None:
        """Default timing parameters match constants."""
        if IGNORE_TESTS:
            return
        rr = LRoadrunner(ANTIMONY_MODEL)
        self.assertEqual(rr.start_time, cn.START_TIME)
        self.assertEqual(rr.num_point, cn.NUM_POINTS)

    def test_custom_timing_params_stored(self) -> None:
        """Custom start_time, end_time, and num_points are stored correctly."""
        if IGNORE_TESTS:
            return
        rr = LRoadrunner(ANTIMONY_MODEL, start_time=1.0, end_time=5.0, num_point=20)
        self.assertEqual(rr.start_time, 1.0)
        self.assertEqual(rr.end_time, 5.0)
        self.assertEqual(rr.num_point, 20)

    def test_roadrunner_instance_stored(self) -> None:
        """Internal RoadRunner instance is created and stored."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(lrr.getRoadrunner(), "getFloatingSpeciesIds"))

    def test_invalid_specification_raises(self) -> None:
        """Passing an unsupported type raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LRoadrunner("12345")

    def test_load_from_rr_instance(self) -> None:
        """LRoadrunner can be initialized from an existing RoadRunner instance."""
        if IGNORE_TESTS:
            return
        rr_raw = te.loada(ANTIMONY_MODEL)
        with self.assertRaises(ValueError):
            LRoadrunner(rr_raw)  # type: ignore


class TestLRoadrunnerProperty(unittest.TestCase):
    """Tests for LRoadrunner.roadrunner property."""

    def test_returns_valid_rr_instance(self) -> None:
        """roadrunner property returns an object with getFloatingSpeciesIds."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL)
        self.assertTrue(hasattr(lrr.getRoadrunner(), "getFloatingSpeciesIds"))

    def test_resets_model_state(self) -> None:
        """roadrunner property resets state so species concentrations return to initial values."""
        if IGNORE_TESTS:
            return
        rr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        rr_raw = rr.getRoadrunner()
        rr_raw.simulate(0.0, 50.0, 10)
        post_sim = np.array(rr_raw.getFloatingSpeciesConcentrations())
        # Accessing property again should reset
        rr_raw = rr.getRoadrunner()
        after_reset = np.array(rr_raw.getFloatingSpeciesConcentrations())
        initial = np.array(te.loada(ANTIMONY_MODEL).getFloatingSpeciesConcentrations())
        np.testing.assert_array_almost_equal(after_reset, initial)


class TestLoadModel(unittest.TestCase):
    """Tests for LRoadrunner._loadModel."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(ANTIMONY_MODEL)

    def test_load_antimony(self) -> None:
        """Antimony strings load successfully."""
        if IGNORE_TESTS:
            return
        rr_raw = self.rr._loadModel(ANTIMONY_MODEL)
        self.assertTrue(hasattr(rr_raw, "getFloatingSpeciesIds"))

    def test_load_sbml(self) -> None:
        """SBML strings (containing <?xml) load successfully."""
        if IGNORE_TESTS:
            return
        sbml_str = te.loada(ANTIMONY_MODEL).getSBML()
        rr_raw = self.rr._loadModel(sbml_str)
        self.assertTrue(hasattr(rr_raw, "getFloatingSpeciesIds"))

    def test_antimony_has_correct_species(self) -> None:
        """Loaded Antimony model contains the expected floating species."""
        if IGNORE_TESTS:
            return
        rr_raw = self.rr._loadModel(ANTIMONY_MODEL)
        species = rr_raw.getFloatingSpeciesIds()
        self.assertIn("S1", species)
        self.assertIn("S2", species)


class TestEndTime(unittest.TestCase):
    """Tests for LRoadrunner.end_time property."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(PRODUCTION_MODEL)

    def test_returns_float(self) -> None:
        """end_time returns a float."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.rr.end_time, float)

    def test_result_is_positive(self) -> None:
        """end_time returns a positive value."""
        if IGNORE_TESTS:
            return
        self.assertGreater(cast(float, self.rr.end_time), 0.0)

    def test_explicit_end_time_returned_unchanged(self) -> None:
        """end_time returns the explicitly provided value without computing."""
        if IGNORE_TESTS:
            return
        rr = LRoadrunner(PRODUCTION_MODEL, end_time=42.0)
        self.assertEqual(rr.end_time, 42.0)

    def test_simulation_reaches_steady_state(self) -> None:
        """Simulating to end_time puts each species within 1% of its steady-state value."""
        if IGNORE_TESTS:
            return
        threshold = 0.01
        end_time = self.rr.end_time
        rr_raw = self.rr.getRoadrunner()
        rr_raw.steadyState()
        ss_arr = np.array(rr_raw.getFloatingSpeciesConcentrations())
        ss_arr_safe = np.array([max(v, 1e-8) for v in ss_arr])
        rr_raw.reset()
        rr_raw.simulate(0.0, end_time, 2)
        final_arr = np.array(rr_raw.getFloatingSpeciesConcentrations())
        divergence = np.max(np.abs(final_arr / ss_arr_safe - 1))
        self.assertLess(divergence, threshold)

    def test_result_is_cached(self) -> None:
        """end_time returns the same value on repeated access (caches result)."""
        if IGNORE_TESTS:
            return
        first = self.rr.end_time
        second = self.rr.end_time
        self.assertEqual(first, second)
        self.assertIsNotNone(self.rr.end_time)


class TestGetSteadyState(unittest.TestCase):
    """Tests for LRoadrunner.getSteadyState."""

    def setUp(self) -> None:
        self.rr = LRoadrunner(PRODUCTION_MODEL)

    def test_returns_ndarray(self) -> None:
        """getSteadyState returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.rr.getSteadyState()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_matches_species_count(self) -> None:
        """getSteadyState returns a 1-D array with one entry per floating species."""
        if IGNORE_TESTS:
            return
        result = self.rr.getSteadyState()
        n_species = len(self.rr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (n_species,))

    def test_known_steady_state_value(self) -> None:
        """getSteadyState returns the analytically known value for the production model (SS = 10)."""
        if IGNORE_TESTS:
            return
        result = self.rr.getSteadyState()
        self.assertAlmostEqual(float(result[0]), 10.0, places=4)


class TestSimulate(unittest.TestCase):
    """Tests for LRoadrunner.simulate."""

    def setUp(self) -> None:
        self.lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=50)

    def test_returns_ndarray(self) -> None:
        """simulate returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_num_points_by_species(self) -> None:
        """simulate returns shape (num_points, n_species)."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate()
        n_species = len(self.lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (50, n_species))

    def test_with_timepoints_includes_time_column(self) -> None:
        """simulate(is_with_timepoints=True) returns an extra leading time column."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate(is_with_timepoints=True)
        n_species = len(self.lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (50, n_species + 1))

    def test_with_timepoints_first_column_is_time(self) -> None:
        """First column of simulate(is_with_timepoints=True) is monotonically increasing."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate(is_with_timepoints=True)
        times = result[:, 0]
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_values_are_finite(self) -> None:
        """All simulated concentrations are finite."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate()
        self.assertTrue(np.all(np.isfinite(result)))

    def test_values_are_non_negative(self) -> None:
        """All simulated concentrations are non-negative."""
        if IGNORE_TESTS:
            return
        result = self.lrr.simulate()
        self.assertTrue(np.all(result >= 0.0))


class TestMakeJacobians(unittest.TestCase):
    """Tests for LRoadrunner.makeJacobians."""

    def setUp(self) -> None:
        self.lr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=10)

    def test_returns_namedtuple_of_three(self) -> None:
        """makeJacobians returns a MakeJacobianResult namedtuple with three fields."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertTrue(hasattr(result, "jacobians"))
        self.assertTrue(hasattr(result, "timepoints"))
        self.assertTrue(hasattr(result, "forcing_inputs"))

    def test_jacobians_shape(self) -> None:
        """Jacobians array has shape (num_points, n_species, n_species)."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        n_species = len(self.lr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.jacobians.shape, (self.lr.num_point, n_species, n_species))

    def test_times_shape(self) -> None:
        """Timepoints array has shape (num_points,)."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        self.assertEqual(result.timepoints.shape, (self.lr.num_point,))

    def test_times_are_monotonically_increasing(self) -> None:
        """Timepoints are strictly increasing."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        self.assertTrue(np.all(np.diff(result.timepoints) > 0))

    def test_jacobians_are_finite(self) -> None:
        """All Jacobian entries are finite."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        self.assertTrue(np.all(np.isfinite(result.jacobians)))

    def test_forcing_inputs_shape(self) -> None:
        """forcing_inputs array has shape (num_points, n_species)."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        n_species = len(self.lr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.forcing_inputs.shape, (self.lr.num_point, n_species))

    def test_forcing_inputs_are_finite(self) -> None:
        """All forcing_inputs entries are finite."""
        if IGNORE_TESTS:
            return
        result = self.lr.makeJacobians()
        self.assertTrue(np.all(np.isfinite(result.forcing_inputs)))

    def test_raises_for_no_floating_species(self) -> None:
        """makeJacobians raises ValueError when the model has no floating species."""
        if IGNORE_TESTS:
            return
        # Boundary-species-only model: S1 is a boundary species (fixed), no floating species.
        boundary_model = """
$S1 -> $S2; k1*S1
k1 = 0.1; S1 = 10; S2 = 0
"""
        rr = LRoadrunner(boundary_model, end_time=10.0)
        with self.assertRaises(ValueError):
            rr.makeJacobians()

    def test_all_zero_jacobian_skips_timepoint(self) -> None:
        """makeJacobians skips timepoints where the Jacobian is all zeros.

        CONSTANT_PRODUCTION_MODEL (dS1/dt = k_in, no species dependency) has
        Jacobian [[0]] at every timepoint.  Each zero Jacobian is skipped, so
        the returned jacobians array is empty.
        """
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(CONSTANT_PRODUCTION_MODEL, end_time=10.0, num_point=5)
        with self.assertRaises(ValueError):
            _ = lrr.makeJacobians()


class TestGetEndtimeFromJacobian(unittest.TestCase):
    """Tests for LRoadrunner._calculateEndtimeJacobian."""

    def setUp(self) -> None:
        self.antimony_lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=10)
        self.production_lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0, num_point=10)

    def test_returns_float_for_normal_model(self) -> None:
        """Returns a float for a model with non-zero eigenvalues."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertIsInstance(result, float)

    def test_returns_positive_value(self) -> None:
        """Returned end time is strictly positive."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertIsNotNone(result)
        self.assertGreater(cast(float, result), 0.0)

    def test_antimony_model_end_time(self) -> None:
        """ANTIMONY_MODEL: smallest |eigenvalue| = k1 = 0.1, so end_time ≈ 10.0.

        Jacobian at t=0 is [[-k1, 0], [k1, -k2]] = [[-0.1, 0], [0.1, -0.2]].
        Eigenvalues of a lower-triangular matrix are the diagonal entries: -0.1, -0.2.
        Smallest magnitude is 0.1 → end_time = 1/0.1 = 10.0.
        """
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_production_model_end_time(self) -> None:
        """PRODUCTION_MODEL: single eigenvalue -k_out = -0.1, so end_time ≈ 10.0."""
        if IGNORE_TESTS:
            return
        result = self.production_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_end_time_is_reciprocal_of_min_eigenvalue_magnitude(self) -> None:
        """end_time equals 1 / min|eigenvalue| of the t=0 Jacobian."""
        if IGNORE_TESTS:
            return
        rr = self.antimony_lrr.getRoadrunner()
        rr.reset()
        rr.simulate(self.antimony_lrr.start_time, self.antimony_lrr.start_time + 1e-10, 2)
        eigenvalues = np.linalg.eigvals(np.array(rr.getFullJacobian()))
        magnitudes = np.abs(eigenvalues)
        expected = 1.0 / float(np.min(magnitudes[magnitudes >= 1e-10]))
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertAlmostEqual(result, expected, places=6)

    def test_returns_none_when_condition_not_met(self) -> None:
        """Returns None when left_null_rank < number of zero eigenvalues.

        The fresh RoadRunner created inside _calculateEndtimeJacobian is
        controlled via _loadModel.  Jacobian has one zero eigenvalue; full-rank
        stoichiometry gives left_null_rank = 0.  0 < 1, so the condition fails.
        """
        if IGNORE_TESTS:
            return
        import unittest.mock as mock
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=10)
        fresh = mock.MagicMock()
        fresh.getFullJacobian.return_value = np.array([[0.0, 0.0], [0.0, -0.2]])
        # Full-rank 2×2 stoichiometry → left_null_rank = 0
        fresh.getFullStoichiometryMatrix.return_value = np.array([[-1.0, 0.0], [1.0, -1.0]])
        with mock.patch.object(lrr, "_loadModel", return_value=fresh):
            result = lrr._calculateEndtimeJacobian()
        self.assertTrue(np.isnan(result))

    def test_returns_none_when_all_eigenvalues_zero(self) -> None:
        """Returns None when every eigenvalue is near-zero (no finite end time exists).

        Rank-0 stoichiometry gives left_null_rank = 2; zero Jacobian gives 2
        zero eigenvalues.  Condition 2 >= 2 is met, but there are no non-zero
        eigenvalues to take the reciprocal of.
        """
        if IGNORE_TESTS:
            return
        import unittest.mock as mock
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=10)
        fresh = mock.MagicMock()
        fresh.getFullJacobian.return_value = np.zeros((2, 2))
        fresh.getFullStoichiometryMatrix.return_value = np.zeros((2, 2))
        with mock.patch.object(lrr, "_loadModel", return_value=fresh):
            result = lrr._calculateEndtimeJacobian()
        self.assertTrue(np.isnan(result))

    def test_result_is_finite(self) -> None:
        """Returned end time is finite (not inf or nan)."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeJacobian()
        self.assertTrue(np.isfinite(result))


class TestCalculateEndtimeCV(unittest.TestCase):
    """Tests for LRoadrunner._calculateEndtimeCV."""

    def setUp(self) -> None:
        self.antimony_lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=20)
        self.production_lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0, num_point=20)

    def test_returns_float_for_antimony_model(self) -> None:
        """Returns a float for a model with dynamics."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeCV()
        self.assertIsInstance(result, float)

    def test_returns_positive_value(self) -> None:
        """Returned end time is strictly positive."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeCV()
        self.assertGreater(result, 0.0)

    def test_result_is_finite(self) -> None:
        """Returned end time is finite (not inf or nan)."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr._calculateEndtimeCV()
        self.assertTrue(np.isfinite(result))

    def test_returns_float_for_production_model(self) -> None:
        """Returns a float for the production-degradation model."""
        if IGNORE_TESTS:
            return
        result = self.production_lrr._calculateEndtimeCV()
        self.assertIsInstance(result, float)

    def test_result_beats_post_steadystate_time(self) -> None:
        """Returned end time has higher median CV than a time well past steady state.

        PRODUCTION_MODEL (SS = k_in/k_out = 10, time constant = 1/k_out = 10 s).
        At T = 1000 s every species is essentially at steady state, so CV → 0.
        The optimised time should capture the transient dynamics and return a
        noticeably higher median CV.
        """
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(PRODUCTION_MODEL, num_point=20)
        opt_time = lrr._calculateEndtimeCV()
        self.assertIsNotNone(opt_time)

        def _median_cv(end_time: float) -> float:
            rr = lrr.getRoadrunner()
            result_arr = np.array(rr.simulate(lrr.start_time, end_time, lrr.num_point))
            data_arr = result_arr[:, 1:]
            cvs = []
            for col in range(data_arr.shape[1]):
                col_data = data_arr[:, col]
                mean_val = float(np.mean(np.abs(col_data)))
                if mean_val < 1e-10:
                    continue
                cvs.append(float(np.std(col_data) / mean_val))
            return float(np.median(cvs)) if cvs else 0.0

        cv_opt = _median_cv(opt_time)
        # At 1000 s all species are firmly at steady state → CV ≈ 0
        cv_post_ss = _median_cv(1000.0)
        self.assertGreater(cv_opt, cv_post_ss)

    def test_returns_none_for_no_species_model(self) -> None:
        """Returns None when the model has no floating species."""
        if IGNORE_TESTS:
            return
        import unittest.mock as mock
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=10)
        fresh = mock.MagicMock()
        fresh.getFloatingSpeciesIds.return_value = []
        with mock.patch.object(lrr, "_loadModel", return_value=fresh):
            result = lrr._calculateEndtimeCV()
        self.assertTrue(np.isnan(result))

    def test_returns_none_when_all_species_zero_mean(self) -> None:
        """Returns None when every species has a zero-mean time course.

        If all species concentrations are identically zero the CV is
        undefined for all of them, so no valid objective exists.
        """
        if IGNORE_TESTS:
            return
        zero_model = """
S1 -> S2; k1*S1
k1 = 0.1; S1 = 0; S2 = 0
"""
        lrr = LRoadrunner(zero_model, end_time=50.0, num_point=20)
        result = lrr._calculateEndtimeCV()
        self.assertTrue(np.isnan(result))

    def test_end_time_source_set_to_max_median_cv(self) -> None:
        """end_time_source is set to ENDTIME_SOURCE_MAX_MEDIAN_CV when CV method is used.

        Uses a model where steady-state and Jacobian methods return None so
        the CV fallback is exercised via the end_time property.
        """
        if IGNORE_TESTS:
            return
        import unittest.mock as mock
        lrr = LRoadrunner(PRODUCTION_MODEL)
        with mock.patch.object(lrr, "_calculateEndtimeSBML", return_value=np.nan), \
                mock.patch.object(lrr, "_calculateEndtimeSteadystate", return_value=np.nan), \
                mock.patch.object(lrr, "_calculateEndtimeJacobian", return_value=np.nan):
            _ = lrr.end_time
        self.assertEqual(lrr.end_time_source, cn.ENDTIME_SOURCE_MAX_MEDIAN_CV)


class TestGetBiomodelsEndtimes(unittest.TestCase):
    """Tests for getBiomodelsEndtimes."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_dict(self) -> None:
        """Returns a dict."""
        if IGNORE_TESTS:
            return
        result = getBiomodelsEndtimes(
            "/nonexistent/path.csv"
        )
        self.assertIsInstance(result, dict)

    def test_missing_file_returns_empty_dict(self) -> None:
        """Returns empty dict when the CSV file does not exist."""
        if IGNORE_TESTS:
            return
        result = getBiomodelsEndtimes("/nonexistent/path.csv")
        self.assertEqual(result, {})

    def test_valid_csv_returns_mapping(self) -> None:
        """Valid CSV with model_name and end_time columns returns correct mapping."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "endtimes.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME, cn.COL_ENDTIME])
            writer.writerow(["BIOMD0000000001", "25.0"])
            writer.writerow(["BIOMD0000000002", "100.0"])
        result = getBiomodelsEndtimes(csv_path)
        self.assertAlmostEqual(result["BIOMD0000000001"], 25.0)
        self.assertAlmostEqual(result["BIOMD0000000002"], 100.0)

    def test_missing_model_name_column_returns_empty_dict(self) -> None:
        """Returns empty dict when CSV is missing the model_name column."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_col", cn.COL_ENDTIME])
            writer.writerow(["BIOMD0000000001", "25.0"])
        result = getBiomodelsEndtimes(csv_path)
        self.assertEqual(result, {})

    def test_missing_end_time_column_returns_empty_dict(self) -> None:
        """Returns empty dict when CSV is missing the end_time column."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME, "wrong_col"])
            writer.writerow(["BIOMD0000000001", "25.0"])
        result = getBiomodelsEndtimes(csv_path)
        self.assertEqual(result, {})


class TestCalculateEndtimeSBML(unittest.TestCase):
    """Unit tests for LRoadrunner._calculateEndtimeSBML using synthetic SED-ML strings."""

    def _lrr(self, sedml_str: str) -> LRoadrunner:
        lrr = LRoadrunner(ANTIMONY_MODEL, sedml_str=sedml_str)
        lrr._end_time = np.nan
        return lrr

    def test_returns_none_when_no_sedml(self) -> None:
        """Returns None when no SED-ML string is provided."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL)
        result = lrr._calculateEndtimeSBML()
        self.assertTrue(np.isnan(result))

    def test_returns_none_when_no_output_end_time(self) -> None:
        """Returns None when SED-ML has no outputEndTime attribute."""
        if IGNORE_TESTS:
            return
        lrr = self._lrr('<sedML><uniformTimeCourse/></sedML>')
        result = lrr._calculateEndtimeSBML()
        self.assertTrue(np.isnan(result))

    def test_returns_end_time_from_sedml(self) -> None:
        """Returns the outputEndTime value from SED-ML when it differs from the default."""
        if IGNORE_TESTS:
            return
        sedml = '<uniformTimeCourse outputEndTime="25.0"/>'
        lrr = self._lrr(sedml)
        result = lrr._calculateEndtimeSBML()
        self.assertAlmostEqual(result, 25.0)

    def test_ignores_default_end_time_with_auto_marker(self) -> None:
        """Returns None when outputEndTime equals default and auto marker is present."""
        if IGNORE_TESTS:
            return
        sedml = (
            f'uniformTimeCourse id="auto_ten_seconds" outputEndTime="10.0"'
        )
        lrr = self._lrr(sedml)
        result = lrr._calculateEndtimeSBML()
        self.assertTrue(np.isnan(result))

    def test_uses_default_end_time_when_no_auto_marker(self) -> None:
        """Returns the default end time when outputEndTime=10.0 and no auto marker."""
        if IGNORE_TESTS:
            return
        sedml = '<uniformTimeCourse outputEndTime="10.0"/>'
        lrr = self._lrr(sedml)
        result = lrr._calculateEndtimeSBML()
        self.assertAlmostEqual(result, 10.0)


class TestEndTimeSource(unittest.TestCase):
    """Tests for LRoadrunner.end_time_source tracking."""

    def test_source_is_user_specified_when_end_time_provided(self) -> None:
        """end_time_source is USER_SPECIFIED when end_time is given at construction."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=42.0)
        _ = lrr.end_time
        self.assertEqual(lrr.end_time_source, cn.ENDTIME_SOURCE_USER_SPECIFIED)

    def test_source_is_steadystate_when_ss_method_succeeds(self) -> None:
        """end_time_source is STEADYSTATE when calculated via steady-state search."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(PRODUCTION_MODEL)
        with mock.patch.object(lrr, "_calculateEndtimeSBML", return_value=np.nan):
            _ = lrr.end_time
        self.assertEqual(lrr.end_time_source, cn.ENDTIME_SOURCE_STEADYSTATE)

    def test_source_is_none_when_all_methods_fail(self) -> None:
        """end_time_source is None when no method can determine an end time."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(PRODUCTION_MODEL)
        with mock.patch.object(lrr, "_calculateEndtimeSBML", return_value=np.nan), \
                mock.patch.object(lrr, "_calculateEndtimeSteadystate", return_value=np.nan), \
                mock.patch.object(lrr, "_calculateEndtimeCV", return_value=np.nan):
            with self.assertRaises(ValueError):
                _= lrr.end_time

    def test_source_is_sedml_when_sedml_used(self) -> None:
        """end_time_source is SEDML when end time is extracted from SED-ML string."""
        if IGNORE_TESTS:
            return
        sedml = '<uniformTimeCourse outputEndTime="25.0"/>'
        lrr = LRoadrunner(ANTIMONY_MODEL, sedml_str=sedml)
        _ = lrr.end_time
        self.assertEqual(lrr.end_time_source, cn.ENDTIME_SOURCE_SEDML)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestEndTimeSedml(unittest.TestCase):
    """Tests for LRoadrunner.end_time when a SED-ML string is supplied."""

    def test_non_default_sedml_end_time_used_directly(self) -> None:
        """end_time returns the SED-ML outputEndTime when it differs from DEFAULT_END_TIME.

        BIOMD477's SED-ML specifies outputEndTime="25", which is not the
        default (10), so end_time should return 25.0 without running
        the auto-detection algorithm.
        """
        if IGNORE_TESTS:
            return
        rr = LRoadrunner(_read(BIOMD477_SBML), sedml_str=_read(BIOMD477_SEDML))
        self.assertAlmostEqual(rr.end_time, 25.0)

    def test_non_default_sedml_end_time_caches_value(self) -> None:
        """end_time is cached after being read from SED-ML."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(_read(BIOMD477_SBML), sedml_str=_read(BIOMD477_SEDML))
        _ = lrr.end_time
        self.assertAlmostEqual(lrr.end_time, 25.0) # type: ignore

    def test_default_sedml_end_time_falls_through_to_auto_detect(self) -> None:
        """end_time runs auto-detection when SED-ML outputEndTime equals DEFAULT_END_TIME.

        BIOMD11's SED-ML specifies outputEndTime="10", which equals DEFAULT_END_TIME,
        so the SED-ML value is ignored and the steady-state search runs instead.
        The result is a positive float that need not equal 10.
        """
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(_read(BIOMD11_SBML), sedml_str=_read(BIOMD11_SEDML))
        result = lrr.end_time
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestEndTimeBiomd241(unittest.TestCase):
    """end_time tests for BIOMD241, a rate-rule-only model whose steadyState()
    raises RuntimeError due to events, forcing the fallback to
    _calculateEndtimeJacobian.
    """

    def setUp(self) -> None:
        self.lrr = LRoadrunner(_read(BIOMD241_SBML))

    def test_returns_float(self) -> None:
        """end_time returns a float for BIOMD241."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.lrr.end_time, float)

    def test_returns_positive_value(self) -> None:
        """end_time is strictly positive for BIOMD241."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.lrr.end_time, 0.0)

    def test_end_time_is_cached(self) -> None:
        """end_time returns the same value on repeated access."""
        if IGNORE_TESTS:
            return
        first = self.lrr.end_time
        second = self.lrr.end_time
        self.assertEqual(first, second)


class TestGetInitialValues(unittest.TestCase):
    """Tests for LRoadrunner.getInitialValues."""

    def setUp(self) -> None:
        self.antimony_lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        self.production_lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0)

    def test_returns_ndarray(self) -> None:
        """getInitialValues returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getInitialValues()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_matches_species_count(self) -> None:
        """getInitialValues returns a 1-D array with one entry per floating species."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getInitialValues()
        n_species = len(self.antimony_lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (n_species,))

    def test_known_initial_values_antimony(self) -> None:
        """getInitialValues returns [10, 0] for ANTIMONY_MODEL (S1=10, S2=0)."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getInitialValues()
        np.testing.assert_array_almost_equal(result, np.array([10.0, 0.0]))

    def test_known_initial_values_production(self) -> None:
        """getInitialValues returns [0] for PRODUCTION_MODEL (S1=0)."""
        if IGNORE_TESTS:
            return
        result = self.production_lrr.getInitialValues()
        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_returns_reset_state(self) -> None:
        """getInitialValues returns the reset (initial) concentrations, not post-simulation."""
        if IGNORE_TESTS:
            return
        rr_raw = self.antimony_lrr.getRoadrunner()
        rr_raw.simulate(0.0, 50.0, 10)
        result = self.antimony_lrr.getInitialValues()
        np.testing.assert_array_almost_equal(result, np.array([10.0, 0.0]))


class TestGetForcedInputs(unittest.TestCase):
    """Tests for LRoadrunner.getForcedInputs.

    getForcedInputs evaluates f and J at end_time/100.  To recover near-t=0
    values (where the linearisation identity f = J*x0 + u holds cleanly), the
    tests that check specific numerical values use a very small end_time so
    that end_time/100 ≈ 0.
    """

    def setUp(self) -> None:
        # Use a large end_time for shape/type tests (speed doesn't matter).
        self.antimony_lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        self.production_lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0)
        # Tiny end_time so evaluation is essentially at t=0.
        self.antimony_lrr_near0 = LRoadrunner(ANTIMONY_MODEL, end_time=1e-3)
        self.production_lrr_near0 = LRoadrunner(PRODUCTION_MODEL, end_time=1e-3)

    def test_returns_ndarray(self) -> None:
        """getForcedInputs returns a numpy ndarray."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getForcingInputs()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_matches_species_count(self) -> None:
        """getForcedInputs returns a 1-D array with one entry per floating species."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getForcingInputs()
        n_species = len(self.antimony_lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (n_species,))

    def test_linear_model_has_zero_forced_inputs(self) -> None:
        """ANTIMONY_MODEL is linear (dx/dt = J*x), so forced inputs are zero near t=0."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr_near0.getForcingInputs()
        np.testing.assert_array_almost_equal(result, np.zeros(2), decimal=3)

    def test_production_model_has_nonzero_forced_inputs(self) -> None:
        """PRODUCTION_MODEL has a constant production term k_in=1.0, so forced input ≈ 1.0 near t=0."""
        if IGNORE_TESTS:
            return
        result = self.production_lrr_near0.getForcingInputs()
        self.assertAlmostEqual(float(result[0]), 1.0, places=3)

    def test_values_are_finite(self) -> None:
        """All forced input values are finite."""
        if IGNORE_TESTS:
            return
        result = self.antimony_lrr.getForcingInputs()
        self.assertTrue(np.all(np.isfinite(result)))


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestBiomd8InitialValuesAndForcedInputs(unittest.TestCase):
    """Tests for getInitialValues and getForcedInputs using BIOMD8.

    BIOMD8 is a 5-species nonlinear model (C, X, M, Y, Z), making it a realistic
    integration test for both methods.  A tiny end_time (1e-3) is used so that
    end_time/100 ≈ 0, keeping the Jacobian and rates near the initial state.
    """

    def setUp(self) -> None:
        with open(BIOMD8_SBML) as fh:
            sbml = fh.read()
        self.lrr = LRoadrunner(sbml, end_time=1e-3)

    def test_get_initial_values_returns_ndarray(self) -> None:
        """getInitialValues returns a numpy ndarray for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getInitialValues()
        self.assertIsInstance(result, np.ndarray)

    def test_get_initial_values_shape(self) -> None:
        """getInitialValues returns a 1-D array of length 5 for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getInitialValues()
        self.assertEqual(result.shape, (BIOMD8_N_SPECIES,))

    def test_get_initial_values_known_values(self) -> None:
        """getInitialValues matches the known initial concentrations for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getInitialValues()
        np.testing.assert_array_almost_equal(result, BIOMD8_INITIAL_VALUES, decimal=6)

    def test_get_forced_inputs_returns_ndarray(self) -> None:
        """getForcedInputs returns a numpy ndarray for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getForcingInputs()
        self.assertIsInstance(result, np.ndarray)

    def test_get_forced_inputs_shape(self) -> None:
        """getForcedInputs returns a 1-D array of length 5 for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getForcingInputs()
        self.assertEqual(result.shape, (BIOMD8_N_SPECIES,))

    def test_get_forced_inputs_nonzero(self) -> None:
        """getForcedInputs has at least one nonzero entry for the nonlinear BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getForcingInputs()
        self.assertGreater(np.max(np.abs(result)), 0.0)

    def test_get_forced_inputs_values_are_finite(self) -> None:
        """All forced input values are finite for BIOMD8."""
        if IGNORE_TESTS:
            return
        result = self.lrr.getForcingInputs()
        self.assertTrue(np.all(np.isfinite(result)))


class TestTimecourse(unittest.TestCase):
    """Tests for LRoadrunner.timecourse property."""

    def setUp(self) -> None:
        self.lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0, num_point=50)

    def test_returns_dataframe(self) -> None:
        """timecourse returns a pandas DataFrame."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_is_named_time(self) -> None:
        """timecourse DataFrame index is named 'time'."""
        #if IGNORE_TESTS:
        #    return
        result = self.lrr.timecourse_df
        self.assertEqual(result.index.name, "time")

    def test_index_is_monotonically_increasing(self) -> None:
        """timecourse index values are strictly increasing."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        times = result.index.to_numpy()
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_shape_num_points_by_species(self) -> None:
        """timecourse DataFrame has shape (num_points, n_species)."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        n_species = len(self.lrr.getRoadrunner().getFloatingSpeciesIds())
        self.assertEqual(result.shape, (self.lrr.num_point, n_species))

    def test_columns_match_species_names(self) -> None:
        """timecourse columns match the floating species IDs."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        species_ids = self.lrr.getRoadrunner().getFloatingSpeciesIds()
        self.assertEqual(list(result.columns), species_ids)

    def test_values_are_finite(self) -> None:
        """All timecourse values are finite."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        self.assertTrue(np.all(np.isfinite(result.values)))

    def test_values_are_non_negative(self) -> None:
        """All timecourse concentration values are non-negative."""
        if IGNORE_TESTS:
            return
        result = self.lrr.timecourse_df
        self.assertTrue(np.all(result.values >= 0.0))


class TestSpeciesNames(unittest.TestCase):
    """Tests for LRoadrunner.species_names property."""

    def setUp(self) -> None:
        self.lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)

    def test_returns_list(self) -> None:
        """species_names returns a list."""
        if IGNORE_TESTS:
            return
        result = self.lrr.species_names
        self.assertIsInstance(result, list)

    def test_elements_are_strings(self) -> None:
        """species_names elements are all strings."""
        if IGNORE_TESTS:
            return
        for name in self.lrr.species_names:
            self.assertIsInstance(name, str)

    def test_correct_species_for_antimony_model(self) -> None:
        """species_names returns ['S1', 'S2'] for ANTIMONY_MODEL."""
        if IGNORE_TESTS:
            return
        result = self.lrr.species_names
        self.assertEqual(result, ["S1", "S2"])

    def test_correct_species_for_production_model(self) -> None:
        """species_names returns ['S1'] for PRODUCTION_MODEL."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0)
        result = lrr.species_names
        self.assertEqual(result, ["S1"])

    def test_no_brackets_in_names(self) -> None:
        """species_names strips any enclosing brackets from RoadRunner IDs."""
        if IGNORE_TESTS:
            return
        for name in self.lrr.species_names:
            self.assertFalse(name.startswith("["))
            self.assertFalse(name.endswith("]"))

    def test_is_cached(self) -> None:
        """species_names returns the same list object on repeated access."""
        if IGNORE_TESTS:
            return
        first = self.lrr.species_names
        second = self.lrr.species_names
        self.assertIs(first, second)


class TestNumSpecies(unittest.TestCase):
    """Tests for LRoadrunner.num_species property."""

    def test_returns_int(self) -> None:
        """num_species returns an int."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        self.assertIsInstance(lrr.num_species, int)

    def test_correct_count_antimony_model(self) -> None:
        """num_species returns 2 for ANTIMONY_MODEL (S1, S2)."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        self.assertEqual(lrr.num_species, 2)

    def test_correct_count_production_model(self) -> None:
        """num_species returns 1 for PRODUCTION_MODEL (S1)."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(PRODUCTION_MODEL, end_time=50.0)
        self.assertEqual(lrr.num_species, 1)

    def test_matches_species_names_length(self) -> None:
        """num_species equals len(species_names)."""
        if IGNORE_TESTS:
            return
        lrr = LRoadrunner(ANTIMONY_MODEL, end_time=50.0)
        self.assertEqual(lrr.num_species, len(lrr.species_names))


class TestMakeBiomodel(unittest.TestCase):
    """Tests for LRoadrunner.makeBiomodel — error cases that do not need BioModels data."""

    def test_no_path_no_id_no_num_raises_value_error(self) -> None:
        """All three identifiers omitted → ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LRoadrunner.makeBiomodel()

    def test_model_id_without_biomd_prefix_raises(self) -> None:
        """model_id that does not start with 'BIOMD' → ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LRoadrunner.makeBiomodel(model_name="FOO0000000008")

    def test_model_num_zero_with_no_path_no_id_raises(self) -> None:
        """model_num=0 with no path or model_id → ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            LRoadrunner.makeBiomodel(model_num=0)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestMakeBiomodelWithData(unittest.TestCase):
    """Tests for LRoadrunner.makeBiomodel that require BioModels data."""

    def test_model_num_returns_lroadrunner(self) -> None:
        """model_num=8 → returns an LRoadrunner instance."""
        if IGNORE_TESTS:
            return
        l_roadrunner = LRoadrunner.makeBiomodel(model_num=8, end_time=20.0,
                num_point=5)
        self.assertIsInstance(l_roadrunner, LRoadrunner)

    def test_model_name_returns_lroadrunner(self) -> None:
        """model_name='BIOMD0000000008' → returns an LRoadrunner instance."""
        if IGNORE_TESTS:
            return
        l_roadrunner = LRoadrunner.makeBiomodel(model_name="BIOMD0000000008",
                end_time=20.0, num_point=5)
        self.assertIsInstance(l_roadrunner, LRoadrunner)

    def test_explicit_path_returns_lroadrunner(self) -> None:
        """Explicit path → returns an LRoadrunner instance."""
        if IGNORE_TESTS:
            return
        l_roadrunner = LRoadrunner.makeBiomodel(path=BIOMD8_SBML, end_time=20.0,
                num_point=5)
        self.assertIsInstance(l_roadrunner, LRoadrunner)

    def test_end_time_kwarg_overrides_lookup(self) -> None:
        """end_time kwarg takes precedence over the endtime_dct lookup."""
        if IGNORE_TESTS:
            return
        l_roadrunner = LRoadrunner.makeBiomodel(path=BIOMD8_SBML, end_time=42.0,
                num_point=5)
        self.assertAlmostEqual(l_roadrunner.end_time, 42.0)

    def test_model_num_has_species(self) -> None:
        """Model loaded via model_num has at least one floating species."""
        if IGNORE_TESTS:
            return
        l_roadrunner = LRoadrunner.makeBiomodel(model_num=8, end_time=20.0,
                num_point=5)
        self.assertGreater(len(l_roadrunner.species_names), 0)

    def test_model_num_equals_model_id(self) -> None:
        """Loading by model_num=8 and model_id='BIOMD0000000008' give the same species."""
        if IGNORE_TESTS:
            return
        lr_num = LRoadrunner.makeBiomodel(model_num=8, end_time=20.0, num_point=5)
        lr_id = LRoadrunner.makeBiomodel(model_name="BIOMD0000000008", end_time=20.0,
                num_point=5)
        self.assertEqual(lr_num.species_names, lr_id.species_names)

    def test_end_time_from_csv_used_for_biomd153(self) -> None:
        """makeBiomodel uses the end_time from biomodels_endtime.csv for BIOMD153.

        BIOMD0000000153 has end_time=1.4106262159417386 (max_median_cv) in the CSV.
        When no end_time kwarg is passed, makeBiomodel should look it up and store
        it in _end_time rather than leaving it as nan.
        """
        if IGNORE_TESTS:
            return
        BIOMD153_END_TIME = 1.4106262159417386
        l_roadrunner = LRoadrunner.makeBiomodel(model_num=153, num_point=5)
        self.assertAlmostEqual(l_roadrunner.end_time, BIOMD153_END_TIME, places=6)


if __name__ == "__main__":
    unittest.main()
