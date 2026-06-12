"""Tests for Timecourse."""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import tellurium as te  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from timecourse import Timecourse  # type: ignore

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(cn.TIMECOURSE_ZIP_PATH)
BIOMODEL_53 = "BIOMD0000000053"

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_POINT = 11
NUM_SPECIES = 2
MODEL_NAME = "test_model"


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL, model_name=MODEL_NAME)


def _makeTimecourse() -> Timecourse:
    """Return a Timecourse pre-populated with synthetic data (no simulation)."""
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
            rng.standard_normal((NUM_POINT, NUM_SPECIES)),
            index=timepoint_arr,
            columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    jacobian_collection_arr = rng.standard_normal((NUM_POINT, NUM_SPECIES, NUM_SPECIES))
    return Timecourse(
            model=_makeModel(),
            timecourse_df=timecourse_df,
            jacobian_collection_arr=jacobian_collection_arr,
    )


class TestTimecourseInit(unittest.TestCase):
    """Tests for Timecourse.__init__."""

    def test_stores_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIsInstance(tc.model, Model)

    def test_stores_start_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), start_time=1.5)
        self.assertEqual(tc.start_time, 1.5)

    def test_stores_num_points(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), num_point=50)
        self.assertEqual(tc.num_point, 50)

    def test_timecourse_df_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertFalse(tc._timecourse_df.empty)

    def test_jacobian_collection_arr_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertGreater(tc._jacobian_collection_arr.size, 0)


class TestTimecourseJacobianCollectionArr(unittest.TestCase):
    """Tests for Timecourse.jacobian_collection_arr property."""

    def test_returns_ndarray(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIsInstance(tc.jacobian_collection_arr, np.ndarray)

    def test_shape(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertEqual(tc.jacobian_collection_arr.shape,
                (NUM_POINT, NUM_SPECIES, NUM_SPECIES))

    def test_returns_prepopulated_array(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        np.testing.assert_array_equal(
                tc.jacobian_collection_arr, tc._jacobian_collection_arr)

    def test_cached_on_second_access(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        first = tc.jacobian_collection_arr
        second = tc.jacobian_collection_arr
        self.assertIs(first, second)

    def test_simulation_populates_timecourse_df(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), end_time=10.0)
        self.assertTrue(tc._timecourse_df.empty)
        _ = tc.jacobian_collection_arr
        self.assertFalse(tc._timecourse_df.empty)


class TestTimecourseSerialize(unittest.TestCase):
    """Tests for Timecourse.serialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_string_path(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIsInstance(path, str)

    def test_file_exists_after_serialize(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertTrue(os.path.isfile(path))

    def test_filename_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIn(MODEL_NAME, os.path.basename(path))

    def test_raises_without_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=Model(ANTIMONY_MODEL))
        with self.assertRaises(ValueError):
            tc.serialize()


class TestTimecourseEq(unittest.TestCase):
    """Tests for Timecourse.__eq__."""

    def test_equal_timecourses(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        self.assertEqual(tc1, tc2)

    def test_different_model_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.model = Model(ANTIMONY_MODEL, model_name="other")
        self.assertNotEqual(tc1, tc2)

    def test_different_start_time_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.start_time = 99.0
        self.assertNotEqual(tc1, tc2)

    def test_different_end_time_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.end_time = 99.0
        self.assertNotEqual(tc1, tc2)

    def test_different_num_point_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.num_point = 999
        self.assertNotEqual(tc1, tc2)

    def test_different_timecourse_df_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2._timecourse_df = tc2._timecourse_df * 2
        self.assertNotEqual(tc1, tc2)

    def test_different_jacobian_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2._jacobian_collection_arr = tc2._jacobian_collection_arr * 2
        self.assertNotEqual(tc1, tc2)

    def test_not_equal_to_non_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIs(tc.__eq__("not a timecourse"), NotImplemented)


class TestTimecourseRoundtrip(unittest.TestCase):
    """Roundtrip tests for Timecourse.serialize / Timecourse.deserialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _serializeAndDeserialize(self) -> tuple:
        """Return (original, restored) Timecourse pair via a temp directory."""
        original = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = original.serialize()
        restored = Timecourse.deserialize(path)
        return original, restored

    def test_roundtrip_equal(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original, restored)

    def test_deserialize_raises_with_empty_path(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(Exception):
            Timecourse.deserialize("")


class TestTimecourseplot(unittest.TestCase):
    """Tests for Timecourse.plot."""

    def setUp(self) -> None:
        self._tc = _makeTimecourse()
        self._fig, self._ax = plt.subplots()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        result = self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertIsInstance(result, PlotOptions)

    def test_plots_one_line_per_species(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(len(self._ax.lines), NUM_SPECIES)

    def test_line_labels_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        labels = [line.get_label() for line in self._ax.lines]
        self.assertEqual(labels, self._tc.model.species_names)

    def test_xdata_matches_timecourse_index(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        for line in self._ax.lines:
            np.testing.assert_array_equal(line.get_xdata(),
                    self._tc.timecourse_df.index)

    def test_ydata_matches_species_values(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        for line, name in zip(self._ax.lines, self._tc.model.species_names):
            np.testing.assert_array_equal(line.get_ydata(),
                    self._tc.timecourse_df[name])

    def test_line_colors_are_distinct(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        colors = [line.get_color() for line in self._ax.lines]
        self.assertEqual(len(set(colors)), NUM_SPECIES)

    def test_default_xlabel_is_time(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(self._ax.get_xlabel(), "time")

    def test_default_ylabel_is_concentration(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(self._ax.get_ylabel(), "concentration")

    def test_title_applied_when_provided(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig, title="My Title")
        self.assertIn("My Title", self._ax.get_title())

    def test_kwargs_forwarded_to_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        result = self._tc.plot(ax=self._ax, fig=self._fig, title="Fwd")
        self.assertEqual(result.title, "Fwd")


def _makePerturbedTimecourse(
        value_fraction: float,
        species_fraction: float) -> Timecourse:
    """Return a pre-populated Timecourse with the given perturbation parameters."""
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
            rng.standard_normal((NUM_POINT, NUM_SPECIES)),
            index=timepoint_arr,
            columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    jacobian_collection_arr = rng.standard_normal((NUM_POINT, NUM_SPECIES, NUM_SPECIES))
    return Timecourse(
            model=_makeModel(),
            timecourse_df=timecourse_df,
            jacobian_collection_arr=jacobian_collection_arr,
            perturbation_value_fraction=value_fraction,
            perturbation_species_fraction=species_fraction,
    )


class TestTimecourseGetPerturbedInitialValues(unittest.TestCase):
    """Tests for Timecourse._getPerturbedInitialValues.

    ANTIMONY_MODEL initial values: S1=10.0, S2=0.0
    Use species_fraction=1.0 to select all species deterministically.
    """

    def test_empty_when_species_fraction_zero(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=0.0)
        self.assertEqual(tc.perturbation_dct, {})

    def test_empty_when_num_perturb_rounds_to_zero(self) -> None:
        if IGNORE_TESTS:
            return
        # int(0.4 * 2) == 0
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=0.4)
        self.assertEqual(tc.perturbation_dct, {})

    def test_all_species_selected_when_fraction_one(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=1.0)
        self.assertEqual(len(tc.perturbation_dct), NUM_SPECIES)

    def test_keys_are_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=1.0)
        self.assertTrue(set(tc.perturbation_dct.keys()).issubset(
                set(tc.model.species_names)))

    def test_values_are_floats(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=1.0)
        for v in tc.perturbation_dct.values():
            self.assertIsInstance(v, float)

    def test_perturbed_value_correct_for_nonzero_species(self) -> None:
        if IGNORE_TESTS:
            return
        # S1=10.0, value_fraction=0.1 → 10.0 + 0.1*10.0 = 11.0
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=1.0)
        self.assertAlmostEqual(tc.perturbation_dct["S1"], 11.0)

    def test_perturbed_value_zero_for_zero_initial_species(self) -> None:
        if IGNORE_TESTS:
            return
        # S2=0.0; perturbation = value_fraction * 0.0 = 0.0
        tc = _makePerturbedTimecourse(value_fraction=0.5, species_fraction=1.0)
        self.assertAlmostEqual(tc.perturbation_dct["S2"], 0.0)

    def test_clamping_prevents_negative_values(self) -> None:
        if IGNORE_TESTS:
            return
        # S1=10.0, value_fraction=-2.0 → 10 + (-2)*10 = -10 → clamped to 0
        tc = _makePerturbedTimecourse(value_fraction=-2.0, species_fraction=1.0)
        self.assertAlmostEqual(tc.perturbation_dct["S1"], 0.0)

    def test_values_unchanged_when_value_fraction_zero(self) -> None:
        if IGNORE_TESTS:
            return
        # perturbation = 0 * original = 0 → perturbed == original
        tc = _makePerturbedTimecourse(value_fraction=0.0, species_fraction=1.0)
        for name, value in tc.perturbation_dct.items():
            self.assertAlmostEqual(value, tc.model.initial_value_dct[name])


class TestTimecourseSetInitialValues(unittest.TestCase):
    """Tests for Timecourse._setInitialValues."""

    def setUp(self) -> None:
        self._tc = _makeTimecourse()
        self._rr = te.loadSBMLModel(self._tc.model.sbml_str)
        self._rr.reset()

    def test_sets_value_for_species_in_dict(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc._setInitialValues(self._rr, {"S1": 15.0})
        self.assertAlmostEqual(self._rr["S1"], 15.0)

    def test_ignores_species_not_in_dict(self) -> None:
        if IGNORE_TESTS:
            return
        original_s2 = float(self._rr["S2"])
        self._tc._setInitialValues(self._rr, {"S1": 15.0})
        self.assertAlmostEqual(self._rr["S2"], original_s2)

    def test_empty_dict_changes_nothing(self) -> None:
        if IGNORE_TESTS:
            return
        original_s1 = float(self._rr["S1"])
        self._tc._setInitialValues(self._rr, {})
        self.assertAlmostEqual(self._rr["S1"], original_s1)


class TestTimecourseNumPerturbed(unittest.TestCase):
    """Tests for Timecourse.num_perturbed."""

    def test_zero_when_species_fraction_zero(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=0.0)
        self.assertEqual(tc.num_perturbation, 0)

    def test_matches_len_perturbation_dct(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makePerturbedTimecourse(value_fraction=0.1, species_fraction=1.0)
        self.assertEqual(tc.num_perturbation, len(tc.perturbation_dct))


class TestMakeTimecourses(unittest.TestCase):
    """Tests for Timecourse.makeTimecourses."""

    model: Model

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _makeModel()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_list(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(self.model, end_time=10.0, is_plot=False)
        self.assertIsInstance(result, list)

    def test_single_combination_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertEqual(len(result), 1)

    def test_two_value_fractions_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertEqual(len(result), 2)

    def test_cartesian_product_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertEqual(len(result), 4)

    def test_all_items_are_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        for item in result:
            self.assertIsInstance(item, Timecourse)

    def test_value_fraction_varies_slowest(self) -> None:
        if IGNORE_TESTS:
            return
        # [0.0, 0.1] x [0.5, 1.0] → indices 0,1 share value_frac=0.0; 2,3 share 0.1
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertEqual(result[0].perturbation_value_fraction,
                result[1].perturbation_value_fraction)
        self.assertNotEqual(result[0].perturbation_value_fraction,
                result[2].perturbation_value_fraction)

    def test_perturbation_value_fraction_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.2],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertAlmostEqual(result[0].perturbation_value_fraction, 0.0)
        self.assertAlmostEqual(result[1].perturbation_value_fraction, 0.2)

    def test_perturbation_species_fraction_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertAlmostEqual(result[0].perturbation_species_fraction, 0.5)
        self.assertAlmostEqual(result[1].perturbation_species_fraction, 1.0)

    def test_num_point_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0, num_point=7, is_plot=False)
        self.assertEqual(result[0].num_point, 7)

    def test_start_time_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, start_time=2.0, end_time=10.0, is_plot=False)
        self.assertAlmostEqual(result[0].start_time, 2.0)

    def test_is_plot_false_creates_no_figure(self) -> None:
        if IGNORE_TESTS:
            return
        before = set(plt.get_fignums())
        Timecourse.makeTimecourses(self.model, end_time=10.0, is_plot=False)
        self.assertEqual(before, set(plt.get_fignums()))

    def test_is_plot_true_creates_one_subplot_per_species(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        self.assertEqual(len(plt.gcf().axes), NUM_SPECIES)

    def test_subplot_titles_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        titles = [ax.get_title() for ax in plt.gcf().axes]
        self.assertEqual(titles, self.model.species_names)

    def test_subplot_xlabel_is_time(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        for ax in plt.gcf().axes:
            self.assertEqual(ax.get_xlabel(), "time")

    def test_subplot_ylabel_is_concentration(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        for ax in plt.gcf().axes:
            self.assertEqual(ax.get_ylabel(), "concentration")


class TestTimecourseEqRealBiomodel(unittest.TestCase):
    """Verifies __eq__ with a real BioModel timecourse loaded from the zip archive."""

    @unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
    def test_biomodel53_equals_itself(self) -> None:
        if IGNORE_TESTS:
            return
        from timecourse_iterator import TimecourseIterator  # type: ignore
        tc = TimecourseIterator().getTimecourse(BIOMODEL_53)
        self.assertEqual(tc, tc)


if __name__ == "__main__":
    unittest.main()
