"""Tests for JacobianSignal."""
import os
import unittest

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.change_point_detector import ChangePointDetector  # type: ignore
from src.jacobian_signal import JacobianSignal  # type: ignore
from src.model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(cn.TIMECOURSE_ZIP_PATH)

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_SPECIES = 2
MODEL_NAME = "test_model"


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL, model_name=MODEL_NAME)


def _makeTimecourse(num_point: int = 20,
        seed: int = 42,
        timecourse_df: pd.DataFrame = None,  # type: ignore
        jacobian_collection_arr: np.ndarray = None,  # type: ignore
        ) -> Timecourse:
    """Return a Timecourse pre-populated with synthetic data (no simulation).

    Species values are shifted away from 0 so that standard deviations and
    ratios used by JacobianSignal's normalization are well behaved.
    """
    rng = np.random.default_rng(seed)
    timepoint_arr = np.linspace(0.0, 10.0, num_point)
    if timecourse_df is None:
        timecourse_df = pd.DataFrame(
                rng.standard_normal((num_point, NUM_SPECIES)) + 5.0,
                index=timepoint_arr,
                columns=["S1", "S2"],
        )
        timecourse_df.index.name = "time"
    if jacobian_collection_arr is None:
        jacobian_collection_arr = rng.standard_normal((num_point, NUM_SPECIES, NUM_SPECIES))
    return Timecourse(
            model=_makeModel(),
            timecourse_df=timecourse_df,
            jacobian_collection_arr=jacobian_collection_arr,
    )


def _makeJacobianSignal(**kwargs) -> JacobianSignal:
    return JacobianSignal(_makeTimecourse(**kwargs))


def _steppedJacobianCollection(num_point: int = 20) -> np.ndarray:
    """A Jacobian collection with a clean step change halfway through, used to
    exercise change-point detection deterministically."""
    half = num_point // 2
    j1 = np.array([[-1.0, 0.5], [0.3, -2.0]])
    j2 = np.array([[-5.0, 2.0], [1.0, -6.0]])
    return np.array([j1] * half + [j2] * (num_point - half))


class TestJacobianSignalInit(unittest.TestCase):
    """Tests for JacobianSignal.__init__ / _makeSignal."""

    def test_stores_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        js = JacobianSignal(tc)
        self.assertIs(js.timecourse, tc)

    def test_stores_num_species(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        self.assertEqual(js._num_species, NUM_SPECIES)

    def test_stores_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        self.assertEqual(js._species_names, ["S1", "S2"])

    def test_signal_arr_length_is_num_timepoint_minus_one(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 15
        js = _makeJacobianSignal(num_point=num_point)
        self.assertEqual(js.signal_arr.shape, (num_point - 1,))

    def test_species_signal_sq_arr_shape(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 15
        js = _makeJacobianSignal(num_point=num_point)
        self.assertEqual(js.species_signal_sq_arr.shape,
                (num_point, NUM_SPECIES, NUM_SPECIES))

    def test_signal_arr_is_nonnegative(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        self.assertTrue(np.all(js.signal_arr >= 0.0))

    def test_signal_arr_is_finite(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        self.assertTrue(np.all(np.isfinite(js.signal_arr)))

    def test_species_signal_sq_arr_is_finite(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        self.assertTrue(np.all(np.isfinite(js.species_signal_sq_arr)))

    def test_zero_signal_when_all_jacobians_identical(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 10
        rng = np.random.default_rng(1)
        constant_jacobian = rng.standard_normal((NUM_SPECIES, NUM_SPECIES))
        jacobian_collection_arr = np.tile(constant_jacobian, (num_point, 1, 1))
        # signal_sq_arr is 0 at every timepoint here, so _makeSignal's
        # species_signal_sq_arr /= signal_sq_arr divides 0/0; the source
        # handles this via a subsequent np.nan_to_num call, but numpy still
        # emits a RuntimeWarning for the intermediate division -- suppress it
        # here since it does not indicate a test failure.
        with np.errstate(invalid="ignore", divide="ignore"):
            js = _makeJacobianSignal(num_point=num_point,
                    jacobian_collection_arr=jacobian_collection_arr)
        np.testing.assert_allclose(js.signal_arr, 0.0, atol=1e-10)

    def test_no_nan_in_signal_when_jacobian_has_nan(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 10
        rng = np.random.default_rng(2)
        jacobian_collection_arr = rng.standard_normal((num_point, NUM_SPECIES, NUM_SPECIES))
        jacobian_collection_arr[3, 0, 1] = np.nan
        js = _makeJacobianSignal(num_point=num_point,
                jacobian_collection_arr=jacobian_collection_arr)
        self.assertFalse(np.any(np.isnan(js.signal_arr)))
        self.assertFalse(np.any(np.isnan(js.species_signal_sq_arr)))

    def test_no_exception_with_zero_variance_species(self) -> None:
        if IGNORE_TESTS:
            return
        # S1 has zero variance (constant); should not raise or produce NaN/inf.
        num_point = 10
        timepoint_arr = np.linspace(0.0, 10.0, num_point)
        rng = np.random.default_rng(3)
        timecourse_df = pd.DataFrame({
                "S1": np.full(num_point, 5.0),
                "S2": rng.standard_normal(num_point) + 5.0,
        }, index=timepoint_arr)
        timecourse_df.index.name = "time"
        js = _makeJacobianSignal(num_point=num_point, timecourse_df=timecourse_df)
        self.assertTrue(np.all(np.isfinite(js.signal_arr)))
        self.assertTrue(np.all(np.isfinite(js.species_signal_sq_arr)))


class TestNormalizeJacobian(unittest.TestCase):
    """Tests for JacobianSignal._normalizeJacobian (private helper, tested
    directly per this repo's existing convention -- see
    TestTimecourseGetPerturbedInitialValues in tests/test_timecourse.py)."""

    def test_preserves_shape(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        result = js._normalizeJacobian(js._jacobian_collection_arr, js._timecourse_df)
        self.assertEqual(result.shape, js._jacobian_collection_arr.shape)

    def test_normalization_uses_column_over_row_std_ratio(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 5
        timepoint_arr = np.linspace(0.0, 4.0, num_point)
        rng = np.random.default_rng(7)
        timecourse_df = pd.DataFrame(
                rng.standard_normal((num_point, NUM_SPECIES)) + 5.0,
                index=timepoint_arr, columns=["S1", "S2"])
        timecourse_df.index.name = "time"
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
        js = _makeJacobianSignal(num_point=num_point, timecourse_df=timecourse_df)
        result = js._normalizeJacobian(jacobian_arr, timecourse_df)
        std_arr = timecourse_df.to_numpy(dtype=float).std(axis=0, ddof=1)
        expected = np.array([[
                [1.0 * (std_arr[0] / std_arr[0]), 2.0 * (std_arr[1] / std_arr[0])],
                [3.0 * (std_arr[0] / std_arr[1]), 4.0 * (std_arr[1] / std_arr[1])],
        ]])
        np.testing.assert_allclose(result, expected)

    def test_zero_variance_column_does_not_divide_by_zero(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 5
        timepoint_arr = np.linspace(0.0, 4.0, num_point)
        timecourse_df = pd.DataFrame({
                "S1": np.full(num_point, 3.0),  # zero variance
                "S2": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }, index=timepoint_arr)
        timecourse_df.index.name = "time"
        jacobian_arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        js = _makeJacobianSignal(num_point=num_point, timecourse_df=timecourse_df)
        result = js._normalizeJacobian(jacobian_arr, timecourse_df)
        self.assertTrue(np.all(np.isfinite(result)))
        # S1's safe std is clamped to 1.0 since its true std is 0.
        std_s2 = timecourse_df["S2"].std(ddof=1)
        expected_01 = 2.0 * (std_s2 / 1.0)
        self.assertAlmostEqual(result[0, 0, 1], expected_01)

    def test_denans_jacobian_before_normalizing(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 5
        timepoint_arr = np.linspace(0.0, 4.0, num_point)
        rng = np.random.default_rng(9)
        timecourse_df = pd.DataFrame(
                rng.standard_normal((num_point, NUM_SPECIES)) + 5.0,
                index=timepoint_arr, columns=["S1", "S2"])
        timecourse_df.index.name = "time"
        jacobian_arr = np.array([[[np.nan, 2.0], [3.0, 4.0]]])
        js = _makeJacobianSignal(num_point=num_point, timecourse_df=timecourse_df)
        result = js._normalizeJacobian(jacobian_arr, timecourse_df)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertAlmostEqual(result[0, 0, 0], 0.0)


class TestDenan(unittest.TestCase):
    """Tests for JacobianSignal._denan (private helper, tested directly per
    this repo's existing convention)."""

    def test_replaces_nan_with_zero(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        arr = np.array([1.0, np.nan, 3.0])
        result = js._denan(arr)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])

    def test_leaves_non_nan_values_unchanged(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        arr = np.array([1.0, 2.0, 3.0])
        result = js._denan(arr)
        np.testing.assert_array_equal(result, arr)

    def test_no_nan_present_returns_equivalent_array(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        arr = np.zeros((3, 3))
        result = js._denan(arr)
        np.testing.assert_array_equal(result, arr)


class TestMakeDetector(unittest.TestCase):
    """Tests for JacobianSignal.fit."""

    def test_returns_change_point_detector(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        detector = js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertIsInstance(detector, ChangePointDetector)

    def test_detector_wraps_signal_arr(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        detector = js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        np.testing.assert_array_equal(detector.data_arr, js.signal_arr)

    def test_detector_has_been_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal()
        detector = js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertGreater(len(detector.subsequences), 0)

    def test_finds_change_point_for_stepped_jacobian(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 20
        js = _makeJacobianSignal(num_point=num_point,
                jacobian_collection_arr=_steppedJacobianCollection(num_point))
        detector = js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertEqual(len(detector.subsequences), 2)

    def test_no_change_point_when_max_change_point_is_zero(self) -> None:
        if IGNORE_TESTS:
            return
        num_point = 20
        js = _makeJacobianSignal(num_point=num_point,
                jacobian_collection_arr=_steppedJacobianCollection(num_point))
        detector = js.fit(max_change_point=0,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertEqual(len(detector.subsequences), 1)


class TestJacobianSignalPlot(unittest.TestCase):
    """Tests for JacobianSignal.plot.

    An earlier version of plot() unconditionally raised ValueError (the
    bottom panel plotted time_arr against self.signal_arr, which are
    mismatched in length by one element) and never called apply() on the
    returned PlotOptions. Both have since been fixed in the source
    (bottom panel now plots against time_arr[:-1]; apply() is called once
    for each panel after both are drawn). These tests cover the fixed,
    positive-path behavior.
    """

    def setUp(self) -> None:
        self.num_point = 20
        self.js = _makeJacobianSignal(num_point=self.num_point)

    def tearDown(self) -> None:
        plt.close("all")

    def _dashed_lines(self, ax):
        return [l for l in ax.lines if l.get_linestyle() == "--"]

    def test_plot_returns_two_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        result = self.js.plot(max_change_point=0,
                min_fractional_reduction=1.0, min_subsequence_length=1)
        self.assertEqual(len(result), 2)
        for po in result:
            self.assertIsInstance(po, PlotOptions)

    def test_plot_with_change_points_present(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal(num_point=self.num_point,
                jacobian_collection_arr=_steppedJacobianCollection(self.num_point))
        top_po, bot_po = js.plot(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertEqual(len(self._dashed_lines(top_po.ax)), 1)
        self.assertEqual(len(self._dashed_lines(bot_po.ax)), 1)

    def test_plot_vline_uses_plus_one_offset(self) -> None:
        if IGNORE_TESTS:
            return
        js = _makeJacobianSignal(num_point=self.num_point,
                jacobian_collection_arr=_steppedJacobianCollection(self.num_point))
        detector = js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        raw_split_idx = detector.subsequences[1].splice_start
        expected_time = js._timecourse_df.index[raw_split_idx + 1]
        top_po, _ = js.plot(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        vline_x = self._dashed_lines(top_po.ax)[0].get_xdata()[0]
        self.assertAlmostEqual(vline_x, expected_time)

    def test_signal_arr_is_one_shorter_than_timecourse_index(self) -> None:
        if IGNORE_TESTS:
            return
        # plot()'s bottom panel relies on this being true (it plots
        # time_arr[:-1] against signal_arr).
        self.assertEqual(len(self.js.signal_arr),
                len(self.js._timecourse_df.index) - 1)


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestJacobianSignalWithBiomodel45(unittest.TestCase):
    """Integration tests exercising JacobianSignal against a real BioModel
    (BIOMD0000000045) using its previously-serialized Timecourse from the
    timecourse zip archive (no live simulation needed). Built once in
    setUpClass since it's shared read-only fixture data."""

    js: JacobianSignal
    num_species: int
    num_point: int
    timecourse: Timecourse

    @classmethod
    def setUpClass(cls) -> None:
        if IGNORE_TESTS:
            return
        timecourse = TimecourseIterator().getTimecourse("BIOMD0000000045")
        cls.js = JacobianSignal(timecourse)
        cls.num_species = timecourse.model.num_species
        cls.num_point = timecourse.num_point

    def test_signal_arr_length_is_num_point_minus_one(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(self.js.signal_arr.shape, (self.num_point - 1,))

    def test_species_signal_sq_arr_shape(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(self.js.species_signal_sq_arr.shape,
                (self.num_point, self.num_species, self.num_species))

    def test_signal_arr_is_finite(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.js.signal_arr)))

    def test_species_signal_sq_arr_is_finite(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(np.isfinite(self.js.species_signal_sq_arr)))

    def test_signal_arr_is_nonnegative(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(self.js.signal_arr >= 0.0))

    def test_make_detector_returns_fitted_detector_with_subsequences(self) -> None:
        if IGNORE_TESTS:
            return
        detector = self.js.fit(max_change_point=1,
                min_fractional_reduction=0.0, min_subsequence_length=1)
        self.assertIsInstance(detector, ChangePointDetector)
        self.assertGreater(len(detector.subsequences), 0)

    def test_plot(self) -> None:
        if IGNORE_TESTS:
            return
        self.js.plot(max_change_point=4, min_fractional_reduction=0.0,
                min_subsequence_length=100)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
