"""Tests for NLCurve."""

import os
import tempfile
import unittest

#matplotlib.use("Agg")  #  type: ignore

import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
import numpy as np  # type: ignore
import pandas as pd # type: ignore

import src.constants as cn  # type: ignore
from nl_curve import NLCurve  # type: ignore


IGNORE_TESTS = False


def _biomd_name(model_num: int) -> str:
    """Construct a system_id matching Model.getBiomodelName's format."""
    return f"BIOMD{model_num:010d}"


def _write_csv(tmpdir: str, rows: list) -> str:
    """Write a small PiecewisePredictions CSV into tmpdir and return its path."""
    path = os.path.join(tmpdir, "predictions.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestNLCurveInit(unittest.TestCase):
    """Tests for NLCurve.__init__."""

    def test_constructs_with_two_boundaries(self) -> None:
        """A two-element input produces one segment of the expected length."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0.0, 5.0])
        self.assertEqual(list(nl._boundaries), [0.0, 5.0])
        np.testing.assert_array_equal(nl._segment_arr, [5.0])

    def test_sorts_unsorted_boundaries(self) -> None:
        """Boundaries are sorted regardless of input order."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([3.0, 0.0, 1.0])
        self.assertEqual(list(nl._boundaries), [0.0, 1.0, 3.0])

    def test_curve_ends_at_one(self) -> None:
        """The cumulative curve always reaches 1.0 at the largest segment length."""
        if IGNORE_TESTS:
            return
        for bounds in [[0, 1, 2], [0, 1, 3], [0, 2, 5]]:
            nl = NLCurve(bounds)  # type: ignore
            self.assertAlmostEqual(float(nl.curve_ser.iloc[-1]), 1.0)

    def test_curve_values_in_unit_interval(self) -> None:
        """All curve values lie in [0, 1]."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4])
        self.assertTrue((nl.curve_ser.to_numpy() >= 0).all())
        self.assertTrue((nl.curve_ser.to_numpy() <= 1.0 + 1e-9).all())

    def test_empty_boundaries_raises(self) -> None:
        """An empty list raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve([])

    def test_single_boundary_raises(self) -> None:
        """A single-element list raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve([5])


class TestNLCurvecopy(unittest.TestCase):
    """Tests for NLCurve.copy()."""

    def test_returns_new_instance(self) -> None:
        """copy() returns an independent NLCurve instance."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3])
        other = nl.copy()
        self.assertIsInstance(other, NLCurve)
        self.assertIsNot(nl, other)

    def test_copy_preserves_curve(self) -> None:
        """The copy has identical curve values."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2])
        other = nl.copy()
        pd.testing.assert_series_equal(nl.curve_ser, other.curve_ser)

    def test_copy_is_independent(self) -> None:
        """Mutating the original does not affect the copy."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2])
        other = nl.copy()
        nl._boundaries.append(5.0)
        self.assertEqual(list(other._boundaries), [0, 1, 2])


class TestNLCurveDist(unittest.TestCase):
    """Tests for NLCurve.dist()."""

    def test_self_distance_is_zero(self) -> None:
        """A curve's distance to itself is exactly 0."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2, 3])
        self.assertAlmostEqual(nl.dist(nl), 0.0)

    def test_distance_symmetric(self) -> None:
        """dist(a, b) == dist(b, a)."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 2, 5])
        self.assertAlmostEqual(a.dist(b), b.dist(a))

    def test_distance_positive_for_different_curves(self) -> None:
        """Different boundaries produce a strictly positive distance."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 1, 3])
        self.assertGreater(a.dist(b), 0.0)

    def test_distance_identical_boundaries_zero(self) -> None:
        """Curves with identical boundaries have distance zero."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 1, 2])
        self.assertAlmostEqual(a.dist(b), 0.0)


class TestMakeMergedCurve(unittest.TestCase):
    """Tests for NLCurve.makeMergedCurve()."""

    def test_returns_series(self) -> None:
        """makeMergedCurve returns a pd.Series."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 2, 5])
        merged = a.makeMergedCurve(b)
        self.assertIsInstance(merged, pd.Series)

    def test_index_is_union_of_both(self) -> None:
        """The merged index contains every unique segment length from both curves."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 3])   # lengths {1, 2}
        b = NLCurve([0, 2, 5])   # lengths {2, 3}
        merged = a.makeMergedCurve(b)
        expected_index = sorted({1.0, 2.0, 3.0})
        import pdb; pdb.set_trace()
        self.assertEqual(list(merged.index), expected_index)

    def test_cumsum_ends_at_one(self) -> None:
        """The returned series cumulates to 1.0."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 2, 5])
        merged = a.makeMergedCurve(b)
        self.assertAlmostEqual(float(merged.iloc[-1]), 1.0)

    def test_cumsum_monotonic_non_decreasing(self) -> None:
        """The returned series is monotonic non-decreasing."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 3])
        b = NLCurve([0, 2, 5])
        merged = a.makeMergedCurve(b)
        diffs = np.diff(merged.to_numpy())
        self.assertTrue((diffs >= -1e-12).all())



class TestMakeNLDensity(unittest.TestCase):
    """Tests for NLCurve._makeNLDensity()."""

    def test_single_segment_density(self) -> None:
        """A single segment produces a density with one entry at value 1.0."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 5])
        density = nl._makeNLDensity()
        self.assertEqual(len(density), 1)
        self.assertIn(5.0/5.0, density.index)
        self.assertAlmostEqual(float(density[1.0]), 1.0)

    def test_duplicate_lengths_grouped(self) -> None:
        """Equal-length segments are grouped into a single index entry."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2, 3])   # three segments of length 1
        density = nl._makeNLDensity()
        self.assertEqual(len(density), 1)
        self.assertIn(1.0/3.0, density.index)

    def test_density_values_sum_to_one(self) -> None:
        """Density values sum to 1.0 (within floating-point tolerance)."""
        if IGNORE_TESTS:
            return
        for bounds in [[0, 1, 2], [0, 1, 3], [0, 2, 5, 7]]:
            nl = NLCurve(bounds)  # type: ignore
            density = nl._makeNLDensity()
            self.assertAlmostEqual(float(density.sum()), 1.0)

    def test_density_index_sorted_ascending(self) -> None:
        """Density index is sorted in ascending order."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 3, 1, 5])
        density = nl._makeNLDensity()
        idx = list(density.index.to_numpy())
        self.assertEqual(idx, sorted(idx))


class TestMakeNLCurve(unittest.TestCase):
    """Tests for NLCurve._makeNLCurve()."""

    def test_curve_is_cumulative(self) -> None:
        """The curve is monotonically non-decreasing."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4, 7])
        curve = nl._makeNLCurve()
        diffs = np.diff(curve.to_numpy())
        self.assertTrue((diffs >= -1e-12).all())

    def test_curve_ends_at_one(self) -> None:
        """The final value of the curve is 1.0."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4])
        self.assertAlmostEqual(float(nl._makeNLCurve().iloc[-1]), 1.0)

    def test_curve_first_value_equals_smallest_segment_fraction(self) -> None:
        """The first curve value equals the smallest segment's share of total time."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3])   # segments {1, 2}; smallest=1/3
        self.assertAlmostEqual(float(nl._makeNLCurve().iloc[0]), 1.0 / 3.0)



class TestParseChangepoints(unittest.TestCase):
    """Tests for NLCurve._parseChangepoints()."""

    def test_string_list_parsed_to_numbers(self) -> None:
        """A string representation of a list is parsed into a Python list."""
        if IGNORE_TESTS:
            return
        result = NLCurve._parseChangepoints("[0.0, 2.5, 4]")
        self.assertEqual(result, [0.0, 2.5, 4])

    def test_int_list_preserved(self) -> None:
        """A list of ints is returned unchanged."""
        if IGNORE_TESTS:
            return
        result = NLCurve._parseChangepoints([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_mixed_int_float_list_accepted(self) -> None:
        """Lists containing both ints and floats are accepted."""
        if IGNORE_TESTS:
            return
        result = NLCurve._parseChangepoints([0, 1.5, 2])
        self.assertEqual(result, [0, 1.5, 2])

    def test_invalid_string_raises(self) -> None:
        """A string that is not a valid literal raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve._parseChangepoints("not a list")

    def test_non_list_type_raises(self) -> None:
        """Passing a non-string, non-list value raises ValueError."""
        if IGNORE_TESTS:
            return
        for value in [123, 4.5, None]:
            with self.assertRaises(ValueError):
                NLCurve._parseChangepoints(value)  # type: ignore

    def test_list_with_non_numeric_element_rejected(self) -> None:
        """A list containing a string element raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve._parseChangepoints([1, "two"])



class TestDataframeColumns(unittest.TestCase):
    """Tests for NLCurve.getDataframeColumns()."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """The returned DataFrame has boundaries, aggregation_type, system_id."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0, 5.0],
                    cn.COL_AGGREGATION_TYPE: "model",
                    cn.COL_SYSTEM_ID: _biomd_name(1),
                    cn.COL_NUM_TIMEPOINT: 10,
                }],
            )
            df = NLCurve.getDataframeColumns(path)
            self.assertEqual(list(df.columns), [
                cn.COL_BOUNDARIES, cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID,
            ])

    def test_boundaries_prepend_zero_and_append_endtime_minus_one(self) -> None:
        """Boundaries are constructed as [0.0] + changepoints + [num_timepoint - 1]."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0, 5.0],
                    cn.COL_AGGREGATION_TYPE: "model",
                    cn.COL_SYSTEM_ID: _biomd_name(1),
                    cn.COL_NUM_TIMEPOINT: 10,
                }],
            )
            df = NLCurve.getDataframeColumns(path)
            self.assertEqual(df[cn.COL_BOUNDARIES].iloc[0], [0.0, 2.0, 5.0, 9])

    def test_missing_path_raises_valueerror(self) -> None:
        """Passing a non-existent path raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve.getDataframeColumns("/tmp/does_not_exist.csv")

    def test_missing_columns_raises_valueerror(self) -> None:
        """A CSV missing one of the required columns raises ValueError."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "predictions.csv")
            pd.DataFrame([{"foo": 1}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                NLCurve.getDataframeColumns(path)



class TestFromPSDPredictions(unittest.TestCase):
    """Tests for NLCurve.fromPSDPredictions()."""

    def _make_csv(self, tmpdir: str, rows: list) -> str:
        path = os.path.join(tmpdir, "predictions.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_model_level_creates_curve(self) -> None:
        """Selecting model-level aggregation returns a valid NLCurve."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0, 4.0],
                    cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL,
                    cn.COL_SYSTEM_ID: _biomd_name(1),
                    cn.COL_NUM_TIMEPOINT: 10,
                }],
            )
            nl, _ = NLCurve.fromPSDPredictions(path, model_num=1)
            self.assertIsInstance(nl, NLCurve)

    def test_species_level_filters_by_name(self) -> None:
        """Passing species_name selects the matching aggregation row."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [
                    {
                        cn.COL_CHANGEPOINTS: [2.0],
                        cn.COL_AGGREGATION_TYPE: "S1",
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_NUM_TIMEPOINT: 5,
                    },
                    {
                        cn.COL_CHANGEPOINTS: [3.0, 4.0],
                        cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL,
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_NUM_TIMEPOINT: 5,
                    },
                ],
            )
            nl, _ = NLCurve.fromPSDPredictions(path, model_num=1, species_name="S1")
            self.assertIn(2.0, nl._boundaries)

    def test_no_matching_rows_raises(self) -> None:
        """When no rows match the filter, ValueError is raised."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0],
                    cn.COL_AGGREGATION_TYPE: "model",
                    cn.COL_SYSTEM_ID: _biomd_name(42),
                    cn.COL_NUM_TIMEPOINT: 5,
                }],
            )
            with self.assertRaises(ValueError):
                NLCurve.fromPSDPredictions(path, model_num=1)

    def test_multiple_matching_rows_raises(self) -> None:
        """When more than one row matches the filter, ValueError is raised."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [
                    {
                        cn.COL_CHANGEPOINTS: [1.0],
                        cn.COL_AGGREGATION_TYPE: "model",
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_NUM_TIMEPOINT: 5,
                    },
                    {
                        cn.COL_CHANGEPOINTS: [2.0],
                        cn.COL_AGGREGATION_TYPE: "model",
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_NUM_TIMEPOINT: 5,
                    },
                ],
            )
            with self.assertRaises(ValueError):
                NLCurve.fromPSDPredictions(path, model_num=1)


class TestPlotNLCurve(unittest.TestCase):
    """Smoke test for NLCurve.plotNLCurve()."""

    def test_plot_does_not_raise(self) -> None:
        """plotNLCurve executes without raising an exception."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
        nl = NLCurve([0, 1, 2, 3, 5], name="Test Curve")
        fig = plt.figure()
        try:
            nl.plotNLCurve()
            #plt.show()
        finally:
            plt.close(fig)

class TestEnd2End(unittest.TestCase):
    """Tests for NLCurve.fromPSDPredictions()."""

    model_num = 234
    model_num = 42
    model_num = 10
    model_num = 343
    model_path = os.path.join(cn.TEST_DIR, "testdata", "piecewise_predictions.csv")
    nl_curve = NLCurve.fromPSDPredictions(model_path, model_num=model_num)[0]

    def test_model_level_creates_curve(self) -> None:
        """Selecting model-level aggregation returns a valid NLCurve."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsInstance(self.nl_curve, NLCurve)
            self.assertGreater(len(self.nl_curve.curve_ser), 0)

    def test_plot_does_not_raise(self) -> None:
        """plotNLCurve executes without raising an exception."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
        nl = self.nl_curve
        fig = plt.figure()
        try:
            nl.plotNLCurve()
            #plt.show()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
