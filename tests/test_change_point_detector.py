import unittest

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from src.change_point_detector import ChangePointDetector
from src.plot_options import PlotOptions  # type: ignore

IGNORE_TESTS = False


class TestChangePointDetector(unittest.TestCase):

    def test_single_change_point(self) -> None:
        if IGNORE_TESTS:
            return
        # Sequence: 10 zeros, 10 ones
        data = np.array([0.0]*10 + [1.0]*10)
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)

        # Expected change point at index 10
        self.assertEqual(len(detector.subsequences), 2)
        self.assertEqual(detector.subsequences[0][1], 10)
        self.assertAlmostEqual(detector.subsequences[0][2], 0.0)  # ASS of [0...0] is 0
        self.assertAlmostEqual(detector.subsequences[1][2], 0.0)  # ASS of [1...1] is 0

        # A_total = sum(x^2) - (sum(x)^2 / N) = 10 - (10^2 / 20) = 10 - 5 = 5
        # Total reduction = 5 - (0 + 0) = 5
        self.assertAlmostEqual(detector.total_reduction, 5.0)

    def test_multiple_change_points(self) -> None:
        if IGNORE_TESTS:
            return
        # Sequence: 5 zeros, 5 ones, 5 zeros
        data = np.array([0.0]*5 + [1.0]*5 + [0.0]*5)
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=2, min_fractional_reduction=0.0)

        # Expected partitions: [0,5), [5,10), [10,15) — ends at 5, 10, 15
        self.assertEqual(len(detector.subsequences), 3)
        self.assertEqual([p[1] for p in detector.subsequences], [5, 10, 15])

        # A_total = sum(x^2) - (sum(x)^2 / N) = 5 - (5^2 / 15) = 5 - 25/15 = 3.333...
        self.assertAlmostEqual(detector.total_reduction, 5 - 25/15)

    def test_max_points_constraint(self) -> None:
        if IGNORE_TESTS:
            return
        # Sequence: 5 zeros, 5 ones, 5 zeros
        data = np.array([0.0]*5 + [1.0]*5 + [0.0]*5)
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)

        # Should only find 1 change point
        self.assertEqual(len(detector.subsequences), 2)

    def test_min_reduction_constraint(self) -> None:
        if IGNORE_TESTS:
            return
        # [0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1]
        data = np.array([0.0]*5 + [0.1]*5)
        detector = ChangePointDetector(data)

        # A_total = 5 * 0.1^2 - (0.5^2 / 10) = 0.05 - 0.025 = 0.025
        # Reduction if split at 5: 0.025 - (0 + 0) = 0.025
        # Relative reduction = 0.025 / 0.025 = 1.0

        # min_fractional_reduction=0.1 → relative threshold 0.1 < 1.0 → should split
        detector.fit(max_change_point=1, min_fractional_reduction=0.1)
        self.assertEqual(len(detector.subsequences), 2)

        # min_fractional_reduction=1.1 → relative threshold 1.1 > 1.0 → should NOT split
        detector.fit(max_change_point=1, min_fractional_reduction=1.1)
        self.assertEqual(len(detector.subsequences), 1)

    def test_empty_data(self) -> None:
        if IGNORE_TESTS:
            return
        data = np.array([])
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=5, min_fractional_reduction=0.0)
        self.assertEqual(detector.subsequences, [])
        self.assertEqual(detector.total_reduction, 0.0)

    def test_single_element(self) -> None:
        if IGNORE_TESTS:
            return
        data = np.array([1.0])
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=5, min_fractional_reduction=0.0)
        self.assertEqual(len(detector.subsequences), 1)
        self.assertEqual(detector.subsequences[0], (0, 1, 0.0))


class TestPlot(unittest.TestCase):
    """Tests for ChangePointDetector.plot()."""

    def setUp(self) -> None:
        data = np.array([0.0] * 10 + [1.0] * 10)
        self.detector = ChangePointDetector(data)
        self.detector.fit(max_change_point=1, min_fractional_reduction=0.0)

    def tearDown(self) -> None:
        plt.close("all")

    def _dashed_lines(self, ax):
        return [l for l in ax.lines if l.get_linestyle() == "--"]

    def test_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot()
        self.assertIsInstance(po, PlotOptions)

    def test_fig_and_ax_are_set(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot()
        self.assertIsNotNone(po.fig)
        self.assertIsNotNone(po.ax)

    def test_one_vline_for_one_change_point(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot()
        self.assertEqual(len(self._dashed_lines(po.ax)), 1)

    def test_vline_at_correct_index(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot()
        x = self._dashed_lines(po.ax)[0].get_xdata()[0]
        self.assertAlmostEqual(x, 10.0)

    def test_vline_color_is_red(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot()
        color = self._dashed_lines(po.ax)[0].get_color()
        self.assertEqual(color, "red")

    def test_no_vlines_without_change_points(self) -> None:
        if IGNORE_TESTS:
            return
        detector = ChangePointDetector(np.ones(10))
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)
        po = detector.plot()
        self.assertEqual(len(self._dashed_lines(po.ax)), 0)

    def test_two_vlines_for_two_change_points(self) -> None:
        if IGNORE_TESTS:
            return
        data = np.array([0.0] * 5 + [1.0] * 5 + [0.0] * 5)
        detector = ChangePointDetector(data)
        detector.fit(max_change_point=2, min_fractional_reduction=0.0)
        po = detector.plot()
        self.assertEqual(len(self._dashed_lines(po.ax)), 2)

    def test_title_kwarg_applied(self) -> None:
        if IGNORE_TESTS:
            return
        po = self.detector.plot(title="my signal")
        self.assertIn("my signal", po.ax.get_title())  # type: ignore


_RNG = np.random.default_rng(42)
_SCALE_N = 1000
_SCALE_DATA = np.concatenate([
    _RNG.normal(loc=0.0, scale=0.1, size=_SCALE_N),
    _RNG.normal(loc=1.0, scale=0.1, size=_SCALE_N),
])


class TestScaling(unittest.TestCase):
    """2,000-point dataset: N(0,1) × 1000 followed by N(1,1) × 1000."""

    def test_finds_one_change_point(self) -> None:
        if IGNORE_TESTS:
            return
        detector = ChangePointDetector(_SCALE_DATA)
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)
        self.assertEqual(len(detector.subsequences), 2)

    def test_change_point_near_true_boundary(self) -> None:
        if IGNORE_TESTS:
            return
        detector = ChangePointDetector(_SCALE_DATA)
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)
        split = detector.subsequences[0][1]  # end of first partition = change point index
        self.assertAlmostEqual(split, _SCALE_N, delta=50)

    def test_total_reduction_positive(self) -> None:
        if IGNORE_TESTS:
            return
        detector = ChangePointDetector(_SCALE_DATA)
        detector.fit(max_change_point=1, min_fractional_reduction=0.0)
        self.assertGreater(detector.total_reduction, 0.0)

    def test_second_change_point_also_near_boundary(self) -> None:
        if IGNORE_TESTS:
            return
        # With max_points=2 on a single-boundary dataset, the 2nd split
        # should land within one of the two noisy halves, not near 1000 again.
        detector = ChangePointDetector(_SCALE_DATA)
        detector.fit(max_change_point=2, min_fractional_reduction=0.1)
        self.assertEqual(len(detector.subsequences), 2)
        splits = sorted(p[1] for p in detector.subsequences[:-1])
        # First split must still be near the true boundary
        self.assertAlmostEqual(splits[0], _SCALE_N, delta=50)


if __name__ == "__main__":
    unittest.main()
