"""Tests for Trajectory."""
import os
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import matplotlib  # type: ignore
matplotlib.use("Agg")

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from trajectory import Trajectory  # type: ignore

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_POINT = 11
NUM_SPECIES = 2


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL)


def _makeArrays(num_point: int = NUM_POINT, num_species: int = NUM_SPECIES):
    """Return consistent (jacobian_collection_arr, timepoint_arr,
    forcing_input_collection_arr, timecourse_df) for unit tests."""
    rng = np.random.default_rng(0)
    jacobian_collection_arr = rng.standard_normal((num_point, num_species, num_species))
    timepoint_arr = np.linspace(0.0, 10.0, num_point)
    forcing_input_collection_arr = rng.standard_normal((num_point, num_species))
    timecourse_df = pd.DataFrame(
            rng.standard_normal((num_point, num_species)),
            index=timepoint_arr,
            columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    return jacobian_collection_arr, timepoint_arr, forcing_input_collection_arr, timecourse_df


def _makeTrajectory(num_point: int = NUM_POINT) -> Trajectory:
    jc, tp, fi, tc = _makeArrays(num_point)
    return Trajectory(
            model=_makeModel(),
            jacobian_collection_arr=jc,
            timepoint_arr=tp,
            forcing_input_collection_arr=fi,
            timecourse_df=tc,
    )


class TestTrajectoryInit(unittest.TestCase):
    """Tests for Trajectory.__init__."""

    def test_constructs_with_valid_args(self) -> None:
        """Constructor does not raise with valid arguments."""
        if IGNORE_TESTS:
            return
        trajectory = _makeTrajectory()
        self.assertIsInstance(trajectory, Trajectory)

    def test_empty_timepoint_arr_raises(self) -> None:
        """Empty timepoint_arr raises ValueError."""
        if IGNORE_TESTS:
            return
        jc, _, fi, tc = _makeArrays()
        with self.assertRaises(ValueError):
            Trajectory(
                    model=_makeModel(),
                    jacobian_collection_arr=np.empty((0, NUM_SPECIES, NUM_SPECIES)),
                    timepoint_arr=np.array([]),
                    forcing_input_collection_arr=np.empty((0, NUM_SPECIES)),
                    timecourse_df=tc.iloc[:0],
            )

    def test_stores_model(self) -> None:
        """model is stored on the instance."""
        if IGNORE_TESTS:
            return
        model = _makeModel()
        jc, tp, fi, tc = _makeArrays()
        trajectory = Trajectory(
                model=model,
                jacobian_collection_arr=jc,
                timepoint_arr=tp,
                forcing_input_collection_arr=fi,
                timecourse_df=tc,
        )
        self.assertIs(trajectory.model, model)

    def test_stores_jacobian_collection_arr(self) -> None:
        """jacobian_collection_arr is stored on the instance."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        trajectory = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=jc,
                timepoint_arr=tp,
                forcing_input_collection_arr=fi,
                timecourse_df=tc,
        )
        np.testing.assert_array_equal(trajectory.jacobian_collection_arr, jc)

    def test_stores_timepoint_arr(self) -> None:
        """timepoint_arr is stored on the instance."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        trajectory = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=jc,
                timepoint_arr=tp,
                forcing_input_collection_arr=fi,
                timecourse_df=tc,
        )
        np.testing.assert_array_equal(trajectory.timepoint_arr, tp)

    def test_stores_forcing_input_collection_arr(self) -> None:
        """forcing_input_collection_arr is stored on the instance."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        trajectory = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=jc,
                timepoint_arr=tp,
                forcing_input_collection_arr=fi,
                timecourse_df=tc,
        )
        np.testing.assert_array_equal(trajectory.forcing_input_collection_arr, fi)

    def test_stores_timecourse_df(self) -> None:
        """timecourse_df is stored on the instance."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        trajectory = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=jc,
                timepoint_arr=tp,
                forcing_input_collection_arr=fi,
                timecourse_df=tc,
        )
        pd.testing.assert_frame_equal(trajectory.timecourse_df, tc)


class TestTrajectoryProperties(unittest.TestCase):
    """Tests for Trajectory scalar properties and cached array properties."""

    def setUp(self) -> None:
        self.jc, self.tp, self.fi, self.tc = _makeArrays()
        self.trajectory = Trajectory(
                model=_makeModel(),
                jacobian_collection_arr=self.jc,
                timepoint_arr=self.tp,
                forcing_input_collection_arr=self.fi,
                timecourse_df=self.tc,
        )

    def test_start_time(self) -> None:
        """start_time equals the first element of timepoint_arr."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.trajectory.start_time, float(self.tp[0]))

    def test_end_time(self) -> None:
        """end_time equals the last element of timepoint_arr."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.trajectory.end_time, float(self.tp[-1]))

    def test_num_point(self) -> None:
        """num_point equals len(timepoint_arr)."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.trajectory.num_point, len(self.tp))

    def test_jacobian_median_arr_shape(self) -> None:
        """jacobian_median_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        shape = self.trajectory.jacobian_median_arr.shape
        self.assertEqual(shape, (NUM_SPECIES, NUM_SPECIES))

    def test_jacobian_median_arr_values(self) -> None:
        """jacobian_median_arr equals element-wise median of jacobian_collection_arr."""
        if IGNORE_TESTS:
            return
        expected = np.median(self.jc, axis=0)
        np.testing.assert_array_almost_equal(
                self.trajectory.jacobian_median_arr, expected)

    def test_jacobian_median_arr_cached(self) -> None:
        """jacobian_median_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        first = self.trajectory.jacobian_median_arr
        second = self.trajectory.jacobian_median_arr
        self.assertIs(first, second)

    def test_jacobian_std_arr_shape(self) -> None:
        """jacobian_std_arr has shape (n_species, n_species)."""
        if IGNORE_TESTS:
            return
        shape = self.trajectory.jacobian_std_arr.shape
        self.assertEqual(shape, (NUM_SPECIES, NUM_SPECIES))

    def test_jacobian_std_arr_values(self) -> None:
        """jacobian_std_arr equals element-wise std of jacobian_collection_arr."""
        if IGNORE_TESTS:
            return
        expected = np.std(self.jc, axis=0)
        np.testing.assert_array_almost_equal(
                self.trajectory.jacobian_std_arr, expected)

    def test_jacobian_std_arr_cached(self) -> None:
        """jacobian_std_arr returns the same object on repeated access."""
        if IGNORE_TESTS:
            return
        first = self.trajectory.jacobian_std_arr
        second = self.trajectory.jacobian_std_arr
        self.assertIs(first, second)


class TestTrajectoryMakeSubmodel(unittest.TestCase):
    """Tests for Trajectory.makeSubmodel."""

    def setUp(self) -> None:
        self.trajectory = _makeTrajectory()

    def test_returns_trajectory(self) -> None:
        """makeSubmodel returns a Trajectory instance."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertIsInstance(sub, Trajectory)

    def test_timepoints_in_range(self) -> None:
        """All timepoints in submodel lie within [start_time, end_time]."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertTrue(np.all(sub.timepoint_arr >= 2.0))
        self.assertTrue(np.all(sub.timepoint_arr <= 7.0))

    def test_boundaries_inclusive(self) -> None:
        """makeSubmodel includes points exactly at start_time and end_time."""
        if IGNORE_TESTS:
            return
        full_start = self.trajectory.start_time
        full_end = self.trajectory.end_time
        sub = self.trajectory.makeSubmodel(full_start, full_end)
        self.assertEqual(sub.num_point, self.trajectory.num_point)

    def test_same_model(self) -> None:
        """Submodel shares the same model object."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertIs(sub.model, self.trajectory.model)

    def test_jacobian_collection_arr_sliced(self) -> None:
        """jacobian_collection_arr in submodel has one row per included timepoint."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertEqual(sub.jacobian_collection_arr.shape[0], sub.num_point)

    def test_forcing_input_collection_arr_sliced(self) -> None:
        """forcing_input_collection_arr in submodel has one row per included timepoint."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertEqual(sub.forcing_input_collection_arr.shape[0], sub.num_point)

    def test_timecourse_df_sliced(self) -> None:
        """timecourse_df in submodel has one row per included timepoint."""
        if IGNORE_TESTS:
            return
        sub = self.trajectory.makeSubmodel(2.0, 7.0)
        self.assertEqual(len(sub.timecourse_df), sub.num_point)


class TestTrajectoryLt(unittest.TestCase):
    """Tests for Trajectory.__lt__."""

    def _makeWithTimepoints(self, start: float, end: float) -> Trajectory:
        num = 5
        jc, _, fi, _ = _makeArrays(num)
        tp = np.linspace(start, end, num)
        tc = pd.DataFrame(
                np.zeros((num, NUM_SPECIES)),
                index=tp,
                columns=["S1", "S2"],
        )
        tc.index.name = "time"
        return Trajectory(model=_makeModel(), jacobian_collection_arr=jc,
                timepoint_arr=tp, forcing_input_collection_arr=fi, timecourse_df=tc)

    def test_strictly_before_is_true(self) -> None:
        """end_time < start_time of other returns True."""
        if IGNORE_TESTS:
            return
        traj1 = self._makeWithTimepoints(0.0, 5.0)
        traj2 = self._makeWithTimepoints(10.0, 20.0)
        self.assertTrue(traj1 < traj2)

    def test_end_equals_start_is_true(self) -> None:
        """end_time == start_time of other also returns True."""
        if IGNORE_TESTS:
            return
        traj1 = self._makeWithTimepoints(0.0, 5.0)
        traj2 = self._makeWithTimepoints(5.0, 10.0)
        self.assertTrue(traj1 < traj2)

    def test_overlapping_is_false(self) -> None:
        """Overlapping intervals return False."""
        if IGNORE_TESTS:
            return
        traj1 = self._makeWithTimepoints(0.0, 7.0)
        traj2 = self._makeWithTimepoints(5.0, 10.0)
        self.assertFalse(traj1 < traj2)

    def test_reversed_order_is_false(self) -> None:
        """traj2 < traj1 is False when traj1 comes first."""
        if IGNORE_TESTS:
            return
        traj1 = self._makeWithTimepoints(0.0, 5.0)
        traj2 = self._makeWithTimepoints(10.0, 20.0)
        self.assertFalse(traj2 < traj1)

    def test_not_lt_to_non_trajectory(self) -> None:
        """Comparing a Trajectory to a non-Trajectory returns NotImplemented."""
        if IGNORE_TESTS:
            return
        traj = _makeTrajectory()
        self.assertIs(traj.__lt__("not a trajectory"), NotImplemented)


class TestTrajectoryEq(unittest.TestCase):
    """Tests for Trajectory.__eq__."""

    def test_equal_trajectories(self) -> None:
        """Two Trajectory instances built from identical data are equal."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        model = _makeModel()
        traj1 = Trajectory(model=model, jacobian_collection_arr=jc,
                timepoint_arr=tp, forcing_input_collection_arr=fi, timecourse_df=tc)
        traj2 = Trajectory(model=model, jacobian_collection_arr=jc.copy(),
                timepoint_arr=tp.copy(), forcing_input_collection_arr=fi.copy(),
                timecourse_df=tc.copy())
        self.assertEqual(traj1, traj2)

    def test_different_timepoints_not_equal(self) -> None:
        """Trajectories with different timepoint_arr are not equal."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        model = _makeModel()
        tp2 = tp + 1.0
        traj1 = Trajectory(model=model, jacobian_collection_arr=jc,
                timepoint_arr=tp, forcing_input_collection_arr=fi, timecourse_df=tc)
        traj2 = Trajectory(model=model, jacobian_collection_arr=jc,
                timepoint_arr=tp2, forcing_input_collection_arr=fi, timecourse_df=tc)
        self.assertNotEqual(traj1, traj2)

    def test_different_end_time_source_not_equal(self) -> None:
        """Trajectories with different end_time_source are not equal."""
        if IGNORE_TESTS:
            return
        jc, tp, fi, tc = _makeArrays()
        model = _makeModel()
        traj1 = Trajectory(model=model, jacobian_collection_arr=jc,
                timepoint_arr=tp, forcing_input_collection_arr=fi,
                timecourse_df=tc, end_time_source="a")
        traj2 = Trajectory(model=model, jacobian_collection_arr=jc,
                timepoint_arr=tp, forcing_input_collection_arr=fi,
                timecourse_df=tc, end_time_source="b")
        self.assertNotEqual(traj1, traj2)

    def test_not_equal_to_non_trajectory(self) -> None:
        """Comparing a Trajectory to a non-Trajectory returns NotImplemented."""
        if IGNORE_TESTS:
            return
        traj = _makeTrajectory()
        self.assertIs(traj.__eq__("not a trajectory"), NotImplemented)


class TestTrajectoryMakeFromSimulation(unittest.TestCase):
    """Integration tests for Trajectory.makeFromSimulation."""
    model: Model
    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model(ANTIMONY_MODEL)
        cls.trajectory = Trajectory.makeFromSimulation(
                cls.model, num_point=NUM_POINT)

    def test_returns_trajectory(self) -> None:
        """makeFromSimulation returns a Trajectory."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.trajectory, Trajectory)

    def test_timecourse_df_columns_match_species_names(self) -> None:
        """timecourse_df columns equal the model's species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.trajectory.timecourse_df.columns),
                self.model.species_names,
        )

    def test_jacobian_collection_arr_shape(self) -> None:
        """jacobian_collection_arr has shape (num_point, n_species, n_species)."""
        if IGNORE_TESTS:
            return
        n = self.model.num_species
        expected = (self.trajectory.num_point, n, n)
        self.assertEqual(self.trajectory.jacobian_collection_arr.shape, expected)

    def test_forcing_input_collection_arr_shape(self) -> None:
        """forcing_input_collection_arr has shape (num_point, n_species)."""
        if IGNORE_TESTS:
            return
        n = self.model.num_species
        expected = (self.trajectory.num_point, n)
        self.assertEqual(self.trajectory.forcing_input_collection_arr.shape, expected)

    def test_num_point_matches(self) -> None:
        """num_point matches the requested value."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.trajectory.num_point, NUM_POINT)

    def test_explicit_end_time(self) -> None:
        """makeFromSimulation respects an explicit end_time."""
        if IGNORE_TESTS:
            return
        traj = Trajectory.makeFromSimulation(
                self.model, end_time=5.0, num_point=NUM_POINT)
        self.assertAlmostEqual(traj.end_time, 5.0, places=5)

    def test_timecourse_df_index_name(self) -> None:
        """timecourse_df index is named 'time'."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.trajectory.timecourse_df.index.name, "time")

    def test_start_time_is_zero(self) -> None:
        """Default start_time is cn.START_TIME (0.0)."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.trajectory.start_time, cn.START_TIME)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestTrajectoryBiomodel(unittest.TestCase):
    """Integration tests using a real BioModels SBML file."""
    model: Model
    trajectory: Trajectory

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.makeBiomodel("BIOMD0000000001")
        cls.trajectory = Trajectory.makeFromSimulation(
                cls.model, num_point=NUM_POINT)

    def test_returns_trajectory(self) -> None:
        """makeFromSimulation returns a Trajectory for a real BioModel."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.trajectory, Trajectory)

    def test_jacobian_collection_arr_shape(self) -> None:
        """jacobian_collection_arr has the correct shape for a real BioModel."""
        if IGNORE_TESTS:
            return
        n = self.model.num_species
        self.assertEqual(
                self.trajectory.jacobian_collection_arr.shape,
                (self.trajectory.num_point, n, n),
        )

    def test_timecourse_df_columns_match_species_names(self) -> None:
        """timecourse_df columns match species_names for a real BioModel."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.trajectory.timecourse_df.columns),
                self.model.species_names,
        )


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestTrajectoryMakeBiomodel(unittest.TestCase):
    """Tests for Trajectory.makeBiomodel."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = Trajectory.makeBiomodel(1, num_point=NUM_POINT)

    def test_returns_trajectory(self) -> None:
        """makeBiomodel returns a Trajectory instance."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.trajectory, Trajectory)

    def test_model_name_formatted_correctly(self) -> None:
        """model.model_name is formatted as BIOMD<10-digit-zero-padded>."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.trajectory.model.model_name, "BIOMD0000000001")

    def test_timecourse_df_columns_match_species_names(self) -> None:
        """timecourse_df columns equal the model's species names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(
                list(self.trajectory.timecourse_df.columns),
                self.trajectory.model.species_names,
        )

    def test_num_point_matches(self) -> None:
        """num_point matches the requested value."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.trajectory.num_point, NUM_POINT)

    def test_explicit_end_time_respected(self) -> None:
        """end_time kwarg is passed through to the simulation."""
        if IGNORE_TESTS:
            return
        traj = Trajectory.makeBiomodel(1, end_time=5.0, num_point=NUM_POINT)
        self.assertAlmostEqual(traj.end_time, 5.0, places=5)

    def test_jacobian_collection_arr_shape(self) -> None:
        """jacobian_collection_arr has shape (num_point, n_species, n_species)."""
        if IGNORE_TESTS:
            return
        n = self.trajectory.model.num_species
        self.assertEqual(
                self.trajectory.jacobian_collection_arr.shape,
                (self.trajectory.num_point, n, n),
        )


class TestTrajectoryPlotTimecourse(unittest.TestCase):
    """Tests for Trajectory.plotTimecourse."""

    def setUp(self) -> None:
        self.trajectory = _makeTrajectory()

    def test_returns_plot_options(self) -> None:
        """plotTimecourse returns a PlotOptions instance."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.plotTimecourse()
        self.assertIsInstance(result, PlotOptions)

    def test_line_count_matches_species(self) -> None:
        """One line per species is drawn."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.plotTimecourse()
        self.assertEqual(len(result.ax.lines), NUM_SPECIES)

    def test_kwargs_passed_to_plot_options(self) -> None:
        """kwargs such as title are forwarded to PlotOptions."""
        if IGNORE_TESTS:
            return
        result = self.trajectory.plotTimecourse(title="Test")
        self.assertEqual(result.ax.get_title(), "Test")


class TestTrajectoryMakeTimecourse(unittest.TestCase):
    """Tests for Trajectory.makeTimecourse."""
    model: Model
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model(ANTIMONY_MODEL)
        cls.df = Trajectory.makeTimecourse(
                cls.model,
                end_time=10.0,
                num_point=NUM_POINT,
        )

    def test_returns_dataframe(self) -> None:
        """makeTimecourse returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_columns_match_species_names(self) -> None:
        """DataFrame columns equal model.species_names."""
        if IGNORE_TESTS:
            return
        self.assertEqual(list(self.df.columns), self.model.species_names)

    def test_num_point_rows(self) -> None:
        """DataFrame has exactly num_point rows."""
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.df), NUM_POINT)

    def test_index_is_monotonically_increasing(self) -> None:
        """Time index is strictly monotonically increasing."""
        if IGNORE_TESTS:
            return
        self.assertTrue(self.df.index.is_monotonic_increasing)

    def test_end_time_respected(self) -> None:
        """Last index value is approximately 10.0."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.df.index[-1], 10.0, places=5)

    def test_start_time_defaults_to_cn_start_time(self) -> None:
        """First index value is approximately cn.START_TIME (0.0)."""
        if IGNORE_TESTS:
            return
        self.assertAlmostEqual(self.df.index[0], cn.START_TIME, places=5)

    def test_nonzero_start_time(self) -> None:
        """With start_time=2.0 the first index value is approximately 2.0."""
        if IGNORE_TESTS:
            return
        df2 = Trajectory.makeTimecourse(
                self.model,
                start_time=2.0,
                end_time=10.0,
                num_point=NUM_POINT,
        )
        self.assertAlmostEqual(df2.index[0], 2.0, places=5)

    def test_nonzero_start_time_num_point(self) -> None:
        """With start_time=2.0 the DataFrame still has exactly NUM_POINT rows."""
        if IGNORE_TESTS:
            return
        df2 = Trajectory.makeTimecourse(
                self.model,
                start_time=2.0,
                end_time=10.0,
                num_point=NUM_POINT,
        )
        self.assertEqual(len(df2), NUM_POINT)

    def test_concentrations_nonnegative(self) -> None:
        """All concentration values in the DataFrame are >= 0."""
        if IGNORE_TESTS:
            return
        self.assertTrue((self.df.values >= 0).all())

    def test_perturbation_changes_trajectory(self) -> None:
        """A non-zero perturbation produces different S1 values than baseline."""
        if IGNORE_TESTS:
            return
        df_perturbed = Trajectory.makeTimecourse(
                self.model,
                perturbation=0.2,
                end_time=10.0,
                num_point=NUM_POINT,
        )
        self.assertFalse(np.allclose(df_perturbed["S1"].values, self.df["S1"].values))

    def test_zero_perturbation_matches_baseline(self) -> None:
        """perturbation=0 returns the same result as the setUpClass baseline."""
        if IGNORE_TESTS:
            return
        df_zero = Trajectory.makeTimecourse(
                self.model,
                perturbation=0,
                end_time=10.0,
                num_point=NUM_POINT,
        )
        pd.testing.assert_frame_equal(df_zero, self.df)

    def test_zero_initial_concentration_unaffected_by_perturbation(self) -> None:
        """S2 starts at 0; perturbation=1.0 leaves initial S2 value at ~0."""
        if IGNORE_TESTS:
            return
        df_big = Trajectory.makeTimecourse(
                self.model,
                perturbation=1.0,
                end_time=10.0,
                num_point=NUM_POINT,
        )
        self.assertAlmostEqual(df_big["S2"].iloc[0], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
