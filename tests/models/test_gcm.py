import numpy as np
from pytest import mark

from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)
from models.GCM.gcm import GCM
from models.GCM.time_stepping import RK2, RK4, euler_forward


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_gcm_initialization(gcm_config, f_schedule, polynomial_parameterization):
    """Test GCM initialization."""
    gcm = GCM(polynomial_parameterization, F_schedule=f_schedule)

    assert gcm.f_schedule is f_schedule
    assert gcm.parameterization is polynomial_parameterization


def test_gcm_rhs_shape(gcm_config, sample_x, polynomial_parameterization):
    """Test that GCM.rhs returns correct shape."""
    gcm = GCM(polynomial_parameterization, F_schedule=gcm_config.f_schedule)

    rhs = gcm.rhs(sample_x, t=0.0)

    assert rhs.shape == sample_x.shape
    assert np.all(np.isfinite(rhs))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_gcm_call_rk2_shapes(
    gcm_config, f_schedule, sample_x, polynomial_parameterization
):
    """Test that GCM integration returns correct shapes with RK2."""
    gcm = GCM(polynomial_parameterization, F_schedule=f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK2,
    )

    nt = int(gcm_config.total_time / gcm_config.si)

    assert hist.shape == (nt + 1, gcm_config.K)
    assert time.shape == (nt + 1,)
    assert np.all(np.isfinite(hist))
    assert np.all(np.isfinite(time))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_gcm_call_rk4_shapes(
    gcm_config, f_schedule, sample_x, polynomial_parameterization
):
    """Test that GCM integration returns correct shapes with RK4."""
    gcm = GCM(polynomial_parameterization, F_schedule=f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK4,
    )

    nt = int(gcm_config.total_time / gcm_config.si)

    assert hist.shape == (nt + 1, gcm_config.K)
    assert time.shape == (nt + 1,)
    assert np.all(np.isfinite(hist))
    assert np.all(np.isfinite(time))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_gcm_call_euler_shapes(
    gcm_config, f_schedule, sample_x, polynomial_parameterization
):
    """Test that GCM integration returns correct shapes with Euler."""
    gcm = GCM(polynomial_parameterization, F_schedule=f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=euler_forward,
    )

    nt = int(gcm_config.total_time / gcm_config.si)

    assert hist.shape == (nt + 1, gcm_config.K)
    assert time.shape == (nt + 1,)
    assert np.all(np.isfinite(hist))
    assert np.all(np.isfinite(time))


def test_gcm_run_initial_condition_preserved(
    gcm_config, sample_x, polynomial_parameterization
):
    """Test that initial condition is correctly set in history."""
    gcm = GCM(polynomial_parameterization, F_schedule=gcm_config.f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK2,
    )

    # First time point should be 0
    assert np.isclose(time[0], 0.0)
    # First state should match initial condition
    assert np.allclose(hist[0], sample_x)


def test_gcm_run_time_progression(gcm_config, sample_x, polynomial_parameterization):
    """Test that time progresses correctly during integration."""
    gcm = GCM(polynomial_parameterization, F_schedule=gcm_config.f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK2,
    )

    # Time should be monotonically increasing
    assert np.all(np.diff(time) > 0)
    # Should reach approximately total_time
    assert np.isclose(time[-1], gcm_config.total_time, rtol=1e-10)


def test_gcm_run_state_evolves(gcm_config, sample_x, polynomial_parameterization):
    """Test that state evolves during integration."""
    gcm = GCM(polynomial_parameterization, F_schedule=gcm_config.f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK2,
    )

    # Final state should be different from initial (with high probability)
    assert not np.allclose(hist[-1], hist[0])


def test_gcm_run_with_ar1_parameterization(
    gcm_config, sample_x, polynomial_ar1_parameterization
):
    """Test GCM integration with AR1 parameterization."""
    gcm = GCM(polynomial_ar1_parameterization, F_schedule=gcm_config.f_schedule)

    hist, time = gcm(
        sample_x,
        si=gcm_config.si,
        total_time=gcm_config.total_time,
        dt=gcm_config.dt,
        time_stepping_func=RK2,
    )

    nt = int(gcm_config.total_time / gcm_config.si)

    assert hist.shape == (nt + 1, gcm_config.K)
    assert time.shape == (nt + 1,)
    assert np.all(np.isfinite(hist))


def test_gcm_forcing_schedule_constant(polynomial_parameterization):
    gcm = GCM(polynomial_parameterization, F_schedule=ConstantForcingSchedule(20.0))

    assert np.isclose(gcm._forcing_at(0.0), 20.0)
    assert np.isclose(gcm._forcing_at(1.5), 20.0)


def test_gcm_forcing_schedule_linear(polynomial_parameterization):
    schedule = LinearForcingSchedule(F0=10.0, F1=20.0, t0=0.0, t1=2.0)
    gcm = GCM(polynomial_parameterization, F_schedule=schedule)

    assert np.isclose(gcm._forcing_at(-1.0), 10.0)
    assert np.isclose(gcm._forcing_at(0.0), 10.0)
    assert np.isclose(gcm._forcing_at(1.0), 15.0)
    assert np.isclose(gcm._forcing_at(2.0), 20.0)
    assert np.isclose(gcm._forcing_at(3.0), 20.0)


def test_gcm_forcing_schedule_oscillating(polynomial_parameterization):
    schedule = OscillatingForcingSchedule(Fmean=10.0, amp=2.0, freq=0.5)
    gcm = GCM(polynomial_parameterization, F_schedule=schedule)

    assert np.isclose(gcm._forcing_at(0.0), 10.0)
    assert np.isclose(gcm._forcing_at(1.0), 10.0)
