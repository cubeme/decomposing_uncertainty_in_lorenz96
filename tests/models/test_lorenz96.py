import numpy as np
import pytest

from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)
from models.L96.lorenz96 import L96


def test_l96_initialization(l96_config):
    """Test L96 model initialization."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=l96_config.f_schedule,
        seed=l96_config.seed,
    )

    assert l96.k == l96_config.K
    assert l96.j == l96_config.J
    assert l96.f_schedule is l96_config.f_schedule
    assert l96.h == l96_config.h
    assert l96.b == l96_config.b
    assert l96.c == l96_config.c
    assert l96.x.shape == (l96_config.K,)
    assert l96.y.shape == (l96_config.K * l96_config.J,)

    assert np.all(np.isfinite(l96.x))
    assert np.all(np.isfinite(l96.y))


@pytest.mark.parametrize(
    "schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_l96_run_shapes_with_schedule(l96_config, schedule):
    """Test that L96.run returns correct shapes across forcing schedules."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=schedule,
        seed=l96_config.seed,
    )

    x_hist, y_hist, time = l96.run(
        si=l96_config.si,
        total_time=l96_config.total_time,
        dt=l96_config.dt,
    )

    nt = int(l96_config.total_time / l96_config.si)

    assert x_hist.shape == (nt + 1, l96_config.K)
    assert y_hist.shape == (nt + 1, l96_config.K * l96_config.J)
    assert time.shape == (nt + 1,)

    assert np.all(np.isfinite(x_hist))
    assert np.all(np.isfinite(y_hist))
    assert np.all(np.isfinite(time))


def test_l96_run_time_progression(l96_config):
    """Test that time array progresses correctly."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=l96_config.f_schedule,
    )

    _, _, time = l96.run(
        si=l96_config.si,
        total_time=l96_config.total_time,
        dt=l96_config.dt,
    )

    # Time should be monotonically increasing
    assert np.all(np.diff(time) > 0)
    # Should reach approximately total_time
    assert np.isclose(time[-1], l96_config.total_time, rtol=1e-10)


def test_l96_run_store_updates_state(l96_config):
    """Test that store=True updates internal state."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=l96_config.f_schedule,
    )

    x_initial = l96.x.copy()

    x_hist, y_hist, time = l96.run(
        si=l96_config.si,
        total_time=l96_config.total_time,
        dt=l96_config.dt,
        store=True,
    )

    # State should be updated to final state
    assert np.allclose(x_initial, x_hist[0])
    assert np.allclose(l96.x, x_hist[-1])
    assert np.allclose(l96.y, y_hist[-1])
    assert l96.t == time[-1]


def test_l96_run_sequential_continuity(l96_config):
    """Test that sequential runs continue from previous state."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=l96_config.f_schedule,
    )

    # First run with store=True
    x_hist1, y_hist1, time1 = l96.run(
        si=l96_config.si,
        total_time=l96_config.total_time,
        dt=l96_config.dt,
        store=True,
    )

    # Second run should start from where first run ended
    x_hist2, y_hist2, time2 = l96.run(
        si=l96_config.si,
        total_time=l96_config.total_time,
        dt=l96_config.dt,
        store=True,
    )

    # First point of second run should equal last point of first run
    assert np.allclose(x_hist2[0], x_hist1[-1])
    assert np.allclose(y_hist2[0], y_hist1[-1])
    # Time should continue
    assert time2[0] == time1[-1]


def test_l96_repr_str(l96_config):
    """Test string representations."""
    l96 = L96(
        l96_config.K,
        l96_config.J,
        l96_config.h,
        l96_config.b,
        l96_config.c,
        F_schedule=l96_config.f_schedule,
    )

    repr_str = repr(l96)
    str_str = str(l96)

    assert "L96" in repr_str
    assert str(l96_config.K) in repr_str
    assert "X=" in str_str
    assert "Y=" in str_str
