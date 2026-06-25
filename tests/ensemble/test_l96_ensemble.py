import numpy as np
import pytest

from ensemble.l96_ensemble import run_l96_parallel, run_single_l96
from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)


def test_run_single_l96_shapes(l96_ensemble_config):
    """Test that run_single_l96 returns correct shapes."""
    np.random.seed(42)
    x_init = np.random.randn(l96_ensemble_config.K)
    y_init = np.random.randn(l96_ensemble_config.K * l96_ensemble_config.J)

    args = (
        x_init,
        y_init,
        l96_ensemble_config.K,
        l96_ensemble_config.J,
        l96_ensemble_config.f_schedule,
        l96_ensemble_config.h,
        l96_ensemble_config.b,
        l96_ensemble_config.c,
        l96_ensemble_config.si,
        l96_ensemble_config.total_time,
        l96_ensemble_config.dt,
    )

    x, y, t = run_single_l96(args)

    nt = int(l96_ensemble_config.total_time / l96_ensemble_config.si) + 1
    assert x.shape == (nt, l96_ensemble_config.K)
    assert y.shape == (nt, l96_ensemble_config.K * l96_ensemble_config.J)
    assert t.shape == (nt,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(t))


@pytest.mark.parametrize(
    "schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=2.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_run_l96_parallel_shapes_f_schedule(
    init_states_x, init_states_y, l96_ensemble_config, schedule
):
    """Test that run_l96_parallel returns correct shapes."""
    l96_ensemble_config.f_schedule = schedule
    x_per_state, y_per_state, t_per_state = run_l96_parallel(
        init_states_x,
        init_states_y,
        l96_ensemble_config,
        num_processes=1,  # Use single process for testing
    )

    nt = int(l96_ensemble_config.total_time / l96_ensemble_config.si) + 1
    assert x_per_state.shape == (
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        nt,
        l96_ensemble_config.K,
    )
    assert y_per_state.shape == (
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        nt,
        l96_ensemble_config.K * l96_ensemble_config.J,
    )
    assert t_per_state.shape == (nt,)

    assert np.all(np.isfinite(x_per_state))
    assert np.all(np.isfinite(y_per_state))
    assert np.all(np.isfinite(t_per_state))


def test_run_l96_parallel_time_consistency(
    init_states_x, init_states_y, l96_ensemble_config
):
    """Test that time arrays are consistent across states."""
    _, _, t_per_state = run_l96_parallel(
        init_states_x,
        init_states_y,
        l96_ensemble_config,
        num_processes=1,
    )

    nt = int(l96_ensemble_config.total_time / l96_ensemble_config.si) + 1
    assert t_per_state.shape == (nt,)
    assert np.all(np.isfinite(t_per_state))
