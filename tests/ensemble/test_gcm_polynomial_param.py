import numpy as np
import pytest

from ensemble.gcm_polynomial_param import (
    run_poly_param_ensemble_parallel_multiprocessing,
    run_poly_param_member,
)
from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)


def test_run_poly_param_member_shapes(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    poly_coefs = gcm_case_data.poly_coefs

    args = (
        0,  # i_init
        0,  # i_member
        0,  # i_model
        init_states[0, 0, 0],
        config.f_schedule,
        config.si,
        config.total_time,
        poly_coefs,
        config.dt,
        config.time_stepping,
    )

    i_init, i_member, i_model, x_pred, t = run_poly_param_member(args)

    nt = int(config.total_time / config.si) + 1
    assert i_init == 0
    assert i_member == 0
    assert i_model == 0
    assert x_pred.shape == (nt, config.K)
    assert t.shape == (nt,)
    assert np.all(np.isfinite(x_pred))
    assert np.all(np.isfinite(t))


@pytest.mark.parametrize(
    "schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=2.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_run_poly_param_ensemble_deterministic_forcing_schedule(
    gcm_case_data, schedule
):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    poly_coefs = gcm_case_data.poly_coefs
    config.f_schedule = schedule

    x_ens_forecast, t_ens_forecast = run_poly_param_ensemble_parallel_multiprocessing(
        init_states,
        config,
        poly_coefs,
        num_processes=1,
    )

    nt = int(config.total_time / config.si) + 1
    assert x_ens_forecast.shape == (
        config.n_init_states,
        config.n_ens_members,
        config.n_models,
        nt,
        config.K,
    )
    assert t_ens_forecast.shape == (nt,)
    assert np.all(np.isfinite(x_ens_forecast))
    assert np.all(np.isfinite(t_ens_forecast))


def test_run_poly_param_ensemble_bayesian(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    bayesian_poly_coefs = gcm_case_data.bayesian_poly_coefs

    x_ens_forecast, t_ens_forecast = run_poly_param_ensemble_parallel_multiprocessing(
        init_states,
        config,
        bayesian_poly_coefs,
        num_processes=1,
    )

    nt = int(config.total_time / config.si) + 1
    assert x_ens_forecast.shape == (
        config.n_init_states,
        config.n_ens_members,
        config.n_models,
        nt,
        config.K,
    )
    assert t_ens_forecast.shape == (nt,)
    assert np.all(np.isfinite(x_ens_forecast))
    assert np.all(np.isfinite(t_ens_forecast))


def test_run_poly_param_ensemble_time_consistency(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    poly_coefs = gcm_case_data.poly_coefs

    _, t_ens_forecast = run_poly_param_ensemble_parallel_multiprocessing(
        init_states,
        config,
        poly_coefs,
        num_processes=1,
    )

    nt = int(config.total_time / config.si) + 1
    assert t_ens_forecast.shape == (nt,)
