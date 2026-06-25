import numpy as np
import pytest

from ensemble.gcm_flow_param_batched import (
    run_flow_param_ensemble_batched_single_gpu,
)
from ensemble.gcm_flow_param_sequential import (
    run_flow_param_ensemble_sequential_single_gpu,
)
from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)
from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import ConditionalRealNVP


def _assert_ensemble_shapes(config, x_ens, t_ens):
    nt = int(config.total_time / config.si) + 1
    assert x_ens.shape == (
        config.n_init_states,
        config.n_ens_members,
        config.n_models,
        nt,
        config.K,
    )
    assert t_ens.shape == (nt,)
    assert np.all(np.isfinite(x_ens))
    assert np.all(np.isfinite(t_ens))


def _assert_seed_variation_when_member_seeds_differ(config, seeds, x_ens):
    member_seeds = seeds[0, :, 0]
    if config.n_ens_members > 1 and np.unique(member_seeds).size > 1:
        diff = np.abs(x_ens[0, 0, 0] - x_ens[0, 1, 0])
        assert np.any(diff > 1e-10)


def _rho_and_sigma(ar_order):
    if ar_order <= 1:
        return 0.2, 0.1
    return np.array([0.3, -0.1, 0.05], dtype=np.float32), 0.4


@pytest.mark.parametrize("ar_order", [0, 1, 3])
def test_run_flow_param_ensemble_sequential(gcm_case_data, flow_model_small, ar_order):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    seeds = gcm_case_data.seeds
    config.ar_order = ar_order
    rho, sigma = _rho_and_sigma(ar_order)

    x_ens, t_ens = run_flow_param_ensemble_sequential_single_gpu(
        init_states,
        config,
        flow_model_small,
        seeds,
        rho=rho,
        sigma=sigma,
        device="cpu",
    )

    _assert_ensemble_shapes(config, x_ens, t_ens)
    _assert_seed_variation_when_member_seeds_differ(config, seeds, x_ens)


@pytest.mark.parametrize(
    ("schedule", "ar_order"),
    [
        (ConstantForcingSchedule(20.0), 0),
        (LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=2.0), 1),
        (OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5), 3),
    ],
)
def test_run_flow_param_ensemble_batched(
    gcm_case_data,
    flow_model_small,
    schedule,
    ar_order,
):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    seeds = gcm_case_data.seeds
    config.f_schedule = schedule
    config.ar_order = ar_order
    rho, sigma = _rho_and_sigma(ar_order)

    x_ens, t_ens = run_flow_param_ensemble_batched_single_gpu(
        init_states,
        config,
        flow_model_small,
        seeds,
        rho=rho,
        sigma=sigma,
        device="cpu",
        time_stepping=config.time_stepping,
    )

    _assert_ensemble_shapes(config, x_ens, t_ens)
    _assert_seed_variation_when_member_seeds_differ(config, seeds, x_ens)


def test_run_flow_param_ensemble_batched_with_tail_forcing(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    seeds = gcm_case_data.seeds
    config.include_forcing_in_cond = True
    config.delta_t = 0
    config.use_flexible_tails = True
    config.ttf_init_lambda = 0.1

    cond_dim = config.K + 1
    flow_model = ConditionalRealNVP(
        dim=config.K,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
        use_flexible_tails=True,
    )

    x_ens, t_ens = run_flow_param_ensemble_batched_single_gpu(
        init_states,
        config,
        flow_model,
        seeds,
        rho=0.2,
        sigma=0.1,
        device="cpu",
        time_stepping=config.time_stepping,
    )

    _assert_ensemble_shapes(config, x_ens, t_ens)


def test_run_flow_param_ensemble_batched_with_arp_base(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    seeds = gcm_case_data.seeds
    config.ar_order = 2

    flow_model = ConditionalRealNVP(
        dim=config.K,
        cond_dim=config.K,
        n_coupling_layers=2,
        hidden_dims=(8,),
        base_dist=ARpBase(dim=config.K, p=2, init_rho=[0.2, -0.05], init_sigma=0.8),
    )

    x_ens, t_ens = run_flow_param_ensemble_batched_single_gpu(
        init_states,
        config,
        flow_model,
        seeds,
        rho=np.array([0.2, -0.05], dtype=np.float32),
        sigma=0.8,
        device="cpu",
        time_stepping=config.time_stepping,
    )

    _assert_ensemble_shapes(config, x_ens, t_ens)


def test_run_flow_param_ensemble_sequential_with_history(gcm_case_data):
    config = gcm_case_data.config
    init_states = gcm_case_data.init_states
    seeds = gcm_case_data.seeds
    config.delta_t = 1
    config.include_forcing_in_cond = False

    cond_dim = config.K * (config.delta_t + 1)
    flow_model = ConditionalRealNVP(
        dim=config.K,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x_ens, t_ens = run_flow_param_ensemble_sequential_single_gpu(
        init_states,
        config,
        flow_model,
        seeds,
        rho=0.2,
        sigma=0.1,
        device="cpu",
    )

    _assert_ensemble_shapes(config, x_ens, t_ens)
