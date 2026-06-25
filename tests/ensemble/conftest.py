import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.forcing_schedule import ConstantForcingSchedule
from parameterization.flow.flow_model import ConditionalRealNVP

GCM_CASES = [
    pytest.param((2, 1, 1), id="ninit_gt1_nens1_nmodels1"),
    pytest.param((2, 2, 1), id="ninit_gt1_nens_gt1_nmodels1"),
    pytest.param((2, 2, 3), id="ninit_gt1_nens_gt1_nmodels_gt1"),
]


def _make_gcm_config(n_init_states: int, n_ens_members: int, n_models: int):
    return SimpleNamespace(
        K=8,
        f_schedule=ConstantForcingSchedule(20.0),
        dt=0.01,
        si=0.05,
        total_time=3,
        n_init_states=n_init_states,
        n_ens_members=n_ens_members,
        n_models=n_models,
        time_stepping="RK2",
        delta_t=0,
        dt_full=1.0,
        include_forcing_in_cond=False,
        use_flexible_tails=False,
        ttf_init_lambda=0.1,
        ar_order=1,
    )


@pytest.fixture
def l96_ensemble_config():
    """Fixture for L96 ensemble configuration."""
    return SimpleNamespace(
        K=8,
        J=32,
        f_schedule=ConstantForcingSchedule(20.0),
        h=1.0,
        b=10.0,
        c=10.0,
        dt=0.005,
        si=0.05,
        total_time=3,  # Short for testing
        n_init_states=2,
        n_ens_members=2,
        seed=42,
        time_stepping="RK2",
    )


@pytest.fixture(params=GCM_CASES)
def gcm_case_data(request):
    """Parameterized GCM case data for (N, M, L) scenarios."""
    n_init_states, n_ens_members, n_models = request.param
    cfg = _make_gcm_config(n_init_states, n_ens_members, n_models)
    rng = np.random.default_rng(0)

    init_states = rng.normal(
        size=(cfg.n_init_states, cfg.n_ens_members, cfg.n_models, cfg.K)
    )
    poly_coefs = np.array([0.341, 1.3, -0.0136, -0.00235], dtype=float)
    bayes_coefs = np.tile(poly_coefs, (cfg.n_ens_members, cfg.n_models, 1))

    if cfg.n_ens_members > 1 and cfg.n_models > 1:
        # Last case: vary seeds only by model, share across N and M.
        model_seeds = np.arange(100, 100 + cfg.n_models, dtype=int)
        seeds = np.broadcast_to(
            model_seeds[None, None, :],
            (cfg.n_init_states, cfg.n_ens_members, cfg.n_models),
        ).copy()
    else:
        n_total = cfg.n_init_states * cfg.n_ens_members * cfg.n_models
        seeds = np.arange(100, 100 + n_total, dtype=int).reshape(
            cfg.n_init_states, cfg.n_ens_members, cfg.n_models
        )

    return SimpleNamespace(
        config=cfg,
        init_states=init_states,
        seeds=seeds,
        poly_coefs=poly_coefs,
        bayesian_poly_coefs=bayes_coefs,
    )


@pytest.fixture
def init_states_x(l96_ensemble_config):
    """Sample initial states for slow variables (X)."""
    np.random.seed(0)
    return np.random.randn(
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        l96_ensemble_config.K,
    )


@pytest.fixture
def init_states_y(l96_ensemble_config):
    """Sample initial states for fast variables (Y)."""
    np.random.seed(1)
    return np.random.randn(
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        l96_ensemble_config.K * l96_ensemble_config.J,
    )


@pytest.fixture
def ar1_params():
    """AR1 parameters for stochastic parameterization."""
    np.random.seed(2)
    phi = float(np.random.uniform(0.2, 0.8))
    sigma_e = float(np.random.uniform(0.01, 0.1))
    return phi, sigma_e


@pytest.fixture
def flow_model_small():
    """Small deterministic flow model for ensemble tests."""
    torch.manual_seed(0)
    return ConditionalRealNVP(
        dim=8,
        cond_dim=8,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )


@pytest.fixture
def gcm_ensemble_config():
    """Default GCM configuration used in storage tests."""
    return _make_gcm_config(n_init_states=2, n_ens_members=2, n_models=1)


@pytest.fixture
def sample_gcm_ensemble_data(gcm_ensemble_config):
    """Sample GCM ensemble simulation results."""
    np.random.seed(42)
    nt = int(gcm_ensemble_config.total_time / gcm_ensemble_config.si) + 1
    x = np.random.randn(
        gcm_ensemble_config.n_init_states,
        gcm_ensemble_config.n_ens_members,
        gcm_ensemble_config.n_models,
        nt,
        gcm_ensemble_config.K,
    )
    t = np.linspace(0, gcm_ensemble_config.total_time, nt)
    return x, t


@pytest.fixture
def sample_l96_ensemble_data(l96_ensemble_config):
    """Sample L96 ensemble simulation results."""
    np.random.seed(42)
    nt = int(l96_ensemble_config.total_time / l96_ensemble_config.si) + 1
    x = np.random.randn(
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        nt,
        l96_ensemble_config.K,
    )
    y = np.random.randn(
        l96_ensemble_config.n_init_states,
        l96_ensemble_config.n_ens_members,
        nt,
        l96_ensemble_config.K * l96_ensemble_config.J,
    )
    t = np.linspace(0, l96_ensemble_config.total_time, nt)
    return x, y, t


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
