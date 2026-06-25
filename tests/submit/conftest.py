from pathlib import Path

import numpy as np
import pytest
import torch

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.storage import save_checkpoint
from utils.config import (
    AR_P_PARAMS_DIR_NAME,
    COEFS_DIR_NAME,
    L96_SINGLE_OUTPUT_SUBDIR,
)
from utils.sweep_utils import get_sweep_name

BASE_DIR = Path(__file__).parent


@pytest.fixture
def configs_dir():
    """Return the directory containing test configs."""
    return BASE_DIR / "configs"


@pytest.fixture
def output_root(tmp_path):
    """Provide a temporary root for results to avoid polluting the repo."""
    return tmp_path / "results"


def _write_initial_states(
    base_dir: Path,
    n_states: int,
    k: int,
    j: int,
    seed: int,
    n_ens_members: int | None = None,
    sweep_names: list[str] | None = None,
):
    rng = np.random.default_rng(seed)
    if n_ens_members is None:
        x = rng.normal(size=(n_states, k))
        y = rng.normal(size=(n_states, k * j))
    else:
        x = rng.normal(size=(n_states, n_ens_members, k))
        y = rng.normal(size=(n_states, n_ens_members, k * j))
    t = np.arange(n_states, dtype=float)

    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        init_dir = target / "initial_states"
        init_dir.mkdir(parents=True, exist_ok=True)
        np.save(init_dir / "x.npy", x)
        np.save(init_dir / "y.npy", y)
        np.save(init_dir / "t.npy", t)


def _write_l96_train(
    base_dir: Path,
    steps: int,
    k: int,
    j: int,
    seed: int,
    sweep_names: list[str] | None = None,
):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(steps, k))
    y = rng.normal(size=(steps, k * j))
    t = np.arange(steps, dtype=float)

    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        l96_dir = target / L96_SINGLE_OUTPUT_SUBDIR
        l96_dir.mkdir(parents=True, exist_ok=True)
        np.save(l96_dir / "x.npy", x)
        np.save(l96_dir / "y.npy", y)
        np.save(l96_dir / "t.npy", t)


def _write_params_det(base_dir: Path, sweep_names: list[str] | None = None):
    # Use true coefficients from Arnold et al. 2013 to avoid numerical errors
    coefs = np.array([0.341, 1.3, -0.0136, -0.00235])
    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        coefs_dir = target / COEFS_DIR_NAME
        coefs_dir.mkdir(parents=True, exist_ok=True)
        np.save(coefs_dir / "coefs.npy", coefs)


def _write_params_ar_p(
    base_dir: Path, sweep_names: list[str] | None = None, ar_orders=[1]
):
    # Use true coefficients from Arnold et al. 2013 to avoid numerical errors
    coefs = np.array([0.341, 1.3, -0.0136, -0.00235])
    rho = 0.25
    sigma = 0.1
    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        coefs_dir = target / COEFS_DIR_NAME
        ar_dir = target / AR_P_PARAMS_DIR_NAME
        coefs_dir.mkdir(parents=True, exist_ok=True)
        ar_dir.mkdir(parents=True, exist_ok=True)
        np.save(coefs_dir / "coefs.npy", coefs)
        for ar_order in ar_orders:
            np.save(ar_dir / f"rho_{ar_order}.npy", rho)
            np.save(ar_dir / f"sigma_{ar_order}.npy", sigma)


def _write_params_bayes(
    base_dir: Path,
    n_ens_members: int,
    n_models: int = 1,
    sweep_names: list[str] | None = None,
):
    # Use true coefficients from Arnold et al. 2013 to avoid numerical errors
    coefs = np.array([0.341, 1.3, -0.0136, -0.00235])
    coefs_ens = np.tile(coefs, (n_ens_members, n_models, 1))
    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        coefs_dir = target / COEFS_DIR_NAME
        coefs_dir.mkdir(parents=True, exist_ok=True)
        np.save(coefs_dir / "bayesian_coefs.npy", coefs_ens)


def _write_params_flow(
    base_dir: Path, sweep_names: list[str] | None = None, ar_order=1
):
    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=8,
        cond_dim=8,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )
    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        model_dir = target / "flow_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_dir, model=flow_model, cfg=flow_model.get_config())
        ar_dir = target / AR_P_PARAMS_DIR_NAME
        ar_dir.mkdir(parents=True, exist_ok=True)
        np.save(ar_dir / f"rho_{ar_order}.npy", np.array([0.3]))
        np.save(ar_dir / f"sigma_{ar_order}.npy", np.array([0.95]))
        coefs_dir = target / "coefs"
        coefs_dir.mkdir(parents=True, exist_ok=True)
        np.save(coefs_dir / "coefs.npy", np.array([0.341, 1.3, -0.0136, -0.00235]))


def _write_params_flow_arp_base(
    base_dir: Path, sweep_names: list[str] | None = None, ar_order=2
):
    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=8,
        cond_dim=8,
        n_coupling_layers=2,
        hidden_dims=(8,),
        base_dist=ARpBase(dim=8, p=ar_order, init_rho=[0.2, -0.05], init_sigma=0.9),
    )
    cfg = flow_model.get_config()
    cfg.update(
        {
            "base_dist_name": "ar_p",
            "ar_order": ar_order,
            "init_rho": [0.2, -0.05],
            "init_sigma": 0.9,
        }
    )
    targets = [base_dir] + [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        model_dir = target / "flow_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_dir, model=flow_model, cfg=cfg)
        ar_dir = target / AR_P_PARAMS_DIR_NAME
        ar_dir.mkdir(parents=True, exist_ok=True)
        np.save(ar_dir / f"rho_{ar_order}.npy", np.array([0.2, -0.05]))
        np.save(ar_dir / f"sigma_{ar_order}.npy", np.array([0.9]))
        coefs_dir = target / "coefs"
        coefs_dir.mkdir(parents=True, exist_ok=True)
        np.save(coefs_dir / "coefs.npy", np.array([0.341, 1.3, -0.0136, -0.00235]))


def _write_params_flow_tail_history_forcing(
    base_dir: Path, ttf_init_lambda: float, sweep_names: list[str] | None = None
):
    # Build condition dim
    delta_t = 2
    k_dim = 8
    cond_dim = k_dim * (delta_t + 1) + 1

    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=k_dim,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
        use_flexible_tails=True,
        ttf_init_lambda=ttf_init_lambda,
    )
    targets = [base_dir / name for name in (sweep_names or [])]
    for target in targets:
        model_dir = target / "flow_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_dir, model=flow_model, cfg=flow_model.get_config())
        ar_dir = target / AR_P_PARAMS_DIR_NAME
        ar_dir.mkdir(parents=True, exist_ok=True)
        if "ar_order_1" in target.name:
            np.save(ar_dir / "rho_1.npy", np.array([0.3]))
            np.save(ar_dir / "sigma_1.npy", np.array([0.95]))
        elif "ar_order_3" in target.name:
            np.save(ar_dir / "rho_3.npy", np.array([0.3, 0.1, 0.05]))
            np.save(ar_dir / "sigma_3.npy", np.array([0.95]))


@pytest.fixture
def submit_input_data(tmp_path):
    """Generate minimal input data for submit_local_run smoke tests."""
    data_root = tmp_path / "submit_input_data"
    data_root.mkdir(parents=True, exist_ok=True)

    l96_init = data_root / "l96_init"
    l96_init_perturbed = data_root / "l96_init_perturbed"
    l96_init_perturbed_cond = data_root / "l96_init_perturbed_cond"
    gcm_init = data_root / "gcm_init"
    gcm_init_perturbed = data_root / "gcm_init_perturbed"
    l96_train = data_root / "l96_train"
    params_det = data_root / "params_det"
    params_ar_p = data_root / "params_ar_p"
    params_bayes = data_root / "params_bayes"
    params_flow = data_root / "params_flow"
    params_flow_arp_base = data_root / "params_flow_arp_base"
    params_flow_tail_history_forcing = data_root / "params_flow_tail_history_forcing"

    # ---------------------------- Initial states L96 ----------------------------
    l96_sweeps = [
        get_sweep_name({"c": c_val, "F": f_val})
        for c_val in [4.0, 10.0]
        for f_val in [18.0, 20.0]
    ]
    # Initial states
    _write_initial_states(
        l96_init, n_states=2, k=8, j=32, seed=0, sweep_names=l96_sweeps
    )

    # Initial states for perturbed state L96 ensemble runs (n_ens_members=2)
    _write_initial_states(
        l96_init_perturbed,
        n_states=2,
        k=8,
        j=32,
        seed=0,
        n_ens_members=2,
        sweep_names=l96_sweeps,
    )

    # L96 sweeps for forcing schedules
    l96_fs_sweeps = [
        get_sweep_name({"f_schedule": fs})
        for fs in [
            {"type": "linear", "F0": 18, "F1": 23, "t0": 0, "t1": 5},
            {"type": "oscillating", "Fmean": 20, "amp": 2, "freq": 5},
        ]
    ]
    _write_initial_states(
        l96_init_perturbed,
        n_states=2,
        k=8,
        j=32,
        seed=0,
        n_ens_members=2,
        sweep_names=l96_fs_sweeps,
    )
    _write_initial_states(
        l96_init_perturbed_cond,
        n_states=2,
        k=8,
        j=32,
        seed=0,
        n_ens_members=2,
        sweep_names=l96_fs_sweeps,
    )

    # ---------------------------- Training data -----------------------------
    # Unswept training data
    _write_l96_train(l96_train, steps=100, k=8, j=32, seed=2)
    # Swept training data
    flow_training_sweeps = [get_sweep_name({"c": val}) for val in [4.0, 10.0]]
    _write_l96_train(
        l96_train, steps=100, k=8, j=32, seed=2, sweep_names=flow_training_sweeps
    )

    det_sweeps = [get_sweep_name({"F": val}) for val in [19.0, 20.0, 21.0]]
    bayes_sweeps = [
        get_sweep_name({"c": c_val, "F": f_val})
        for c_val in [8.0, 10.0]
        for f_val in [19.0, 20.0]
    ]
    flow_gcm_sweeps = [get_sweep_name({"F": val}) for val in [19.0, 20.0]]
    flow_arp_base_sweeps = [get_sweep_name({"F": val}) for val in [19.0, 20.0]]
    flow_history_forcing_sweeps = [
        get_sweep_name(d)
        for d in [
            {"noise_type": "white", "ar_order": 0},
            {"noise_type": "ar_p", "ar_order": 1},
            {"noise_type": "ar_p", "ar_order": 3},
        ]
    ]
    _write_params_det(params_det, det_sweeps)
    _write_params_ar_p(params_ar_p, ar_orders=[1, 2])
    _write_params_bayes(
        params_bayes, n_ens_members=10, n_models=2, sweep_names=bayes_sweeps
    )
    _write_params_flow(params_flow, flow_gcm_sweeps)
    _write_params_flow_arp_base(params_flow_arp_base, flow_arp_base_sweeps)
    _write_params_flow_tail_history_forcing(
        params_flow_tail_history_forcing,
        ttf_init_lambda=0.1,
        sweep_names=flow_history_forcing_sweeps,
    )

    # ---------------------------- Initial states GCMs ----------------------------
    # GCMS that expected perturbed initial states with n_ens_members=2
    gcm_perturbed_sweeps = []
    gcm_perturbed_sweeps.extend(det_sweeps)
    gcm_perturbed_sweeps.extend(flow_gcm_sweeps)
    gcm_perturbed_sweeps.extend(flow_arp_base_sweeps)
    _write_initial_states(
        gcm_init_perturbed,
        n_states=2,
        k=8,
        j=32,
        seed=0,
        sweep_names=gcm_perturbed_sweeps,
        n_ens_members=2,
    )

    # Other GCMs all use perfect intiial states
    gcm_perfect_sweeps = []
    gcm_perfect_sweeps.extend(bayes_sweeps)
    gcm_perfect_sweeps = list(dict.fromkeys(gcm_perfect_sweeps))
    _write_initial_states(
        gcm_init, n_states=10, k=8, j=32, seed=0, sweep_names=gcm_perfect_sweeps
    )

    # Perturbed initial states for l96 sensitivity studies (perturb stds need to be the same as in config files)
    for std in [0.1, 0.2]:
        _write_initial_states(
            l96_init_perturbed / f"perturb_std_{std}",
            n_states=2,
            k=4,
            j=2,
            seed=0,
            n_ens_members=1,
        )

    return {
        "l96_init": l96_init,
        "l96_init_perturbed": l96_init_perturbed,
        "l96_init_perturbed_cond": l96_init_perturbed_cond,
        "gcm_init": gcm_init,
        "gcm_init_perturbed": gcm_init_perturbed,
        "l96_train": l96_train,
        "params_det": params_det,
        "params_ar_p": params_ar_p,
        "params_bayes": params_bayes,
        "params_flow": params_flow,
        "params_flow_arp_base": params_flow_arp_base,
        "params_flow_tail_history_forcing": params_flow_tail_history_forcing,
    }
