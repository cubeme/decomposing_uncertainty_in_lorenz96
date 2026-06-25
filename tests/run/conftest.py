from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.storage import save_checkpoint
from utils.config import (
    AR_P_PARAMS_DIR_NAME,
    COEFS_DIR_NAME,
    L96_SINGLE_OUTPUT_SUBDIR,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"


def _load_kj(config_path: Path) -> tuple[int, int]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return int(data["K"]), int(data["J"])


def _load_init_state_requirements(config_dir: Path) -> dict[str, dict[str, int]]:
    requirements: dict[str, dict[str, int]] = {}
    for config_path in config_dir.glob("*.yaml"):
        if not config_path.name.startswith(("gcm_", "l96_")):
            continue
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        init_states_dir = data.get("init_states_dir")
        if not init_states_dir:
            continue

        name = Path(init_states_dir).name
        n_init_states = data.get("n_init_states")
        n_ens_members = data.get("n_ens_members")
        n_models = data.get("n_models")

        current = requirements.setdefault(
            name, {"n_init_states": 0, "n_ens_members": 1, "n_models": 1}
        )
        if n_init_states is not None:
            current["n_init_states"] = max(current["n_init_states"], int(n_init_states))
        if n_ens_members is not None:
            current["n_ens_members"] = max(current["n_ens_members"], int(n_ens_members))
        if n_models is not None:
            current["n_models"] = max(current["n_models"], int(n_models))
    return requirements


GCM_K, GCM_J = _load_kj(CONFIGS_DIR / "gcm_flow.yaml")
L96_K, L96_J = _load_kj(CONFIGS_DIR / "l96_ensemble.yaml")
L96_TRAIN_K, L96_TRAIN_J = _load_kj(CONFIGS_DIR / "parameter_fitting_baseline.yaml")
GCM_KJ = GCM_K * GCM_J
L96_KJ = L96_K * L96_J
L96_TRAIN_KJ = L96_TRAIN_K * L96_TRAIN_J


INIT_STATE_REQUIREMENTS = _load_init_state_requirements(CONFIGS_DIR)
GCM_N_INIT = INIT_STATE_REQUIREMENTS["gcm_init"]["n_init_states"]
GCM_N_INIT_PERT = INIT_STATE_REQUIREMENTS["gcm_init_perturbed"]["n_init_states"]
GCM_N_ENS_PERT = INIT_STATE_REQUIREMENTS["gcm_init_perturbed"]["n_ens_members"]
GCM_N_MODELS = max(
    INIT_STATE_REQUIREMENTS["gcm_init"]["n_models"],
    INIT_STATE_REQUIREMENTS["gcm_init_perturbed"]["n_models"],
)
L96_N_INIT = INIT_STATE_REQUIREMENTS["l96_init"]["n_init_states"]
L96_N_INIT_PERT = INIT_STATE_REQUIREMENTS["l96_init_perturbed"]["n_init_states"]
L96_N_ENS_PERT = INIT_STATE_REQUIREMENTS["l96_init_perturbed"]["n_ens_members"]


@pytest.fixture(autouse=True)
def _ensure_test_data():
    """Create tiny data assets on disk for the run scripts to consume."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # GCM initial states
    gcm_init = DATA_DIR / "gcm_init" / "initial_states"
    gcm_init.mkdir(parents=True, exist_ok=True)
    np.save(gcm_init / "x.npy", np.random.randn(GCM_N_INIT, GCM_K))
    np.save(gcm_init / "y.npy", np.random.randn(GCM_N_INIT, GCM_KJ))
    np.save(gcm_init / "t.npy", np.linspace(0.0, 0.1, GCM_N_INIT))

    # GCM perturbed initial states (n_states, n_ens_members, ...)
    gcm_init_perturbed = DATA_DIR / "gcm_init_perturbed" / "initial_states"
    gcm_init_perturbed.mkdir(parents=True, exist_ok=True)
    np.save(
        gcm_init_perturbed / "x.npy",
        np.random.randn(GCM_N_INIT_PERT, GCM_N_ENS_PERT, GCM_K),
    )
    np.save(
        gcm_init_perturbed / "y.npy",
        np.random.randn(GCM_N_INIT_PERT, GCM_N_ENS_PERT, GCM_KJ),
    )
    np.save(gcm_init_perturbed / "t.npy", np.linspace(0.0, 0.1, GCM_N_INIT_PERT))

    # L96 initial states
    l96_init = DATA_DIR / "l96_init" / "initial_states"
    l96_init.mkdir(parents=True, exist_ok=True)
    np.save(l96_init / "x.npy", np.random.randn(L96_N_INIT, L96_K))
    np.save(l96_init / "y.npy", np.random.randn(L96_N_INIT, L96_KJ))
    np.save(l96_init / "t.npy", np.linspace(0.0, 0.1, L96_N_INIT))

    # L96 perturbed initial states (n_states, n_ens_members, ...)
    l96_init_perturbed = DATA_DIR / "l96_init_perturbed" / "initial_states"
    l96_init_perturbed.mkdir(parents=True, exist_ok=True)
    np.save(
        l96_init_perturbed / "x.npy",
        np.random.randn(L96_N_INIT_PERT, L96_N_ENS_PERT, L96_K),
    )
    np.save(
        l96_init_perturbed / "y.npy",
        np.random.randn(L96_N_INIT_PERT, L96_N_ENS_PERT, L96_KJ),
    )
    np.save(l96_init_perturbed / "t.npy", np.linspace(0.0, 0.1, L96_N_INIT_PERT))

    # Parameters (deterministic)
    params_det = DATA_DIR / "params_det" / COEFS_DIR_NAME
    params_det.mkdir(parents=True, exist_ok=True)
    np.save(params_det / "coefs.npy", np.array([0.1, 0.2, 0.3]))

    # Parameters (AR1)

    params_ar = DATA_DIR / "params_ar" / COEFS_DIR_NAME
    params_ar.mkdir(parents=True, exist_ok=True)
    np.save(params_ar / "coefs.npy", np.array([0.1, 0.2, 0.3]))

    ar_params_dir = DATA_DIR / "params_ar" / AR_P_PARAMS_DIR_NAME
    ar_params_dir.mkdir(parents=True, exist_ok=True)
    for ar_order in [1, 2]:
        rho = np.array([0.45, -0.15], dtype=np.float32) if ar_order == 2 else 0.2
        sigma = 0.08 if ar_order == 2 else 0.06
        np.save(ar_params_dir / f"rho_{ar_order}.npy", rho)
        np.save(ar_params_dir / f"sigma_{ar_order}.npy", sigma)

    # Parameters (Bayesian)
    params_bayes = DATA_DIR / "params_bayes" / COEFS_DIR_NAME
    params_bayes.mkdir(parents=True, exist_ok=True)
    np.save(
        params_bayes / "bayesian_coefs.npy",
        np.ones((GCM_N_ENS_PERT, GCM_N_MODELS, GCM_K)),
    )
    params_bayes = DATA_DIR / "params_bayes_uncertainty" / COEFS_DIR_NAME
    params_bayes.mkdir(parents=True, exist_ok=True)
    np.save(
        params_bayes / "bayesian_coefs.npy",
        np.ones((1, GCM_N_MODELS, GCM_K)),
    )

    # Parameters (Flow)
    # Normal flow model
    params_flow = DATA_DIR / "params_flow"
    params_flow.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=GCM_K,
        cond_dim=GCM_K,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )
    params_flow_model_dir = params_flow / "flow_model"
    params_flow_model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        params_flow_model_dir, model=flow_model, cfg=flow_model.get_config()
    )

    # Flow with tail transform
    params_flow_tail = DATA_DIR / "params_flow_tail"
    params_flow_tail.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=GCM_K,
        cond_dim=GCM_K,
        n_coupling_layers=2,
        hidden_dims=(8,),
        use_flexible_tails=True,
        ttf_init_lambda=0.1,
    )
    params_flow_model_dir = params_flow_tail / "flow_model"
    params_flow_model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        params_flow_model_dir, model=flow_model, cfg=flow_model.get_config()
    )

    # History, forcing flow model
    params_flow_history_forcing = DATA_DIR / "params_flow_history_forcing"
    params_flow_history_forcing.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    delta_t = 2

    cond_dim = GCM_K * (delta_t + 1) + 1
    flow_model = ConditionalRealNVP(
        dim=GCM_K,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )
    params_flow_model_dir = params_flow_history_forcing / "flow_model"
    params_flow_model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        params_flow_model_dir, model=flow_model, cfg=flow_model.get_config()
    )

    # AR(p) values per flow config
    ar_params = {
        "params_flow": (np.array([0.3, -0.1, 0.05], dtype=np.float32), 0.98, 3),
        "params_flow_tail": (0.2, 0.98, 1),
        "params_flow_history_forcing": (
            np.array([0.25, -0.05], dtype=np.float32),
            0.98,
            2,
        ),
    }
    for d in [params_flow, params_flow_tail, params_flow_history_forcing]:
        params_flow_ar_dir = d / AR_P_PARAMS_DIR_NAME
        params_flow_ar_dir.mkdir(parents=True, exist_ok=True)
        rho, sigma, ar_order = ar_params[d.name]
        np.save(params_flow_ar_dir / f"rho_{ar_order}.npy", rho)
        np.save(params_flow_ar_dir / f"sigma_{ar_order}.npy", sigma)

    # Flow trained with AR(p) base
    params_flow_arp_base = DATA_DIR / "params_flow_arp_base"
    params_flow_arp_base.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    flow_model = ConditionalRealNVP(
        dim=GCM_K,
        cond_dim=GCM_K,
        n_coupling_layers=2,
        hidden_dims=(8,),
        base_dist=ARpBase(dim=GCM_K, p=2, init_rho=[0.2, -0.05], init_sigma=0.9),
    )
    cfg = flow_model.get_config()
    cfg.update(
        {
            "base_dist_name": "ar_p",
            "ar_order": 2,
            "init_rho": [0.2, -0.05],
            "init_sigma": 0.9,
        }
    )
    params_flow_arp_model_dir = params_flow_arp_base / "flow_model"
    params_flow_arp_model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(params_flow_arp_model_dir, model=flow_model, cfg=cfg)
    params_flow_arp_ar_dir = params_flow_arp_base / AR_P_PARAMS_DIR_NAME
    params_flow_arp_ar_dir.mkdir(parents=True, exist_ok=True)
    np.save(params_flow_arp_ar_dir / "rho_2.npy", np.array([0.2, -0.05]))
    np.save(params_flow_arp_ar_dir / "sigma_2.npy", np.array(0.9))

    # L96 training data
    l96_train = DATA_DIR / "l96_train" / L96_SINGLE_OUTPUT_SUBDIR
    l96_train.mkdir(parents=True, exist_ok=True)
    np.save(l96_train / "x.npy", np.random.randn(100, L96_TRAIN_K))
    np.save(l96_train / "y.npy", np.random.randn(100, L96_TRAIN_KJ))
    np.save(l96_train / "t.npy", np.linspace(0.0, 0.2, 100))


@pytest.fixture
def configs_dir():
    """Return the directory containing static test configs."""
    return BASE_DIR / "configs"


@pytest.fixture
def output_root(tmp_path):
    """Provide a temporary root for results to avoid polluting the repo."""
    return tmp_path / "results"
