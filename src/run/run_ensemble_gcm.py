"""Run reduced-model ensembles with a configured parameterization."""

import os
import random

import numpy as np
import torch
from absl import app, flags, logging

from ensemble.gcm_flow_param_batched import (
    run_flow_param_ensemble_batched_single_gpu as run_flow_ensemble,
)
from ensemble.gcm_polynomial_ar_p_param import (
    run_poly_ar_p_param_ensemble_parallel_multiprocessing as run_ar_p_ensemble,
)
from ensemble.gcm_polynomial_param import (
    run_poly_param_ensemble_parallel_multiprocessing as run_poly_ensemble,
)
from ensemble.storage import (
    save_output_gcm_ensemble,
)
from parameterization.flow.storage import load_checkpoint
from parameterization.utils.storage import (
    load_ar_p_parameters,
    load_bayesian_poly_coefficients,
    load_poly_coefficients,
    save_bayesian_poly_coefficients,
)
from utils.config import ConfigGCM
from utils.loading import INIT_STATES_DIR, load_initial_states
from utils.run_helpers import configure_logging
from utils.saving import save_seeds
from utils.sweep_utils import keep_only_load_sweep

# ------------------------------- FLAGS ---------------------------------------
FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "config",
        None,
        "Path to the YAML configuration file.",
        required=True,
    )


# =============================================================================
# Utility functions
# =============================================================================


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_model_seeds(config: ConfigGCM):
    model_start_seed = int(config.model_start_seed)
    N, M, L = config.n_init_states, config.n_ens_members, config.n_models

    if M > 1 and L > 1:
        # per-model seeds only, broadcast across init states and members
        seeds_1d = np.arange(model_start_seed, model_start_seed + L, dtype=int)
        return np.broadcast_to(seeds_1d[None, None, :], (N, M, L)).copy()

    # otherwise: unique seed per (N, M, L)
    n_total = N * M * L
    seeds = np.arange(model_start_seed, model_start_seed + n_total, dtype=int)
    return seeds.reshape(N, M, L)


def _process_initial_states(x_init: np.ndarray, config: ConfigGCM) -> np.ndarray:
    if config.n_init_states != x_init.shape[0]:
        if config.n_init_states > len(x_init):
            raise ValueError(
                f"Requested {config.n_init_states} initial states but only "
                f"{len(x_init)} available in {config.init_states_dir}"
            )
        x_init = x_init[: config.n_init_states]

    if config.init_states_type == "perfect":
        if x_init.ndim != 2:
            raise ValueError(
                "For init_states_type='perfect', loaded initial states must be 2D (N_init_states, K)."
            )

        if config.n_ens_members != 1:
            raise ValueError("For init_states_type='perfect', n_ens_members must be 1.")

        # Broadcast for computation
        x_init = np.broadcast_to(
            x_init[:, None, None, :],
            (config.n_init_states, 1, config.n_models, config.K),
        ).copy()

        logging.info("Using perfect initial states...")

    elif config.init_states_type == "perturbed":
        # Expect already-ensembled initial states
        if x_init.ndim != 3:
            raise ValueError(
                "For init_states_type='perturbed', loaded initial states must be 3D "
                "(N_init_states, N_ens_members, K)."
            )
        if config.n_ens_members != x_init.shape[1]:
            if config.n_ens_members > x_init.shape[1]:
                raise ValueError(
                    f"Requested {config.n_ens_members} ensemble members but only "
                    f"{x_init.shape[1]} available in {config.init_states_dir}"
                )

            # Slice to requested number of ensemble members
            x_init = x_init[:, : config.n_ens_members, :]

        # Broadcast for computation
        x_init = np.broadcast_to(
            x_init[:, :, None, :],
            (config.n_init_states, config.n_ens_members, config.n_models, config.K),
        ).copy()
    else:
        raise ValueError(f"Unknown init_states_type='{config.init_states_type}'")

    return x_init


def _get_load_dir(base_dir, subdir, config):
    def _dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)

    if _dir_exists(base_dir / subdir):
        # Load from base dir directly
        return base_dir
    elif _dir_exists(base_dir / config.sweep_name / subdir):
        # Load from full sweep dir in base dir
        return base_dir / config.sweep_name
    elif _dir_exists(
        base_dir / keep_only_load_sweep(config.sweep_name, config.load_sweep) / subdir
    ):
        # Load from load_sweep-only dir in base dir
        return base_dir / keep_only_load_sweep(config.sweep_name, config.load_sweep)
    else:
        raise ValueError(f"No data found in {base_dir}")


def _process_bayesian_coefficients(coefs, config):
    M, L, poly_order_plus_one = coefs.shape
    if M != config.n_ens_members:
        if M < config.n_ens_members:
            raise ValueError(
                f"Loaded Bayesian coefficients have {M} ensemble members, but config requests {config.n_ens_members}."
            )
        coefs = coefs[: config.n_ens_members, :, :]
    if L != config.n_models:
        if L < config.n_models:
            raise ValueError(
                f"Loaded Bayesian coefficients have {L} models, but config requests {config.n_models}."
            )
        coefs = coefs[:, : config.n_models, :]
    return coefs


# =============================================================================
# Run function
# =============================================================================


def run_from_config_path(config_path: str):
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigGCM(config_path)
    if config.simulation_type != "ensemble":
        raise ValueError(
            f"This script requires simulation_type='ensemble', "
            f"but got '{config.simulation_type}'"
        )

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    _seed_everything(config.seed)

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    init_states_dir = _get_load_dir(config.init_states_dir, INIT_STATES_DIR, config)
    logging.info(f"Loading initial states from {init_states_dir}...")
    x_init, _, _ = load_initial_states(init_states_dir)
    x_init = _process_initial_states(x_init, config)

    # ---------------------------------------------------------------------------

    # Load parameterization parameters based on type
    coefs = None
    ar_p_rho = None
    ar_p_sigma = None
    ar_p_seeds = None

    if config.parameterization_type in [
        "baseline_det",
        "baseline_ar_p",
    ]:
        params_dir = _get_load_dir(config.params_dir, config.coefs_dir_name, config)
        logging.info(f"Loading polynomial coefficients from {params_dir}...")
        coefs = load_poly_coefficients(params_dir / config.coefs_dir_name)

    if config.parameterization_type == "baseline_ar_p":
        params_dir = _get_load_dir(config.params_dir, config.coefs_dir_name, config)
        ar_order = config.ar_order
        logging.info("Loading AR(%d) parameters from %s...", ar_order, params_dir)
        ar_p_rho, ar_p_sigma = load_ar_p_parameters(
            params_dir / config.ar_parameters_dir_name, ar_order=ar_order
        )

    if config.parameterization_type == "bayesian_regression":
        params_dir = _get_load_dir(config.params_dir, config.coefs_dir_name, config)
        logging.info(f"Loading Bayesian polynomial coefficients from {params_dir}...")
        coefs = load_bayesian_poly_coefficients(params_dir / config.coefs_dir_name)

        coefs = _process_bayesian_coefficients(coefs, config)

    if config.parameterization_type == "flow":
        params_dir = _get_load_dir(
            config.params_dir, config.flow_model_dir_name, config
        )
        logging.info(f"Loading flow models from {params_dir}...")

        flow_model, _ = load_checkpoint(
            params_dir / config.flow_model_dir_name,
            load_optimizer=False,
        )
        flow_model = flow_model.to(torch.device(config.flow_device)).eval()

        flow_rho = 0.0
        flow_sigma = None
        if config.noise_type == "ar_p":
            logging.info(
                f"Loading rho, sigma from {params_dir} with ar_order={config.ar_order}..."
            )
            flow_rho, flow_sigma = load_ar_p_parameters(
                params_dir / config.ar_parameters_dir_name, ar_order=config.ar_order
            )

    # ---------------------------------------------------------------------------
    # Run ensemble simulation based on parameterization type
    # ---------------------------------------------------------------------------
    logging.info(
        f"Running {config.parameterization_type} ensemble with "
        f"{config.n_init_states} initial states and {config.n_ens_members} members..."
    )

    if config.parameterization_type == "baseline_det":
        x, t = run_poly_ensemble(
            x_init,
            config,
            coefs,
            num_processes=config.cpu_count,
        )

    elif config.parameterization_type == "baseline_ar_p":
        ar_p_seeds = _generate_model_seeds(config)

        x, t = run_ar_p_ensemble(
            x_init,
            config,
            coefs,
            ar_p_rho,
            ar_p_sigma,
            ar_p_seeds,
            num_processes=config.cpu_count,
        )
        save_seeds(out_dir, ar_p_seeds)

    elif config.parameterization_type == "bayesian_regression":
        x, t = run_poly_ensemble(
            x_init,
            config,
            coefs,
            num_processes=config.cpu_count,
        )
        save_bayesian_poly_coefficients(out_dir, coefs)

    elif config.parameterization_type == "flow":
        flow_seeds = _generate_model_seeds(config)

        x, t = run_flow_ensemble(
            x_init,
            config,
            flow_model,
            seeds=flow_seeds,
            rho=flow_rho,
            sigma=flow_sigma,
            device=config.flow_device,
            time_stepping=config.time_stepping,
        )
        save_seeds(out_dir, flow_seeds)

    else:
        raise ValueError(
            f"Unknown parameterization type: {config.parameterization_type}"
        )
    # ---------------------------------------------------------------------------
    # Run ensemble simulation based on parameterization type
    # ---------------------------------------------------------------------------

    logging.info("Storing results...")
    save_output_gcm_ensemble(
        config.output_dir(out_dir), x, t, backend=config.save_backend
    )

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
