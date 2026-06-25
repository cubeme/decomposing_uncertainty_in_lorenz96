"""Run fully resolved Lorenz '96 ensemble simulations."""

import gc
import os

import matplotlib

# Set non-interactive backend
matplotlib.use("PDF")

import numpy as np
from absl import app, flags, logging

from ensemble.l96_ensemble import run_l96_parallel
from ensemble.storage import (
    merge_output_l96_ensemble,
    save_output_l96_ensemble,
    save_output_l96_ensemble_split,
)
from utils.config import ConfigL96
from utils.loading import INIT_STATES_DIR, load_initial_states
from utils.run_helpers import configure_logging

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
# Run functions
# =============================================================================
def _get_init_states_dir(base_dir, config):
    def _dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)

    if _dir_exists(base_dir / INIT_STATES_DIR):
        return base_dir
    elif _dir_exists(base_dir / config.sweep_name / INIT_STATES_DIR):
        return base_dir / config.sweep_name
    else:
        raise ValueError(f"Not data found in {base_dir}")


def run_from_config_path(config_path: str):
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigL96(config_path)
    if config.simulation_type != "ensemble":
        raise ValueError(
            f"This script requires simulation_type='ensemble', "
            f"but got '{config.simulation_type}'"
        )

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    np.random.seed(config.seed)

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    init_states_dir = _get_init_states_dir(config.init_states_dir, config)
    logging.info(f"Loading initial states from {init_states_dir}...")
    x_init, y_init, _ = load_initial_states(init_states_dir)

    if config.n_init_states != x_init.shape[0]:
        if config.n_init_states > len(x_init):
            raise ValueError(
                f"Requested {config.n_init_states} initial states but only "
                f"{len(x_init)} available in {config.init_states_dir}"
            )
        x_init = x_init[: config.n_init_states]
        y_init = y_init[: config.n_init_states]

    if config.init_states_type == "perfect":
        if x_init.ndim != 2 or y_init.ndim != 2:
            raise ValueError(
                "For init_states_type='perfect', loaded initial states must be 2D "
                "(N_init_states, K) and (N_init_states, J*K)."
            )
        logging.info("Using perfect initial states...")

        # Expand to 3D with n_ens_members=1
        x_init = x_init[:, None, :]  # (n_init_states, 1, K)
        y_init = y_init[:, None, :]  # (n_init_states, 1, J*K)

    elif config.init_states_type == "perturbed":
        # Expect already-ensembled initial states
        if x_init.ndim != 3 or y_init.ndim != 3:
            raise ValueError(
                "For init_states_type='perturbed', loaded initial states must be 3D "
                "(N_init_states, N_ens_members, K) and (N_init_states, N_ens_members, J*K)."
            )
        if config.n_ens_members > x_init.shape[1]:
            raise ValueError(
                f"Requested {config.n_ens_members} ensemble members but only "
                f"{x_init.shape[1]} available in {config.init_states_dir}"
            )

        # Slice to requested number of ensemble members
        x_init = x_init[:, : config.n_ens_members, :]
        y_init = y_init[:, : config.n_ens_members, :]
    else:
        raise ValueError(f"Unknown init_states_type='{config.init_states_type}'")

    # ---------------------------------------------------------------------------
    # Run ensemble simulation
    # ---------------------------------------------------------------------------
    # Total number of simulations ("runs") you will execute
    n_init_states = x_init.shape[0]
    n_ens_members = x_init.shape[1]
    n_runs = n_init_states * n_ens_members

    if n_runs > config.states_mem_limit:
        # Split along init_states axis so each chunk has at most states_mem_limit runs
        # runs_per_init_state = n_ens_members
        max_init_per_split = max(1, config.states_mem_limit // n_ens_members)
        splits = int(np.ceil(n_init_states / max_init_per_split))

        logging.info(
            "Running %d-split ensemble (max %d init states per split => <= %d runs per split)...",
            splits,
            max_init_per_split,
            max_init_per_split * n_ens_members,
        )

        for s in range(splits):
            start_idx = s * max_init_per_split
            end_idx = min((s + 1) * max_init_per_split, n_init_states)

            logging.info(
                "Processing split %d/%d (init states %d to %d; runs %d)...",
                s + 1,
                splits,
                start_idx,
                end_idx,
                (end_idx - start_idx) * n_ens_members,
            )

            x_split = x_init[start_idx:end_idx]  # (N_split, M, K)
            y_split = y_init[start_idx:end_idx]  # (N_split, M, J*K)

            x, y, t = run_l96_parallel(
                x_split,
                y_split,
                config,
                num_processes=config.cpu_count,
            )

            logging.info("Store results split %d out of %d...", s + 1, splits)
            save_output_l96_ensemble_split(
                config.output_dir(out_dir),
                x,
                y,
                t,
                s + 1,
                save_y=config.save_y,
                backend=config.save_backend,
            )

            del x, y, t, x_split, y_split
            gc.collect()

        logging.info("Merging split outputs using %s backend...", config.save_backend)
        merge_output_l96_ensemble(
            config.output_dir(out_dir),
            merge_y=config.save_y,
            backend=config.save_backend,
            delete_parts=True,
        )

    else:
        x, y, t = run_l96_parallel(
            x_init,
            y_init,
            config,
            num_processes=config.cpu_count,
        )
        logging.info("Store results...")
        save_output_l96_ensemble(
            config.output_dir(out_dir),
            x,
            y,
            t,
            save_y=config.save_y,
            backend=config.save_backend,
        )

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
