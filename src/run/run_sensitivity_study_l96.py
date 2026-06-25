"""Run Lorenz '96 initial-condition sensitivity ensembles."""

import gc
import re
from pathlib import Path

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
from utils.loading import load_initial_states
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

REQUIRED = ("x.npy", "y.npy", "t.npy")


def find_init_states_dir(base_dir: Path, sweep_name: str) -> Path:
    # pull (optional) tokens out of sweep_name (works whether separated by _, -, /, etc.)
    m_std = re.search(r"(perturb_std_[0-9.]+)", sweep_name)
    m_c = re.search(r"(c_[0-9.]+)", sweep_name)
    m_f = re.search(r"(F_[0-9.]+)", sweep_name)

    std_name = m_std.group(1) if m_std else None
    c_name = m_c.group(1) if m_c else None
    f_name = m_f.group(1) if m_f else None

    # c/F directory name (single dir, may be c only, F only, or both)
    cf_name = None
    if c_name and f_name:
        cf_name = f"{c_name}-{f_name}"
    elif c_name:
        cf_name = c_name
    elif f_name:
        cf_name = f_name

    # build candidate parent dir that contains "initial_states"
    parent = base_dir
    if std_name:
        parent = parent / std_name
    if cf_name:
        parent = parent / cf_name

    init_dir = parent / "initial_states"

    if init_dir.is_dir() and all((init_dir / f).is_file() for f in REQUIRED):
        # return the directory that contains "initial_states"
        return parent

    # fallback: recursive search, return first valid parent (deterministic order)
    for d in sorted(base_dir.rglob("initial_states")):
        if d.is_dir() and all((d / f).is_file() for f in REQUIRED):
            return d.parent

    raise FileNotFoundError(
        f"No matching initial_states under {base_dir} for sweep_name='{sweep_name}'. "
        f"Tried: {init_dir}"
    )


def run_from_config_path(config_path: str):
    logging.info("Loading configuration from %s...", config_path)
    config = ConfigL96(config_path)
    if config.simulation_type != "sensitivity_study":
        raise ValueError(
            f"This script requires simulation_type='sensitivity_study', "
            f"but got '{config.simulation_type}'"
        )

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info("Set random seed %i...", config.seed)
    np.random.seed(config.seed)

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------

    logging.info("Finding initial states directory...")
    init_states_dir = find_init_states_dir(config.init_states_dir, config.sweep_name)

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

    # n_ens_members=1 for sensitivity study, so keep the first perturbed member
    if x_init.ndim == 2:
        # (n_states, K) -> (n_states, 1, K)
        x_init = x_init[:, None, :]
        y_init = y_init[:, None, :]

    elif x_init.ndim == 3:
        # accept (n_states, M, K) but keep only the first member
        x_init = x_init[:, :1, :]  # keeps dim, shape (n_states, 1, K)
        y_init = y_init[:, :1, :]  # shape (n_states, 1, J*K)

    else:
        raise ValueError(
            f"Loaded initial states have invalid shape for sensitivity study: x_init.shape={x_init.shape}"
        )

    # ---------------------------------------------------------------------------
    # Run ensemble simulation
    # ---------------------------------------------------------------------------
    logging.info(
        "Running L96 ensemble with %d initial states and total_time=%f...",
        config.n_init_states,
        config.total_time,
    )

    if x_init.shape[0] > config.states_mem_limit:
        # Split ensemble to fit in memory
        splits = int(np.ceil(x_init.shape[0] / config.states_mem_limit))
        logging.info("Running %d-split ensemble...", splits)

        for s in range(splits):
            # Calculate split indices
            start_idx = s * config.states_mem_limit
            end_idx = min((s + 1) * config.states_mem_limit, x_init.shape[0])

            logging.info(
                "Processing split %d out of %d (states %d to %d)...",
                s + 1,
                splits,
                start_idx,
                end_idx,
            )

            # Extract split data
            x_split = x_init[start_idx:end_idx]
            y_split = y_init[start_idx:end_idx]

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
                backend=config.save_backend,
            )

            # Explicitly delete large arrays and force garbage collection
            del x, y, t, x_split, y_split
            gc.collect()

        logging.info("Merging split outputs using %s backend...", config.save_backend)
        merge_output_l96_ensemble(
            config.output_dir(out_dir),
            merge_y=True,
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
            config.output_dir(out_dir), x, y, t, backend=config.save_backend
        )

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
