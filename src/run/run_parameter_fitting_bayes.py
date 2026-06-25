"""Fit a Bayesian polynomial parameterization to Lorenz '96 data."""

import os
from pathlib import Path

import matplotlib

# Set non-interactive backend
matplotlib.use("PDF")

import numpy as np
from absl import app, flags, logging

from models.forcing_schedule import ConstantForcingSchedule, forcing_at_array
from models.storage import load_output_l96
from parameterization.bayes.compute_bayesian_regression import fit_bayesian_regression
from parameterization.utils.helpers import compute_coupling_from_x
from parameterization.utils.storage import save_bayesian_poly_coefficients
from plotting.plot_parameter_fitting_bayes import plot_all
from utils.config import ConfigParamsFit
from utils.run_helpers import clear_tmp_dir, configure_logging, slurm_jobs_matching

# -------------------------- Temp directory ----------------------------------
# Default temp directory is /tmp which is apparently too small on CHC
# for PyMC sampling with PyTensor
tmp_dir = str(Path("../tmp").resolve())
os.makedirs(tmp_dir, exist_ok=True)

os.environ["TMPDIR"] = tmp_dir
os.environ["TMP"] = tmp_dir
os.environ["TEMP"] = tmp_dir

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
# MAIN
# =============================================================================


def run_from_config_path(config_path: str):
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigParamsFit(config_path)

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    np.random.seed(config.seed)

    logging.info(f"Loading L96 data from {config.l96_data_dir}...")
    x, y, t = load_output_l96(
        config.l96_data_dir / config.sweep_name / config.l96_output_sub_dir,
        backend=config.l96_load_backend,
    )
    del y

    if config.l96_load_backend == "zarr":
        x = np.asarray(x)  # loads into memory

    # ---------------------------------------------------------------------------
    # Compute coupling term from X (FULL series first)
    # ---------------------------------------------------------------------------
    logging.info("Computing coupling term from X (full series)...")
    F_values = forcing_at_array(config.f_schedule, t)
    u, x = compute_coupling_from_x(x, config.si, F_values, config.h, config.b, config.c)
    # u_all/x_all are length N-1 relative to original x/t
    t = t[:-1]

    # ---------------------------------------------------------------------------
    # Select training data percentage
    #   - constant: contiguous prefix (as before)
    #   - linear/oscillating: sample contiguous chunks of 100 TU from (u_all, x_all)
    # ---------------------------------------------------------------------------
    f_type = (
        "constant"
        if isinstance(config.f_schedule, ConstantForcingSchedule)
        else "time-varying"
    )
    N_all = u.shape[0]

    if f_type == "constant":
        train_index = int(config.train_perc * N_all)
        logging.info("Selecting %d samples for training (contiguous).", train_index)

        u_train = u[:train_index]
        x_train = x[:train_index]
        t_train = t[:train_index]

    else:
        chunk_TU = config.chunk_length  # in time units (TU)
        chunk_len = int(round(chunk_TU / config.si))  # indices per chunk
        chunk_len = max(chunk_len, 1)

        n_chunks = N_all // chunk_len
        if n_chunks < 1:
            raise ValueError(
                f"Not enough data for chunking: N={N_all}, chunk_len={chunk_len}"
            )

        n_train_chunks = int(round(config.train_perc * n_chunks))
        n_train_chunks = max(1, min(n_train_chunks, n_chunks))

        rng = np.random.default_rng(config.seed)
        chunks = np.arange(n_chunks)
        rng.shuffle(chunks)
        train_chunks = np.sort(
            chunks[:n_train_chunks]
        )  # sort only for nicer time order

        idx = np.concatenate(
            [
                np.arange(ch * chunk_len, ch * chunk_len + chunk_len)
                for ch in train_chunks
            ]
        )

        logging.info(
            "Selecting %d/%d chunks for training (chunk_len=%d steps ≈ %.1f TU) => %d samples.",
            n_train_chunks,
            n_chunks,
            chunk_len,
            chunk_len * config.si,
            idx.size,
        )

        u_train = u[idx]
        x_train = x[idx]
        t_train = t[idx]

    # ---------------------------------------------------------------------------
    # Fit parameterization parameters
    # ---------------------------------------------------------------------------
    return_samples = config.n_ens_members * config.n_models

    logging.info("Fitting Bayesian polynomial regression...")
    bayes_coefs = fit_bayesian_regression(
        x_train,
        u_train,
        config.poly_order,
        chains=config.chains,
        draws=config.draws,
        tune=config.tune,
        return_samples=return_samples,
    )

    bayes_coefs = bayes_coefs.reshape(
        (
            config.n_ens_members,
            config.n_models,
            config.poly_order + 1,
        )
    )

    logging.info("Storing Bayesian polynomial coefficients...")
    save_bayesian_poly_coefficients(config.coefs_dir(out_dir), bayes_coefs)

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    if config.plot:
        logging.info("Plotting results...")

        plot_all(
            x_train,
            u_train,
            bayes_coefs,
            poly_order=config.poly_order,
            base_dir=out_dir / "plots",
        )

    # ---------------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------------
    # Check if there are still parameter fitting jobs running on SLURM
    pattern = "fitted_parameters*"
    running = slurm_jobs_matching(pattern)
    if running:
        logging.info(
            "Found SLURM jobs matching '%s' (not clearing tmp): %s" % (pattern, running)
        )
    else:
        logging.info("Clear tmp directory...")
        clear_tmp_dir(tmp_dir)

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
