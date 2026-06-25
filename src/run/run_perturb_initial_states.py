"""Generate perturbed ensembles from Lorenz '96 initial states."""

import numpy as np
from absl import app, flags, logging

from models.storage import load_output_l96
from perturbations.iid import perturb_iid
from perturbations.wilks import perturb_wilks
from utils.config import ConfigPerturbInitialStates
from utils.loading import load_initial_states
from utils.run_helpers import configure_logging
from utils.saving import save_initial_states
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
# Main
# =============================================================================


def run_from_config_path(config_path: str) -> None:
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigPerturbInitialStates(config_path)

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info("Set random seed %d...", config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    logging.info(f"Loading L96 data from {config.init_states_dir}...")
    x_init, y_init, t_init = load_initial_states(
        config.init_states_dir
        / keep_only_load_sweep(config.sweep_name, config.load_sweep)
    )

    if config.n_init_states is not None:
        if config.n_init_states <= 0:
            raise ValueError("n_states must be positive when provided.")
        if config.n_init_states > x_init.shape[0]:
            raise ValueError(
                "n_states cannot exceed the number of available initial states."
            )

        logging.info("Using first %s initial states...", config.n_init_states)
        x_init = x_init[: config.n_init_states]
        y_init = y_init[: config.n_init_states]
        t_init = t_init[: config.n_init_states]

    logging.info("Perturb initial states...")

    if config.perturb_iid:
        logging.info("Using IID Gaussian perturbations with std=%s", config.perturb_std)
        x_init = perturb_iid(x_init, config.n_ens_members, config.perturb_std, rng)
        y_init = perturb_iid(y_init, config.n_ens_members, config.perturb_std, rng)

    else:
        # Long runs used to find analogues
        x_long, y_long, t = load_output_l96(
            config.l96_data_dir
            / keep_only_load_sweep(config.sweep_name, config.load_sweep)
            / config.l96_output_sub_dir,
            backend=config.l96_load_backend,
        )
        sigma_clim_x = np.std(x_long, ddof=1)
        sigma_clim_y = np.std(y_long, ddof=1)

        logging.info(
            "Using Wilks perturbations (local analogue covariance), target std=0.05*sigma_clim "
            "(sigma_clim_x=%s, sigma_clim_y=%s)",
            sigma_clim_x,
            sigma_clim_y,
        )
        x_init = perturb_wilks(
            x_init,
            config.n_ens_members,
            x_long,
            sigma_clim_x,
            rng,
            num_workers=config.cpu_count,
        )
        y_init = perturb_wilks(
            y_init,
            config.n_ens_members,
            y_long,
            sigma_clim_y,
            rng,
            num_workers=config.cpu_count,
        )

    logging.info("Store results in %s...", out_dir)
    save_initial_states(out_dir, x_init, y_init, t_init)

    logging.info("DONE. Results saved under %s", out_dir)


def main(argv):
    del argv
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
