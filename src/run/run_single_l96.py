"""Run a single fully resolved Lorenz '96 simulation."""

import matplotlib

# Set non-interactive backend
matplotlib.use("PDF")

import numpy as np
from absl import app, flags, logging

from models.execute import initialize_l96, run_l96
from models.storage import save_output_l96
from plotting.plot_run_single_l96 import plot_all
from utils.config import ConfigL96
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


def run_from_config_path(config_path: str):
    """
    Does the full work: logging setup, seeding, sim, save, plotting.
    """
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigL96(config_path)

    if config.simulation_type != "single":
        raise ValueError(
            f"This script requires simulation_type='single', "
            f"but got '{config.simulation_type}'"
        )

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    np.random.seed(config.seed)

    logging.info("Initializing L96 model...")
    m = initialize_l96(config)

    logging.info(f"Running L96 for {config.total_time} time units...")
    x, y, t = run_l96(m, config)

    logging.info("Store results...")
    save_output_l96(config.output_dir(out_dir), x, y, t, backend=config.save_backend)

    if config.plot:
        logging.info("Plot results...")
        plot_start_idx = int(config.plot_start_time / config.si)

        plot_all(
            x,
            y,
            t,
            plot_start=plot_start_idx,
            base_dir=out_dir / "plots",
        )

    logging.info("DONE. Results stored in %s", str(out_dir))


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
