"""Generate Lorenz '96 initial states by spin-up or trajectory selection."""

import numpy as np
from absl import app, flags, logging

from models.execute import initialize_l96, run_l96
from utils.config import ConfigL96
from utils.run_helpers import configure_logging
from utils.saving import save_initial_states, save_seeds

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
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigL96(config_path)

    if config.simulation_type != "IC_generation":
        raise ValueError(
            f"This script requires simulation_type='IC_generation', "
            f"but got '{config.simulation_type}'"
        )

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    np.random.seed(config.seed)

    # ---------------------------------------------------------------------------
    # Generate initial states
    # ---------------------------------------------------------------------------

    init_states_x = np.zeros((config.n_init_states, config.K))
    init_states_y = np.zeros((config.n_init_states, config.K * config.J))
    init_states_t = np.zeros((config.n_init_states,))

    if config.generate_method == "spin_up":
        logging.info(
            f"Generating {config.n_init_states} initial states via independent spin-ups..."
        )
        init_seeds = np.arange(
            config.seed, config.seed + config.n_init_states + 1, dtype=int
        )

        for i in range(config.n_init_states):
            m = initialize_l96(config, seed=init_seeds[i])

            init_states_x[i] = m.x
            init_states_y[i] = m.y
            init_states_t[i] = m.t

    elif config.generate_method == "selection":
        logging.info(
            f"Generating {config.n_init_states} initial states via selection "
            f"from single run (selection_mtu={config.selection_mtu})..."
        )

        # Calculate required simulation time
        n_selection_steps = int(config.selection_mtu / config.si)
        required_time = (
            int(
                (config.n_init_states * n_selection_steps + n_selection_steps)
                * config.si
            )
            + 1
        )

        # Update config total_time for this run
        config.total_time = required_time
        logging.info(f"Running L96 for {required_time} time units...")

        # Run single long L96 simulation
        m = initialize_l96(config, seed=config.seed)
        x, y, t = run_l96(m, config)

        # Select initial states at regular intervals
        for i in range(1, config.n_init_states + 1):
            select_index = i * n_selection_steps

            init_states_x[i - 1] = x[select_index, :]
            init_states_y[i - 1] = y[select_index, :]
            init_states_t[i - 1] = t[select_index]

    else:
        raise ValueError(
            f"Unknown generate_method: {config.generate_method}. "
            "Must be 'spin_up' or 'selection'."
        )

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    logging.info("Store results...")
    save_initial_states(
        config.output_dir(out_dir), init_states_x, init_states_y, init_states_t
    )
    if config.generate_method == "spin_up":
        save_seeds(config.output_dir(out_dir), init_seeds)

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
