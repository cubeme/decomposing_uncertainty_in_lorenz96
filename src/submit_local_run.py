"""Launch parameter sweeps on the local machine."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from absl import app, flags

from utils.run_helpers import determine_run_module
from utils.sweep_utils import generate_run_configs

# ------------------------------- FLAGS ---------------------------------------
FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "config",
        None,
        "Path to the YAML configuration file.",
        required=True,
    )


# ----------------------------- Python executable ------------------------------

executable = "conda run -n uncertainty python"

# ----------------------------- Local functions --------------------------------


def run_shell_file(configs: list[Dict[str, Any]], run_module: str) -> None:
    """Generate shell submit scripts and launch them."""
    # Base of the shell file
    base = [
        "#!/bin/bash",
        "",
    ]

    n_runs = len(configs)

    for i, cfg in enumerate(configs):
        # Run directory
        run_dir = Path(cfg["results_dir"]) / cfg["experiment_name"] / cfg["sweep_name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config to run directory as YAML
        config_location = run_dir / "config.yaml"
        with open(config_location, "w") as fp:
            yaml.dump(cfg, fp, default_flow_style=False, sort_keys=False)

        # Create the run command
        runsh = f"printf '\\n\\nRunning {i + 1}/{n_runs}: {cfg['sweep_name']}...\\n'\n"
        runsh += executable
        runsh += " -u -m "
        runsh += run_module
        runsh += f" --config {config_location}"

        base.append(runsh)
        base.append("")

    base.append("printf '\\n\\nALL DONE\\n'")

    # Dump the string to the file
    with open("run.sh", "w") as file:
        file.write("\n".join(base))

    # Run the shell file
    os.system("chmod +x run.sh")
    os.system("./run.sh  > run.out 2>&1 &  ")


def run_from_config_path(config_path: str):
    """Initiate multiple runs."""
    # Load the base config file
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    configs, sweep = generate_run_configs(base_config)

    experiment_dir = Path(base_config["results_dir"]) / base_config["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Store sweep dictionary to {experiment_dir}...")
    with open(experiment_dir / "sweep.yaml", "w") as fp:
        yaml.safe_dump(sweep, fp, sort_keys=False)

    # Set run module
    run_module = determine_run_module(base_config)

    print(f"Generate all {len(configs)} run commands and start them...")
    run_shell_file(configs, run_module)

    print("ALL DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
