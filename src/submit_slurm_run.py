"""Submit a parameter sweep to Slurm."""

import os
from copy import deepcopy
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
        "",
        "Path to the YAML configuration file.",
        required=False,
    )

    # Resource requirements *per run*
    flags.DEFINE_integer("num_cpus", 8, "Number of CPUs per run.", required=False)
    flags.DEFINE_integer("num_gpus", 0, "Number of GPUs per run.", required=False)
    flags.DEFINE_string(
        "qos", "cpu_preemptible", "Quality of Service per run.", required=False
    )
    flags.DEFINE_integer("mem_mb", 128000, "Memory (in MB) per run.", required=False)
    flags.DEFINE_string(
        "max_runtime",
        "00-01:00:00",
        "Maximum runtime (DD-HH:MM:SS) per run.",
        required=False,
    )


# ----------------------------- Python executable ------------------------------

executable = "conda run -n uncertainty python"

# ----------------------------- Local functions --------------------------------


def submit_all_jobs(configs: list[Dict[str, Any]], run_module: str) -> None:
    """Generate shell submit scripts and launch them."""
    # Base of the shell file
    base = [
        "#!/bin/bash",
        "",
    ]

    base.append(f"#SBATCH --cpus-per-task={FLAGS.num_cpus}")
    base.append(f"#SBATCH --mem={FLAGS.mem_mb}")
    base.append(f"#SBATCH --time={FLAGS.max_runtime}")
    base.append("#SBATCH --nice=10000")
    if FLAGS.num_gpus > 0:
        base.append("#SBATCH --partition=gpu_p")
        base.append(f"#SBATCH --gres=gpu:{FLAGS.num_gpus}")
    else:
        base.append("#SBATCH --partition=cpu_p")
    base.append(f"#SBATCH --qos={FLAGS.qos}")
    # base.append("#SBATCH --dependency=afterok:33836363:33836364")

    for i, cfg in enumerate(configs):
        lines = deepcopy(base)

        lines.append(
            f"#SBATCH --job-name={cfg['experiment_name']}_{cfg['sweep_name']}{'_gpu' if FLAGS.num_gpus > 0 else ''}"
        )
        # Run directory
        run_dir = Path(cfg["results_dir"]) / cfg["experiment_name"] / cfg["sweep_name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config to run directory as YAML
        config_location = run_dir / "config.yaml"
        with open(config_location, "w") as fp:
            yaml.dump(cfg, fp, default_flow_style=False, sort_keys=False)

        # The output, logs, and errors from running the scripts
        logs_name = run_dir / "slurm"
        lines.append(f"#SBATCH -o {logs_name}.out")
        lines.append(f"#SBATCH -e {logs_name}.err")

        # Queue job
        lines.append("")
        runsh = executable
        runsh += " -u -m "
        runsh += run_module
        runsh += f" --config {config_location}"

        lines.append(runsh)
        lines.append("")

        # Now dump the string into the file.
        with open("run_job.sh", "w") as file:
            file.write("\n".join(lines))

        print(f"Submitting {i + 1}...")
        os.system("sbatch run_job.sh")


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

    print(f"Generate all {len(configs)} submit script and launch them...")
    submit_all_jobs(configs, run_module)

    print("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
