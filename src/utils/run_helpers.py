"""Provide shared helpers for experiment execution."""

import fnmatch
import getpass
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil
from absl import logging
from absl.flags import FLAGS


def get_flag(key: str, value: Any) -> str:
    """Format a command-line flag for a given key and value."""
    if isinstance(value, bool):
        return f" --{key}" if value else f" --no{key}"
    else:
        return f" --{key} {value}"


def remove_key(arg, key):
    """Remove a key from a dictionary if present."""
    try:
        del arg[key]
    except KeyError:
        pass


def clear_tmp_dir(tmp_dir):
    """Clear the contents of a temporary directory."""
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Current memory usage: {memory_mb:.1f} MB")


def slurm_jobs_matching(pattern, user=None):
    user = user or getpass.getuser()
    try:
        proc = subprocess.run(
            ["squeue", "-u", user, "--noheader", "-o", "%i %j %T"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logging.warning("squeue not found; skipping SLURM job check and clearing tmp.")
        return []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    matches = []
    for ln in lines:
        parts = ln.split(None, 2)
        if len(parts) >= 2:
            jobid, name = parts[0], parts[1]
            state = parts[2] if len(parts) >= 3 else ""
            if fnmatch.fnmatch(name, pattern):
                matches.append((jobid, name, state))
    return matches


def determine_run_module(base_config):
    try:
        run_module = base_config["run_module"]
    except KeyError as exc:
        raise ValueError('Missing required "run_module" in config.') from exc

    if not isinstance(run_module, str) or "." in run_module:
        raise ValueError(
            'Invalid "run_module" in config; expected file name of run module without extension.'
        )

    return f"run.{run_module}"


def configure_logging(out_dir: Path) -> None:
    """
    Preserve your existing absl logging behavior for CLI runs.
    In tests you can still call this, or skip it if you want quieter logs.
    """
    FLAGS.alsologtostderr = True

    # Ensure directory exists before absl log file handler
    out_dir.mkdir(parents=True, exist_ok=True)

    # absl uses this for log files
    FLAGS.log_dir = str(out_dir)

    # Write absl logs to out_dir with program_name="run"
    logging.get_absl_handler().use_absl_log_file(program_name="run")
