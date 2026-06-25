from pathlib import Path

import pytest
import yaml

from src import submit_slurm_run
from utils.run_helpers import determine_run_module
from utils.sweep_utils import generate_run_configs

CONFIG_FILES = [
    "fit_params_flow_ar_p_base_no_sweep.yaml",
    "fit_params_flow_hidden_dims_sweep.yaml",
    "fit_params_flow_tail_history_forcing_no_sweep.yaml",
    "gcm_baseline_ar1_sweep_ar_order.yaml",
    "gcm_baseline_det_sweep_f.yaml",
    "gcm_bayes_n_models_no_sweep.yaml",
    "gcm_bayes_sweep_f_c.yaml",
    "gcm_flow_arp_base_sweep_f.yaml",
    "gcm_flow_sweep_f.yaml",
    "gcm_flow_tail_history_forcing_sweep_noise_type.yaml",
    "generate_initial_states_sweep_n_init.yaml",
    "l96_ensemble_sweep_f_c_mem_limit.yaml",
    "l96_ensemble_sweep_schedule_conditional_params.yaml",
    "l96_sensitivity_study_sweep.yaml",
    "parameter_fitting_baseline_ar_order_no_sweep.yaml",
    "parameter_fitting_bayes_no_sweep.yaml",
    "perturb_initial_states_iid_conditional_params.yaml",
    "perturb_initial_states_wilks_no_sweep.yaml",
    "single_l96_sweep_f.yaml",
]


def _ensure_flags_parsed():
    if "num_cpus" not in submit_slurm_run.FLAGS:
        submit_slurm_run.define_flags()
    if not submit_slurm_run.FLAGS.is_parsed():
        submit_slurm_run.FLAGS(["test"])


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_submit_slurm_run_smoke(config_name, configs_dir, tmp_path, monkeypatch):
    """Verify submit_slurm_run.main writes job scripts and configs for all examples."""
    _ensure_flags_parsed()
    submit_slurm_run.FLAGS["num_cpus"].value = 3
    submit_slurm_run.FLAGS["num_gpus"].value = 1
    submit_slurm_run.FLAGS["qos"].value = "gpu_normal"
    submit_slurm_run.FLAGS["mem_mb"].value = 12345
    submit_slurm_run.FLAGS["max_runtime"].value = "00-00:10:00"

    cfg_path = configs_dir / config_name
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(tmp_path / "results")
    configs, sweep = generate_run_configs(base_config)
    run_module = determine_run_module(base_config)

    config_yaml = tmp_path / "config.yaml"
    with open(config_yaml, "w") as fp:
        yaml.safe_dump(base_config, fp)

    submitted: list[str] = []
    monkeypatch.setattr(
        submit_slurm_run.os, "system", lambda cmd: submitted.append(cmd)
    )
    monkeypatch.chdir(tmp_path)
    submit_slurm_run.run_from_config_path(str(config_yaml))

    experiment_dir = Path(base_config["results_dir"]) / base_config["experiment_name"]
    # Config files lie in subdirs for each sweep combination
    config_files = sorted(experiment_dir.glob("**/config.yaml"))
    assert len(config_files) == len(configs)
    assert (experiment_dir / "sweep.yaml").exists()
    assert yaml.safe_load((experiment_dir / "sweep.yaml").read_text()) == sweep

    run_job = tmp_path / "run_job.sh"
    assert run_job.exists()
    run_job_text = run_job.read_text()
    assert run_module in run_job_text
    assert "#SBATCH --cpus-per-task=3" in run_job_text
    assert "#SBATCH --mem=12345" in run_job_text
    assert "#SBATCH --time=00-00:10:00" in run_job_text
    assert "#SBATCH --partition=gpu_p" in run_job_text
    assert "#SBATCH --gres=gpu:1" in run_job_text
    assert "#SBATCH --qos=gpu_normal" in run_job_text
    assert submitted == ["sbatch run_job.sh"] * len(configs)
