import time
from pathlib import Path

import pytest
import yaml

from src import submit_local_run
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


def _set_load_directories(submit_input_data, base_config):
    experiment_name = base_config.get("experiment_name", "")

    # conditional input states
    if experiment_name == "l96_ensemble_sweep_schedule_conditional_params":
        rules = base_config["conditional_params"]["init_states_dir"]["values"]

        for r in rules:
            t = r["when"].get("type")
            if t == "linear":
                r["set"] = str(submit_input_data["l96_init_perturbed"])
            elif t == "oscillating":
                r["set"] = str(submit_input_data["l96_init_perturbed_cond"])

        base_config["conditional_params"]["init_states_dir"]["values"] = rules
    elif experiment_name.startswith("gcm_baseline_det"):
        base_config["init_states_dir"] = str(submit_input_data["gcm_init_perturbed"])
        base_config["params_dir"] = str(submit_input_data["params_det"])
    elif experiment_name.startswith("gcm_baseline_arp"):
        base_config["init_states_dir"] = str(submit_input_data["gcm_init_perturbed"])
        base_config["params_dir"] = str(submit_input_data["params_ar_p"])
    elif experiment_name.startswith("gcm_bayes"):
        base_config["init_states_dir"] = str(submit_input_data["gcm_init"])
        base_config["params_dir"] = str(submit_input_data["params_bayes"])
    elif experiment_name == "gcm_flow_tail_history_forcing_ensemble":
        base_config["init_states_dir"] = str(submit_input_data["gcm_init_perturbed"])
        base_config["params_dir"] = str(
            submit_input_data["params_flow_tail_history_forcing"]
        )
    elif experiment_name == "gcm_flow_arp_base_ensemble":
        base_config["init_states_dir"] = str(submit_input_data["gcm_init_perturbed"])
        base_config["params_dir"] = str(submit_input_data["params_flow_arp_base"])
    elif experiment_name.startswith("gcm_flow"):
        base_config["init_states_dir"] = str(submit_input_data["gcm_init_perturbed"])
        base_config["params_dir"] = str(submit_input_data["params_flow"])
    elif experiment_name.startswith("l96_ensemble"):
        init_key = (
            "l96_init_perturbed"
            if base_config.get("init_states_type") == "perturbed"
            else "l96_init"
        )
        base_config["init_states_dir"] = str(submit_input_data[init_key])
    elif experiment_name.startswith("l96_sensitivity_study"):
        base_config["init_states_dir"] = str(submit_input_data["l96_init_perturbed"])
    elif experiment_name.startswith("perturb_initial_states_iid"):
        base_config["init_states_dir"] = str(submit_input_data["l96_init"])
    elif experiment_name.startswith("perturb_initial_states_wilks"):
        base_config["init_states_dir"] = str(submit_input_data["l96_init"])
        base_config["l96_data_dir"] = str(submit_input_data["l96_train"])
    elif experiment_name.startswith("fitted_parameters"):
        base_config["l96_data_dir"] = str(submit_input_data["l96_train"])
    elif experiment_name.startswith("fit_params_flow"):
        base_config["l96_data_dir"] = str(submit_input_data["l96_train"])


@pytest.mark.slow
@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_submit_local_run_smoke(
    config_name, configs_dir, tmp_path, monkeypatch, submit_input_data
):
    """Verify submit_local_run.main writes scripts and configs for all examples."""
    cfg_path = configs_dir / config_name
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(tmp_path / "results")
    _set_load_directories(submit_input_data, base_config)

    configs, sweep = generate_run_configs(base_config)
    run_module = determine_run_module(base_config)

    config_yaml = tmp_path / "config.yaml"
    with open(config_yaml, "w") as fp:
        yaml.safe_dump(base_config, fp)

    # Run in isolated temp directory and allow system calls to execute
    monkeypatch.chdir(tmp_path)
    submit_local_run.run_from_config_path(str(config_yaml))

    run_script = tmp_path / "run.sh"
    assert run_script.exists()
    assert run_module in run_script.read_text()

    run_out = tmp_path / "run.out"
    expected_runs = len(configs)
    run_out_text = ""
    for _ in range(1000):
        if run_out.exists():
            run_out_text = run_out.read_text()
            if "ALL DONE" in run_out_text:
                break
        time.sleep(0.1)

    assert run_out.exists()
    assert "ALL DONE" in run_out_text

    for cfg in configs:
        assert cfg["sweep_name"] in run_out_text
    for i in range(1, expected_runs + 1):
        assert f"Running {i}/{expected_runs}" in run_out_text
    assert "error" not in run_out_text.lower()

    experiment_dir = Path(base_config["results_dir"]) / base_config["experiment_name"]
    # Config files lie in subdirs for each sweep combination
    config_files = sorted(experiment_dir.glob("**/config.yaml"))
    assert len(config_files) == len(configs)
    if "no_sweep" not in config_name:
        assert {p.parent.name for p in config_files} == {
            cfg["sweep_name"] for cfg in configs
        }

    sweep_path = experiment_dir / "sweep.yaml"
    assert sweep_path.exists()
    assert yaml.safe_load(sweep_path.read_text()) == sweep
