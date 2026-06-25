"""Submit multiple experiment configurations to Slurm."""

import subprocess
from pathlib import Path

RESOURCE_SETTINGS = {
    "ensemble_gcm_short_cpu": {
        "num_cpus": 32,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 128000,
        "max_runtime": "00-03:00:00",
    },
    "ensemble_gcm_flow": {
        "num_cpus": 8,
        "num_gpus": 1,
        "qos": "gpu_normal",
        "mem_mb": 128000,
        "max_runtime": "00-05:00:00",
    },
    "ensemble_gcm_long_cpu": {
        "num_cpus": 16,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 128000,
        "max_runtime": "00-01:00:00",
    },
    "ensemble_l96_short": {
        "num_cpus": 16,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 256000,
        "max_runtime": "00-05:00:00",
    },
    "ensemble_l96_long": {
        "num_cpus": 16,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 256000,
        "max_runtime": "00-00:30:00",
    },
    "parameter_fitting_baseline": {
        "num_cpus": 8,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 32000,
        "max_runtime": "00-00:30:00",
    },
    "parameter_fitting_bayes": {
        "num_cpus": 20,
        "num_gpus": 0,
        "qos": "cpu_normal",
        "mem_mb": 512000,
        "max_runtime": "00-24:00:00",
    },
    "generate_initial_conditions": {
        "num_cpus": 8,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 32000,
        "max_runtime": "00-00:30:00",
    },
    "perturb_initial_conditions_iid": {
        "num_cpus": 8,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 16000,
        "max_runtime": "00-00:10:00",
    },
    "perturb_initial_conditions_wilks": {
        "num_cpus": 40,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 512000,
        "max_runtime": "00-12:00:00",
    },
    "l96_single_run": {
        "num_cpus": 8,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 128000,
        "max_runtime": "00-00:30:00",
    },
    "l96_spinup": {
        "num_cpus": 8,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 16000,
        "max_runtime": "00-00:10:00",
    },
    "l96_sensitivity": {
        "num_cpus": 32,
        "num_gpus": 0,
        "qos": "cpu_preemptible",
        "mem_mb": 256000,
        "max_runtime": "00-00:30:00",
    },
    "flow_training": {
        "num_cpus": 8,
        "num_gpus": 1,
        "qos": "gpu_normal",
        "mem_mb": 32000,
        "max_runtime": "00-02:00:00",
    },
}

CONFIGS = [
    # # ================= Training data =================
    # (
    #     "configs/generate_l96_training_data.yaml",
    #     "l96_single_run",
    # ),
    # # ================= Generate initial states =================
    # (
    #     "configs/generate_initial_states.yaml",
    #     "generate_initial_conditions",
    # ),
    # # ================= Perturb initial states =================
    # (
    #     "configs/perturb_initial_states.yaml",
    #     "perturb_initial_conditions_iid",
    # ),
    # # ================= Parameters bayes =================
    # ("configs/fit_params_bayes.yaml", "parameter_fitting_bayes"),
    # # ================= Parameters baseline =================
    # ("configs/fit_params_baseline.yaml", "parameter_fitting_baseline"),
    # # ================= Flow parameter fitting =================
    # ("configs/fit_params_ar_base_flow.yaml", "flow_training"),
    # ("configs/fit_params_flow.yaml", "flow_training"),
    # ("configs/fit_params_forcing_flow.yaml", "flow_training"),
    # ("configs/fit_params_history_flow.yaml", "flow_training"),
    # ("configs/fit_params_tail_flow.yaml", "flow_training"),
    # # ================= Constant forcing ensembles =================
    # ---- L96 ensembles
    # ("configs/L96/ensemble_l96_long.yaml", "ensemble_l96_long"),
    # ("configs/L96/ensemble_l96_short.yaml", "ensemble_l96_short"),
    # ("configs/L96/ensemble_l96_sensitivity.yaml", "ensemble_l96_short"),
    # # ---- GCM baselines det
    # (
    #     "configs/baseline_det/ensemble_gcm_baseline_det_full.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/baseline_det/ensemble_gcm_baseline_det_long.yaml",
    #     "ensemble_gcm_long_cpu",
    # ),
    # (
    #     "configs/baseline_det/ensemble_gcm_baseline_det_mix.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/baseline_det/ensemble_gcm_baseline_det_perfect.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # # ---- GCM baselines AR1
    # (
    #     "configs/baseline_ar1/ensemble_gcm_baseline_ar1_full.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/baseline_ar1/ensemble_gcm_baseline_ar1_long.yaml",
    #     "ensemble_gcm_long_cpu",
    # ),
    # (
    #     "configs/baseline_ar1/ensemble_gcm_baseline_ar1_mix.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/baseline_ar1/ensemble_gcm_baseline_ar1_perfect.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # # ---- GCM bayes ensembles
    # (
    #     "configs/bayes/ensemble_gcm_bayes_full.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/bayes/ensemble_gcm_bayes_long.yaml",
    #     "ensemble_gcm_long_cpu",
    # ),
    # (
    #     "configs/bayes/ensemble_gcm_bayes_mix.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # (
    #     "configs/bayes/ensemble_gcm_bayes_perfect.yaml",
    #     "ensemble_gcm_short_cpu",
    # ),
    # # ==== GCM flow ensembles
    # # ---- flow with AR1 base distribution
    # (
    #     "configs/ar_base_flow/ensemble_gcm_ar_base_flow_full.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/ar_base_flow/ensemble_gcm_ar_base_flow_long.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/ar_base_flow/ensemble_gcm_ar_base_flow_mix.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/ar_base_flow/ensemble_gcm_ar_base_flow_perfect.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # # ---- normal flow
    # ("configs/flow/ensemble_gcm_flow_full.yaml", "ensemble_gcm_flow"),
    # ("configs/flow/ensemble_gcm_flow_long.yaml", "ensemble_gcm_flow"),
    # ("configs/flow/ensemble_gcm_flow_mix.yaml", "ensemble_gcm_flow"),
    # (
    #     "configs/flow/ensemble_gcm_flow_perfect.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # #---- forcing flow
    # (
    #     "configs/forcing_flow/ensemble_gcm_forcing_flow_full.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/forcing_flow/ensemble_gcm_forcing_flow_long.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/forcing_flow/ensemble_gcm_forcing_flow_mix.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/forcing_flow/ensemble_gcm_forcing_flow_perfect.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # # ---- history flow
    # (
    #     "configs/history_flow/ensemble_gcm_history_flow_full.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/history_flow/ensemble_gcm_history_flow_long.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/history_flow/ensemble_gcm_history_flow_mix.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/history_flow/ensemble_gcm_history_flow_perfect.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # # ---- tail flow
    # (
    #     "configs/tail_flow/ensemble_gcm_tail_flow_full.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/tail_flow/ensemble_gcm_tail_flow_long.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/tail_flow/ensemble_gcm_tail_flow_mix.yaml",
    #     "ensemble_gcm_flow",
    # ),
    # (
    #     "configs/tail_flow/ensemble_gcm_tail_flow_perfect.yaml",
    #     "ensemble_gcm_flow",
    # )
]

# Run script in same directory
SCRIPT = Path(__file__).resolve().parent / "submit_slurm_run.py"


def run_one(config_path: str, resources: dict) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        "uncertainty",
        "python",
        str(SCRIPT),
        f"--config={config_path}",
        f"--num_cpus={resources['num_cpus']}",
        f"--num_gpus={resources['num_gpus']}",
        f"--qos={resources['qos']}",
        f"--mem_mb={resources['mem_mb']}",
        f"--max_runtime={resources['max_runtime']}",
    ]
    # subprocess.run(cmd, check=True)

    p = subprocess.run(cmd, text=True, capture_output=True)
    print("\nreturncode:", p.returncode)
    print("\nSTDOUT:\n", p.stdout)
    print("\nSTDERR:\n", p.stderr)
    p.check_returncode()


def run_all(CONFIGS, RESOURCE_SETTINGS):
    for config_path, resource_key in CONFIGS:
        config_path = str(Path(config_path))
        resources = RESOURCE_SETTINGS[resource_key]
        print(f"\n\n=== Submitting {Path(config_path).name} with {resource_key} ===")
        run_one(config_path, resources)


if __name__ == "__main__":
    run_all(CONFIGS, RESOURCE_SETTINGS)
