#!/bin/bash

#SBATCH --job-name=run_eval_metrics_mix
#SBATCH --cpus-per-task=36
#SBATCH --mem=128000
#SBATCH --time=00-00:30:00
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_preemptible
#SBATCH -o <absolute path>/stochastic_uncertainty/notebooks/evaluation/scripts/logs/slurm_%j.out
#SBATCH -e <absolute path>/stochastic_uncertainty/notebooks/evaluation/scripts/logs/slurm_%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Workdir: $(pwd)"
echo ""

# "ensemble_gcm_baseline_det", "ensemble_gcm_baseline_ar1",
# "ensemble_gcm_bayes", "ensemble_gcm_flow", "ensemble_gcm_ar_base_flow",
# "ensemble_gcm_forcing_flow", "ensemble_gcm_history_flow",
# "ensemble_gcm_tail_flow"


PROJECT_ROOT="$(cd "$(pwd)/../../.." && pwd)"

# Default (manual) model if no CLI argument is given
MODEL_DEFAULT="ensemble_gcm_baseline_det"
# If first argument is provided, use it; else use the default
MODEL="${1:-$MODEL_DEFAULT}"

scontrol update JobId=$SLURM_JOB_ID JobName=run_compute_metrics_mix_${MODEL}

MODEL_DIR="${PROJECT_ROOT}/results/${MODEL}"
L96_DIR="${PROJECT_ROOT}/results/ensemble_l96"
OUT_DIR="${PROJECT_ROOT}/notebooks/evaluation/output/ensemble_evaluation/mix"

N_INIT_STATES=300
N_ENS_MEMBERS=20

conda run -n uncertainty python -u eval_metrics_mix_driver.py \
  --model_dir "${MODEL_DIR}" \
  --l96_dir "${L96_DIR}" \
  --out_dir "${OUT_DIR}" \
  --n_init_states "${N_INIT_STATES}" \
  --n_ens_members "${N_ENS_MEMBERS}" \

echo "Done: $(date)"
