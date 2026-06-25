#!/bin/bash

#SBATCH --job-name=run_eval_correlation
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
# "ensemble_gcm_tail_flow", "ensemble_l96"


PROJECT_ROOT="$(cd "$(pwd)/../../.." && pwd)"

# Default (manual) model if no CLI argument is given
MODEL_DEFAULT="ensemble_gcm_baseline_det"
# If first argument is provided, use it; else use the default
MODEL="${1:-$MODEL_DEFAULT}"

# derive model type
MODEL_TYPE="long"
if [[ "$MODEL" == *"l96"* ]]; then
    MODEL_TYPE="truth"
fi

scontrol update JobId=$SLURM_JOB_ID JobName=run_eval_correlation_${MODEL}_${MODEL_TYPE}

DATA_DIR="${PROJECT_ROOT}/results/${MODEL}"
OUT_DIR="${PROJECT_ROOT}/notebooks/evaluation/output/ensemble_evaluation/${MODEL_TYPE}"

MAX_LAG=""

CMD=(conda run -n uncertainty python -u eval_correlation_driver.py
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
)

if [ -n "$MAX_LAG" ]; then
  CMD+=(--max_lag "$MAX_LAG")
fi

"${CMD[@]}"
