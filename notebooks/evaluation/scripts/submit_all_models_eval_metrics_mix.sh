#!/bin/bash
set -euo pipefail

SCRIPT="run_eval_metrics_mix.sh"

MODELS=(
  "ensemble_gcm_baseline_det"
  "ensemble_gcm_baseline_ar_p"
  "ensemble_gcm_bayes"
  "ensemble_gcm_flow"
  "ensemble_gcm_ar_base_flow"
  "ensemble_gcm_forcing_flow"
  "ensemble_gcm_history_flow"
  "ensemble_gcm_tail_flow"
)


for m in "${MODELS[@]}"; do
  sbatch "$SCRIPT" "$m"
done