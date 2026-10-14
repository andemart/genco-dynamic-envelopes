#!/bin/bash
# Train GENCO (PF mode) on ENWL1 + Fluvius states, on VUB Hydra.
# Expects the repo in $VSC_DATA/genco-dynamic-envelopes with .venv built and the
# datakit dataset synced (raw parquet + splits/ for split_from_existing_files).
# Submit: sbatch scripts/hydra_train.slurm [config] [data_path] [exp_name]
#SBATCH --job-name=genco-enwl1
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

CONFIG="${1:-configs/genco_pf_enwl1_tiny.yaml}"
DATA="${2:-data_out/datakit}"
EXP="${3:-$(basename "$CONFIG" .yaml)}"
echo "config=$CONFIG data=$DATA exp=$EXP job=$SLURM_JOB_ID"
cat "$CONFIG"

export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
# gencodoe.graphkit_cli = gridfm_graphkit CLI with the LV/prior model variants registered
.venv/bin/python -m gencodoe.graphkit_cli train \
  --config "$CONFIG" \
  --data_path "$DATA" \
  --log_dir mlruns \
  --exp_name "$EXP" \
  --num_workers "$SLURM_CPUS_PER_TASK"
