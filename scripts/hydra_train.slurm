#!/bin/bash
# Train GENCO Tiny (PF mode) on ENWL1 + Fluvius states, on VUB Hydra.
# Expects the repo in $VSC_DATA/genco-dynamic-envelopes with .venv built and
# data_out/datakit/enwl1_fluvius/raw synced from the datakit run.
# Submit: sbatch scripts/hydra_train.slurm [config]
#SBATCH --job-name=genco-enwl1-tiny
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

export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/gridfm_graphkit train \
  --config "${1:-configs/genco_pf_enwl1_tiny.yaml}" \
  --data_path data_out/datakit \
  --log_dir mlruns \
  --exp_name genco-pf-enwl1-tiny \
  --num_workers "$SLURM_CPUS_PER_TASK"
