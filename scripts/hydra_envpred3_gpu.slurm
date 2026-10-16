#!/bin/bash
# Envelope prediction v3 (docs/envpred_protocol.md, amendments 3/3a): GENCO feature extraction on an A100.
# Smoke:    sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_envpred3_gpu.slurm smoke
# Features: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-15 scripts/hydra_envpred3_gpu.slurm features 16
#SBATCH --job-name=envpred3-gpu
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd=$1 commit=${GENCODOE_COMMIT:-unknown} task=${SLURM_ARRAY_TASK_ID:-none}"
if [ "$1" = smoke ]; then
  .venv/bin/python -m gencodoe.envpred3 smoke --device cuda --out results/envpred3_smoke.json
else
  .venv/bin/python -m gencodoe.envpred3 features --device cuda --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "$2"
fi
