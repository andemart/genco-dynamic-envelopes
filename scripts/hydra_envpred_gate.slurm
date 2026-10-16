#!/bin/bash
# Envelope prediction (docs/envpred_protocol.md): exact gate + fallback + fresh feasibility for all four arms,
# one array task per test file (180). Needs data_out/envpred/predictions.parquet from `envpred fit`.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-179 scripts/hydra_envpred_gate.slurm
#SBATCH --job-name=envpred-gate
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
echo "task=$SLURM_ARRAY_TASK_ID commit=${GENCODOE_COMMIT:-unknown}"
.venv/bin/python -m gencodoe.envpred gate --file-index "$SLURM_ARRAY_TASK_ID"
