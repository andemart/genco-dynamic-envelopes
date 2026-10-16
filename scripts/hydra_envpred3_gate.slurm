#!/bin/bash
# Envelope prediction v3: exact gate + fallback + fresh feasibility, all arms x shrink points, one test file per task.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-179 scripts/hydra_envpred3_gate.slurm
#SBATCH --job-name=envpred3-gate
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
echo "task=$SLURM_ARRAY_TASK_ID commit=${GENCODOE_COMMIT:-unknown}"
.venv/bin/python -m gencodoe.envpred3 gate --file-index "$SLURM_ARRAY_TASK_ID"
