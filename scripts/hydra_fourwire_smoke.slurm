#!/bin/bash
# Unbalanced GENCO smoke (protocol smoke gate 4): PhaseGENCO embeddings for train/val, then a short
# PhaseGraph and PhaseGENCO fit on train/val only.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_fourwire_smoke.slurm <epochs>
#SBATCH --job-name=fw-smoke
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=4
: "${GENCODOE_COMMIT:?}"
[ "$(git rev-parse HEAD)" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD mismatch"; exit 2; }
git diff --quiet HEAD -- src scripts || { echo "ABORT: tracked files modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for s in train val; do .venv/bin/python -m gencodoe.fourwire_train embeddings --split $s --device cuda; done
for m in phasegraph phasegenco; do
  .venv/bin/python -m gencodoe.fourwire_train fit --model $m --epochs "$1" --tag smoke --device cuda
done
