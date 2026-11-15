#!/bin/bash
# Four-wire scale study, Amendment 3 CUDA validation gate (not an E2 timing run).
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_fourwire_scale_cuda.slurm
#SBATCH --job-name=fwscale-cuda
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${GENCODOE_WORKTREE:-$VSC_DATA/genco-dynamic-envelopes}"
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # shared .venv must import this pinned worktree
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
.venv/bin/python -m gencodoe.fourwire_scale validate-sweeps --device cuda --processes "$SLURM_CPUS_PER_TASK"
