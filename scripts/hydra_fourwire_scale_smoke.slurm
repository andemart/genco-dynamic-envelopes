#!/bin/bash
# Four-wire scale study, §9 gate 5: PG-pre smoke fit on one fold (train and val only; not a final fit).
# Submit from the worktree:
#   sbatch --export=ALL,GENCODOE_COMMIT=<full sha>,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder> \
#       scripts/hydra_fourwire_scale_smoke.slurm
#SBATCH --job-name=fwscale-smoke
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${GENCODOE_WORKTREE:-$VSC_DATA/genco-dynamic-envelopes}"
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # shared .venv must import this pinned worktree
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
: "${GENCODOE_LABEL_ROOT:?GENCODOE_LABEL_ROOT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
.venv/bin/python -m gencodoe.fourwire_scale_train smoke --label-root "$GENCODOE_LABEL_ROOT" --fold 0 --epochs 3 --device cuda
