#!/bin/bash
# Four-wire scale study: is PG-pre training reproducible on the A100? Four short processes (32 steps each):
# two with default settings and two with PyTorch's deterministic algorithms. Train and val states only.
# Submit from an immutable worktree of the pinned commit:
#   sbatch --export=ALL,GENCODOE_COMMIT=<full sha>,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder> \
#       scripts/hydra_fourwire_scale_determinism.slurm
#SBATCH --job-name=fwscale-determinism
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${GENCODOE_WORKTREE:?GENCODOE_WORKTREE must be set}"
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
OUT="data_out/fourwire_scale/determinism/job$SLURM_JOB_ID"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"                                               # fails if this job's folder exists: nothing is overwritten
for run in default_a default_b; do
  .venv/bin/python -m gencodoe.fourwire_scale_train determinism --label-root "$GENCODOE_LABEL_ROOT" --device cuda --out "$OUT/$run.json"
done
for run in deterministic_a deterministic_b; do
  .venv/bin/python -m gencodoe.fourwire_scale_train determinism --label-root "$GENCODOE_LABEL_ROOT" --device cuda --deterministic --out "$OUT/$run.json"
done
# exits 4 if a pair has no valid verdict
.venv/bin/python -m gencodoe.fourwire_scale_train determinism-report --out-dir "$OUT" --out "results/fourwire_scale_determinism_job$SLURM_JOB_ID.json"
