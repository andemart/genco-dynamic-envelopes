#!/bin/bash
# Four-wire scale study, Amendment 10: PG-pre under the stabilised recipe (exploratory; train and val only).
# Not a final run of the study: it writes under data_out/fourwire_scale/runs/stabilised and results/ only.
# Submit from a worktree created for the commit that holds this script and never moved, after
# `python -m gencodoe.fourwire_scale_eval networks`:
#
#   E="--export=ALL,GENCODOE_COMMIT=$(git rev-parse HEAD),GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder>"
#   S=scripts/hydra_fourwire_scale_stabilised.slurm
#   eq=$(sbatch --parsable "$E,GENCODOE_STAGE=equivalence" --array=0 --time=00:30:00 $S)
#   sbatch --parsable "$E,GENCODOE_STAGE=fit" --array=0-11 --time=12:00:00 --dependency=afterok:$eq $S
#
# equivalence: the frozen fit routine and the script's loop with the frozen switches on one reduced fit
#              (fold 2, seed 1, 650 train and 256 val states per network, 4 epochs); it fails unless they are
#              equal bit for bit, and the fits then do not start.
# fit:         task = 3 x fold + seed, as the pgpre stage of the final runs. A run folder is never replaced.
#SBATCH --job-name=fwscale-stab
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
cd "${GENCODOE_WORKTREE:?GENCODOE_WORKTREE must be set}"
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # shared .venv must import this pinned worktree
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
: "${GENCODOE_LABEL_ROOT:?GENCODOE_LABEL_ROOT must be set}"
: "${GENCODOE_STAGE:?GENCODOE_STAGE must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
T=$SLURM_ARRAY_TASK_ID
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK stage=$GENCODOE_STAGE task=$T"
PY=.venv/bin/python
S=scripts/fourwire_scale_pgpre_stabilised.py

case "$GENCODOE_STAGE" in
  equivalence)
    $PY $S equivalence --fold 2 --seed 1 --n-train 650 --n-val 256 --epochs 4 \
      --label-root "$GENCODOE_LABEL_ROOT" --device cuda \
      --work data_out/fourwire_scale/stabilised_equivalence \
      --out results/fourwire_scale_pgpre_stabilised_equivalence.json ;;
  fit)
    $PY $S fit --recipe stabilised --fold $((T / 3)) --seed $((T % 3)) \
      --label-root "$GENCODOE_LABEL_ROOT" --device cuda ;;
  *) echo "ABORT: unknown stage $GENCODOE_STAGE"; exit 4 ;;
esac
RC=$?
echo "rc=$RC"
exit $RC
