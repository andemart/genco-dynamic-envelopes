#!/bin/bash
# Four-wire scale study: the final runs (docs/fourwire_scale_protocol.md §8, Amendments 7 to 9).
# One script, five stages; the stage and the array say what a task does. Submit from the worktree that was
# created for the frozen commit and is never moved, after `python -m gencodoe.fourwire_scale_eval networks`:
#
#   E="--export=ALL,GENCODOE_COMMIT=<full sha>,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder>"
#   sbatch $E,GENCODOE_STAGE=pgpre     --array=0-11          --time=06:00:00 scripts/hydra_fourwire_scale_final.slurm
#   sbatch $E,GENCODOE_STAGE=target    --array=1-12,14-25    --time=03:00:00 scripts/hydra_fourwire_scale_final.slurm
#   sbatch $E,GENCODOE_STAGE=ft        --array=1-12,14-25    --time=01:30:00 --dependency=afterok:<pgpre job> ...
#   sbatch $E,GENCODOE_STAGE=evaluate  --array=1-12,14-25    --time=01:30:00 --dependency=afterok:<target>:<ft> ...
#   sbatch $E,GENCODOE_STAGE=timing    --array=25,10,1,7,2,17 --time=08:00:00 --cpus-per-task=16 --dependency=afterok:<evaluate> ...
#   sbatch $E,GENCODOE_STAGE=envelopes --array=25,10,1,7,2,17 --time=04:00:00 --dependency=afterok:<evaluate> ...
#
# pgpre: task = 3 x fold + seed. Every other stage: task = network number.
# A fit that already has a record is skipped: a completed fit is never rerun (Amendment 7).
#SBATCH --job-name=fwscale-final
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
RUNS=data_out/fourwire_scale/runs/final
FAILED=0

fit() {  # fit <run name> <arguments of the fit command>
  local name=$1; shift
  if [ -f "$RUNS/$name/record.json" ]; then echo "SKIP $name: it has a record"; return; fi
  echo "FIT $name"
  $PY -m gencodoe.fourwire_scale_train fit --label-root "$GENCODOE_LABEL_ROOT" --device cuda "$@" \
    || { echo "FAILED $name"; FAILED=1; }
}

case "$GENCODOE_STAGE" in
  pgpre)
    fit "pgpre_fold$((T / 3))_seed$((T % 3))" --kind pgpre --fold $((T / 3)) --seed $((T % 3)) ;;
  target)
    fit "pgnet_net$T" --kind pgnet --network "$T"
    fit "scratch_net${T}_b200_seed0" --kind scratch --network "$T" --budget 200 --seed 0
    for s in 0 1 2; do fit "scratch_net${T}_b1000_seed$s" --kind scratch --network "$T" --budget 1000 --seed $s; done
    fit "scratch_net${T}_b5000_seed0" --kind scratch --network "$T" --budget 5000 --seed 0 ;;
  ft)
    fit "ft_net${T}_b200_seed0" --kind ft --network "$T" --budget 200 --seed 0
    for s in 0 1 2; do fit "ft_net${T}_b1000_seed$s" --kind ft --network "$T" --budget 1000 --seed $s; done
    fit "ft_net${T}_b5000_seed0" --kind ft --network "$T" --budget 5000 --seed 0 ;;
  evaluate)
    $PY -m gencodoe.fourwire_scale_eval evaluate --network "$T" --label-root "$GENCODOE_LABEL_ROOT" --device cuda || FAILED=1 ;;
  timing)
    $PY -m gencodoe.fourwire_scale_e2 timing --network "$T" --device cuda || FAILED=1 ;;
  envelopes)
    $PY -m gencodoe.fourwire_scale_e2 envelopes --network "$T" --device cuda || FAILED=1 ;;
  *) echo "ABORT: unknown stage $GENCODOE_STAGE"; exit 4 ;;
esac
exit $FAILED
