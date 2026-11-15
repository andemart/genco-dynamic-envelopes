#!/bin/bash
# Four-wire scale study: the final runs (docs/fourwire_scale_protocol.md §8, Amendments 7 to 9).
# One script, six stages; the stage and the array say what a task does. Submit from the worktree that was
# created for the frozen commit and is never moved, after `python -m gencodoe.fourwire_scale_eval networks`:
#
#   scripts/hydra_fourwire_scale_submit.sh <label folder>
# which submits, with their dependencies:
#   stage      array                 time limit   cores
#   pgpre      0-11                  12 h         8
#   target     1-12,14-25            6 h          8      PG-net and the five PG-scratch fits of a network
#   ft         1-12,14-25            3 h          8      after every pgpre task
#   evaluate   1-12,14-25            1.5 h        8      after every target and ft task
#   timing     25,10,1,7,2,17        8 h          16     after every evaluate task
#   envelopes  25,10,1,7,2,17        4 h          8      after every evaluate task
#
# pgpre: task = 3 x fold + seed. Every other stage: task = network number.
# A run whose record is valid and passes every check is skipped (exit code 10): it is never run again.
# Whether an attempt that left no valid record may be run again is decided by the command (Amendment 7).
# An error inside a final run halts every later final command (data_out/fourwire_scale/runs/final/HALT.json).
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
FAILED=0

run() {  # run <what> <python module and arguments>: 10 means "it has a valid record", which is a skip
  local what=$1; shift
  echo "RUN $what"
  $PY -m "$@"
  local rc=$?
  if [ $rc -eq 10 ]; then echo "SKIP $what: it has a valid record"
  elif [ $rc -ne 0 ]; then echo "FAILED $what (exit $rc)"; FAILED=1; fi
}
fit() {  # fit <run name> <arguments of the fit command>
  local name=$1; shift
  run "$name" gencodoe.fourwire_scale_train fit --label-root "$GENCODOE_LABEL_ROOT" --device cuda "$@"
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
    run "evaluation of network $T" gencodoe.fourwire_scale_eval evaluate --network "$T" --label-root "$GENCODOE_LABEL_ROOT" --device cuda ;;
  timing)
    run "timing of network $T" gencodoe.fourwire_scale_e2 timing --network "$T" --device cuda ;;
  envelopes)
    run "envelopes of network $T" gencodoe.fourwire_scale_e2 envelopes --network "$T" --device cuda ;;
  *) echo "ABORT: unknown stage $GENCODOE_STAGE"; exit 4 ;;
esac
exit $FAILED
