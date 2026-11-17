#!/bin/bash
# Four-wire scale study, Amendment 12 (E5): the E4 experiment with the positive-sequence model as a third
# design model, under three technology mixes (base, half, pv). CPU only; no learned model, no label read.
# Submit from a worktree created for the commit that holds this script and never moved, after
# `python -m gencodoe.fourwire_scale_eval networks`:
#
#   E="--export=ALL,GENCODOE_COMMIT=$(git rev-parse HEAD),GENCODOE_WORKTREE=$PWD"
#   S=scripts/hydra_fourwire_scale_e5.slurm
#   a=$(sbatch --parsable "$E,GENCODOE_STAGE=run,GENCODOE_MIX=base" --array=1-12,14-25 $S)
#   b=$(sbatch --parsable "$E,GENCODOE_STAGE=run,GENCODOE_MIX=half" --array=1-12,14-25 $S)
#   c=$(sbatch --parsable "$E,GENCODOE_STAGE=run,GENCODOE_MIX=pv" --array=1-12,14-25 $S)
#   sbatch --parsable "$E,GENCODOE_STAGE=report" --array=0 --time=00:30:00 \
#          --dependency=afterok:$a:$b:$c --kill-on-invalid-dep=yes $S
#
# run:    task = network number, GENCODOE_MIX = the technology mix. One record,
#         results/fourwire_scale_e5/<mix>/network_N.json, published at the end; every start leaves a file in
#         results/fourwire_scale_e5/<mix>/attempts. A record that exists is never replaced (the command fails).
# report: results/fourwire_scale_e5.json from the 72 records.
# If a run task fails, Slurm removes the report job. Without a record of that network and mix: submit that
# task again alone (--array=N with its GENCODOE_MIX) and the report after it. With a complete record (the task
# failed after publishing it): keep the record, do not run it again, submit the report again alone.
# The 16 cores are required by the command.
#SBATCH --job-name=fwscale-e5
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
cd "${GENCODOE_WORKTREE:?GENCODOE_WORKTREE must be set}"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # shared .venv must import this pinned worktree
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
: "${GENCODOE_STAGE:?GENCODOE_STAGE must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
T=$SLURM_ARRAY_TASK_ID
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK stage=$GENCODOE_STAGE mix=${GENCODOE_MIX:-} task=$T"
PY=.venv/bin/python
S=scripts/fourwire_scale_balanced_envelopes_e5.py

case "$GENCODOE_STAGE" in
  run)    $PY $S run --network "$T" --mix "${GENCODOE_MIX:?GENCODOE_MIX must be set}" ;;
  report) $PY $S report ;;
  *) echo "ABORT: unknown stage $GENCODOE_STAGE"; exit 4 ;;
esac
RC=$?
echo "rc=$RC"
exit $RC
