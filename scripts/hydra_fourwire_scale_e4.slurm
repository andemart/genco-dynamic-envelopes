#!/bin/bash
# Four-wire scale study, Amendment 11 (E4): balanced-model envelopes against four-wire-verified envelopes.
# CPU only; no learned model, no label read. Submit from a worktree created for the commit that holds this
# script and never moved, after `python -m gencodoe.fourwire_scale_eval networks`:
#
#   E="--export=ALL,GENCODOE_COMMIT=$(git rev-parse HEAD),GENCODOE_WORKTREE=$PWD"
#   S=scripts/hydra_fourwire_scale_e4.slurm
#   run=$(sbatch --parsable "$E,GENCODOE_STAGE=run" --array=1-12,14-25 $S)
#   sbatch --parsable "$E,GENCODOE_STAGE=report" --array=0 --time=00:30:00 --dependency=afterok:$run \
#          --kill-on-invalid-dep=yes $S
#
# run:    task = network number. One record, results/fourwire_scale_e4/network_N.json, published at the end;
#         every start leaves a file in results/fourwire_scale_e4/attempts. A record that exists is never
#         replaced (the command then fails).
# report: results/fourwire_scale_e4.json from the 24 records.
# If a run task ends without a record, Slurm removes the report job. Submit that network again alone
# (--array=N) and the report after it (--dependency=afterok:<that job>); Amendment 11, corrections 3 and 13.
# The 16 cores are required by the command (corrections, item 5).
#SBATCH --job-name=fwscale-e4
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK stage=$GENCODOE_STAGE task=$T"
PY=.venv/bin/python
S=scripts/fourwire_scale_balanced_envelopes.py

case "$GENCODOE_STAGE" in
  run)    $PY $S run --network "$T" ;;
  report) $PY $S report ;;
  *) echo "ABORT: unknown stage $GENCODOE_STAGE"; exit 4 ;;
esac
RC=$?
echo "rc=$RC"
exit $RC
