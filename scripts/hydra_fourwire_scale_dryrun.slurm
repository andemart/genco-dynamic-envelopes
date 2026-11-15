#!/bin/bash
# Four-wire scale study: dry run of the whole chain on the A100, before the freeze. Short fits, then the
# evaluation, the timing harness and the envelope check, with train and val labels only (Amendment 8, item 14).
# Checks that every step runs on CUDA with deterministic algorithms; it produces no result.
#   sbatch --export=ALL,GENCODOE_COMMIT=<full sha>,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder>,GENCODOE_NETWORK=17 \
#       scripts/hydra_fourwire_scale_dryrun.slurm
#SBATCH --job-name=fwscale-dryrun
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${GENCODOE_WORKTREE:?GENCODOE_WORKTREE must be set}"
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
: "${GENCODOE_LABEL_ROOT:?GENCODOE_LABEL_ROOT must be set}"
N="${GENCODOE_NETWORK:?GENCODOE_NETWORK must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK network=$N affinity=$(nproc)"
PY=.venv/bin/python
FOLD=$($PY -c "from gencodoe import fourwire_scale_train as t; print(t.fold_of($N))")
FIT="$PY -m gencodoe.fourwire_scale_train fit --label-root $GENCODOE_LABEL_ROOT --device cuda"
$PY -m gencodoe.fourwire_scale_eval networks
$FIT --kind pgpre --fold "$FOLD" --seed 0 --dry-train 64 --dry-val 64 --dry-epochs 2
$FIT --kind pgnet --network "$N" --dry-train 1000 --dry-val 3000 --dry-epochs 2
$FIT --kind scratch --network "$N" --budget 200 --seed 0 --dry-train 200 --dry-val 3000 --dry-epochs 2
$FIT --kind ft --network "$N" --budget 200 --seed 0 --dry-train 200 --dry-val 3000 --dry-epochs 2
$PY -m gencodoe.fourwire_scale_eval evaluate --network "$N" --label-root "$GENCODOE_LABEL_ROOT" --device cuda --dry
$PY -m gencodoe.fourwire_scale_e2 timing --network "$N" --device cuda --dry --warmup 1 --repeats 3
$PY -m gencodoe.fourwire_scale_e2 envelopes --network "$N" --device cuda --dry --days 1 --hours 2
echo DRY_RUN_OK
