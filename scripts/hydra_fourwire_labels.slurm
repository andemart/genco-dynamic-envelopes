#!/bin/bash
# Four-wire OpenDSS labels (docs/unbalanced_genco_protocol.md). One shard per array task.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-<k-1> scripts/hydra_fourwire_labels.slurm <split> <k>
#SBATCH --job-name=fw-labels
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts || { echo "ABORT: tracked files modified"; exit 3; }
echo "split=$1 shard=$SLURM_ARRAY_TASK_ID/$2 commit=$HEAD node=$(hostname)"
.venv/bin/python -m gencodoe.fourwire_labels generate --split "$1" --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$2"
