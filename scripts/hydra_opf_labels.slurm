#!/bin/bash
# GENCO-OPF layer (docs/genco_opf_protocol.md, amendment 2): multi-scenario OPF labels, one array task per day.
# Days 0-79 train, 80-99 validation; test days are refused by gencodoe.opf_labels itself.
# Submit a single day first so Julia precompiles PowerModels/Ipopt once in the shared depot, then the rest:
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=99      scripts/hydra_opf_labels.slurm
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-98    scripts/hydra_opf_labels.slurm
# Output: data_out/opf_case33/labels/dayNNN.npz (atomic; an existing day is skipped, so resubmission resumes).
#SBATCH --job-name=opf-labels
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
echo "commit=$HEAD (verified) day=$SLURM_ARRAY_TASK_ID node=$(hostname)"
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m gencodoe.opf_labels --days "$SLURM_ARRAY_TASK_ID"
