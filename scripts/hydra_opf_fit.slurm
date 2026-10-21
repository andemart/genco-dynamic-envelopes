#!/bin/bash
# GENCO-OPF layer (docs/genco_opf_protocol.md, amendment 2): fit the GENCO-style graph model (task 0) and the MLP
# baseline (task 1) on the multi-scenario OPF labels of the training days, validation days for early stopping.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-1 scripts/hydra_opf_fit.slurm
# Output: data_out/opf_case33/runs/{gnn,mlp}_final/{best.pt,meta.json}. Needs data_out/opf_case33/labels/day000-099.
#SBATCH --job-name=opf-fit
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
[ "$(ls data_out/opf_case33/labels/day0[0-9][0-9].npz | wc -l)" -eq 100 ] || { echo "ABORT: expected 100 label days"; exit 4; }
MODELS=(gnn mlp); MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "commit=$HEAD (verified) model=$MODEL node=$(hostname)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python -m gencodoe.opf_gnn fit --model "$MODEL" --tag final --device cpu
