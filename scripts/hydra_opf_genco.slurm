#!/bin/bash
# GENCO-OPF amendment 3, after the graphkit PF run (hydra_train.slurm with configs/genco_pf_case33_lv_prior.yaml):
# gate G3b on the val-day states, frozen GENCO embeddings of the label days, one fit of gnn_genco.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_opf_genco.slurm [mlruns/<exp>/<run>]
# Stops (exit 5) if the gate fails; then no embedding and no fit, and results/genco_case33_gate.json says so.
#SBATCH --job-name=opf-genco
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
[ "$(ls data_out/opf_case33/labels/day0[0-9][0-9].npz | wc -l)" -eq 100 ] || { echo "ABORT: expected 100 label days"; exit 4; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN_ARG=(); [ $# -ge 1 ] && RUN_ARG=(--run "$1")
echo "commit=$HEAD (verified) node=$(hostname) run=${1:-newest}"
.venv/bin/python -m gencodoe.opf_genco gate "${RUN_ARG[@]}"
.venv/bin/python -m gencodoe.opf_genco embed "${RUN_ARG[@]}" --days $(seq 0 99)
.venv/bin/python -m gencodoe.opf_gnn fit --model gnn_genco --tag final --device cpu
sha256sum results/genco_case33_gate.json data_out/opf_case33/runs/gnn_genco_final/best.pt
