#!/bin/bash
# GENCO-OPF amendment 3: the one added method (gnn_genco) on the same test days, one array task per day,
# through the same exact check and scale-down; the frozen stage A rows are not re-run:
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=100-119 scripts/hydra_opf_test_genco.slurm
# Needs runs/gnn_genco_final and a gated graphkit run (hydra_opf_genco.slurm). Output: data_out/opf_case33/test_genco/dayNNN.json.
# Tasks start staggered by a few seconds (shared case files; see hydra_opf_test.slurm).
#SBATCH --job-name=opf-test-genco
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
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
[ -f data_out/opf_case33/runs/gnn_genco_final/best.pt ] || { echo "ABORT: missing run gnn_genco_final"; exit 4; }
.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open(\"results/genco_case33_gate.json\"))[\"pass\"] else 5)" || { echo "ABORT: gate G3b not passed"; exit 5; }
DAY=$SLURM_ARRAY_TASK_ID
OUT=data_out/opf_case33/test_genco/day$(printf %03d "$DAY").json
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }
echo "commit=$HEAD (verified) day=$DAY node=$(hostname)"
sha256sum data_out/opf_case33/runs/gnn_genco_final/best.pt results/genco_case33_gate.json
sleep $(( (DAY - 100) * 5 ))
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m gencodoe.opf_envelopes --split test --days "$DAY" --methods gnn_genco --tag final --out "$OUT"
