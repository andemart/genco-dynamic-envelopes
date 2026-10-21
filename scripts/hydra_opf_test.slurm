#!/bin/bash
# GENCO-OPF layer (docs/genco_opf_protocol.md, amendment 2): the frozen test-day evaluation, one array task per
# test day (100-119), all six methods through the same exact check and scale-down:
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=100-119 scripts/hydra_opf_test.slurm
# Needs data_out/opf_case33/runs/{gnn,mlp}_final (hydra_opf_fit.slurm). Output: data_out/opf_case33/test/dayNNN.json.
# Tasks start staggered by a few seconds: oc.OPF rewrites the shared case33_units_*.m files on start-up.
#SBATCH --job-name=opf-test
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
for r in gnn mlp; do [ -f data_out/opf_case33/runs/${r}_final/best.pt ] || { echo "ABORT: missing run ${r}_final"; exit 4; }; done
DAY=$SLURM_ARRAY_TASK_ID
OUT=data_out/opf_case33/test/day$(printf %03d "$DAY").json
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }
echo "commit=$HEAD (verified) day=$DAY node=$(hostname)"
sha256sum data_out/opf_case33/runs/*_final/best.pt
sleep $(( (DAY - 100) * 5 ))
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m gencodoe.opf_envelopes --split test --days "$DAY" --methods uniform lp ipopt multi gnn mlp --tag final --out "$OUT"
