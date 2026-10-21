#!/bin/bash
# Amendment 5 option 1 (docs/genco_opf_protocol.md): the official pretrained GENCO as warm start for IPOPT on 500
# held-out case118_flex scenarios. Runs in the case118 worktree ($VSC_DATA/genco-case118) with the main checkout's
# venv and Julia environment; timing on one core.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_amendment5_warmstart.slurm
#SBATCH --job-name=a5-warmstart
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
MAIN="$VSC_DATA/genco-dynamic-envelopes"; cd "$VSC_DATA/genco-case118"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts configs results grids || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
HUB=data_out/hub/genco-opf-datakit/case118_ieee/base/seed0
sha256sum $HUB/last.pt $HUB/normalizer_stats.pt $HUB/case118_ieee_base_seed0.yaml grids/case118_flex.m
export PYTHONPATH="$PWD/src" PYTHON_JULIAPKG_PROJECT="$MAIN/.venv/julia_env" PYTHON_JULIAPKG_OFFLINE=yes
PY="$MAIN/.venv/bin/python"
echo "commit=$HEAD (verified) node=$(hostname)"
RAW=data_out/case118_heldout/case118_flex/raw
if [ ! -f "$RAW/bus_data.parquet" ] && [ ! -d "$RAW/bus_data.parquet" ]; then
  export OMP_NUM_THREADS=1
  "$MAIN/.venv/bin/gridfm_datakit" generate configs/datakit_opf_case118_flex_heldout.yaml | grep -v "Processing scenarios" | tail -3
fi
sha256sum "$RAW/args.log" $(ls "$RAW"/scenarios_agg_load_profile.parquet 2>/dev/null)
export OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
"$PY" -m gencodoe.opf_warmstart --raw "$RAW" --case grids/case118_flex.m --hub "$HUB" --n 500 --out results/genco_opf_amendment5_warmstart.json
sha256sum results/genco_opf_amendment5_warmstart.json
