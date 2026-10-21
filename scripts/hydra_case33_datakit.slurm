#!/bin/bash
# GENCO-OPF amendment 3: PF training states for the actual GENCO on case33bw (gencodoe.datakit_case33).
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_case33_datakit.slurm
# Writes data_out/datakit_case33 (20,000 states, days 0-99, splits by day) and data_out/datakit_case33_gate
# (2,000 val-day states for gate G3b). Then: sbatch scripts/hydra_train.slurm configs/genco_pf_case33_lv_prior.yaml
# data_out/datakit_case33 genco_pf_case33_lv_prior
#SBATCH --job-name=case33-datakit
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
echo "commit=$HEAD (verified) node=$(hostname)"
export OMP_NUM_THREADS=1
.venv/bin/python -m gencodoe.datakit_case33 --scenarios 20000 --processes 8 --augment 0.5
.venv/bin/python -m gencodoe.datakit_case33 --split gate --scenarios 2000 --processes 8 --augment 0.5
sha256sum grids/case33bw_opf.m data_out/datakit_case33/case33bw_opf/raw/bus_data.parquet data_out/datakit_case33_gate/case33bw_opf/raw/bus_data.parquet
