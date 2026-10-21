#!/bin/bash
# GENCO-OPF amendment 4: the frozen partial-metering evaluation, one array task per test day (100-119).
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=100-119 scripts/hydra_opf_se_test.slurm
# Needs results/se_calibration.json and results/se_estimators.txt (hydra_opf_se.slurm). Output: data_out/opf_case33/se/dayNNN.json.
#SBATCH --job-name=opf-se-test
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
VARIANT=${GENCO_VARIANT:-recon}; TAG=${SE_TAG:-recon}                 # amendment 4: VARIANT=se TAG=""; 4b: recon
CAL=results/se_calibration${TAG:+_$TAG}.json; ESTF=results/se_estimators${TAG:+_$TAG}.txt; OUTDIR=data_out/opf_case33/se${TAG:+_$TAG}
EST=$(sed -n 's/^estimators: //p' "$ESTF")
[ -n "$EST" ] || { echo "ABORT: no estimators recorded"; exit 4; }
DAY=$SLURM_ARRAY_TASK_ID
OUT=$OUTDIR/day$(printf %03d "$DAY").json
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }
echo "commit=$HEAD (verified) day=$DAY estimators=$EST variant=$VARIANT node=$(hostname)"
sha256sum "$CAL"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python -m gencodoe.opf_se run --split test --days "$DAY" --estimators $EST --genco-variant "$VARIANT" --calibration "$CAL" --out "$OUT"
