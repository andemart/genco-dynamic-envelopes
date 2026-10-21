#!/bin/bash
# GENCO-OPF amendment 4, after the graphkit SE run (hydra_train.slurm with configs/genco_se_case33_lv.yaml):
# gate for GENCO-SE on 2,000 val-day states at f = 0.5, then calibration of WLS and GENCO-SE on the validation days.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_opf_se.slurm [mlruns/<exp>/<run>]
# If the gate fails (exit 5) the calibration still runs for WLS only, and the test array evaluates prior and WLS.
#SBATCH --job-name=opf-se-cal
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN_ARG=(); [ $# -ge 1 ] && RUN_ARG=(--genco-run "$1")
echo "commit=$HEAD (verified) node=$(hostname) run=${1:-newest}"
EST="prior wls"
if .venv/bin/python -m gencodoe.opf_se gate "${RUN_ARG[@]}" --out results/genco_se_case33_gate.json; then EST="prior wls genco"; else echo "GATE FAILED: GENCO-SE arm dropped"; fi
echo "estimators: $EST" | tee results/se_estimators.txt
.venv/bin/python -m gencodoe.opf_se calibrate --split val --days $(seq 80 99) --estimators $EST "${RUN_ARG[@]}" --out results/se_calibration.json
sha256sum results/se_calibration.json $(ls results/genco_se_case33_gate.json 2>/dev/null)
