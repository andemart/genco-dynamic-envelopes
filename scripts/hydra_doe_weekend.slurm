#!/bin/bash
# Weekend analyses W1-W3 (docs/weekend_protocols.md; post-hoc, exploratory): one array task per test day,
# hourly, exact four-wire oracle only (no GENCO). Same scenarios/limits/threshold/verification as the annual run.
# Submit: sbatch --array=0-35 --time=HH:MM:SS scripts/hydra_doe_weekend.slurm <tag> <household_seed> <method> [flags...]
#   W1: <tag>=w1_placement  <seed>=202|303|404|505  b_opendss_exact
#   W2: <tag>=w2_three_phase <seed>=101|202          b_opendss_exact --three-phase-hp-ev
#   W3: <tag>=w3_spatial    <seed>=101               s_spatial_exact --spatial
#SBATCH --job-name=weekend-doe
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr
TAG=$1; HH=$2; METHOD=$3; shift 3
DAY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/annual_test_days.txt)
echo "day=$DAY tag=$TAG household_seed=$HH method=$METHOD flags=$* limits=$GENCODOE_VOLTAGE_LIMITS commit=${GENCODOE_COMMIT:-unknown}"
mkdir -p "data_out/doe/weekend/$TAG"
.venv/bin/python -m gencodoe.doe_day --day "$DAY" --steps-per-hour 1 --scenarios 200 --beta 0.05 --seed 101 \
  --household-seed "$HH" --verify-draws 1000 --unbalanced 200 --exact-four-wire --methods ac "$METHOD" "$@" \
  --calib data_out/datakit_v2_calib --out "data_out/doe/weekend/$TAG/${DAY}_${METHOD}_sph1_seed101_hh${HH}_uk_esqcr.json"
