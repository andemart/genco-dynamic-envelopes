#!/bin/bash
# Pre-registered annual extension (docs/preregistration_annual_extension.md):
# one array task per test day, hourly resolution, one method per array.
# Submit: sbatch --array=0-35 scripts/hydra_doe_annual.slurm <method: b_opendss_exact | c_genco_4w_opendss>
#SBATCH --job-name=annual-doe
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr
DAY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/annual_test_days.txt)
echo "day=$DAY method=$1 limits=$GENCODOE_VOLTAGE_LIMITS commit=${GENCODOE_COMMIT:-unknown}"
mkdir -p data_out/doe/annual
.venv/bin/python -m gencodoe.doe_day --day "$DAY" --steps-per-hour 1 --scenarios 200 --beta 0.05 --seed 101 \
  --verify-draws 1000 --unbalanced 200 --four-wire --methods ac "$1" \
  --run mlruns/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee --config configs/genco_pf_enwl1_lv_prior_v2.yaml \
  --calib data_out/datakit_v2_calib --out "data_out/doe/annual/${DAY}_annual_4w_$1_sph1_seed101_uk_esqcr.json"
