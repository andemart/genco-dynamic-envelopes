#!/bin/bash
# Envelope-prediction labels (docs/envpred_protocol.md, amendment 2): exact B on train/val/calib dates.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-329 scripts/hydra_envpred_labels.slurm <household_seed>
#SBATCH --job-name=envpred-labels
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
HH=$1
DAY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/envpred_label_days.txt)
echo "day=$DAY household_seed=$HH limits=$GENCODOE_VOLTAGE_LIMITS commit=${GENCODOE_COMMIT:-unknown}"
mkdir -p data_out/doe/envpred_labels
.venv/bin/python -m gencodoe.doe_day --day "$DAY" --steps-per-hour 1 --scenarios 200 --beta 0.05 --seed 101 \
  --household-seed "$HH" --verify-draws 1000 --unbalanced 0 --exact-four-wire --any-split --methods ac b_opendss_exact \
  --calib data_out/datakit_v2_calib --out "data_out/doe/envpred_labels/${DAY}_b_opendss_exact_sph1_seed101_hh${HH}_uk_esqcr.json"
