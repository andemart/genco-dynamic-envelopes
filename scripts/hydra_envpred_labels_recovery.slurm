#!/bin/bash
# Corrected envelope-prediction labels (docs/envpred_protocol.md, amendment 3c): exact B on train/val/calib dates.
# Aborts unless GENCODOE_COMMIT == git HEAD, tracked files are unmodified and every pinned hash matches.
# Full:  sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-329 scripts/hydra_envpred_labels_recovery.slurm full <hh>
# Smoke: sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-3   scripts/hydra_envpred_labels_recovery.slurm smoke
#SBATCH --job-name=envpred-labels-fix
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=1
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts configs data/feeder_topology results/envpred_label_days.txt results/envpred_split_v2.json \
  || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
.venv/bin/python -m gencodoe.pin verify results/envpred_label_recovery_manifest.json || { echo "ABORT: pin"; exit 4; }
MODE=$1
if [ "$MODE" = smoke ]; then
  read -r DAY HH <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/envpred_label_recovery_smoke_blocks.txt)"
  OUTDIR=data_out/doe/envpred_labels_recovery_smoke
else
  HH=$2
  DAY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/envpred_label_days.txt)
  OUTDIR=data_out/doe/envpred_labels
fi
echo "mode=$MODE day=$DAY household_seed=$HH head=$HEAD limits=$GENCODOE_VOLTAGE_LIMITS node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p "$OUTDIR"
.venv/bin/python -m gencodoe.doe_day --day "$DAY" --steps-per-hour 1 --scenarios 200 --beta 0.05 --seed 101 \
  --household-seed "$HH" --verify-draws 1000 --unbalanced 0 --exact-four-wire --any-split --methods ac b_opendss_exact \
  --calib data_out/datakit_v2_calib --out "$OUTDIR/${DAY}_b_opendss_exact_sph1_seed101_hh${HH}_uk_esqcr.json"
