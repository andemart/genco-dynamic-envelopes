#!/bin/bash
# Unbalanced GENCO: the final fit of one model (amendment 1). Submit once per model.
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=4
: "${GENCODOE_COMMIT:?}"
[ "$(git rev-parse HEAD)" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD mismatch"; exit 2; }
git diff --quiet HEAD -- src scripts || { echo "ABORT: tracked files modified"; exit 3; }
if [ "$1" = eval ]; then
  .venv/bin/python -m gencodoe.fourwire_eval --device cuda --tag final --out results/fourwire_eval_test.json
else
  .venv/bin/python -m gencodoe.fourwire_train fit --model "$1" --epochs 60 --tag final --device cuda
fi
