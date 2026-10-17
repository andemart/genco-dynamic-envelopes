#!/bin/bash
# Envelope prediction v3: the replacement valid head-fitting run (amendment 3c; one invalid fit 13338884 is quarantined) (four frozen arms x two directions), Zen4 CPU node.
# Verifies both integrity reports and every input hash first; stops on any mismatch.
# Submit: sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_envpred3_fit.slurm
#SBATCH --job-name=envpred3-fit
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr OMP_NUM_THREADS=8
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
echo "commit=$HEAD (verified)"
.venv/bin/python -m gencodoe.envpred3 fit-job --commit "$GENCODOE_COMMIT" --expect-code results/envpred3_fit_code_pin_valid.json
