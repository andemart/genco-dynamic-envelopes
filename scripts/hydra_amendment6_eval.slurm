#!/bin/bash
# Amendment 6: confirmation at N = 100 on the unopened sorted test positions 1000-2999 (2,000 scenarios), one task per
# training seed 0-4: fine-tuned and scratch N = 100 of that seed, same gate as amendment 5.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-4 scripts/hydra_amendment6_eval.slurm      (in $VSC_DATA/genco-case118)
# then: python scripts/amendment6_analysis.py
#SBATCH --job-name=a6-eval
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
MAIN="$VSC_DATA/genco-dynamic-envelopes"; cd "$VSC_DATA/genco-case118"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts configs results grids || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
PY="$MAIN/.venv/bin/python"
SEED=$SLURM_ARRAY_TASK_ID; SUFFIX=""; [ "$SEED" = 0 ] || SUFFIX="_s$SEED"
RUNS=()
for exp in a5_finetune_N100 a5_scratch_N100; do
  d=""
  for m in mlruns/*/meta.yaml; do grep -qx "name: $exp$SUFFIX" "$m" && d=$(dirname "$m"); done
  [ -n "$d" ] || { echo "ABORT: no mlflow experiment $exp$SUFFIX"; exit 4; }
  r=$(ls -td "$d"/*/artifacts/model/best_model_state_dict.pt | head -1 | xargs dirname | xargs dirname | xargs dirname)
  echo "$exp$SUFFIX -> $r"; sha256sum "$r/artifacts/model/best_model_state_dict.pt"
  RUNS+=("$exp=$r")
done
echo "commit=$HEAD (verified) seed=$SEED node=$(hostname)"
OUT=results/genco_opf_amendment6_seed$SEED.json
"$PY" -m gencodoe.opf_transfer --raw data_out/case118_flex/raw --case grids/case118_flex.m --splits data_out/case118_flex/splits \
  --runs "${RUNS[@]}" --offset 1000 --n 2000 --out "$OUT"
sha256sum "$OUT"
