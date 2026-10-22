#!/bin/bash
# Amendment 5 option 2, step 3: evaluate the six runs (and the reference) on the first 1,000 test-split scenarios.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_amendment5_eval.slurm      (in $VSC_DATA/genco-case118)
#SBATCH --job-name=a5-eval
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
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
RUNS=()
for exp in a5_finetune_N100 a5_scratch_N100 a5_finetune_N1000 a5_scratch_N1000 a5_finetune_N10000 a5_scratch_N10000; do
  d=""
  for m in mlruns/*/meta.yaml; do grep -qx "name: $exp" "$m" && d=$(dirname "$m"); done
  [ -n "$d" ] || { echo "ABORT: no mlflow experiment $exp"; exit 4; }
  r=$(ls -td "$d"/*/artifacts/model/best_model_state_dict.pt | head -1 | xargs dirname | xargs dirname | xargs dirname)
  echo "$exp -> $r"; sha256sum "$r/artifacts/model/best_model_state_dict.pt"
  RUNS+=("$exp=$r")
done
echo "commit=$HEAD (verified) node=$(hostname)"
"$PY" -m gencodoe.opf_transfer --raw data_out/case118_flex/raw --case grids/case118_flex.m --splits data_out/case118_flex/splits \
  --runs "${RUNS[@]}" --n 1000 --out results/genco_opf_amendment5_transfer.json
sha256sum results/genco_opf_amendment5_transfer.json
