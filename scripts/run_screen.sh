#!/bin/bash
# Train each config with graphkit (LV variant registered), then evaluate its best
# checkpoint in physical units on the val-day states of the same split folder.
# Usage: scripts/run_screen.sh configs/experiments/q_*.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p data_out/eval data_out/logs
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  splits=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open('$cfg'))['data']['split_from_existing_files'])")
  data=$(dirname "$splits")   # the dataset dir holds enwl1_fluvius/raw and the split folders
  start=$(date +%s)
  .venv/bin/python -m gencodoe.graphkit_cli train --config "$cfg" --data_path "$data" \
    --log_dir mlruns --exp_name "$name" > "data_out/logs/$name.log" 2>&1
  run=$(ls -td mlruns/*/*/artifacts/model/best_model_state_dict.pt | head -1 | xargs dirname | xargs dirname | xargs dirname)
  echo "$name: trained in $(( $(date +%s) - start )) s, run $run"
  .venv/bin/python -m gencodoe.evaluate --data "$data" --scenario-file "$splits/val.pt" \
    --run "$run" --config "$cfg" --device cpu --out "data_out/eval/val_$name.json" 2>&1 | grep -v -i warn
done
