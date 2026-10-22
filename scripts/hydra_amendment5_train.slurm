#!/bin/bash
# Amendment 5 option 2, step 2: six runs on one A100 each: array task -> (arm, N) in
# 0 finetune_N100, 1 scratch_N100, 2 finetune_N1000, 3 scratch_N1000, 4 finetune_N10000, 5 scratch_N10000.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> --array=0-5 scripts/hydra_amendment5_train.slurm
#   seed replicates: --export=ALL,GENCODOE_COMMIT=<sha>,SEED=1 (experiment names get _s1)
#SBATCH --job-name=a5-train
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
MAIN="$VSC_DATA/genco-dynamic-envelopes"; cd "$VSC_DATA/genco-case118"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts configs results grids || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
ARMS=(finetune scratch finetune scratch finetune scratch); NS=(100 100 1000 1000 10000 10000)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; N=${NS[$SLURM_ARRAY_TASK_ID]}; CFG=configs/amendment5/${ARM}_N${N}.yaml; EXP=a5_${ARM}_N${N}
SEED=${SEED:-0}                                  # seed replicates (exploratory addendum): same config, seed line changed
if [ "$SEED" != 0 ]; then
  sed -e "s/^seed: 0$/seed: $SEED/" "$CFG" > /tmp/a5_${ARM}_N${N}_s${SEED}_$SLURM_JOB_ID.yaml
  CFG=/tmp/a5_${ARM}_N${N}_s${SEED}_$SLURM_JOB_ID.yaml; EXP=${EXP}_s$SEED
  grep -qx "seed: $SEED" "$CFG" || { echo "ABORT: seed not set in $CFG"; exit 5; }
fi
HUB=data_out/hub/genco-opf-datakit/case118_ieee/base/seed0
echo "commit=$HEAD (verified) arm=$ARM N=$N seed=$SEED cfg=$CFG node=$(hostname)"
sha256sum data_out/case118_flex/splits/N$N/train.pt $HUB/last.pt
export PYTHONPATH="$PWD/src" MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
PY="$MAIN/.venv/bin/python"
if [ "$ARM" = finetune ]; then
  "$PY" -m gencodoe.amendment5_util convert --hub "$HUB" --out data_out/hub/pretrained_state_dict.pt
  "$PY" -m gridfm_graphkit finetune --config "$CFG" --model_path data_out/hub/pretrained_state_dict.pt --data_path data_out --log_dir mlruns --exp_name "$EXP" --num_workers "$SLURM_CPUS_PER_TASK"
else
  "$PY" -m gridfm_graphkit train --config "$CFG" --data_path data_out --log_dir mlruns --exp_name "$EXP" --num_workers "$SLURM_CPUS_PER_TASK"
fi
