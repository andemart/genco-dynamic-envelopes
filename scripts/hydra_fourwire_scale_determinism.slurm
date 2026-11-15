#!/bin/bash
# Four-wire scale study: is PG-pre training reproducible on the A100? Four short processes (32 steps each):
# two with default settings and two with PyTorch's deterministic algorithms. Train and val states only.
# Submit from an immutable worktree of the pinned commit:
#   sbatch --export=ALL,GENCODOE_COMMIT=<full sha>,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=<label folder> \
#       scripts/hydra_fourwire_scale_determinism.slurm
#SBATCH --job-name=fwscale-determinism
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${GENCODOE_WORKTREE:?GENCODOE_WORKTREE must be set}"
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # shared .venv must import this pinned worktree
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
: "${GENCODOE_LABEL_ROOT:?GENCODOE_LABEL_ROOT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "commit=$HEAD node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
OUT="data_out/fourwire_scale/determinism/job$SLURM_JOB_ID"
mkdir -p "$OUT"
for run in default_a default_b; do
  .venv/bin/python -m gencodoe.fourwire_scale_train determinism --label-root "$GENCODOE_LABEL_ROOT" --device cuda --out "$OUT/$run.json"
done
for run in deterministic_a deterministic_b; do
  .venv/bin/python -m gencodoe.fourwire_scale_train determinism --label-root "$GENCODOE_LABEL_ROOT" --device cuda --deterministic --out "$OUT/$run.json"
done
.venv/bin/python - "$OUT" "results/fourwire_scale_determinism_job$SLURM_JOB_ID.json" <<'PY'
import json, sys
from pathlib import Path
runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*.json"))}
def pair(a, b):
    a, b = runs[a], runs[b]
    if "error" in a or "error" in b:
        return {"error": a.get("error") or b.get("error")}
    d = [abs(x - y) for x, y in zip(a["step_losses"], b["step_losses"])]
    return {"same_initial_weights": a["initial_weights_sha256"] == b["initial_weights_sha256"],
            "same_final_weights": a["final_weights_sha256"] == b["final_weights_sha256"],
            "identical_step_losses": a["step_losses"] == b["step_losses"], "steps": len(d),
            "first_step_that_differs": next((i for i, x in enumerate(d) if x > 0), None), "max_abs_step_loss_difference": max(d)}
out = {"what": "two processes per setting, 32 PG-pre training steps on networks 10 and 17 (train/val only)",
       "default": pair("default_a", "default_b"), "deterministic_algorithms": pair("deterministic_a", "deterministic_b"),
       "default_vs_deterministic": pair("default_a", "deterministic_a"), "runs": runs}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("default", "deterministic_algorithms", "default_vs_deterministic")}, indent=1))
PY
