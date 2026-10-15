#!/bin/bash
# Stage G on Hydra: GENCO throughput on an A100 (model-only and end to end) vs the
# radial sweep, pandapower Newton-Raphson and OpenDSS (unbalanced) on the node's CPU.
# Submit: sbatch scripts/hydra_benchmark.slurm <run_dir> <config>
#SBATCH --job-name=genco-bench
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
export OMP_NUM_THREADS=1   # the radial sweep and generic solvers on one core, as on the laptop
.venv/bin/python -m gencodoe.benchmark --run "$1" --config "$2" --device cpu --batch 1000 --model-batch 1000 \
  --repeats 2 --generic-rows 100 --out data_out/eval/benchmark_hydra_cpu.json
unset OMP_NUM_THREADS
for mb in 1024 4096; do
  for tf in "" "--tf32"; do
    .venv/bin/python -m gencodoe.benchmark --run "$1" --config "$2" --device cuda --batch 1000 4000 16000 \
      --model-batch $mb --repeats 3 $tf --out "data_out/eval/benchmark_a100_mb${mb}${tf:+_tf32}.json" || true
  done
done
