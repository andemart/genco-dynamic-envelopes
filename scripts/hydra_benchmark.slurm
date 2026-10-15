#!/bin/bash
# Stage G on Hydra: end-to-end GENCO throughput on an A100 vs the CPU AC sweep.
# Submit: sbatch scripts/hydra_benchmark.slurm <run_dir> <config>
#SBATCH --job-name=genco-bench
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
for mb in 256 1024 4096; do
  .venv/bin/python -m gencodoe.benchmark --run "$1" --config "$2" --device cuda --batch 1000 4000 16000 \
    --model-batch $mb --repeats 3 --out "data_out/eval/benchmark_a100_mb$mb.json"
done
