#!/bin/bash
# Exact classical solvers (radial sweep, generic Z-bus) on a Hydra CPU node, same rows as the A100 benchmark.
#SBATCH --job-name=genco-cpu-baselines
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name|^CPU\(s\)"
.venv/bin/python -m gencodoe.benchmark --batch 1000 4000 16000 --repeats 3 --out data_out/eval/benchmark_hydra_zen4_classical.json
