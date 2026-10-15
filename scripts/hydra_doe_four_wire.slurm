#!/bin/bash
# Certified four-wire DOE day on a Hydra CPU node: option B (exact OpenDSS search) or
# option C (GENCO + four-wire conformal margins, OpenDSS fallback), with balanced AC as reference.
# Submit: sbatch scripts/hydra_doe_four_wire.slurm <method: b_opendss_exact | c_genco_4w_opendss> <day>
#SBATCH --job-name=genco-doe-4w
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
lscpu | grep -E "Model name"
.venv/bin/python -m gencodoe.doe_day --day "$2" --steps-per-hour 1 --scenarios 200 --beta 0.05 \
  --verify-draws 1000 --unbalanced 200 --four-wire --methods ac "$1" \
  --run mlruns/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee --config configs/genco_pf_enwl1_lv_prior_v2.yaml \
  --calib data_out/datakit_v2_calib --out "data_out/doe/${2}_certified_4w_$1.json"
