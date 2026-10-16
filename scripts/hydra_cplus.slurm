#!/bin/bash
# Formal C+ run (docs/cplus_protocol.md + amendment 1): one array task per stored C run.
# Submit: sbatch --array=0-37 scripts/hydra_cplus.slurm
#SBATCH --job-name=cplus-gate
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$VSC_DATA/genco-dynamic-envelopes"
module load Python/3.12.3-GCCcore-13.3.0
export GENCODOE_VOLTAGE_LIMITS=uk_esqcr
SRC=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/cplus_inputs.txt)
case "$SRC" in
  data_out/doe/annual/*) OUT=data_out/doe/cplus_annual; BDIR=data_out/doe/annual ;;
  *)                     OUT=data_out/doe/cplus_uk;     BDIR=data_out/doe ;;
esac
echo "src=$SRC out=$OUT bdir=$BDIR limits=$GENCODOE_VOLTAGE_LIMITS commit=${GENCODOE_COMMIT:-unknown} node=$(hostname)"
sha256sum "$SRC" "$BDIR/$(basename "${SRC/c_genco_4w_opendss/b_opendss_exact}")"
.venv/bin/python -m gencodoe.cplus --runs "$SRC" --b-dir "$BDIR" --out-dir "$OUT"
