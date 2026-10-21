#!/bin/bash
# Amendment 5 option 2, step 1: 30,000 case118_flex AC-OPF labels (Evgenii's config) and the nested splits.
#   sbatch --export=ALL,GENCODOE_COMMIT=<sha> scripts/hydra_amendment5_labels.slurm      (in $VSC_DATA/genco-case118)
#SBATCH --job-name=a5-labels
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
MAIN="$VSC_DATA/genco-dynamic-envelopes"; cd "$VSC_DATA/genco-case118"
module load Python/3.12.3-GCCcore-13.3.0
: "${GENCODOE_COMMIT:?GENCODOE_COMMIT must be set}"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$GENCODOE_COMMIT" ] || { echo "ABORT: HEAD $HEAD != GENCODOE_COMMIT $GENCODOE_COMMIT"; exit 2; }
git diff --quiet HEAD -- src scripts configs results grids || { echo "ABORT: tracked files modified relative to HEAD"; exit 3; }
echo "commit=$HEAD (verified) node=$(hostname)"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1
sed -e 's/^  num_processes: 8$/  num_processes: 16/' configs/datakit_opf_case118_flex.yaml > /tmp/a5_labels_$SLURM_JOB_ID.yaml
"$MAIN/.venv/bin/gridfm_datakit" generate /tmp/a5_labels_$SLURM_JOB_ID.yaml | grep -v "Processing scenarios" | tail -3
RAW=data_out/case118_flex/raw
"$MAIN/.venv/bin/python" scripts/amendment5_splits.py --raw "$RAW" --out data_out/case118_flex/splits
sha256sum "$RAW/args.log" data_out/case118_flex/splits/manifest.json
find "$RAW" -name "*.parquet" -type f | sort | xargs sha256sum | sha256sum | sed 's/-$/  (hash of all parquet hashes)/'
