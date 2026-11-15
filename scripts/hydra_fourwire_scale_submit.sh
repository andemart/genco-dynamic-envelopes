#!/bin/bash
# Four-wire scale study: submit the final runs (docs/fourwire_scale_protocol.md, Amendments 7 to 9).
# Run once, on Hydra, from the worktree created for the commit that holds results/fourwire_scale_freeze.json,
# after `python -m gencodoe.fourwire_scale_eval networks`:
#   scripts/hydra_fourwire_scale_submit.sh <label folder> [<job ids the fits must wait for, colon-separated>]
# The optional job ids are those of the A100 dry runs of the same commit (scripts/hydra_fourwire_scale_dryrun.slurm):
# no final fit then starts unless the whole chain has run on the A100 with train and val labels.
# Order (Amendment 8, item 14): fits; evaluation of all 24 networks as one array, after every fit;
# timing and envelopes of the six E2 networks, after the evaluation.
set -euo pipefail
L=${1:?usage: hydra_fourwire_scale_submit.sh <label folder> [<job ids to wait for>]}
AFTER=${2:-}
C=$(git rev-parse HEAD)
git diff --quiet HEAD -- src scripts results || { echo "ABORT: tracked source, scripts or results modified"; exit 3; }
git ls-files --error-unmatch results/fourwire_scale_freeze.json > /dev/null || { echo "ABORT: no committed freeze file"; exit 4; }
mkdir -p logs
E="--export=ALL,GENCODOE_COMMIT=$C,GENCODOE_WORKTREE=$PWD,GENCODOE_LABEL_ROOT=$L"
S=scripts/hydra_fourwire_scale_final.slurm
NETS=1-12,14-25
E2=25,10,1,7,2,17
DEP=${AFTER:+--dependency=afterok:$AFTER}
pre=$(sbatch --parsable "$E,GENCODOE_STAGE=pgpre" --array=0-11 --time=12:00:00 $DEP $S)
tgt=$(sbatch --parsable "$E,GENCODOE_STAGE=target" --array=$NETS --time=06:00:00 $DEP $S)
ft=$(sbatch --parsable "$E,GENCODOE_STAGE=ft" --array=$NETS --time=03:00:00 --dependency=afterok:$pre $S)
evl=$(sbatch --parsable "$E,GENCODOE_STAGE=evaluate" --array=$NETS --time=01:30:00 --dependency=afterok:$tgt:$ft $S)
tim=$(sbatch --parsable "$E,GENCODOE_STAGE=timing" --array=$E2 --time=08:00:00 --cpus-per-task=16 --dependency=afterok:$evl $S)
env=$(sbatch --parsable "$E,GENCODOE_STAGE=envelopes" --array=$E2 --time=04:00:00 --dependency=afterok:$evl $S)
echo "commit=$C after=${AFTER:-none} pgpre=$pre target=$tgt ft=$ft evaluate=$evl timing=$tim envelopes=$env" | tee -a logs/final_submission.txt
