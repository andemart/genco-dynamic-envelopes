Audit round 10 (read-only). HEAD is bd1cc9c. Tracked files equal HEAD and stay stable while you read.

Since e34d242 (the corrected diagnostic you asked for in round 9):
- 511577a adds docs/fourwire_scale_audits/ (your briefs and reports, copied with local paths made relative). Out of scope unless something there is wrong.
- 3c27703 adds Amendment 10 at the end of docs/fourwire_scale_protocol.md, scripts/fourwire_scale_pgpre_stabilised.py and scripts/hydra_fourwire_scale_stabilised.slurm. This is the stabilised-recipe robustness analysis you proposed, prepared at the author's request.
- bd1cc9c corrects one sentence (the label reader also serves calib).

State of the study: all 276 final fits ended with valid records. Evaluation (24 tasks) and envelopes (6) completed on the cluster; timing has 4 of 6 done and 2 running. No evaluation record, timing record or report has been opened. The corrected diagnostic jobs of e34d242 are queued and have produced no output. The first-version diagnostic jobs ran before the correction; their output was not read or copied. Nothing of Amendment 10 has been submitted.

The twelve frozen PG-pre losses in the amendment's table come from the records on the cluster, which are not in the repository yet; you cannot verify them here. Treat them as claims.

Rules as before: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access. Small in-memory or CPU probes on train/val labels are fine. Try to refute, derive numbers yourself, be concise.

A. Source identity. Is src/ byte-identical to a4068ef? Are results/fourwire_scale_freeze.json and scripts/hydra_fourwire_scale_final.slurm untouched, so the freeze still holds?

B. Equivalence of the loop. Is train() in scripts/fourwire_scale_pgpre_stabilised.py with RECIPES["frozen"] the same computation as fourwire_scale_train.fit for a PG-pre fit: random-number consumption, order of operations, loss expression, scheduler construction and calls, skip of batches with no converged state, selection, patience, checkpoint writing? Does measuring the gradient norm in frozen mode change anything? Is the `equivalence` command a sufficient gate at its reduced size (fold 2, seed 1, 650 train and 256 val states per network, 4 epochs)? Name what it cannot exercise.

C. The four switches. (1) Is the learning-rate schedule what Amendment 10 states: linear per step from zero to the peak at the last step of epoch 5, the peak during epoch 6, then CosineAnnealingLR(T_max=55) stepped once per epoch? Check the scheduler's recursive formula against the manual changes of the group learning rate. (2) Is clipping applied after the finiteness check and before the optimiser step, and is the recorded norm the norm before clipping? (3) Does the round-robin order visit every training state once per epoch, with one batch per pool network per round? What does filtering batches without converged states after the ordering do to a round? (4) Anything in these that is not what the amendment says?

D. Isolation. Can the script read calib or test labels? Can it write into data_out/fourwire_scale/runs/final or runs/dry, or replace an existing file? Does `fit` at full size enforce the checks of a final command? Can a failure of a stabilised fit disturb the frozen study (for example a HALT file, a claim, a lock, a results/ file the final reports read)? The fits run in a separate worktree at their own commit.

E. collect. Are its checks enough to prevent mixing recipes, dry runs, other freezes or other states/scalers? Is the weak threshold what the amendment says (five times the median of the twelve frozen losses, about 0.106)? Any quantity it reports that could mislead?

F. scripts/hydra_fourwire_scale_stabilised.slurm. Array mapping, the afterok dependency on the equivalence job, exit codes, anything that could let the fits start after a failed equivalence check.

G. Amendment 10 and its correction. Any statement stronger than the code or the evidence? Is the planned reading sound (threshold, what may be said in each case, no significance test)? Where the amendment fixes a choice your proposal left open (5 warm-up epochs, round robin and not accumulation, no EMA) or reports a property (clipping active on more than 90% of steps in a small dry run), do you consider anything material enough to change before running?

H. Verdict: ready to submit the equivalence job and the twelve fits, or the blockers in order.
