Audit round 12 (read-only), very narrow. HEAD is the commit after bb89f92 (it only adds your round-11 brief and report under docs/fourwire_scale_audits/). Tracked files equal HEAD and stay stable while you read.

bb89f92 answers the two blockers of round 11. It changes only scripts/fourwire_scale_pgpre_stabilised.py (guard, fit, collect, one constant) and appends "Corrections to Amendment 10, second set" to docs/fourwire_scale_protocol.md. src/ is untouched; train() is unchanged. Nothing of Amendment 10 has been submitted.

Same rules: read-only; never read data_out/fourwire_scale/labels/test or /calib; no fixture test suite; no cluster. Do not re-audit what you cleared. In-memory probes are fine. Be brief.

1. Attempt limit: the count of earlier attempts and the call of fourwire_scale_train.claim_run_folder now run under one lock per fit (.<name>.attempts.lock). Can claim_run_folder still reach its HALT write from this script, including your sequence of round 11? Does a completed fit with two earlier attempts now return exit code 10? Is the lock always released, and what happens if a process dies while holding it?
2. Path guard: it now looks for runs/final and runs/dry in the path as given (os.path.abspath) and in the resolved path. Is your symlink example closed? Is the stated limit in the second corrections (item 2) accurate?
3. collect now requires the equivalence record to be the prescribed gate (GATE). Consistent with the job script?
4. The second set of corrections, items 1 to 6: any statement stronger than the code?

Verdict: ready to submit the equivalence job and the twelve fits, or the remaining blockers.
