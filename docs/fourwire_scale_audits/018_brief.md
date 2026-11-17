Audit round 18 (read-only). HEAD is 0f5cb80. Tracked files equal HEAD and stay stable while you read.

Since 4d74a1f (round 17: E4 "not ready", five points left; stabilised output numerically sound, one sentence to correct):
- 78e9d1d answers round 17. scripts/fourwire_scale_balanced_envelopes.py: the report now checks the ledger (ledger_problems) and the consistency of the differing decisions with the two counts (record_problems, _count, _number); the run command is unchanged (check that: `git diff 4d74a1f 78e9d1d -- scripts/fourwire_scale_balanced_envelopes.py`). scripts/hydra_fourwire_scale_e4.slurm: the header states the two recovery cases. docs/fourwire_scale_protocol.md: "Corrections to Amendment 11, second set (after the seventeenth audit)" (5 items) and "Correction to the output of the stabilised-recipe analysis (after the seventeenth audit)".
- 0f5cb80 adds your round-17 brief and review to docs/fourwire_scale_audits/.

Nothing of E4 has run on a held-out day. Rules as in rounds 16 and 17: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day (day index d with d % 5 == 4 and (d // 5) % 2 == 1); do not run the script without --dry. In-memory probes on validation days and on fabricated records are fine. Try to refute, be concise.

A. src/, results/fourwire_scale_freeze.json and scripts/hydra_fourwire_scale_final.slurm: still byte-identical to a4068ef?

B. The five points of round 17, each resolved or not:
 1. Ledger membership and content: can a record still pass whose attempt is missing, belongs to another network or run, or lists earlier attempts that are not in the ledger? Can a correctly produced record and ledger (one attempt, or a rerun after a task that left no record) be refused?
 2. Differing decisions: with a = feasible by the sweep, b = feasible by OpenDSS (both counts over 200) and d = decisions that differ, the report requires x = (d + a - b)/2 and y = (d - a + b)/2 to be whole, 0 <= x <= min(a, 200 - b), 0 <= y <= min(b, 200 - a). Is that the exact set of triples a real run can produce? Try fabricated triples on both sides of it, and the type cases (0.5, true, a string).
 3. Second set, item 3 (the extra voltage checks of the balanced model): is the corrected statement right, including "it can only make the balanced model stricter ... so it cannot be the reason for a balanced envelope that is too large"?
 4. Second set, item 4 (where the four-wire rule admits no envelope): is the corrected statement right and complete?
 5. Second set, item 5 and the job-script header: are the two recovery cases right, given that run() refuses when the record exists before it claims an attempt?

C. Did 78e9d1d introduce anything: a changed run path, a check that the text does not state or a statement the code does not implement, a refusal of a valid complete set of records?

D. The corrected sentence of the stabilised output: does it now say only what the records show?

E. Verdict for E4: ready to submit the 24 runs and the report on the cluster from a worktree of 0f5cb80, or the blockers in order.
