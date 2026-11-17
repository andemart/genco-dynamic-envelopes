Audit round 17 (read-only). HEAD is 4d74a1f. Tracked files equal HEAD and stay stable while you read.

Since 9d2af25 (round 16, "not ready to submit"):
- 8bb4061 answers round 16: scripts/fourwire_scale_balanced_envelopes.py and scripts/hydra_fourwire_scale_e4.slurm are corrected, and "Corrections to Amendment 11 (after the sixteenth audit)" is appended to docs/fourwire_scale_protocol.md (13 items).
- 45ab6c9 adds your round-16 brief and review to docs/fourwire_scale_audits/.
- 4d74a1f adds the output of the stabilised-recipe analysis of Amendment 10: results/fourwire_scale_pgpre_stabilised.json, results/fourwire_scale_pgpre_stabilised_equivalence.json, accounting lines appended to results/fourwire_scale_hydra_job_evidence.txt, and the section "Output of the stabilised-recipe analysis of Amendment 10" at the end of the protocol.

Nothing of E4 has run on a held-out day. Dry runs on validation days were made as the corrections disclose (their records are not in the repository; treat their figures as claims).

Rules as in round 16: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day (day index d with d % 5 == 4 and (d // 5) % 2 == 1); do not run the script without --dry. In-memory probes on validation days and on fabricated records are fine. Try to refute, derive things yourself, be concise.

Part 1. E4 after the corrections.

A. src/, results/fourwire_scale_freeze.json and scripts/hydra_fourwire_scale_final.slurm: still byte-identical to a4068ef?

B. Each round-16 finding, resolved or not, with what is left:
 1. dry mode reaching a held-out day (hours, the per-step split check);
 2. output paths (no command takes one; where each mode writes; can anything still be written into a run folder of the study or replace a file?);
 3. publishing (publish(): partial file, link, no replacement; what remains after a death at each point; the attempt ledger claim_attempt()/attempts() and how the report lists attempts);
 4. record validation in report() (record_problems(): the exact interval list, settings, required results for positive and zero envelopes, provenance keys). Try fabricated records that should be refused and are not;
 5. the 16 cores (run and report);
 6. the edge rules (no positive balanced envelope on the OpenDSS days; engines compared for every network; per-direction intervals and OpenDSS figures; the conditional bootstrap);
 7. the E2 comparison (tolerance kept, manifest binding via results/fourwire_scale_final_outputs_sha256.txt, the list of differences);
 8. the generator check on every hour;
 9-12. the text: what differs between the two models, what fresh draws are, the completed disclosure (including the sweep and OpenDSS differences already seen and the label-derived Thevenin source), the claim wording;
 13. the start check and the recovery of a failed task in the job script (--kill-on-invalid-dep, resubmission).

C. Anything the corrections introduced: a new way to write outside the three places, a rule in the text that the code does not implement (or the reverse), a case in which a valid complete set of records is refused, or a statement in the 13 items that is stronger than the code or the evidence.

D. Verdict for E4: ready to submit the 24 runs and the report, or the blockers in order.

Part 2. Output of the stabilised-recipe analysis (exploratory; Amendment 10 fixed how it is read).

E. Recompute from results/fourwire_scale_pgpre_stabilised.json every figure of the output section: the table of twelve frozen and stabilised losses with selected epoch and epochs run, the weak threshold and the statement that no stabilised fit is weak, the ratios (median over twelve; over the ten fits not weak under the frozen recipe; how many are lower), the two fits that are worse and stopped early with their per-network losses, the per-network statement, the clipping shares and gradient-norm medians, the attempts and numerical failures. Check the run times against the accounting lines appended to results/fourwire_scale_hydra_job_evidence.txt (35 h 43 min for the twelve fits). Do the hashes quoted in the section match the files?

F. Is the reading within what Amendment 10 allows ("How it is read", "What it is not")? Is anything overstated, in particular "consistent with a sensitivity of the frozen optimisation", the explanation offered for the two early stops (patience with the lower learning rate), and the closing paragraph on what it does not show? Is anything that should be said missing?

G. Does results/fourwire_scale_pgpre_stabilised_equivalence.json show what the section says about the gate (equal in every item, 1,512 steps, the frozen environment)?

H. Verdict for the output section: sound as written, or the corrections in order.
