Audit round 22 (read-only). HEAD is 61b38f3. Tracked files equal HEAD and stay stable while you read, with one exception that is not part of this audit: scripts/paper_figures.py is modified in the working tree by unrelated work (ignore it). The untracked folder paper/ is also unrelated work in progress.

Since 8e46433 (round 21: not ready to submit E5; two code gaps and one disclosure gap; item 6 ready):
- 3bac869 answers round 21: scripts/fourwire_scale_balanced_envelopes_e5.py (check_reduction, scalar_power_flow, the positive_sequence_check field of a record and its validation in record_problems; e4_comparison and its use in report) and "Corrections to Amendment 12 (after the twenty-first audit)" at the end of docs/fourwire_scale_protocol.md (4 items). See `git diff 8e46433 3bac869`.
- 61b38f3 adds your round-21 brief and review to docs/fourwire_scale_audits/.

Nothing of E5 has run on a held-out day, and the table of item 6 has not been computed. Rules as in round 21 (read-only; never read labels/test or labels/calib; no fixture test suite; no cluster access; no envelope, scenario or sweep state on a held-out day; no `run` without --dry, no `report`; do not compute the table of item 6 from the E4 records). In-memory probes on validation days and on fabricated records are fine. Try to refute, be concise.

A. Finding 1 of round 21 (the positive-sequence check).
 1. With the corrected run(), is there still a path to a published record that the report refuses because of the check, or to a published record without a valid check? A day without a usable state now raises before publishing: confirm, and confirm that nothing is left behind except the attempt file.
 2. The 16 states (8 design scenarios and the same at one tenth of their power), the usability rule (the sweep converged and the scalar power flow's last update below 1e-12 kV within 500 iterations): sound? Can the scalar convergence rule exclude states on which the sweep converged, so often that a correct run stops? Can a NaN or a non-converged scalar state reach the comparison?
 3. record_problems: do 0, -1, 0.5, "none", 17, true and null as a day's state count, a list of the wrong length, a non-dict check, and NaN differences now fail? Can a correct record fail?
 4. Corrections item 1: is every statement right, including "equal in magnitude" and what the check covers and does not cover?

B. Finding 2 (the comparison with the E4 records).
 1. Does e4_comparison now compare all three results of every envelope with an equal positive level, list every interval and model with any difference, and gate on levels, on results and on the classification of all 24 networks plus the list of counted networks? Repeat your two fabricated cases (all feasible counts set to 0 with levels unchanged; decisions that differ changed from 0 to 2 with both counts at 100).
 2. Is the gate as corrections item 2 states it? Is there any remaining way for the recomputed E4 cell to disagree materially with E4 and the report still be written? Can a set of base-mix records that reproduces E4 exactly be refused?

C. Finding 3 (corrections item 3): is the list of differences from the ENWL1 balanced model now complete and accurately worded (thermal limits, voltage limits, transformer, scenario draws)?

D. Did 3bac869 change anything else: the run path of the three models, the fresh check, the classification, the dry/final isolation, the job script, scripts/fourwire_scale_e4_robustness.py (must be unchanged)? Any statement of the corrections that the code does not implement, or code behaviour the text does not state?

E. Verdict: ready to submit the 72 E5 runs and the report from a worktree of 61b38f3, or the blockers in order.
