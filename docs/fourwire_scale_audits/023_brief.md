Audit round 23 (read-only). HEAD is 07738d3. Tracked files equal HEAD and stay stable while you read, with one exception that is not part of this audit: scripts/paper_figures.py is modified in the working tree by unrelated work (ignore it). The untracked folder paper/ is also unrelated work in progress.

Since 61b38f3 (round 22: findings 1 and 3 resolved; one gap left in the comparison with the E4 records; not ready to submit E5):
- e5392c4: e4_comparison() in scripts/fourwire_scale_balanced_envelopes_e5.py and "Corrections to Amendment 12, second set (after the twenty-second audit)" in docs/fourwire_scale_protocol.md. See `git diff 61b38f3 e5392c4`.
- 756e9cd: results/fourwire_scale_e4_robustness.json (the specification table of Amendment 12 item 6, computed in a clean checkout of 61b38f3 after round 21 found item 6 ready) and the protocol section "Output of the specification table for E4".
- 07738d3: your round-22 files in docs/fourwire_scale_audits/, and a short protocol note on the authors of the four-wire dataset.

Nothing of E5 has run on a held-out day. Rules as before (read-only; never read labels/test or labels/calib; no fixture test suite; no cluster access; no envelope, scenario or sweep state on a held-out day; no E5 `run` without --dry, no E5 `report`). You may now recompute the specification table from the committed E4 records with your own code (it is committed and its output is under audit); do not run scripts/fourwire_scale_e4_robustness.py itself, since it writes a file. Try to refute, be concise.

A. The gap of round 22. Does e4_comparison now count every envelope that is positive in either record, with agreement only at an exactly equal level and three equal results, and gate on at most 0.5% not agreeing (besides the levels gate and the classification)? Repeat your case (every level moved by one representable float, feasible counts changed from 100 to 0, classifications preserved): refused? Is there any remaining way for base-mix records to pass without their results having been compared with E4, or for records that reproduce E4 exactly to be refused? Is corrections (second set) item 1 an exact description, and is item 2 a fair statement of why exact equality is expected?

B. Did e5392c4 change anything other than e4_comparison and its two constants' comments (the run path, the fresh check, the classification, record_problems, the job script, the robustness script)?

C. The specification table (756e9cd).
 1. Recompute from the 23 committed E4 records: the registered combination (21; networks 10 and 25 not unsafe), the nine one-change-at-a-time counts with their not-unsafe networks, and over the 162 combinations the distribution of counts (20: 4, 21: 124, 22: 28, 23: 6), which networks ever change (10, 18, 25), the combinations in which network 18 is not unsafe (4), network 25 unsafe (34), network 10 unsafe (6), the range without network 18 (20 to 22), and that no combination has a network without a positive envelope. Does results/fourwire_scale_e4_robustness.json agree with your recomputation, and does it carry the commit and hashes it should (code_commit 61b38f3, tracked_files_modified false, the E4 report's and scripts' hashes)?
 2. The output section: every number and every statement, including the "Reading" paragraph. Is anything overstated for an exploratory description made after the E4 outcome was known? Is the description of which rule values produce the counts of 20 and 23 exact?

D. The protocol note on the dataset's authors: consistent with data/feeder_topology/ENWL_CSIRO/README.md and its licence file?

E. Verdict: ready to submit the 72 E5 runs and the report from a worktree of 07738d3, or the blockers in order. Separately: is the output of the specification table accurate as it stands?
