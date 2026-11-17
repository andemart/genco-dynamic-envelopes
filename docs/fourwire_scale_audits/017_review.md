**Part 1 — E4**

**A. Pass.** `src/`, the freeze JSON and the original final Slurm script remain byte-identical to `a4068ef`.

**B. Round-16 findings**

1. **Resolved.** Hours are restricted to 1–24; every generated step is checked against the selected day and split. [Script:201](scripts/fourwire_scale_balanced_envelopes.py:201), [253](scripts/fourwire_scale_balanced_envelopes.py:253).

2. **Resolved for the commands.** No output-path arguments remain; outputs use the three prescribed locations and existing records are refused. [205](scripts/fourwire_scale_balanced_envelopes.py:205), [518](scripts/fourwire_scale_balanced_envelopes.py:518).

3. **Partly resolved.** Publication is complete and exclusive: death before linking leaves only a partial; death after linking leaves a complete record, possibly alongside its partial. However, the report never checks that the named attempt exists or validates its contents. A record naming a nonexistent attempt passes. [134](scripts/fourwire_scale_balanced_envelopes.py:134), [416](scripts/fourwire_scale_balanced_envelopes.py:416), [493](scripts/fourwire_scale_balanced_envelopes.py:493).

4. **Partly resolved.** Exact intervals, settings, provenance and mandatory results are checked. But fabricated records with **0.5 differing decisions**, or **sweep share 1, OpenDSS share 0, zero differing decisions**, pass. Require integral disagreement counts consistent with the two marginal counts. [434](scripts/fourwire_scale_balanced_envelopes.py:434).

5. **Resolved.** Final runs require 16 allocated cores and 16 OpenDSS workers; report validation requires the prescribed worker count. [203](scripts/fourwire_scale_balanced_envelopes.py:203), [244](scripts/fourwire_scale_balanced_envelopes.py:244), [419](scripts/fourwire_scale_balanced_envelopes.py:419).

6. **Resolved.** The unknown-engine policy, comparisons across all networks, directional reporting and conditional bootstrap now match the text. [352](scripts/fourwire_scale_balanced_envelopes.py:352), [389](scripts/fourwire_scale_balanced_envelopes.py:389), [504](scripts/fourwire_scale_balanced_envelopes.py:504).

7. **Resolved under the prescribed Slurm execution.** E2 records are checked against the manifest; every difference is listed. The retained tolerance permits **17/3,456** comparisons beyond tolerance. [447](scripts/fourwire_scale_balanced_envelopes.py:447).

8. **Resolved.** Generator equality is checked every hour. [249](scripts/fourwire_scale_balanced_envelopes.py:249).

9. **Partly resolved.** The changed voltage-monitoring domain is disclosed. “This adds little” remains unsupported: equal phase powers do not guarantee nearly equal phase voltages in this four-wire network. [Protocol:1061](docs/fourwire_scale_protocol.md:1061).

10. **Resolved.** Fresh draws are accurately described as conditional resampling. [Protocol:1062](docs/fourwire_scale_protocol.md:1062).

11. **Resolved as disclosure.** Prior solver discrepancies and the label-derived Thevenin source are stated. The unavailable dry-run figures remain unverified claims. [Protocol:1063](docs/fourwire_scale_protocol.md:1063).

12. **Partly resolved.** The restricted replication claim is appropriate, but **a zero four-wire envelope does not imply baseline infeasibility or failure of a positive balanced envelope’s fresh check**. Search truncation, different draws and the design/check thresholds can produce otherwise. A fabricated zero-reference/positive-balanced interval with fresh share 1 passes the check. Correct the categorical explanation. [Protocol:1070](docs/fourwire_scale_protocol.md:1070).

13. **Partly resolved.** Initial provenance and dependency cancellation are fixed. Recovery text misses death **after publication**: rerunning then fails because the record exists. A valid published record should be retained and the report resubmitted; rerun only when no record exists. [Script:207](scripts/fourwire_scale_balanced_envelopes.py:207), [Protocol:1072](docs/fourwire_scale_protocol.md:1072).

**C–D. Not ready to submit E4.** Remaining blockers, in order: validate ledger membership/content and disagreement counts; correct the zero-envelope explanation and unsupported monitoring assertion; document recovery after successful publication followed by job failure. I found no additional rejection of a correctly generated complete cohort.

**Part 2 — stabilised-recipe output**

**E. All numerical figures reproduce**, including all twelve table entries, selected epochs, epochs run and per-network rankings.

- Weak threshold: **0.1058773727**; stabilised range **0.0107464–0.0514081**; **0/12 weak**.
- Median stabilised/frozen ratios: **0.6282478** overall, **0.6408099** among the ten originally nonweak fits; **10/12** and **8/10** improve.
- Worse fits: fold 0/seed 2, ratio **1.88319**, selected/stopped **11/19**; fold 1/seed 1, ratio **1.47377**, **18/26**.
- Clipping: full-length fits **27.0966–34.3135%**, norm medians **0.28147–0.48190**; early stops **52.5124%/1.10137** and **46.7132%/0.90038**. Reported network-level ranges also reproduce.
- Records list no earlier attempts or numerical failures. Accounting totals **35:43:23**, matching “35 h 43 min”; all twelve tasks are recorded COMPLETED.
- Both output hashes and the quoted script hash match.

Evidence: [summary JSON](results/fourwire_scale_pgpre_stabilised.json), [accounting](results/fourwire_scale_hydra_job_evidence.txt).

**F. One wording correction.** The quoted losses **0.26/0.23** and **0.17/0.16** are from the **selected epochs**, not the stopping epochs. The records do not establish that lower learning rate caused these early stops; four components changed together. State the observations and label any explanation hypothetical. Otherwise, “consistent with sensitivity,” the preregistered reading and the closing limitations are appropriately restrained. [Protocol:1092](docs/fourwire_scale_protocol.md:1092).

**G. Gate supported.** All equality flags pass, both runs complete four epochs without numerical failure, and both record **1,512 steps**: \(4×18×\lceil648/32\rceil\). Frozen source, software, A100 model, deterministic mode and unchanged-checkout fields match. [Gate JSON](results/fourwire_scale_pgpre_stabilised_equivalence.json).

**H. Numerically sound; correct the selected-epoch/early-stop sentence. No additional stabilised runs are needed.**