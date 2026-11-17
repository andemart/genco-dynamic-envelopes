Audit round 14 (read-only). HEAD is 11bcf1f. Tracked files equal HEAD and stay stable while you read.

Since 9aadb1d:
- d53088f: the wording corrections and job evidence you asked for in round 13 (docs/fourwire_scale_protocol.md, "Corrections to the diagnostic output section"; 96 accounting lines appended to results/fourwire_scale_hydra_job_evidence.txt; your round-13 files).
- 11bcf1f: the outputs of the final runs, copied byte for byte from the frozen worktree on the cluster, and the section "Outcome of the final runs (2026-11-17)" at the end of the protocol:
  results/fourwire_scale_fits.json (276 fit records), results/fourwire_scale_eval/network_N.json (24), results/fourwire_scale_e2_timing/network_N.json (6), results/fourwire_scale_e2_envelopes/network_N.json (6), results/fourwire_scale_e3.json, results/fourwire_scale_e2.json.

These are the first test outcomes of the study. Rules: read-only; never read data_out/fourwire_scale/labels/test or /calib (the committed records are outputs and may be read); no fixture test suite; no cluster. Try to refute; derive numbers yourself.

Two errors in the outcome section are already known and will be corrected with your findings: (a) the claim wording is in Amendment 9, item 6, not item 5; (b) "because the baseline load is not feasible there" holds for all zero envelopes except one interval each on networks 7 and 17, where the search itself returned 0.

A. Records. Are all 276 fit records, 24 evaluation records and 12 E2 records valid, made under freeze ea17e8f4…1dccb with the frozen source, libraries and GPU model, unchanged checkouts, no earlier attempts, no numerical failure? Do the report files name the records they were computed from (hashes), and do those hashes match the committed records? Anything that makes a record unusable?

B. E3 report against the evaluation records. Recompute from results/fourwire_scale_eval/: the primary statistic at B = 1,000 (PG-scratch minus PG-ft fallback share, mean over seeds, then over the 23 fresh networks), its paired bootstrap interval (2,000 replicates, seed 0, as the code does it), the false-safe and cleared-indeterminate sums, the two secondary comparisons, the sensitivity with network 1, and the seed-0 method table. Is "criterion not met" the right reading of §7?

C. E2 report against the timing and envelope records. Recompute the speed-ups and intervals at batch 4,096 for the six networks, the criterion per network, the break-even size, the sensitivity series, the U1 screen under the same rule, the exponents, and the envelope summaries (share within tolerance, unsafe-envelope intervals, zero envelopes).

D. The outcome section. Check every figure in it against the two reports: the seed and fold table (computed by me from per_network_per_seed, not by the report), the statements about fold 2 and the U1 screen, the method table, the E2 table, batches 1 and 256, the exponents, the envelope paragraph. Report each mismatch with your value.

E. Claims. Is anything in "What may be claimed" or elsewhere in the section stronger than §7, §10, Amendment 9 or the data allow? Are the after-the-fact descriptions (seed and fold pattern, the 17 networks of folds 0, 1 and 3, the concentration on few networks) fenced clearly enough as descriptions? Is the account of the weak seeds consistent with the note of 2026-11-16 and your guidance? Is the one false-safe decision of PG-zs, the six unsafe-envelope intervals of network 10 and the zero envelopes stated plainly enough?

F. Anything the section omits that §7 or the amendments require to be reported with these results.

Verdict: sound as the committed outcome, or the corrections needed, in order of importance.
