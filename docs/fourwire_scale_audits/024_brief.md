Audit round 24 (read-only). HEAD is 6764d7b. Tracked files equal HEAD and stay stable while you read, except scripts/paper_figures.py, which is modified in the working tree by unrelated work and is out of scope (as are the untracked paper/ and docs/hf_post_*).

Since 07738d3 (round 23: ready to submit E5):
- c551953 adds your round-23 brief and review to docs/fourwire_scale_audits/ and corrects one sentence of the reading of the specification table.
- The 72 E5 runs and the report ran on the cluster from a worktree of 07738d3.
- 6764d7b commits their output: results/fourwire_scale_e5/<mix>/network_N.json (72 records), results/fourwire_scale_e5/<mix>/attempts/ (72 attempt files), results/fourwire_scale_e5.json (the report), 663 lines appended to results/fourwire_scale_hydra_job_evidence.txt (accounting, log lines, each array task's own job id, cluster-side hashes of the 145 files), and the protocol section "Outcome of E5 (2026-11-17)" at the end of docs/fourwire_scale_protocol.md.

Rules as before: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day (day index d with d % 5 == 4 and (d // 5) % 2 == 1); do not run the script's `run` without --dry, and do not run `report` (it writes a file): import the script and call its functions in memory, or recount with your own code. The E5 and E4 records hold no label and may be read in full. Try to refute, be concise.

A. Unchanged inputs. Are src/, results/fourwire_scale_freeze.json and scripts/hydra_fourwire_scale_final.slurm still byte-identical to a4068ef? Are scripts/fourwire_scale_balanced_envelopes_e5.py, scripts/fourwire_scale_balanced_envelopes.py and scripts/hydra_fourwire_scale_e5.slurm identical to 07738d3, and are the two script hashes those in every record and in the report? Do the committed files have the cluster-side hashes listed in the evidence file?

B. The records and the ledgers. Do all 72 records pass record_problems against the report's provenance (in memory), with the ledgers as committed? One attempt per network and mix, none run twice? Anything in a record that contradicts the execution rules of Amendment 12 item 8 and Amendment 11 item 7 with their corrections (commit, 16 cores, 36 days, 24 hours, the three OpenDSS days, 1,728 intervals in order, the mix and its shares, the positive-sequence check with 1 to 16 states on each of 36 days within 1e-6 pu)?

C. The report. Recompute results/fourwire_scale_e5.json from the 72 records with your own code: for each of the six cells the classification of every network, the lists and the count, the per-direction and combined summaries, the bootstrap intervals (2,000 replicates over days, seed 0), and the OpenDSS-day figures per engine; and the comparison of the base mix with the committed E4 records (82,944 levels; every envelope positive in either record with its level and three results; the classification of the 24 networks against results/fourwire_scale_e4.json). Any figure that does not reproduce? Is the cell base/posseq really computed from the posseq columns (on networks 10 and 25 the two design models have equal levels in every interval, and on network 18 in most, so identical figures there are expected; confirm that from the records rather than from the report)?

D. The outcome section, number by number, against the report, the records and the accounting lines. In particular:
 1. Runs: times, one attempt per run, the three tasks above one hour, 31.1 h, the hashes quoted, the dry-run jobs named.
 2. The positive-sequence check figures (16 states on every day of every run; 1.7e-15 and 1.6e-9 pu).
 3. The criteria table: 21 of 23 in each cell; the six pairs of counts and shares; the reference shares with their per-network ranges; networks 25 and 10 with their sizes and shares per mix; network 18's 64/64, 112/114, 45/46, all imports.
 4. The three per-network tables, every cell.
 5. Reading: the level comparison of the two design models (32,803 / 34,709 / 32,043 above, 8,642 / 6,740 / 9,406 equal, 27 / 23 / 23 below, all on network 18; the median ratios); network 18's equal levels; the cap counts on networks 25 and 10; network 25 imports alone (40.5%, 48.7%, 45.4% with their intervals); the ranges of mean feasibility and the six networks below 0.01; "in no interval ... is the level of a design model below the four-wire-verified level"; the counts in intervals where the four-wire rule admits nothing and that none of them reached 0.95; the reference shortfall; the intervals over days.
 6. OpenDSS days: the states per cell; network 18's table (states that differ by model, those feasible for OpenDSS only, envelopes that change status, the counts of envelopes reaching 0.95 by each engine); the single differing states on networks 9, 12, 17, 15 and 2; "every envelope that changes status reaches 0.95 by the sweep and not by OpenDSS".
 7. Network 1 and the published ENWL1 figures (README.md line 27, docs/claims.md line 30).

E. Claims. Does the section stay within Amendment 12 item 9 and item 5 and the corrections, and within what Amendment 11 item 8 excludes? Is anything overstated, or anything missing that a reader needs to judge the result? Consider at least:
 - The primary-cell sentence and "under all three mixes and both models": allowed as worded?
 - "The six cells are not six independent tests": is that caveat accurate and sufficient, given that the same 21 networks are counted in every cell?
 - Network 18 under the half mix: 123 of 128 four-wire-verified envelopes of the three days reach 0.95 by the sweep and 106 by OpenDSS. Does this threaten any statement of the section (the reference share of the 21 networks, the classification, the count), and is "20 of 23 in every cell without network 18" right?
 - Networks 25 and 10 not counted while network 25's imports alone are below one half in every mix: handled as fixed before the runs?
 - The section does not repeat the per-direction tables of Amendment 11 item 5 and points to the report file for them. Is that acceptable under Amendment 12 item 7, or must they be in the protocol?
 - Is there a reading of these records under which the primary cell would not be met?

F. Evidence file: are the appended lines consistent with the records (job ids through the mapping, nodes, start and run times, mixes), and free of account identifiers?

G. Verdict: is the outcome section accurate and within the amendment as it stands, or what must be corrected, in order of importance.
