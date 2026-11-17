Audit round 19 (read-only). HEAD is f377217. Tracked files equal HEAD and stay stable while you read.

Since 0f5cb80 (round 18: ready to submit E4):
- 3eea1f2 adds your round-18 brief and review to docs/fourwire_scale_audits/.
- The 24 E4 runs and the report ran on the cluster from a worktree of 0f5cb80.
- f34e85f commits their output: results/fourwire_scale_e4/network_N.json (24 records), results/fourwire_scale_e4/attempts/ (24 attempt files), results/fourwire_scale_e4.json (the report), 130 accounting and log lines appended to results/fourwire_scale_hydra_job_evidence.txt, and the protocol section "Outcome of E4 (2026-11-17)" at the end of docs/fourwire_scale_protocol.md.
- f377217 appends to the evidence file the mapping from array task id (13602480_N) to the task's own job id, which the records and attempt files carry.

Rules as before: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day (day index d with d % 5 == 4 and (d // 5) % 2 == 1); do not run the script's `run` without --dry, and do not run `report` (it writes a file): import the script and call its functions in memory, or recount with your own code. The E4 records hold no label and may be read in full. Try to refute, be concise.

A. Unchanged inputs. Are src/, results/fourwire_scale_freeze.json and scripts/hydra_fourwire_scale_final.slurm still byte-identical to a4068ef? Are scripts/fourwire_scale_balanced_envelopes.py and scripts/hydra_fourwire_scale_e4.slurm identical to 0f5cb80, and is the script's SHA-256 the one in every record and in the report?

B. The records and the ledger. Do all 24 records pass record_problems against the report's provenance (in memory), with the ledger as committed? One attempt per network, none run twice? Anything in a record that contradicts the execution rules of Amendment 11 item 7 and its corrections (commit, 16 cores, 36 days, 24 hours, the three OpenDSS days, 1,728 intervals in order)? The field `cpus` is 64 in the records; the outcome section says it is the node's core count and that the allocation of 16 is shown by numba_threads, opendss_processes and the accounting lines. Is that right, and enough?

C. The report. Recompute results/fourwire_scale_e4.json from the 24 records with your own code: the classification of every network, the lists and the count 21 of 23, the per-direction and combined summaries, the bootstrap intervals (2,000 replicates over days, seed 0), the OpenDSS-day figures per engine, and the E2 comparison (the 3,456 levels against evaluator A of the E2 envelope records bound by results/fourwire_scale_final_outputs_sha256.txt). Any figure that does not reproduce?

D. The outcome section, number by number, against the report, the records and the accounting lines. In particular:
 1. Runs: times, one attempt per network, the hashes quoted, the dry-run jobs named.
 2. Criterion: 21 of 23; 20 networks with none; network 18 with 64 of 1,728; 64 of 32,662 and 20,821 of 22,038 (94.5%; 92.7% to 95.9%); networks 25 and 10 with their sizes and shares.
 3. The three tables, every cell.
 4. Reading: the cap statement for networks 25 and 10 and their imports and exports; network 25 imports alone at 40.5% (29.6 to 51.9%); the ranges of mean feasibility and of the median ratio; "in no interval of any network is the balanced-model level below the four-wire-verified one"; the split 10,624 / 22,038 and where the 64 lie; the 5.5%; the intervals over days.
 5. OpenDSS days: 1,046,000 states of 5,230 envelopes; network 18's 615 of 53,000 with 613 and 2, by day, by direction and by model; the six envelopes that change status with their counts; 4 and 0 of 144, 114 and 112 of 121.
 6. Network 1 and the published ENWL1 figures (README.md line 27, docs/claims.md line 30).

E. Claims. Does the section stay within Amendment 11 item 8, corrections item 12 and second set item 4? Is anything overstated, or anything missing that a reader needs to judge the result? Consider at least:
 - "The count of 21 does not rest on the intervals in which the four-wire rule admits nothing": is it supported as worded?
 - Network 18: is "the classification cannot turn on it, because a stricter engine lowers the balanced share" right, and is "cause not established" the right limit of what can be said? Does the sweep's leniency there threaten any other statement of the section, for example the reference share of the other networks, where the engines agreed on every state of the three days?
 - The reference: is the account of the 5.5% of four-wire-verified envelopes that fall short accurate and not an excuse?
 - Network 25 imports below one half while the network is not counted: is the handling what the amendment fixed before the runs?
 - Is there a reading of these records under which the criterion would not be met (another reasonable handling of a rule the amendment left open)?

F. Evidence file: are the appended lines consistent with the records (job ids through the mapping, nodes, start and run times), and free of account identifiers?

G. Verdict: is the outcome section accurate and within the amendment as it stands, or what must be corrected, in order of importance.
