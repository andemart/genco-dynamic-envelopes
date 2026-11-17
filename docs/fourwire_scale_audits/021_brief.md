Audit round 21 (read-only). HEAD is 8e46433. Tracked files equal HEAD and stay stable while you read, with one exception that is not part of this audit: scripts/paper_figures.py is modified in the working tree by unrelated work (ignore it; do not rely on it).

Since bb98792 (round 20: E4 closed, no remaining findings):
- 9c8d92f, 12b1716: your round-20 files in docs/fourwire_scale_audits/, and the times of rounds 19 and 20 in its README corrected.
- 8e46433: "Amendment 12" at the end of docs/fourwire_scale_protocol.md (family E5: the positive-sequence model as a third design model and two further technology mixes; item 6: an exploratory specification table for E4), with scripts/fourwire_scale_balanced_envelopes_e5.py, scripts/hydra_fourwire_scale_e5.slurm and scripts/fourwire_scale_e4_robustness.py.

Nothing of E5 has run on a held-out day, and the table of item 6 has not been computed. Rules: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day (day index d with d % 5 == 4 and (d // 5) % 2 == 1); do not run the E5 script's `run` without --dry and do not run `report`; do not compute the table of item 6 from the E4 records (review scripts/fourwire_scale_e4_robustness.py by reading and with made-up rows). In-memory probes on validation days and on fabricated records are fine. Try to refute, be concise.

A. Unchanged. src/, results/fourwire_scale_freeze.json, scripts/hydra_fourwire_scale_final.slurm identical to a4068ef? scripts/fourwire_scale_balanced_envelopes.py and scripts/hydra_fourwire_scale_e4.slurm identical to 0f5cb80 (SHA-256 of the former e817d926...14fed25, the constant E4_SCRIPT_SHA256)? The E4 records and report unchanged since f34e85f?

B. The positive-sequence model (Amendment 12, item 3; positive_sequence, reduced_case, scalar_power_flow, check_reduction).
 1. Is positive_sequence() the reduction of gencodoe.enwl.read_linecodes, applied to the 4x4 matrix of each retained branch? Is the amendment's account of how that differs from the ENWL1 model of the challenge entry right and complete ("the same for a branch of one line code"; source; flags per day)?
 2. Source: e1 = conj(u) . e / 3 and z1s = conj(u) . Zth . u / 3 with u = (1, a^2, a): the positive-sequence voltage and impedance for the phase order of the frozen Thevenin equivalents?
 3. Is the Numba sweep on reduced_case() exactly the positive-sequence power flow, including the three limits (customer phase-to-neutral voltage, cable loading, the transformer quantity)? Is "the neutral keeps its own impedance and carries nothing" right for the sweep's code?
 4. check_reduction compares the sweep with a scalar power flow built from the same z1, e1, z1s. What can it miss? Is 1e-6 pu, on 8 design scenarios of the first hour of each day, a sound gate, and can it stop a correct run (for example when fewer than 8 states converge, or none)? Note the report requires days_checked == 36: what if no state of a day's first hour converges?
 5. Probe on a validation day if you can: does the reduced network give three equal phase voltages and the scalar solution?

C. Mixes (item 2; scenarios()). Same generator as the E4 script and fourwire_scale_e2.day_scenarios for the base mix? Flags nested across mixes as stated? Does anything else in a run still depend on e2.MIX? Is "independent shares" an accurate description?

D. The fresh check (run()). Same as E4 for each model? Is solving equal levels once exactly equivalent to solving them separately? Any path by which a positive envelope gets no result or a zero envelope gets one?

E. Records and report. Column storage and rows_of(); record_problems(); e4_rows() feeding e4.summarise and e4.classify: is each cell's classification exactly that of Amendment 11 item 4 and its corrections with the cell's model in the place of the balanced model? The six cells, the criterion, the primary cell. e4_comparison(): bound to the E4 report's record hashes, the tolerance rule, the list of differences. The ledger per mix. Can a valid complete set of 72 records be refused? Can an invalid, incomplete or mixed set (wrong mix in a folder, another commit or script, a record of E4's layout) pass? Try fabricated records in memory.

F. Isolation and execution. Can a dry run touch a held-out day or write inside results/? Final-run requirements (start, unchanged, 16 cores, E4 script hash). The job script: three arrays, the report's dependency on all three, the recovery cases. Anything that could let a final run start from a wrong commit or with a modified script?

G. scripts/fourwire_scale_e4_robustness.py against item 6: the 162 combinations, the classification under each choice (in particular the engine choices and the restriction to intervals with a positive four-wire-verified envelope), the gate that the registered combination reproduces the E4 report, the count without network 18. Does it read anything but the E4 records and report? Is anything in item 6 not implemented, or implemented and not stated?

H. The amendment text. Is the disclosure accurate and complete (what was seen, who decided what and when, the dry runs, what they let one expect)? Are the criteria and edge rules fully determined before the data exist? Is the claim wording of item 9 within what E5 can show? Any statement the code does not implement, or behaviour of the code the text does not state? Is the primary / secondary structure free of a forking path?

I. Verdict: ready to submit the 72 E5 runs and the report from a worktree of 8e46433, and to compute the table of item 6 at that commit; or the blockers, in order.
