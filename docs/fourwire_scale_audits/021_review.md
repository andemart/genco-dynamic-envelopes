**Two code blockers and one disclosure gap remain.**

1. **The reduction gate mishandles a legitimate non-convergence outcome.** If none of the first eight states converges, `check_reduction()` returns zero states without failing. `run()` excludes that day, publishes a record marked valid, then the report refuses it because `days_checked < 36`. The existing record prevents retry. Freeze a deterministic policy for an inconclusive day before running. Also validate the check’s state counts: fabricated values **0, −1, 0.5 and `"none"` all pass** when `days_checked=36`. [E5:204](scripts/fourwire_scale_balanced_envelopes_e5.py:204), [320](scripts/fourwire_scale_balanced_envelopes_e5.py:320), [375](scripts/fourwire_scale_balanced_envelopes_e5.py:375).

2. **The E4 agreement check can pass substantial, incompletely reported disagreement.** With every level unchanged, fabricated records changing every feasible count from 200 to 0 remain valid and `e4_comparison()["passed"]` remains true. Separately, changing paired disagreement counts from 0 to 2, with both marginal counts at 100, produces **no listed difference**. The comparison never checks `fresh_decisions_differ`; its gate considers only levels. Freeze the count-discrepancy policy, compare all promised counts, and enforce the promised agreement of the recomputed E4 classification. [E5:436](scripts/fourwire_scale_balanced_envelopes_e5.py:436), [452](scripts/fourwire_scale_balanced_envelopes_e5.py:452), [Amendment:1311](docs/fourwire_scale_protocol.md:1311).

3. **The ENWL1 comparison is correct but incomplete.** Besides the listed differences, the old balanced model checks **sending-end apparent-power ratings and all retained non-slack bus voltages**; E5 checks **phase-current ampacity and customer-bus voltages**, with the LV-root transformer proxy. The scenario-draw ordering also differs beyond changing placement daily. Disclose these differences; no model change is required. Clarify that the three phase voltages have equal **magnitudes**, not equal phasors. [Original ratings](src/gencodoe/enwl.py:9), [flows](src/gencodoe/feeder.py:203), [sampling](src/gencodoe/doe_day.py:58).

**A. Unchanged — passes.** All requested frozen files, E4 scripts, records and report are unchanged. The E4 script’s hash matches `E4_SCRIPT_SHA256`.

**B. Reduction — otherwise correct.** The branch reduction matches `enwl.read_linecodes`; reduction before/after merging agrees for branches comprising one line code. The source projections use the correct phase order. Balanced inputs yield the scalar positive-sequence solution, negligible neutral current, and the stated voltage/current/root-power quantities.

On **network 25, validation day 3**, all eight probe states converged: phase-magnitude spread **5.78e−16 pu**, scalar disagreement **1.57e−11 pu**. This CPU probe used JIT disabled. The gate checks only converged voltage magnitudes, shares the reduction parameters with the scalar solver, and does not independently validate angles, currents, limits or envelope operating points. The scalar solver also returns no convergence flag.

The network-data disclosure reproduces: negative-sequence ratio **6.499–6.509e−11**, maximum source asymmetry **3.23e−14 Ω**, maximum Kron change **0.44874%**.

**C–D. Mixes and fresh checks — pass.** The base generator matched E2 and E4 exactly in the validation probe. Flags are independent Bernoulli assignments within a mix and deliberately nested across mixes. Other mixes do not accidentally retain `e2.MIX`. Equal-level reuse is equivalent because sweeps start flat and OpenDSS resets each chunk. Positive/zero result handling is consistent.

**E–F. Records and execution — otherwise pass.** All **72 fabricated valid records** passed; wrong mix, commit and E4-script identity were refused. Column conversion and model substitution are correct. Fixed paths, dry-day checks, provenance checks and the three-array dependency are sound. Recovery must wait for every outstanding task; incomplete records cannot produce a report.

**G. Specification table — ready.** The **162 combinations**, **nine single-rule changes**, engine selections, positive-reference restriction, strict thresholds and count without network 18 match item 6 in made-up-row probes. The registered-result gate is appropriate for the known E4 cohort. Scientific inputs are the E4 records/report; other reads supply code and provenance metadata. **I did not compute the actual table.**

**H–I.** Primary/secondary choices and permitted claims are otherwise sufficiently fixed. Private dry-run figures and decision chronology remain author disclosures.

**Not ready to submit E5 at `8e46433`: resolve findings 1–2 and complete the comparator disclosure. Item 6 can be computed as written.**