I found no submission blocker.

1. **Source identity: supported.** The nine changed paths between `e85164c` and `a4068ef` are all under `results/`. Tracked `src/`, `scripts/`, `tests/` and `docs/` are byte-identical.

2. **Freeze: matches.** I independently reproduced the source hash (`661c1d26…1c4be87`), all **56 per-file hashes**, job-script hash, both input hashes and all four folds. I also reproduced the hashes over **192 variant-network files** and the eight profile groups. All six dry-run records match the freeze’s software, source and input fields. No missing or wrongly formed field was found in the three freeze readers. The earlier `frozen_at_commit=e85164c` is compatible with the later results-only commits. [Freeze](results/fourwire_scale_freeze.json), [checks](src/gencodoe/fourwire_scale_train.py:631).

3. **A100 dry runs: internally consistent.** Both accounting entries say `COMPLETED`; both logs end with `DRY_RUN_OK`. GPU identity, numerical modes and timing’s **16 cores/threads/workers** agree with the code. Printed metrics, timing medians, selected epochs, evaluation-record hashes and checkpoint links match the records. Evaluation/envelopes use deterministic mode; timing uses default mode; TF32 is off throughout. [Evidence](results/fourwire_scale_hydra_job_evidence.txt:116).

   Across primary and rating-variant evaluation screens, all reported false-safe, cleared-indeterminate and false-unsafe counts are **zero**. Complete sweep/OpenDSS decisions agree, although transformer flags differ on **5 states in network 17 and 2 in network 7**.

   - **a. Zero envelopes:** the acceptance threshold is **196/200 = 0.98**. Recorded zero-level feasible shares range from **0 to 0.435**. Each evaluator examined exactly **400 states per interval**—the cap and zero—and returned the documented zero sentinel. These zeros do **not** certify that the baseline is feasible. No defect is indicated. [envelopes.py:175](src/gencodoe/envelopes.py:175).
   - **b. Rating variants:** network 17’s retained ratings actually change on **15 and 6 branches**; network 7’s change on **22 and 93**. The variants are active. Identical aggregate screen metrics can coexist with changed predictions; the JSONs do not prove identical individual decisions. Network 7’s `cu16_89` changes the reported outcomes.
   - **c. Zero OpenDSS calls:** screened pipelines first solve only uncertain states. At the largest batch, their fallback sweeps handle **966 states** on network 17 and **1,311/1,318** for U1/PG-net on network 7, with no sweep failures in those subsets. Baselines solve all 3,000 states and forward **85/350** failures to OpenDSS. The counters describe **fallback calls in the first timed repeat**, not totals across every repeat. [screened](src/gencodoe/fourwire_scale_e2.py:179).

   Coverage limits remain: the interior envelope search and screen-to-OpenDSS failure path were not exercised by these dry runs. The committed JSONs also cannot independently establish every individual timed decision without their referenced sidecars.

4. **Time limits:** no likely overrun is demonstrated by the measurements.

   | Stage | Approximate projection | Limit |
   |---|---:|---:|
   | PG-pre, maximum epochs | 3.27–3.30 h | 12 h |
   | Target stage, all six fits | 0.71–0.85 h | 6 h |
   | Five PG-ft fits | 0.18–0.23 h | 3 h |
   | Evaluation | Several minutes | 1.5 h |
   | Timing, networks 7/17 | 2.06/2.74 h | 8 h |
   | Envelopes | Least well established | 4 h |

   These are projections, not bounds. Training estimates exclude setup/I/O; final fits have eight cores versus sixteen in the dry runs. Timing estimates include 103 rounds, scaling the largest batch to 4,096, plus eleven large OpenDSS batches. The envelope limit has the weakest evidence: an interior search can evaluate **4,400 states per evaluator**, versus 400 here—**11×** the state count. [Fit/timing measurements](results/fourwire_scale_hydra_job_evidence.txt:120), [limits](scripts/hydra_fourwire_scale_submit.sh:19).

5. **Determinism: supported, with indexing clarified.** Independently comparing the raw arrays gives identical deterministic-mode losses at all **32 steps**, both validation losses and the recorded final-weight hashes. Default mode first differs at **index 2—the third training step**; maximum loss difference is **6.723403930664062e−5**. This is the disclosed older-source, short probe, not a reproducibility certificate for complete final fits. No Amendment 7 violation follows. [Probe](results/fourwire_scale_determinism_job13587703.json).

**Remarks that can wait:** retain the referenced sidecars/checkpoints; distinguish infeasible-baseline zero sentinels when reporting envelope agreement; make final speed/safety claims only from the prescribed final reports.

**Ready to submit — no blockers found.**