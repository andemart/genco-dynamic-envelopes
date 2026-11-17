**Not ready to submit.** The core power mapping and four-wire reference match the intended experiment, but isolation, record validation and several preregistered edge cases need fixing first.

**A. Source identity — passes.** `src/`, the freeze file and the original final-job script are byte-identical to `a4068ef`. The recomputed source hash matches the freeze.

**B. Scenarios and reference — correct.** The four-wire design path reproduces evaluator A: same customer mapping, scenarios, search, quantities and non-convergence rule. The stream-1 generator is equivalent by inspection; synthetic probes reproduced all 24 hours on validation days 3 and 8. The first-hour assertion alone would not establish that.

Stream 2 gives independent *conditional resampling*: the same technology flags, profiles, hour and sampling law, with a different random stream. It does not supply new meters or independent calendar data.

**C. Balanced model — mapping correct; clarify the comparison.** Each customer’s total P/Q and envelope are preserved and divided into three shares. Existing three-phase customers are not divided twice. Source, impedances, ratings and solver remain unchanged.

However, voltage monitoring also changes: the balanced model checks **all three phases at every customer bus**, whereas the reference checks actual connections. State this explicitly. At fixed voltages, those additional checks can be more conservative. There is no general ordering against a positive-sequence reduction; this is a defensible **balanced-load four-wire surrogate**, not a quantitative replication of that comparator. [Mapping:175](scripts/fourwire_scale_balanced_envelopes.py:175), [voltage summary:134](src/gencodoe/fourwire_sweep_opt.py:134)

**D. Fresh check — correct.** Export yields consumed power `pf − level`; import yields `pf + level`; reactive consumption remains `qf`. Fresh verification always uses the actual phase connections. Strictly positive returned levels are used as specified. `reaches()` classified all **201 possible counts** correctly: 190–200 pass.

**E. OpenDSS — mapping correct; enforce the worker count.** OpenDSS receives exactly the same complex consumption arrays as the sweep and is judged using its own quantities and convergence flag. Positions 0/12/24 give **days 9/129/249: January 10, May 9, September 6**.

Chunking preserves output order, but changes the preceding solution used within each chunk and can therefore affect convergence and decisions. **Sixteen workers are requested by Slurm but not enforced by Python or the report.**

**F. Classification and reporting — incomplete.**

- **No positive balanced envelope on OpenDSS days:** `classify()` sets agreement to `None` and **still counts an otherwise unsafe network**. Freeze that policy explicitly before observing E4 outcomes; the amendment leaves it implicit. A synthetic probe confirmed this behaviour. [classify:338](scripts/fourwire_scale_balanced_envelopes.py:338)
- **Report validation is insufficient.** It checks interval *length*, not unique expected day/step/direction membership, settings or mandatory fresh/OpenDSS results. Synthetic records with duplicate import intervals, missing export intervals, missing OpenDSS checks and wrong settings passed. Validate the complete interval set, provenance/settings, and required results for every positive envelope. [report:373](scripts/fourwire_scale_balanced_envelopes.py:373)
- Bootstrap intervals and OpenDSS discrepancy summaries are **joint across directions**, although item 5 promises reporting per direction. Zero-positive bootstrap replicates are omitted and counted; document that conditional interval rule. The global disagreement list also omits disagreements on networks not classified unsafe overall.
- The E2 tolerance permits **17 of 3,456** comparisons beyond the level tolerance. Exact bit equality is unnecessarily brittle across CPUs, but retain mismatch details and bind the old E2 inputs by hash. Do not relax the gate after seeing results.

The criterion is not tautological: the disclosed validation claims include both passing and failing examples. Counting a small positive offer when the four-wire rule returns zero is legitimate under the stated definition; it does not prove the extra injection caused the underlying infeasibility.

**G. Execution and isolation — blockers.**

1. **Dry mode can enter held-out days.** `--dry --days 2 --hours 48` includes validation day 8 at hour 24, which is **day 9**. Enforce `1 ≤ hours ≤ 24` and validate reductions. I checked only this index arithmetic, not those states. [scenario indexing:142](scripts/fourwire_scale_balanced_envelopes.py:142)
2. **Outputs lack protected-path guards.** For example, `--dry --out OTHER_WORKTREE/.../runs/final/HALT.json` can create a frozen-study halt file. Existing files cannot be replaced, but new files can be written inside protected folders. Guard dry and report outputs. [output selection:162](scripts/fourwire_scale_balanced_envelopes.py:162)
3. **Publication is exclusive but not atomic.** Death during `json.dump` leaves an existing truncated record that cannot be retried. Preserve interrupted attempts, publish completed records atomically without replacement, and implement the promised rerun ledger. [write_new:117](scripts/fourwire_scale_balanced_envelopes.py:117)

Dropping GPU and PyTorch deterministic-mode requirements is justified. Dropping the old halt dependency is defensible for this separate analysis. Fold checks are irrelevant here. The original final-job script is omitted initially but still checked by the inherited end check—an unnecessary inconsistency.

Slurm’s mapping, UK limits, thread environment and `afterok` chain are correct. Untracked E4 records do not fail `git diff`. If an original array task fails, rerunning it separately will not repair the original report dependency; document recovery.

**H–I. Disclosure and claims — qualify before running.** E4 cannot change frozen E1–E3 results through its default paths. It is nevertheless a new extension devised after their outcomes were available. Disclose the already-seen E3 sweep/OpenDSS discrepancies, particularly network 18. “Reads no label” is accurate; “uses no label” overlooks the label-derived Thevenin source.

“Replicates on a majority” should explicitly mean the **new operational criterion under this balanced-load surrogate**, rather than reproduction of the positive-sequence result. Preserve the listed restrictions and report the four-wire reference beside every balanced result. E4 can support a conclusion about phase modelling; it does not establish a GENCO benefit.

**Verdict:** fix, in order, **dry-day isolation; protected/atomic outputs and retry accounting; complete record validation and worker enforcement; explicit OpenDSS/empty-bootstrap edge rules and reporting scope; comparator and prior-outcome disclosure.**