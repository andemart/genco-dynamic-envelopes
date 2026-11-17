**A. Finding 1 is resolved.**

1. A day without a usable state now raises before publication; no E5 record or partial is created. The final-run attempt remains. [E5:214](scripts/fourwire_scale_balanced_envelopes_e5.py:214), [328](scripts/fourwire_scale_balanced_envelopes_e5.py:328).
2. The usability rule is sound. The stricter scalar tolerance can exclude slowly converging states; tenth-power probes reduce that risk without guaranteeing convergence universally. Failure is now a predetermined stop. On network 25, validation days 3 and 8, **all 16 states passed**; maximum differences were **5.78e−16 pu** between phases and **1.98e−11 pu** against scalar. NaN scalar states were excluded; non-converged scalar states cannot enter the comparison.
3. Every requested invalid type/count, wrong list length, non-dict check and NaN difference was refused. Valid daily counts of **1 and 16** passed. [E5:389](scripts/fourwire_scale_balanced_envelopes_e5.py:389).
4. Correction 1 accurately describes magnitude equality and the gate’s limited coverage.

**B. The original counterexamples are fixed, but a coverage gap remains.**

- Exact reproduction passes.
- Unchanged levels with feasible counts changed to zero fail.
- Changing disagreement counts from 0 to 2 with both marginal counts at 100 is now listed and fails when above the declared allowance.
- Per-network classification and counted-list checks are enforced.

However, results are compared **only at exactly equal levels**. In a fabricated complete record, I moved every level upward by **one representable float** and changed feasible counts from **100 to 0**, preserving classifications. The result was:

- zero levels beyond tolerance;
- **zero envelopes compared for results**;
- every changed interval listed;
- **`passed=True`**.

Thus the gate implements correction 2 literally but can pass without testing any fresh-result agreement. Freeze a minimum comparison-coverage rule, or another rule covering results at unequal-but-within-tolerance levels, before outcomes exist. [E5:456](scripts/fourwire_scale_balanced_envelopes_e5.py:456), [468](scripts/fourwire_scale_balanced_envelopes_e5.py:468).

**C. Finding 3 is resolved.** The thermal, voltage-monitoring, transformer and scenario-draw differences are now accurately disclosed. [Protocol:1362](docs/fourwire_scale_protocol.md:1362).

**D. No other computational change found.** The three design paths, fresh check, classification and isolation remain unchanged apart from the intended gate changes. The job script and E4 robustness script are unchanged. The new private dry-run and cancellation statements remain author disclosures.

**E. Not ready to submit E5 at `61b38f3`: close the result-comparison coverage gap. No other blocker found.**