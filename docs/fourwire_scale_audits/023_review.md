**A. Round-22 gap resolved.** Every envelope positive in either record enters the denominator; agreement requires an identical level and all three identical results. The previous one-float-step/100→0 counterexample now fails. Exact reproduction passes; envelopes present in only one record are counted as disagreements. No remaining coverage bypass found within the validated cohort and declared 0.5% allowance. [E5:453](scripts/fourwire_scale_balanced_envelopes_e5.py:453).

Second-set item 1 describes the implementation exactly. Item 2 reasonably motivates an expectation of exact agreement; it does not guarantee it.

**B. No other executable change.** AST comparison identifies only `e4_comparison()` as changed. Constant values, run path, fresh check, classification, validation, job script and robustness script are unchanged.

**C. Every specification-table number reproduces.**

- Registered: **21**, with **10 and 25** not unsafe.
- Nine single-change counts, in printed order: **21, 21, 21, 22, 22, 21, 21, 21, 21**. The two counts of 22 exclude only network 10; the others exclude 10 and 25.
- Full distribution: **20: 4; 21: 124; 22: 28; 23: 6**.
- Only **10, 18 and 25** change classification. Network 18 is not unsafe in **4** combinations; networks 25 and 10 are unsafe in **34** and **6**, respectively.
- Without network 18: **20–22**. No combination has an empty positive-envelope denominator.
- The descriptions of counts 20 and 23 are exact. Count 20 permits either interval filter and either sweep time span; count 23 permits either interval filter and all three engine choices.

The entire recomputed grid matches the JSON. Commit `61b38f3`, the recorded clean-checkout flag, script hashes, E4 report hash and quoted output hash are consistent.

**One wording correction:** “the count … does not depend on the rules” overstates the finding—the count changes from 20 to 23. Replace it with **“the majority conclusion remains unchanged across these listed alternatives.”** The remaining exploratory qualifications are appropriate. [Protocol:1412](docs/fourwire_scale_protocol.md:1412).

**D. Attribution is consistent.** The local README credits Urquhart **and Murray Thomson** for cable data; its generic CC BY 4.0 licence contains no author roster. The three dataset authors—Heidarihaei, Geth and Claeys—are independently corroborated by the [author’s university citation](https://eecs.uq.edu.au/node/12312).

**E. Ready to submit the 72 E5 runs and report from `07738d3`. No remaining E5 blocker.** Separately, the specification output is numerically accurate; correct the single reading sentence above.