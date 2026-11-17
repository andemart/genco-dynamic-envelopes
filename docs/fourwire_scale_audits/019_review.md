**The criterion is met and the records are usable. Two corrections are needed.**

1. **Network 18: “the classification cannot turn on it” is too strong.** OpenDSS checked 144 of its 1,728 balanced envelopes. It is not uniformly stricter: two states differ in the reverse direction. The checked days show **4/144 reaching the target by sweep versus 0/144 by OpenDSS**; the other 33 days remain unmeasured. Say that the **registered classification is unchanged**, rather than guaranteeing an all-days OpenDSS classification. Even declining to count network 18 leaves **20 counted networks**, above the required 12. [Protocol:1234](docs/fourwire_scale_protocol.md:1234).

2. **Maximum median ratio: change 4.15 to 4.14.** Network 20 gives **4.1446078431**; the minimum is **1.1587837838** on network 6. [Protocol:1225](docs/fourwire_scale_protocol.md:1225).

**A–B. Identity and records pass.** The frozen files are unchanged from `a4068ef`; both E4 scripts are unchanged from `0f5cb80`. The script hash matches all records and the report. All **24 records pass `record_problems`**, with 24 matching attempt files, no earlier attempts, the expected 36 days, three OpenDSS days and 1,728 ordered intervals each. Source, software and freeze provenance match.

`cpus=64` comes from `os.cpu_count()`: describe it precisely as the system-visible **logical CPU count**. Allocation of 16 is adequately supported by accounting, the start check, and the recorded 16 Numba threads/OpenDSS workers. [Provenance:144](src/gencodoe/fourwire_scale_train.py:144).

**C–D. The report and remaining figures reproduce.**

- Independently recomputed every network’s classification, summaries, directional OpenDSS figures and 2,000-replicate bootstrap intervals. No substantive differences; maximum floating-point discrepancy **7.1×10⁻¹⁵**.
- **All 960 table cells match.** Apart from the prose ratio above, the numerical outcome statements reproduce.
- **21/23** counted; 20 have no reaching balanced envelope. Balanced: **64/32,662**; reference: **20,821/22,038 = 94.4777%**.
- Reference failures: **1,217/22,038 = 5.5223%**; pooled mean feasibility **0.975124**.
- **10,624** balanced envelopes have zero reference levels; none reaches the target. All 64 successes lie among the other **22,038**. Removing zero-reference intervals leaves the same 21 networks below one half, supporting the stated robustness description.
- Cap counts, network 25’s import result and interval, feasibility ranges, and absence of any balanced level below its reference all reproduce.
- OpenDSS totals **1,046,000 states / 5,230 envelopes**. Network 18’s **615 = 613 + 2**, all directional/day/model decompositions, six status changes and **114→112 of 121** reference envelopes reproduce.
- All **3,456 E2 levels match exactly**, with matching manifest-bound hashes.
- Network 1’s figures and the quoted README/claims figures match their cited sources.

**E. Otherwise within the amendment.** The reference shortfall is plainly disclosed; it should remain an empirical result, without attributing it solely to sampling. “Cause not established” is appropriate. Three-day engine agreement does not establish agreement on the remaining days, including on the other 23 networks. Network 25’s combined-direction classification follows the frozen rule. No unresolved edge case changes whether the criterion is met.

**F. Evidence is consistent.** Job-id mappings, nodes, attempt times, logs and accounting agree. Total elapsed task time is **8:11:36**, correctly rounded to 8 h 12 min; range **6:58–42:59**, report **21 s**. Appended identifiers are redacted. Accounting supports the queue delay, but not its asserted cause; remote-copy hash equality is not independently documented beyond the internally matching artefacts.

**Verdict: sound numerical outcome and criterion met; correct the network-18 inference and ratio. No rerun is required.**