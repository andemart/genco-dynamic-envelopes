**A. Pass.** `src/`, the freeze JSON and the original final Slurm script remain byte-identical to `a4068ef`.

**B. All five points resolved.**

1. **Ledger:** fabricated missing attempts, wrong network/job/node/commit/script, and missing earlier attempts are refused. Correct one-attempt and retry ledgers pass. [Script:419](scripts/fourwire_scale_balanced_envelopes.py:419).

2. **Differing decisions:** the constraints are necessary and sufficient for a two-by-two table of 200 paired decisions. An exhaustive in-memory comparison covering **8,201,403 integer triples** found **zero mismatches**. Valid boundary triples pass; impossible counts, wrong parity, and `d = 0.5`, `true`, or a string fail. A *share* of `0.5` correctly represents 100 states; Boolean/string shares fail. [Script:412](scripts/fourwire_scale_balanced_envelopes.py:412), [465](scripts/fourwire_scale_balanced_envelopes.py:465).

3. **Extra voltage checks:** correct. For the same balanced-model solution, monitoring additional phases only removes feasible decisions; it cannot explain excessive permissiveness. The unsupported magnitude assertion is withdrawn. [Protocol:1104](docs/fourwire_scale_protocol.md:1104).

4. **Zero reference envelope:** correct. The correction distinguishes design/search outcomes from fresh-check outcomes and withdraws the causal inference. Positive balanced envelopes are classified by their own fresh results. [Protocol:1105](docs/fourwire_scale_protocol.md:1105).

5. **Recovery:** correct. An existing published record causes refusal **before another attempt is claimed**. Missing-record failures permit retry; published records are retained and validated, with invalid records requiring the stated resolution. [Script:207](scripts/fourwire_scale_balanced_envelopes.py:207), [job header:16](scripts/hydra_fourwire_scale_e4.slurm:16), [Protocol:1106](docs/fourwire_scale_protocol.md:1106).

**C. No new blocker found.** The diff changes validation helpers and the report’s validation call; `run()` and its computational path are unchanged. The fabricated valid complete records passed.

**D. Resolved.** The stabilised-output correction identifies the quoted losses as selected-epoch values and labels the lower-learning-rate explanation a hypothesis. It now stays within the records’ evidence. [Protocol:1112](docs/fourwire_scale_protocol.md:1112).

**E. Ready to submit the 24 E4 runs and the report from a worktree of `0f5cb80`. No remaining blockers within this audit’s scope.**