# C+ protocol: GENCO screened search plus a final exact gate (post-hoc)

**Status: post-hoc safety enhancement.**

- The pre-registered B (exact four-wire search) and C (GENCO + four-wire conformal screen) results remain the primary results and are not modified.
- C+ was motivated by one observed C false acceptance: 2024-04-29 13:00, import; B = 0.0308 kW, C = 0.3647 kW; C's exact design-set feasibility was 0.970 against the 0.98 rule.
- This protocol is committed before the formal C+ run.

## Disclosure: exploratory runs before this protocol

Before this protocol was written, an implementation of C+ (`src/gencodoe/cplus.py`, not yet committed) was run exploratorily:
- on 2024-04-29: the known false acceptance was caught and repaired to 0.0321 kW;
- on 7 annual days (2024-01-10 … 2024-03-10), before the run was stopped.

Seen from those runs:
- the per-day gate and repair counts printed to the log;
- the full outputs of 2024-04-29 and 2024-01-30.

On 2024-01-30, one repair (13:00, import) had C = B = 0.1208 kW. The gate's re-solve found 195/200 feasible where B's search found 196/200. The cause is OpenDSS solve-history dependence: one scenario's worst cable loading was 1.00005 in one solve order and 0.99989 in another, the default 1e-4 tolerance.

These outputs are archived in `data_out/doe/exploratory_cplus_pre_protocol/` and are **not** reported as C+ results. The formal run starts from scratch.

## Definition

C+ takes each stored C result (one interval and direction) and does the following.

1. **If C's level is zero:** keep zero. No gate.
   - Zero is the existing sentinel for "no feasible non-negative optional envelope at the design threshold". It is **not** a claim that the passive network is safe.
2. **If C's level L_C > 0, gate it:** replay the same 200 design scenarios (same seeds, same draw order as the run) and compute the exact four-wire feasible share at L_C.
   - Accept L_C if share ≥ the run's acceptance threshold (0.98 for n = 200, ε = 0.05, β = 0.05).
3. **If the gate fails, repair it:** an exact k-section search on [0, L_C], with the same rule, the same levels per round (4) and the same tolerance (1e-3 × cap) as the envelope search.
   - Return the largest exactly-accepted level ≤ L_C.
   - If level 0 itself fails, return the zero sentinel and flag `zero_infeasible_sentinel`.
4. **Fresh feasibility:** for repaired intervals only, recompute the held-out feasibility of the new level on the run's fresh four-wire draws (first 200 of the replayed verification draws). Non-repaired intervals keep C's held-out values.

## Fixed choices

| | Choice |
|---|---|
| Scenarios, seeds, households, limits, threshold, tolerance | identical to the stored run (replayed; seeds 101/101, UK limits) |
| OpenDSS configuration for gate and repair | **identical to B**: same build of the four-wire model, constant-PQ loads 0.5–1.5 pu, default tolerance, retry-from-no-load on non-convergence, solves in scenario order |
| Solver-noise classification (secondary, pre-specified) | for gate failures where \|L_C − L_B\| ≤ tolerance, re-solve the design scenarios at L_C each from the no-load state (`_reset` before every solve). Classify the failure as **solver-history noise** if the canonical share ≥ threshold, else as a **screen error**. This classification never changes the C+ envelope. |
| B and C files | read only; never rewritten (checked by SHA-256 before and after) |

## Accounting (recorded per interval and direction)

- C's own: GENCO-screened rows (`power_flows`), exact calls (`ac_calls`), search wall-clock (`runtime_s`).
- C+ additions: `gate_calls`, `repair_calls`, `gate_s`, `repair_s`, `gated`, `repaired`, `exact_design_share_at_c`, `c_limit_kw`, `zero_infeasible_sentinel`.
- Totals: C+ exact calls = C exact calls + gate calls + repair calls; C+ search time = C time + gate time + repair time.
- Solver-noise re-solves and the fresh-feasibility recomputation are reported separately and **not** counted as search cost.

## Runs

- **Annual:** the 36 stored C runs (hourly), output `data_out/doe/cplus_annual/`.
- **Pre-registered 15-minute days:** 2024-06-28 (154 non-zero C envelopes) and 2024-12-15 (none, so no gates), output `data_out/doe/cplus_uk/`.
- Run on Hydra CPU nodes, one task per file. Job IDs, commands and the commit go in a run manifest.

## Reported (by `gencodoe.report_final` / `report_annual`, extended for C+)

- B / C / C+ envelope agreement: within tolerance, C or C+ exceeding B, both non-zero, both zero.
- C+ vs B capacity (envelope-hours).
- Exact calls: B, C, C+.
- Combined exact-call saving vs B: C 14.65% (measured), and C+ (expected ≤ 10.44% before repairs).
- Number of C envelopes failing the gate, split into solver noise and screen errors.
- Repair magnitude (L_C − L_C+).
- Held-out feasibility of repaired envelopes.
- Search wall-clock. This is CPU-node descriptive; the controlled comparison is the separate same-node A100 benchmark (Phase 3).

## Expected cost

- **Gates:** ≥ 183,200 exact solves (annual) + 30,800 (summer).
- **Repairs:** about 2,400–2,600 exact solves each (observed in the exploratory run).
- **Wall-clock:** at ~30 ms per solve on a Hydra zen4 core, about 1.8 core-hours for the annual gates, run in parallel per day; about 15 minutes for summer.
- **Exact-call saving vs B (annual):** falls from 14.65% (C) to at most 10.44% (C+) before repairs.

## Amendment 1 (before the formal launch, no C+ result produced under this protocol yet)

1. **The diagnostic covers every gate failure.**
   - The primary gate and repair stay B-compatible.
   - The canonical reset-from-no-load re-solve of the 200 design scenarios at L_C is now run for **every** gate failure, not only when C ≈ B.
   - Classification: **solver-history noise** if the canonical share ≥ threshold, otherwise **screen error**.
   - It is diagnostic only and never changes a C+ envelope. When the matching B result is available, |L_C − L_B| is recorded alongside.
2. **What "identical to B" means.**
   - C+ matches B's OpenDSS **configuration** (model build, constant-PQ loads, tolerance, retry rule) and its **scenario order within each gate or repair evaluation**.
   - It does **not** reproduce B's complete solve history: OpenDSS starts each solve from the previous solution, and B's search visited different levels in a different sequence. This is why a borderline state can flip, and why the canonical diagnostic exists.
3. **Accounting is separate per category:** gate calls and seconds, repair calls and seconds, verification calls and seconds, diagnostic calls and seconds.
   - Search cost = C's exact calls and time + gate + repair.
   - Verification and diagnostic are reported, never counted as search cost.
4. **Every four-wire metric is recomputed after a repair**, on the run's fresh draws at the new level:
   - `prob_ok_unbalanced`, `unbalanced_v_min`, `unbalanced_v_max`, `unbalanced_line_loading_max`, `unbalanced_draws`;
   - the balanced held-out `prob_ok_ac_fresh` on the run's balanced fresh draws;
   - the design-set share at the final level, as `prob_ok_oracle`, from the exact repair.
5. **Zero results are classified explicitly.**
   - `zero_infeasible_sentinel`: the final level is 0 and the design share at 0 is below the threshold, i.e. no feasible non-negative optional envelope.
   - `zero_feasible`: the final level is 0 but 0 passes.
   - For zeros that C produced itself (not gated), the class is taken from C's own design share (`prob_ok_oracle`) and labelled as C-screened, not exact.
6. **Inputs are protected by hash.**
   - Every source C file, and every B file used for the diagnostic, is hashed (SHA-256) before and after processing. The run fails if any hash changes.
   - The hashes are written into each C+ output and into the run manifest.
7. **The B/C report schema is preserved.**
   - `gencodoe.report_final` and `report_annual` keep their existing keys and values for B and C exactly.
   - C+ appears only as new rows (method "C+") and new keys with a `_cplus` suffix.
   - Consistency tests check this.
