# GENCO-OPF envelopes: protocol (new exploratory layer)

**Status: exploratory.** Branch `ander-genco-opf`.
- Committed before any OPF label is generated and before any model is fitted.
- Existing results (B/C/C+, annual, W1–W3, Option 2, envpred3, the four-wire extension) are untouched.
- Proposed by E. (GENCO team): compute **optimal** envelopes with OPF instead of uniform or proportional envelopes found by power flows.

## 1. Question

> Does a GENCO-OPF proposal, followed by an exact check and scale-down, give AC-optimal-quality, safe envelopes at a fraction of the cost of solving AC-OPF with IPOPT?

**Why this gap matters.** Our only optimal method so far, the max-total LinDistFlow LP, fails the AC check (P(ok) 0.856 against a 0.95 target, in the case33bw demo). Per-scenario AC-OPF with IPOPT is correct, but costs about 20 ms per solve.

## 2. Two stages

| Stage | Network | Exact check | Role |
|---|---|---|---|
| **A** | case33bw (balanced, 12.66 kV, 32 load buses) | batched exact balanced AC sweep | GENCO's native setting: the clean test of the claim |
| **B** | ENWL1 four-wire benchmark with Fluvius households | iterated four-wire sweep (validated to 3.6e-6 pu against OpenDSS), with OpenDSS confirmation on the final envelopes | the application |

**A runs first. B starts only after A's result is committed.** B's ENWL1-specific choices are frozen in an amendment before B runs.

## 3. Envelope method (identical for every optimiser)

For each interval and direction, with design scenarios s = 1..S (S = 200, ε = β = 0.05, acceptance threshold 0.98):

1. **Per-scenario OPF.**
   - Each flexible customer i is a dispatchable unit at its bus:
     - export: 0 ≤ P_i ≤ cap_i;
     - import: −cap_i ≤ P_i ≤ 0.
   - The passive load is that scenario's net load.
   - Constraints: AC power flow, voltage limits, line ratings, transformer rating.
   - **Objectives:**
     - **hosting** (primary): maximise Σ|P_i|;
     - **fairness** (secondary): maximise Σ|P_i| − w·Σ P_i² with w fixed in A's amendment, so allocations spread.
   - Each scenario gives an optimal allocation x_i^s.
2. **Quantile proposal.** x_i = the ε-quantile of {x_i^s} across scenarios (the lower ε-quantile of |x_i|, signed by direction).
3. **Exact check and scale-down.**
   - Check the vector x jointly on all S design scenarios with the exact model.
   - If fewer than 98% are feasible, find the largest λ ∈ [0, 1] such that λ·x passes, by k-section search (4 levels per round, tolerance 1e-3).
   - This exact step **enforces the design-sample rule for every method**, whatever proposed x.
4. **Held-out verification.** Fresh scenarios (1,000), reported separately.

## 4. Methods compared

| Method | Proposal x | Notes |
|---|---|---|
| **Uniform** | existing exact uniform search | the baseline that hosting gains are measured against |
| **LP** | max-total LinDistFlow LP per scenario | the existing optimal-but-approximate method, now with the same exact scale-down |
| **IPOPT-OPF** | AC-OPF per scenario (PowerModels + IPOPT) | the reference optimum |
| **GENCO-OPF** | GENCO in OPF mode per scenario | the method under test |

## 5. GENCO-OPF model

- **Task:** graphkit `OptimalPowerFlow`, with customers as generators (Pg bounds, linear or quadratic costs). Outputs are Pg per unit and bus Vm, Va, Qg.
- **Encoder:** our LV variant with asinh admittance features. The linear PF prior is not used: it needs known injections, which OPF does not have.
- **Labels:** datakit OPF mode with customer generators.
- **Disclosed workarounds:**
  - datakit's slack pinning (already handled for PF);
  - negative-cost units. Preferred formulation: a positive cost on slack import with zero-cost customers for export; a positive cost on customer Pg ∈ [−cap, 0] for import. Losses are included in the objective; this is disclosed.
- **Data:** OPF labels on training intervals only, with validation intervals for early stopping. Evaluation intervals are disjoint (section 6).
- **Runs:** one smoke fit and one final fit per stage. The configuration is frozen in an amendment before the final fit.

## 6. Stage A setting (frozen now)

- **Network:** pandapower `case33bw`, exported to MATPOWER, slack voltage 1.0 pu.
  - Voltage limits 0.95–1.05 pu.
  - Line ratings from the case: placeholder ratings of 1,000 A or more become inf. If no finite ratings exist, line rating = 1.5 × each line's peak base-case loading. Fixed now, applied by code, reported.
- **Flexible units:** one per load bus (32), cap_i = 1.5 × the bus's nominal active load, in both directions.
- **Scenarios:**
  - Load = nominal × a fixed 24-hour residential profile × lognormal day factor (σ 0.15) × per-bus Gaussian copula noise (relative std 10%, correlation 0.6).
  - PV on every second load bus: capacity = 1.0 × nominal load, a clear-sky 24-hour shape × day cloud factor U(0.3, 1.0), relative std 20%.
- **Days:** 120 synthetic days (seed 0): 80 train, 20 val, 20 test. Hourly intervals.
  - Training labels: 2,000 scenario states per training day (160,000 OPF solves), drawn from the training days' scenario distributions.
  - Evaluation: the 20 test days × 24 h × 2 directions; S = 200 design and 1,000 fresh scenarios per interval.
- **Seeds:** a scenario stream separate from training (seed 101).

## 7. Metrics (test intervals; per method and direction, and combined)

- **Hosting:** total envelope (Σ_i limit × 1 h) per day, and its gain over uniform (%).
- **Gap to IPOPT-OPF:** (hosting_IPOPT − hosting_method) / hosting_IPOPT, after the exact scale-down.
- **Safety:** design share at the issued envelope (≥ 0.98 by construction); held-out P(ok) mean, and share of intervals ≥ 0.95.
- **Fairness:** Jain's index of the per-customer limits; share of customers with limit 0.
- **Cost:**
  - proposal wall-clock (GENCO: A100 batch inference; IPOPT: **parallel**, N CPU cores, N reported);
  - exact check and scale-down time and calls;
  - end-to-end time.
  - Hardware is always stated.

## 8. Success criteria (Stage A; all required for "GENCO-OPF positive")

1. **Near-optimal:** GENCO-OPF hosting ≥ 95% of IPOPT-OPF hosting, combined over directions.
2. **Better than the cheap optimal method:** GENCO-OPF hosting ≥ LP hosting × 1.05, both after exact scale-down.
3. **Safe:** held-out P(ok) no worse than IPOPT-OPF's minus 0.005 (mean) and minus 1 point (share ≥ 0.95).
4. **Cheaper:** GENCO-OPF proposal wall-clock below parallel IPOPT's at the stated hardware.
   - Criterion 4 is reported separately. If 1–3 pass and 4 fails, the claim is "near-optimal and safe, not faster", **not** positive on cost.

**Uncertainty.** Day-block bootstrap over the 20 test days (2,000 replicates, seed 0) for the hosting ratios.

## 9. Claim discipline

- **The exact check guarantees the design-sample rule for every method.** Differences between methods are in hosting and cost, never in nominal safety.
- No claim of unconditional safety; no claim beyond the stated networks.
- Negative results are reported. In particular, the gain may come from accuracy over the LP rather than speed over parallel IPOPT.

## Amendment 1 (2026-10-19, before any label, fit or method comparison)

Found while building stage A. Checked on **training days only** (days 0, 13, 27, 41, 55 and 69; 5 scenarios per hour; 720 solves per direction). No evaluation day was used.

1. **The slack is pinned at 1.0 pu** (V_min = V_max = 1.0 at the root). Unpinned, the OPF raised the substation voltage to 1.05 pu to host more import. The protocol already specified 1.0 pu; the code now enforces it.

2. **Export caps raised to 3.0 × nominal load. Import caps stay at 1.5 ×.**

   | Setting | Network-limited solves (< 99% of caps) | Mean hosted / caps |
   |---|---|---|
   | Export at 1.5× | 2.4% | 99.9% |
   | **Export at 3.0×** | **100%** | **78.7%** |
   | Export at 5.0× | 100% | 49.5% |
   | Import at 1.5× | 100% (network always binds) | 55.1% |

   At 1.5× every method would return about the caps, so export would be uninformative. Import is already informative at 1.5×.

3. **Infeasible per-scenario OPF.** 124 of 720 import solves were infeasible: the passive load already violates the voltage limits.
   - For such a scenario, x_i^s = 0 for every unit (no feasible non-negative allocation; the zero-sentinel analogue).
   - The ε-quantile then includes these zeros.
   - GENCO-OPF does not detect infeasibility. Its prediction is used as is, and the exact check and scale-down decide.
   - Counts are reported per method.

4. **Fairness weight (the secondary objective):** a quadratic cost w·P_i² on every unit, with w = 1 / (2 · mean cap of the direction) and a slack cost c1 = 1. The primary objective stays maximum hosting.

5. **IPOPT timing.**
   - The reported per-solve time is the `solve_ac_opf` wall-clock.
   - Python↔Julia data-update overhead is reported separately.
   - "Parallel IPOPT" means N independent Julia processes on one Hydra zen4 node, N reported; throughput = states / wall-clock.

## Amendment 2 (2026-10-19, before any label, fit or test-day evaluation): option 1, the multi-scenario OPF

### Finding that motivated it (validation day 80 only; development data)

Per-scenario AC-OPF + per-customer ε-quantile **does not keep optimality**:
- each per-scenario optimum sits on its own scenario's voltage limit, and uses different customers' headroom;
- the stitched quantile vector then fails jointly and is scaled down hard.

| Import, 03:00 | Proposal after quantile | After the exact check (λ) |
|---|---|---|
| IPOPT per scenario | 2.89 MW | 0.38 MW (λ = 0.13) |
| LinDistFlow LP per scenario | 1.44 MW | 1.10 MW (λ = 0.77) |

The cheap LP beat per-scenario IPOPT after the check in most tested intervals. GENCO-OPF imitating per-scenario IPOPT would inherit this and fail criterion 2 by construction.

**Also corrected:** the per-scenario OPF let units supply reactive power, which the exact check does not model. From now on units inject **active power only**, and each OPF imposes only the constraints relevant to its direction, exactly as the exact check does: export checks overvoltage and loading; import checks undervoltage and loading.

### New reference optimum: the multi-scenario AC-OPF (`gencodoe.opf_multi`)

**Formulation.** For each interval and direction, one AC-OPF with a **shared** limit vector x across the design scenarios (the scenario approach):

> maximise Σ x_i, subject to, for each kept scenario: polar AC power flow with unit injection ±x_i (P only), the direction-relevant voltage bound, sending-end branch |S| ≤ rating, and 0 ≤ x_i ≤ cap_i.

**Discarding** follows the design rule, which allows S − ⌈0.98·S⌉ = 4 of 200 scenarios to fail:
1. Scenarios infeasible at x = 0 are discarded first. If more than 4 are, the envelope is 0 (zero sentinel).
2. The rest of the allowance is spent greedily: drop the kept scenario with the smallest direction-relevant headroom at the current x, re-solve, and stop when hosting no longer grows.

**Solver.** JuMP (PowerModels' internal copy) with IPOPT; the exact check and scale-down follow as for every method.

**Validation-day 80 result** (development): it passes the exact check unchanged (λ = 1), hosts 6–13% more than the LP after the check in the 5 tested non-zero intervals, and costs **7–24 s per interval and direction**.

### GENCO's role (replaces section 5)

A **GENCO-style graph model amortises the multi-scenario OPF**. It maps the interval's scenario statistics to the shared optimal vector, x̂_i / cap_i ∈ [0, 1], followed by the same exact check and scale-down.

**Architecture.**
- The encode–process–decode TransformerConv processor family used in the four-wire extension, with asinh-scaled impedance and rating edge features and no final LayerNorm in the encoders.
- **Trained from scratch** on case33bw. There is no pretrained GENCO checkpoint for this network, and this is disclosed.

**Inputs.**
- **Node:** per bus, the 7 statistics (mean, min, p02, p10, p90, p98, max) over the 200 design scenarios of net P and of Q; cap; PV flag; root flag; depth; direction flag.
- **Edge:** r, x, rating.

**Labels.** Multi-scenario OPF optima on **training** days, with validation days for early stopping. Test days are only evaluated once, in the frozen evaluation.

**Baselines for the learned model.** An MLP on the same features flattened (same budget, same training rule), so that any gain can be attributed to the graph structure.

### Methods in the final evaluation (test days)

- uniform;
- LP (per scenario, quantile, check);
- IPOPT per scenario (quantile, check);
- **multi-scenario OPF (the reference)**;
- GENCO-style graph model + check;
- MLP + check.

### Success criteria (replace section 8)

GENCO-style versus the reference and the LP, after the exact check:

1. Hosting ≥ 95% of the multi-scenario OPF.
2. Hosting ≥ 1.05 × LP.
3. Held-out P(ok) no worse than the multi-scenario OPF's minus 0.005 (mean) and minus 1 point (share ≥ 0.95).
4. Proposal plus check wall-clock below the multi-scenario OPF's, reported per hardware.
   - The claim "near-optimal at a fraction of the cost" needs 1–4.

**Also reported:** GENCO-style versus the MLP. A graph-structure claim needs the GENCO-style model to beat the MLP by at least 2 points of hosting relative to the reference, with a day-bootstrap lower bound above 0.
