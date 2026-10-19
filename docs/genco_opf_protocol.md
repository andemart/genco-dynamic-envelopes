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
