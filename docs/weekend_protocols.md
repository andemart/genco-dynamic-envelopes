# Weekend analyses: protocols (post-hoc, exploratory)

All three analyses below are post-hoc and exploratory. They are reported separately from the pre-registered B / C / C+ results, which are unchanged.

This file was committed before any of the runs were submitted.

Fixed throughout, unless stated otherwise:
- UK limits (`uk_esqcr`);
- the 36 test days, hourly;
- 200 design scenarios; ε = 0.05; β = 0.05 (acceptance ≥ 98%);
- verification on 1,000 balanced and 200 four-wire fresh draws per interval;
- scenario seed 101;
- the exact four-wire search (B-type, OpenDSS) as the only oracle. No GENCO is involved.

## W1. Placement robustness of the exact DOE result

**Question.** How much do the annual envelope-hours and the share of certifiable intervals depend on which households own heat pumps, EVs and PV, and on which phase they sit?

**Design.** Exact four-wire search (method `b_opendss_exact`) on the 36 test days, with household seeds **202, 303, 404, 505**. Seed 101 is the existing result. Everything else is identical to the annual extension.

**Reported.**
- Per seed: annual export and import envelope-hours per household (the `report_annual` definition), share of intervals with a non-zero envelope, share of baseline-infeasible sentinels, and held-out four-wire feasibility of non-zero envelopes (mean, share ≥ 0.95).
- Across the 5 seeds: minimum, maximum and mean.

The spread across seeds is reported as **placement sensitivity**. It is not a confidence interval.

## W2. Idealised multi-phase connection of heat-pump and EV households (sensitivity)

**Question.** How much of the four-wire congestion comes from connecting every household single-phase?

**Design.** As W1, but households with a heat pump or an EV are connected **three-phase**: balanced over phases a, b and c, with the same total kW and kvar. Other households stay single-phase. Seeds 101 and 202.

For a three-phase household, the worst of its three phase-neutral voltages counts.

This is labelled an **idealised multi-phase customer connection**. It is not a claim about how existing Belgian HP/EV homes are actually connected.

**Reported.** The same metrics as W1, compared seed by seed with the single-phase result.

## W3. Spatial (group-differentiated) exact envelopes

**Question.** In intervals where a uniform envelope exists, how much more optional capacity do location- and phase-aware envelopes release, with the same joint certification?

**Groups.** Main feeder × phase: the subtree under each child of the transformer bus, crossed with the customer phase. That gives 12 groups.

**Method (per interval and direction).**
1. **Solo headroom.** For each group g, the exact uniform envelope for that group's customers only, with all other customers at zero flexible injection: w_g ∈ [0, cap]. Same search rule, threshold, levels per round and tolerance as B.
2. **Joint certification.** One exact search on λ ∈ [0, 1], with each customer's limit = λ · w_{g(i)}. Every group uses its limit simultaneously, and acceptance is the same 98% rule on the same 200 design scenarios, so the issued vector is jointly certified like B.
3. **Verification.** Of the issued vector, on fresh draws as in B.

If the baseline is infeasible (the zero sentinel), every w_g and λ are 0, exactly as for B. Spatial envelopes cannot unlock such intervals under a feeder-wide chance constraint; that is stated, not hidden.

**Seed.** Household seed 101, so the results pair with B and C.

**Reported.**
- Mean per-household envelope (kW) and envelope-hours: spatial vs uniform B, per direction, season and day.
- Per-group envelopes and the share of customers better or worse off than uniform.
- Exact calls.
- Held-out feasibility, as in B.
- A fairness note: customers in the binding groups may receive less than under uniform.

## W4. Runtime scaling (speed only)

**Question.** At what network size does GENCO-style inference become cheaper than dense classical solvers?

**Design.**
- Synthetic radial networks of 398, 2,000, 10,000 and 50,000 buses, with random trees and impedances in the ENWL1 range.
- Per-state time of:
  - GENCO inference: the same architecture (GNS_heterogeneous, hidden 12, 12 layers) on the A100, with untrained weights and **no** linear prior, because the prior uses a dense Z;
  - dense Z-bus fixed point;
  - the radial sweep on CPU.
- Batch 1,000 states; 3 repetitions; median and spread.

**Not claimed:** any accuracy on these networks. This is speed only.

## Amendment (2026-10-16): W4 dropped before any run

W4 is dropped. GENCO inference, the radial sweep and sparse fixed-point solvers all scale linearly in the number of buses; only the dense Z-bus scales worse. A comparison against the dense solver would therefore be a straw man, and the submission already states that GENCO has no speed advantage over specialised radial solvers here. No W4 job was submitted.
