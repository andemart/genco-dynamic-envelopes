# Reporting plan for the amended pre-registered evaluation

This plan was committed after the corrected-limit jobs were submitted (13331847–13331850) and **before any of their outputs were opened**.

The script `python -m gencodoe.report_final` implements exactly these definitions. Its output is the only source of the numbers that are reported.

## Notation

- Days d ∈ {2024-12-15, 2024-06-28}. Directions: export and import.
- Intervals t = 1..96, with Δt = 0.25 h.
- L_{m,t} is the per-household envelope (kW) of method m ∈ {ac (balanced), B (exact four-wire), C (GENCO + four-wire screen)}.
- Feasibility is checked on independent fresh draws per interval and method:
  - four-wire: n = 200 draws, k feasible;
  - balanced: n = 1,000 draws.

## 1. Feasibility (per interval, direction and method)

- Empirical feasibility: p̂ = k / n.
- **One-sided 95% Clopper–Pearson lower bound:** LB = Beta⁻¹(0.05; k, n − k + 1), and LB = 0 when k = 0.
- Both are reported, for the four-wire draws (primary) and the balanced draws (secondary).

## 2. Safety summary

For each day, direction and method:
- the number of intervals with LB ≥ 0.95, the certified intervals;
- the number with p̂ ≥ 0.95;
- the minimum p̂.

These are pointwise statements per interval, not a whole-day guarantee.

## 3. Static baseline

L_static,d = min_t L_{B,t,d}, separately for each day and direction.

This is the largest flat per-household limit that the exact four-wire method accepted in every interval of that day.

## 4. Capacity

- Per household: C_m = Σ_t L_{m,t} Δt, in kW·h per household per day.
- Feeder total: 200 × C_m.
- Reported for ac, B, C and static. Mean limits are reported as well.
- Relative uplift (C_B − C_static) / C_static is reported **only if C_static > 0**; otherwise only absolute values.

## 5. B–C agreement

Over all intervals, separately per day and direction:
- mean and maximum |L_C − L_B|;
- the fraction of intervals agreeing within the search tolerance, tol = 1e-3 × cap (5 kW export, 7.4 kW import);
- the number of intervals with L_C > L_B + tol, and the largest excess;
- the number of intervals where both are zero, reported separately, because agreement at zero is weak evidence.

## 6. Runtime and exact calls

- **Envelope-search wall-clock per interval** (independent verification excluded): total, median, 95th percentile and maximum.
- **Exact four-wire (OpenDSS) calls:** total and per interval.
- B and C ran as separate jobs, possibly on different nodes. Runtime ratios are therefore **descriptive measurements**, not hardware-controlled speed-ups.

## 7. Integrity checks (the script fails if any of these does not hold)

- Every input file records `voltage_limits.jurisdiction == "uk_esqcr"` and vmin/vmax in pu matching `gencodoe.limits`.
- Scenario and household seeds as pre-registered (101 / 101); `steps_per_hour == 4`; 96 intervals per direction and method.
- The SHA-256 of each input file is written to the report.
