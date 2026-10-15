# Pre-registered annual extension: all 36 test days

Committed **before** any annual job was submitted, and **before** the corrected summer result (2024-06-28, jobs 13331849/13331850) was opened. At that point only the winter 2024-12-15 corrected result had been seen.

## Why

DOE benefit is seasonal, so two days cannot support annual statements. Only the 36 test days are clean for option C: GENCO and its margins were trained, selected and calibrated on train, val and calib days. The test days are the d mod 5 = 4 blocks, spread over the whole of 2024, so they form a stratified sample of the year (about 10% of days).

The days are listed in `results/annual_test_days.txt`: 9 winter (Dec–Feb), 9 spring, 8 summer, 10 autumn.

## Protocol (identical to the pre-registered days except the resolution)

| | Value |
|---|---|
| Resolution | 24 hourly intervals per day (`--steps-per-hour 1`), for cost; the 15-minute evaluation remains the two pre-registered days |
| Methods | balanced AC (reference), B (exact four-wire search), C (GENCO + four-wire margins, OpenDSS fallback) |
| Policy | uniform envelope per household; **no group envelopes** |
| Search | 200 scenarios, ε = 0.05, β = 0.05 (≥ 98% feasible), α = 0.01 over four criteria |
| Seeds | scenario seed 101; household seed 101 (same population as the pre-registered days) |
| Verification | 1,000 balanced and 200 four-wire fresh draws per interval |
| Limits | UK 216.2–253.0 V (`uk_esqcr`), pinned in the job script |
| Model | frozen GENCO (`mlruns/877677247744645366/d42d5c34…`) |

Days 2024-06-28, 2024-07-08 and 2024-12-15 are test days, so they are included at hourly resolution as part of the sample.

## Reporting (same definitions as `docs/reporting_plan.md`, with Δt = 1 h)

- **Per day and direction:** safety (intervals with one-sided 95% lower bound ≥ 0.95), capacity Σ_t L_t Δt for ac / B / C / static, B–C agreement, search time and exact calls.
- **Static baseline:**
  - per day: L_static,d = min_t L_B,t,d;
  - a single **annual** static limit: L_static = min over all 36 days and intervals of L_B.
- **Annual estimates:** mean daily capacity over the 36 days × 366, per method and direction, with a bootstrap 95% CI over days, stratified by season. Relative uplift is reported only where the static capacity is > 0.
- **Seasonal breakdown:** the same metrics per season.
- These are **estimates from a 36-day stratified sample**, not a full-year simulation.
