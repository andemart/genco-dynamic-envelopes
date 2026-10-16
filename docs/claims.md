# Claims: what we state, and what we do not

Every number below is regenerated from committed results:
- `results/report_winter_2024-12-15.json`
- `results/report_summer_2024-06-28.json`
- `results/annual_report.json`
- C+ outputs, once added

Wording follows the external audit of 16 Oct 2026.

## Setting (always stated with any result)

- **Network:** an ENWL1 four-wire benchmark populated with Flemish (Fluvius 2024) demand profiles. It is not an observed Flemish feeder.
- **Population:** one household technology assignment (seed 101) and one technology mix (30% HP / 30% EV / 40% PV), with ENWL1's original single-phase connections.
- **Limits:** UK statutory voltage limits, 216.2–253.0 V.
- **Envelope:** uniform per household, optional and incremental, starting from zero. It is the extra import or export allowed on top of the passive forecast.
- **Feasibility is direction-specific:**
  - export envelopes are checked for overvoltage, cable loading and transformer loading;
  - import envelopes are checked for undervoltage, cable loading and transformer loading.
  This is a "do not worsen the direction-relevant constraint" policy, not unconditional statutory feasibility. In the annual B results, 60 of 472 non-zero export intervals had at least one fresh draw below the lower voltage limit, and 110 of 444 non-zero import intervals had at least one above the upper limit.
- **Guarantees are per interval.** They do not hold simultaneously over a day or a year.

## Claims we make

1. **Temporal DOEs recover optional headroom that a single limit cannot.**
   - On 36 held-out test days sampled through 2024, no positive uniform year-round optional allowance could be certified across the sampled intervals.
   - Annualized available envelope-hours per household:
     - export **4.81 MWh** (season-stratified day-bootstrap interval 4.35–5.30);
     - import **2.50 MWh** (2.11–2.87);
     - equivalent energy only if the limit were fully used in every interval.
   - These are envelope-hours, not delivered energy.
2. **Issued envelopes mostly meet the target empirically.** Non-zero exact-search (B) envelopes reached **97.4%** mean held-out four-wire feasibility, and **93.9%** of them (860/916) met the 95% target empirically. Only **47.3%** (433/916) passed the one-sided 95% lower-bound criterion: the 200-draw verification is underpowered for proving 95%.
3. **A balanced approximation can be badly unsafe on this unbalanced feeder.** Balanced-model envelopes are about 2–3× larger, and on the summer day they hold in only 2% (export) and 24% (import) of four-wire draws on average.
4. **GENCO can triage expensive four-wire simulations with almost exact agreement.**
   - C reproduced **1,727 of 1,728** exact-search decisions (hourly, 36 days, both directions) with **9–20% fewer** exact simulations.
   - C did **not** reduce CPU wall-clock time: total envelope search was +9%, measured on different nodes, descriptive only.
   - In one interval C returned **11.8×** B's import limit (2024-04-29 13:00). This shows that the pointwise conformal guarantee does not certify the final adaptive envelope decision.
   - C+ (post-hoc final exact gate) addresses this; its results and added cost are reported separately.
5. **GENCO adaptation to LV is necessary and explained.** Three transfer failures were found and repaired (impedance blindness, physics-loss collapse, flow reconstruction). The hybrid is about 5× more accurate than its physics prior on envelope rows.

## Claims we do not make

- That zero flexibility makes the network safe. Passive baselines are frequently infeasible, especially on winter evenings.
- Unconditional or statutory four-wire safety. Feasibility is direction-specific and pointwise.
- Delivered EV or PV energy, curtailment avoided, or customer utilization. No coherent EV sessions or utilization model is implemented.
- Results for a real Flemish feeder, other household placements, other technology mixes, other years, or other feeders.
  - The confidence intervals cover day-to-day variation within the 36 sampled days only.
  - Household placement alone changes the winter baseline violation rate from 15.8% to 35.1%.
- That hourly results equal 15-minute results. They were close on one comparison day: export 14.30 vs 14.20 kWh and import 19.13 vs 18.95 kWh on 28 June, the only non-zero 15-minute day.
- That GENCO is faster than exact solvers. On CPU it is not; specialised exact solvers (Z-bus, sweep) are comparable or faster on this 398-bus balanced feeder.
- That frozen C is strictly certified. It needs the final exact gate (C+).
- Multi-feeder generalisation. One model, one feeder topology.
- Reinforcement deferral or economic value. No cost or growth model.
