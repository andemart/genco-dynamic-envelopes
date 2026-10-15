# Pre-registered final evaluation

Written and committed **before** any run on these days. The method is frozen at this commit:

- GENCO hidden 12, LV + linear prior (Hydra job 13326572);
- binned conformal margins;
- four-wire margins with transformer criterion and alpha split over 4 criteria;
- finite-sample acceptance rule.

Machine-readable copy: `results/preregistration.json`.

## Evaluation days

| Season | Day | How chosen |
|---|---|---|
| Winter | 2024-12-15 | seeded draw (`default_rng(20261015)`) from the 9 Dec–Feb test days |
| Summer | 2024-06-28 | same draw from the 8 Jun–Aug test days, excluding 2024-07-08 |

2024-07-08 was used during development, so it is excluded.

## Protocol

- **Resolution:** all 96 quarter-hours.
- **Envelopes:** 200 scenarios per interval; ε = 0.05; β = 0.05, so a level is accepted only if ≥ 98% of scenarios are feasible.
- **Four-wire screen:** α = 0.01, split over voltage-min, voltage-max, line loading and transformer loading.
- **Verification:** independent fresh draws per interval:
  - 1,000 on the balanced AC model;
  - 200 on the four-wire OpenDSS model.
- **Seed:** 101.
- **Methods:**
  - balanced AC;
  - option B (exact four-wire search);
  - option C (GENCO + four-wire margins, OpenDSS fallback).

## What is claimed from it, and nothing more

- **Per interval and direction:** four-wire out-of-sample feasibility (empirical, with a binomial confidence interval) against the 0.95 target. This is a pointwise statement, not a whole-day one.
- **Capacity:** of C vs B vs balanced, and vs a static four-wire-safe limit.
- **OpenDSS calls and measured end-to-end wall-clock** of B and C.

## Status of earlier data

- The v2 test days were inspected while developing the four-wire screen (collapse handling, bin logic, choice of α). They are therefore an **exploratory** set, not a pristine holdout.
- The v2 calibration days were used only to fit margins.
