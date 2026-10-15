# Pre-registration amendment 1: voltage limits

Committed before any corrected-limit run. It amends `docs/preregistration.md` and `results/preregistration.json`.

## Statement

On 15 October 2026, after the original runs had begun, we discovered that voltage limits supplied in per unit had been interpreted on the feeder's 240.18 V no-load base, rather than converted from the applicable absolute limits.

Before any corrected-limit run, we amended the primary criterion to the ENWL/UK statutory range of **216.2–253.0 V** (230 V −6% / +10%). On the model base this is **0.9002–1.0534 pu**.

The following remain unchanged:
- the model (frozen GENCO, hidden 12);
- the calibration residuals and conformal bin edges;
- the scenario protocol;
- the evaluation dates (2024-12-15, 2024-06-28);
- the seeds (scenario seed 101; households seed 0);
- the search procedure (200 scenarios, ε = 0.05, β = 0.05, α = 0.01 split over four criteria).

Results computed with 0.94–1.10 pu (225.8–264.2 V) are kept only as a superseded audit record:
- `data_out/doe/superseded_wrong_voltage_base/` on the laptop and on Hydra;
- `results/superseded_wrong_voltage_base/`.

Their speed-up is not used as a headline.

## Why these limits, and why the same days

- **The limits.** ENWL applies the UK statutory range on its network, and the feeder model is ENWL1, so this is the unambiguous primary choice. It is externally mandated, not tuned to the results.
- **The days.** The corrected evaluation reruns the **same two pre-registered days with unchanged seeds**, as a paired rerun. Choosing new days after seeing the winter result could look like selecting favourable conditions. New randomly selected days may be added later only as a robustness extension, never as a replacement.

## Sensitivity (secondary)

The **EN 50160 voltage band**: 207–253 V, 0.8619–1.0534 pu, selected with `GENCODOE_VOLTAGE_LIMITS=en50160_band`.

This is a voltage-band sensitivity for Belgian practice, not a claim of full EN 50160 compliance. The standard also has measurement-duration and aggregation requirements that this study does not model.

## What is recomputed

The architecture and trained weights are unaffected. Everything that depends on the feasibility thresholds is recomputed:
- screen shares (safe / uncertain / violated);
- fallback and OpenDSS-call counts;
- false-safe validation;
- runtime;
- all envelopes and capacity comparisons;
- near-limit error metrics.

The conformal residuals and the trained surrogate stay frozen, because they estimate prediction errors, not feasibility.

## Implementation

- `src/gencodoe/limits.py` holds the 416 V / √3 base and the absolute limits per jurisdiction, and derives the per-unit values.
- Every check imports from it: `feeder`, `evaluate`, `four_wire`, `unbalanced`, `diagnose_baseline`, `conformal_eval`, `doe_day`.
- Every result file records `voltage_limits`: jurisdiction, base, absolute and per-unit limits.
- Tests:
  - boundary tests just above and below both limits, for the balanced and four-wire checks;
  - a consistency test that all modules use the same constants.
