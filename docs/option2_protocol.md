# Option 2 protocol: GENCO embeddings + a learned four-wire head (post-hoc, exploratory)

**Status: post-hoc, exploratory.** Committed before any Option 2 result is computed.

It is reported separately from the pre-registered B / C / C+ results, which are not modified.

## Question

Do GENCO's learned representations help predict **four-wire** risk better than a classical balanced solver does, when both get the same phase-composition information?

Context: with the frozen four-wire conformal adapter, GENCO, exact balanced AC (Z-bus/sweep) and a single fixed-point iteration all screen equally poorly (`results/screen_comparison_genco_vs_classical.json`).

## Data (same calendar-day splits as everywhere)

| Split | States | Use |
|---|---|---|
| v2 train days | 14,648 | fit the heads |
| v2 val days | 5,066 | early stopping and model choice only |
| v2 calib days | 2,953 | conformal margins of the heads |
| v2 test days | 2,968 | evaluation (exploratory: also used in earlier development) |

**Four-wire truth** comes from `gencodoe.unbalanced` (constant-PQ loads, UK limits) for every state. It already exists for calib and test, and is computed for train and val with the same function.

## Targets (state level)

Four-wire worst customer phase-neutral voltage (min and max), worst cable phase loading, and transformer loading.

Non-converged (collapse) states are excluded from head *training*. They are kept in calibration, where they get an infinite score in their bins exactly as in the frozen adapter (`FourWireMargins`), and in evaluation, where they count as violating.

## Feature arms (identical head and protocol; only the balanced input differs)

All arms share the same **phase-composition features**: net P and Q summed per (main feeder × phase) group (4 feeders × 3 phases × P, Q = 24), per-phase totals (6) and feeder totals (2). The phases and the feeder membership come from the ENWL1 model.

1. **phase-only:** phase-composition features alone; no balanced model.
2. **classical:** + the exact balanced-AC state summary (min V, max V, max line loading, transformer loading), i.e. what Z-bus or the sweep provides.
3. **GENCO:** + the GENCO-predicted state summary (same four quantities) + GENCO bus embeddings pooled over buses (mean and max of the 96-dimensional final-layer embedding, 192 features). The frozen GENCO is used, deployment-equivalent inference, float32.

## Head and calibration (fixed now)

- **Head:** one `sklearn.ensemble.HistGradientBoostingRegressor` per target; squared error; `max_iter` 500; `learning_rate` 0.05; early stopping on the val days (validation_fraction unused, explicit val set); `random_state` 0.
- **No hyperparameter search.** The same settings for every arm.
- **Calibration:** the head predictions replace the balanced state summary in the frozen four-wire screen (`gencodoe.four_wire.FourWireMargins`): same Mondrian bins, α = 0.01 split over 4 criteria, fitted on the calib days.

## Reported (test states, per direction and arm)

- Share of states screened safe, certainly violating, and sent to OpenDSS.
- False-safe count against four-wire truth.
- Share of feasible states not cleared.
- Head accuracy on test (MAE of each target), as a secondary measure.
- The frozen-adapter GENCO screen as reference.

## What counts as evidence for GENCO

The GENCO arm must clear clearly more states safely than the classical arm, with false-safe staying at 0 or comparable. "Clearly" means at least 5 percentage points more states screened safe in a direction. The comparison is exploratory, and the result is reported whichever way it goes.

## Not done in this step

DOE-level integration (envelopes and capacity with the new head) follows only if the screening result supports it. It would then be a separate exploratory run.
