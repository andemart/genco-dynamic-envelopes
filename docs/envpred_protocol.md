# Envelope prediction with an exact gate: protocol (post-hoc, exploratory)

**Status: post-hoc, exploratory.** Committed together with the frozen split (`results/envpred_split.json`) before any feature is extracted and before any result is computed. It is reported separately from the pre-registered B / C / C+ results, which are not modified.

## Question

Can a learned predictor of the exact four-wire envelope replace most of B's exact search, and does GENCO make that predictor better than the same predictor without GENCO?

## Scope (fixed)

- Only the existing 180 labelled B files: the 36 annual test dates × 5 household placements (seed 101 from `data_out/doe/annual/`, seeds 202/303/404/505 from W1). No new B labels.
- GENCO is not retrained: the frozen v2 LV GENCO (`d42d5c34…`) as used everywhere.
- No four-wire fixed-point solver. OpenDSS stays the exact oracle.
- UK limits; 200 design scenarios; ε = β = 0.05; acceptance threshold 0.98; hourly.

## Data and split

- **Unit:** one interval and direction for one date and placement: 36 × 5 × 24 × 2 = 8,640 rows.
- **Label:** B's exact uniform level `L_B` (kW) from the stored file.
- **Inputs:** the interval's 200 design scenarios, replayed exactly as `run_day` drew them (scenario seed 101, the file's household seed, the same draw order). The run's fresh draws, also replayed, are used only for held-out feasibility.
- **Split by date** (`results/envpred_split.json`): season-stratified, 16 train / 4 validation / 8 calibration / 8 test dates. All five placements of a date stay together. The split was drawn by a fixed rule (seeded shuffle within season) and committed before feature extraction.
- The test dates were also used in earlier development (B / C / C+ reports), so the result is exploratory.

## Features (per interval, summarised over its 200 design scenarios)

Every quantity below is summarised by its mean, 90th percentile, 98th percentile and maximum over the 200 scenarios.

- **Phase-composition block (P):** net P and Q of each (main feeder × phase) group (12 groups), of each phase (3), and of the feeder (1). 32 series × 4 statistics = 128 features.
- **Classical block (C):**
  - the stored balanced-AC envelope level (`ac` row of the same file, same interval and direction), 1 feature;
  - the exact balanced sweep at zero flexibility: min V, max V, max line loading, transformer loading; 4 × 4 = 16 features.
- **GENCO block (G):**
  - GENCO's predicted state summary (same 4 quantities) at zero flexibility; 4 × 4 = 16 features;
  - GENCO final-layer bus embeddings pooled by mean and max over buses (192), averaged over the 200 scenarios; 192 features.
  - Deployment-equivalent inference (`GencoOracle`, float32, CPU).

## Arms

| Arm | Features | Contains GENCO |
|---|---|---|
| `P` phase-only | P | no |
| `PC` phase + classical | P + C | no |
| `PG` phase + GENCO | P + G | yes |
| `PCG` phase + classical + GENCO | P + C + G | yes |

## Head (frozen Option 2 settings)

- One `HistGradientBoostingRegressor` per arm and direction: squared error, `max_iter` 500, `learning_rate` 0.05, early stopping on the validation dates (explicit `X_val`, `y_val`), `random_state` 0. No hyperparameter search.
- Trained on all training-date rows, zeros included. Target: `L_B` in kW.
- No fitted preprocessing: the tree model takes raw features. Any preprocessing that were needed would be fitted on training dates only.
- Predictions are clipped to [0, cap] (cap = 5 kW export, 7.4 kW import).

## Issuing an envelope (identical for every arm)

1. **Shrink (gate-efficiency calibration).**
   - On the calibration dates, per arm and direction: residuals `r = L̂ − L_B`.
   - δ = the ⌈(n+1)·0.95⌉-th smallest residual, floored at 0.
   - Shrunk level `L_s = max(L̂ − δ, 0)`.
   - The shrink only sets how often the gate passes. **It is not the source of safety.**
2. **Predicted zero.** If `L_s` ≤ the search tolerance (1e-3 × cap), issue 0 with no exact call.
3. **Exact gate.**
   - Exact four-wire feasible share at `L_s` on the 200 design scenarios: OpenDSS, B-compatible configuration, scenario order (`cplus.FourWireOracle`).
   - If share ≥ 0.98, issue `L_s`.
   - **The gate enforces B's design-sample rule, so every issued non-zero envelope satisfies the same acceptance rule as B.**
4. **Fallback.**
   - If the gate fails: the exact k-section search on [0, `L_s`] (`cplus.repair`: 4 levels per round, tolerance 1e-3 × cap).
   - If level 0 fails, issue the zero sentinel (no feasible non-negative envelope).

## Metrics (test dates; per arm and direction, and combined)

- **Retention** = Σ min(issued, `L_B`) / Σ `L_B` over intervals (envelope-hours).
  - Using min(issued, `L_B`) means that solver noise cannot push retention above 100%.
  - **Excess over B** = Σ max(issued − `L_B`, 0) / Σ `L_B`, and the number of intervals with issued > `L_B` + tolerance, reported separately.
- **Exact calls:** gate + fallback, against B's stored exact calls on the same intervals.
- **Gate:** pass rate. **Fallback:** rate, and calls per fallback.
- **Zeros:** predicted zeros (no calls); predicted zeros where `L_B` > 0 (capacity lost); zero sentinels issued by the fallback; agreement with B's sentinels.
- **Runtime:**
  - feature extraction per interval, by block;
  - head inference;
  - gate and fallback seconds.
  - Descriptive, on one laptop CPU.
- **Fresh feasibility,** for every issued non-zero envelope:
  - the four-wire feasible share on the run's first 200 fresh draws (as in B);
  - the balanced share on all 1,000 fresh draws;
  - mean, and share ≥ 0.95, compared with B on the same intervals.
  - Not counted as search cost.
- **Head accuracy:** MAE of `L̂` against `L_B` (secondary).

## Choosing the best arm of each family (before the test dates)

- "Best non-GENCO arm" (P or PC) and "best GENCO-containing arm" (PG or PCG) are chosen on the **validation dates**, without any exact solve.
- Selection proxy: proxy retention Σ min(`L_s`, `L_B`) · 1[`L_s` ≤ `L_B` + tol] / Σ `L_B`, i.e. it assumes a gate pass exactly when `L_s` ≤ `L_B`. Combined over both directions.
- Ties go to the simpler arm (P over PC, PG over PCG).

## Decision rule: when is GENCO positive

GENCO is called positive **only if all four** hold on the test dates, for the best GENCO arm G against the best non-GENCO arm N (combined over directions):

1. Retention(G) − Retention(N) ≥ 3 percentage points.
2. Exact calls(G) ≤ Exact calls(N).
3. **Paired day-block bootstrap.**
   - Resample the 8 test dates within season (2 per season), with all placements, intervals and directions of a date kept together; 2,000 replicates, seed 0.
   - The 2.5th percentile of Retention(G) − Retention(N) must be > 0.
4. **Held-out feasibility is not worse.** Over each arm's issued non-zero envelopes:
   - the mean fresh four-wire share of G is at least that of N minus 0.005;
   - the share of envelopes with fresh four-wire share ≥ 0.95 for G is at least that of N minus 1 percentage point.

Otherwise the result is reported as "GENCO not positive". **There is no tuning on the test dates, and the analysis stops there.**

All four arms are reported in full whatever the outcome, including the comparison with B.

## Order of work

1. Commit this protocol and the split. *(This commit.)*
2. Implement feature extraction, heads, the gate and the report as a committed module, with tests.
3. Extract features for all 180 files, then fit the heads, the shrink and the validation selection.
4. Run the gate on the test dates (all four arms), then the report.
5. Log the result in `docs/experiment_log.md`, whichever way it goes.

## Amendment 1 (before any feature extraction or result)

**Problem.** The summary statistics above (mean, 90th and 98th percentile, max) capture only the upper tail over scenarios. Several binding quantities live in the lower tail: minimum voltage for imports, and the most negative net consumption (PV export) for exports.

**Change.** Every quantity is summarised by **7 statistics**: mean, min, 2nd, 10th, 90th and 98th percentile, and max over the 200 scenarios.

**Feature counts become:**
- P: 32 × 7 = 224;
- C: 1 + 4 × 7 = 29;
- G: 4 × 7 + 192 = 220.

No feature had been computed and no data inspected when this was written. Everything else is unchanged.
