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

## Amendment 2 (before any head is fitted or any result computed): more labels, full-year test

**Why.** To get the best-powered answer, the user decided to use Hydra for new B labels. At the time of writing:
- the features of the 180 existing files have been computed (only their shapes and timings were checked);
- no head has been fitted;
- no prediction, gate or metric exists.

**New split** (`results/envpred_split_v2.json`; supersedes `results/envpred_split.json`, which is kept). It is the global calendar-day split already used for GENCO (`datakit_fluvius.day_split`):

| Role | Dates | Placements | Labels |
|---|---|---|---|
| train | 220 (day % 5 ∈ {0, 1, 2}) | 101, 202 | new B runs |
| val (early stopping, family selection) | 73 (day % 5 = 3) | 101, 202 | new B runs |
| calib (shrink δ) | 37 (held-out, (day // 5) even) | 101, 202 | new B runs |
| **test** | **all 36 annual test dates** | 101, 202, 303, 404, 505 | the existing 180 files |

**New labels.**
- Command identical to the annual / W1 B runs: `doe_day`, hourly, 200 scenarios, β = 0.05, scenario seed 101, `--verify-draws 1000`, methods `ac` and `b_opendss_exact`, UK limits.
- Two differences only:
  - `--any-split`, which lets the runner process non-test dates;
  - `--unbalanced 0`: no four-wire fresh verification, which labels do not need.
- The random stream, and so the design scenarios, are unchanged.
- Script: `scripts/hydra_envpred_labels.slurm`; list of dates: `results/envpred_label_days.txt`.

**Completeness rule (fixed now).**
- Heads are fitted only after every label task has finished.
- A task that fails twice is excluded, listed in the report, and its date is dropped for that placement.

**Bootstrap.** Over the 36 test dates, resampled within season (9 per season).

**The test gate** runs on Hydra, one task per test file, with the same code.

**Disclosure: GENCO in-sample dates.** GENCO was trained on states from the train and val dates. The design scenarios are new draws, but they come from the same days. GENCO features on those dates may therefore be more accurate than on new dates. The shrink is calibrated on calib dates and the test is on test dates; neither was used in GENCO training. The bias, if any, affects only how well the GENCO arms' heads are fitted, not the gate's safety.

Everything else (features incl. amendment 1, arms, head, shrink, gate, fallback, metrics, decision rule) is unchanged.

## Amendment 3 (before any head is fitted, any prediction made, or any result inspected): action-conditioned GENCO features, integrity, strata, frontier

**State when committed.**
- No head has been fitted.
- No prediction, gate result or test metric exists.
- The 660 amendment-2 label jobs are still running; none had finished.

### Why: feature symmetry and action alignment

- The classical block contains the balanced-AC envelope `L_ac`. That is already an *action-conditioned* optimisation result: the outcome of a feasibility search over flexibility levels.
- The GENCO block contains state predictions and embeddings only at **zero** flexibility. GENCO is therefore asked to infer the response to flexibility from one operating point, while the classical arm gets the directly relevant envelope.
- This amendment gives GENCO the analogous action-conditioned information. The G block is **augmented**, not tuned after results.

### What is unchanged

- The P and C feature definitions (amendment 1).
- The exact B labels.
- The split (amendment 2), the head and its settings, the gate and fallback.
- The primary operating point (95% shrink).
- The strict primary decision rule, including criterion 4 as defined above.
- Every test result is reported, negative results included.

### New block GA (per interval *and direction*, from its 200 design scenarios)

All GA features use the frozen GENCO checkpoint as a **balanced** feasibility oracle (`envelopes.exact(GencoOracle, feeder)`). The balanced, direction-specific feasibility definition, UK limits, uniform policy, signs and caps (5 kW export, 7.4 kW import) are the same as elsewhere. No OpenDSS, no four-wire truth, no four-wire conformal adapter and no B envelope is used.

**A. GENCO balanced envelope proposal `L_G`** (1 feature)
- `envelopes.scalar_envelope` with GENCO as the oracle: the same 200 scenarios, ε = β = 0.05 (threshold 0.98), 4 levels per round, tolerance 1e-3 × cap.
- `L_G` is the GENCO analogue of `L_ac`.

**B. GENCO response at the proposal** (85 features)
- At `L_G`, GENCO's predicted min V, max V, max cable loading and transformer loading, each over the 200 scenarios with the 7 frozen statistics (mean, min, p02, p10, p90, p98, max): 28.
- Change from zero to `L_G`, per scenario, then the 7 statistics: 28.
- Headroom at `L_G` relative to the active limits, then the 7 statistics: 28.
  - Headroom quantities: min V − VMIN, VMAX − max V, 1 − cable loading, 1 − transformer loading.
  - Disclosure: these are affine transforms of the absolute summaries, so they are redundant for a tree model. They are included as specified.
- GENCO feasible share at `L_G`: 1.
- Indicator `L_G` = 0: 1.

**C. Fixed response curve** (117 features)
- Levels: 0%, 25%, 50%, 75% and 100% of the direction-specific cap. The grid is fixed now.
- At each level: the GENCO balanced feasible share, and the 7-statistic summaries of the 4 state quantities.
- At 0% the state summaries are identical to the G block's zero-state summaries and are not duplicated; only the 0% share is added.
- Count: 5 shares + 4 levels × 28 = 117.

**D. Embeddings at `L_G`** (384 features)
- The mean-over-scenarios pooled embedding at `L_G` (192), and its difference from the zero-flexibility pooled embedding (192).
- The existing zero-flexibility embeddings stay in G.
- No other embedding variant is introduced.

**GA total: 1 + 85 + 117 + 384 = 587 features.**

### Arms (names unchanged)

| Arm | Blocks | Features |
|---|---|---|
| P | P | 224 |
| PC | P + C | 253 |
| PG | P + G + GA | 224 + 220 + 587 = 1,031 |
| PCG | P + C + G + GA | 1,060 |

No other GENCO arm is added.

### Extraction (feature version v3)

- New directory `data_out/envpred/features_v3/`. The amendment-2 CPU features in `data_out/envpred/features/` are kept and not overwritten.
- For device consistency, **all** blocks (P, C, G, GA) of all 840 files (180 test + 660 label) are recomputed in v3. GENCO runs on the Hydra A100s in FP32 with TF32 **off** (deployment-equivalent float32); P and C are deterministic and unchanged in definition.
- Time is recorded per block: phase, classical, GENCO zero-state, GENCO action features, and the complete total.

### Label integrity before fitting (replaces the amendment-2 completeness rule)

Before any fit, `envpred integrity` writes `results/envpred_label_integrity.json`. It checks:
- exactly 330 non-test dates × 2 placements, with no duplicate date/placement block and no missing output;
- 24 hourly intervals × 2 directions per file;
- scenario seed 101 and the correct household seed;
- UK limits and acceptance threshold 0.98;
- methods `ac` and `b_opendss_exact`;
- the code commit, and the SHA-256 of every file.

Failed or timed-out jobs are **retried**. If a date genuinely cannot be completed, it is dropped for **both** label placements and disclosed before fitting.

**Sample sizes (corrects the earlier "about 33,000 training rows").**

| Split | Rows (combined) | Per direction |
|---|---|---|
| train | 21,120 | 10,560 |
| validation | 7,008 | |
| calibration | 3,552 | |
| test | 8,640 | |

That is 31,680 non-test rows in total.

### Placement strata (test)

1. **Familiar placements:** household seeds 101 and 202, which also appear in training.
2. **Unseen placements:** seeds 303, 404 and 505.
3. **All five combined.** The frozen combined decision remains primary.

The unseen stratum is an important secondary hypothesis: *do GENCO's bus-level representations help when household technology placement changes?* A GENCO benefit that occurs only on familiar placements is not presented as generalisation.

Uncertainty within each stratum uses the date-block bootstrap (dates resampled within season; placements, intervals and directions of a date kept together).

### Capacity-versus-cost frontier (secondary)

- Shrink quantiles {80%, 90%, 95%, 99%}, fixed now. The 95% point is primary.
- The exact gate and fallback run on the test set **for every point and every arm**. Every point is reported, including a plot of retained B envelope-hours against exact OpenDSS calls.
- No quantile is chosen with test outcomes.
- **Matched-budget comparison.**
  - For each arm, retention is linearly interpolated along its frontier at two budgets: the exact calls of the best non-GENCO arm at 95%, and those of the best GENCO arm at 95%.
  - Budgets outside an arm's frontier are reported as "not reached"; there is no extrapolation.
  - Pairs compared: P vs PG (does GENCO add value beyond phase information?), PC vs PCG (beyond phase-aware classical physics?), and best non-GENCO vs best GENCO arm.

### Safety and cost reporting

- **Wording.** The gate applies the same design-sample rule and OpenDSS configuration as B.
  - It does not produce unconditional statutory safety.
  - OpenDSS threshold results have known solve-history non-reproducibility (see C+).
  - Held-out feasibility is reported separately.
- **Fresh four-wire feasibility, for the selected G and N arms at 95%, on four sets:**
  - all issued non-zero envelopes of each arm;
  - intervals where both issue non-zero (paired);
  - intervals where only G issues;
  - intervals where only N issues.
  - Primary criterion 4 stays as frozen; the paired sets are reported next to it.
- **Cost, per arm:**
  - phase-feature time, classical-feature time, GENCO zero-state time, GENCO action-feature time;
  - head inference;
  - exact gate calls and time, fallback calls and time;
  - complete end-to-end runtime.
  - Devices differ (GENCO on A100, OpenDSS on CPU), and exact-call savings are not automatically wall-clock savings.

### Interpretation tiers (descriptive; the strict primary decision is unchanged)

Differences are in retention at the primary 95% point, combined over directions, with the date-block bootstrap.

1. **Strong GENCO-positive:** the primary decision is positive.
2. **Useful but non-unique GENCO value:** tier 1 fails, and either
   - PG − P ≥ 3 percentage points with a bootstrap lower bound > 0, or
   - PCG − PC has a bootstrap lower bound > 0 and is < 3 percentage points.

   Wording: "GENCO improves the learned proposal relative to phase-only information, but does not outperform the specialised classical baseline." (Or the PCG-over-PC analogue.)
3. **No measurable GENCO value:** neither tier 1 nor tier 2. Reported directly, and the analysis stops.

### Framing

- **GENCO is described as** "a learned proposal engine for expensive four-wire envelope optimisation, with OpenDSS as the final authority."
- **GENCO is not described as** a replacement for exact power flow, universally faster than classical methods, certified unconditionally, or superior unless the frozen comparison shows it.
- **If the result is positive**, the statement takes the form: "GENCO's action-conditioned representation proposed near-optimal dynamic envelopes that retained X% of exact four-wire capacity using Y% fewer OpenDSS evaluations. Every non-zero proposal passed the same exact design-sample gate used by the reference workflow." The classical comparison goes in the same paragraph.

### Order of work

1. Commit this amendment.
2. Implement v3 extraction, integrity, frontier, strata and tiers, with tests; commit.
3. Extract v3 features on Hydra A100s, with a manifest.
4. Integrity report once the labels are complete.
5. Fit once.
6. Gate the frontier on the test set.
7. Report.
