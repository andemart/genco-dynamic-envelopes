# Experiment log: GENCO power-flow surrogate on ENWL1

## How to read this log

All metrics are in physical units from `python -m gencodoe.evaluate`:

- **Vm** in pu, **Va** in degrees;
- **loading** as a fraction of the rating, from voltages via Kirchhoff's current law (KCL);
- **false-safe** = a state that truly violates a limit but is predicted to be fine.

"val" means val-day states, never used for weights or the normalizer. Calibration and test days are separate datasets.

**Balanced vs unbalanced.** Every number here is for the balanced positive-sequence ENWL1 model. Truth is datakit's AC solution (PowerModels), which matches our sweep to within 6e-8 pu. The unbalanced OpenDSS check is not done yet.

## E0: physics + MSE, stock model (14 Oct, laptop)

- **Config.** `0.1 LayeredWeightedPhysics + 0.9 MaskedBusMSE`; random split over 20k training-day states; 44 epochs.
- **Result.** Collapse toward zero voltage. On val days: Vm MAE 0.986 pu, Va MAE 8.7°.
- **Why.** Measured, not assumed:
  - Normalizer baseMVA is 2.1 kW, so normalized admittances are 1e3–3e5.
  - At the true AC solution the graphkit physics residual is 0.02, so the data and physics agree.
  - A non-uniform voltage error of 1e-3 pu gives residuals far above the residual of the all-zero-voltage state (33). With these scales the physics loss rewards the trivial solution.
- **Artifacts.** `mlruns/702338596317565606/2b68b49cf7c44ba3ac92b580c2e1156b`.

## Finding 1: the stock edge encoder cannot see impedance magnitude

- graphkit's edge encoder is Linear → LeakyReLU → Linear → LayerNorm. With inputs of order 1e4 the biases are negligible, so it is positively homogeneous and LayerNorm cancels |Y|.
- Multiplying every admittance by 10 changes the edge embedding by 0.2% (cosine similarity 1.000000), at random init and in the trained E0 model alike.
- So the stock model cannot tell a long cable from a short one.
- Covered by `tests/test_genco.py::test_lv_edge_encoder_sees_impedance_magnitude_stock_does_not`.

## Finding 2: flows from predicted voltages must use KCL

- For E0-like predictions, (v_parent − v)/z gives loading MAE 1.7: a 1e-4 pu voltage error over a short LV segment is an O(100%) flow error.
- `feeder.branch_flows_kcl` rebuilds each branch current from the downstream injections, so the error is O(dV).
- The evaluator, `GencoOracle` and conformal calibration all use KCL.

## Finding 3: real meter states do not reach export congestion

- v1 data (20k train/val, 3k calib, 3k test):
  - undervoltage 4–5% of states;
  - line overload 3–4%;
  - **overvoltage 0%**;
  - **transformer overload 0%**.
- The DOE search evaluates "every customer at level L" rows, which sit outside this distribution.
- v2 data adds such rows to half of the states. Export is up to 5 kW and import up to 7.4 kW per connection; these caps are assumptions.
- v2 coverage, consistent across splits:

  | Condition | Share of states |
  |---|---|
  | undervoltage | 23% |
  | overvoltage | 7.5% |
  | line overload | 36% |
  | transformer overload | 15% |
  | reverse flow | 30% |

- 1–1.5% of v2 states fail datakit's OPF step at extreme import.
- datakit then renumbers `scenario`. The draw ↔ state join now uses `load_scenario_idx`; there is a regression test for it.

## Baselines (1,000 val-day states, v1)

| Predictor | Vm MAE | Vm p99 | Vm max | Va MAE (°) | Loading MAE | False-safe / violating | Runtime |
|---|---|---|---|---|---|---|---|
| flat 1.0 pu | 0.0144 | 0.0585 | 0.120 | 0.092 | 2.3e-4 | 32/63 | — |
| train-day mean | 0.0093 | 0.0494 | 0.099 | 0.065 | 1.5e-4 | 31/63 | — |
| LinDistFlow | 3.8e-4 | 0.0033 | 0.0122 | — | 2.3e-4 | 10/63 | 0.03 s |
| fixed point, 1 iteration | 3.8e-4 | 0.0033 | 0.0122 | 0.0009 | 8.8e-6 | 8/63 | 0.02 s |
| exact AC sweep | 1e-9 | 2e-8 | 5e-8 | 1e-8 | 1e-11 | 0/63 | 0.3–0.4 s |

## Stage C screen

- **Setup.** 4,000 train-day and 1,000 val-day states (v1); MaskedBusMSE only; 15 epochs; Apple M4/MPS.

| Run | Variant | Vm MAE | Vm p99 | Va MAE (°) | Verdict |
|---|---|---|---|---|---|
| q_stock12 | stock, 12 layers | 0.0107 | 0.051 | 0.26 | fails G3a (worse than train mean) |
| q_lv12 | asinh edge/residual encoders, 12 layers | 0.0119 | 0.057 | 0.45 | fails G3a |
| q_lv12_prior | + linear prior (zero-init head, scale 1e-3) | 3.80e-4 | 0.0030 | 0.00095 | passes G3a/G3b, mostly thanks to the prior |
| q_lv12_prior_ctx | + global context | plateau equal to q_lv12_prior (val MSE 2.425e-7 vs 2.42e-7) | | | no gain |

- **Hybrid vs prior alone** (same 1,000 val states):

  | Metric | Prior alone | Hybrid |
  |---|---|---|
  | Vm RMSE | 7.95e-4 | 6.95e-4 |
  | Vm p95 | 1.59e-3 | 1.23e-3 |
  | Vm p99 | 3.33e-3 | 2.97e-3 |
  | false-safe | 8/63 | 7/63 |

  Val MSE plateaus from epoch 4. A global channel does not change the plateau, so the 12-hop receptive field is not the limit. Next: depth 24 and width 48 on v2.

- **First attempt at q_lv12_prior.** The head was not zero-initialised and the prior was added unscaled. Val MSE started at 1.2e-3, about 2,000× worse than the prior alone, because the random head swamped the prior.
  - Fix: zero-initialise the last layer and add 1e-3 × head output. An untrained model now reproduces the prior exactly.
  - The aborted run is kept in `mlruns/436740004777710085`.

## Stage C on v2 (4k/1k envelope-augmented states, 15 epochs, M4)

**Bug found and fixed.** graphkit's `SaveBestModelStateDict` keeps the monitored metric *tensor* as its best score. In two runs the exported `best_model_state_dict.pt` stayed at epoch 0 (the head's zero init), although validation loss fell every epoch; they evaluated identical to the prior.

- `gencodoe.graphkit_cli` now stores the best score as a float after each save.
- The two runs were re-evaluated from `checkpoints/last.ckpt`, which is also their best epoch because validation loss decreased monotonically.
- All other exports were checked against `last.ckpt` times and are current.

| Run | Vm MAE | RMSE | p95 | p99 | max | False-safe |
|---|---|---|---|---|---|---|
| fixed point x1 (prior alone) | 5.43e-3 | 0.0169 | 0.0248 | 0.0872 | 0.212 | 7 |
| 12 layers, hidden 12 | 4.16e-3 | 0.0153 | 0.0188 | 0.0812 | 0.206 | 3 |
| 24 layers, hidden 12 | 4.74e-3 | 0.0156 | 0.0204 | 0.0824 | 0.208 | 0 |
| 12 layers, hidden 48 | 3.94e-3 | 0.0133 | 0.0133 | 0.0696 | 0.195 | 2 |

Width helps; depth does not (24 layers is worse than 12 at equal epochs).

## Stage B/C full runs on Hydra A100 (v2, 14,648 train-day states, 80 epochs, ~55 s/epoch)

Evaluated on the same 1,000 v2 val-day states:

| Job | Variant | Vm MAE | RMSE | p95 | p99 | max | False-safe |
|---|---|---|---|---|---|---|---|
| 13326572 | LV + prior, hidden 12 | **1.01e-3** | 3.3e-3 | **2.7e-3** | **6.6e-3** | 0.102 | **0** |
| 13326573 | LV + prior + global context | 9.6e-4 | 2.7e-3 | 3.7e-3 | 7.9e-3 | 0.082 | 3 |

- Artifacts: `mlruns_hydra/877677247744645366/d42d5c34…` (selected) and `mlruns_hydra/499876217681087884/31c5a2ec…`.
- The selected model is 5.4× more accurate than its own prior on envelope rows and passes G3b. It is still less accurate than 5 fixed-point iterations (MAE 1.2e-4).
- Job 13328751 (hidden 48) was queued after this.

## Fixed-point convergence (exact AC is cheap here)

| States | Iterations | Vm MAE | Vm max | ms/state (CPU) |
|---|---|---|---|---|
| v1 val (real) | 1 | 3.8e-4 | 1.2e-2 | 0.04 |
| v1 val | 2 | 1.5e-5 | 1.3e-3 | 0.06 |
| v1 val | 3 | 7.4e-7 | 1.5e-4 | 0.10 |
| v1 val | 5 | 4e-9 | 1.7e-6 | 0.15 |
| v2 val (envelope rows) | 1 | 4.9e-3 | 0.21 | 0.03 |
| v2 val | 5 | 1.2e-4 | 0.033 | 0.12 |
| v2 val | converged (exact) | 1e-9 | 1e-7 | ~1.8 |

GENCO hybrid inference on CPU is 7–22 ms/state. On real states the plain fixed point beats GENCO on speed and accuracy after two iterations. GENCO has room only on the extreme envelope rows, and only if it beats ~2 ms/state for exact AC.

## Stage E: conformal margins (v2 calib → v2 test days, fixed-point prior)

| α | Margins | Test coverage | Near-limit coverage | Cleared safe | AC fallback | False-safe | Conservatism |
|---|---|---|---|---|---|---|---|
| 0.05 | global | 0.951 | **0.742** | 0.55 | 0.087 | 2/1116 | 0.12 |
| 0.01 | global | 0.990 | 0.950 | 0.00 | 0.64 | 0/1116 | 1.00 |
| 0.01 | per element | 0.992 | 0.960 | 0.00 | 0.72 | 0/1116 | 1.00 |
| 0.01 | **binned by predicted Vm** | 0.991 | **0.986** | **0.60** | **0.037** | **0/1116** | **0.035** |

- Marginal (global) conformal under-covers near the limits.
- Mondrian binning on the predicted voltage (`conformal.calibrate_binned`) restores near-limit coverage. With AC verification of the uncertain 3.7% it gives zero false-safe test states.
- The DOE screen uses the binned margins (`ScreenedFeasibility` takes a margin callable).

With the Hydra GENCO hybrid (job 13326572), same calib → test protocol:

| α | Margins | Coverage | Near-limit coverage | Cleared safe | AC fallback | False-safe | Conservatism |
|---|---|---|---|---|---|---|---|
| 0.05 | global | 0.951 | 0.784 | 0.615 | 0.011 | 0/1116 | 0.015 |
| 0.01 | global | 0.991 | 0.951 | 0.603 | 0.027 | 0/1116 | 0.033 |
| 0.01 | **binned** | 0.990 | **0.991** | **0.609** | **0.019** | **0/1116** | **0.024** |

Compared with the prior alone at α = 0.01 binned, GENCO halves the AC fallback (3.7% → 1.9%) and cuts conservatism from 3.5% to 2.4%.

## Stage F: full test day 2024-07-08 (96 intervals, 50 scenarios, ε = 0.05)

- Uniform policy, caps of 5 kW export / 7.4 kW import per connection.
- Assumed mix: 30% heat pump, 30% EV, 40% PV.
- Exact balanced AC on 50 fresh draws per interval.

| Direction | Method | Mean total | Gap vs AC | Intervals with P(ok) < 0.9 | AC calls | Runtime/day |
|---|---|---|---|---|---|---|
| export | AC | 347.5 kW | — | 10 | 105,600 | 30 s |
| export | fixed point + binned conformal + AC verify | 347.5 kW | 0 | 10 | 5,026 | 13 s |
| export | LinDistFlow | 332.1 kW | −15.4 kW | 0 | 0 | 5 s |
| import | AC | 247.0 kW | — | 13 | 105,600 | 53 s |
| import | fixed point + binned conformal + AC verify | 247.0 kW | 0 | 13 | 38,676 | 21 s |
| import | LinDistFlow | 263.4 kW | +16.4 kW | **73** | 0 | 6 s |

- The worst interval (all methods, both directions) took 2.5 s against the 3-minute budget.
- Even exact AC averages out-of-sample P(ok) of 0.94–0.945 against the 0.95 target. This is the finite-scenario (n = 50) generalisation gap of the chance constraint, separate from solver error. Next: more scenarios or a scenario-approach margin.

## Stage G: runtime on Apple M4, quiet machine (rows of v2 calibration states)

| Method | ms/row (1k–4k rows) |
|---|---|
| exact AC sweep, numpy, CPU | 0.30–0.42 |
| fixed point x1 | 0.02 |
| GENCO end to end, CPU | 5.6–7.7 |
| GENCO end to end, MPS | 6.3–7.1 |

- The model forward pass dominates: 5.5 s per 1,000 graphs on CPU. Graph preparation takes about 0.5 s and the prior 0.02 s.
- MPS is no faster. A100 benchmark: job 13328787.

## Runtime context

- The exact AC sweep solves about 2,500–4,000 states/s on one M4 CPU core for this feeder.
- A full test day of hourly export+import envelopes (20 scenarios, k = 4 levels per round) took 30 s in total with AC. The single-feeder 3-minute budget is met without a surrogate.
- LinDistFlow import envelopes on the same day were **unsafe**: P(ok) under exact AC on fresh draws went down to 0.0 in the worst hour.

## Stage G (final): runtime on Hydra (job 13329021 A100-PCIE-40GB; job 13329587 EPYC 9384X)

Per state, end to end. GENCO is the frozen hidden-12 LV+prior model (job 13326572):

| Method | Hardware | ms/state |
|---|---|---|
| GENCO, TF32 (model only) | A100 | 0.32 (0.28) |
| GENCO, FP32 (model only) | A100 | 0.44 (0.42) |
| generic Z-bus fixed point (exact, works on meshed too) | EPYC 9384X, 8 cores | 0.49-0.56 |
| radial sweep (exact) | EPYC 7282 / 9384X | 0.52-0.80 |
| generic Z-bus / radial sweep | Apple M4 | 0.20 / 0.30 |
| fixed point, 1 iteration | any | 0.04-0.05 |
| OpenDSS four-wire (unbalanced), one core | EPYC 7282 | 32 |
| pandapower Newton-Raphson, one core | EPYC 7282 | 103 |

- Model batch 4,096 vs 1,024 makes no difference.
- TF32 gives 1.35x. Its effect on accuracy (float32 jitter) is not yet checked.

Hourly test-day DOE (both directions), from exact call counts x measured costs (`gencodoe.time_accounting`):

| Exact AC engine | AC only | GENCO A100 screen + AC | Fixed-point screen + AC |
|---|---|---|---|
| pandapower | 5,430 s | 724 s (7.5x) | 1,150 s (4.7x) |
| radial sweep | 27.6 s | 20.5 s (1.35x) | 8.1 s (3.4x) |

**Conclusion.** On a 398-bus LV feeder, GENCO on an A100 is two orders of magnitude faster than general-purpose AC solvers and comparable to exact solvers specialised for small feeders (Z-bus, sweep). It does not beat them.

- As a conformal screen, GENCO cuts exact AC calls by 87% vs AC only and by 40% vs the linear screen.
- With pandapower as the exact fallback, 98% of the screen's time is fallback calls. At ~1,000 scenarios per interval the worst interval would take ~17 min, so a "meets the 3-minute window where general solvers fail" claim is **not** supported.
- GENCO's speed case needs larger networks, where dense or sparse classical solvers scale poorly, or a cheaper exact fallback.
- Hidden-48 run (job 13328751): early-stopped at epoch 67, then crashed with a bus error in graphkit's test step. Its checkpoint gives Vm MAE 2.5e-3 vs 1.0e-3 for hidden 12, so it was dropped and the model is frozen at hidden 12.

## Unbalanced reality check (validated)

The four-wire OpenDSS checker reproduces the balanced model when every customer is reconnected as a three-phase load:
- passive load: identical;
- +1.7 kW per customer: loading 1.20 vs 1.18.

The checker is therefore correct. With the real single-phase connections (74/68/58 customers per phase, current returning through the neutral), the same injections give:
- export at +1.7 kW: Vmax 1.096 vs 1.071, loading 1.37 vs 1.18;
- import at +1.7 kW: Vmin 0.904 vs 0.945, loading 1.10 vs 0.83.

With passive load alone, 2.1% of snapshots are undervoltage on the four-wire model. Balanced envelopes are genuinely unsafe on the real feeder (~3% pass rate); this is physics, not a bug. Next: a calibrated unbalance derating and OpenDSS-exact envelopes.

## Finite-sample certification (test day 2024-07-08, 96 intervals)

200 scenarios, eps = 0.05, beta = 0.05 (accept a level only if >= 98% of scenarios are feasible), 1,000 fresh AC draws per interval:
- mean P(ok): 0.974 export, 0.976 import;
- intervals below 0.95: 3 export and 4 import, none significantly below (one-sided p >= 0.087);
- capacity cost vs the plain empirical rule: 1.3% export, 3.7% import.

## Correction: four-wire checker loads were not constant power (15 Oct)

OpenDSS loads default to `Vminpu=0.95` / `Vmaxpu=1.05`: outside that band a load becomes constant impedance. That understated undervoltage and distorted export rows above 1.05 pu.

- `gencodoe.unbalanced` now sets `Vminpu=0.5 Vmaxpu=1.5`, i.e. constant power as in the balanced model and datakit, and retries a non-converged solve from the no-load solution.
- **Invalid, superseded by the numbers below:** the earlier "+1.7 kW export: Vmax 1.096, loading 1.37" figures and the "~3% unbalanced P(ok)" of the hourly DOE run.

Revalidated at 13:00 on 2024-07-08:

| Case | Balanced | Four-wire, three-phase loads | Four-wire, real single-phase loads |
|---|---|---|---|
| +1.7 kW export | Vmax 1.071, L 1.18 | Vmax 1.078, L 1.18 | Vmax 1.083, L 1.22 |
| +1.7 kW import | Vmin 0.945, L 0.83 | Vmin 0.938, L 0.85 | Vmin 0.904, L 1.10 |

The export gap is small; the import gap is large.

### Four-wire truth on v2 calib/test (2,953 / 2,968 states; `python -m gencodoe.unbalanced`)

- 9-10% of states do not converge on four wires. All are extreme imports that already violate on the balanced model (voltage collapse); they count as violating.

| State type | dVmin (four-wire − balanced) | dVmax | Loading ratio (four-wire / balanced) |
|---|---|---|---|
| import rows | mean −0.07, 5th pct −0.17 | — | median 1.23, 95th pct 1.56 |
| export rows | — | mean +0.031, 95th pct +0.061 | median 1.13, 95th pct 2.2 |
| real states | mean −0.036 | mean +0.012 | median 1.43, 95th pct 2.35 |

- 22% of states are balanced-safe but four-wire-violating. Calib and test agree.

### Option C: GENCO screen with four-wire conformal margins (`gencodoe.four_wire`)

Setup:
- Mondrian split conformal on state-level worst cases: min V and max V offsets, loading ratio.
- Fitted on calib days against OpenDSS; collapsed states give +inf for undervoltage and loading.
- Evaluated on test days against OpenDSS.

| α | Direction | Safe | Violated | OpenDSS fallback | False-safe | Conservative |
|---|---|---|---|---|---|---|
| 0.05 | import | 0.127 | 0.434 | 0.439 | 11/1775 | 0.69 |
| 0.05 | export | 0.447 | 0.379 | 0.174 | 12/1311 | 0.21 |
| 0.01 | import | 0.000 | 0.403 | 0.597 | 0/1775 | 1.00 |
| 0.01 | export | 0.295 | 0.374 | 0.331 | 3/1311 | 0.47 |

Two bugs were found and fixed on the way:
- Collapsed states poisoned the overvoltage bins.
- `~` on a Python bool gives −2, not False.

Reading:
- The screen is safe-ish at α = 0.05 (false-safe 0.6-0.9% of violating states) and strict at α = 0.01.
- Import screening is weak: balanced-model error grows with import, so most import rows need OpenDSS.
- Next: DOE days with option C vs option B (exact OpenDSS envelopes) to measure capacity and OpenDSS calls.

## Corrections and method freeze (15 Oct, after Codex audit)

**False-safe claim corrected.** In the previous screen at α = 0.01, export had **3 of 1,311** violating states screened safe. Import had 0 only because no import state was screened safe. An earlier chat summary said the strict screen "never wrongly clears a state"; that was false.

**Method changes before the freeze:**
1. Transformer loading is now an explicit screen criterion (four-wire / predicted ratio). Before, it was only redundant by accident: no test state was a transformer-only violation.
2. The α error budget is split over the four criteria (α/4 each, union bound). Per state, P(screened safe but violating) ≤ α under exchangeability. This is pointwise, not over a day.

Revised screen on the exploratory v2 test set, α = 0.01:

| Direction | Safe | Violated | OpenDSS | False-safe | Conservative |
|---|---|---|---|---|---|
| import | 0.000 | 0.199 | 0.801 | 0/1775 | 1.00 |
| export | 0.122 | 0.199 | 0.679 | 0/1311 | 0.78 |

GENCO's four-wire contribution is modest: it clears 12% of export states, and about 20% of rows are recognised as certainly violating.

**Monotonicity check.** 96 checks: 4 seasons × 4 times of day × 3 states × 2 directions, levels 0–5 kW. One case is non-monotone: a winter-night passive state already overloads a cable at level 0, and a small export relieves it. The search then issues a zero export envelope, which is conservative, not unsafe. No feasibility "hole" between feasible levels was found. The issued envelope is verified independently anyway.

**Runs.**
- The hourly runs on 2024-07-08 (24 intervals) are development runs, an "hourly subsample of a day", not full days.
- Hydra job 13330990 (old option C) was cancelled. Job 13330989 (option B, exact four-wire) continues as the development reference.
- The final evaluation is pre-registered in `docs/preregistration.md`.

## Pre-registered winter day 2024-12-15 and the voltage-limit base error (15 Oct)

### Pre-registered result (as run, under the stated limits)

Options B and C give zero optional incremental envelopes in all 96 intervals, in both directions.

- **Weak agreement check.** At level 0 the acceptance rule already failed, so the search only evaluated the endpoints. B = C here is weak evidence of agreement; summer must test a nonzero boundary.
- **Search work, B vs C:**
  - OpenDSS calls: 76,800 (B) vs 38,845 (C), 49% fewer;
  - envelope-search wall-clock: 3,001 s vs 1,648 s, 1.82x faster. This times the search only; independent verification is not included.
- **Interpretation:** no safe *optional incremental envelope containing zero* exists. This does not mean the feeder "cannot host export": moderate mandatory export could relieve existing import overloads, but the frozen method does not consider it.
- **Confidence intervals.** 95% Clopper-Pearson upper bound of four-wire P(ok) below 0.95: B export 86/96 intervals, B import 96/96. Even zero envelopes are not certifiable.

### Baseline diagnosis

Command: `python -m gencodoe.diagnose_baseline`, 50 fresh no-envelope draws per interval, seed 202. Output: `results/superseded_wrong_voltage_base/winter_2024-12-15_baseline_diagnosis.json`. **Exploratory:** it used household seed 0, not the pre-registered 101, so it describes a different household/phase allocation (see the amendment erratum).

| | Four-wire | Balanced |
|---|---|---|
| Draws with any violation | 63% | 0.5% |
| Worst-customer voltage, median (5-95%) | 0.933 (0.869-0.959) | 0.968 |

Composition of the four-wire violations:
- Almost all are undervoltage. Line overload occurs in 5.9% of draws, always together with undervoltage.
- No transformer overload and no non-convergence. The feeder is lightly loaded: median 148 kW, transformer about 20% loaded.
- Phase 1 (74 customers) is the worst phase. Transformer phase-current imbalance (max/mean) is 1.16 median and 1.42 at the 95th percentile.
- Neutral current: median 62 A, 95th percentile 151 A.

### Error: voltage limits on the wrong base

- 1 pu in this study is the no-load transformer secondary: 416 V line-to-line, 240.2 V phase-neutral.
- The limits 0.94 / 1.10 pu (from the handoff) were applied on that base, giving 225.8 V / 264.2 V.
- Statutory limits for ENWL (UK) are 230 V −6% / +10% = 216.2 V / 253.0 V, i.e. **0.900 / 1.053 pu** on our base.
- Under EN 50160 (Belgium), 230 V ±10% gives 207 / 253 V, i.e. 0.862 / 1.053 pu.

Consequences:
- The lower limit was about 10 V too strict and the upper limit about 11 V too lax.
- This applies to every balanced and four-wire result so far. Export envelopes were overstated; winter undervoltage was overstated.

Winter baseline, share of draws violating under each lower limit:

| Lower limit | Undervoltage | Any violation |
|---|---|---|
| 225.8 V (as implemented) | 63.3% | 63.3% |
| 216.2 V (UK) | 14.9% | 15.8% |
| 207 V (EN 50160) | 3.9% | 7.2% |

~~Methods are unaffected~~ (corrected after review). The architecture, trained weights and conformal error residuals are unaffected. **Everything that depends on the feasibility thresholds is affected and is recomputed:**
- screen shares, fallback and OpenDSS-call counts;
- false-safe validation;
- runtimes;
- envelopes and capacity comparisons;
- near-limit error metrics.

The pre-registered runs stand as a record under their stated limits.

## Pre-registration amendment 1 (voltage limits): applied before any corrected run

- The primary criterion is now the UK statutory range 216.2–253.0 V, i.e. 0.9002–1.0534 pu on the 240.18 V base, set in `gencodoe.limits`.
- The EN 50160 voltage band (207–253 V) is a secondary sensitivity.
- Old-limit outputs were moved to `superseded_wrong_voltage_base/`. The summer jobs still running under the old limits (13331014, 13331015) were cancelled.
- The corrected B and C runs repeat 2024-12-15 and 2024-06-28 with unchanged seeds; see `docs/preregistration_amendment_1.md`.

## Erratum after Codex review (16:10)

- **Household seed.** The pre-registered DOE runs use household seed 101, equal to `--seed`; the amendment text claimed 0. The winter diagnosis used seed 0, so it is exploratory and is rerun with seed 101.
- **Reporting plan frozen.** `docs/reporting_plan.md` and `gencodoe.report_final` were committed before the corrected outputs were opened.
- **Runtime comparisons.** B and C run as separate jobs, possibly on different nodes, so runtime ratios are descriptive measurements.
- **Old evaluation outputs archived.** Threshold-dependent outputs in `data_out/eval` computed under the old limits are moved to `data_out/eval/superseded_wrong_voltage_base/`: coverage, false-safe, near-limit, screening and data-check shares. Timing benchmarks are kept, because they do not depend on the limits.
- **Not to circulate yet:** `README.md`, `docs/workplan.md` and the one-pager (`docs/genco_challenge_onepager.*`) still contain old numbers and limits.

## Winter 2024-12-15 baseline diagnosis, pre-registered population (household seed 101, UK limits)

`results/winter_2024-12-15_baseline_diagnosis_hh101_uk.json`: 50 no-envelope draws × 96 intervals, draw seed 202, household seed 101.

| | Value |
|---|---|
| Any four-wire violation | 35.1% (balanced: 1.1%) |
| Undervoltage | 29.7% |
| Line overload | 26.6% |
| Overvoltage | 0.8% |
| Transformer overload | 0% |
| Non-converged | 0.04% |
| Overlaps | undervoltage + line overload 983; undervoltage only 405; line overload only 256; all three 38; none 3,116 (of 4,800) |
| Worst line on phase 1 / 2 / 3 | 3,337 / 1,230 / 231 draws |
| Most frequent worst lines | line1917, line1919, line1948, line1920 (one lateral) |
| Phase-current imbalance (median / 95th pct) | 1.26 / 1.56 |

**Reading.** The congestion is local: phase 1 of one lateral, where this household allocation concentrates heat-pump and EV homes. It is not a feeder-wide shortage.

**Placement sensitivity.** The same day under household seed 0 (exploratory) gives 15.8% under UK limits. Any penetration study must therefore vary household placement over several seeds.

This is a baseline analysis only. The corrected DOE outputs were not opened when it was produced.

## Results of the amended pre-registered evaluation (UK limits; reports from `gencodoe.report_final` / `report_annual`)

### Winter 2024-12-15, 15-minute (`results/report_winter_2024-12-15.json`)

- B = C = 0 in all 96 intervals, both directions. Both-zero agreement is weak evidence.
- Static limit 0.
- Four-wire mean feasibility: 0.73 export, 0.65 import.
- Balanced envelopes: export 2.18 kW/household with four-wire feasibility 0.067; import 0.26 kW with 0.36.

| | B | C |
|---|---|---|
| Exact calls (export / import) | 38,400 / 38,400 | 19,658 / 19,200 |
| Search time (export / import) | 680 / 2,262 s | 639 / 629 s |

### Summer 2024-06-28, 15-minute (`results/report_summer_2024-06-28.json`)

- **B = C exactly**, including 59 export and 95 import intervals with non-zero envelopes. C never exceeds B.
- Static limit 0.

| | Capacity (kW·h per household per day) | Mean limit |
|---|---|---|
| B = C, export | 14.2 | 0.59 kW |
| B = C, import | 18.95 | 0.79 kW |
| Balanced, export | 33.3 | — |
| Balanced, import | 32.4 | — |

Four-wire safety, pre-registered metric:
- export: 31 of 96 intervals certified (one-sided lower bound ≥ 0.95), 59 with p̂ ≥ 0.95, mean 0.77;
- import: 51 certified, 95 with p̂ ≥ 0.95, mean 0.976.

Descriptive split (not pre-registered): where the envelope is > 0, mean four-wire feasibility is 0.976 in both directions, with p̂ ≥ 0.95 in 56/59 export and 94/95 import intervals. The zero-envelope export intervals (37) have baseline violations (mean 0.43). Certification needs about 98% of 200 draws to pass, so the power of the verification sample limits it.

| | B | C |
|---|---|---|
| Exact calls, export | 274,400 | 247,012 (−10%) |
| Exact calls, import | 418,400 | 343,663 (−18%) |
| Search time, export | 4,828 s | 6,619 s (+37%) |
| Search time, import | 11,440 s | 9,518 s (−17%) |

### Annual extension, 36 test days, hourly (`results/annual_report.json`)

- Annual static limit: 0 in both directions.

| | Annual capacity (kWh per household per year, 95% CI) | Feeder |
|---|---|---|
| B, export | 4,810 (4,346-5,304) | 962 MWh |
| B, import | 2,502 (2,113-2,875) | 500 MWh |
| Balanced (unsafe, 0% certified), export | 15,524 | — |
| Balanced, import | 7,451 | — |

Seasonal means (kWh per household per day; winter / spring / summer / autumn):
- export: 6.1 / 12.6 / 16.8 / 17.0;
- import: 0.6 / 6.5 / 15.1 / 5.1.

B-C agreement over 864 intervals per direction:
- export: 100% (472 both non-zero);
- import: 99.88% (444 both non-zero), with **one interval where C > B by 0.334 kW**.

| | B | C |
|---|---|---|
| Exact calls, export | 2.24 M | 2.03 M (−9%) |
| Exact calls, import | 2.12 M | 1.69 M (−20%) |
| Search time, export | 39,773 s | 60,136 s (+51%) |
| Search time, import | 64,283 s | 52,973 s (−18%) |
| Search time, total | 104,056 s | 113,109 s (+9%) |

Runtime ratios are descriptive (different CPU nodes).

### The one C > B interval: a false-safe acceptance

Interval: 2024-04-29 13:00, import. B = 0.0308 kW, C = 0.3647 kW.

Replaying the run's random draws reproduces the 200 design scenarios exactly. Exact OpenDSS feasibility of those scenarios is:
- 0.990 at 0;
- 0.980 at B's level (accepted at the threshold, so B is reproduced);
- 0.975 at 0.05-0.25 kW;
- 0.970 at C's level.

Feasibility is monotone here. C accepted a level whose exact design-set share (0.970) is below the 0.98 rule, so the screen cleared at least 2 of 200 scenarios as safe that OpenDSS rejects. This is within the per-state α budget, but **a per-state screening guarantee does not carry through to the envelope-level acceptance rule**.

Impact:
- C's issued envelope holds in 97% of fresh four-wire draws (one-sided lower bound 0.942);
- it adds about 3 kWh per household per year (+0.14%);
- it is 1 of 1,728 paired intervals.

Fix, a post-hoc method change if adopted: re-verify only the final accepted level with OpenDSS (200 exact solves per interval and direction). C would then be certified exactly like B.

## C+ formal run (protocol 2253f1f + amendment 84f1564; implementation 4269211; Hydra job 13333842)

38 stored C runs: 36 annual hourly days plus the two 15-minute days. All completed, with no input-hash changes. Outputs are in `data_out/doe/cplus_annual/` and `cplus_uk/`. Reports:
- `results/annual_report_with_cplus.json`
- `results/report_2024-06-28_with_cplus.json`
- `results/report_2024-12-15_with_cplus.json`

The frozen B/C entries in these reports are byte-identical to the original reports.

### Gate outcomes (all runs)

| | Count |
|---|---|
| Non-zero C envelopes gated | 1,070 |
| Gate failures, repaired | **72** |
| Pre-specified diagnostic: "screen error" | 58 |
| Pre-specified diagnostic: "solver-history noise" | 14 |
| Zero-infeasible sentinels created by repair | 0 |

Extra exact calls:

| Category | Calls | Seconds |
|---|---|---|
| Gate | 214,000 | 3,682 |
| Repair | 220,800 | 3,783 |
| Diagnostic | 14,400 | 274 |
| Verification | 14,400 | 258 |

**Factual split, from the recorded |L_C − L_B|:**

- **71 of the 72 failures occurred at L_C = L_B exactly.** B's own exact search had accepted the level at 196/200; the gate's exact re-evaluation of the same replayed scenarios found 195/200.
  - These are **threshold non-reproducibility of the exact four-wire oracle**: OpenDSS default tolerance plus solve-history dependence. They affect B equally and are not GENCO errors.
  - About 6.6% (71/1,070) of non-zero exact envelopes are not reproducible at the 0.98 threshold on re-evaluation.
  - The pre-specified diagnostic labels a failure "screen error" whenever the canonical solve also fails. For these 71 cases that only means the canonical solve agrees with the gate rather than with B's search. **The label does not imply a GENCO error.**
- **One failure is a genuine screen-driven over-acceptance:** 2024-04-29 13:00, import, C 0.3647 → C+ 0.0321 kW (B 0.0308).
- **Repair sizes** in the other 71 cases are 0.001–0.007 kW. Repair costs about 3,000 exact solves each, because the protocol's repair re-searches [0, L_C] from zero. It is inefficient for small corrections; this is a cost property of the frozen protocol.

### Annual (36 days, hourly)

| | Export | Import |
|---|---|---|
| Envelope-hours per household per year: B / C / C+ | 4,810 / 4,810 / 4,809 | 2,502 / 2,505 / 2,501 |
| C+ vs B within tolerance | 100% | 100% |
| C+ > B | 0 intervals | 0 intervals |
| Exact calls: B / C / C+ | 2.238 M / 2.029 M / 2.261 M | 2.122 M / 1.693 M / 1.835 M |
| Search time: B / C+ | 39,773 / 64,149 s | 64,283 / 55,420 s |
| Held-out p̂ ≥ 0.95: B / C+ | 0.610 / 0.610 | 0.587 / 0.588 |
| One-sided LB ≥ 0.95: B / C+ | 0.304 / 0.304 | 0.259 / 0.259 |

- **Combined exact-call saving vs B: C 14.65%, C+ 6.04%.** For export, C+ uses 1.1% *more* exact calls than B.
- C+ total search time is 119,569 s vs 104,056 s for B (+15%). This is descriptive, as the runs were on different CPU nodes.

### 15-minute days

| Day | Direction | Exact calls B / C / C+ | Search time B / C / C+ |
|---|---|---|---|
| Summer 2024-06-28 | export | 274,400 / 247,012 / 275,812 | 4,828 / 6,619 / 7,103 s |
| Summer | import | 418,400 / 343,663 / 374,663 | 11,440 / 9,518 / 10,040 s |
| Winter 2024-12-15 | export | 38,400 / 19,658 / 19,658 | 680 / 639 / 639 s |
| Winter | import | 38,400 / 19,200 / 19,200 | 2,262 / 629 / 629 s |

- Summer: C+ = B within tolerance, and capacity is unchanged.
- Winter: no non-zero envelopes, so no gates and C+ = C.

## Option 2: learned four-wire heads (post-hoc, exploratory; docs/option2_protocol.md)

Implementation committed (`gencodoe.option2`) before the result was computed. Regenerate with
`python -m gencodoe.option2 --run <GENCO run> --config configs/genco_pf_enwl1_lv_prior_v2.yaml --out results/option2_screen.json`.
Train 14,648 / val 5,066 / calib 2,953 / test 2,968 states; UK limits; α = 0.01 over 4 criteria. The test days were also used in earlier development.

| Arm (test states) | Import: safe / to OpenDSS / false-safe | Export: safe / to OpenDSS / false-safe |
|---|---|---|
| Frozen adapter, GENCO summary (reference) | 2.2% / 78.0% / 0 | 0.0% / 80.1% / 0 |
| Head, phase-only features | 25.8% / 50.1% / 2 | 21.0% / 55.0% / 0 |
| Head, + exact balanced summary (classical) | 26.4% / 49.7% / 1 | 20.6% / 55.9% / 0 |
| Head, + GENCO summary + pooled embeddings | 23.9% / 52.0% / 1 | 16.7% / 59.4% / 0 |

- The reference row reproduces `results/screen_comparison_genco_vs_classical.json` exactly.
- **A learned four-wire head helps a lot:** states cleared without OpenDSS go from ~2% to ~21–26%; states sent to OpenDSS drop from ~79% to ~50–56%.
- **The gain comes from phase-composition information, not from GENCO.** Phase-only features alone reach it. Adding the exact balanced summary changes little. The GENCO arm is *worse* than both (−2.5 pp import, −3.9 pp export vs classical).
- **The pre-specified evidence criterion for GENCO (≥ 5 pp more safe than classical) is not met.** Reported as is.
- False-safe states: 1–2 out of 2,968 per arm. That is within what α = 0.01 per state allows (pointwise guarantee). It is not zero, unlike the frozen adapter, which cleared almost nothing.
- Head test MAE (converged states): v_pn_min ≈ 0.011 pu, v_pn_max ≈ 0.005 pu, line loading ≈ 0.04, transformer ≈ 0.002–0.003, similar across arms.
- Per the protocol, DOE-level integration would follow only if the screen supported GENCO. It does not support GENCO over classical. A phase-feature head could be integrated as a non-GENCO screen, but that is outside the GENCO claim.

## Envelope prediction v3: A100 determinism and device diagnostic (features only, before fitting)

Job 13338852. Environment: torch 2.11.0+cu126, CUDA 12.6, torch_geometric 2.8.1, torch_scatter 2.1.2, driver 610.57.04, NVIDIA A100-PCIE-40GB; TF32 off.

**Five A100 repetitions** of the 12 fixed smoke items, production path:

| What | Max change |
|---|---|
| `L_G` | 0 (identical in all 60 runs) |
| Feasible shares, zero indicator, fixed-grid shares | 0 |
| Per-scenario feasibility decisions, constraint flags | 0 |
| Voltage summaries | ≤ 1.0 × 10⁻⁴ pu |
| Loading summaries | ≤ 6.2 × 10⁻⁵ |
| Headroom and change summaries | ≤ 1.0 × 10⁻⁴ |
| Zero-state embeddings | ≤ 4.0 × 10⁻³ (2.1% of the smoke-sample std) |
| `L_G` embeddings and deltas | ≤ 1.74 × 10⁻² (`GA:embLG_max12`, 2.8% of its std; mean change ≈ 9 × 10⁻⁴) |

- The 0.017 seen in the smoke test is this max-pooled embedding channel. It is continuous and not decision-relevant.
- Reproducibility is therefore **tolerance-based, not bitwise**.
- `torch.use_deterministic_algorithms(True)` ran without error. It was not adopted, because no decision-relevant feature changes and adopting it would require re-extraction.

**Uneven chunking** (batch 333 instead of the production single chunk):
- `L_G` is identical in all 12 items, and every share is identical.
- On 2024-04-01, one per-scenario feasibility decision during the search flipped.
- 2 constraint flags changed (2024-04-01 and 2024-07-21), with no effect on any feature.
- Production extraction used a fixed batching (every call ≤ 800 rows in one 1,024 template), so this does not affect v3.

**CPU float32 vs A100 FP32.** The devices are **not interchangeable**:
- `L_G` differs in 3 of 12 items, by 0.0016–0.0024 kW (2024-01-01 import, 2024-04-01 export, 2024-05-11 export). This is below the search tolerance (5 W export, 7.4 W import), but the values are not identical.
- Per-scenario feasibility decisions differ by 0–5 per item, and constraint flags by 0–5.
- One feasible share differs by 1/200.
- Zero-state voltage differs by up to 4.5 × 10⁻³ pu (the known float32 input-path jitter).
- All v3 features, for training and evaluation, use the frozen A100 FP32 path.

**Decision: proceed without re-extraction** (amendment 3b).

## Envelope prediction: label defect found after the fit (2026-10-17). Fit invalidated, nothing gated

**What happened.**
- The single v3 fit (job 13338884) completed, but every head stopped after 10 boosting iterations.
- Checking why, using training and validation rows only, showed that **all 31,680 B labels in the 660 amendment-2 label files are 0**, with design share exactly 0.0 in 31,626 rows.
- No test label, test prediction or gate result was examined.

**Cause.** An OpenDSS engine whose very first solve diverges never recovers.
- In the label runs (`--unbalanced 0`), the first solve of each fresh process was B's top level: every customer at the 5 kW export cap.
- That solve diverged, and every later solve in the process failed, including the no-load reset.
- Reproduced locally and on a Hydra node: on a fresh engine the search returns share 0.0 at zero; after one no-load priming solve, 0.9 for the same 40 scenarios.
- All earlier runs used `--unbalanced 200`. There, the `ac` method's four-wire verification solved ordinary states before B's first search, priming the engine.
- An audit of every stored run (annual B/C, 15-minute days, C+, W1, W2, W3) finds no failure signature.

**Fix.**
- `UnbalancedENWL1` now solves the no-load case at construction, and raises if that solve fails.
- A regression test fails on the old code and passes on the new.
- The label-integrity check now flags the failure signature (all exact design shares 0). It flags all 660 label files.

**Status.**
- Label jobs 13337097 and 13337098 are invalid.
- Fit 13338884 and its predictions are invalid: they were trained on constant zero labels.
- The v3 features (job 13337441) are unaffected. They never use labels or OpenDSS, and `L_ac` comes from the valid `ac` rows.
- Next steps await the user's decision.

## Envelope prediction v3: frozen primary result (q95; `results/envpred3_primary_q95.json`, commit ddef0ae)

**Status.** Frozen prospective analysis within an already explored benchmark.
- One invalid fit (13338884) is quarantined, and this result comes from the replacement valid fit (13342839).
- All shrink points were computed in one cached evaluation, but the frozen q95 primary result was reported and committed before the secondary frontier was examined.
- GENCO is evaluated as a learned proposal engine for expensive four-wire envelope optimisation; exact OpenDSS remains the final decision authority.

**Arms selected on validation dates only:** best non-GENCO arm P; best GENCO-containing arm PCG.

**Test results: 36 dates × 5 placements = 8,640 interval-directions per arm.**

| Arm | Retained B envelope-hours | Exact calls alone (vs B 24.53 M) | Gate pass rate | Fallbacks | Fresh four-wire feasibility, mean / share ≥ 0.95 |
|---|---|---|---|---|---|
| P | 79.0% | 1.651 M (6.7%) | 93.4% | 341 | 0.991 / 0.999 |
| PC | 78.6% | 1.561 M (6.4%) | 94.3% | 291 | 0.991 / 0.999 |
| PG | 78.8% | 1.688 M (6.9%) | 93.4% | 334 | 0.991 / 0.999 |
| PCG | 79.0% | 1.662 M (6.8%) | 93.7% | 320 | 0.991 / 0.999 |

**Other figures.**
- Excess issued above B: 1.3–1.4% of B's envelope-hours, in 135–141 intervals per arm.
- Predicted zero where B > 0: 385–407 intervals per arm. This is the main source of lost capacity.
- Physical exact calls across all arms and shrink points, after caching: 57.99 M. These are not credited to any arm.

**Primary decision (PCG vs P): GENCO not positive.**

| Criterion | Result |
|---|---|
| Retention difference | −0.06 percentage points; date-block 95% interval [−0.46, +0.32] |
| ≥ 3 points | no |
| No more exact calls | no (1.662 M vs 1.651 M) |
| Bootstrap lower bound > 0 | no |
| Feasibility not worse | yes |

**Descriptive pairs.**
- PG − P = −0.21 points, interval [−0.61, +0.17]: no value beyond phase-only information.
- PCG − PC = +0.39 points, interval [+0.10, +0.68]: **detectable but not practically material** (below 3 points). This is not a positive finding.

**Placement strata.**
- Familiar placements (101, 202): PCG − P = −0.51 points, interval [−0.81, −0.19]; PG − P = −0.53.
- Unseen placements (303, 404, 505): PCG − P = +0.22 points, interval [−0.39, +0.79]; PG − P = 0.00; PCG − PC = +0.73, interval [+0.29, +1.14].
- Retention is 80.0–80.5% on familiar and 77.6–78.3% on unseen placements, for every arm.
- There is no GENCO-specific generalisation benefit.

**Held-out feasibility, paired.**
- Both P and PCG issue on 4,884 intervals, with fresh four-wire means of 0.9914 (P) and 0.9914 (PCG).
- Only PCG issues on 108 intervals (0.985); only P issues on 122 intervals (0.985).
- The gate applies the same design-sample rule and OpenDSS configuration as B. This does not give unconditional statutory safety.

**Cost.**

| Item | Value |
|---|---|
| GENCO action features, test set | 25,350 s on the A100 |
| `L_ac`, test set | 11,998 s on CPU |
| Exact search, P alone | ≈ 29,400 s on CPU (1.65 M solves at 17.8 ms) |
| Exact search, B | 566,789 s on CPU (24.5 M solves) |

GENCO adds proposal cost without improving the proposal.

**Interpretation (frozen rule): no measurable GENCO value.** The PCG − PC difference is detectable but below the 3-point materiality threshold.

**What the experiment does show, independently of GENCO.** A learned proposal followed by one exact gate retains about 79% of the exact envelope-hours with about 6.7% of B's exact OpenDSS calls, roughly 15× fewer. Held-out four-wire feasibility of the issued envelopes is 0.991. This holds for the phase-only arm, so it needs no GENCO.

### Secondary frontier (exploratory; `results/envpred3_frontier_report.json`, generated after the primary was committed)

Retained B envelope-hours against the exact calls each arm needs alone (B: 24.53 M), at the shrink quantiles 99 / 95 / 90 / 80%:

| Arm | Retention by shrink quantile, 99 → 80% | Exact calls (M), 99 → 80% |
|---|---|---|
| P | 52.3 / 79.0 / 85.6 / 90.4% | 0.95 / 1.65 / 2.93 / 5.19 |
| PC | 53.3 / 78.6 / 85.5 / 90.3% | 0.98 / 1.56 / 2.82 / 5.01 |
| PG | 49.0 / 78.8 / 85.9 / 90.5% | 0.97 / 1.69 / 3.27 / 5.59 |
| PCG | 50.0 / 79.0 / 85.9 / 90.4% | 0.98 / 1.66 / 3.17 / 5.56 |

**Matched budgets** (linear interpolation; descriptive, not observed operating points):
- At P's q95 budget (1.65 M calls): P 79.0, PC 79.1, PG 77.3, PCG 78.5%.
- At PCG's q95 budget (1.66 M calls): P 79.1, PC 79.2, PG 77.8, PCG 79.0%.
- GENCO-containing arms do not retain more at equal exact-call budgets.

**Workflow trade-off.** A learned proposal plus one exact gate retains about 90% of B's envelope-hours with about 21% of its exact calls, or 79% with 6.7%.

**Note.** The frontier script's built-in tier label ("no measurable GENCO value") predates the amendment-3a wording. The committed primary artifact is authoritative: "detectable but not practically material" (PCG − PC = +0.39 points). Both mean that GENCO is not positive.
