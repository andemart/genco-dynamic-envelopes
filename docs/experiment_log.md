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

## W3: spatial (feeder × phase) envelopes, complete (36 dates, seed 101; `results/weekend_report.json`)

Post-hoc and exploratory. Analysed only after the envelope-prediction primary result was frozen.

| | Export | Import |
|---|---|---|
| Envelope-hours per household per day: uniform B → spatial | 13.14 → 14.58 (**+11%**) | 6.83 → 14.64 (**2.14×**) |
| Joint λ = 1 (every group at its solo headroom) | 99% of non-zero intervals | 9% |
| Fresh four-wire feasibility of non-zero envelopes, mean / share ≥ 0.95 | 0.998 / 100% | 0.968 / **85.8%** (uniform B on the same dates: ≈ 0.973 / 93.5%) |
| Customer-intervals worse off than under uniform | 41% | 25% |
| Exact calls: B → spatial | 2.24 M → 25.35 M (11×) | 2.12 M → 25.41 M (12×) |

**Reading.**
- Location- and phase-aware envelopes release substantially more import capacity under the same joint design-sample rule.
- The held-out four-wire robustness of the issued import vectors is lower than for uniform envelopes.
- They cost about 11–12× the exact solves.
- Fairness is a real trade-off: up to 41% of customer-intervals receive less than under uniform.
- No GENCO is involved.

## Unbalanced GENCO extension: smoke stage (train/val only; branch ander-unbalanced-genco)

**Labels.** 25,635 four-wire states (Hydra jobs 13343323–26; train/val resubmitted as 13343787/88 after out-of-memory at 8 GB). Semantic integrity passes:
- KCL closes to ≤ 0.16 A;
- the root neutral is 0;
- V_pn and the flags match OpenDSS;
- converged counts match the existing four-wire truth;
- no reset failures.

**Prior check** (1,500 converged val states):

| Method | V_pn MAE | Max error | Time per state |
|---|---|---|---|
| U1 (one sweep iteration) | 3.0e-3 pu | 0.22 pu | 0.08 ms |
| Iterated sweep U3 | 3.0e-6 pu | 6.2e-4 pu | 2.2 ms (CPU) |

**Smoke fits** (A100, 3 epochs, seed 0, 1,500 converged val states):

| Model | Job | Params | V_pn MAE (U1: 3.04e-3) | p99 | Epoch time | GPU memory |
|---|---|---|---|---|---|---|
| PhaseGraph | 13343814 | 216,840 | 9.49e-4 (−69%) | 8.4e-3 | 18 s | 343 MB |
| PhaseGENCO, before the fix | 13343814 | 222,984 | 1.54e-3 (−49%) | | | |
| PhaseGENCO, after the fix | 13343821 | 222,984 | 9.31e-4 (−69%) | 7.9e-3 | 35 s | |

- **Smoke defect found and fixed** (commit 054991b): PhaseGENCO's embedding inputs were not standardised. PhaseGENCO was therefore smoke-run twice, a disclosed deviation from "one smoke fit per model". Train/val only.
- **Learning is real:** the predicted residual varies across states, and its correlation with the true residual is 0.97 (PhaseGraph).

## Unbalanced GENCO extension: frozen test result (`results/fourwire_eval_test.json`, commit c20b96b)

**Runs.**
- Final fits: jobs 13343837 (PhaseGraph, best epoch 59 of 60) and 13343838 (PhaseGENCO, early-stopped, best epoch 19).
- Evaluation: job 13343839. All at commit 49a6045.

**Test set.** 2,968 test states (2,699 converged) on 36 test dates. Calibration on calib only.

**State accuracy** (converged test states):

| Method | V_pn MAE (pu) | p99 | Worst-customer error, max | Neutral current MAE | Phase-loading MAE |
|---|---|---|---|---|---|
| U1 (one sweep iteration) | 3.20e-3 | 3.8e-2 | 0.233 | 0.244 A | 8.4e-4 |
| **PhaseGraph (U2)** | **2.91e-4** | 2.8e-3 | 0.047 | 0.019 A | 5.6e-5 |
| PhaseGENCO | 4.41e-4 | 4.1e-3 | 0.059 | 0.028 A | 8.7e-5 |
| Iterated four-wire sweep (U3) | 3.6e-6 | 6.1e-5 | 0.003 | 0.001 A | 3.3e-6 |

**Decision level** (global conformal screen, amendment 1; mean of both directions):

| Method | Sent to OpenDSS | Exact calls | Observed false-safe | Near-limit balanced accuracy | End-to-end time per state |
|---|---|---|---|---|---|
| U1 | 35.8% | 2,125 | 0 | 0.926 | 10.9 ms |
| **PhaseGraph** | **5.8%** | 345 | 0 | 0.992 | **2.1 ms** |
| PhaseGENCO | 11.8% | 702 | 0 | 0.991 | 4.6 ms |
| U3 | 0.1% | 6 | 0 | 0.993 | 5.5 ms |
| Balanced GENCO / balanced sweep | 63.0% | 3,742 | 0 | 0.77 | 19.5 / 19.9 ms |

**Timing.**
- End-to-end time = method time + fallback share × measured OpenDSS time (29.8 ms per state).
- Learned models: the U1 prior and features on CPU, the model on the A100 (model only: 0.14 ms per state). PhaseGENCO adds the balanced GENCO pass.
- U3 ran as frozen (tolerance 1e-8, at most 50 iterations, vectorised numpy on the node's CPU). It was **not optimised**; a looser tolerance or per-state stopping would be faster.

**Primary criterion, PhaseGraph (U2) vs U1: met on every point.**

| Criterion | Result |
|---|---|
| V_pn MAE reduction | **−91%** (≥ 30% required) |
| Near-limit balanced accuracy | 0.926 → 0.992 |
| OpenDSS fallbacks | **−30.0 points** (35.8% → 5.8%; ≥ 5 required) |
| False-safe | 0 vs 0 |
| Runtime | reported |

**GENCO-specific criterion 9b (PhaseGENCO − PhaseGraph): not met. GENCO embeddings made things worse.**
- Fallback **+6.0 points** for PhaseGENCO, date-bootstrap CI [+5.3, +6.7]. The interval excludes zero, but on the wrong side.
- V_pn MAE 4.4e-4 vs 2.9e-4.
- About 2× the end-to-end cost.
- PhaseGENCO early-stopped at epoch 19, with validation loss 0.073 against PhaseGraph's 0.038.

**Interpretation (claim discipline).**
- A phase-aware, GENCO-architecture model (edge-conditioned TransformerConv processor, trained from scratch on four-wire states) removed the balanced-model information bottleneck. Relative to its four-wire physics prior it cut voltage error by 91% and exact OpenDSS fallbacks from 35.8% to 5.8%. The exact four-wire gate preserved the frozen acceptance rule and produced zero observed false-safe decisions on the held-out evaluation.
- **The improvement is attributed to the unbalanced representation, not to GENCO's pretrained balanced embeddings.** Conditioning on node-aligned GENCO embeddings degraded accuracy and fallbacks.
- **The specialised iterated four-wire sweep is about 80× more accurate (0.1% fallback).** As run in the frozen setup it was about 2.6× slower end to end than PhaseGraph (5.5 vs 2.1 ms per state). That speed gap reflects an unoptimised sweep on CPU against a model on the A100. It is not a claimed advantage over classical solvers.
- Scope: one feeder, single-feeder training. The test dates' global summaries were seen in earlier work, so this is an exploratory extension.

### Correction to the interpretation above (unbalanced protocol amendment 2, after the external audit)

The sentence "The improvement is attributed to the unbalanced representation…" is **too strong**.

**Why.** U1 is itself phase- and neutral-aware, so PhaseGraph vs U1 shows the value of a **learned GENCO-style residual over a one-step four-wire prior**. It does not isolate the causal value of phase-aware representation. The balanced baselines (63% fallback) support the importance of phase information without isolating it.

**Corrected summary:**

> "A GENCO-style phase-aware GNN can correct a cheap four-wire prior (V_pn MAE −91%, OpenDSS fallbacks 35.8% → 5.8%, zero observed false-safe decisions on the held-out evaluation), but pretrained balanced-GENCO representations add no value here, and a specialised classical four-wire sweep remains the accuracy leader."

**Also disclosed in amendment 2:**
- The Thevenin root equivalent was a least-squares fit on training labels.
- The embedding scaler used 2,000 training states.
- The near-limit mask is a union over boundaries.
- The original label checker did not check the cable and transformer flags. It is now extended; every split passes, test included.

## GENCO-OPF stage A (case33bw): frozen test result (`results/genco_opf_stageA_report.json`, run manifest `results/run_manifest_genco_opf.json`)

Protocol: `docs/genco_opf_protocol.md` with amendments 1–2. Test days 100–119, 24 h, both directions, S = 200 design and 1,000 fresh scenarios; six methods through the same exact check and scale-down. Hydra zen4, 1 CPU per task (jobs 13359329/13359330 labels, 13360254 fits, 13360327 test; commits 44ee439, ec18686, eae6857). The report ran on the laptop with commit 6fb268b on the copied day files. Both learned models were fitted once on the training days with the validation days for early stopping; nothing was tuned on test days.

**Hosting after the exact check (MWh per test day, 20-day mean) and gap to the multi-scenario reference.**

| Method | Export | Import | Combined | Gap to reference (combined) |
|---|---|---|---|---|
| uniform | 101.6 | 3.0 | 104.6 | 48.2% |
| LP per scenario + quantile | 179.5 | 11.6 | 191.1 | 5.4% |
| IPOPT per scenario + quantile | 172.4 | 6.5 | 179.0 | 11.4% |
| **multi-scenario OPF (reference)** | 189.4 | 12.7 | 202.1 | 0 |
| **GENCO-style graph model** | 187.7 | 11.9 | 199.6 | 1.2% |
| MLP | 186.3 | 12.0 | 198.3 | 1.9% |

**Criteria (amendment 2), GENCO-style model; day-block bootstrap over the 20 test days, 2,000 replicates, seed 0.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Hosting ≥ 95% of the reference | 98.8% [98.6, 98.9] | met |
| 2. Hosting ≥ 1.05 × LP | 1.044 [1.041, 1.048] | **not met** (short by 0.6 points; the interval excludes 1.05) |
| 3. Held-out P(ok) ≥ reference − 0.005 (mean), share ≥ 0.95 within 1 point | 0.733 vs 0.722; share 0.65 vs 0.49 | met |
| 4. Proposal + check wall-clock below the reference | 0.02 s vs 28.1 s per interval and direction, CPU | met |
| Graph structure: GENCO-style − MLP ≥ 2 points of the reference, bootstrap lower bound > 0 | +0.64 points [0.54, 0.78] | **not met** (positive, but below 2 points) |

**Verdict: "GENCO-OPF positive" is not met** (criterion 2). The supported claim is narrower:

> "A GENCO-style graph model trained from scratch amortises the multi-scenario AC-OPF to within 1.2% of its hosting after the exact check, with held-out feasibility no worse than the reference, at about 1/1000 of its CPU cost. It hosts 4.4% more than the LinDistFlow LP after the same check, short of the preregistered 5% bar, and the graph structure adds 0.6 points over an MLP on the same features, short of the 2-point bar."

**Also observed.**
- **Import is largely a zero problem.** 72.5% of import intervals end at a zero envelope for every method (the passive load violates the undervoltage limit in more than 4 of 200 design scenarios; the sentinel). The comparison is therefore driven by export. Held-out P(ok) in import is about 0.49 for every method including uniform, because a zero envelope's held-out share is the base case's feasibility on fresh draws; it says nothing about the methods.
- **The reference is the least conservative on fresh draws.** Export held-out P(ok): reference 0.956 with 69% of intervals ≥ 0.95, against 0.974–0.977 and 96–99% for the scaled-down methods. The reference uses the full design sample and lands close to the nominal 1 − ε; every other method is scaled down by the exact check and ends more conservative. Criterion 3 passing partly reflects this.
- Per-scenario IPOPT + quantile remains the worst optimiser after the check (11.4% gap), the finding that motivated amendment 2; the LP is at 5.4%.
- Timing is serial CPU for every method (1 task per day). The reference costs 49 s per export and 7 s per import interval; per-scenario IPOPT 8.7 s serial for 200 solves; LP 0.4 s; learned models 0.02 s including the check.
- Training labels: 4,800 intervals, 69% import zero-sentinel, one IPOPT failure (day 67 h 1 export, labelled 0). Both models learned the sentinel (predicted 0.3% of cap on sentinel validation intervals); validation weighted MSE 0.0035 (graph) and 0.0037 (MLP).

## GENCO-OPF amendment 3 (exploratory, post-hoc): the actual GENCO on case33bw (`results/genco_opf_amendment3_report.json`, gate `results/genco_case33_gate.json`)

Protocol amendment 3 (commit 25cd114), written after the stage A verdict and before any case33bw PF state existed. Hydra jobs: datakit 13360775 (20,000 training states, days 0–99, 50% with an envelope injection; 2,000 gate states from val days), graphkit training 13360801 (A100, 74 min, 80 epochs, commit 4051a7c), gate + embeddings + fit 13361551 and the single test evaluation 13361552 (commit d35d2a6). The stage A rows were not re-run; the report merges the one added method into the frozen evaluation.

**The GENCO PF model is excellent on this network.** graphkit `GNS_heterogeneous_lv` + linear prior (hidden 12, 12 layers, 164k parameters of the backbone), trained from scratch on case33bw: on 2,000 fresh val-day states (1,710 near a voltage limit) Vm MAE 5.9e-5 pu, p99 3.5e-4, max 3.8e-3, loading MAE 3.0e-5. Gate G3b passed with a 30× margin (bar 2e-3 / 1e-2). For comparison the ENWL1 model had Vm MAE 3.8e-4.

**Conditioning the envelope model on it does not help; it hurts slightly.** `gnn_genco` = the stage A GENCO-style model with the frozen GENCO's final-layer bus embeddings (96 per bus) of the interval's mean design scenario appended to the 19 node features; same labels, loss, training rule and seed. Validation weighted MSE 0.0037 (best epoch 32) against 0.0035 for `gnn`.

| Test days 100–119, after the exact check | gnn (stage A) | mlp | **gnn_genco** |
|---|---|---|---|
| Hosting, combined, MWh/day | 199.6 | 198.3 | 198.6 |
| Gap to the multi-scenario reference | 1.2% | 1.9% | 1.7% |
| Export / import gap | 0.9% / 6.5% | 1.6% / 6.0% | 1.4% / 6.8% |
| Held-out P(ok) mean, share ≥ 0.95 | 0.733, 0.65 | 0.733, 0.64 | 0.734, 0.64 |
| Proposal + check, s | 0.02 | 0.02 | 0.04 |

**Criterion (amendment 3): not met.** `gnn_genco` − `gnn` = **−0.49 points** of the reference, day-bootstrap interval [−0.58, −0.40], against a bar of +2 points with a lower bound above 0. The interval excludes zero on the wrong side. Criteria 1 (98.3% of the reference [98.1, 98.4]), 3 and 4 still hold for `gnn_genco`; criterion 2 does not (1.039 × LP).

**Reading.** On case33bw the scenario statistics in the 19 node features already carry what the envelope needs; a near-exact learned power-flow state of the mean scenario adds 96 correlated inputs and a little overfitting (validation loss up 5%, best epoch earlier). This is the fourth preregistered role in the project where a GENCO component does not clear its bar, and the first where the GENCO model itself is demonstrably accurate on the network, so the negative result is about the role, not about the model's quality. The honest summary for stage A as a whole: a GENCO-style model, with or without the actual GENCO inside it, amortises the multi-scenario AC-OPF to within 1–2% at a thousandth of the cost, and the value of the graph structure and of the pretrained-style backbone is 0.6 and −0.5 points respectively.

**Also disclosed.** Exploratory and post-hoc: the six stage A methods were evaluated on these test days first; `gnn_genco` once, after development on validation days only (one fit, no tuning). Timing: GENCO embedding on CPU adds about 0.02 s per interval.

## GENCO-OPF amendment 4 (partial metering and state estimation, case33bw): frozen result (`results/genco_opf_amendment4_report.json`; run 1 under `*_run1`)

Protocol amendment 4 (commit c71012c, preregistered) with the disclosed correction 4a (f46db76) after run 1. Hydra: graphkit SE training 13361830 (A100, 47 min, 80 epochs, `configs/genco_se_case33_lv.yaml`), run 1 gate/calibration 13362188 and test 13362189 (commit 43c1e9d), run 2 calibration 13362271 and test 13362272 (commit dc18730). Test days 100–119, 24 h, both directions, f ∈ {0.1, 0.3, 0.5, 0.7, 1.0}, LP proposal + stage A exact check; one realised state per interval; the report ran on the laptop on the copied day files.

**GENCO-SE: no usable estimator (gate failed in both runs).** On 2,000 validation-day states at f = 0.5 the graphkit StateEstimation checkpoint reaches Vm MAE 1.7e-3 pu (bar 2e-3) but an MAE of the unmeasured active loads of **329 kW against 7.6 kW for the prior mean**; the decoded injections are uncorrelated with the truth even at the metered homes whose load was an input. Cause: graphkit's SE task trains on voltages only and derives injections from the predicted voltages through its physics layer, which on a distribution feeder amplifies a 1e-3 pu voltage error into hundreds of kW (the same amplification as in the four-wire study). The masked-reconstruction formulation that would predict loads directly cannot run on the heterogeneous model in graphkit 0.9.0.3 (two-wide voltage mask hard-coded in PF mode) without a custom head; not attempted before the deadline. The arm was dropped per protocol; prior and WLS were evaluated.

**Value of information (run 2, issued envelopes; hosting combined over directions, MWh per test day; realised feasibility of issued envelopes; zero-envelope share).**

| f metered | prior MWh/d | prior / full | prior realised ok | WLS MWh/d | WLS / prior | WLS realised ok | unmeasured-load MAE prior / WLS, kW |
|---|---|---|---|---|---|---|---|
| 0.1 | 193.4 | 0.868 [0.861, 0.875] | 0.982 | 203.0 | 1.050 [1.041, 1.058] | 0.951 | 8.0 / 6.7 |
| 0.3 | 199.5 | 0.896 [0.889, 0.903] | 0.983 | 207.0 | 1.038 [1.031, 1.044] | 0.945 | 7.7 / 6.6 |
| 0.5 | 204.0 | 0.916 [0.910, 0.923] | 0.983 | 209.3 | 1.026 [1.019, 1.033] | 0.945 | 7.7 / 6.6 |
| 0.7 | 210.3 | 0.945 [0.936, 0.952] | 0.979 | 214.0 | 1.017 [1.013, 1.023] | 0.938 | 7.8 / 6.9 |
| 1.0 | 222.7 | 1 | 0.975 | 222.7 | 1 | 0.975 | – |

Zero-envelope share (import mostly): 26–36% of intervals. Realised feasibility over all intervals, zero envelopes included, is about 0.70–0.73 for every arm: those are base-case violations, not envelope violations.

**Reading.**
- **Metering a tenth of the homes already yields 87% of the full-metering hosting, half of them 92%.** The exact check turns the narrower scenario set directly into capacity, with the issued envelopes staying at 0.98 realised feasibility for the prior at every f. The remaining 8–13% is the unmeasured homes' uncertainty, which five voltage readings cannot remove.
- **WLS adds 5 points at f = 0.1, falling to 2 points at f = 0.7, and costs about 3 points of realised feasibility.** Under the preregistered rule (issued envelopes ≥ 0.95) WLS is safe only at f = 0.1 and f = 1.0; at f = 0.3–0.7 it sits at 0.938–0.945, so its calibrated Gaussian spread (one validation-day RMSE per bus) under-covers slightly. Inflating the spread would trade the gain back; not done post hoc.
- **Criterion 2 (GENCO-SE beats WLS): not evaluable;** criterion 3: WLS beats the prior's hosting at every f ≤ 0.7 but not at the safety line.
- Scope: one network, synthetic realised states from the prior's own generator (no model mismatch), fixed five-bus voltage sensing, random metered subsets.

## GENCO-OPF layer: corrections after the external audit (2026-10-22, Codex review of commit b6fe32d)

The audit reproduced every committed OPF report byte for byte from the local raw day files and found the arithmetic sound. It found the following overstatements and omissions, which are corrected here and in `results/run_manifest_genco_opf.json`; earlier entries are left as written.

1. **Test-day use on case33bw.** Days 100–119 have been evaluated **four** times (stage A 13360327; amendment 3 13361552; amendment 4 run 1 13362189; amendment 4a run 2 13362272), and amendment 4b will be the fifth. Earlier wording said three. Amendments 3, 4a and 4b are post-hoc relative to previously observed test results; their bootstrap intervals quantify day-to-day sampling variation, not the adaptive-development bias accumulated across these evaluations.
2. **"Safe".** The supported wording is "passes the preregistered comparative safety criterion" (held-out P(ok) mean within 0.005 of the reference's, share ≥ 0.95 within 1 point; point estimates, as preregistered). Export held-out P(ok) of the GENCO-style model is 0.977; the combined figure of 0.733 includes the import intervals, 72.5% of which are zero-sentinel base cases. In amendment 4 the 0.98 realised feasibility is for issued envelopes; over all intervals it is 0.70–0.73. WLS at f = 0.1 has a point estimate of 0.951 with a day-bootstrap interval of about [0.938, 0.967], so it is not demonstrably at or above 0.95. Day-bootstrap intervals for these safety metrics are now in the reports (`fresh_ok_mean_ci`, `realised_feasibility_ci`).
3. **"Reference optimum".** The reference is the **multi-scenario IPOPT solution** with the greedy scenario-discard heuristic of `opf_multi.multi_envelope`: a local solution of a non-convex problem with no global-optimality certificate and no exhaustive choice of the discarded scenarios. The GENCO-style model hosts more than this reference in 30 of 960 test intervals, by up to 1.57%. "98.8% of the reference" means 98.8% of this heuristic reference, not of a certified AC optimum. The words "optimum" and "AC-optimal" in earlier entries should be read accordingly.
4. **Reproducibility.** The committed reports regenerate only from ignored local inputs. `results/opf_artifact_hashes.json` now records SHA-256 of every raw test-day file, label file, embedding file, checkpoint and metadata file; `scripts/reproduce_check.py` now regenerates the four OPF reports. Checkpoints and raw rows remain uncommitted (size); the hashes identify them.
5. **Provenance error, amendment 3.** The datakit job 13360775 ran at commit **ce56b6d**, not 4051a7c, and ended in Slurm state FAILED (exit 1) after both datasets were generated completely: the failure was the checksum line applied to partitioned parquet directories, fixed in 4051a7c without regenerating the data. The manifest is corrected. The failed setup jobs 13360734 (commit check), 13361841 (gate crash, generator feature width) and 13362269 (clean-tree check after tracked result files were deleted on hydra) are now recorded in the manifest.
6. **Amendment 4b.** No result yet. A smoke test now instantiates `GNS_heterogeneous_lv_recon` and runs `GencoRecon` inference on a synthetic datakit-style template, so checkpoint loading and the measurement mask are exercised before the gate.
7. **Single seed.** Amendment 3 and the graph-versus-MLP comparison use one training seed each; the intervals cover test-day variation, not training randomness. "This fit was 0.5 points worse" is supported; "conditioning generally hurts" is not.

Regeneration note: `results/genco_opf_amendment4_report_run1.json` is now produced by the current report code (realised feasibility on issued envelopes, zero share and intervals added), so its `realised_feasibility` and `unsafe` entries differ from the file first committed with run 1, which used the pre-4a definition; the run-1 rows themselves are unchanged (hashes in `results/opf_artifact_hashes.json`). The stage A, amendment 3 and amendment 4 run-2 reports regenerate with every previous value unchanged and the interval fields added.


## GENCO-OPF amendment 5, option 1: the official pretrained GENCO as warm start for IPOPT on case118_flex (`results/genco_opf_amendment5_warmstart.json`, branch `ander-case118`)

Protocol amendment 5 (4158e66) with 5a (guard) and 5b, built on Evgenii's `evgenii/opf-envelopes` (case118 with 99 flexible export units). Official checkpoint `gridfm/genco-opf-datakit` case118_ieee/base/seed0 (`last.pt`, SHA-256 c2934420b6f2…) on graphkit 0.9.0.3 main. Hydra zen4 (AMD EPYC 9384X), one core, Julia 1.13.2, PowerModels 0.21.6, Ipopt.jl 1.15.0; job 13362503, commit 3cde16e; 500 held-out scenarios (seed 888). Step 0 before it (commit 319a967): the checkpoint loads and produces finite, plausible outputs with accuracy of the same order as the published result on an easier local slice (Pg RMSE 9.6 MW on 300 base-topology scenarios; card 17.1 MW on its perturbed test set), and runs on the flex case (three-scenario smoke).

| Start of IPOPT | converged | iterations median / p90 | wall-clock median / p90 (s) | objective within 1e-4 of flat start | within 1e-6 |
|---|---|---|---|---|---|
| flat | 500 / 500 | 54 / 68 | 0.275 / 0.416 | – | – |
| previous scenario's solution | 500 / 500 | 47 / 57 | 0.236 / 0.345 | 38% | 8% |
| **pretrained GENCO dispatch and voltages** | 500 / 500 | 58 / 71 | 0.288 / 0.418 | 56% | 27% |

GENCO inference on CPU adds 0.082 s per scenario, not counted above.

**Criterion (≥ 20% median wall-clock cut against the previous-solution start, bootstrap lower bound > 0): not met.** The GENCO start is **22% slower** than the previous-solution start, scenario-bootstrap interval [-26%, -20%], and 23% more iterations; it is also slower than the flat start. The 5a quality guard also fails: the GENCO-started solve lands within 1e-4 of the flat-start objective in 56% of scenarios (bar 99%); the previous-solution start manages only 38%, which says the flex OPF has many nearby local optima, so objective agreement at this tolerance is a property of the problem, not of the start. Convergence is 500 of 500 for every start.

**Supported sentence.** "On case118 with 99 flexible export units, starting IPOPT from the official pretrained GENCO's dispatch and voltages gives no material warm-start gain: the median solve is 22% slower than starting from the previous scenario's solution and slower than a flat start, on one zen4 core, with convergence unchanged." The zero-shot checkpoint's flex dispatch is 37% below the optimum on this case (step 0), which is consistent: a start far from the optimum in the flexible units costs IPOPT iterations rather than saving them. The from-scratch arm (iv) was not run: no checkpoint is committed on the branch.

## GENCO-OPF amendment 5, option 2: official pretrained GENCO fine-tuned versus trained from scratch at scarce labels (`results/genco_opf_amendment5_transfer.json`, branch `ander-case118`)

Protocol amendment 5 with 5a (freeze) and 5b (the exact check), and 5c (implementation incident: the first evaluation, job 13363424, loaded the N = 10,000 checkpoints for every N and is void; its output is kept as `*_buggy.json`). Official checkpoint `gridfm/genco-opf-datakit` case118_ieee/base/seed0. 30,000 case118_flex AC-OPF labels (job 13362542), nested splits; six A100 runs, one seed each, 80 epochs; evaluation on the first 1,000 test-split scenarios on one zen4 node (job 13375605, commit 567caa6). Hashes of labels, splits, test IDs, checkpoints and result are in `results/run_manifest_genco_opf.json` under `amendment5.option2_transfer`.

| Labels N | Fine-tuned: hosting after the check (% of reference) | From scratch | Fine-tuned − scratch [scenario bootstrap 95%] |
|---|---|---|---|
| 100 | 96.2 | 82.1 | +14.2 [13.6, 14.8] |
| **1,000** | **89.4** | 82.5 | **+6.9 [5.9, 7.8]** |
| 10,000 | 98.5 | 90.1 | +8.4 [7.7, 9.2] |

**Criterion (at N = 1,000: fine-tuned − scratch ≥ 2 points with lower bound > 0, and fine-tuned ≥ 95% of the reference): not met.** The gain over scratch is met; the 95% bar is not (89.4%).

Qualifications:
- **The check is an IPOPT convergence-and-redispatch gate.** The flexible units are fixed at λ × proposal and the AC-OPF re-solved; λ is the largest value on {1, 0.75, 0.5, 0.25} that converges. The reference (IPOPT label dispatch) passed through the same gate keeps only 96.4% (λ mean 0.9635), so the gate also scales back valid optimal dispatches, and the 95% bar sits 1.4 points below the reference's own gated share. Hosting shares are relative to the ungated reference.
- **The grid is coarse.** Every model's λ was 1 or 0.75 in every scenario; no level below 0.75 was used. One non-convergence at full scale costs 25% in that scenario, which likely amplifies differences between models. Per-scenario reference λ is saved from this commit on (`per_scenario.reference_lambda`).
- **Raw proposals are almost never feasible on their own** (0.1–4.9% of scenarios across the six models). The result is a hybrid: the network proposes the flexible dispatch and IPOPT redispatches the rest. It is not a standalone neural OPF.
- **One training seed per arm.** The intervals cover test scenarios conditional on the six fits, not training variability. The fine-tuned arm is non-monotone in N (96.2, 89.4, 98.5): at N = 100 every proposal passed at λ = 1 (flexible-unit MAE 8.5 MW), at N = 1,000 414 of 1,000 were cut to 0.75 (MAE 3.3 MW). Why is not established; distance from the pretrained weights was not measured. Seed replicates are running (exploratory addendum).
- **Transfer, not generalisation.** The checkpoint was trained on the same case118 base system with a closely related load distribution; this tests transfer to the flexible-hosting formulation, not to an unseen network, and overlap with the checkpoint's training profiles is not ruled out.

**Supported sentence.** "On case118 with 99 flexible export units, fine-tuning the official pretrained GENCO exceeded training from scratch by 14.2, 6.9 and 8.4 points of reference hosting at 100, 1,000 and 10,000 labels (one seed each, IPOPT redispatch gate), but at the preregistered 1,000-label point it kept 89.4% of the reference, below the 95% bar."

## GENCO-OPF amendment 6: data-informed confirmation at N = 100 on unopened scenarios (`results/genco_opf_amendment6.json`, branch `ander-case118`)

Protocol amendment 6 (14b7cf5), frozen before any evaluation of sorted test positions 1000–2999 and before any seed-replicate result. Official checkpoint fine-tuned versus scratch at N = 100 labels, training seeds 0–4 (seed 0 = the amendment 5 pair; seeds 1–4 trained at 1eb6ea5), the amendment 5 gate unchanged, 2,000 scenarios never evaluated before. Evaluation jobs 13381345–13381349 at 322664a from a second checkout (the queued training jobs pinned the main checkout; the evaluation code is the frozen one). Run, checkpoint, test-ID and result hashes: `results/run_manifest_genco_opf.json` under `amendment6`.

| Seed | Fine-tuned (% of reference) | Scratch | Fine-tuned − scratch (points) |
|---|---|---|---|
| 0 | 95.6 | 83.7 | +11.9 |
| 1 | 92.1 | 83.7 | +8.4 |
| 2 | 95.2 | 82.2 | +13.0 |
| 3 | 96.0 | 84.6 | +11.5 |
| 4 | 94.7 | 82.6 | +12.1 |
| **Mean** | **94.7** | 83.4 | **+11.4** [9.8, 12.5] |

**Criterion: not met.** (1) mean fine-tuned ≥ 95%: **94.7%, not met**; (2) mean gain ≥ 2 points: met; (3) paired seed-and-scenario bootstrap lower bound > 0: met (9.8); (4) fine-tuned better in ≥ 4 of 5 seeds: met (5 of 5).

- The pretraining gain replicates on unseen scenarios and across training seeds: about 11 points, in every seed. The absolute bar is missed by 0.3 points on the mean; seed 1 (92.1%) is the low one, and three of five seeds are at or above 95% individually. The verdict is a fail.
- Context, not a rescue: the IPOPT reference through the same gate keeps 96.7% on these scenarios (267 of 2,000 scaled down; now recorded per scenario). The criterion was frozen against the ungated reference and is reported as such. The amendment 5 qualifications (redispatch gate, coarse grid, hybrid rather than standalone neural OPF, same base network as the checkpoint) apply unchanged.

**Supported sentence.** "In a frozen follow-up at 100 labels on 2,000 unseen scenarios with five training seeds, fine-tuning the official GENCO beat training from scratch by 11.4 points [9.8, 12.5], in 5 of 5 seeds, but kept 94.7% of the reference, missing the 95% bar."

## GENCO-OPF amendment 5, exploratory addendum: training-seed replicates of option 2 (`results/genco_opf_amendment5_transfer_s{1,2}.json`)

Frozen before any replicate result (1eb6ea5): all six option 2 arms refitted with seeds 1 and 2, same configs, splits, epochs, gate and the same first 1,000 test scenarios (training jobs 13381147, 13381148; evaluations 13381149, 13381150 at 1eb6ea5). **Post-hoc and exploratory; the preregistered verdict stays the seed-0 result (fail).**

Hosting after the check, % of the reference: fine-tuned / scratch / difference (points).

| Labels N | Seed 0 | Seed 1 | Seed 2 | Fine-tuned range | Scratch range |
|---|---|---|---|---|---|
| 100 | 96.2 / 82.1 / +14.2 | 92.5 / 82.3 / +10.2 | 95.0 / 79.2 / +15.9 | 92.5–96.2 | 79.2–82.3 |
| 1,000 | 89.4 / 82.5 / +6.9 | 84.0 / 91.1 / -7.1 | 95.3 / 78.0 / +17.3 | 84.0–95.3 | 78.0–91.1 |
| 10,000 | 98.5 / 90.1 / +8.4 | 98.9 / 88.2 / +10.7 | 99.1 / 88.1 / +11.0 | 98.5–99.1 | 88.1–90.1 |

- **At 100 and 10,000 labels, fine-tuning beats scratch in all three seeds** (+10.2 to +15.9 and +8.4 to +11.0 points).
- **At 1,000 labels, the training seed dominates**: the difference ranges from −7.1 (seed 1, scratch better) to +17.3 (seed 2). Fine-tuned hosting spans 84.0–95.3% and scratch 78.0–91.1%. The seed-0 verdict at the preregistered point is not a stable property of the method.
- **The N = 100 > N = 1,000 ordering of the fine-tuned arm** holds in seeds 0 and 1 but not in seed 2 (95.0 vs 95.3). It is training noise, not a property of the label budget.
- Seed 2 alone would have met the option 2 criterion at N = 1,000 (95.3%, +17.3). It is reported, not used.

**Supported sentence.** "Across three training seeds, fine-tuning the official GENCO beat scratch at 100 and 10,000 labels in every seed, but at 1,000 labels the outcome depended on the seed (−7.1 to +17.3 points); single-seed results at that budget are not reliable."

