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
