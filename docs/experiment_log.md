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
