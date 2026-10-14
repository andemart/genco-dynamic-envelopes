# Work plan: GENCO surrogate for congestion-preventing DOEs on ENWL1 (to 25 Oct 2026)

Three results are kept apart throughout:

- the **balanced GENCO** surrogate;
- the **exact balanced AC** baseline (our backward/forward sweep, checked against datakit/PowerModels);
- the eventual **unbalanced OpenDSS** reality check.

## Gates

Each gate must pass, in physical units, before the next stage spends compute.

| Gate | Pass criterion | Status |
|---|---|---|
| G1 data | sweep vs datakit Vm < 1e-6 pu on every state; branch flows rebuilt from datakit Y; losses ≥ 0; coverage reported per split | **passed** (26,000 states, see experiment log) |
| G2 leakage-free split | weights, normalizer and early stopping use train/val days only; calib and test days in separate datasets | **in place** (`day_split`, graphkit split files) |
| G3a beats trivial baselines | Vm MAE on val days < train-day-mean baseline (0.0093 pu) | stock GENCO **failed** (0.0107) |
| G3b useful surrogate | Vm MAE < 0.002 pu, p99 < 0.01 pu, and p99 < 0.01 pu near the 0.94/1.10 limits; loading MAE < 0.01 | pending |
| G4 DOE accuracy | GENCO+conformal envelopes within 5% of the AC envelope in total kW on test days; AC-verified P(violation) ≤ ε | pending |
| G5 runtime | one interval (both directions, all scenarios) settled < 3 min end to end, warm and cold reported | pending; exact AC already meets it on CPU (see below) |

## Calendar-day splits (`gencodoe.datakit_fluvius.day_split`)

| Day of year d (2024, 366 days) | Split | Use |
|---|---|---|
| d mod 5 ∈ {0, 1, 2} | train | weights, normalizer |
| d mod 5 = 3 | val | early stopping, checkpoint and model selection |
| d mod 5 = 4, ⌊d/5⌋ even | calib (37 days) | conformal margins only |
| d mod 5 = 4, ⌊d/5⌋ odd | test (36 days) | final numbers, touched once |

Every state uses one 15-minute step for all 200 customers, so weather and coincidence are kept.

## Stages and fallbacks

**A. Pipeline validation.** Done.

- In-process inference (`GencoModel`) uses graphkit's own pipeline.
- The evaluator (`gencodoe.evaluate`) writes JSON and CSV.
- Baselines: flat 1.0 pu, train-day mean, LinDistFlow and exact AC.
- Scenario/bus mapping is tested.

**B. Supervised baseline.**

- Train with MaskedBusMSE only, checkpointing on validation loss (val days).
- The stock model failed G3a, so stage C runs before any long run.

**C. Architecture.** Both causes are measured, not assumed.

1. *Scale.* Normalized admittances are 1e3 to 3e5. The stock edge encoder is positively homogeneous, so LayerNorm cancels |Y|: scaling every Y by 10 changes the embedding by 0.2%. Fix: `GNS_heterogeneous_lv`, which applies asinh before the edge and residual encoders; the physics layers are unchanged.
2. *Receptive field.* The feeder depth is 38, and leaf-to-leaf paths are up to about 76 hops; a 12-layer model cannot see the total load. Fixes:
   - `global_context` (a virtual node);
   - more layers.

Screen on 4k/1k states first (about 25 min each on an M4); scale only the winner. Fallback if no variant passes G3a: report the negative result and use LinDistFlow+AC for the DOE. LinDistFlow has Vm MAE 4e-4 and is the strongest cheap baseline.

**D. Physics fine-tune.** Only from a checkpoint that passes G3b.

- Use a small physics weight with warm-up.
- The residual scale is about 1e4 per 1e-3 pu of voltage-difference error on this feeder, so the weight must be set from measured loss/gradient ratios.
- Reject the run if Vm MAE worsens.

**E. Held-out evaluation and conformal margins.**

- Report per-bus and global quantiles on calib days.
- Report coverage, conservatism and false-safe rate on test days, stratified near limits.

**F. DOE integration.** Full test days with real Fluvius states. Methods compared:

- AC bisection;
- LinDistFlow;
- GENCO;
- GENCO+conformal;
- GENCO+conformal+AC verify.

Every envelope is verified with exact AC.

**G. Runtime.** Benchmark:

- CPU AC sweep;
- GENCO on MPS;
- GENCO on the Hydra A100;

with conversion and post-processing included, warm and cold.

## Key design decisions

- **Branch flows from predicted voltages use Kirchhoff's current law** (`feeder.branch_flows_kcl`). The voltage-difference form (v_parent − v)/z turns a 1e-4 pu voltage error into an O(100%) loading error on short LV segments (measured: loading MAE 1.7 for a model with Vm MAE 0.011).
- **Envelope-augmented data (v2).** Real meter states never reach overvoltage or transformer overload (0.0% in all splits). Half of the v2 states add one flexible injection level for all customers: export up to 5 kW, import up to 7.4 kW (assumed single-phase inverter and 32 A charger). These match the rows the DOE search evaluates. Calibration and test sets are augmented the same way.
- **Runtime context.** The exact AC sweep solves about 4,000 states/s on one CPU for this 398-bus feeder. A DOE interval of 500 scenarios × 40 levels is 20,000 power flows, about 5 s. GENCO's speed case on a single feeder must therefore be shown, not assumed; batched multi-feeder or unbalanced cases are where it can pay off.

## Risks

- Voltage errors near limits are what matter. Report conditional errors near 0.94/1.10 pu, and use per-element conformal quantiles if global ones are too loose.
- The training distribution may not cover the DOE search's extreme rows. The v2 augmentation addresses this; check coverage again.
- Balanced approximation: single-phase customers are lumped as balanced loads. The OpenDSS check (unbalanced) is reported separately.
