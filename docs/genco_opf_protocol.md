# GENCO-OPF envelopes: protocol (new exploratory layer)

**Status: exploratory.** Branch `ander-genco-opf`.
- Committed before any OPF label is generated and before any model is fitted.
- Existing results (B/C/C+, annual, W1–W3, Option 2, envpred3, the four-wire extension) are untouched.
- Proposed by E. (GENCO team): compute **optimal** envelopes with OPF instead of uniform or proportional envelopes found by power flows.

## 1. Question

> Does a GENCO-OPF proposal, followed by an exact check and scale-down, give AC-optimal-quality, safe envelopes at a fraction of the cost of solving AC-OPF with IPOPT?

**Why this gap matters.** Our only optimal method so far, the max-total LinDistFlow LP, fails the AC check (P(ok) 0.856 against a 0.95 target, in the case33bw demo). Per-scenario AC-OPF with IPOPT is correct, but costs about 20 ms per solve.

## 2. Two stages

| Stage | Network | Exact check | Role |
|---|---|---|---|
| **A** | case33bw (balanced, 12.66 kV, 32 load buses) | batched exact balanced AC sweep | GENCO's native setting: the clean test of the claim |
| **B** | ENWL1 four-wire benchmark with Fluvius households | iterated four-wire sweep (validated to 3.6e-6 pu against OpenDSS), with OpenDSS confirmation on the final envelopes | the application |

**A runs first. B starts only after A's result is committed.** B's ENWL1-specific choices are frozen in an amendment before B runs.

## 3. Envelope method (identical for every optimiser)

For each interval and direction, with design scenarios s = 1..S (S = 200, ε = β = 0.05, acceptance threshold 0.98):

1. **Per-scenario OPF.**
   - Each flexible customer i is a dispatchable unit at its bus:
     - export: 0 ≤ P_i ≤ cap_i;
     - import: −cap_i ≤ P_i ≤ 0.
   - The passive load is that scenario's net load.
   - Constraints: AC power flow, voltage limits, line ratings, transformer rating.
   - **Objectives:**
     - **hosting** (primary): maximise Σ|P_i|;
     - **fairness** (secondary): maximise Σ|P_i| − w·Σ P_i² with w fixed in A's amendment, so allocations spread.
   - Each scenario gives an optimal allocation x_i^s.
2. **Quantile proposal.** x_i = the ε-quantile of {x_i^s} across scenarios (the lower ε-quantile of |x_i|, signed by direction).
3. **Exact check and scale-down.**
   - Check the vector x jointly on all S design scenarios with the exact model.
   - If fewer than 98% are feasible, find the largest λ ∈ [0, 1] such that λ·x passes, by k-section search (4 levels per round, tolerance 1e-3).
   - This exact step **enforces the design-sample rule for every method**, whatever proposed x.
4. **Held-out verification.** Fresh scenarios (1,000), reported separately.

## 4. Methods compared

| Method | Proposal x | Notes |
|---|---|---|
| **Uniform** | existing exact uniform search | the baseline that hosting gains are measured against |
| **LP** | max-total LinDistFlow LP per scenario | the existing optimal-but-approximate method, now with the same exact scale-down |
| **IPOPT-OPF** | AC-OPF per scenario (PowerModels + IPOPT) | the reference optimum |
| **GENCO-OPF** | GENCO in OPF mode per scenario | the method under test |

## 5. GENCO-OPF model

- **Task:** graphkit `OptimalPowerFlow`, with customers as generators (Pg bounds, linear or quadratic costs). Outputs are Pg per unit and bus Vm, Va, Qg.
- **Encoder:** our LV variant with asinh admittance features. The linear PF prior is not used: it needs known injections, which OPF does not have.
- **Labels:** datakit OPF mode with customer generators.
- **Disclosed workarounds:**
  - datakit's slack pinning (already handled for PF);
  - negative-cost units. Preferred formulation: a positive cost on slack import with zero-cost customers for export; a positive cost on customer Pg ∈ [−cap, 0] for import. Losses are included in the objective; this is disclosed.
- **Data:** OPF labels on training intervals only, with validation intervals for early stopping. Evaluation intervals are disjoint (section 6).
- **Runs:** one smoke fit and one final fit per stage. The configuration is frozen in an amendment before the final fit.

## 6. Stage A setting (frozen now)

- **Network:** pandapower `case33bw`, exported to MATPOWER, slack voltage 1.0 pu.
  - Voltage limits 0.95–1.05 pu.
  - Line ratings from the case: placeholder ratings of 1,000 A or more become inf. If no finite ratings exist, line rating = 1.5 × each line's peak base-case loading. Fixed now, applied by code, reported.
- **Flexible units:** one per load bus (32), cap_i = 1.5 × the bus's nominal active load, in both directions.
- **Scenarios:**
  - Load = nominal × a fixed 24-hour residential profile × lognormal day factor (σ 0.15) × per-bus Gaussian copula noise (relative std 10%, correlation 0.6).
  - PV on every second load bus: capacity = 1.0 × nominal load, a clear-sky 24-hour shape × day cloud factor U(0.3, 1.0), relative std 20%.
- **Days:** 120 synthetic days (seed 0): 80 train, 20 val, 20 test. Hourly intervals.
  - Training labels: 2,000 scenario states per training day (160,000 OPF solves), drawn from the training days' scenario distributions.
  - Evaluation: the 20 test days × 24 h × 2 directions; S = 200 design and 1,000 fresh scenarios per interval.
- **Seeds:** a scenario stream separate from training (seed 101).

## 7. Metrics (test intervals; per method and direction, and combined)

- **Hosting:** total envelope (Σ_i limit × 1 h) per day, and its gain over uniform (%).
- **Gap to IPOPT-OPF:** (hosting_IPOPT − hosting_method) / hosting_IPOPT, after the exact scale-down.
- **Safety:** design share at the issued envelope (≥ 0.98 by construction); held-out P(ok) mean, and share of intervals ≥ 0.95.
- **Fairness:** Jain's index of the per-customer limits; share of customers with limit 0.
- **Cost:**
  - proposal wall-clock (GENCO: A100 batch inference; IPOPT: **parallel**, N CPU cores, N reported);
  - exact check and scale-down time and calls;
  - end-to-end time.
  - Hardware is always stated.

## 8. Success criteria (Stage A; all required for "GENCO-OPF positive")

1. **Near-optimal:** GENCO-OPF hosting ≥ 95% of IPOPT-OPF hosting, combined over directions.
2. **Better than the cheap optimal method:** GENCO-OPF hosting ≥ LP hosting × 1.05, both after exact scale-down.
3. **Safe:** held-out P(ok) no worse than IPOPT-OPF's minus 0.005 (mean) and minus 1 point (share ≥ 0.95).
4. **Cheaper:** GENCO-OPF proposal wall-clock below parallel IPOPT's at the stated hardware.
   - Criterion 4 is reported separately. If 1–3 pass and 4 fails, the claim is "near-optimal and safe, not faster", **not** positive on cost.

**Uncertainty.** Day-block bootstrap over the 20 test days (2,000 replicates, seed 0) for the hosting ratios.

## 9. Claim discipline

- **The exact check guarantees the design-sample rule for every method.** Differences between methods are in hosting and cost, never in nominal safety.
- No claim of unconditional safety; no claim beyond the stated networks.
- Negative results are reported. In particular, the gain may come from accuracy over the LP rather than speed over parallel IPOPT.

## Amendment 1 (2026-10-19, before any label, fit or method comparison)

Found while building stage A. Checked on **training days only** (days 0, 13, 27, 41, 55 and 69; 5 scenarios per hour; 720 solves per direction). No evaluation day was used.

1. **The slack is pinned at 1.0 pu** (V_min = V_max = 1.0 at the root). Unpinned, the OPF raised the substation voltage to 1.05 pu to host more import. The protocol already specified 1.0 pu; the code now enforces it.

2. **Export caps raised to 3.0 × nominal load. Import caps stay at 1.5 ×.**

   | Setting | Network-limited solves (< 99% of caps) | Mean hosted / caps |
   |---|---|---|
   | Export at 1.5× | 2.4% | 99.9% |
   | **Export at 3.0×** | **100%** | **78.7%** |
   | Export at 5.0× | 100% | 49.5% |
   | Import at 1.5× | 100% (network always binds) | 55.1% |

   At 1.5× every method would return about the caps, so export would be uninformative. Import is already informative at 1.5×.

3. **Infeasible per-scenario OPF.** 124 of 720 import solves were infeasible: the passive load already violates the voltage limits.
   - For such a scenario, x_i^s = 0 for every unit (no feasible non-negative allocation; the zero-sentinel analogue).
   - The ε-quantile then includes these zeros.
   - GENCO-OPF does not detect infeasibility. Its prediction is used as is, and the exact check and scale-down decide.
   - Counts are reported per method.

4. **Fairness weight (the secondary objective):** a quadratic cost w·P_i² on every unit, with w = 1 / (2 · mean cap of the direction) and a slack cost c1 = 1. The primary objective stays maximum hosting.

5. **IPOPT timing.**
   - The reported per-solve time is the `solve_ac_opf` wall-clock.
   - Python↔Julia data-update overhead is reported separately.
   - "Parallel IPOPT" means N independent Julia processes on one Hydra zen4 node, N reported; throughput = states / wall-clock.

## Amendment 2 (2026-10-19, before any label, fit or test-day evaluation): option 1, the multi-scenario OPF

### Finding that motivated it (validation day 80 only; development data)

Per-scenario AC-OPF + per-customer ε-quantile **does not keep optimality**:
- each per-scenario optimum sits on its own scenario's voltage limit, and uses different customers' headroom;
- the stitched quantile vector then fails jointly and is scaled down hard.

| Import, 03:00 | Proposal after quantile | After the exact check (λ) |
|---|---|---|
| IPOPT per scenario | 2.89 MW | 0.38 MW (λ = 0.13) |
| LinDistFlow LP per scenario | 1.44 MW | 1.10 MW (λ = 0.77) |

The cheap LP beat per-scenario IPOPT after the check in most tested intervals. GENCO-OPF imitating per-scenario IPOPT would inherit this and fail criterion 2 by construction.

**Also corrected:** the per-scenario OPF let units supply reactive power, which the exact check does not model. From now on units inject **active power only**, and each OPF imposes only the constraints relevant to its direction, exactly as the exact check does: export checks overvoltage and loading; import checks undervoltage and loading.

### New reference optimum: the multi-scenario AC-OPF (`gencodoe.opf_multi`)

**Formulation.** For each interval and direction, one AC-OPF with a **shared** limit vector x across the design scenarios (the scenario approach):

> maximise Σ x_i, subject to, for each kept scenario: polar AC power flow with unit injection ±x_i (P only), the direction-relevant voltage bound, sending-end branch |S| ≤ rating, and 0 ≤ x_i ≤ cap_i.

**Discarding** follows the design rule, which allows S − ⌈0.98·S⌉ = 4 of 200 scenarios to fail:
1. Scenarios infeasible at x = 0 are discarded first. If more than 4 are, the envelope is 0 (zero sentinel).
2. The rest of the allowance is spent greedily: drop the kept scenario with the smallest direction-relevant headroom at the current x, re-solve, and stop when hosting no longer grows.

**Solver.** JuMP (PowerModels' internal copy) with IPOPT; the exact check and scale-down follow as for every method.

**Validation-day 80 result** (development): it passes the exact check unchanged (λ = 1), hosts 6–13% more than the LP after the check in the 5 tested non-zero intervals, and costs **7–24 s per interval and direction**.

### GENCO's role (replaces section 5)

A **GENCO-style graph model amortises the multi-scenario OPF**. It maps the interval's scenario statistics to the shared optimal vector, x̂_i / cap_i ∈ [0, 1], followed by the same exact check and scale-down.

**Architecture.**
- The encode–process–decode TransformerConv processor family used in the four-wire extension, with asinh-scaled impedance and rating edge features and no final LayerNorm in the encoders.
- **Trained from scratch** on case33bw. There is no pretrained GENCO checkpoint for this network, and this is disclosed.

**Inputs.**
- **Node:** per bus, the 7 statistics (mean, min, p02, p10, p90, p98, max) over the 200 design scenarios of net P and of Q; cap; PV flag; root flag; depth; direction flag.
- **Edge:** r, x, rating.

**Labels.** Multi-scenario OPF optima on **training** days, with validation days for early stopping. Test days are only evaluated once, in the frozen evaluation.

**Baselines for the learned model.** An MLP on the same features flattened (same budget, same training rule), so that any gain can be attributed to the graph structure.

### Methods in the final evaluation (test days)

- uniform;
- LP (per scenario, quantile, check);
- IPOPT per scenario (quantile, check);
- **multi-scenario OPF (the reference)**;
- GENCO-style graph model + check;
- MLP + check.

### Success criteria (replace section 8)

GENCO-style versus the reference and the LP, after the exact check:

1. Hosting ≥ 95% of the multi-scenario OPF.
2. Hosting ≥ 1.05 × LP.
3. Held-out P(ok) no worse than the multi-scenario OPF's minus 0.005 (mean) and minus 1 point (share ≥ 0.95).
4. Proposal plus check wall-clock below the multi-scenario OPF's, reported per hardware.
   - The claim "near-optimal at a fraction of the cost" needs 1–4.

**Also reported:** GENCO-style versus the MLP. A graph-structure claim needs the GENCO-style model to beat the MLP by at least 2 points of hosting relative to the reference, with a day-bootstrap lower bound above 0.

## Amendment 3 (2026-10-21, after the stage A result was frozen; exploratory and post-hoc): the actual GENCO on case33bw

**Why.** Stage A benchmarked a GENCO-*style* architecture trained from scratch, because no GENCO checkpoint existed for case33bw and graphkit has no task for the shared-envelope target. That answers "can a GENCO-style GNN amortise the OPF", not "does GENCO help". This amendment adds the missing arm. It is written before any case33bw PF state is generated and before any fit. The stage A result (`results/genco_opf_stageA_report.json`) stays frozen and is not re-run.

**Disclosure.** The test days 100–119 have been evaluated once for the six stage A methods. The one method added here is evaluated once on the same days, after development on validation days only. Because the baselines were seen first, the comparison is exploratory, and this is stated wherever the result is reported.

**GENCO PF model (`gencodoe.datakit_case33`, graphkit training).**
- The ENWL1 recipe, unchanged: graphkit `GNS_heterogeneous_lv` (asinh encoders) with the linear prior, hidden 12, 12 layers, 8 heads, `MaskedBusMSE`, AdamW 5e-4, batch 64, at most 80 epochs, patience 12 on validation loss (`configs/genco_pf_case33_lv_prior.yaml`). Final-layer bus embedding width 96.
- Training states: 20,000 PF states from the stage A scenario generator (section 6) on days 0–99 only, hour uniform; with probability 0.5 a state carries an envelope injection: direction export or import (50/50), every unit at one level uniform in [0, cap] (the augmentation of the ENWL1 v2 dataset). Scenario seed separate from the OPF streams (datakit seed 42 + 3000). Splits by day: 0–79 train (weights, normalizer), 80–99 val (early stopping, model selection). Test days are never drawn.
- MATPOWER export for datakit with loose limits (0.5–1.5 pu, no ratings) and the slack pinned at 1.0 pu, as for ENWL1; the stage A limits and ratings are applied by the exact check only.
- **Gate G3b (as in the workplan), on 2,000 fresh val-day states (same generator, seed + 4000):** Vm MAE < 0.002 pu and p99 < 0.01 pu against the exact sweep. If the gate fails, the GENCO arm is reported as "no usable case33bw GENCO" and nothing below runs.

**GENCO-conditioned envelope model (`gnn_genco`).**
- The stage A GENCO-style model (same architecture, config, loss, training rule, seed, labels, early stopping) with the frozen GENCO's final-layer bus embeddings of one state per interval and direction, the per-bus mean of the 200 design scenarios, concatenated to the 19 node features (19 + 96). The backbone is never trained; no pooled representation (the PhaseGENCO recipe of the four-wire extension).
- One fit, tag `final`, on the stage A labels (days 0–79, val 80–99).
- Evaluation: the stage A pipeline, method `gnn_genco` only, on the test days, through the same exact check and scale-down, appended to the frozen rows; no other method is re-run.

**Criterion (mirrors the graph-structure claim).** "GENCO helps" needs `gnn_genco` to beat the from-scratch `gnn` by at least 2 points of hosting relative to the multi-scenario reference, combined over directions, with a day-bootstrap lower bound above 0 (2,000 replicates, seed 0); and criteria 1 and 3 of amendment 2 must still hold for `gnn_genco`. Criterion 2 against the LP and criterion 4 are also reported for `gnn_genco`. Anything else, including a gain between 0 and 2 points, is reported as "GENCO does not materially help on case33bw".

## Amendment 4 (2026-10-21, before any state is estimated): partial metering, state estimation and the value of information

**Why.** Stage A assumed the operator knows the scenario distribution and nothing about the realised state. A DSO that issues an envelope for the coming interval has smart-meter readings from a share of the participating homes and voltage measurements at a few points. Estimating the unmeasured homes narrows the scenarios, and the exact check turns narrower scenarios into larger envelopes at the same safety level. State estimation is also the one role where a learned model does inference rather than imitation, so it is the fairest test of GENCO left in this project (suggested by R., GridFM team). Written before any measurement is simulated, any estimator is fitted or any envelope is computed under it.

**Setting (case33bw, stage A network, limits, caps and scenario generator).**
- **Realised state.** Per interval one draw from the scenario generator, seed 303, never used elsewhere. Development on validation days 80–99; the frozen run on test days 100–119, evaluated once per estimator and metered share.
- **Metering.** A share f of the 32 flexible households is metered, f ∈ {0.1, 0.3, 0.5, 0.7, 1.0}; the subset is drawn per (day, f) with seed 404 and is the same for every estimator. A metered home reports P and Q with 1% relative Gaussian noise. Voltage magnitude is measured at the substation and at four fixed buses, the feeder end and the end of each lateral (case33bw buses 18, 22, 25, 33), with 0.2% relative Gaussian noise (about 0.002 pu). No outliers. Measurements are redrawn per interval.
- **Prior.** The interval's 200 design draws from the stage A distribution (seed 101), as in stage A.

**Estimators of the unmeasured households** (each gives 200 design scenarios; metered homes are fixed at their readings in every scenario).
1. **prior:** unmeasured homes from the prior draws, unchanged. The floor: metering without estimation.
2. **WLS:** Gauss–Newton weighted least squares on the voltage measurements with the prior mean as pseudo-measurements of the unmeasured homes (weight 1/prior variance per bus), sensitivities from the exact sweep by finite differences at the prior mean. The classical distribution-system state estimator.
3. **GENCO-SE:** graphkit `StateEstimation` task on the LV variant (no linear prior: it needs known injections), trained from scratch on the amendment 3 PF states (days 0–99, splits by day) with graphkit's measurement simulation: power injections masked at ratio 0.5 with 1% relative noise, voltage magnitude masked at ratio 0.85 (about five of 33 buses) with 0.2% relative noise, no outliers; at most 80 epochs, patience 12, seed 0, one fit. At inference the measured buses are exactly the metered homes and the five voltage buses. **Gate:** on 2,000 validation-day states at f = 0.5, Vm MAE < 2e-3 pu and MAE of the unmeasured active loads below the prior mean's; otherwise "no usable case33bw GENCO-SE" and arm 3 is not run.
- **Design scenarios for WLS and GENCO-SE:** Gaussian draws around the estimate with a per-bus, per-f standard deviation calibrated as the residual std of that estimator on the validation days (the same rule for both), truncated so load sign is kept. The training mask ratio of 0.5 does not match every f; this is disclosed, and no per-f retraining is done.

**Envelope.** The LinDistFlow LP proposal (per scenario, quantile) followed by the stage A exact check and scale-down (S = 200, acceptance 0.98) on the estimator's scenarios, both directions. The LP is used instead of the multi-scenario OPF to keep the sweep to minutes; the choice is the same for every estimator and is disclosed.

**Metrics** (per f, estimator and direction, combined over directions; day-block bootstrap, 2,000 replicates, seed 0).
- Hosting after the check, MWh per day, and its ratio to f = 1.0 (all homes metered) and to the prior at the same f.
- **Realised feasibility:** the share of intervals whose issued envelope is feasible at the realised state (exact sweep). Target ≥ 0.95. An estimator below 0.95 at some f is reported as unsafe at that f whatever its hosting; the exact check only guarantees the design rule against the estimator's own scenarios.
- Estimation error of the unmeasured active loads (MAE, kW) per estimator and f; cost per interval.

**Criteria.**
1. **Value of information:** reported as the hosting curve over f for each estimator; no bar.
2. **GENCO-SE beats WLS** if, at both f = 0.3 and f = 0.5, hosting ≥ 1.02 × WLS with the bootstrap lower bound of the ratio above 1, and realised feasibility ≥ 0.95. Otherwise "GENCO-SE does not beat WLS on case33bw".
3. Both estimators are reported against the prior; an estimator that does not beat the prior's hosting at f ≤ 0.5 is reported as adding nothing.

**Claim discipline.** Exploratory layer; one network; synthetic realised states from the same generator as the prior (the estimators are not tested under model mismatch, and this is stated). Negative results are reported.

### Amendment 4a (2026-10-21, after the first test run of amendment 4; correction, disclosed)

The first run (`results/genco_opf_amendment4_report_run1.json`, hydra jobs 13362188/13362189, commit 43c1e9d) exposed two flaws in the amendment 4 design, both visible without any estimator:
1. **Metered readings entered the scenarios without their noise.** With every home metered the 200 scenarios collapse to one noisy point, the envelope is sized to its boundary, and realised feasibility of issued export envelopes fell to 0.33 at f = 1.0 (0.98 at f = 0.1, 0.95 at f = 0.5 for the prior). Correction: each scenario carries its own draw of the 1% measurement noise on the metered homes (same seed stream for every estimator).
2. **Zero envelopes counted as unsafe.** A zero import envelope is issued when the passive load already violates a limit; its "feasibility at the realised state" is the base case's, not the envelope's. Correction: realised feasibility is reported for issued (non-zero) envelopes, with the zero-envelope share alongside; the all-interval figure is kept in the report.

The GENCO-SE gate failed in the first run as it would in any run (Vm MAE 1.7e-3 pu but unmeasured-load MAE 329 kW against the prior's 7.6 kW: graphkit's StateEstimation task derives injections from the predicted voltages by physics, which is unusable at this accuracy on a distribution feeder); the arm stays dropped. The test days are re-run once with the corrected scenario set for prior and WLS; the first run is kept and both are reported.

### Amendment 4b (2026-10-21, before any fit of the variant): GENCO masked reconstruction as the state estimator

**Why.** graphkit's StateEstimation task cannot estimate loads here (amendment 4a). GENCO's own pretraining objective is masked reconstruction, where masked bus quantities are predicted directly, which is the formulation that fits a load-estimation task. In graphkit 0.9.0.3 the heterogeneous model's PF mode has a two-wide voltage head, so a direct load prediction needs a head of our own.

**Model (`GNS_heterogeneous_lv_recon`, registered in `gencodoe.genco_model`).** The LV variant in PF mode with graphkit's random masking (every bus quantity PD, QD, VM, VA, QG masked independently with probability 0.5; generator PG and branch flows likewise), plus a linear reconstruction head on the final bus embedding that predicts [PD, QD]. The bus output is [Vm, Va, Pg, Qg] from graphkit's physics decoder followed by [PD, QD] from the head; loss `MaskedReconstructionMSE` (masked entries of all six). Measured voltages are kept fixed by graphkit's PF masking, so a voltage reading is enforced, not just an input. No measurement noise in training (graphkit's random masking has none); the 1% / 0.2% noise is applied at inference and this mismatch is disclosed. Same data, splits, optimiser, epochs, patience and seed as amendment 4; one fit.

**Inference pattern.** Known: P and Q at the metered homes (noisy), P = Q = 0 at the substation, Q generation 0 at every load bus, |V| at the five voltage buses (noisy). Masked: everything else, all generator P and all branch flows. The unmeasured loads are the head's [PD, QD] at those buses.

**Gate, calibration, criterion:** as in amendment 4 (gate on 2,000 validation-day states at f = 0.5: Vm MAE < 2e-3 and unmeasured-load MAE below the prior mean's; calibration on validation days; criterion 2 against WLS at f = 0.3 and 0.5; realised feasibility ≥ 0.95). Evaluated once on the test days in a separate output directory together with prior and WLS, whose rows are deterministic and identical to run 2. The variant is named `genco` in the rows; the run manifest records which checkpoint.

## Amendment 5 (2026-10-21, before any label, fit or evaluation on case118): the official pretrained GENCO on case118, warm start and scarce-label transfer

**Why.** Every GENCO result in this project so far (ENWL1 roles, stage A, amendments 3–4b) used a model trained from scratch with tens of thousands of labels. The official checkpoints on the Hub (`gridfm/genco-opf-datakit`, `gridfm/genco-pf-transfer`, Apache-2.0) were never loaded. The GENCO paper's claim is that pretrained weights pay off when labels are scarce; that regime is untested here. This amendment is built on Evgenii's branch `evgenii/opf-envelopes` (case118 with 99 flexible export units: `scripts/export_flex_case.py`, `grids/case118_flex.m`, the datakit and graphkit configs, `src/gencodoe/opf_envelopes.py` with batched GENCO OPF inference, the PowerModels/IPOPT baseline, the exact AC check and GENCO-warm-started IPOPT), without rebasing or rewriting it. Case118 is a fresh network. The case33bw test days 100–119 are not touched.

**Checkpoint.** `gridfm/genco-opf-datakit`, `case118_ieee/base/seed0`: `last.pt` (the model card's choice), `normalizer_stats.pt`, `case118_ieee_base_seed0.yaml` (GNS_heterogeneous, hidden 48, 12 layers, 8 heads; trained on 250,000 datakit OPF scenarios of plain case118). SHA-256 of the downloaded files is recorded in the run manifest. Our installed graphkit is 0.9.0.3 (main); the card warns that main may not reproduce the paper branch's numbers, and whatever we get is recorded.

**Step 0, pre-checks (laptop-sized; outcome reported before anything else runs).**
- (a) Evaluate the checkpoint on plain case118 datakit OPF scenarios generated locally with the load settings of the checkpoint's own training set (`gridfm/opf_small_case118_ieee`, `args.log`), seed distinct from any training seed: report Pg RMSE and Vm MAE against the labels, next to the card's `metrics.csv` (Pg RMSE 17.1 MW, optimality gap 1.64%).
- (b) Build the graph of one `case118_flex` datakit OPF scenario (99 extra generators) with the checkpoint's graph builder and normaliser and run inference. Pass = runs, finite outputs, flexible dispatch within [0, cap] after clipping. If (b) fails, option 2 is off and option 1 runs on plain case118.

**Option 1 (no training): pretrained GENCO as warm start for the exact OPF.** On 500 held-out scenarios (flex case if (b) passes, plain case118 otherwise; datakit OPF scenarios with a dedicated seed, used for nothing else), PowerModels/IPOPT AC-OPF from
(i) flat start; (ii) warm start from the previous scenario's solution (the honest baseline); (iii) warm start from the pretrained GENCO's dispatch and voltages; (iv) same as (iii) from a from-scratch checkpoint if one exists on the branch (none is committed; reported as not available otherwise).
Reported: IPOPT iterations and wall-clock per solve (median, p90), convergence failures, and objective match to the flat-start optimum (1e-6 relative). Hardware stated (hydra zen4, one core, Julia/IPOPT versions). **Preregistered success:** (iii) cuts the median wall-clock by at least 20% against (ii), not (i), with a scenario-bootstrap lower bound above 0. Anything less is "no material warm-start gain".

**Option 2 (only if (a) and (b) pass and hydra is up; go/no-go 23 Oct 09:00): pretrained versus from-scratch at scarce labels, as envelope proposal.** Datakit OPF labels on `case118_flex` (Evgenii's config, 30k scenarios, split by load scenario; the test split is never used for selection). Fine-tune the pretrained OPF checkpoint and train from scratch at N = 100, 1,000 and 10,000 labels, one seed each, identical config otherwise, early stopping on validation. Each model's predicted flexible dispatch is the proposal; the exact check is the IPOPT re-dispatch check already in `opf_envelopes.py`; scale-down by the k-section rule. Reference = IPOPT AC-OPF per scenario. Metrics on the test split: hosting after the check as a share of the reference, AC feasibility of the raw proposal, proposal time. **Preregistered criterion:** at N = 1,000, fine-tuned hosting share exceeds from-scratch by at least 2 points with a scenario-bootstrap lower bound above 0, and fine-tuned is within 95% of the reference. The full curve is reported either way.

**Claim discipline.** "Exact-call reduction" is not "speed-up" unless wall-clock was measured on stated hardware (option 1 measures it). No "certified". The IPOPT solution is a local optimum; "reference" not "optimum". Negative results are reported. Deliverables: `results/genco_opf_amendment5_warmstart.json`, and if run `results/genco_opf_amendment5_transfer.json`; run manifest with commits, job IDs and hashes; experiment-log entry with the exact supported sentence.

### Amendment 5a (2026-10-21, after step 0 and before any option 1 result is read; corrections from the external audit and the option 2 freeze)

**Step 0 wording.** (b) was a three-scenario flex smoke batch, not one scenario. (a) shows that the checkpoint loads correctly on graphkit 0.9.0.3 main and produces finite, plausible outputs with accuracy of the same order as the published checkpoint result, on an easier local slice (base topology, no cost or admittance perturbation); the 9.6 MW Pg RMSE is not directly comparable with the card's 17.1 MW. **Pass rule for (a), stated now:** the checkpoint loads, outputs are finite, and Pg RMSE on the local slice is at most twice the card's figure (≤ 34 MW). (a) passes.

**Option 1, solution-quality guard.** A warm-start "success" also requires: (iii)'s convergence failures no more than (ii)'s plus 1 point of the 500, and (iii)'s objective within **1e-4 relative** of the flat-start objective in at least 99% of the scenarios where both converged. The 1e-6 share is reported alongside; the three-scenario smoke on the laptop showed objective differences above 1e-6 for both warm starts, so a 1e-6 requirement would fail every warm start on solver tolerance alone. The 1e-4 level is the preregistered tolerance; the per-scenario objectives are in the result file so any other tolerance can be read off. The job 13362503 runs code that computes `pass` by the timing rule alone; the guard is applied by `opf_warmstart --recompute` on its result file and both verdicts are recorded.

**Option 2, frozen.**
- Labels: `configs/datakit_opf_case118_flex.yaml` (30,000 scenarios, seed 42) generated on hydra; SHA-256 of the raw tables recorded.
- Splits (`scripts/amendment5_splits.py`, seed 0): a fixed random permutation of the solved scenario ids; the first 3,000 are the **test** split, the next 3,000 **validation**, the rest the training pool. Label subsets are **nested**: the first N of the training pool for N ∈ {100, 1,000, 10,000}. The split files' SHA-256 are recorded; the test split is never used for selection.
- Arms, identical except for initialisation: graphkit `finetune --model_path <official last.pt>` (all layers trainable; graphkit's finetune has no freezing) versus `train` from scratch; the same config (`GNS_heterogeneous`, hidden 48, 12 layers, 8 heads, AdamW 5e-4, batch 64, the OPF loss mix of Evgenii's config), the same budget (at most 80 epochs, patience 12 on validation loss, one seed 0), and **target normalisation for both** (graphkit fits the normaliser on the training subset; the pretrained arm therefore sees case118_flex statistics, not the source statistics; disclosed).
- Evaluation on the first 1,000 test-split scenarios (sorted ids): proposal = the model's flexible dispatch clipped to [0, cap]; exact check = the IPOPT re-dispatch check of `opf_envelopes.pf_violation` with the conventional units re-dispatched; scale-down = the largest λ ∈ [0, 1] on a 4-level k-section with the proposal feasible (worst violation ≤ 1e-4); hosting after the check = λ·Σ flexible dispatch; reference = the IPOPT label dispatch Σ flexible. Metrics: hosting share of the reference (sum over scenarios), AC feasibility of the raw proposal (share with worst violation ≤ 1e-4), proposal time. Scenario bootstrap (2,000 replicates, seed 0) of the hosting-share difference fine-tuned minus scratch.
- Criterion as in amendment 5. **Uncertainty statement:** one training seed per arm; the bootstrap measures test-scenario uncertainty conditional on these two fits, not training-seed uncertainty.

### Amendment 5b (2026-10-21, before any option 2 label, fit or evaluation): the exact check on case118_flex

The three-scenario smoke of the option 2 evaluation (`gencodoe.opf_transfer`, local flex scenarios, no model trained) showed two properties of the flex case that the amendment 5a check did not allow for:
1. **Zero flexible export can be infeasible.** With the flexible units fixed at 0, or switched off, IPOPT reports local infeasibility from a flat start in every smoke scenario: at these load levels the network needs the export. Feasibility is therefore not monotone in the scale factor, and a bisection from 0 is undefined.
2. **The label itself fails the check from a flat start.** The IPOPT label dispatch fixed exactly (pmin = pmax) is reported locally infeasible unless IPOPT starts near it.

**Check, as implemented and frozen now.** For a proposal p̂ (flexible dispatch clipped to [0, cap]) and scale λ: the flexible units are fixed at λ·p̂, the conventional units are re-dispatched by IPOPT **started from the scenario's plain OPF solution** (flexible units free; computed once per scenario and shared by every model and λ), and the proposal at λ is feasible iff IPOPT converges. λ is the largest value on the grid {1, 0.75, 0.5, 0.25} that is feasible, else 0 (at most 4 solves per scenario and model). Hosting after the check = λ·Σ p̂. The reference labels pass this check at λ = 1 in the smoke (3 of 3), the pretrained checkpoint's zero-shot proposal passes at λ = 1 with 63% of the reference's export. The raw full-solution feasibility (`pf_violation` with the model's own conventional dispatch and set-points) is reported alongside and is not the check. Costs: about 0.3 s per scenario and model on the laptop.

### Exploratory addendum to amendment 5 (2026-10-22, after the option 2 result was read): training-seed replicates

The option 2 criterion failed at N = 1,000 (fine-tuned 89.4% of the reference against the 95% bar; gain over scratch +6.9 points, interval [5.9, 7.8]), and the fine-tuned model hosted more at N = 100 (96.2%) than at N = 1,000. Each arm was fitted once, so the bootstrap covers test scenarios only. To measure training-seed variability we refit all six arms with seeds 1 and 2: same configs, splits, epochs and evaluation, only the `seed:` line changed (`SEED` in `scripts/hydra_amendment5_train.slurm`); each seed is evaluated separately into `results/genco_opf_amendment5_transfer_s{1,2}.json`. **This is post-hoc and exploratory. The preregistered verdict stays the seed-0 result (fail) whatever the replicates show.** We will report, per N and arm, the three hosting shares and their range, and whether the N = 100 > N = 1,000 ordering of the fine-tuned arm holds in each seed.

