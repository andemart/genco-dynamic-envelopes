# Four-wire GENCO across 25 ENWL networks: protocol (scaling, replication, transfer)

**Status: frozen** by the commit that adds this file on branch `ander-fourwire-scale`.
- Exploratory extension of `docs/unbalanced_genco_protocol.md` (the ENWL1 four-wire study). Its frozen results are untouched.
- Committed before any Fluvius state is generated on the new networks, before any label is generated and before any model is trained or timed on them.

**What was seen before this protocol (disclosure).**
- **Pilot timings** on synthetic random loads (normal, 0.6 ± 1.5 kW per customer, PF 0.95), for the NumPy sweep, the one-step prior and OpenDSS on all 25 networks. No GENCO model was run. These timings show that **the NumPy sweep's cost grew faster than network size over this sample and loading regime**. They do not show that four-wire sweeps scale superlinearly in general: that needs the optimised sweep and Fluvius states (E2).
- **Conversion checks:** network_1 equals ENWL1 exactly (397 retained buses, identical branch impedances). On 24 networks the NumPy sweep agrees with OpenDSS (default tolerance) to ≤ 1.3e-4 pu at those loads.
- **The network_13 diagnosis** (§2.3), the light-load screen of all 128 feeders, and the cable-rating sources (§2.4).
- **ENWL1 = network_1.** Its global test-split results are already published. network_1 is therefore not a fresh test network, for any family.

**Terminology.** A *network* is one ENWL substation: its feeders merged under one 800 kVA transformer, as in ENWL1. "Feeder" means one CSIRO feeder folder. **GENCO** here is the four-wire GENCO-style model of the ENWL1 study, `GNS_heterogeneous_4wire` ("PhaseGraph"). It is **not** the official balanced GENCO checkpoint, which cannot represent four-wire states.

## 1. Research questions

1. **Replication (E1).** Do all 25 networks convert to the four-wire representation and agree with OpenDSS?
2. **Computational scaling (E2).** As networks grow from 42 to 883 customers, does the GENCO screening pipeline, with exact fallbacks included, become faster end to end than an **optimised four-wire backward/forward sweep**, without losing safety?
3. **Model transfer (E3).** Does a GENCO model pretrained on many networks transfer to unseen networks? Compared at equal target fitting-label budgets:
   - fine-tuned against trained from scratch;
   - pretrained weights used directly (zero-shot in weights, and fully zero-shot), as a statement of current limits.

**E3 is what makes this a GENCO (grid foundation model) study.** E2 alone, with one model per network, shows repeatability, not transfer.

## 2. Data

### 2.1 Source
- CSIRO "Four-wire low voltage power network dataset" (Heidarihaei, Geth, Claeys, Urquhart; DOI 10.25919/jaae-vc35), `Four-wire/` subset: 25 networks, 128 feeders, 640 files.
- Derived from ENWL Low Voltage Network Solutions. Dataset licence CC BY 4.0. Its 4×4 line impedances derive from Urquhart & Thomson, "Cable impedance data" (Loughborough University), **CC BY-NC 3.0** (non-commercial).
- **Integrity:** SHA-256 over the sorted per-file SHA-256 list of `Four-wire/`, `LICENSE.txt` and `README.md` = `0c8411cb398992facd5f9b65794b6f18411c35ede93c09b4ca907cfdf1d6e83c`.
- Committed unchanged in `data/feeder_topology/ENWL_CSIRO/`, with its original README and licence.

### 2.2 Conversion to a network (frozen; implemented in `src/`, validated by E1)
1. **Source and transformer:** ENWL1's `Master.dss`. That is an 11 kV source (ISC3 3000, ISC1 2500), an 800 kVA Dyn 11/0.416 kV transformer (XHL 1%) and the neutral earthed only at `TRbus`. Every CSIRO `Transformers.txt` specifies the same 800 kVA unit.
2. **Feeder bus `1`** (the transformer LV terminal) becomes `TRbus`. Every other bus b of feeder k becomes `k_b`. The CSIRO stub `LINE0` (sourcebus → 1, 1 m) is dropped.
3. **Line codes** are merged across feeders by their original cable name, with one global code per cable type. Ratings are as in §2.4.
4. **Single-phase loads** are kept, with the same phase and the neutral return.
5. **Three-phase loads** (9 in total: Network_8 1, network_12 7, network_16 1) become three single-phase phase-to-neutral customers of S/3 each, sharing one meter profile. For a balanced constant-power wye load with a neutral, this is electrically identical. They count as one customer for profile allocation and envelopes, and as three phase-connections for the solvers.
6. The reduction (pruning, series merging, Thevenin root) is the ENWL1 one, unchanged.

### 2.3 Eligibility (frozen; applied to every feeder of every network, not just to network_13)
- **Light-load screen.** At a uniform 0.5 kW per customer, PF 0.95, the minimum customer phase voltage must be at least 0.9002 pu of 240.18 V (the UK lower limit). This must hold in **both** ENWL's original three-wire model and the CSIRO four-wire feeder, each solved standalone.
- A network is **eligible for E2 and E3** only if all its feeders pass.
- **Result (seen before this protocol):** 127 of 128 feeders pass. network_13/Feeder_4 fails: 0.751 pu in ENWL's original model and 0.673 pu four-wire, and already 0.61 pu at 1 kW per customer in the original.
  - That feeder has 154 of its 173 customers on phase b in ENWL's original data, not only in the four-wire conversion. No realistic load keeps it within limits.
  - **network_13 is excluded from E2 and E3** under this rule. It stays in E1, which reports it with this diagnosis.
- **Eligible:** 24 networks, of which 23 are fresh (network_1 is not; see the disclosure above).

### 2.4 Cable ratings (frozen)
`NormAmps` per cable type, as used for the cable-overload label, the screen and the edge features.
- Ampacity depends on installation conditions. These values are **operational ratings for an assumed installation**, not intrinsic properties of the cable designation.
- CP203 = ENWL, *Current ratings – underground cables*, CP203 Issue 7, May 2023, https://www.enwl.co.uk/globalassets/get-connected/cic/icpsidnos/g81-policy/policy-library-documents/design-and-planning/cp203-i7---cable-ratings.pdf . CP203 states that it is the approved source of ratings for ENWL underground cables, and that BS 6480 itself does not specify ratings.

| Cable (original name in the dataset) | NormAmps (A) | Source |
|---|---|---|
| BS 5467 1997 table 10, 16 | 110 | ENWL1 `LineCode_3f4w.dss` (commit 1222968); original source not recovered; not in CP203 |
| BS 5467 1997 table 10, 25 | 141 | as above |
| BS 5467 1997 table 10, 35 | 174 | as above |
| BS 5467 1997 table 10, 70 | 269 | as above |
| BS 6480 1988 table 17, Cu 16 | 101 | ENWL1 file, source not recovered. **Differs from CP203 Table 12** (4-core stranded Cu, paper insulated, lead sheathed, steel-tape armoured, 16 mm²: 89 A direct in ground, 76 A in duct). See the note below. |
| **BS 6480 1988 table 17, Cu 120** | **280** | CP203 Table 12, 120 mm² Cu: continuous rating direct in ground, g = 1.2 K·m/W, Tg = 15 °C, for the equivalent construction. **Not in ENWL1's file.** |
| Waveform 4-core 95 (IEC 60287) | 235 | ENWL1 file; equals CP203 Table 6 (95 mm² Al Waveform, 1.2 K·m/W, 15 °C, direct in ground) |
| Waveform 4-core 185 (IEC 60287) | 335 | as above, 185 mm² |
| Waveform 4-core 300 (IEC 60287) | 435 | as above, 300 mm² |

- **Cu 16: 101 A stays primary,** so that network_1 reproduces ENWL1's published loading labels exactly. The source of 101 A is unrecovered; the one documented source, CP203 Table 12, gives 89 A for the equivalent construction. Cu 16 is used in all 25 networks (678 segments in network_1).
- **Rating sensitivity (frozen; no criterion uses it):**
  - one variant with Cu 120 = 240 A (CP203 Table 12, one cable per duct);
  - one variant with Cu 16 = 89 A (CP203 Table 12, direct in ground).
  - For each, the E1 violation shares and the E2/E3 decision metrics are re-reported on the test states, with the frozen screens and models (cable ratings are edge inputs, so each variant re-evaluates the models with its ratings). Labels change only in their cable-overload flags.

### 2.5 Household profiles and states
The ENWL1 generator (`gencodoe.datakit_fluvius`), applied per network:
- Each state draws one 15-minute 2024 calendar step from the split's days, and a technology mix (HP, EV, PV shares, each uniform in [0, 1]).
- Each customer gets a group from that mix and a random Fluvius meter of that group, without replacement within the state.
- **Meter shortage (new, frozen).** If a state needs more meters of a group than exist, the excess customers reuse meters of that group (drawn with replacement). Reported per network and technology group: the share of repeated assignments and the unique-meter fraction.
- Offtake at PF 0.95. Injection at unity power factor.
- **Envelope augmentation 0.5. A new, prospective stress-design choice,** disclosed as such. The generator default is 0.0. The ENWL1 v2 setting was not recovered, so 0.5 is not presented as a replication of it. Half of the states carry a uniform envelope injection (export ≤ 5.0 kW, import ≤ 7.4 kW per customer).
- **Seeds:** generator seed = 1000 × network number + split index (train 0, val 1, calib 2, test 3).
- **States per network:** train 10,000, val 3,000, calib 2,000, test **4,096** (all distinct, so that a timing batch of 4,096 needs no duplicated state).

### 2.6 Splits
- **By time:** the existing global calendar-day rule, identical for every network. train d%5 ∈ {0, 1, 2}; val d%5 = 3; calib d%5 = 4 with ⌊d/5⌋ even; test d%5 = 4 with ⌊d/5⌋ odd.
- **By network (E3): grouped four-fold, leave-six-networks-out.**
  - The 24 eligible networks are sorted by retained-bus count (ties broken by network number) and assigned round-robin to folds 0–3. That gives 6 networks per fold, with every size class in every fold.
  - Each network is unseen exactly once; the pretraining pool is the other 18.
  - **This is not leave-one-network-out** and is never described as such.
- **Test isolation:** as in the ENWL1 protocol §3. Test labels are never read by training, debugging or calibration code; a test-isolation check enforces this.

## 3. Exact solvers

| Solver | Role | Setting (frozen) |
|---|---|---|
| **OpenDSS** (four-wire, snapshot) | **authoritative accuracy reference and labels** | `Set tolerance=1e-6`, `maxiterations=100`, constant-PQ loads with Vminpu 0.5 / Vmaxpu 1.5, no-load reset after a failure (as `UnbalancedENWL1`); failure rule below |
| **Optimised four-wire sweep** | **the conventional speed baseline that GENCO must beat** | the U3 algorithm (ENWL1 protocol §5), CPU and GPU implementations, **FP64** (below) |
| NumPy sweep (`fourwire_net.sweep`) | transparent reference implementation, never the speed baseline | FP64, as above |

**OpenDSS failure rule:**
1. Solve with OpenDSS's default fixed-point algorithm (`Set algorithm=normal`).
2. If it fails to converge, reset and retry with `Set algorithm=newton`, at the same tolerance and iteration limit.
3. If both fail, the state is **solver-indeterminate**:
   - **Not called a physical violation.**
   - Treated as **operationally unsafe** for screening: clearing it is an error, reported as *cleared-indeterminate*, separately from false-safe.
   - Excluded from state-accuracy metrics.
   - Counted and reported per network and split, together with the number of states rescued by Newton.

**Common sweep settings** (every implementation):
- Convergence: maximum voltage update ≤ 1e-8 pu, at most 100 iterations.
- Flat start (nominal phasors) for every state. No warm start for any method, GENCO included.
- Thevenin root: fitted per network as in ENWL1 (amendment 2, item 1), from training-split labels only. **Fitting set and description amended: see Amendment 1.**

**Optimised sweep implementations, both FP64:**
- **CPU:** depth-level batched (all buses at one tree depth in one vectorised operation), compiled with Numba, parallel over states, on all cores of the timing node.
- **GPU:** the same depth-level algorithm in PyTorch on the A100, `complex128`.
- **Before any GENCO timing,** both must agree with the NumPy sweep to ≤ 1e-9 pu on the val states of all eligible networks. Their implementation may be tuned until then, on synthetic and val states only, never against GENCO timings.
- **Speed baseline = the faster of the two,** per network and batch size.

**Sweep non-convergence:** a state not converged within 100 iterations is sent to OpenDSS (with the failure rule above). The extra cost counts in the sweep's end-to-end time. Reported per network.

**Four-wire Z-bus:** not implemented. No claim relative to Z-bus solvers is made.

## 4. Methods

| ID | Method | Families |
|---|---|---|
| U1 | One-step four-wire prior | E2, E3 baseline |
| **PG-net** | PhaseGraph trained from scratch on one network's full train split: the ENWL1 amendment 1 configuration, unchanged (hidden 64, 8 layers, 4 heads, AdamW 1e-3, cosine, batch 32, ≤ 60 epochs, patience 8, same loss weights), FP32 | E2, E3 |
| **PG-pre** | PhaseGraph pretrained on the 18 networks outside the target's fold: 5,000 train states per network (the first 5,000 in generation order), same configuration, validation on those networks' val states | E3 |
| **PG-ft(B)** | PG-pre fine-tuned on B target train states (first B in generation order), early stopping on the target's val states, lr 3e-4, ≤ 30 epochs, patience 5 | E3 |
| **PG-scratch(B)** | PG-net trained on the same B target states, same stopping rule | E3 |
| **PG-pre-tc** | PG-pre on the target without target training; screen calibrated on the target's calib states. **Zero-shot in its weights only** | E3 |
| **PG-zs** | PG-pre on the target, screen calibrated on the pretraining pool's calib states. **Fully zero-shot**: no target label used | E3 |

**Target fitting-label budget B** ∈ {200, 1,000, 5,000}. **Primary: B = 1,000.**
- B counts the target **training** states only.
- Every target-adapted method (PG-ft, PG-scratch, PG-net) also reads the target's fixed **validation (3,000)** and **calibration (2,000)** sets; PG-pre-tc reads the calibration set.

**Seeds:**
- PG-pre: **seeds 0, 1, 2** (12 pretraining runs).
- Primary comparison (B = 1,000): PG-ft and PG-scratch with seeds 0, 1, 2; PG-ft with seed s starts from PG-pre seed s.
- Everything else (PG-net, B = 200 and 5,000, PG-pre-tc, PG-zs): seed 0, using PG-pre seed 0.

**Input scaling:** asinh then standardisation.
- Statistics from the training networks: PG-net and PG-scratch use the target's train states; PG-pre, PG-ft, PG-pre-tc and PG-zs use the pretraining pool.
- Every input is graph-size agnostic, so one model applies to any network: per-bus injections, customer counts, depth, electrical distance, U1 prior; per-edge 4×4 impedances, rating, length.

## 5. Screen and fallback (identical for every learned method and U1)
- ENWL1 amendment 1 screen: one global split-conformal margin per criterion (minimum V_pn, maximum V_pn, maximum cable loading, transformer loading). α = 0.01 split over the 4 criteria, fitted on the converged calib states.
- Applied per direction (import: minimum V and loading; export: maximum V and loading).
- **Outcomes:** cleared safe, rejected as violating, or sent to the exact fallback. A method without a finite state defers.
- **Calibration source:** the target network's calib states for every method except PG-zs, which uses the pretraining pool's calib states.
- **Fallback engine:**
  - In E2 timing: the optimised sweep (the cheapest exact solver, validated by E1), with OpenDSS under the failure rule for sweep non-convergence.
  - Decision truth in every family: OpenDSS under the failure rule.

## 6. Metrics and tolerances (frozen)

**Limits (as ENWL1):**
- V_pn in [0.9002, 1.0534] pu of 240.18 V (UK ESQCR 216.2–253.0 V);
- cable current ≤ `NormAmps` (§2.4);
- transformer ≤ 800 kVA.

**State accuracy** on converged test states:
- V_pn MAE, p95, p99 and maximum;
- neutral-voltage error;
- phase- and neutral-current error, normalised by rating;
- transformer-loading error.

**Decisions:**
- **False-safe (primary safety metric):** cleared safe while converged OpenDSS truth violates in that direction. Reported as a count, as a rate over the truly violating converged states, and with its one-sided 95% Clopper–Pearson upper bound.
- **Cleared-indeterminate (reported separately):** cleared safe on a solver-indeterminate state.
- **False-unsafe (reported separately):** rejected while OpenDSS says safe.
- **Fallback share:** share of test states sent to the exact fallback, averaged over both directions.

**Timing procedure:**
- **Hardware:** Hydra A100-PCIE-40GB for models and the GPU sweep; the same node's CPU (all allocated cores, count recorded) for the CPU sweep. OpenDSS runs one process per core.
- **Precision:** models in FP32 with TF32 off; sweeps in FP64. No `torch.compile` for the models. Numba compile time is reported but excluded from steady-state time.
- **Repeats:** 3 warm-up repeats, then **100 timed repeats**; `torch.cuda.synchronize()` around every GPU timing.
- **Batch sizes:** 1, 256 and 4,096 states, from the network's distinct test states in a fixed order (batch 4,096 = the whole test split).
- **Internal microbatching** (e.g. for GPU memory) is allowed for any method. It runs inside the timed region and counts in its end-to-end time.
- **End-to-end time includes:** host-to-device transfer, feature construction, U1, model, screen, and the exact fallback solves for the deferred states. It excludes reading labels from disk.
- **Reported:** p50 and p95 latency per batch, and throughput (states/s) per method, network and batch size.

**Envelope-level check** (E2 networks only):
- Uniform hourly envelopes on 12 test days (every third test day in calendar order), both directions, with the ENWL1 envelope rule: 200 design scenarios, ε = β = 0.05, acceptance 0.98, k-section 4 levels per round, relative tolerance 1e-3.
- Run twice: with the optimised sweep as the evaluator, and with the GENCO screen + sweep fallback.
- **Agreement criterion:** |envelope difference| ≤ 1e-3 × cap in ≥ 95% of intervals.
- Any GENCO-pipeline envelope above the sweep envelope by more than 1e-3 × cap is reported as **unsafe-envelope**.

## 7. Experiment families and criteria

### E1 Numerical replication (all 25 networks)
- **Per network:** 200 Fluvius states from val days, generated as §2.5, all solved by OpenDSS (failure rule) and by the NumPy sweep.
- **Pass:** on states where OpenDSS converges, max |V_pn sweep − V_pn OpenDSS| ≤ 1e-4 pu, and max branch-current difference ≤ 1e-3 of the rating.
- A network failing E1 is excluded from E2 and E3. That exclusion is reported with the discrepancy.
- Also reported: Newton-rescued and solver-indeterminate shares, sweep iterations (median, p95) and violation shares (also under both rating-sensitivity variants of §2.4).

### E2 Computational scaling (preregistered six networks)

| Network | Customers | Retained buses (pilot) |
|---|---|---|
| network_25 | 42 | 79 |
| network_10 | 64 | 94 |
| network_1 (= ENWL1, not fresh) | 200 | 397 |
| network_7 | 471 | 876 |
| network_2 | 567 | 1,089 |
| network_17 | 883 | 1,668 |

**Methods timed:** OpenDSS, NumPy sweep, optimised sweep (CPU, GPU), U1 screen + fallback, and PG-net screen + fallback, with the PG-net trained on that network.

**Speed criterion, per network** (batch 4,096, primary):
- **Speed-up** = median end-to-end time of the speed baseline / median end-to-end time of the PG-net pipeline, over the 100 timed repeats.
- **Interval:** percentile bootstrap 95% interval, resampling the 100 repeats of each method independently, 2,000 replicates, seed 0.
- GENCO is **faster** on a network if all three hold:
  - the speed-up point estimate is **≥ 1.10**;
  - its 95% interval **excludes 1.0**;
  - false-safe = 0 and cleared-indeterminate = 0 on that network's test states.
- **Break-even size:** the smallest of the six networks, in retained buses, from which the criterion holds on every larger network.
- Also reported: log–log scaling exponents of time against retained buses for every method, and the speed-ups at batch 1 and 256.

**Claim rule:**
- "Faster than the optimised four-wire sweep" only for networks where the criterion holds, quoting the speed-up, its interval and the batch size.
- If the criterion holds nowhere, the result is reported as such.
- **Never:** "faster than conventional / traditional solvers" in general.

### E3 Model transfer (24 eligible networks, grouped four-fold; primary means over the 23 fresh networks)
- **Per target network:**
  - PG-scratch(B) and PG-ft(B) for B ∈ {200, 1,000, 5,000};
  - PG-pre-tc and PG-zs;
  - PG-net (full 10,000) as the per-network reference.
- **Primary criterion (B = 1,000):** fine-tuning beats training from scratch in fallback share by ≥ 2 percentage points.
  - Statistic: per network, the mean over seeds 0–2 of PG-scratch minus PG-ft fallback share; then the mean over the **23 fresh networks** (network_1 excluded).
  - The 95% interval must exclude 0. It is a paired bootstrap over networks, 2,000 replicates, seed 0.
  - The false-safe count of PG-ft must be no greater than PG-scratch's, summed over networks and seeds. The same holds for cleared-indeterminate.
  - Seed spread is reported per network.
- **Secondary (seed 0):**
  - the same comparison at B = 200 and B = 5,000;
  - PG-ft(B) against PG-net (target fitting labels needed to match the per-network model);
  - V_pn MAE.
- **Zero-shot:** PG-pre-tc and PG-zs, reported with fallback share, false-safe, cleared-indeterminate and V_pn error, against U1 and PG-net. **No success criterion.** It establishes the current limit, whatever the outcome.
- **network_1** is reported, and a sensitivity analysis with it included (24 networks) is given, but it never enters the primary mean.

## 8. Runs allowed
- **PG-pre:** 12 pretraining runs (4 folds × seeds 0, 1, 2).
- **Fine-tune and scratch fits:** one final fit per (method, network, budget, seed), with seeds as in §4.
- **PG-net:** one final fit per network, seed 0.
- **Debugging and smoke fits:** train and val only.
- **Test evaluation:** once per fitted model, with the frozen code.
- **Recorded:** code, data, schema, checkpoint and configuration hashes in a run manifest.

## 9. Gates before any final fit
1. Conversion and loaders in `src/`, with unit tests: series-sum preservation, the three-phase split, network_1 identical to ENWL1, bus-name maps, ratings as §2.4.
2. E1 complete and reported.
3. Optimised sweeps validated (§3).
4. Label integrity checks (ENWL1 extended checker) pass on every network and split.
5. A smoke fit of PG-pre on one fold (train/val only). Reported: memory, epoch time and the projected compute for all runs of §8.

## 10. Claim discipline
**Permitted when the criteria hold:**
- "Across N networks of 42–883 customers, the four-wire GENCO screening pipeline was X× [interval] faster end to end than an optimised four-wire sweep on networks of ≥ M buses, at batch 4,096 on an A100, with zero observed false-safe decisions."
- "A GENCO model pretrained on 18 networks reduced exact fallbacks on 23 unseen networks by Y points relative to training from scratch, with 1,000 target fitting states plus fixed validation and calibration sets."

**Never claimed:**
- speed-ups over solvers not timed here (Z-bus, Newton, commercial tools);
- formal safety from finite tests;
- that solver-indeterminate states are physical violations;
- that network_1 is a fresh test;
- leave-one-network-out evaluation;
- that this is the official GENCO checkpoint;
- a universal LV foundation model from 24 UK networks of one operator.

## Amendment 1 (2026-11-15): Thevenin fitting set (before any E1 label is generated)
Committed after the converter (commit 8d1c2b9) and before any Fluvius state or label is generated on the new networks. It replaces the fitting set in §3 ("Common sweep settings", Thevenin root) and corrects how the fit is described.

**Fitting set.**
1. For each network, generate the **first 500 candidate training states** in the frozen generator order (§2.5: train split, seed 1000 × network number + 0). Generation is sequential per state, so these are exactly the first 500 states of the full 10,000-state training split.
2. Solve them with OpenDSS under the §3 failure rule. Fit the Thevenin equivalent on **all OpenDSS-converged states among them** (converged by the default algorithm or rescued by Newton). Solver-indeterminate states are left out.
3. **Freeze that equivalent for E1, E2 and E3.** Generating the remaining training labels later does not trigger a refit.
4. **Report per network:** the number of fitted states, the design-matrix rank and its condition number.
5. **Rank rule:** if the design matrix is not of full rank (4), add successive frozen blocks of 500 training states, in the same order (states 501–1,000, then 1,001–1,500, then 1,501–2,000), up to 2,000 states, refitting after each block. This rule depends only on the fitting set, never on E1 results, and E1 results are not inspected before every network's equivalent is frozen. If a network is still rank-deficient at 2,000 states, it is reported and excluded from E1–E3.

**Accurate description of the fit** (the existing implementation, `fourwire_net.fit_thevenin`, kept unchanged for ENWL1 replication):
- Model: V_abc = E − Z_th · I_abc at `TRbus`. V_abc are the root phase voltages, I_abc the phase currents leaving the root (both from OpenDSS); kV, kA, ohm.
- Ordinary complex least squares, with design matrix [1, −I_a, −I_b, −I_c] (S × 4) and the three root phase voltages as responses. It estimates an **unconstrained** complex 3-vector E and an **unconstrained** complex 3 × 3 Z_th.
- No symmetry is imposed on Z_th and E is not fixed to nominal. ENWL1 amendment 2, item 1 ("E = 240.18 V nominal phasors with a symmetric Z_th") describes what that fit returned, not a constraint of the method.

**Consequence for network_1.** Its equivalent here comes from 500 training states, not from the full ENWL1 training split, so it can differ slightly from the ENWL1 study's. The ENWL1 study's frozen results are untouched.

## Amendment 2 (2026-11-15): transformer-based cap on the envelope augmentation (before any E1 label)
**What motivated it (disclosure).** The Amendment 1 identification run (code commit 17d3f9c) was run with the §2.5 augmentation unchanged. It is kept as an **uncapped pre-E1 design pilot**: `results/fourwire_scale_identification_pilot_uncapped.json` (label hashes, fit diagnostics, solver-indeterminate counts per network). It was not frozen.
- Among its first 500 training states, up to 115 per network were solver-indeterminate (network_13: 351), almost all with import augmentation. In the large networks 75–85% of import-augmented states failed: every customer importing up to 7.4 kW requests more than 1 MW through an 800 kVA transformer.
- Re-solving failing network_1 states at tolerance 1e-4 or with 1,000 iterations did not converge either (last iterates 0.3–0.7 pu), so these are loadings beyond the network's capability, not a tolerance artefact.
- Left in, they would make E2 partly measure the rejection of grossly infeasible synthetic requests rather than screening speed. This pilot motivated the global rule below. Only training states were seen; no val, calib or test state, and no E1 result, had been generated.

**Rule (replaces the augmentation in §2.5; input-only, global, prospective).**
1. Draw the state and its augmentation exactly as in §2.5 (same random streams, same order).
2. Let S(γ) = |Σ_i (P_i − γ·f) + j Σ_i Q_i| be the aggregate customer apparent power in kVA, summed over all customers i of the network (a three-phase customer counts once, with its total power), where f is the drawn per-customer augmentation (+ = export) and P_i, Q_i the unaugmented customer net consumption. Losses are not included.
3. Scale the augmentation by the largest γ ∈ [0, 1] with S(γ) ≤ 1.10 × 800 kVA = 880 kVA. The same rule applies to import and export. If the unaugmented state already exceeds 880 kVA, γ = 0. If the full augmentation fits, γ = 1 and the state is bit-identical to the uncapped draw.
4. γ depends only on the drawn inputs. No solver outcome, network-specific tuning or per-network cap is used.
5. Stored per state: the drawn f, γ and the applied γ·f. Reported per network and split: the share of augmented states with γ < 1 and with γ = 0.

**Consequences.**
- The cap still allows transformer overload (up to 10% before losses) and every cable and voltage violation; remaining solver-indeterminate states stay operationally unsafe and are reported as in §3.
- **No further change to the state distribution** after the capped identification run, whatever its indeterminate shares. All 25 identification runs are redone with the cap and their Thevenin equivalents frozen (Amendment 1), and only then is E1 run.

## Amendment 3 (2026-11-15): validation gate and timing rules for the optimised sweeps (before they are implemented)
Clarifies §3 ("Optimised sweep implementations") and the timing procedure of §6. Committed before any optimised sweep code exists and before any timing. It also records one E1 bookkeeping correction.

**E1 bookkeeping correction (already applied, commits 4413c71 and 9572669).** §7 compares every state where OpenDSS converges. The first E1 report compared only states where the sweep had also converged, which left out one network_7 state (it converges at iteration 108 and agrees with OpenDSS to 1.5e-5 pu at iteration 100). The report was rerun with the frozen mask: 24 of 25 networks pass, as before.

**Reference.** The NumPy sweep run one state at a time, flat start (`fourwire_scale.reference_sweep_state`), as in E1.
- A state is **converged** if the voltage update of its last performed iteration is below 1e-8 pu, within 100 iterations. The reference code tests `update < 1e-8`; every implementation uses the same strict test.
- **Per-state semantics:** in a batch, each state stops at its own convergence iteration and returns that iterate. A batched implementation must therefore return, for every state, what the one-state reference returns.

**Validation set.** The full val split of each of the 24 eligible networks (3,000 states per network, §2.5, with the Amendment 2 cap). Inputs only: no OpenDSS label is needed.

**Gate, for each implementation** (Numba CPU, PyTorch):
1. **On reference-converged states:** max |V − V_ref| ≤ 1e-9 pu of 240.18 V, over all retained buses and all four conductors (complex difference).
2. **On reference-nonconverged states:** the implementation must also classify the state as non-converged at 100 iterations. The final iterates of such states are unstable and are **not** gated.
3. **Reported per network and implementation:** convergence-mask differences (count, in each direction), iteration-count differences (count and largest), and the largest branch-current difference relative to the rating on reference-converged states.
4. **Devices:** the PyTorch implementation is validated locally on CPU (the same code path with `device="cpu"`). The validation is **repeated on the A100 CUDA path, on the same states, before any timing**. No device is timed unless its own validation has passed.

**Implementation notes (disclosed design, not a change of algorithm).**
- **CPU (Numba):** parallel over states. Within a state the buses are processed level by level (by tree depth) in compiled loops, which is what a vectorised depth-level operation compiles to.
- **GPU/CPU (PyTorch, complex128):** tensors batched over states; each tree depth is one tensor operation in the backward (current) and forward (voltage) pass. States that have converged leave the working batch.
- Both return, per state: node voltages, branch currents from the final voltages (as the reference), the iteration count and the convergence flag. They also compute the four screening quantities with the ENWL1 definitions (`fourwire_model.summaries`): minimum and maximum V_pn, maximum phase loading of the non-root branches, and the LV-root transformer-loading proxy.

**Timing rules (add to §6; identical for every timed method, the GENCO pipelines included).**
- **Excluded, one-off:** compilation (Numba JIT, CUDA kernel loading), the 3 warm-up repeats, moving network constants (topology, impedances, ratings, the Thevenin equivalent, model weights) to the device, and allocating reusable buffers.
- **Included, every batch:** every operation from the batch's inputs in host memory (customer P and Q) to its deliverable in host memory. That covers host-to-device transfer of the batch, the solve or the model with its screen, convergence checks, the final-current and screening-quantity computation, device-to-host transfer of the deliverable, and the fallback solves (§3, §5).
- **Deliverable, the same for every method:** per state, the four screening quantities, the decision per direction and the convergence flag. Full voltage and current tensors stay on the device and are not part of the timed transfer for any method.
- **CUDA:** `torch.cuda.synchronize()` immediately before the start and immediately before the end of every timed measurement.

## Amendment 4 (2026-11-15): label generation and the label-integrity gate (before the full label run)
Specifies how the labels of §2.5 are generated and what §9 gate 4 checks. Committed before the full label run. No state distribution, solver setting or criterion changes.

**Labels that exist at this point.** The 500 identification states per network (Amendment 1) and the 200 E1 val states per network. In addition, a smoke test of the label code, run from uncommitted source in a scratch folder and discarded: network_25 (1,500 train and 500 val states) and network_8 (500 train and 500 val states). No test or calib state has been labelled.

**Why a rule is needed.** OpenDSS starts each snapshot solve from the previous solution. A label therefore depends, at the level of the solver tolerance, on the states solved before it in the same engine.

**Generation (frozen).**
1. One unit of work per network and split, for the 24 eligible networks. States in the frozen generator order, with the Amendment 2 cap.
2. **Shards of 500 consecutive states. Every shard starts from a fresh OpenDSS engine** and solves its states in order, under the §3 failure rule. A shard then depends only on its own states, so an interrupted and resumed unit gives the same labels.
3. Consequence: the first training shard repeats the identification solves, and the first 200 states of the first val shard repeat the E1 solves. On the machine that produced those labels, both must come out bit-identical (checked below).
4. Stored per state, in FP64: the generator outputs (step, mix, drawn f, γ, applied γ·f, technology groups, meters, customer P and Q) and the OpenDSS outputs (node voltages, branch currents, the four summary quantities, the four violation flags, converged, retried, Newton-rescued).
5. Each shard is written atomically. Per network and split, a manifest records a content hash per shard (SHA-256 over array names, dtypes, shapes and bytes), the solver counts of §3, the meter-reuse report of §2.5, the cap report of Amendment 2, the code commit, the platform and the OpenDSS version.
6. This run is made on the laptop that produced the identification and E1 labels, so that item 3 can be checked. Label arrays stay outside git; the manifests and the integrity report are committed.

**§9 gate 4, per network and split.** The thresholds are fixed here and are not adjusted after the run.
- All shards present and contiguous, with the §2.5 state count.
- Every shard's recomputed content hash equals its manifest entry.
- One code commit, with unmodified source.
- The stored inputs equal the states regenerated from the seed, bit for bit, and their calendar days belong to the split.
- The ENWL1 checker (`fourwire_labels.semantic_check`), **with its ENWL1 thresholds unchanged**, on the converged states of every shard: KCL residual at every retained bus below 2e-4 kA; root neutral below 1 V; recomputed minimum and maximum V_pn within 1e-4 pu of the stored summary; recomputed worst cable loading within 1e-3; the four flags consistent with the summaries; all values finite.
- The identification states (train) and the E1 states (val) are bit-identical to the labels stored by those runs, where those files are on disk.
- **Reported, not gated:** the number of converged states with a customer voltage outside 0.5–1.5 pu, where OpenDSS no longer models the load as constant power (§3; the cause of the network_13 exclusion).
- A failed check is reported as it is.

**Test split.** Test labels are generated in this run and pass through the gate. The gate is a consistency check: it computes no model metric and no violation share. As §3 requires for every network and split, the counts of converged, Newton-rescued and solver-indeterminate test states are reported. Test shards live in their own directory tree (`labels/test/`), which training, debugging and calibration code never reads (§2.6).

## Amendment 4, correction (2026-11-15): voltage base of the reported constant-power diagnostic
Appended after the full label run and its first integrity report (commit f0fbbdd). The Amendment 4 text above is left as written.

**What was wrong.** Amendment 4 reports, without gating on it, the converged states "with a customer voltage outside 0.5–1.5 pu, where OpenDSS no longer models the load as constant power". The checker applied 0.5 and 1.5 to the study base of 240.18 V. OpenDSS applies `Vminpu` and `Vmaxpu` to each load's declared rating, and every load of the 25 networks declares `kV=0.23`. The load model therefore switches at 115 V and 345 V, which is 0.479 and 1.436 pu on the study base.

**First report (kept in the git history, f0fbbdd).** It counted 113 converged states: network_7 108, network_8 3, network_22 2. In the training states among them, the affected customers sit between 0.479 and 0.500 pu of the study base, and KCL with constant-power loads closes to below 1 mA. These states are inside OpenDSS's constant-power range.

**Corrected rule.** A converged state is counted when a customer's phase-to-neutral voltage is below 0.5 × 230 V or above 1.5 × 230 V. A voltage exactly at a limit counts as inside. The declared rating is covered by a unit test, and the network files are pinned by hash to the identification run.

**Scope.** Only this reported diagnostic changes. No label, no threshold and no gate check changes: the gate result of the first report (96 of 96 network-splits) does not depend on the diagnostic. `label-integrity` is rerun on the same labels (commit 6ad2af6). The corrected report replaces the first one, which stays in the git history.

## Amendment 5 (2026-11-15): pooled PG-pre training and the gate-5 smoke (before the prescribed smoke run)
Clarifies the multi-network details left implicit by §4 and fixes the §9 gate-5 run before it is submitted. It changes no data, model architecture, loss weight, optimiser or success criterion.

**Prior engineering check, disclosed.** Before this amendment, the real PG-pre code path was run locally on CPU for two epochs using the train and val labels of networks 8 and 25 only. Validation loss fell from 2.38 for the untrained U1 residual to 0.38 after the first epoch (0.43 after the second). This was a code-path check, not the prescribed gate-5 smoke; it used no calib or test label and set no threshold. No 18-network or A100 smoke result had been seen.

**Training-side isolation and provenance.**
- The training loader has no default label root: the caller supplies the commit-qualified root explicitly.
- It serves only `train`, `val` and `calib`; `test` raises before a test path is constructed, and there is no override. Shard names come from the committed manifest rather than a directory listing.
- Before returning states, it checks label commit 6ad2af6, clean provenance, the frozen state count, contiguous shards and each shard's content hash. Tests fail on any open or directory listing inside a decoy test tree while exercising both the loader and smoke trainer.

**PG-pre across networks.**
1. For a fold, training uses the first 5,000 train states of each of its 18 pool networks; validation uses all 3,000 val states of each. Solver-indeterminate states have zero loss weight, as in ENWL1.
2. The asinh/standardisation statistics for node inputs and target scales pool the pool's training data, with each state-bus row counting once. Edge statistics pool directed edge rows once per network. No val, calib or test value enters a scaler.
3. A batch contains 32 states of one network, because graph sizes differ. Every epoch visits every training state once; the per-network batches are shuffled together. Since every pool network contributes 5,000 states, networks contribute equal state counts.
4. Architecture, FP32 precision, TF32-off setting, loss, AdamW settings and cosine schedule remain the ENWL1/PG-net configuration of §4. PG-pre and PG-net allow at most 60 epochs with patience 8. `PG-scratch(B)` uses that PG-net stopping rule (60 epochs, patience 8); `PG-ft(B)` retains its distinct 30-epoch, patience-5 rule.

**Gate-5 smoke.** One seed-0 PG-pre run on fold 0 for three epochs on the A100, using train and val only. Fold 0 holds out networks 4, 6, 7, 16, 18 and 25; its pool is the other 18 eligible networks. Epoch 1 is warm-up for timing; epochs 2–3 provide per-network step and validation timings. The report records the clean code and label commits, folds, state and convergence counts, peak host/GPU memory, preparation time, epoch time, the loss curve and the requested §8 projection. The smoke is a feasibility/compute gate and has no accuracy threshold.

**Compute projection.** The primary projection is fit-only GPU time: measured training-step and validation-state time are fitted linearly against retained buses, never below the smallest measured time, then applied to every run allowed by §8 at its maximum epoch count and with no early stopping. It counts 12 PG-pre fits, 24 PG-net fits, 120 PG-scratch fits and 120 PG-ft fits. Data preparation, calibration and test evaluation are excluded from that number and disclosed separately; thus it is not presented as total allocation or end-to-end study time.

## Amendment 6 (2026-11-15): audit resolutions and the gate-5 record (before any final fit)
Records the findings of an independent read-only audit of the work through commit 20be7b1 and of both smoke reports, and how each was resolved. No final fit has been run. No test label has been read by training, calibration or evaluation code; the label generator and the integrity checker are the only code that has read them, and the checker computes no model metric and no violation share. The audit found no val, calib or test value entering training or a scaler.

**1. Gate-5 record: two smoke runs, both kept, neither selected.**
- Job 13586491 started at commit b377e92, eight minutes after Amendment 5 was committed (fddb921) and without it in its checkout. Four minutes into the run the Hydra worktree was moved to a0b5e19. Its report names a0b5e19 because the smoke read the commit at the end of the run; the job log names b377e92.
- Job 13586516 was pinned to a0b5e19, which contains Amendment 5.
- `src/` is identical at b377e92 and a0b5e19, so both jobs ran the same training code.
- Files, byte-identical to the job outputs: `results/fourwire_scale_smoke_pgpre_job13586491.json` (SHA-256 57d84510788c50d5d2fb163cf21eb0eaad2265521812bfc1d94990e19ee2d93d) and `results/fourwire_scale_smoke_pgpre_job13586516.json` (SHA-256 863ea2842dc26ee78c235e9cb73178f0e722d3bd0e96630b6da230540efbd9a6).
- **Gate-5 quantities (the two runs agree):** 111 s per epoch (epochs 2–3), 2,009 MiB peak GPU memory allocated, 15.5 GiB peak host memory, 142–143 s data preparation, and a fit-only projection of 34 GPU-hours for the 276 fits of §8.
- **The projection is fit-only and empirical,** not an upper bound: a straight-line timing fit (R² 0.85 for the training step) at the maximum epoch counts. Data preparation, the initial validation pass, checkpoint writes, calibration, test evaluation and timing runs come on top.
- **The two runs differ in validation loss** at the same seed, code, data and node: 0.615 / 0.373 / 0.218 against 0.838 / 0.557 / 0.293 after epochs 1–3. The cause is not established. Initial weights were not hashed, and the zero-initialised decoder makes the untrained loss identical whatever the hidden weights are. How this enters the final fits is fixed in a separate amendment, before any final fit.
- **The smoke is not the start of a final fit:** with three epochs the cosine schedule ran at 1e-3, 7.5e-4 and 2.5e-4.

**2. Corrections to code. No label and no gate threshold changes.**
- **Run provenance.** A run reads the commit before it starts, `GENCODOE_COMMIT` must equal HEAD, and the run fails if the checkout moved. Reports and run folders are named per job and never overwritten. Library versions, node, and the hashes of the initial and final weights are recorded. Final fits run from a checkout that is created for one commit and never moved.
- **Training loader.** The manifest and the label commit are fixed in code; the caller passes only the label root. A shard is read only if its manifest name is `shard_NNN.npz` and its resolved path equals its literal path, so a symbolic link or `..` cannot redirect a read into another split.
- **Integrity checker.** It took the label commit from the unit manifests. It now reads it from every shard file and requires one commit across all units. Rerun on the same labels: 96 of 96 network-splits pass, one label commit (6ad2af6) in all 936 shards; no other field changes.
- **Erratum, informational sweep timing.** In `results/fourwire_scale_sweep_validation_cpu.json` and `..._cuda.json` the field `numba.informational_ms_per_state` also contains the time of the comparison against the reference. It overstates the Numba sweep's time by a factor of about 3 (laptop, solve only: 0.144 ms per state on network_17 against 0.470 in the file; 0.004 against 0.014 on network_10). The validation gate is unaffected. The field is not a solve time and is not quoted as one. The files are left as produced; the code is corrected for later runs.

**3. Disclosures about the frozen design.**
- **Target labels in the physical prior.** Every network's Thevenin equivalent is fitted on OpenDSS root voltages and currents of its first 500 training states (Amendment 1). U1, the sweep, and the U1 input of every model therefore use target-network labels. That includes PG-zs and PG-pre-tc, and for B = 200 the identification uses more target states than the fitting budget. §4's "Fully zero-shot: no target label used" is wrong as worded: PG-zs uses no target label for its weights or its calibration, and nothing more is claimed. B still counts model-fitting states only. The paper states this dependence wherever it reports a budget or a zero-shot result.
- **Validation loss, inherited from ENWL1 and kept.** Per batch of 128 consecutive states the loss is the near-limit-weighted mean; the split's loss is the mean of those batch values weighted by their converged-state counts. It is fixed for given weights, but it is not the global weighted mean. Checkpoint selection and early stopping use it for every method.
- **Cosine schedule.** T_max is the fit's maximum epoch count (60, or 30 for PG-ft), as in ENWL1.
- **Target scales.** For a pool of one network they equal the ENWL1 formulas, except that the V_pn scale uses the population standard deviation in FP64 where ENWL1 used the sample standard deviation in FP32 (relative difference about 6e-6 on network_25).

## Amendment 6, corrections (2026-11-15): wording, evidence and limits found by the second audit round
Appended after a second read-only audit, of commits 42f61bc to cbb0a02 and of the Amendment 6 text. That text is left as written. No label, threshold or criterion changes.

**1. The opening statement of Amendment 6.**
- "No val, calib or test value entering training or a scaler" is too broad. Validation labels select the checkpoint and stop training, as §4 prescribes. Corrected: no validation, calibration or test label enters a gradient loss or a scaler.
- "The label generator and the integrity checker are the only code that has read them" is wrong. Test shards were also read by the transfer verification (file hashes) and by the independent auditor, who recomputed the constant-power diagnostic from test-label voltages and read every shard's provenance. None of these computes a model metric or a violation share. No training, calibration or evaluation code has read a test label.

**2. Gate-5 timeline: evidence now archived.** `results/fourwire_scale_hydra_job_evidence.txt` holds the Slurm accounting lines, the reflog of the Hydra worktree and the job logs, with account identifiers redacted.
- Job 13586491 was submitted with commit b377e92 and started at 18:32:27 on node401, 7 min 53 s after the Amendment 5 commit (18:24:34).
- The worktree moved to a0b5e19 at 18:35:53. That is 3 min 26 s into the run, not four minutes.
- Job 13586516 was submitted with commit a0b5e19 and ran from 18:41:53 to 18:50:27 on node401.
- Not established: that both jobs loaded the same library versions. The two smoke reports do not record them. Both jobs used the same environment folder.

**3. What the protections guarantee.** Amendment 6 states them too strongly.
- **"The run fails if the checkout moved."** A run compares HEAD, the modified flag and a hash of `src/gencodoe/*.py` at its start and at its end, and marks its record invalid if they differ (from commit ee2479f). A checkout that is changed and restored during a run is not detected. That a final fit runs from a checkout that is never moved is an operating rule, not something the code enforces.
- **"Never overwritten."** A run refuses an existing run folder or report file. This guards against reuse by our own code. It does not stop another process from writing there.
- **Loader.** A symbolic link, a hard link or `..` below the label root is refused, and the manifest is pinned by its SHA-256 (ee2479f). The label root itself is trusted as given, and the check and the read are two steps.
- **Integrity report.** The report of commit b23b286 was stamped with the checker's commit at the end of its run. From ee2479f the checker reads its commit at the start and compares at the end.

**4. Two quantities.**
- **Thevenin fitting sets (Amendment 1).** 500 candidate states are solved per network and the fit uses the converged ones: 385 to 500 states, fewer than 500 in 7 of the 24 eligible networks (network_8 385, network_5 437, network_7 443, network_2 482, network_12 and network_17 492, network_1 497).
- **Sweep timing erratum.** The solve-only values quoted in Amendment 6 (0.144 and 0.004 ms per state) are a laptop re-measurement that was not archived. "A factor of about 3" is therefore not an established correction, on the laptop or on the A100 node. What stands: the field includes the comparison with the reference and is not a solve time. Solve times come from E2 only.

**5. Determinism probe, first run (job 13587453, commit cbb0a02, node402).** The file is kept as produced: `results/fourwire_scale_determinism_job13587453.json` (SHA-256 b5a878611c6cb380447e6da3213ddba4f3de4521793ce56437d4080fcad2b305).
- Two processes with default settings start from identical weights and differ from the third training step (largest step-loss difference 6.7e-5).
- Two processes with PyTorch's deterministic algorithms are bit-identical over 32 steps.
- **Limits of this run:** it recorded the requested setting, not the setting in effect, and its summary would also have given a verdict on an invalid run. It is kept as a record. The probe is repeated with corrected code (ee2479f), and the rule for the final fits is fixed in Amendment 7.

## Amendment 7 (2026-11-15): deterministic final fits and what counts as a fit's record (before any final fit)
**Why.** Two smoke runs with the same seed, code and data gave different losses (Amendment 6). The first determinism probe (Amendment 6, corrections, item 5) shows that PyTorch's default GPU mode is not reproducible between processes, and that its deterministic mode was bit-reproducible over 32 steps on one A100 node.

**Rule.**
1. **Numerical mode and software.** Every final fit runs on a Hydra A100-PCIE-40GB with `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN benchmark off and TF32 off, with the library versions recorded in the freeze file (Amendment 8, item 14). The settings are read at the start of the run and again at its end. A run with other versions is refused; a record whose settings were not in effect or changed is invalid.
2. **One fit.** One final fit per (method, fold or network, budget, seed), as §8. **The first attempt that completes with a valid record is the fit.** A fit with a valid record is never run again, and no choice is ever made between two completed runs.
3. **Attempts.** Every attempt has its own run folder, with the job that owns it and the loss curve written epoch by epoch. No folder is deleted.
   - A fit may be attempted again only if the earlier attempt left no valid record and one of these holds: (a) Slurm accounting shows that its job ended as FAILED, NODE_FAIL, TIMEOUT, OUT_OF_MEMORY or PREEMPTED, or was cancelled by the system or an administrator; (b) its record is invalid for a reason that does not depend on the fit's outcome: the checkout or the numerical mode changed during the run.
   - **We do not cancel a running final fit.** If one is cancelled by us, that is recorded as a deviation with its reason.
   - While an attempt is running, a second attempt of the same fit is refused.
   - **A rerun must retrace the earlier attempt.** The fit record compares every epoch the earlier attempt recorded (training and validation loss) with the rerun. With deterministic algorithms they should be identical; a difference is stated in the record and in the paper. A rerun therefore cannot give another outcome unnoticed.
   - At most two reruns per fit. A third failure, an operation without a deterministic implementation, or an out-of-memory error that repeats is a defect of the code or the setup. The final runs then stop until a dated amendment (Amendment 8, item 14).
   - The accounting lines of every job of the final runs are committed, and the record of a rerun names the earlier attempts with their jobs and how Slurm says they ended.
4. **Terminal outcomes, never retried:** a non-finite gradient; a target scale that is zero or not finite; no epoch with a finite validation loss. The fit ends there. Its checkpoint is the one selected among its completed epochs (Amendment 8, item 5). If there is none, the method has no model for that case and defers on every state. If a PG-pre fit has no model, neither have the PG-ft fits that start from it, nor PG-pre-tc and PG-zs of that fold. These are valid records.
5. **Valid record:** `GENCODOE_COMMIT` equals HEAD at the start; the source is unmodified and its hash equals the frozen one at the start and at the end; the label manifest has its pinned hash; the library versions are the frozen ones; the numerical mode of item 1 is in effect at the start and at the end.

**What this does and does not establish.**
- The probe is a falsification test: 32 steps, two networks, one node type, one software stack. It is not a certificate that a 60-epoch fit is bit-reproducible, and no criterion needs that.
- Uncertainty enters the primary E3 criterion as frozen in §7: three seeds at B = 1,000 and the bootstrap over networks. Run-to-run variation at a fixed seed is not estimated. A rerun that retraces an interrupted attempt is evidence about it, where one happens.
- The probe is being repeated with corrected code (job 13587703, commit ee2479f). Its report is committed when it finishes, whatever it shows. The rule above does not depend on it: if deterministic mode turns out not to be reproducible, every fit is still run once, the first valid attempt is the record, and the paper says that the fits are not bit-reproducible.
- Deterministic algorithms are slower. The fit-only projection of 34 GPU-hours (Amendment 6) was measured in default mode, so the final fits will need more. The probe is too short to say how much.
- The two smoke runs stay in the record. Neither is a final fit, and neither is used to start one.

## Amendment 8 (2026-11-15): final fits, the screen, test evaluation and the freeze (before any final fit)
Fixes what §4 to §8 and Amendment 5 leave open. No data, architecture, loss weight or success criterion changes.

**What was seen before this amendment (disclosure).**
- The two gate-5 smoke runs and the first determinism probe: train and val labels of pool networks only. The val-only comparison of the sweep with OpenDSS described in Amendment 9.
- **Between the first draft of Amendments 7 to 9 and their commit:** an independent read-only audit of the draft, which led to the attempt rule of Amendment 7, to items 8, 9 and 14 below and to several items of Amendment 9; and dry runs of the code with train and val labels only. One ran on the laptop (network 25: fits of 1 to 3 epochs on at most 500 states, a screen calibrated on the first 1,500 val states and applied to the other 1,500, the timing harness, two envelope hours on a val day). Two of the same kind were submitted to the A100 (networks 17 and 7). They check that the code runs; no number from them decides anything in this text.
- No PG-net, PG-scratch or PG-ft fit beyond those dry runs has been run. No calib or test label has been read by model code.

**1. One training routine.** Every fit uses the PG-pre routine of Amendment 5 (`fourwire_scale_train.fit`): the same model, loss, loss weights, near-limit weight, batches of 32 states of one network, AdamW with weight decay 1e-4, a cosine schedule over the fit's maximum number of epochs, and the validation loss described in Amendment 6. PG-net, PG-scratch and PG-ft are that routine with a pool of one network.

| Fit | Training states | Validation states | lr | Max epochs | Patience | Initial weights | Scalers and target scales |
|---|---|---|---|---|---|---|---|
| PG-pre (fold, seed) | first 5,000 train states of each of the 18 pool networks | all val states of the pool | 1e-3 | 60 | 8 | random | pool training states (Amendment 5) |
| PG-net (network) | all 10,000 train states | the network's 3,000 | 1e-3 | 60 | 8 | random, seed 0 | its 10,000 train states |
| PG-scratch(B, seed) | first B train states | the network's 3,000 | 1e-3 | 60 | 8 | random | its B states only |
| PG-ft(B, seed) | first B train states | the network's 3,000 | 3e-4 | 30 | 5 | the selected checkpoint of PG-pre, fold of the target, same seed | PG-pre's, unchanged |

**2. Budget and scaling boundaries.**
- **B counts states in generation order, converged or not.** Solver-indeterminate states have loss weight 0 and enter no target scale. Their inputs do enter the input scaler, as in ENWL1. The number of converged states is reported for every fit.
- **PG-scratch(B):** every fitted statistic comes from its B states only: the input scaler, the residual scale, the V_pn scale and the voltage-drop scale. The inherited rule "first 2,000 converged train states" for the last two becomes "the converged states among the B" when there are fewer. The edge scaler uses the target's branch data (impedance, rating, length), which are not labels.
- **PG-ft(B), PG-pre-tc and PG-zs:** every scaler and scale is PG-pre's, unchanged, the edge scaler included. No statistic of the target network enters.
- **PG-net:** its 10,000 train states, with the inherited rules.
- **What the primary comparison compares.** PG-ft keeps PG-pre's input scaling and loss scales; PG-scratch estimates its own from its B states. With the same loss weights, the two objectives are therefore scaled differently. The criterion compares the two procedures as specified here, not pretrained weights in isolation, and the paper says so.
- As disclosed in Amendment 6, every network's Thevenin equivalent uses the converged states among its first 500 train states, for every method and budget.

**3. Seeds.** The seed sets the weight initialisation (except PG-ft, which starts from PG-pre) and the batch order: `torch.manual_seed(seed)`, `numpy.random.seed(seed)` and batch order from `numpy.random.default_rng(seed)`.

**4. Fine-tuning.** All parameters are trainable. A new AdamW (lr 3e-4, weight decay 1e-4) and a new cosine schedule over 30 epochs. No optimiser state is carried over from PG-pre.

**5. Checkpoint selection, every fit.** The validation loss is evaluated before training (epoch 0) and after every epoch.
- The selected checkpoint is the epoch ≥ 1 with the lowest validation loss; on a tie, the earliest. An epoch whose validation loss is not finite is never selected.
- Training stops after `patience` consecutive epochs without a new lowest value.
- **Epoch 0 is recorded and not eligible.** It is the U1 prior for a fit from random weights, and PG-pre itself for PG-ft. This is the ENWL1 rule, unchanged. It cannot favour PG-ft: where PG-pre is already the best model on the target's val states, PG-ft is selected with a higher validation loss than its starting point.
- That starting point is in every PG-ft record (its epoch-0 validation loss). On the test states it is evaluated for seed 0 only, as PG-pre-tc.

**6. Failures and reruns:** Amendment 7, items 2 to 5.

**7. Fit records (the run manifest of §8).** Per fit: commit, source hash, label manifest hash, the hash of the identification record that holds the Thevenin equivalents and the network-file hashes, configuration, seed, state and converged counts, scalers, loss curve, selected epoch, SHA-256 of the checkpoint file, for PG-ft the PG-pre record and checkpoint it started from, the earlier attempts and whether the fit retraced them, library versions, node, GPU, and the numerical mode in effect. Checkpoints stay outside git; the records are committed.

**8. The screen.**
- **Implementation:** `fourwire_eval.GlobalMargins` of the ENWL1 study, unchanged, α = 0.01, fitted on the converged states of the calibration set. The hash of that file is in every evaluation record.
- **Predicted quantities:** from the predicted node voltages, branch currents by KCL with the state's customer powers, then `fourwire_model.summaries`: minimum and maximum V_pn, maximum phase loading of the non-root branches, and the LV-root loading proxy. U1 is screened the same way, from its own voltages.
- **Calibration truth:** the OpenDSS label summary (minimum and maximum V_pn, worst cable loading, transformer loading).
- **Calibration set:** the target's 2,000 calib states for U1, PG-net, PG-scratch, PG-ft and PG-pre-tc. For PG-zs, the calib states of the 18 pool networks together (36,000 states, each once), giving one set of margins. PG-zs thus has 18 times more calibration states, all from other networks. Whether margins from other networks hold on the target is part of what PG-zs tests; its result is read with that in mind.
- A state is screened in both directions, whatever its own injection (as in ENWL1).
- **Unusable calibration.** If one or more predicted quantities are not finite on a converged calibration state, the method has no screen and defers on every state. A margin that is infinite because there are fewer than 399 converged calibration states is kept: that criterion never clears.
- At test time a state with a non-finite predicted quantity defers (inherited).

**9. Decision metrics, per method, network and direction.** Truth is the OpenDSS label. A state is *safe in a direction* if it converged and has no cable overload, no transformer overload and no undervoltage (import) or overvoltage (export).
- **False-safe:** cleared by the screen, converged, not safe in that direction.
- **Cleared-indeterminate:** cleared by the screen, solver-indeterminate.
- **False-unsafe:** rejected by the screen, safe in that direction.
- **Fallback share:** share of the 4,096 test states that the screen sends to the fallback, mean of the two directions.
- **Rates and bounds are per direction.** The false-safe rate is the count over the converged states that are not safe in that direction, with its one-sided 95% Clopper–Pearson upper bound. Counts summed over the two directions are given as counts only: the two directions share the state, so their pairs are not independent trials.
- **These metrics concern screen outcomes,** as §5 and §6 define them. What the whole pipeline delivers, including the decisions of its exact fallback, is assessed in E2 (Amendment 9, item 6).

**10. State accuracy,** per method and network, on the converged test states with a finite prediction.
- The number of converged states and the number of non-finite predictions among them are reported with every metric. If no converged state has a finite prediction, the metrics are missing, not zero.
- **V_pn:** | |V_pn| predicted − |V_pn| OpenDSS | in pu of 240.18 V, over states and customer connections: MAE, p95, p99, maximum.
- **Complex voltage:** |V predicted − V OpenDSS| / 240.18 V over states, retained buses and the four conductors: MAE, maximum.
- **Neutral voltage:** | |V_n| predicted − |V_n| OpenDSS | / 240.18 V over states and retained buses: MAE, maximum.
- **Phase and neutral currents:** | |I| predicted − |I| OpenDSS | divided by the branch rating, non-root branches: MAE and maximum, separately for the phases and the neutral.
- **Transformer loading:** | LV-root proxy of the predicted state − OpenDSS transformer loading |: MAE, maximum. The OpenDSS label is the authority. This differs on purpose from the ENWL1 evaluator, which compared the model's proxy with the proxy of the OpenDSS state. The proxy of the OpenDSS state is reported next to it: it shows how much of the error is the proxy itself (Amendment 9, disclosure).
- **Aggregation:** per network. Means over networks in E3 are unweighted means of the per-network values.

**11. Rating sensitivity (§2.4).** For each of the two variants, every model is re-evaluated on the test states with the variant's ratings as edge inputs and in the loading quantity. Margins stay those of the primary calibration. The truth's cable-overload flag is recomputed from the label currents with the variant's ratings. The decision metrics are re-reported; no criterion uses them.

**12. E3 statistics:** as §7. The paired bootstrap resamples the 23 fresh networks with replacement (2,000 replicates, seed 0, percentile interval) on the per-network differences.

**13. Numerical mode of inference.** Calibration, test evaluation and the envelope check use one device path: U1 and the features computed on the device, the model in FP32 with TF32 off, microbatches of 256 states, deterministic algorithms on. Timing uses the same path in PyTorch's default mode (Amendment 9, item 4).

**14. Freeze and test isolation.**
- **Freeze.** The code and inputs of fits, calibration, evaluation, timing and envelopes are frozen by committing `results/fourwire_scale_freeze.json`. It is written on Hydra, from committed source, and holds: the SHA-256 over `src/gencodoe/*.py`; the hashes of the label manifest, of the identification record (Thevenin equivalents and network files), of the rating-variant network files, of the meter profiles (content hash) and of the job script; the library versions; and the folds. Every final command refuses to run if the source is modified, if it is not pinned to HEAD, or if any of these that it uses differs (the rating-variant files are used by the evaluation, the meter profiles by the envelope check).
- **Order:** freeze; final fits; evaluation, once per target network, after all of that network's fits have valid records; timing and envelopes for the E2 networks; reports.
- **Test labels** are read only by the final `evaluate` command of `fourwire_scale_eval`. It refuses if an evaluation record of that network exists, if anything differs from the freeze, or if a fit record of that network is missing, invalid or made with other source. The training module cannot read them (the Amendment 5 loader, with tests).
- **No selective release.** The evaluation of the 24 networks is submitted as one job array. A final evaluation prints no result, and no evaluation record is opened before all 24 exist.
- **Dry runs before the freeze** use train and val only: short fits flagged `dry_run`; the first half of a network's val states stands in for its calibration set and the second half for its test set. A dry-run model is never evaluated on test.
- **Compute.** A fit job has one A100 and 8 cores. Several fits of one network run one after another in one job, each in its own process.
- **A defect found after the freeze** is fixed under a dated amendment that states the defect, what had been seen, and the change. **What is redone follows from what the change touches, never from a result:**
  - a change to the fitting code or to an input of the fits: all 276 fits, and everything after them;
  - a change to the calibration or evaluation code: the evaluation of all 24 networks, and everything after it;
  - a change to the timing or envelope code: those runs, for all six networks;
  - a change to a report: the report.
  - Never a subset of networks, seeds or methods. Every earlier output is kept with its freeze and reported next to the new one.

## Amendment 9 (2026-11-15): E2 execution and the envelope check (before any timing)
Fixes what §5 to §7 and Amendment 3 leave open for E2, and lists every departure from them. The speed criterion of §7 keeps its thresholds; its safety condition becomes stricter (item 6).

**What was seen before this amendment (disclosure).**
- **Informational timings, none of them E2:** the validation pass of the gate-5 smoke (0.07 to 0.63 ms per state for networks of 94 to 1,668 buses, loss included), the sweep times of the validation runs (the contaminated field of Amendment 6), OpenDSS label times on the laptop (17 to 270 ms per state), and the laptop dry run of Amendment 8's disclosure (network 25 on a CPU).
- **A hypothesis from them, stated before any timing:** the PG-net model alone is slower per state than the optimised sweep on every network, and where a sweep has non-converged states, the OpenDSS solves that §3 prescribes for them dominate its end-to-end time. If so, a speed-up for GENCO would come from that rule and not from faster power-flow prediction. The sensitivity series of item 6 is there to test this, whatever the outcome.
- **Val-only comparison of the Numba sweep with the OpenDSS labels** (72,000 val states, 24 networks; `scripts/fourwire_scale_val_sweep_vs_opendss.py`, `results/fourwire_scale_val_sweep_vs_opendss.json`):
  - **Convergence.** The sweep does not converge on 1,577 states and OpenDSS is indeterminate on 1,547, all of them among the 1,577. On the other 30 the sweep fails and OpenDSS converges; OpenDSS finds all 30 unsafe in both directions. On no state does the sweep converge while OpenDSS is indeterminate.
  - **Flags, where both converge.** The voltage and cable flags agree on every state. The transformer flag differs on 73 states (0 to 7 per network): on 58 the sweep's is clear and OpenDSS's is set, on 15 the reverse. OpenDSS's transformer loading is the apparent power at the transformer's HV terminal; the sweep's is the LV-root proxy, typically 0.5 to 1% lower above 80% loading.
  - **Complete decisions** differ far less, because those states usually break another limit too. The sweep declares no state safe in a direction where OpenDSS does not (0 of 144,000 state-directions). It declares 2 import decisions unsafe that OpenDSS calls safe. Sending the 30 non-converged states to OpenDSS changed no decision.
  - These are capped val states. They say nothing about the uncapped states of an envelope search.

**1. Allocation.** One job per network: one A100-PCIE-40GB and 16 cores of the same node (the count is recorded), with 16 Numba threads, 16 PyTorch CPU threads and 16 OpenDSS processes. The node is shared with other jobs.

**2. Inputs and batches.** The per-connection P and Q (FP64) of the network's test states in generation order, in host memory, regenerated from the seeds and checked against the input hash in the label manifest. The timing code reads no label. Repeat r = 0, 1, 2, … of batch size B (warm-ups first) uses the B states starting at (r · B) mod 4,096, the same for every series. (A choice: §6 only says "in a fixed order". At batch 4,096 every repeat is the whole test split.)

**3. Timed series, per network and batch size.** The deliverable is that of Amendment 3, in host memory: per state the four quantities, the decision per direction and the convergence flag.
- **Decision of an exact engine:** safe in a direction if the state converged and its quantities are within the limits of that direction. A state that is not converged after the §3 rule is unsafe in both directions.
- **Convergence flag of a state decided by the screen:** no solver runs for it, so the flag is delivered as true. Whether each state was decided by the screen or by an exact solve is stored with the decisions.
- **(a) OpenDSS.** 16 worker processes, each with one engine that is built and primed with a no-load solve before timing and kept for the whole run. A batch is cut into at most 16 contiguous chunks; each goes to a free worker.
  - **Before each chunk the engine solves the no-load case,** inside the timed region. A chunk's answers and time then do not depend on what that worker solved before.
  - Within a chunk the states are solved in order under the §3 failure rule, each from the previous solution, as OpenDSS does and as in label generation. This is OpenDSS's own behaviour and an exception to "no warm start for any method" in §3, which concerns the sweeps and GENCO.
- **(b) NumPy sweep.** One state at a time, as the Amendment 3 reference, non-converged states to OpenDSS. **Timed at batch 1 only** (3 warm-ups, 100 repeats): one state takes of the order of a second on the large networks. Only its batch-1 latency is reported; nothing is derived from it for larger batches. A departure from the batch sizes of §6, for a series that enters no criterion.
- **(c) CPU sweep, (d) GPU sweep.** `NumbaSweep` and `TorchSweep` (FP64) on the whole batch. States not converged within 100 iterations go to the OpenDSS workers of (a); OpenDSS's quantities and decisions replace the sweep's.
- **(e) U1 screen + fallback, (f) PG-net screen + fallback,** each timed twice: with (c) and with (d) as its fallback engine.
- **OpenDSS at batch 4,096** is timed with 1 warm-up and 10 repeats, not 3 and 100: one repeat takes of the order of a minute on the large networks. Its p95 is therefore less precise than that of the other series. The series enters no criterion.
- **Microbatches.** §6 allows them for any method. The model uses 256 states, the ENWL1 value. The sweeps use none: the Numba sweep has nothing to split, and 4,096 states of the largest network fit in GPU memory. Neither choice was tuned for speed.

**4. The screen pipelines (e) and (f), per batch.**
1. The batch goes to the device.
2. U1: one sweep iteration from the flat start, FP64, on the device.
3. Node features and scaling, on the device. 4. PG-net in FP32, microbatches of 256 states. (Series (e) skips 3 and 4.)
5. KCL currents and the four quantities.
6. The screen, in both directions, with the margins of the evaluation record.
7. A state that is uncertain in at least one direction is solved once by the exact engine, with its OpenDSS rule. The exact quantities and decisions replace the screen's for both directions of that state, also in a direction where the screen had decided. Item 6 accounts for this.
8. The deliverable goes to host memory.
- Weights, margins, network constants, compiled code and the OpenDSS engines are prepared before timing. Nothing computed in one repeat or series is reused in another; the OpenDSS engines are reset before every chunk (item 3a).
- **Numerical mode:** PyTorch's default algorithms for every GPU series, GENCO and sweep alike, because deterministic algorithms would slow both. TF32 off, no `torch.compile`. **This is not the deterministic path on which the screen was calibrated and evaluated.** Item 6 therefore assesses the decisions that the timed run itself delivered.

**5. Order and repeats.** For each batch size, the series run round-robin: one repeat of each per round; 3 warm-up rounds, then 100 timed rounds.
- **Order within a round:** CPU sweep, GPU sweep, U1 with CPU fallback, U1 with GPU fallback, PG-net with CPU fallback, PG-net with GPU fallback, then the same six without OpenDSS solves (item 6). OpenDSS and the NumPy sweep come first where they are in the rounds.
- **The starting position advances by one every round,** so that no series always runs after the same one.
- Exceptions: (b), and (a) at batch 4,096, as in item 3. `torch.cuda.synchronize()` as in Amendment 3.

**6. Statistics and what is reported with them.**
- **Speed baseline,** per network and batch size: whichever of (c) and (d) has the lower median. **PG-net pipeline time:** whichever of its two fallback variants has the lower median. The speed-up is the ratio of those two medians.
- **Interval (the criterion, §7):** each bootstrap replicate resamples the 100 repeats of each of the four series independently and recomputes both minima and the ratio (2,000 replicates, seed 0, percentile). **Sensitivity:** the same with whole rounds resampled, which keeps what the series of one round share. Both are reported.
- **Safety condition of the §7 criterion, stricter than §7.** Both must hold on the network's test states:
  - the screen of PG-net has no false-safe and no cleared-indeterminate decision (evaluation record; Amendment 8, items 8 and 9);
  - **what the faster timed PG-net variant delivered at batch 4,096** has none either: every delivered decision, from the screen or from the exact fallback, against the OpenDSS labels.
- **Reported next to every speed-up, outside the criterion:**
  - **Sensitivity series.** (c), (d), (e) and (f) again, with the non-converged sweep states declared unsafe without an OpenDSS solve; the speed-up computed from them; and their delivered decisions against OpenDSS.
  - **Delivered decisions of every timed series** against the OpenDSS labels, per direction: false-safe, cleared-indeterminate, false-unsafe. This includes the sweep baselines.
  - The number of timed decisions that differ from the evaluation record's deterministic path, and from repeat to repeat.
  - The log–log exponents and the speed-ups at batch 1 and 256 (§7).
- **Claim wording.**
  - "Zero observed false-safe decisions" about a pipeline refers to its delivered decisions. The screen's own metric is reported separately.
  - If GENCO is faster with the §3 rule and not in the sensitivity series, the paper says "faster than the sweep pipeline with mandatory OpenDSS solves for its non-converged states", gives the comparison without those solves and its decision differences in the same place, and does not attribute the gain to faster power-flow prediction or to solvers in general.

**7. Envelope check (§6), E2 networks.**
- **Days:** the 36 test days in calendar order, every third one starting with the first: 2024-01-10, 02-09, 03-10, 04-09, 05-09, 06-08, 07-08, 08-07, 09-06, 10-06, 11-05 and 12-05. Their day indices, counted from 0 on 1 January, are 9, 39, …, 339.
- **Intervals:** the first 15-minute step of each hour, 24 per day, both directions: 576 envelopes per network and evaluator, 288 per direction.
- **Design scenarios** (the ENWL1 day rule of `doe_day.DayScenarios`, with the meter-shortage rule of §2.5):
  - Each customer gets its technology flags once per network and day, with the ENWL1 DOE shares (heat pump 0.3, EV 0.3, PV 0.4): one draw of `uniform(size=(customers, 3))` from `numpy.random.default_rng([network, day index, 0])`, customers in `Loads.dss` order.
  - One generator `numpy.random.default_rng([network, day index, 1])` serves the day: hours in order; within an hour the 200 scenarios in order; within a scenario the technology groups in ascending order of their (heat pump, EV, PV) flags. For each group, one draw of meters for all its customers at once (without replacement while the group has enough meters, §2.5 otherwise), read at the interval's step. Offtake at PF 0.95, injection at unity power factor.
  - The same scenarios serve both directions and both evaluators. The Amendment 2 cap does not apply: the search sets the injection.
- **Envelope:** `envelopes.scalar_envelope`, unchanged. Uniform policy: one level L per customer in [0, cap], cap 5.0 kW for export and 7.4 kW for import, added to every customer at unity power factor. A three-phase customer's L is split equally over its three connections. ε = β = 0.05, so a level is accepted if at least 98% of the 200 scenarios are feasible; 4 levels per round; relative tolerance 1e-3.
  - **Its own order of checks is kept:** if the cap is accepted, the envelope is the cap; otherwise, if fewer than 98% are feasible at L = 0, the envelope is 0; otherwise the search.
  - The search assumes that feasibility does not come back once lost as L grows (as in ENWL1). A screen need not behave that way; the comparison of B with A shows where it matters.
- **Feasible in a direction:** the exact-engine decision of item 3.
- **Evaluator A:** the Numba sweep. **Evaluator B:** the PG-net screen, then the Numba sweep for the states the screen sends to the fallback in the envelope's direction.
- **Departure from §3, for both evaluators:** a state on which the sweep does not converge counts as infeasible, without an OpenDSS solve. At high levels most states do not converge, and the check needs millions of states. **The check is therefore relative to the sweep. It is not evidence about envelopes judged by OpenDSS.** On the capped val states the same rule changed no decision; the search's states are not capped.
- The check reads no label. Its reference is evaluator A.
- **Agreement criterion (§6):** |L_B − L_A| ≤ 1e-3 × cap in at least 95% of the network's 576 envelopes, the two directions together. The share is also reported per direction.
- **Also reported per network and direction:** the number of unsafe-envelope intervals (L_B > L_A + 1e-3 × cap), the largest differences, the number of zero envelopes, and the share of evaluator B's states sent to the fallback.

## Amendments 7 to 9: a disclosure and corrections (2026-11-15, before the freeze)
Appended after a fourth read-only audit, of the committed Amendments 7 to 9 (commit 2356f51) and of the code written for them (commits babf88a to 73442a2). The amendments above are left as written. Nothing had been frozen and no final fit, final evaluation or timing had been run.

**A. Disclosure: unit tests evaluated 12 test states.**
- **What happened.** The fixture of `tests/test_fourwire_scale_eval.py` (commits 83c3454 to 73442a2) generated labels, in a temporary folder, for the first 12 states of network_25's test split (states 0 to 11 of its 4,096; generator seed 25003) and for the first 12 calib states of networks 10 and 25. The tests then ran the evaluation code on them, with the U1 prior and with throwaway models trained for one or two epochs on 24 train states.
- **How often.** About ten runs of that test file on the laptop on 2026-11-15, before the audit found it. The auditor did not run those tests.
- **What was computed and seen.** With 12 calibration states no margin is finite, so every method deferred every state. The numbers that were displayed are a fallback share of 1.0 and zero false-safe and cleared-indeterminate decisions for every method. Two assertions also passed: that a V_pn error is not negative, and that PG-net's differs from U1's. No accuracy value was displayed, kept or used. The records went to pytest's temporary folders.
- **What it changes.**
  - The sentence "No calib or test label has been read by model code" in Amendment 8's disclosure was **false when it was committed**. Model code had read 12 test states of network_25 and 12 calib states of networks 10 and 25.
  - Those 12 test states are 0.3% of network_25's test split and 0.012% of all test states. **They stay in the test set.** Nothing is dropped or replaced.
  - §8 limits debugging to train and val. Using 12 calib states in a unit test of the calibration code broke that as well.
- **Earlier fixture.** Since commit b377e92 the fixture of `tests/test_fourwire_scale_train.py` generated the same 12 calib states of networks 10 and 25 to test the loader. No model or screen used them.
- **What was done.** In both fixtures, what plays calib and test is now drawn from val days with other seeds. `tests/conftest.py` makes the state generator refuse the calib and test splits in every unit test, and a test checks that.

**B. What had been seen by the time of these corrections,** beyond Amendment 8's disclosure: a second local dry run, on network 1 with train and val labels only (fits of 1 to 3 epochs on at most 1,000 states; calibration on the first 1,500 val states, evaluation on the other 1,500; timing on a laptop CPU).
- Fallback shares of about 0.35 to 0.40 for every method, U1 included. With such fits these say nothing about the methods.
- Timing of 3,000 val states: the Numba sweep took 0.07 s without and 2.6 s with the OpenDSS solves of its 55 non-converged states. The U1 screen pipeline took 0.17 s and sent no state to OpenDSS. PG-net ran on the CPU there (5.7 s), which says nothing about the A100.
- This agrees with the hypothesis written in Amendment 9. **It is why item D8 below was added.**

**C. Corrections to Amendment 7.**
1. **Reruns are authorised by the code, before the earlier attempt is set aside** (item 3). Slurm accounting must report FAILED, NODE_FAIL, TIMEOUT, OUT_OF_MEMORY, PREEMPTED or BOOT_FAIL, or a cancellation that is not by our account. A cancellation by our account needs a declared deviation, whose text goes into the record. If Slurm gives no answer, nothing is rerun. A run folder without an owner is refused.
2. **Halt.** An error raised inside a final run, after it has claimed its folder, writes a halt file. Every later final command refuses to start until the halt is resolved under an amendment. This is how "the final runs then stop" (item 3) is enforced.
3. **The same attempt rule covers evaluation, timing and envelope runs.** Each claims a folder; a network has one valid run of each; an attempt that died is set aside and named in the record of the next one.
4. **Device.** A final command refuses any device other than an "NVIDIA A100-PCIE-40GB", the CPU included.
5. Library versions and input hashes are read at the start of a run and compared at its end, like the checkout and the numerical mode.

**D. Corrections to Amendments 8 and 9.**
1. **Epoch 0 (Amendment 8, item 5).** "It cannot favour PG-ft" is withdrawn. The rule is the same for every fit. Its effect need not be: it costs PG-ft where PG-pre is already the best model, and it costs PG-scratch where the U1 prior is. No neutrality is claimed.
2. **Freeze lineage (Amendment 8, item 14).** Every final record carries the hash of the freeze file. A record is used only under that freeze, or under a later freeze that lists it as accepted for that stage. A change to the evaluation code is thus made with a new freeze that accepts the earlier freeze's fits; a change to the fitting code accepts nothing. A freeze file is never replaced in place, and it must be committed before a final command runs.
3. **The test reader (Amendment 8, item 14).** It serves one network, to the process that holds the claim on that network's evaluation, under the freeze, until the record is written. Timing and envelope runs start only when all 24 networks have a valid evaluation record.
4. **Records are checked when they are used:** which fit they are (method, network, budget, seed, fold), validity, freeze, inputs. Per-state files are checked against the hash in their record. The reports refuse anything else.
5. **Missing metrics (Amendment 8, item 10)** stay missing in the reports.
6. **Delivered decisions (Amendment 9, item 6) are scored over every timed repeat** at batch 4,096. A state counts as cleared if any repeat cleared it, and as rejected if any repeat rejected it.
7. **What the timed region contains (Amendment 9, items 2 to 4).**
   - The batch input is one complex array S = P + jQ per connection (FP64 parts) in host memory. Packing P and Q into it is outside the timed region, for every series alike.
   - The screen runs on the host: the four quantities come back from the device and the ENWL1 `GlobalMargins.screen` is applied to them there, inside the timed region.
   - The GPU sweep brings back the four quantities and the convergence flags, nothing else.
   - A final timing run refuses any core count other than 16, and every OpenDSS worker reports ready before anything is timed.
8. **U1 screen (Amendment 9, item 6).** The report applies the same rule to the U1 screen pipeline: speed-up of at least 1.10, interval above 1, no false-safe and no cleared-indeterminate decision of its screen or in what it delivered. This is reported next to the §7 criterion, which concerns PG-net. It was added after the dry run of item B: if a screen without a learned model gets the same speed-up, the paper must say so.
9. **A PG-net fit without a model** (Amendment 7, item 4) is timed and enveloped as a screen that defers every state.
10. **Envelope check (Amendment 9, item 7).** Evaluator B's fallback share is reported per direction. The E2 report needs the complete, valid envelope records of all six networks.

## Amendments 7 to 9: second set of corrections (2026-11-16, before the freeze)
Appended after a fifth read-only audit, of commit ed17a1a. The texts above are left as written. Still nothing frozen, and no final fit, final evaluation or timing run.

**A. The disclosure of the 12 test states, completed.**
- The evaluation that the unit tests ran on those states also computed, and wrote to pytest's temporary folders, state-accuracy metrics, the two rating-sensitivity screens, the sweep's agreement with OpenDSS and per-state outputs. None of these was displayed, beyond what part A above says.
- The calib states were 24: 12 of network 10 and 12 of network 25. The throwaway models were PG-pre and PG-net on 24 train states and PG-scratch and PG-ft on 12.
- **"About ten runs" and what was displayed come from the working session's own record. They cannot be verified from the repository.**
- **What the test guard covers** (`tests/conftest.py`): the study's state generator, the datakit generator, and `numpy.load` on the real calib and test label folders. It does not stop a test that reads a label file by other means. No test of the study does.

**B. The dry-run timings quoted in part B above** were those of a first run on network 1. Its files were removed when the dry run was repeated with the revised code. The repeat is kept, with the laptop's name removed: `results/fourwire_scale_dryrun_laptop_network1_timing.json` and `..._evaluation.json`.
- It gives, for 3,000 val states: Numba sweep 0.083 s without and 3.12 s with the OpenDSS solves of its 55 non-converged states; U1 screen pipeline 0.20 s with no OpenDSS solve; PG-net on the CPU 6.0 s.
- Fallback shares of 0.350 to 0.397.

**C. What the code enforces, stated as it now is.** Where this differs from parts C and D above, this is the rule.
1. **Claims.** One process at a time inspects, sets aside and claims a run folder; a second one is refused, not queued. A folder appears with its owner in it.
2. **Skips.** A run is skipped only if its record is valid and passes every check below, and for a fit only if its checkpoint is the one the record names. Anything else is an error, not a skip.
3. **Publication.** An evaluation, timing or envelope record is written to its run folder and then copied to `results/`. If a run dies between the two, the next call publishes the record it finds. It does not read a label again.
4. **The test reader** checks its authorisation when it is made and at every read: the freeze, the claim held by this process, no record yet, and the list of fits that this evaluation has checked.
5. **Record checks.** A fit record must be the fit its name stands for in §8: kind, fold, network, budget, seed and pool. Every final record must be valid, made with today's input hashes, and made under this freeze with its source hash and library versions, or under an earlier freeze that this one accepts for the stage.
6. **End of a run.** Besides Amendment 7 corrections, item C5: the job script and, where the stage uses them, the rating-variant networks (evaluation) and the meter profiles (envelopes) are compared with the freeze again.
7. **Halt.** Also written when a fit has used its two reruns. `collect` and both reports refuse while a halt is in force.
8. **Rerun reason.** A cancellation counts only if accounting names an account other than ours. A bare "CANCELLED" does not.
9. **Library versions in the freeze:** Python, NumPy, SciPy, PyTorch with its CUDA build, PyTorch Geometric and its optional native backends, Numba, OpenDSSDirect. **cuDNN is recorded per run and not pinned:** its version cannot be read on the machine where the freeze is written, and the model has no operation that uses it.
10. **Envelope records.** The report accepts a network's envelope record only if it holds exactly the 576 intervals of Amendment 9, item 7, each once, with the frozen settings, and a summary that follows from its intervals.
11. **Submission.** The final job chain is submitted by hand, after the A100 dry run of the same source has ended with `DRY_RUN_OK` and its output has been read. There is no automatic dependency on the dry run.

**D. What these guards are.** They protect the study against our own mistakes: a wrong folder, a repeated job, a stale record, two jobs for one run. They work on one file system and assume nobody edits records on purpose. They are not security.

## Amendments 7 to 9: third note on the run claims (2026-11-16, before the freeze)
After a sixth read-only audit, of commit 6013516. It cleared the reader of the test labels, the record checks, the end-of-run checks and the envelope-record check, and found three ordering faults in the claim of a run folder. This note replaces items C1 to C3 and C8 of the second set of corrections.

1. **Claims.** One process at a time inspects, sets aside and claims a run folder; a second one is refused. **The owner's liveness is checked first:** while the attempt that owns a folder is running, nothing in that folder is set aside, validated or counted, whatever it holds. A fit's two reruns are therefore used up, and the halt written, only when its last attempt has ended.
2. **Skips and publication happen inside the claim.** If the folder holds a valid record, that record and the files it names are checked, the record is published to `results/` where the stage has a published copy, and only then is the run reported as done. A record that fails a check is an error, not a skip. No label is read on that path.
3. **Rerun reason.** A cancellation counts if accounting names an account other than ours. A cancellation by our account, or one that names no account, counts only as a declared deviation, whose text is recorded.

## Note on two weak PG-pre seeds and an exploratory diagnostic (2026-11-16, during the final fits, before any evaluation record is opened)

Written while the final runs of freeze `a4068ef` are in progress. No evaluation record, timing record or report of the final runs has been opened.

**What was seen (training and validation loss only).** The curves of the twelve PG-pre fits, read from the job logs at 10:50:

- Fold 2, seed 1: validation loss 0.4985 at epoch 7; at epoch 8 the training loss rose from 0.60 to 1.91 and no later epoch improved on epoch 7; the fit stopped at epoch 15 by the patience rule. Selected epoch: 7.
- Fold 2, seed 2: validation loss 1.62 at epoch 19 (2.30 before training), still running.
- Fold 2, seed 0, and the nine fits of the other folds: between 0.05 and 0.21 at epochs 18 to 23.

**What does not change.**

1. No run is stopped, changed or run again. These are valid records; under Amendment 7 the first valid attempt is the fit.
2. They are reported as weak seeds, not as failures.
3. Every result is reported per seed and per network. No checkpoint is replaced by another seed's or by an average.
4. PG-zs and PG-pre-tc use PG-pre seed 0 (§4). The two weak seeds are the starting points of PG-ft with seeds 1 and 2 at B = 1,000 on the six networks of fold 2 (11, 12, 14, 15, 19, 20): 12 of the 120 PG-ft fits. Whether fine-tuning recovers from a weak starting point is part of the result and is not assumed.

**Caveat for the discussion.** One PG-pre checkpoint is the starting point for the six target networks of its fold, so its effect on them is correlated within the fold. The network bootstrap of §7 stays the primary analysis; the spread over seeds per network and the pattern per fold are shown next to it.

**Exploratory diagnostic (train and val states only; not a criterion).** `scripts/fourwire_scale_pgpre_diagnostic.py`, committed with this note. It reads the run folders and does not write into them.

- `checkpoints`: for each of the twelve selected PG-pre checkpoints, on the val states of its pool: the validation loss per pool network and its four terms (voltage, phase-to-neutral voltage, current, voltage-drop consistency), the root-mean-square size of the predicted and of the true correction to the U1 prior, and weight norms. The pooled loss is recomputed the way the fit computes it; a fit whose recorded loss is not reproduced within a relative 1e-5 is marked and its diagnostic is not interpreted.
- `retrace`: the fit of fold 2, seed 1 is run once more with the frozen fit routine, the same seed and deterministic algorithms on the A100, into a separate folder, keeping every step loss. Each step is given its network by replaying the fit's shuffling. This shows which networks carry the rise at epoch 8. The rerun's curve and weight hashes are compared with the record; if they differ, that is reported, the retrace is not interpreted, and the record stays the fit. The rerun is not a fit of the study and replaces nothing.

The diagnostic is run and its output committed before any evaluation record or report is opened. It is not used to select, change or exclude a run, a seed or a network.

**Not part of this study.** A stabilised training recipe (a lower peak learning rate, a linear warm-up, gradient-norm clipping, batches balanced across networks) is a candidate for a later robustness analysis. It would need its own dated amendment written before it is run, would cover all four folds and all three seeds, and would be reported as exploratory next to the frozen results, never in their place.

### Corrections to the note above (2026-11-16, after the ninth audit)

The audit of commit `a8a2612` found one fault in the script and several statements to qualify. No evaluation record, timing record or report has been opened.

1. **"No run is stopped, changed or run again"** means: no run of the frozen study is retried or replaced. The rerun in `retrace` is a separate, disclosed diagnostic on train and val states.
2. **Fold 2, seed 2 was still running** when the note was written. The figures given for it are a reading of its job log at 10:50, not its record. Whether its selected checkpoint is weak is stated from its record. Until then one weak seed is established (fold 2, seed 1).
3. **In the primary E3 comparison** (B = 1,000, the 23 fresh networks) the fits that start from the two fold-2 seeds are 12 of the 69 PG-ft fits, on 6 of the 23 networks.
4. **The script, corrected:**
   - It refuses an output path inside the run folders and any output file that exists. The first version checked only the rerun's folder.
   - Every record is checked by the study's own validator (the fit of that name, valid, made under this freeze with its source, libraries and inputs, with its checkpoint). For a final record the diagnostic refuses to run with another source, library set or GPU model, and it records whether the checkout was unchanged at the end.
   - `checkpoints` requires all twelve PG-pre fits of §8 and reads no other folder.
   - `retrace` now observes which network each forward pass used, in addition to replaying the shuffling. The two orders must be equal, otherwise nothing is written. The first version checked only step counts, which a wrong replay can satisfy.
   - The per-network loss is the fit's own validation quantity: a mean with the near-limit weights inside each batch of 128 states, the batches then averaged by their number of converged states. A plain mean over the converged states is reported next to it. The correction sizes are unweighted.
5. **Jobs queued with the first version** (13593190 and 13593191, with output paths outside the run folders): they are cancelled, or their output is discarded if they ran before this correction. What happened to them is stated with the diagnostic's output.

## Amendment 10 (2026-11-16): exploratory robustness analysis of PG-pre under a stabilised recipe (before it is run, and before any evaluation record is opened)

Written after the 276 final fits of freeze `a4068ef` ended and their fit records (training and validation loss) were read. The evaluation and envelope stages have completed on the cluster and the timing stage is running; none of their records, and no report, has been opened. The exploratory diagnostic of the note above is queued and has produced no output yet. The auditor proposed this analysis when the two weak seeds were seen; it is added at the author's request.

**Question.** Are the two weak PG-pre fits a property of the frozen optimisation recipe? The analysis is exploratory. It changes no frozen result, and no criterion of §7 depends on it.

**What the frozen PG-pre fits gave** (selected validation loss, from their records):

| Fold | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0 | 0.0308 | 0.0232 | 0.0273 |
| 1 | 0.0175 | 0.0223 | 0.0186 |
| 2 | 0.0201 | 0.4985 (epoch 7 of 15) | 1.5353 (epoch 60) |
| 3 | 0.0148 | 0.0168 | 0.0179 |

Fold 2, seed 2 is therefore weak by its record, as fold 2, seed 1 is (correction 2 of the note above left this open).

**The stabilised recipe.** PG-pre as frozen (model, loss, the first 5,000 train states and the val states of each pool network, scalers, batch 32, AdamW with weight decay 1e-4, at most 60 epochs, patience 8, the selection rule, folds, seeds 0 to 2, deterministic algorithms on the A100, the frozen source and libraries), with four changes:

1. Peak learning rate 3e-4 (frozen: 1e-3).
2. Warm-up: over the first 5 epochs the learning rate rises linearly, step by step, from zero to the peak, which it reaches at the last step of epoch 5. The cosine decay of the frozen recipe then runs over the remaining 55 epochs, once per epoch: epoch 6 trains at the peak.
3. The global gradient norm is clipped to 1.0 before every optimiser step.
4. Order of the batches: an epoch is a sequence of rounds, and a round holds one batch of every pool network, in a random order. As in the frozen recipe a batch holds states of one network and an epoch visits every training state once; the frozen recipe puts the batches of all networks in one random order.

No averaged (EMA) checkpoint is used. Where the auditor's proposal left a choice, this amendment fixes it: 5 warm-up epochs (proposed: 3 to 5), round-robin batches and not gradient accumulation (which would change the number of optimiser steps), no EMA (proposed as optional).

**Runs.** Twelve fits: the four folds and the three seeds, never fold 2 alone. Train and val states only; the label reader of the fits serves no other split. About 3.2 hours each on the A100, about 40 GPU-hours in all. They run from their own worktree and write under `data_out/fourwire_scale/runs/stabilised`, never into the run folders of the study. Code: `scripts/fourwire_scale_pgpre_stabilised.py` and `scripts/hydra_fourwire_scale_stabilised.slurm`; `src/` is unchanged, so the freeze stays valid, and every fit must pass the checks of a final command (pinned commit, the frozen source, inputs, libraries and GPU model).

**Equivalence check, before the fits.** The script has its own training loop, because the frozen routine has no switch for these changes. With the four changes off, that loop must be the frozen routine. `equivalence` runs both on one reduced fit on the A100 (fold 2, seed 1, 650 train and 256 val states per network, 4 epochs) and compares the curve, every step loss, the initial, selected and final weight hashes and the selected checkpoint file. The twelve fits start only if all are equal. On the laptop's CPU the same check (fold 0, seed 0, 80 train and 40 val states per network, 3 epochs, 162 steps) gave equality in every item.

**Attempts.** The first completed attempt of a fit is the fit. A fit is run again only after an infrastructure failure (the Slurm states of Amendment 7, item 3); the folder of the failed attempt is kept under the name `<fit>.attemptK` and is listed in the output. A numerical failure is a result and is not run again.

**What is recorded beyond the frozen fits.** After every epoch the validation loss of every pool network and its four terms; for every step the loss, the gradient norm before clipping and the learning rate.

**How it is read.** A fit is called weak when its selected validation loss is above five times the median of the twelve frozen fits (0.0212, so above 0.106), or when it selected no checkpoint. The same threshold is used for both recipes: for a given fold they share states, scalers and loss. `collect` writes `results/fourwire_scale_pgpre_stabilised.json` with, for each of the twelve fits, the frozen and the stabilised loss, selected epoch and number of epochs, their ratio, the loss per pool network at the selected epoch and a summary of the gradient norms. There is no significance test. What may be said:

- If no stabilised fit is weak: the weak seeds did not recur under the stabilised recipe, which is consistent with a sensitivity of the frozen optimisation. It does not show which change matters (four change together), and three seeds per fold do not bound how often a weak fit occurs.
- If a stabilised fit is weak: that is reported, and the recipe is not said to remove the problem.
- In both cases the level is reported too: whether the stabilised losses of the other fits are above or below the frozen ones.

**What it is not.** No stabilised checkpoint is fine-tuned, calibrated or evaluated on calib or test states, and no result of E2 or E3 uses one. The frozen fits and everything computed from them stay the results of the study, weak seeds included. Any further use of the stabilised checkpoints needs its own amendment before it is made.

**Known before the runs.** In a laptop dry run of the stabilised recipe (64 train states per network, 8 epochs) more than 90% of the steps had a gradient norm above 1.0. Early in training the clipping is therefore active on most steps, and the size of a step is set by the optimiser and the learning rate more than by the size of the gradient. The share of clipped steps is recorded per epoch. The clipping value is the auditor's and was not tuned.

**Order.** The recipe, the runs and the reading are fixed here, before any evaluation record of the final runs is opened. The stabilised fits may therefore finish after the final reports are made without being informed by them.

### Correction to Amendment 10 (2026-11-16, before any run)

"Runs" says that the label reader of the fits serves no other split than train and val. That is wrong. The reader (`fourwire_scale_data.LabelStore`) serves train, val and calib states and cannot serve test states. The script asks it for train and val states only; it never requests a calib state. The script's own description said the same and is corrected.

### Corrections to Amendment 10 (2026-11-16, after the tenth audit; nothing of Amendment 10 has been submitted)

The audit of `bd1cc9c` confirmed that `src/`, the freeze file and the final job script are unchanged, and that the script's loop with the four changes off is the frozen computation. It found the checks around the analysis insufficient and several statements too strong. No evaluation record, timing record or report has been opened.

1. **Equivalence check.** It now passes only if the two fits are equal in every item, both completed every epoch without a numerical failure, and checkout, script, libraries and inputs were unchanged at the end. On the cluster it runs only as a final command of the study runs (pinned commit, the frozen source, inputs, libraries and GPU model, deterministic algorithms). Its size is now 648 train and 312 val states per network, so that the last training batch holds 8 states and the last validation batch 56, as at full size. It cannot exercise: stopping by patience; the cosine steps after epoch 4 and a decay over 60 epochs; target scales computed from more states (the function that computes them is shared by both loops); and the four changes themselves, which no equivalence covers.
2. **Evidence kept.** The laptop check quoted in the amendment (80 train and 40 val states) was not kept. It was repeated with the corrected script and is kept: `results/fourwire_scale_pgpre_stabilised_equivalence_laptop.json` (fold 0, seed 0, 72 train and 184 val states per network, 3 epochs, 162 steps; passed). The laptop dry run of the stabilised recipe is kept as `results/fourwire_scale_pgpre_stabilised_dryrun_laptop.json`; its curve shows the learning rate of the first and last step of every epoch. Both were made with the script as committed here (its SHA-256 is in the records) while HEAD was still `bd1cc9c`; the laptop's name is removed.
3. **"Known before the runs", restated.** In that one laptop dry run (64 train states per network, 8 epochs of 36 steps) between 92% and 100% of the steps of each epoch had a gradient norm above 1.0. Nothing is claimed about the full fits. The sentence about what sets the size of a step is withdrawn.
4. **Schedule.** The first update uses the peak divided by five times the number of steps of epoch 1, not zero. The rise is linear in the fraction of the epoch completed, so if two warm-up epochs differ in their number of steps, the increment per step differs with it. Epoch 60 trains at the peak times (1 + cos(54π/55))/2; zero would be reached after it.
5. **Batches without a converged state.** As in the frozen recipe such a batch is not a step, and its states get no forward pass in that epoch. "An epoch visits every training state once" and "a round holds one batch of every pool network" hold up to such batches; a round then lacks that network.
6. **Attempts.** They are handled by the study's own claim of a run folder (Amendment 7, item 3), not by renaming by hand. A fit with a valid record is never run again. A folder without a record is set aside, as `<fit>.incompleteK`, only when Slurm accounting shows an infrastructure end state, or as a declared and recorded deviation after our own cancellation. A fit is invalid, and its command fails, if checkout, script, libraries or inputs changed during the run; its folder is then set aside as `<fit>.invalidK`. At most two earlier attempts. `collect` refuses a fit when the attempt folders on disk are not those its record lists, or when one of them was not set aside by the claim.
7. **Outputs.** No output may lie in a folder `runs/final` or `runs/dry` of any checkout, or in the folder of frozen runs given to `collect`. Full-size fits go to the worktree's own `runs/stabilised` and nowhere else. The outputs of `equivalence` and `collect` are created only if they do not exist, and the output of `equivalence` must be outside its work folder.
8. **`collect`.** Every stabilised record must be valid, full size, the stabilised recipe with the frozen training configuration, made under this freeze with its source, libraries, inputs, GPU model and numerical mode, with the checkpoint and step file its record names, and with the states and scalers of the frozen fit of the same name. The twelve fits and the equivalence check must come from one commit and one script (each record holds the script's SHA-256), and the equivalence check must have passed in the frozen environment. A fit that ended before its first step is reported as weak, without a gradient summary. The output states that the gradient summaries cover every step, the epochs after the selected one included; that the median ratio is taken over the fits in which both recipes selected a checkpoint, with their number; and that the loss per network is the fit's validation quantity (near-limit weights inside each batch of 128 states), not one weighted mean per network.
9. **"Every fit must pass the checks of a final command"** holds at the start of the fit and again at its end.
