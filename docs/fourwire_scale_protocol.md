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
