# Unbalanced (four-wire) GENCO: protocol (exploratory extension)

**Status: exploratory extension**, on branch `ander-unbalanced-genco`.
- Committed before any label is generated or any model is fitted.
- The frozen B / C / C+, annual DOE, Option 2, envpred3 and W1–W3 results are untouched.
- The existing conclusions are rewritten only if this experiment meets the frozen success criterion below.
- **Disclosure.** Global test-split summaries of this feeder were inspected in earlier work (balanced GENCO accuracy, four-wire screens, DOE results). This experiment is therefore exploratory, not a pristine pre-registered test.

**Dates.** Hard stop for model experimentation: 21 Oct 2026. Challenge deadline: 25 Oct AoE.

**Scope.** One feeder-specific proof of concept on ENWL1. No claim of multi-feeder generalisation, nor of a pretrained unbalanced foundation model.

## 1. Research question

> Can a GENCO-style four-wire model close the balanced-to-four-wire accuracy gap and reduce exact OpenDSS fallbacks for dynamic-envelope decisions?

## 2. Methods compared

| ID | Method | Role |
|---|---|---|
| **U1** | One-iteration four-wire backward/forward sweep from nominal phasors (the physics prior) | primary baseline |
| **U2 = PhaseGraph** | U1 + the unbalanced GENCO residual model (`GNS_heterogeneous_4wire`): four-wire prior, phase-specific injections, topology and impedance features | primary method |
| **PhaseGENCO** | exactly PhaseGraph (same architecture after the input projection, capacity, training procedure, seed and states), additionally conditioned on **node-aligned** final-layer bus embeddings of the frozen balanced GENCO | representation-value ablation |
| U3 | Iterated four-wire sweep (to tolerance) | strong classical reference, only if its agreement with OpenDSS is demonstrated on the validation dates first. Otherwise reported as "iterated sweep" and never called exact |
| BG | Existing balanced GENCO (frozen v2 LV model) | contextual |
| BS | Exact balanced radial sweep | contextual |
| T | OpenDSS four-wire | truth |

**The central tests:**
- **U2 (PhaseGraph) against U1** asks whether a phase-aware learned model helps.
- **PhaseGENCO against PhaseGraph** asks whether GENCO's bus-level representations add value.

A comparison against balanced GENCO alone is never presented as the result.

## 3. Data and splits (frozen)

- **States:** the existing v2 datakit states. Each customer has its own bus, so a state's bus Pd/Qd is the customers' net consumption.
- **Splits:** the existing calendar-day splits, unchanged.

| Split | States | Of which four-wire non-converged (known structurally from the existing truth files) |
|---|---|---|
| train (`datakit_v2`, `splits/train.pt`) | 14,648 | 1,353 |
| val (`splits/val.pt`) | 5,066 | 471 |
| calib (`datakit_v2_calib`) | 2,953 | 253 |
| test (`datakit_v2_test`) | 2,968 | 296 |

- **Non-converged states** (voltage collapse) have no valid state targets:
  - they are excluded from the state losses and state metrics;
  - in decision-level evaluation they count as violating, and they get an infinite score in calibration, exactly as in `FourWireMargins`.
- **Test isolation.** No test label, prediction or metric is opened until the architecture, preprocessing, training rule and checkpoint rule are frozen (amendment before the final fit). Test labels are generated together with the rest, but they are not read by any training, debugging or calibration code; a test-isolation check enforces this.

## 4. Four-wire data representation

**Topology.**
- Parse the full 4×4 R and X matrices of every line code. Capacitance is zero in ENWL1.
- Start at the transformer secondary bus `TRbus`.
- Keep `TRbus`, every customer bus and every branching bus.
- Prune leaves without customers.
- Merge series chains by **adding their 4×4 impedance matrices**. This is exact without shunts. A merged chain's rating is the minimum along the chain.
- Neutral and mutual coupling are kept. **No Kron reduction.**
- Explicit maps between OpenDSS `bus.conductor` names and graph nodes are stored.

**Root and transformer.**
- `TRbus` is grounded through a 1e-10 Ω reactor, so its neutral is ≈ 0.
- The source and the Dyn transformer are represented at `TRbus` by a 3×3 phase-domain Thevenin equivalent (open-circuit phasors E and impedance Z_th), identified **once** from OpenDSS with small test injections. It is a fixed network parameter, not a per-state label.
- Its accuracy is reported.

**Bus inputs** (per retained bus):
- Pa, Qa, Pb, Qb, Pc, Qc (customer net consumption by phase);
- customer count by phase;
- root indicator;
- radial depth;
- electrical distance from the root (cumulative phase-a series resistance);
- U1 prior: Re/Im of Va, Vb, Vc, Vn.

**Edge inputs** (parent → child plus reverse, with a direction flag):
- the 16 R and 16 X entries;
- rating (A);
- length (m);
- direction.

All inputs are scaled with asinh, then standardised with **training-date statistics**.

**Targets** (OpenDSS, absolute phasors in the OpenDSS frame):
- per bus: complex Va, Vb, Vc, Vn, and phase-to-neutral magnitudes;
- per retained branch: complex Ia, Ib, Ic, In at the parent terminal of its first original segment. In a merged chain without taps the current is constant along the chain;
- per state: transformer phase powers and loading, convergence, and the four violation labels (undervoltage, overvoltage, cable overload, transformer overload) under the UK limits.

**Storage.**
- float32, after checking that the float64 → float32 error is < 1e-6 pu.
- Atomic writes, schema hash, engine-health counters.
- Semantic integrity checks, including:
  - KCL at every bus;
  - Vn(TRbus) ≈ 0;
  - V_pn recomputed from the stored phasors equals the stored V_pn;
  - the violation labels are consistent with the stored states;
  - healthy engines and finite values.

## 5. Four-wire physics prior (U1) and iterated sweep (U3)

**Sweep step:**
1. Start from E at the root and nominal phasors downstream, with Vn = 0.
2. Customer current: I = conj(S / (V_phase − V_neutral)), leaving on the phase conductor and returning on the neutral conductor.
3. Backward pass: sum the conductor currents from leaves to root.
4. Root: V_root = E − Z_th · I_root,abc; Vn,root = 0.
5. Forward pass: V_child = V_parent − Z_4×4 · I_branch.

**U1** is exactly one such iteration. **U3** iterates to a maximum update of 1e-8 pu, or at most 50 iterations.

**Before any training**, both are validated against OpenDSS on normal, export-heavy, import-heavy and strongly imbalanced train/val states. The discrepancies are reported, and the agreement must be shown before U3 may be called exact.

## 6. Unbalanced GENCO model (`GNS_heterogeneous_4wire`)

GENCO-style encode–process–decode on physical bus nodes, with conductor-vector states.

**Disclosure.** It reuses GENCO's processor family (PyG `TransformerConv` with edge features, residual updates and per-layer normalisation) and is registered in graphkit's model registry. However, it is trained with our own loop: graphkit's datakit/normalizer pipeline cannot carry four-wire states.

**Encoders.**
- Bus MLP on the bus inputs.
- Edge MLP on the 35 edge inputs, after asinh scaling and standardisation, **without** a final LayerNorm. This avoids the magnitude-cancellation problem found in the stock encoder.

**Processor.**
- L layers of bidirectional `TransformerConv` (parent→child and child→parent edges, edge-conditioned attention), with residual connections.
- A per-graph global context channel (mean of node states, projected and added back).

**Decoder.**
- MLP to an 8-vector: ΔRe and ΔIm of Va, Vb, Vc, Vn, added to U1.
- The **last layer is zero-initialised**, so the untrained model reproduces U1 exactly.

**Derived quantities.**
- Phase-to-neutral voltage from the predicted phasors.
- Branch currents by **KCL**: customer currents from the predicted voltages, summed over the subtree.
- **Deviation from the brief, stated:** the brief asks for currents from voltages and line matrices, Z⁻¹·ΔV. On short LV segments that is ill-conditioned: a 1e-4 pu voltage error becomes an O(100%) flow error, as found in this project's balanced work. KVL is therefore enforced as a **physics-consistency loss**, not used to compute the currents.

**Loss.** All terms are normalised by training-set scales and their weights are fixed now:

| Term | Weight |
|---|---|
| Complex-voltage residual: MSE of (V − V_true) for all 4 conductors | 1.0 |
| Phase-to-neutral magnitude, MSE | 1.0 |
| Branch phase and neutral current, MSE on KCL currents (normalised by rating) | 0.5 |
| KVL consistency: ‖V_parent − V_child − Z·I_KCL‖² (normalised) | 0.1 |

States "near a limit" get **weight 3**. Near a limit means the true worst V_pn is within 0.02 pu of 0.9002 or 1.0534 pu, or the true worst cable or transformer loading is within 0.9–1.1.

### 6b. PhaseGENCO conditioning (the ablation)

1. Run the frozen selected balanced GENCO (v2 LV + prior, run `d42d5c34…`) on the **same state** (the datakit bus Pd/Qd). Use the frozen A100 FP32 path with TF32 off.
2. Keep its **final per-bus hidden representation**: the input of the bus output head at the last layer, width 96.
3. **Never mean- or max-pool it.**
4. Align every embedding explicitly to the same physical ENWL bus in the four-wire graph, by bus name. The four-wire reduction uses the same keep-set as the balanced one; `sourcebus` has no four-wire node and is dropped.
5. At each bus, concatenate the embedding with Pa..Qc and the U1 prior, then apply the input projection. **Only the input projection differs** from PhaseGraph. Parameter counts are reported.
6. The balanced GENCO backbone is **frozen** (no gradients). Embeddings are precomputed once per state.

**Assertions (code and tests):**
- complete one-to-one bus alignment (every four-wire bus has exactly one balanced bus, and vice versa except `sourcebus`);
- embedding width 96;
- identical bus ordering across all states and batches;
- no pooled or global embedding replaces the per-bus representation;
- no parameter of the balanced checkpoint receives a gradient.

## 7. Training rule (frozen before the final fit)

- **Debugging** of architecture and optimiser uses train/val only. Defaults: hidden 64, 8 layers, 4 heads, AdamW lr 1e-3, cosine schedule, batch 32, at most 60 epochs, early stopping with patience 8 on validation loss. Seed 0.
- **Runs allowed:** one deterministic smoke fit per model (a few epochs, train/val only) and **one final fit per model**, i.e. PhaseGraph and PhaseGENCO, two in total. Same seed, same states, same schedule. The architecture and optimiser are frozen in an amendment committed before the final fits.
- **Checkpoint:** the lowest validation loss.
- **Calibration:** conformal margins on calib only. Inference uses the same A100 FP32 path (TF32 off) for calib and test.
- **Recorded:** code, data, schema, checkpoint and configuration hashes.

## 8. Evaluation (test, after freezing)

### State prediction
On converged test states, for U2, U1, U3 (if validated), BG and BS where applicable:
- **Errors:** V_pn MAE, p95, p99; worst-customer V_pn error; neutral-voltage magnitude error; phase-loading MAE; neutral-current MAE; transformer-loading MAE.
- **Classification:** precision and recall of the four violation labels.
- **Near-limit subsets:** the same metrics restricted to the states near each boundary.

### Decision level
- Each method's state-level worst cases (min V_pn, max V_pn, max phase loading, transformer loading) feed a `FourWireMargins` screen with the same Mondrian bins and α = 0.01 over 4 criteria, calibrated on calib.
- Applied to test per direction (import: min V; export: max V; both: loading).
- Reported per method and direction:
  - share of states cleared safe;
  - share rejected as violating;
  - share sent to OpenDSS;
  - observed false-safe count;
  - total exact calls;
  - model-only and end-to-end runtime, including feature construction and the U1 prior.

## 9. Success criterion (primary; U2 against U1, all required)

On held-out converged four-wire test states:

1. **Accuracy:** V_pn MAE reduced by **≥ 30%**.
2. **Near-limit classification:** improved, i.e. higher mean balanced accuracy across the four violation labels on near-limit states, where each label with both classes present counts.
3. **Fallbacks:** share of test states needing an exact OpenDSS fallback, averaged over both directions, reduced by **≥ 5 percentage points**.
4. **Safety:** observed false-safe decisions after the frozen calibration **no greater** than U1's.
5. **Cost:** total runtime reported, including feature construction and inference.

Everything else is secondary. The comparison with U3, when validated, is always reported next to it.

## 9b. GENCO-specific representation-value criterion (PhaseGENCO − PhaseGraph; all required)

- **Fallbacks:** PhaseGENCO reduces the share of test states needing an exact OpenDSS fallback, averaged over both directions, by **≥ 5 percentage points** against PhaseGraph.
- **Uncertainty:** the paired bootstrap 95% interval of that difference excludes zero.
  - The resampling unit is the **date**. Datakit states are drawn per date; there is no household-placement dimension in this data.
  - Dates are resampled within season, 2,000 replicates, seed 0.
- **Safety:** no increase in observed false-safe decisions.
- **Cost:** end-to-end cost reported, including the balanced GENCO embedding pass.

**Reported for both models:**
- OpenDSS fallback rate, and its difference with the interval;
- exact calls actually made;
- observed false-safe count;
- V_pn and neutral-current errors;
- model-only and end-to-end runtime.

**Attribution rule.** If only PhaseGraph improves on U1 while PhaseGENCO ≈ PhaseGraph, the improvement is attributed to the **unbalanced representation, not to GENCO embeddings**. No causal GENCO claim is made unless criterion 9b passes.

## 10. Claim discipline

**If successful,** the strongest permitted statement is:

> "We extended GENCO from balanced positive-sequence prediction to a feeder-specific four-wire representation with explicit phase and neutral states. On held-out ENWL1 states, the unbalanced model reduced [metric] relative to its four-wire physics prior and reduced exact OpenDSS fallbacks by [X], showing that phase-aware representation — not merely better balanced accuracy — is necessary for this LV application."

**If it fails:**

> "A full phase-aware representation removed the balanced-model information bottleneck, but the specialised four-wire sweep remained as good or better. This isolates the remaining challenge as learning efficiency/generalisation rather than missing phase information."

**Safety wording.**
- Preferred: "...while the exact four-wire gate preserved the frozen acceptance rule and produced zero observed false-safe decisions on the held-out evaluation" (when true).
- Not used: "maintained decision safety" as an unconditional guarantee. A finite evaluation supports an empirical statement only.

**Never claimed:**
- a universal unbalanced foundation model;
- multi-network generalisation;
- superiority over all classical solvers;
- formal safety from a finite empirical test;
- an end-to-end speed-up that was not measured.

## 11. Smoke gates (all must pass before the final fit)

1. **Labels:** semantic integrity checks pass on every file, with no unhealthy engine.
2. **Prior:** U1 and U3 validated against OpenDSS on train/val states, with discrepancies reported.
3. **Tests:** the new unit tests pass, plus the existing suite. They cover:
   - 4×4 parsing and series-sum preservation;
   - node maps;
   - the synthetic four-wire sweep;
   - balanced inputs giving a small neutral current;
   - imbalanced inputs giving neutral displacement;
   - zero-init reproducing U1;
   - batch isolation;
   - shapes;
   - finite gradients;
   - date isolation;
   - unhealthy-engine failure.
4. **Smoke fit** on train/val only. Reported: shapes, memory, epoch time, initial (= U1) metrics, the validation curve, evidence of learning beyond a constant correction (the variance of the predicted residual across states, and residual-versus-target correlation), and the projected cost of the full run.

If any gate fails, stop and diagnose. **No final fit.**

## Amendment 1 (2026-10-18): freeze before the final fits; decision-screen correction

Committed before the final fits and before any test state is opened.

### Frozen training configuration (identical for PhaseGraph and PhaseGENCO)

| Item | Value |
|---|---|
| Architecture | hidden 64, 8 TransformerConv layers, 4 heads, global context channel, zero-initialised residual head |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule over 60 epochs |
| Batch, seed | 32, seed 0 |
| Early stopping | patience 8 on validation loss; maximum 60 epochs |
| Checkpoint | the lowest validation loss |
| Loss weights | V 1.0, V_pn 1.0, current 0.5, KVL 0.1; near-limit states weighted 3 |
| Input scaling | asinh then standardisation from training statistics, including PhaseGENCO's embeddings (smoke fix 054991b) |
| Parameters | PhaseGraph 216,840; PhaseGENCO 222,984 (only the input projection differs) |

**Runs.** One final fit per model (`--tag final`). The frozen evaluation is `gencodoe.fourwire_eval`: calibration on calib, then test.

### Decision-screen correction (found in a validation dry run)

- **Dry run.** Before freezing, `fourwire_eval` was run on calib + **validation** states with the smoke checkpoints (`--eval-split val`). Validation decision results were therefore seen. No model, feature or criterion was changed because of them.
- **Defect found.** The Mondrian (binned) `FourWireMargins` screen gave infinite margins in most bins.
  - With α/4 = 0.0025, a bin needs about 400 calibration states for a finite margin. With 2,953 calibration states over 13 voltage bins, most bins fall short.
  - Even the near-exact iterated sweep U3 therefore sent 49% of states to OpenDSS. Criterion 3 could not discriminate between methods.
- **Correction, applied identically to every method.** One global (unbinned) split-conformal margin per criterion, fitted on **converged** calibration states, with α = 0.01 split over the 4 criteria.
  - The scores and decision logic are those of `FourWireMargins`.
  - Non-converged test states still count as violating, so clearing one counts as false-safe.
  - A method without a finite state defers to OpenDSS.
- **Effect on the validation dry run** (fallback share, mean of both directions): U3 0.1%, PhaseGraph 26%, PhaseGENCO 28%, U1 36%, balanced models 64%. False-safe 0 for all.

### Unchanged

Every success criterion (section 9 and 9b), its thresholds, the splits and the claim discipline.
