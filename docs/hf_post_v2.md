# GENCO meets the neutral wire: four-wire-gated dynamic operating envelopes for low-voltage grids

*What breaks when a balanced grid foundation model reaches household networks, and how a phase-aware extension repairs it.*

*GridFM GENCO Challenge 2026. Ander Martinez Alonso, Evgenii Genov (EVERGi, VUB) and Remy Cleenwerck (VUB, UGent). Code, protocols, figures and results: [release commit 2401dfd](https://github.com/ujohn33/genco-dynamic-envelopes/tree/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef); official-checkpoint experiments: [commit 48a1218](https://github.com/ujohn33/genco-dynamic-envelopes/tree/48a1218a0ea7ce2e35f31ff8914ffc034fe31245). Contact: ander.martinez.alonso@vub.be.*

---

## The question, and the short answer

European low-voltage grids are four-wire systems: households connect to individual phases, and the neutral carries the imbalance. Whether one more kilowatt of EV charging or rooftop export is safe depends on that physics. GENCO represents the grid as balanced. So we asked a deliberately adversarial question: **can GENCO support household dynamic operating envelopes, and what would it take to extend it to the distribution level?**

Used directly, no. On a 200-customer benchmark feeder, envelopes from a balanced model were two to three times too large, and none met the four-wire feasibility target in any evaluated hour. But the failure was informative. We found the representation bottlenecks, extended the GENCO architecture to phase-aware four-wire graphs, and put a full four-wire OpenDSS gate around every operational decision. **The phase-aware learned residual cut OpenDSS fallbacks from 35.8% to 5.8%** relative to a one-step physics prior, with zero observed false-safe screening decisions on 2,968 evaluation states. A specialised iterated solver remains more accurate. The contribution is a transparent hybrid workflow, and a map of what GENCO must change to operate on LV networks.

## 1. Why balanced GENCO misses household-grid physics

![Balanced versus four-wire voltages](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/figures/fig0_feeder.png)

*ENWL feeder 1. (a) Customer phase. (b, c) One state with 186 kW net export: the balanced model puts every customer at 240–247 V; the four-wire solution of the same injections ranges from 205 V (phase c) to above 270 V (phase a).*

- **Setting.** The ENWL1 four-wire LV benchmark (UK, 800 kVA, 200 single-phase customers), populated with Fluvius 2024 smart-meter households (30% heat pump, 30% EV, 40% PV) and UK statutory limits of 216.2–253.0 V. It is a benchmark feeder, not an observed Flemish one.
- **What a balanced model sees.** It averages each bus over three phases, so it cannot see one phase sagging while another rises, or the neutral carrying hundreds of amperes.
- **Consequence for envelopes.** Over every hour of the 36 evaluation days, not one positive balanced-model envelope met 0.95 four-wire feasibility on fresh draws (0 of 864 export hours, 0 of 823 import hours).

## 2. Extending GENCO to the distribution level

**Step 1: make the architecture work on an LV feeder.** We trained graphkit's GENCO architecture (`GNS_heterogeneous`) on this feeder. Out of the box it did worse than predicting the training mean. We traced three bottlenecks:

1. **Impedance magnitude.** The edge encoder ends in a LayerNorm. With LV admittances of order 1e4 the encoder is nearly positively homogeneous, so the LayerNorm strongly suppresses the magnitude information that distinguishes a long cable from a short one. We restore it with asinh-scaled admittance features.
2. **Physics-loss collapse** at LV numerical scales. We add a linear power-flow prior with a zero-initialised, scaled residual head, so training starts exactly at the prior.
3. **Ill-conditioned flows.** Flows from voltage differences turn a 1e-4 pu voltage error into an O(100%) flow error on short cables. We compute flows from Kirchhoff's current law.

![GENCO development](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/figures/fig3_genco_adaptation.png)

Restoring the admittance features was necessary but not sufficient on its own. The decisive improvement came from combining the corrected representation with the linear prior and residual head: voltage error 1.0e-3 pu on envelope states, 5.4× better than the prior alone. (The two panels are separate runs on different state sets, not one ablation sequence.) Along the way we found two tooling bugs: graphkit's best-model export silently kept epoch-0 weights, and datakit pinned the slack at 1.5 pu.

**Step 2: make it phase-aware.** A balanced model cannot represent the binding physics, however well trained. We therefore built a GENCO-style encode–process–decode GNN on the explicit four-wire graph: 397 buses with conductors a, b, c and neutral, full 4×4 line impedances without Kron reduction, and an edge-conditioned TransformerConv processor. It predicts a zero-initialised residual on a cheap one-step four-wire sweep.

## 3. What we built: a GENCO screen behind a full four-wire gate

![Workflow](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/figures/fig8_workflow.png)

- **Scenarios.** For each interval, 200 scenarios resample real meters of the same technology group at the same date and time. This is an empirical uncertainty model, not a day-ahead forecast.
- **Envelope.** One optional allowance per household above the passive demand, issued only if the full nonlinear four-wire OpenDSS model finds it feasible in at least 98% of the 200 design scenarios (the finite-sample rule for ε = β = 0.05). This is a **design-set check, not a certificate of future safety**: on fresh draws, issued non-zero envelopes average 97.4% four-wire feasibility, but only 47% pass a one-sided 95% lower-bound test, because 200 draws are too few to prove 95%.
- **The learned screen** decides the clear cases and sends unsure ones to OpenDSS.

## 4. What the phase-aware extension buys

![OpenDSS fallbacks](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/figures/fig4a_fallback.png)

On 2,968 evaluation states, with the same final gate for every method:

- **OpenDSS fallbacks fall from 35.8% to 5.8%** relative to the one-step prior, with **zero observed false-safe decisions**.
- Phase-to-neutral voltage error falls by 91% (on the 2,699 converged states), and near-limit violation classification improves from 0.926 to 0.992 balanced accuracy.
- Balanced screens, whether a balanced sweep or balanced GENCO, must send 63% of states to OpenDSS. Representing the phases is what makes screening useful.
- One training seed per model, so the intervals exclude training variability.

## 5. What it means operationally

![Capacity](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/figures/fig1_capacity.png)

With every envelope passing the full four-wire gate, dynamic envelopes free **4.81 MWh of export** (day-bootstrap interval 4.35–5.30) and **2.50 MWh of import** (2.11–2.87) envelope-hours per household per year over the 36 evaluation days. These are hours of allowed headroom, not delivered energy. A constant optional add-on, one allowance valid in every interval, frees nothing: in about half of all hours no positive envelope passes the rule at all. Across five household-technology placements the capacity is 4.8–6.2 MWh export and 2.5–3.2 MWh import; the headline placement is the lowest.

These capacity figures come from the full OpenDSS search. The GENCO screen was evaluated under the same gate on 2,968 states, not yet run through the annual study. We also compared other workflow variants:

| Workflow variant | Result |
|---|---|
| GENCO screening **without** a final OpenDSS gate | One material false acceptance in 1,728 intervals: the gate is necessary |
| GENCO screening **with** a final OpenDSS gate | Same envelopes as the full search in 100% of intervals, 6% fewer OpenDSS calls; classical balanced screens do as well |
| Learned envelope proposal + one final gate | 79% of the full search's envelope-hours with 6.7% of its OpenDSS calls (a call reduction, not a measured speed-up) |

## 6. Where pretraining helped, and where it did not

Two results look contradictory but are not:

- **Balanced embeddings into a four-wire model: no help.** Feeding embeddings from our feeder-specific balanced GENCO into the phase-aware GNN increased fallbacks from 5.8% to 11.8% (+6.0 points [5.3, 6.7]). Adding the same embeddings to the learned envelope proposal changed retention by −0.06 points [−0.46, +0.32]. A balanced representation does not carry the four-wire information.
- **Official pretraining on its native representation: clear help, but short of our bar.** On case118 with 99 flexible export units, fine-tuning the official `gridfm/genco-opf-datakit` checkpoint beat training from scratch at every label budget: +14.2, +6.9 and +8.4 points of reference hosting at 100, 1,000 and 10,000 labels. A frozen follow-up at 100 labels, with five training seeds on 2,000 unseen scenarios, replicated the gain: **+11.4 points [9.8, 12.5], better in 5 of 5 seeds**. Both preregistered criteria nevertheless failed on an absolute target. At 1,000 labels the fine-tuned model kept 89.4% of the raw IPOPT reference against a 95% bar, and in the follow-up it kept 94.7%. These use an IPOPT corrective-redispatch gate. The networks propose the flexible dispatch and IPOPT redispatches the rest, so this is a hybrid, not a standalone neural OPF. ([log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/48a1218a0ea7ce2e35f31ff8914ffc034fe31245/docs/experiment_log.md))

Pretraining transfers when the representation matches. It does not substitute for a representation that is missing.

## 7. What remains classical, and why the learned part still matters

The specialised iterated four-wire sweep is the accuracy leader on this feeder: 3.6e-6 pu error and 0.1% fallback, about 80× more accurate than the GNN. We do not claim the GNN beats it, and we make no wall-clock claim, because our timings compare an A100 model with an unoptimised CPU sweep. The learned component still matters for three reasons:

- it shows that the GENCO architecture can absorb phase-aware corrections once the representation allows it;
- it cuts calls to the expensive OpenDSS model six-fold relative to the cheap prior it corrects;
- its potential value is amortisation across harder networks and workflows where an equally strong specialised solver is not available.

On this feeder, today, the right deployment is physics-first with a learned screen and a final gate.

## 8. Runnable artifacts, evidence status and limitations

- **Run it yourself, no data access needed.** `python scripts/demo_neutral_wire.py` solves one synthetic household state on the public ENWL1 files with both models in a few seconds. The balanced model reports 0 customers outside the limits; four-wire OpenDSS reports 34, from 205 to 257 V, with 401 A in the transformer neutral.
- **Code, protocols and results.** These are at the release commit (Apache-2.0): numbered amendments, the [experiment log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/docs/experiment_log.md), run manifests with SHA-256 hashes, and [claims and non-claims](https://github.com/ujohn33/genco-dynamic-envelopes/blob/2401dfd2d07ebe09f6b06a7341dfe70af24ebdef/docs/claims.md). There are 84 automated tests. `scripts/reproduce_check.py` regenerates every headline number byte for byte from the recorded run files.
- **Not released, and why.** Until Fluvius confirms that its licence permits redistributing data derived from individual meters, we do not publish the LV checkpoints, four-wire labels or per-day run files. Their hashes are in the manifests, and everything needed to regenerate them is in the repository.
- **Evidence status.** The full OpenDSS search and the GENCO screen were pre-registered. Everything else is exploratory, with protocols frozen before results. The 36 evaluation days were also used during development, so they are not a pristine held-out set. Each learned model was trained once unless stated otherwise.
- **Scope.** One benchmark feeder. We make no claims of generalisation to other feeders or years, delivered energy, or economic value.

**Next step.** The household envelopes on a feeder sum to a flexibility range at the MV/LV transformer. Carrying that range up through MV to the transmission level needs exactly what this entry identified: magnitude-preserving edge features, level-specific (phase-aware) representations, and a full physics gate at every tier.

---

**Team:** Ander Martinez Alonso, EVERGi, Vrije Universiteit Brussel (VUB), ander.martinez.alonso@vub.be · Evgenii Genov, EVERGi, VUB · Remy Cleenwerck, VUB and Ghent University (UGent). Built on GENCO with gridfm-graphkit 0.9.0.3 and gridfm-datakit 1.1.0.
