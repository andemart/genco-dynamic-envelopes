# GENCO meets the neutral wire: four-wire-gated dynamic operating envelopes for low-voltage grids

*Extending GENCO to three-phase four-wire low-voltage distribution networks: what breaks, and how a phase-aware extension repairs it.*

**Ander Martinez Alonso¹,³, Evgenii Genov¹, Rémy Cleenwerck¹,²**  
¹ EVERGi, Vrije Universiteit Brussel · ² EnSy/Lemcko, Ghent University · ³ DTU Wind and Energy Systems, Technical University of Denmark  
*GridFM GENCO Challenge 2026. Code, protocols, figures and results: [release c02b342](https://github.com/ujohn33/genco-dynamic-envelopes/tree/c02b3420e5f53ea9483e74e029253db8be44b93b); official-checkpoint experiments: [commit 994d5d4](https://github.com/ujohn33/genco-dynamic-envelopes/tree/994d5d416a0c45fdbf6045c7103f3cd9c7519b53). Contact: ander.martinez.alonso@vub.be.*

---

## Summary

GENCO models the power grid as balanced. Can it be extended to low-voltage distribution networks, where every household is connected to a single phase? We test this on household dynamic operating envelopes and find the following.

- **A balanced model is unsafe at LV.** Across every hour of 36 evaluation days, not one positive envelope designed with the balanced model met the four-wire feasibility target.
- **A phase-aware extension of the GENCO architecture makes it useful.** Used as a pre-screen, it cuts the number of full four-wire OpenDSS simulations six-fold (fallbacks 35.8% → 5.8%), with zero observed false-safe decisions on 2,968 evaluation states.
- **Dynamic envelopes release real headroom:** 4.81 MWh of export and 2.50 MWh of import envelope-hours per household per year, where a fixed allowance valid in every hour would be zero.
- **The official pretrained GENCO helps where the network representation matches.** On case118, fine-tuning it beat training from scratch by 11.4 points (5 of 5 seeds).
- **Outlook.** Envelopes add up at the MV/LV transformer into the flexibility a distribution operator can offer upstream. A grid foundation model spanning LV, MV and HV needs the three changes identified here.

## 1. The problem

Distribution operators are replacing fixed connection limits with **dynamic operating envelopes (DOEs)**: a time-varying allowance that tells each household how much extra power it may import (to charge an EV or run a heat pump) or export (from rooftop PV) in each interval, without the network leaving its operational limits. Summed over a feeder, these envelopes are also the flexibility the operator can offer to the medium-voltage (MV) and transmission levels. Computing them means solving many power flows under uncertainty, and that is the kind of task a fast learned grid model could take over.

GENCO is such a model. It is a grid foundation model that represents the network as **balanced**, with one equivalent line per bus and no neutral conductor. It is available pretrained on transmission-level cases. European low-voltage (LV) networks are different: they are **three-phase four-wire** systems, where each home is connected between one phase and the neutral. The phases are loaded unevenly, and the neutral carries the resulting imbalance. One phase can drop below the voltage limit while another rises above it, which a balanced model cannot capture. We therefore asked: **what would it take to extend GENCO to four-wire LV networks, and can it then support household envelopes?**

![Four-wire network and a dynamic envelope](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig9_concept.png)

**Figure 1.** *(a) A four-wire LV network: each home sits on one phase and the neutral carries the imbalance. The balanced model GENCO works with lumps homes per bus and removes the neutral. (b) One day (28 July 2024) of envelopes on our feeder that passed the four-wire check: the allowance each household gets per hour, for import (up) and export (down).*

**Definitions**

- **Envelope.** One allowance in kW, the same for every household, on top of each household's own demand, for one hour and one direction (import or export).
- **Envelope-hours.** The allowance summed over the hours of a year, per household: headroom made available, not energy actually used.
- **OpenDSS.** An open-source distribution-system simulator. We use its full three-phase four-wire power flow as the ground truth.
- **Design scenarios and acceptance rule.** For each hour we draw 200 plausible household states. An envelope is issued only if OpenDSS finds the network within limits in at least 98% of them. This is the finite-sample rule for a 5% violation level at 95% confidence.
- **Screen, fallback, false-safe.** A screen is a fast model that labels a network state *safe*, *unsafe* or *unsure*; unsure states fall back to OpenDSS. A *false-safe* decision is a state the screen calls safe that OpenDSS finds unsafe. It is the error that matters, because it could let an unsafe envelope through.
- **Kron reduction.** Folding the neutral conductor into the phase conductors, so a four-wire line becomes a three-wire one. It is the first step of building a balanced model.
- **One-step four-wire sweep.** A single iteration of the backward/forward sweep, the standard radial power-flow method, on the four-wire network. It is cheap because it needs no iterations to convergence, and it serves as a rough physics estimate.
- **Residual head.** The output layer of a network that predicts a *correction* to a physics estimate rather than the voltages themselves.

## 2. Why the balanced representation fails at LV

![Balanced versus four-wire voltages](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig0_feeder.png)

**Figure 2.** *ENWL feeder 1. (a) Connection phase of each customer, and the heat pumps (HP), EVs and PV per phase in the headline household placement. (b, c) One validation state with 186 kW net export (net load per phase −79, −60 and −47 kW). The balanced model puts every customer between 240 and 247 V, inside the 216.2–253.0 V limits, so it reports no problem. The four-wire solution of the same injections ranges from 205 V on phase c to above 270 V on phase a: both limits are violated at the same moment.*

A balanced model cannot capture voltage deviations on the individual phases, such as a voltage rise on one phase and a drop on another. That voltage unbalance drives elevated neutral currents and shifts the neutral point. On this feeder, the consequence for envelopes is decisive: across every hour of the 36 evaluation days, not one positive balanced-model envelope met the four-wire feasibility target (0 of 864 export hours, 0 of 823 import hours).

## 3. Method

### 3.1 Making the GENCO architecture work on an LV feeder

We trained graphkit's GENCO architecture (`GNS_heterogeneous`, as distributed, which we call the *stock* model) on this feeder. It did worse than a trivial baseline that predicts each bus's average training-set voltage. We traced this to **three bottlenecks in GENCO at LV**:

1. **The edge encoder cannot see impedance magnitude.** It ends in a layer normalisation. With LV admittances of order 1e4, the normalisation strongly suppresses the magnitude of the admittances, which is what distinguishes a long cable from a short one: multiplying every admittance by 10 changes the edge embedding by only 0.2%. *Fix:* asinh-scaled admittance features.
2. **The physics loss collapses at LV numerical scales.** *Fix:* a residual head on a linear power-flow estimate, initialised at zero, so training starts exactly at the physics estimate.
3. **Line flows are ill-conditioned.** Flows computed from voltage differences turn a 1e-4 pu voltage error into an O(100%) flow error on short cables. *Fix:* flows from Kirchhoff's current law.

![GENCO development](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig3_genco_adaptation.png)

**Figure 3.** *Voltage error (mean absolute error, pu, log scale) of each modelling step. Left: 1,000 validation states, 15-epoch screening runs. Right: 1,000 validation states that include envelope-loaded conditions, final runs. The two panels are separate experiments, not one ablation sequence. Coral: fails; grey: baselines.*

The admittance features were necessary but not sufficient on their own. The decisive step was combining them with the physics estimate and residual head. On envelope-loaded states, the error fell to 1.0e-3 pu, 5.4× lower than the physics estimate alone.

We also found two bugs in the official GridFM tooling and handled both before any reported run. graphkit's best-model export could save the untrained epoch-0 weights instead of the best epoch; we fixed it and checked every reported export, including the stock model, against its last checkpoint. datakit's optimal-power-flow step raises an unpinned slack bus to its upper voltage bound, which was 1.5 pu with the loose bounds of our export. We fixed the slack voltage at 1.0 pu, so no reported data used 1.5 pu.

### 3.2 Making it phase-aware

What decides feasibility on this feeder is the voltage on each phase and the neutral-point displacement. A balanced representation cannot represent them, however long it is trained. We therefore extended the GENCO architecture with a full three-phase four-wire network model: phase and neutral conductors, each line described by its 4×4 impedance matrix, with no Kron reduction. Its message-passing layers (TransformerConv, an attention-based graph layer that weighs each neighbour using the line's features) carry those impedances. As in step 1, the model predicts a correction to a cheap physics estimate, here the one-step four-wire sweep.

### 3.3 GENCO pre-screening to reduce OpenDSS simulations

![Workflow](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig8_workflow.png)

**Figure 4.** *The workflow. The phase-aware GENCO screen decides clear cases; unsure states, and every envelope before it is issued, go through the full four-wire OpenDSS check.*

The final OpenDSS check is necessary: without it, a GENCO screen let one materially unsafe envelope through in 1,728 intervals. It verifies each envelope on its 200 design scenarios, not on all possible futures. On new scenarios, issued non-zero envelopes average 97.4% four-wire feasibility, but only 47% pass a one-sided 95% lower-bound test. Proving 95% with confidence needs more draws, and each extra draw adds OpenDSS simulations for every candidate envelope; 200 was the pre-registered design size.

## 4. Experimental setup

- **Network.** The open-source benchmark network ENWL1 (three-phase four-wire model, UK): 800 kVA, 200 single-phase customers, 4×4 cable impedances from BS 5467 data. It is modified to represent TT earthing, with the neutral earthed only at the transformer. This is an important modelling choice that makes the case unusual, and it sets the size of the neutral shift. The balanced model Kron-reduces the neutral, takes the positive-sequence impedance of each line, and lumps each customer as a three-phase load at its bus.
- **Households.** Load profiles allocated to households are derived from Fluvius 2024 smart-meter data. The technology mix is 30% heat pumps, 30% electric vehicles and 40% rooftop PV.
- **Operational limits.** Voltage may deviate by no more than −6% and +10% from 230 V, i.e. 216.2–253.0 V (UK ESQCR).
- **What the check covers.** Export envelopes are checked for overvoltage and for phase-conductor and transformer loading. Import envelopes are checked for undervoltage and the same loadings.
- **Scenarios.** The 200 design scenarios of an hour resample real meters of the same technology group at the *same date and time*. The spread therefore reflects how differently households behave at that moment, not seasonal variation.
- **Evaluation.** Envelopes were evaluated on 36 days spaced ten days apart through 2024 (10 January to 25 December), hourly, in both directions. Screens were evaluated on 2,968 four-wire power-flow states from those days, including states with envelopes applied, all under the same final check. Learned models were trained and calibrated on separate 2024 days. Baselines: a balanced sweep, a balanced GENCO screen, the one-step four-wire sweep, and an iterated four-wire sweep.
- **Evidence status.** The full OpenDSS envelope search and the GENCO screen were pre-registered. Everything else is exploratory, with protocols frozen before results. The evaluation days were also used during development. Each learned model was trained with one seed unless stated otherwise.

## 5. Results

### 5.1 Screening performance

![OpenDSS fallbacks](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig4a_fallback.png)

**Figure 5.** *Share of the 2,968 evaluation states each screen cannot decide and sends to OpenDSS. Every method runs under the same final check, and none made a false-safe decision. Teal: the phase-aware GENCO extension; coral: balanced-GENCO variants; grey: physics baselines.*

- **Fallbacks fall from 35.8% to 5.8%** relative to the one-step four-wire sweep, with **zero observed false-safe decisions**: six-fold fewer OpenDSS simulations.
- Phase-to-neutral voltage error falls by 91% (on the 2,699 converged states), and near-limit violation classification improves from 0.926 to 0.992 balanced accuracy.
- Balanced screens, whether a balanced sweep or balanced GENCO, must send 63% of states to OpenDSS. **Representing the phases is what makes screening useful.**
- Per state, the screen runs in 2.1 ms on an A100 GPU against 5.5 ms for our CPU four-wire sweep. These run on different hardware, so the main computational gain is the six-fold reduction in OpenDSS simulations.

### 5.2 Operational value

![Capacity](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/c02b3420e5f53ea9483e74e029253db8be44b93b/figures/fig1_capacity.png)

**Figure 6.** *Envelope-hours per household per year over the 36 evaluation days: a fixed allowance valid in every hour, dynamic envelopes that passed the four-wire check (bars with season-stratified day-bootstrap 95% intervals), and dynamic envelopes from the balanced model, which fail the four-wire check.*

Dynamic envelopes release **4.81 MWh of export** (interval 4.35–5.30) and **2.50 MWh of import** (2.11–2.87) envelope-hours per household per year. A single fixed allowance that must hold in every hour would be zero here, because in about half of all hours no positive allowance passes the check. Across five household-technology placements the result is 4.8–6.2 MWh export and 2.5–3.2 MWh import; the headline placement is the lowest. In practice, this is headroom an operator could offer for EV charging, heat pumps or PV export on the existing feeder. We do not estimate how much of it would be used, or its value for reinforcement deferral. These figures come from the full OpenDSS search; running the phase-aware screen through the full-year study is the next step. A learned envelope proposal checked once by OpenDSS kept 79% of the envelope-hours with 6.7% of the OpenDSS calls.

### 5.3 Transfer of GENCO representations

- **Balanced embeddings do not help a four-wire model.** Adding embeddings from our feeder-specific balanced GENCO to the phase-aware model *increased* fallbacks from 5.8% to 11.8% (+6.0 points [5.3, 6.7]). We did not isolate the cause. The embeddings describe a balanced state that disagrees with the four-wire one precisely in the stressed states where screening matters, so they add no usable information there.
- **The official pretrained GENCO helps on its native representation.** On case118 with 99 flexible export units, a network proposes the flexible dispatch and an AC optimal power flow (IPOPT) re-dispatches the rest. The reference is the IPOPT optimum. Fine-tuning the official `gridfm/genco-opf-datakit` checkpoint beat training from scratch at 100 labels by **+11.4 points of reference hosting [9.8, 12.5], in 5 of 5 seeds**, on 2,000 unseen scenarios. The gain also held at 10,000 labels in every seed, while at 1,000 labels it depended on the seed. Our pre-registered absolute target of 95% of the reference was narrowly missed (94.7%). ([log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/994d5d416a0c45fdbf6045c7103f3cd9c7519b53/docs/experiment_log.md))

### 5.4 Comparison with classical solvers

The iterated four-wire sweep is the accuracy leader on this feeder: 3.6e-6 pu error and 0.1% fallback, about 80× more accurate than the learned screen. The learned screen's value is different. It shows that the GENCO architecture can absorb phase-aware physics once the representation allows it. It reduces OpenDSS simulations six-fold relative to the estimate it corrects. And it can be amortised across networks and workflows where an equally strong specialised solver is not available.

## 6. Conclusions and outlook

1. **Balanced grid models, GENCO included, are not safe proxies for household envelopes on four-wire LV networks.**
2. **The GENCO architecture can be extended to the distribution level** with magnitude-preserving edge features, a physics residual, and an explicit phase-and-neutral network model. Used as a pre-screen, it cuts full four-wire simulations six-fold without an observed false-safe decision.
3. **Pretraining transfers when the representation matches** (case118), but it cannot supply the missing phases.
4. **On this feeder today, the right deployment is physics-first:** a learned screen to triage, and a full four-wire check to decide.

**Limitations.**
- The check assumes every household uses its full envelope; partial uptake, which can shift the neutral further, is not checked.
- Each direction is checked separately, so export hours are not checked for undervoltage on other phases.
- Neutral-conductor current and voltage unbalance are not checked.
- One benchmark feeder.

**Outlook.** The household envelopes on a feeder sum to a flexibility range at the MV/LV transformer. Carrying that range up to MV and transmission, in one model across voltage levels, needs what this entry identified: magnitude-preserving edge features, level-specific (phase-aware) representations, and a full physics check at every level.

## 7. Reproducibility

- **Run it yourself, no data access needed.** `python scripts/demo_neutral_wire.py` solves one synthetic household state on the public ENWL1 files with both models in a few seconds. The balanced model reports no customer outside the voltage limits. Four-wire OpenDSS reports 34, from 205 to 257 V, with 401 A in the transformer neutral, and phase and neutral currents above the cable ratings on some segments (up to 1.21× and 1.19×). The state is deliberately stressed.
- **Code, protocols and results** are at the release commit (Apache-2.0): numbered amendments, the [experiment log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/c02b3420e5f53ea9483e74e029253db8be44b93b/docs/experiment_log.md), run manifests with SHA-256 hashes, [claims and non-claims](https://github.com/ujohn33/genco-dynamic-envelopes/blob/c02b3420e5f53ea9483e74e029253db8be44b93b/docs/claims.md), and 84 automated tests. `scripts/reproduce_check.py` regenerates every headline number byte for byte from the recorded run files.
- **Not released, and why.** Until Fluvius confirms that its licence permits redistributing data derived from individual meters, we do not publish the LV checkpoints, four-wire labels or per-day run files. Their hashes are in the manifests, and everything needed to regenerate them is in the repository.

---

**Team:** Ander Martinez Alonso¹,³ (ander.martinez.alonso@vub.be), Evgenii Genov¹, Rémy Cleenwerck¹,². ¹ EVERGi, Vrije Universiteit Brussel; ² EnSy/Lemcko, Ghent University; ³ DTU Wind and Energy Systems, Technical University of Denmark. Built on GENCO with gridfm-graphkit 0.9.0.3 and gridfm-datakit 1.1.0.
