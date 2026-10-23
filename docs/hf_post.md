# GENCO meets the neutral wire: four-wire-gated dynamic operating envelopes for low-voltage grids

*Extending GENCO to three-phase four-wire low-voltage networks: what breaks, and how a phase-aware extension repairs it.*

**Ander Martinez Alonso¹,³, Evgenii Genov¹, Rémy Cleenwerck¹,²**  
¹ EVERGi research group, Vrije Universiteit Brussel · ² EnSy/Lemcko research group, Ghent University · ³ Department of Wind and Energy Systems, Technical University of Denmark  
*GridFM GENCO Challenge 2026. Code, protocols, figures and results: [release 49f3cd9](https://github.com/ujohn33/genco-dynamic-envelopes/tree/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4); official-checkpoint experiments: [commit 994d5d4](https://github.com/ujohn33/genco-dynamic-envelopes/tree/994d5d416a0c45fdbf6045c7103f3cd9c7519b53). Contact: ander.martinez.alonso@vub.be.*

---

## 1. The problem

Distribution operators are replacing fixed connection limits with **dynamic operating envelopes**: a time-varying allowance that tells each household how much extra it may import (EV charging, heat pumps) or export (rooftop PV) in each interval without breaking the network's limits. Computing an envelope means solving many power flows under uncertainty.

On European low-voltage (LV) networks those power flows are **four-wire**. Each home is connected between one phase and the neutral, so the three phases load unevenly and the neutral carries the imbalance. One phase can sag below the voltage limit while another rises above it. GENCO, like most grid models, represents the network as **balanced**: one equivalent line per bus, with the neutral removed. We asked: **can GENCO support household envelopes, and what would it take to extend it to four-wire distribution networks?**

![Four-wire network and a dynamic envelope](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig9_concept.png)

*(a) The four-wire network, and the balanced view GENCO works with. (b) One day of four-wire-verified envelopes on our feeder: the allowance each household gets per hour, for import (up) and export (down).*

**Definitions used throughout**

- **Envelope.** One allowance in kW, the same for every household, on top of each household's own passive demand, for one hour and one direction (import or export).
- **Envelope-hours.** The allowance summed over the hours of a year, per household. It measures headroom made available, not energy actually used.
- **Design scenarios and the acceptance rule.** For each hour we draw 200 plausible household states. An envelope is issued only if the full four-wire power flow (OpenDSS) finds the network within limits in at least 98% of them. This is the finite-sample rule for a 5% violation level at 95% confidence.
- **Screen, fallback, false-safe.** A screen is a fast model that labels a network state *safe*, *unsafe* or *unsure*. Unsure states fall back to OpenDSS. A *false-safe* decision is a state the screen calls safe that OpenDSS finds unsafe: the error that matters.
- **One-step prior.** A single iteration of the four-wire backward/forward sweep: a cheap physics estimate of the four-wire voltages.

## 2. Highlights

- **A balanced model is unsafe here.** Across every hour of 36 evaluation days, not one positive envelope designed with the balanced model met the four-wire feasibility target (0 of 864 export hours, 0 of 823 import hours).
- **A phase-aware GENCO extension makes screening useful.** A GENCO-style GNN that corrects the one-step prior cut OpenDSS fallbacks from 35.8% to **5.8%**, with **zero observed false-safe decisions** on 2,968 evaluation states. Balanced screens, GENCO included, fall back on 63%.
- **Dynamic envelopes release real headroom.** 4.81 MWh of export and 2.50 MWh of import envelope-hours per household per year, where a constant allowance valid in every hour releases nothing.
- **Official GENCO pretraining helps where the representation matches.** On case118, fine-tuning the official checkpoint beat training from scratch by +11.4 points [9.8, 12.5] in 5 of 5 seeds at 100 labels. Both preregistered absolute targets were still missed.
- **Physics stays in charge.** An iterated four-wire sweep remains the most accurate method, and every issued envelope passes a final OpenDSS check.

## 3. Method and design choices

### 3.1 Why the four-wire check is the reference

The balanced and four-wire models can disagree completely on the same household injections.

![Balanced versus four-wire voltages](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig0_feeder.png)

*ENWL feeder 1. (a) Customer phase, and the technologies per phase in the headline household placement. (b, c) One validation state with 186 kW net export (net load per phase −79, −60 and −47 kW): the balanced model puts every customer at 240–247 V; the four-wire solution ranges from 205 V (phase c) to above 270 V (phase a).*

We therefore treat the full four-wire OpenDSS model as the reference, and every issued envelope must pass it. Learned models are used only to decide *which* states need that check.

### 3.2 Step 1: make the GENCO architecture work on an LV feeder

We trained graphkit's GENCO architecture (`GNS_heterogeneous`) on this feeder. Out of the box it did worse than simply predicting the average training voltage. We traced three bottlenecks:

1. **Impedance magnitude.** The edge encoder ends in a LayerNorm. With LV admittances of order 1e4, it strongly suppresses the magnitude that distinguishes a long cable from a short one: multiplying every admittance by 10 changes the edge embedding by only 0.2%. *Fix:* asinh-scaled admittance features.
2. **Physics-loss collapse** at LV numerical scales. *Fix:* the model predicts a correction (a "residual") to a linear power-flow estimate. The correction starts at zero, so training starts exactly at the physics estimate.
3. **Ill-conditioned flows.** Flows computed from voltage differences turn a 1e-4 pu voltage error into an O(100%) flow error on short cables. *Fix:* flows from Kirchhoff's current law.

![GENCO development](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig3_genco_adaptation.png)

The admittance features were necessary but not sufficient. The decisive step was combining them with the physics estimate and residual: voltage error 1.0e-3 pu, 5.4× better than the estimate alone. (The two panels are separate runs, not one ablation sequence.) We also found two tooling issues and handled both before any reported run. graphkit's best-model export could keep the epoch-0 weights; we fixed it and checked every reported export, including the stock baseline. datakit's OPF step raises an unpinned slack bus to its upper voltage bound; we pin it at 1.0 pu.

### 3.3 Step 2: make it phase-aware

What decides feasibility on this feeder is the voltage on each phase and the shift of the neutral. A balanced representation averages exactly that away, so more training cannot recover it. We therefore built a GENCO-style encode–process–decode GNN on the explicit four-wire network, with phase and neutral conductors and no Kron reduction, and an edge-conditioned TransformerConv processor. As in step 1, it predicts a zero-initialised residual on a cheap physics estimate, here the one-step four-wire prior.

### 3.4 The workflow

![Workflow](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig8_workflow.png)

The screen decides clear cases; unsure states and every issued envelope go through OpenDSS. Why a final gate at all: without it, a GENCO screen let through one materially unsafe envelope in 1,728 intervals. The gate is a **design-set check, not a certificate of future safety**. On new scenarios, issued non-zero envelopes average 97.4% four-wire feasibility, but only 47% pass a one-sided 95% lower-bound test, because 200 draws are too few to prove 95%.

## 4. Experimental setup

- **Network.** ENWL feeder 1, a public UK four-wire LV benchmark: 800 kVA, 200 single-phase customers. Cables have 4×4 phase-and-neutral impedances (BS 5467 data), with the neutral earthed at the transformer only. The balanced model Kron-reduces the neutral, takes the positive-sequence impedance, and lumps each customer as a three-phase load at its bus.
- **Households.** Fluvius 2024 smart-meter profiles, with 30% heat pump, 30% EV and 40% PV. UK statutory voltage limits are 216.2–253.0 V.
- **Scenarios.** For each hour, the 200 design scenarios resample real meters of the same technology group at the same date and time. This is an empirical uncertainty model, not a day-ahead forecast.
- **What the check covers.** Export envelopes: overvoltage, phase-conductor and transformer loading. Import envelopes: undervoltage and the same loadings.
- **Evaluation.** Envelopes on 36 evaluation days of 2024 (hourly, both directions). Screens on 2,968 four-wire power-flow states from those days, all under the same final gate. Baselines: the balanced sweep, balanced GENCO, the one-step prior, and an iterated four-wire sweep.
- **Evidence status.** The full OpenDSS search and the GENCO screen were pre-registered. Everything else is exploratory, with protocols frozen before results. The evaluation days were also used during development. Each learned model was trained once unless stated otherwise.
- **Official checkpoint.** case118 with 99 flexible export units; the official `gridfm/genco-opf-datakit` checkpoint fine-tuned versus trained from scratch, at 100–10,000 labels. A network proposes the flexible dispatch, and an IPOPT corrective redispatch checks it. The reference is the IPOPT optimum.

## 5. Results

### 5.1 Screening: what the phase-aware extension delivers

![OpenDSS fallbacks](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig4a_fallback.png)

- **Fallbacks fall from 35.8% to 5.8%** relative to the one-step prior, with **zero observed false-safe decisions**.
- Phase-to-neutral voltage error falls by 91% (on the 2,699 converged states), and near-limit violation classification improves from 0.926 to 0.992 balanced accuracy.
- Balanced screens, whether a balanced sweep or balanced GENCO, must send 63% of states to OpenDSS. Representing the phases is what makes screening useful.

### 5.2 Operational value

![Capacity](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/figures/fig1_capacity.png)

With every envelope passing the full four-wire check, dynamic envelopes release **4.81 MWh of export** (day-bootstrap interval 4.35–5.30) and **2.50 MWh of import** (2.11–2.87) envelope-hours per household per year. A constant allowance valid in every hour releases nothing, because in about half of all hours no positive envelope passes at all. Across five household-technology placements the capacity is 4.8–6.2 MWh export and 2.5–3.2 MWh import; the headline placement is the lowest. These figures come from the full OpenDSS search; the phase-aware screen has not yet been run through the annual study. A learned envelope proposal checked once by OpenDSS kept 79% of the envelope-hours with 6.7% of the OpenDSS calls.

### 5.3 Where pretraining helped, and where it did not

- **Balanced embeddings into a four-wire model: no help.** Adding embeddings from our feeder-specific balanced GENCO to the phase-aware GNN increased fallbacks from 5.8% to 11.8% (+6.0 points [5.3, 6.7]). A balanced representation does not carry four-wire information.
- **Official pretraining on its native representation: real help, short of our bar.** On case118, fine-tuning beat training from scratch by +14.2, +6.9 and +8.4 points of reference hosting at 100, 1,000 and 10,000 labels. Seed replicates confirmed the gain at 100 and 10,000 labels in every seed; at 1,000 labels the outcome depended on the seed (−7.1 to +17.3). A frozen follow-up at 100 labels, with five seeds on 2,000 unseen scenarios, replicated the gain: **+11.4 points [9.8, 12.5], 5 of 5 seeds**. Both preregistered absolute targets were missed: 89.4% (first seed, 1,000 labels) and 94.7% (follow-up) of the IPOPT reference, against a 95% bar. ([log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/994d5d416a0c45fdbf6045c7103f3cd9c7519b53/docs/experiment_log.md))

Pretraining transfers when the representation matches. It does not substitute for a representation that is missing.

### 5.4 What remains classical

The iterated four-wire sweep is the accuracy leader on this feeder: 3.6e-6 pu error and 0.1% fallback, about 80× more accurate than the GNN. End to end, the GNN takes 2.1 ms per state on an A100 GPU and our sweep 5.5 ms on a CPU; that comparison mixes hardware and an unoptimised sweep, so we make no speed claim.

## 6. Conclusions

1. **Balanced grid models, GENCO included, are not safe proxies for household-level envelopes on four-wire networks.**
2. **The GENCO architecture can be extended to the distribution level** with magnitude-preserving edge features, a physics residual and an explicit phase-and-neutral graph. The extension cuts expensive four-wire checks six-fold relative to the physics estimate it corrects.
3. **Pretraining helps when the representation matches**, as on case118, but cannot supply the missing phases.
4. **On this feeder, today, the right deployment is physics-first:** a learned screen to triage, and a full four-wire gate to decide. The learned part's value grows on networks and workflows where an equally strong specialised solver is not available.

**Limitations.**
- The check assumes every household uses its full envelope; partial uptake, which can shift the neutral further, is not checked.
- Each direction is checked separately, so export hours are not checked for undervoltage on other phases. Neutral-conductor current and voltage unbalance are not checked.
- One benchmark feeder; no claims about other feeders or years, delivered energy, or economic value.

**Next step.** The household envelopes on a feeder sum to a flexibility range at the MV/LV transformer. Carrying that range up through MV to transmission needs what this entry identified: magnitude-preserving edge features, level-specific representations, and a full physics gate at every tier.

## 7. Reproducibility

- **Run it yourself, no data access needed.** `python scripts/demo_neutral_wire.py` solves one synthetic household state on the public ENWL1 files with both models in a few seconds. The balanced model reports 0 customers outside the limits; four-wire OpenDSS reports 34, from 205 to 257 V, with 401 A in the transformer neutral.
- **Code, protocols and results** are at the release commit (Apache-2.0): numbered amendments, the [experiment log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/docs/experiment_log.md), run manifests with SHA-256 hashes, [claims and non-claims](https://github.com/ujohn33/genco-dynamic-envelopes/blob/49f3cd9e60800252fc3b28f0c6898a9d3e6853d4/docs/claims.md), and 84 automated tests. `scripts/reproduce_check.py` regenerates every headline number byte for byte from the recorded run files.
- **Not released, and why.** Until Fluvius confirms that its licence permits redistributing data derived from individual meters, we do not publish the LV checkpoints, four-wire labels or per-day run files. Their hashes are in the manifests, and everything needed to regenerate them is in the repository.

---

**Team:** Ander Martinez Alonso¹,³ (ander.martinez.alonso@vub.be), Evgenii Genov¹, Rémy Cleenwerck¹,². ¹ EVERGi research group, Vrije Universiteit Brussel; ² EnSy/Lemcko research group, Ghent University; ³ Department of Wind and Energy Systems, Technical University of Denmark. Built on GENCO with gridfm-graphkit 0.9.0.3 and gridfm-datakit 1.1.0.
