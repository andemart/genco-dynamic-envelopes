# GENCO meets the neutral wire: four-wire dynamic operating envelopes for low-voltage grids

*Taking GENCO to three-phase four-wire low-voltage distribution networks: what breaks, and how a phase-aware GENCO repairs it.*

**Ander Martinez Alonso¹,³, Evgenii Genov¹, Rémy Cleenwerck¹,²**  
¹ EVERGi, Vrije Universiteit Brussel · ² EnSy/Lemcko, Ghent University · ³ DTU Wind and Energy Systems, Technical University of Denmark  
**Repository:** [github.com/ujohn33/genco-dynamic-envelopes](https://github.com/ujohn33/genco-dynamic-envelopes)  
**Contact:** ander.martinez.alonso@vub.be  
*GridFM GENCO Challenge 2026 [1]. Submitted snapshot: [release 72e2b53](https://github.com/ujohn33/genco-dynamic-envelopes/tree/72e2b5395ae887c3df612d4c2f49d70e6fce6481); official-checkpoint experiments: [commit 994d5d4](https://github.com/ujohn33/genco-dynamic-envelopes/tree/994d5d416a0c45fdbf6045c7103f3cd9c7519b53) (branch `ander-case118`).*

---

## Abstract

*GENCO represents the power grid as balanced. We take it to the low-voltage (LV) distribution level, where every household is connected to a single phase. To our knowledge, this is the first application of GENCO to three-phase four-wire LV networks. We test it on household dynamic operating envelopes, on one benchmark feeder.*

- **On this feeder, a balanced model is not a usable proxy.** Across every hour of 36 evaluation days, not one positive envelope designed with the balanced model met the four-wire feasibility target.
- **A phase-aware GENCO makes learned screening useful.** As a pre-screen in front of OpenDSS, it cuts full four-wire simulations six-fold (fallbacks 35.8% → 5.8%), with zero observed false-safe decisions on 2,968 evaluation states.
- **The official pretrained GENCO helps on its native representation.** Fine-tuning the official checkpoint on case118 beat training from scratch by **+11.4 points [9.8, 12.5], in 5 of 5 seeds**, on 2,000 unseen scenarios.
- **Dynamic envelopes release headroom:** 4.81 MWh of export and 2.50 MWh of import envelope-hours per household per year, where a fixed allowance valid in every hour would be zero.
- **Toward integrated DSO–TSO tools (outlook).** Household envelopes add up at the MV/LV transformer into the flexibility a distribution system operator (DSO) can offer the transmission system operator (TSO). Our LV and case118 studies are separate demonstrations at the two ends of that chain. Together they identify what a single model across voltage levels would need.

## 1. Background and motivation

Distribution operators are replacing fixed connection limits with **dynamic operating envelopes (DOEs)** [5]. A DOE is a time-varying allowance that tells each household how much extra power it may import (EV charging, heat pumps) or export (rooftop PV) in each interval. Summed over a feeder, the envelopes are also the flexibility the DSO can offer upstream. Today DSOs and TSOs coordinate with separate tools built on separate physics. A grid foundation model covering every voltage level could become a shared layer, but only if it handles LV physics.

GENCO [2] is a grid foundation model, available pretrained on transmission cases. It represents the network as **balanced**: one equivalent line per bus, no neutral conductor. European LV networks are **three-phase four-wire**. Each home is connected between one phase and the neutral, the phases are loaded unevenly, and the neutral carries the imbalance. One phase can drop below the voltage limit while another rises above it. We asked: **what would it take to bring GENCO to four-wire LV networks, and can it then support household envelopes?**

![Four-wire network and a dynamic envelope](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig9_concept.png)

**Figure 1.** *(a) A four-wire LV network, and the balanced view GENCO works with: homes lumped per bus, neutral removed. (b) One day (28 July 2024) of envelopes on our feeder that passed the four-wire test, per household and hour, for import (up) and export (down).*

## 2. Problem definition

- **Envelope / envelope-hours.** One allowance in kW, the same for every household, on top of its own demand, per hour and direction. Summed over a year, it gives the headroom made available, not energy actually used.
- **The four-wire test (directional).** An envelope passes our frozen test when OpenDSS [6] finds at least 196 of 200 design scenarios within the limits that test covers. For **export**, those are overvoltage and phase-conductor and transformer loading. For **import**, they are undervoltage and the same loadings. Every household is assumed to use its full envelope at the same time. Opposite-side voltage, neutral-conductor current, partial uptake and future states are **not** certified. This is the finite-sample rule for a 5% violation level at 95% confidence.
- **Screen, fallback, false-safe.** A screen labels a network state as passing, failing or *unsure* with respect to that test; unsure states fall back to OpenDSS. A *false-safe* decision passes a state that OpenDSS fails.
- **Kron reduction** folds the neutral into the phase conductors. **The one-step four-wire sweep** is a single iteration of the standard radial backward/forward sweep on the four-wire network: a cheap, rough physics estimate.

![Balanced versus four-wire voltages](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig0_feeder.png)

**Figure 2.** *ENWL feeder 1. (a) Connection phase of each customer, and the heat pumps (HP), EVs and PV per phase in the headline household placement. (b, c) One validation state with 186 kW net export (−79, −60 and −47 kW on phases a, b and c). The balanced model puts every customer between 240 and 247 V, inside the 216.2–253.0 V limits. The four-wire solution of the same injections ranges from 205 V on phase c to above 270 V on phase a: both limits are violated at once.*

A balanced model cannot capture voltage deviations on individual phases, nor the neutral-point displacement they cause. On this feeder that is decisive. Across every hour of the 36 evaluation days, not one positive balanced-model envelope met the four-wire feasibility target (0 of 864 export hours, 0 of 823 import hours).

## 3. Methodology

### 3.1 Making the GENCO architecture work on an LV feeder

We trained the GENCO architecture in gridfm-graphkit [3] (`GNS_heterogeneous` as distributed, the *stock* model) on this feeder. It did worse than a trivial baseline that predicts each bus's average training-set voltage. We traced this to **three bottlenecks of GENCO at LV**:

1. **The edge encoder cannot see impedance magnitude.** It ends in a layer normalisation that, with LV admittances of order 1e4, strongly suppresses the magnitude that distinguishes a long cable from a short one. Multiplying every admittance by 10 changes the edge embedding by only 0.2%. *Fix:* asinh-scaled admittance features.
2. **The physics loss collapses at LV numerical scales.** *Fix:* a residual head that predicts a correction to a linear power-flow estimate, initialised at zero, so training starts exactly at the physics estimate.
3. **Line flows are ill-conditioned.** Flows from voltage differences turn a 1e-4 pu voltage error into an O(100%) flow error on short cables. *Fix:* flows from Kirchhoff's current law.

![GENCO development](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig3_genco_adaptation.png)

**Figure 3.** *Voltage error (mean absolute error, pu, log scale) of each modelling step. Left: 1,000 validation states, 15-epoch screening runs. Right: 1,000 validation states including envelope-loaded conditions, final runs. Separate experiments, not one ablation sequence. Coral: fails; grey: baselines.*

The admittance features were necessary but not sufficient. The decisive step was combining them with the physics estimate and residual head: on envelope-loaded states the error fell to 1.0e-3 pu, 5.4× below the estimate alone. We also found and fixed two bugs in the GridFM tooling before any reported run. graphkit's best-model export could save the untrained epoch-0 weights, so we checked every reported export, including the stock model. datakit's optimal-power-flow step raises an unpinned slack bus to its upper voltage bound, so we fixed the slack at 1.0 pu.

### 3.2 A phase-aware, four-wire GENCO

A balanced representation cannot carry per-phase voltages or the neutral-point displacement, however long it is trained. We therefore built a **four-wire GENCO**: GENCO's encode–process–decode architecture and TransformerConv processor on an explicit phase-and-neutral representation (4×4 line impedance matrices, no Kron reduction). Because graphkit's pipeline cannot carry a four-wire state, it is implemented in this repository with its own training pipeline. As in 3.1, it predicts a correction to a cheap physics estimate, here the one-step four-wire sweep.

### 3.3 GENCO pre-screening to reduce OpenDSS simulations

![Workflow](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig8_workflow.png)

**Figure 4.** *The screen decides clear cases. Unsure states, and every envelope before it is issued, go through the four-wire OpenDSS test of section 2.*

The final test is necessary: without it, a GENCO screen let one materially failing envelope through in 1,728 intervals. On new scenarios, issued non-zero envelopes average 97.4% feasibility under the same directional test. But only 47% pass a one-sided 95% lower-bound test, because 200 draws are too few to prove 95%. More draws would add OpenDSS simulations for every candidate envelope; 200 was the pre-registered design size.

## 4. Experiment design

- **Network.** The open-source benchmark ENWL1 from Electricity North West's Low Voltage Network Solutions project [10] (three-phase four-wire OpenDSS model, UK): 800 kVA, 200 single-phase customers, 4×4 cable impedances from BS 5467 data, modified to TT earthing with the neutral earthed only at the transformer. The balanced model Kron-reduces the neutral, takes each line's positive-sequence impedance, and lumps each customer as a three-phase load at its bus.
- **Households.** Fluvius open data: 2,400 anonymised residential digital meters, quarter-hourly, 2024, in eight technology groups [4]. The mix is 30% heat pumps, 30% EVs and 40% rooftop PV, so every household has at least one of them.
- **Limits.** Voltage within −6%/+10% of 230 V, i.e. 216.2–253.0 V (UK ESQCR [7]). The checks are directional, as defined in section 2.
- **Scenarios.** The 200 design scenarios of an hour resample real meters of the same technology group at the *same date and time*, so the spread reflects household diversity at that moment, not seasonal variation.
- **Evaluation.** Envelopes on 36 days spaced ten days apart through 2024 (10 January to 25 December), hourly, both directions. Screens on 2,968 four-wire states from those days, including envelope-loaded states, all under the same final test. Learned models were trained on separate 2024 days. Baselines: a balanced sweep, a balanced GENCO screen, and one-step and iterated four-wire sweeps.
- **Evidence status.** The full OpenDSS envelope search and the GENCO screen were pre-registered. Everything else is exploratory, with protocols frozen before results. The evaluation days were also used in development. Each learned model uses one seed unless stated otherwise.

## 5. Results

### 5.1 Screening performance

![OpenDSS fallbacks](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig4a_fallback.png)

**Figure 5.** *Share of the 2,968 evaluation states each screen sends to OpenDSS, all under the same final test; none made a false-safe decision. Teal: the phase-aware GENCO; coral: balanced-GENCO variants; grey: physics baselines.*

- **Fallbacks fall from 35.8% to 5.8%** relative to the one-step four-wire sweep, with **zero observed false-safe decisions**: six-fold fewer OpenDSS simulations.
- Phase-to-neutral voltage error falls by 91% (on the 2,699 converged states), and near-limit violation classification improves from 0.926 to 0.992 balanced accuracy.
- Balanced screens, a balanced sweep or balanced GENCO, send 63% of states to OpenDSS. These balanced baselines support the importance of phase information. The comparison between the one-step sweep and the GNN isolates the value of the learned residual over a prior that is already phase-aware.
- Adding embeddings from our feeder-specific balanced GENCO *raised* fallbacks to 11.8% (+6.0 points [5.3, 6.7]). We did not isolate why.
- Per state, the screen takes 2.1 ms on an A100 GPU against 5.5 ms for our CPU four-wire sweep. The hardware differs, so the main computational gain is the six-fold reduction in OpenDSS simulations.

### 5.2 The official pretrained GENCO

On case118 with 99 flexible export units, a network proposes the flexible dispatch and an AC optimal power flow (PowerModels [9] with IPOPT [8]) re-dispatches the rest; the reference is the IPOPT optimum. Fine-tuning the official [`gridfm/genco-opf-datakit`](https://huggingface.co/gridfm/genco-opf-datakit) checkpoint beat training from scratch at 100 labels by **+11.4 points of reference hosting [9.8, 12.5], in 5 of 5 seeds**, on 2,000 unseen scenarios. The gain also held at 10,000 labels in every seed; at 1,000 labels it depended on the seed. Our pre-registered absolute target of 95% of the reference was narrowly missed (94.7%). This is transfer to a new flexible-hosting objective on the checkpoint's native case118 network, not transfer to an unseen network, and overlap with the checkpoint's original training profiles is not ruled out. ([log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/994d5d416a0c45fdbf6045c7103f3cd9c7519b53/docs/experiment_log.md))

### 5.3 Operational value

![Capacity](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/72e2b5395ae887c3df612d4c2f49d70e6fce6481/figures/fig1_capacity.png)

**Figure 6.** *Envelope-hours per household per year over the 36 evaluation days: a fixed allowance valid in every hour, dynamic envelopes that passed the four-wire test (with season-stratified day-bootstrap 95% intervals), and balanced-model envelopes, which fail it.*

Dynamic envelopes release **4.81 MWh of export** (4.35–5.30) and **2.50 MWh of import** (2.11–2.87) envelope-hours per household per year. A fixed allowance valid in every hour would be zero, because in about half of all hours no positive allowance passes the test. Across five household-technology placements the result is 4.8–6.2 MWh export and 2.5–3.2 MWh import; the headline placement is the lowest. This is headroom an operator could offer on the existing feeder, subject to the test's scope; we do not estimate its uptake or value. These figures come from the full OpenDSS search; running the phase-aware screen through the full-year study is the next step.

## 6. Conclusion

1. **On this feeder and envelope formulation, balanced grid models, GENCO included, are not usable proxies for household envelopes.**
2. **A phase-aware GENCO makes learned screening useful at LV.** The ingredients are magnitude-preserving edge features, a physics residual and an explicit phase-and-neutral representation. As a pre-screen it cuts full four-wire simulations six-fold without an observed false-safe decision.
3. **The official pretrained GENCO helps on its native case118 representation** (5 of 5 seeds), but pretraining cannot supply missing phases.
4. **Physics stays in charge.** The iterated four-wire sweep remains the accuracy leader (3.6e-6 pu, 0.1% fallback, about 80× more accurate). The learned screen's value is fewer OpenDSS simulations, and amortisation where an equally strong specialised solver is not available.

**Limitations.**
- The test is directional and assumes full simultaneous uptake. In the recorded runs, 60 of 472 positive export intervals (12.7%) showed an undervoltage, and 110 of 444 positive import intervals (24.8%) an overvoltage, in at least one fresh scenario. The extremes were 0.741 and 1.118 pu. The runs store only extrema, so we cannot say how often these exceed 5% of scenarios.
- Neutral-conductor current and voltage unbalance are not checked, and partial uptake can shift the neutral further.
- One benchmark feeder with TT earthing and one technology mix. The evaluation days were also used in development.

**Outlook: toward integrated DSO–TSO tools.** We have not run a coupled study. The next step would be to aggregate the envelopes into a P/Q flexibility range per MV/LV transformer, pass it through an MV stage (exploratory MV work on case33bw is in the repository), and on to transmission. One model across these levels needs what this entry identified: magnitude-preserving edge features, level-specific representations (phase-aware at LV, positive-sequence at HV), and a physics check at every level.

## Code and data

- **Public demo.** `python scripts/demo_neutral_wire.py` solves one deliberately stressed synthetic state on the public ENWL1 files with both models in a few seconds. The balanced model shows no customer outside the limits. Four-wire OpenDSS shows 34, from 205 to 257 V, with 401 A in the transformer neutral and currents above cable ratings on some segments.
- **Code, protocols and results** are at the release commit (code under Apache-2.0; third-party data under their own terms, see `DATA_LICENSES.md`). They include numbered amendments, the [experiment log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/72e2b5395ae887c3df612d4c2f49d70e6fce6481/docs/experiment_log.md), run manifests with SHA-256 hashes, [claims and non-claims](https://github.com/ujohn33/genco-dynamic-envelopes/blob/72e2b5395ae887c3df612d4c2f49d70e6fce6481/docs/claims.md), and 84 tests (on a fresh clone 74 run and 10 skip, because they need the Fluvius download or retained artifacts).
- **Reproduction.** `scripts/reproduce_check.py` deterministically regenerates the aggregate reports from locally retained run artifacts, and verifies the A100 evaluation against its recorded hash. The source datasets are public (ENWL1 in the repository, Fluvius open data [4]). The derived run files and LV checkpoints are not yet redistributed, so a fresh clone cannot currently reproduce every headline result end to end.

*Built on GENCO with gridfm-graphkit 0.9.0.3 and gridfm-datakit 1.1.0 [3].*

## References

1. GridFM GENCO Challenge 2026. https://huggingface.co/spaces/gridfm/genco-challenge
2. GENCO – A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis. arXiv:2608.09921, 2026. https://arxiv.org/abs/2608.09921
3. gridfm-graphkit and gridfm-datakit (OpenGridFM): https://github.com/gridfm/gridfm-graphkit and https://github.com/gridfm/gridfm-datakit. Official checkpoint: https://huggingface.co/gridfm/genco-opf-datakit
4. Fluvius Open Data. Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden voor een volledig jaar. https://opendata.fluvius.be/explore/assets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/
5. K. Petrou, A. T. Procopiou, L. Gutierrez-Lagos, M. Z. Liu, L. F. Ochoa, T. Langstaff, J. M. Theunissen. Ensuring distribution network integrity using dynamic operating limits for prosumers. *IEEE Transactions on Smart Grid* 12(5), 2021. https://doi.org/10.1109/TSG.2021.3081371
6. EPRI. OpenDSS: open-source distribution system simulator. https://www.epri.com/pages/sa/opendss
7. The Electricity Safety, Quality and Continuity Regulations 2002 (UK SI 2002/2665). https://www.legislation.gov.uk/uksi/2002/2665
8. A. Wächter, L. T. Biegler. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. *Mathematical Programming*, 2006. https://doi.org/10.1007/s10107-004-0559-y
9. C. Coffrin, R. Bent, K. Sundar, Y. Ng, M. Lubin. PowerModels.jl: an open-source framework for exploring power flow formulations. *PSCC* 2018. https://doi.org/10.23919/PSCC.2018.8442948
10. Electricity North West. Low Voltage Network Solutions (Low Carbon Networks Fund project). https://www.enwl.co.uk/future-energy/innovation/smaller-projects/low-carbon-networks-fund/low-voltage-network-solutions/
