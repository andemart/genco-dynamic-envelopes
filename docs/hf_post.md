# Four-wire-verified dynamic operating envelopes with GENCO: where a grid foundation model helps on LV networks, and where it does not

*GridFM GENCO Challenge 2026. Ander Martinez Alonso, Evgenii Genov (EVERGi, VUB) and Remy Cleerckewer (VUB, UGent). Code, protocols, figures and result files: [github.com/ujohn33/genco-dynamic-envelopes @ 1c172c82e54edeeb17b67d897f1b03ebe68fcac2](https://github.com/ujohn33/genco-dynamic-envelopes/tree/1c172c82e54edeeb17b67d897f1b03ebe68fcac2). Contact: ander.martinez.alonso@vub.be.*

---

## TL;DR

1. **Balanced GENCO fails on four-wire LV physics.** On a four-wire LV benchmark feeder, customers sit on single phases and the neutral carries current. A balanced (positive-sequence) model, which is what GENCO is, misses this and issues envelopes that are badly unsafe.
2. **We diagnosed why and built a verified workflow around it.** Three LV failure modes of the GENCO architecture were found and fixed, and every envelope issued by the final workflow passes an exact four-wire OpenDSS check.
3. **A four-wire learned residual helps relative to a cheap prior.** A GENCO-style GNN on the explicit four-wire graph cuts exact-solver fallbacks from 35.8% to 5.8% relative to a one-step four-wire sweep, with no observed false-safe decision.
4. **Classical iterated physics still wins, and balanced GENCO embeddings do not transfer.** An iterated four-wire sweep is about 80× more accurate, and adding embeddings from our balanced GENCO model made every learned arm no better or worse.

---

## 1. The problem: envelopes on a four-wire feeder

A **dynamic operating envelope** (DOE) gives each household a time-varying import and export allowance instead of a fixed connection limit. Computing one is a search over power flows, and on European LV feeders those power flows are **four-wire**.

![Feeder](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/figures/fig0_feeder.png)

*ENWL feeder 1. (a) Customer phase. (b, c) One validation state with 186 kW net export. The balanced model puts every customer at 240–247 V; the four-wire solution of the same injections ranges from 205 V (phase c) to above 270 V (phase a).*

- **Network.** The ENWL1 four-wire LV benchmark (UK, 800 kVA, 200 single-phase customers; 397 LV buses in the four-wire graph, 398 in the balanced model, which adds the 11 kV source bus). It is a benchmark feeder, not an observed Flemish one.
- **Households.** Fluvius 2024 smart-meter profiles; technology mix 30% heat pump, 30% EV, 40% PV; UK statutory limits 216.2–253.0 V.
- **Uncertainty model.** For each interval, 200 scenarios resample real meters of the same technology group at the same date and time. This is an empirical uncertainty model; it does not test a day-ahead forecast model or distribution shift.
- **Envelope.** Uniform per household, optional and incremental above the passive demand. It is issued only if the exact four-wire model finds it feasible in **≥ 98% of the 200 design scenarios** (the finite-sample rule for ε = β = 0.05).

## 2. The verified workflow and what it releases

```
Fluvius meters → 200 scenarios per interval → proposal → exact four-wire OpenDSS check → envelope
                                                (exact search | GENCO-screened | learned)
```

Over **36 test-split days** of 2024:

![Capacity](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/figures/fig1_capacity.png)

- **Dynamic, four-wire-verified envelopes:** 4.81 MWh export (season-stratified day-bootstrap interval 4.35–5.30) and 2.50 MWh import (2.11–2.87) envelope-hours per household per year. These are envelope-hours, not delivered energy.
- **Constant add-on:** no positive constant optional allowance above the passive demand passes in every sampled interval. This is narrower than a conventional fixed connection capacity, and we do not claim every static connection policy yields zero.
- **Balanced model:** its envelopes are 2–3× larger and four-wire feasible in only 2% (export) and 24% (import) of draws on a summer day.
- **What "verified" means.** On fresh draws, non-zero exact envelopes reach 97.4% mean four-wire feasibility, but only 47% pass a one-sided 95% lower-bound test: 200 verification draws are too few to prove 95%. This is not certified or unconditional safety.
- **Robustness (exploratory).** Across five household-technology placements, capacity is 4.8–6.2 MWh (export) and 2.5–3.2 MWh (import); the headline placement is the lowest. In an idealised sensitivity on two placements, connecting HP/EV homes three-phase raises capacity about 2–2.5×, so single-phase connection is a major contributor to congestion here.

**Where the capacity sits.** Export envelopes close around midday from spring to autumn, when rooftop PV already pushes phase voltages up, and import envelopes vanish on winter evenings, when heat pumps and EVs load the single-phase circuits. In about 45% (export) and 49% (import) of test-split hours no positive envelope can be verified at all, so a single constant add-on that must hold in every hour is zero.

![Envelope heatmap](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/7a88cb022cb35ab675b0ca2613d35099cfb4f038/figures/fig6_envelope_heatmap.png)

*Four-wire-verified envelope per household for every hour of the 36 test-split days (rows by date). Grey: no positive envelope verified.*

**Balanced is unsafe in every hour, not just on one day.** Checking the balanced model's envelopes against the four-wire model over all test-split hours (a descriptive analysis on the same days): not one positive balanced envelope meets 0.95 four-wire feasibility (0 of 864 export, 0 of 823 import hours), and in about 380–390 hours per direction the four-wire check admits no envelope at all while the balanced model issues one.

![Balanced-model safety](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/7a88cb022cb35ab675b0ca2613d35099cfb4f038/figures/fig7_balanced_safety.png)

*(a, b) Balanced-model envelope against the four-wire-verified envelope for the same hour; dashed line = equal. (c) Four-wire feasibility on fresh draws of every non-zero envelope; dotted line = 0.95.*

**Evidence status.** The exact search (B) and the GENCO-screened search (C) were pre-registered. Everything else (C+, placements, spatial envelopes, envelope proposals, the four-wire extension) is exploratory, with protocols frozen before results were produced. The test dates were also used during development.

## 3. Making the GENCO architecture work on LV

Unless stated otherwise, "GENCO" here is graphkit's GENCO architecture (`GNS_heterogeneous`) **trained from scratch on this feeder**, not the official pretrained weights.

![Adaptation](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/figures/fig3_genco_adaptation.png)

The stock model did worse than predicting the training mean. Three causes, each fixed:

1. **Impedance blindness.** The edge encoder ends in a LayerNorm. With LV admittances of order 1e4, the encoder is positively homogeneous and LayerNorm cancels |Y|: the model cannot tell a long cable from a short one. *Fix:* asinh-scaled admittance features.
2. **Physics-loss collapse** at LV scales. *Fix:* a linear power-flow prior with a zero-initialised, scaled correction head.
3. **Ill-conditioned flows.** Flows from voltage differences turn a 1e-4 pu voltage error into an O(100%) flow error on short cables. *Fix:* Kirchhoff-current-law flows.

The result is a voltage MAE of 1.0e-3 pu on envelope states, 5.4× better than the prior alone; five classical fixed-point iterations are still better (1.2e-4). The two panels come from separate runs on different state sets, not one ablation sequence. We also found two tooling bugs: graphkit's best-model export silently kept epoch-0 weights, and datakit pinned the slack at 1.5 pu.

## 4. The four-wire extension, and an honest benchmark

GENCO is balanced, and on this feeder what binds is single-phase, four-wire physics. We therefore built a GENCO-style encode–process–decode GNN on the explicit four-wire graph (conductors a, b, c, n; full 4×4 impedances; edge-conditioned TransformerConv), predicting a zero-initialised residual on a **one-step four-wire sweep**.

![Four-wire](https://raw.githubusercontent.com/ujohn33/genco-dynamic-envelopes/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/figures/fig4_fourwire.png)

Under the same exact gate, on the test split (fallbacks on 2,968 screening states, voltage error on the 2,699 converged ones; one training seed per model, so intervals exclude training variability):

- **Against the one-step prior:** voltage error −91%, OpenDSS fallbacks **35.8% → 5.8%**, near-limit balanced accuracy 0.926 → 0.992, **zero observed false-safe decisions**.
- **Against classical physics:** the iterated four-wire sweep reaches 3.6e-6 pu and 0.1% fallback. End-to-end times (2.1 vs 5.5 ms per state) compare an A100 model with an unoptimised CPU sweep, so we make no speed claim.

Every role we gave GENCO, against the strongest classical alternative:

| GENCO role | Result | Classical comparison |
|---|---|---|
| Screen in the exact search (C, and C+ with a final exact gate) | C had one material false acceptance in 1,728 intervals; C+ matches the exact search on 100% with 6% fewer exact calls | Exact balanced sweep and fixed-point screens do as well |
| Features for envelope proposals + one exact check | 79% of exact capacity with 6.7% of the exact calls (an exact-call reduction, not a measured speed-up) | GENCO features: −0.06 points, CI [−0.46, +0.32] |
| Frozen embeddings from our balanced GENCO in the four-wire GNN | **+6.0 points more** fallbacks, CI [+5.3, +6.7] | The same GNN without them is better |

Only B, C+ and the proposal path end with a final exact gate; C alone does not.

## 5. What we claim, and what we do not

**We claim** that, on this benchmark feeder, dynamic four-wire-verified envelopes release capacity a constant add-on cannot; that balanced-model envelopes are unsafe; that the GENCO architecture can be adapted to LV and guarded by an exact gate; and that a four-wire GNN residual corrects a cheap four-wire prior.

**We do not claim** unconditional safety, a GENCO advantage over specialised classical solvers, a wall-clock speed-up, or generalisation beyond this one feeder. All post-hoc results are exploratory. The full list is in [`docs/claims.md`](https://github.com/ujohn33/genco-dynamic-envelopes/blob/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/docs/claims.md).

## 6. Open science

- Code, protocols, numbered amendments, the [experiment log](https://github.com/ujohn33/genco-dynamic-envelopes/blob/1c172c82e54edeeb17b67d897f1b03ebe68fcac2/docs/experiment_log.md), configs, date splits, run manifests with SHA-256 hashes, and aggregate result files are in the repository (Apache-2.0); 84 automated tests.
- `scripts/reproduce_check.py` regenerates every headline number byte for byte **from the recorded per-day run files**. Those run files, the model checkpoints and the four-wire labels are derived from Fluvius smart-meter data.

  **What is not released, and why.** Until Fluvius confirms that its licence permits redistributing data derived from individual meters, we do not publish the model checkpoints (trained on Fluvius-derived states), the four-wire power-flow labels, or the per-day DOE run files. Their SHA-256 hashes are recorded in the run manifests (`results/run_manifest_*.json`), so a party with Fluvius data access can verify a regenerated artifact against ours. Everything needed to regenerate them (code, configs, seeds, date splits, Slurm scripts) is in the repository.
- Raw Fluvius data is not redistributed.

## Additional experiment: the official pretrained GENCO on case118

In a separate frozen transfer test on case118 with 99 flexible export units, using the official `gridfm/genco-opf-datakit` checkpoint, fine-tuning beat training from scratch by 14.2, 6.9 and 8.4 percentage points of reference hosting at 100, 1,000 and 10,000 labels. The preregistered primary criterion nevertheless failed: at 1,000 labels, the fine-tuned model kept 89.4% of the raw IPOPT reference, below the required 95%. These results use one training seed per arm and an IPOPT corrective-redispatch gate. Protocol, result, logs and hashes: [branch `ander-case118` @ f034a0c](https://github.com/ujohn33/genco-dynamic-envelopes/blob/f034a0c3d3cd3cc516c5e63f31c6e5298dc36fea/docs/experiment_log.md).

## 7. Next step

The household envelopes on a feeder sum to a flexibility range at the MV/LV transformer. The natural next test is a two-tier chain: feed that range into an MV envelope stage and check hosted flexibility and safety against a two-tier exact reference. Our LV results say what a cross-level model would need: magnitude-preserving edge features, level-specific (phase-aware) representations, and an exact gate at every tier.

---

## Team

- **Ander Martinez Alonso**, EVERGi, Vrije Universiteit Brussel (VUB). Contact: ander.martinez.alonso@vub.be
- **Evgenii Genov**, EVERGi, Vrije Universiteit Brussel (VUB)
- **Remy Cleerckewer**, Vrije Universiteit Brussel (VUB) and Ghent University (UGent)

Built on GENCO with gridfm-graphkit 0.9.0.3 and gridfm-datakit 1.1.0.
