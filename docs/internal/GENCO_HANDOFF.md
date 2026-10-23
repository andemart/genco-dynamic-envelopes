# Handoff: GENCO Challenge entry — congestion-preventing DOEs

## The challenge (facts)
- GridFM GENCO Challenge, Porto 2026. Open challenge, no leaderboard; a jury judges **only the Community-tab post** (no pitch, no Q&A).
- **Deadline: Sun 25 Oct 2026, end of day AoE.** Our target: post on **24 Oct**. Winners announced 30 Oct, Porto.
- Must use **GENCO** + OpenGridFM toolkits (`gridfm-datakit`, `gridfm-graphkit`).
- Judging criteria: originality & impact · technical quality · exploration/extension of GENCO · communication & visuals · open science & reproducibility.
- Post must include: title, write-up, visuals, links to code/models/data, contact email, team names and affiliations.
- Template to mirror: example entry "[EXAMPLE] Contingency Analysis with GENCO" (summary → motivation → method → results → runtime → conclusion → next steps). Its stated next step is **conformal prediction** — we use that.
- Links:
  - Challenge: https://gridfm.org/porto/challenge/
  - Space and submission: https://huggingface.co/spaces/gridfm/genco-challenge (Community tab)
  - Example entry: https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1
  - GENCO paper: https://arxiv.org/abs/2608.09921
  - datakit: https://github.com/gridfm/gridfm-datakit
  - graphkit: https://github.com/gridfm/gridfm-graphkit
  - Colab tutorial: https://colab.research.google.com/github/gridfm/gridfm-graphkit/blob/lfe-tutorial/examples/notebooks/Tutorial_opf_colab.ipynb

## Key constraint
GENCO is a **balanced, transmission-trained** solver (PF / OPF / state estimation). datakit takes MATPOWER `.m` or PGLib cases. DOEs and congestion live in **distribution**, often unbalanced. We don't hide this gap — it is part of the research question.

## Pitch
> Can GENCO compute congestion-preventing dynamic operating envelopes (DOEs) on real distribution feeders fast enough to re-issue them every interval, and safely?

Working title: **"Certified congestion-preventing DOEs with GENCO"**

## Method (decided)
1. **Scope.** MV-level, balanced positive-sequence model. One DOE per **distribution transformer**: the aggregate import or export limit for the LV network or aggregator behind it.
2. **Make thermal limits bind.** Add EV and heat-pump load (evening import congestion) and PV (midday export congestion). Constraints: line and transformer loading ≤ 100%, voltage 0.94–1.10 pu.
3. **DOE engine (main).** Batched bisection with GENCO PF:
   - equal-allocation limit per time step, about 10 batched calls per interval;
   - per-node maximum import or export, with others at forecast, as N parallel bisections.
4. **Preventive vs corrective.** Preventive = the DOEs above. Corrective = GENCO **OPF mode**, with flexible assets modelled as generators carrying activation costs, to remove forecast overloads at least cost. Compare the two on the same feeder.
5. **Safety layer.** Split-conformal calibration of GENCO loading and voltage errors. Shrink envelopes so that P(violation) ≤ α. Plot curtailment vs α.
6. **Transfer study** (covers the "exploration" criterion). Compare GENCO pretrained zero-shot, fine-tuned and from-scratch (Tiny). Report voltage error, envelope error in kW, and violation rate.
7. **Reality check.** Apply the envelopes to the full **unbalanced** MV-LV model in OpenDSS on real days. Count phase-level overloads and violations, to quantify the cost of the balanced approximation.
8. **Probabilistic extension (if time).** Push forecast scenarios through GENCO in one batch to get P(congestion) per asset per interval.
9. **Baselines.** The same bisection with AC PF (PowerModels via datakit), with DC PF, and a static export or import limit.

## Metrics and figures
- Metrics:
  - overload hours and energy above rating per asset, with and without DOEs;
  - loading duration curves for the zone-substation transformer and the most stressed lines;
  - flexibility curtailed (kWh);
  - runtime;
  - envelope accuracy against AC;
  - violation rate.
- Figures:
  - feeder map coloured by envelope;
  - one day of DOE vs static limit;
  - GENCO vs AC envelope scatter;
  - violation rate vs safety margin;
  - overloads removed vs flexibility sacrificed;
  - runtime bars;
  - unbalanced-violation histogram.

## Data

### Networks
| Source | Notes |
|---|---|
| **Team-Nando MV-LV** https://github.com/Team-Nando/MV-LV-Networks | **Main case.** 4 real AusNet 22 kV feeders (urban HPK11: 44 transformers; CRE21: 79; rural SMR8 and KLO14: 700+). Excel network data, pseudo LV networks, real 30-min demand profiles, Python/OpenDSS time-series code. BSD-3. Start with an urban feeder. |
| **CSIRO MV + 33 LV** (`csiro:65408`) | 11 kV suburban feeder in OpenDSS. CC BY-NC-SA 4.0 (share-alike). Used in CSIRO's four-wire OPF DOE report, so it serves as a reference. The user's original link (`csiro:62996`) couldn't be opened; confirm which record was meant. |
| **BMOPF draft data** https://github.com/frederikgeth/BMOPFDraftData | Packaged OpenDSS models (CSIRO MV/LV set, ENWL four-wire variants). Could save cleaning time. |
| **IEEE European LV test feeder** https://cmte.ieee.org/pes-testfeeders/resources/ | Small unbalanced check. Also a European home for the Fluvius profiles. |

### Consumption and generation
| Source | Notes |
|---|---|
| **Fluvius 2,400 digital meters** (Flanders) | 15-min, full year. 8 groups × 300 meters: with/without PV × with/without HP × with/without EV. **Best fit for congestion.** Catalogue: https://www.vlaanderen.be/DataCatalogRecord/f71940e4-c5be-3d96-b73b-3a5c36c6cb9e — Zenodo derivative is CC BY 4.0; check the original licence. |
| **Team-Nando AusNet profiles** | Real 30-min demand matching the network. |
| **Ausgrid Solar Home** | 300 homes, half-hourly gross PV + consumption + PV capacity. CC BY. Gives gross PV. |
| **Low Carbon London** | 5,567 households, half-hourly. About 1,100 on dynamic ToU in 2013, so it shows measured demand response. Use it to check whether the curtailment we require is realistic. |
| **CKW (Switzerland)** https://open.data.axpo.com | Large 15-min individual-meter pool. Optional second test set. |
| Fallbacks | ElaadNL (EV sessions), When2Heat (heat pumps), PVGIS (PV). |

### Profile assignment rules
- Every customer at time t comes from the **same calendar day**, to keep weather correlation and coincident peaks.
- **Sum individual profiles** per transformer, using the real customer counts. Never scale one average profile.
- Penetration knob: move a share of households from the no-EV/HP group to the EV or HP groups (0%, 20%, 40%, 60%). Find where overloads begin.
- Smart meters measure **net** flow at the connection point. That is fine for congestion. For device-level DOEs, estimate the device share by comparing groups, or state that the DOE is per connection point.
- If mixing datasets: Belgian data on the Australian grid needs a 6-month seasonal shift, resampling from 15 to 30 min, and peak scaling to match ratings. Cleaner option: Fluvius on a European feeder, AusNet profiles on the AusNet feeder, giving two consistent cases.
- **Training** states come from datakit perturbations. **Test** uses real profile days only.

## Pipeline
1. Team-Nando Excel → pandapower (positive sequence; each LV network aggregated to load + DER at its transformer).
2. Export with `pandapower.converter.to_mpc` → `.m`.
3. datakit with `network.source: file` → PF (and OPF) data.
4. graphkit: train or fine-tune GENCO.
5. DOE engine, conformal layer, baselines.
6. OpenDSS reality check, using Team-Nando `backend.py` or OpenDSS directly.

## Schedule (today: 14 Oct)
| Dates | Work | Exit criterion |
|---|---|---|
| 14–15 Oct | Run the Colab tutorial and the Team-Nando notebook. Convert one feeder to `.m`. Check line ratings exist. Check Fluvius columns. Post a [Question] on HF about distribution feeders and PV as generators. | datakit solves AC PF on our feeder |
| 16–17 Oct | datakit data (~20k states, including EV/HP-heavy and reverse-flow states). Train Tiny from scratch and fine-tune. | Voltage error well below 0.01 pu |
| 18–19 Oct | Bisection DOE engine, AC and DC baselines, runtime | DOE time series for a real week |
| 20 Oct | Conformal margins, violation vs curtailment curve | α = 1% holds on test days |
| 21 Oct | Corrective OPF-mode comparison; unbalanced OpenDSS check | Violation and overload tables |
| 22–23 Oct | Figures; HF dataset and model upload; clean repo with one notebook | Someone else can rerun it |
| 24 Oct | Write and post the entry | Submitted |

## Open questions to settle on day 1
- [ ] Does datakit load a radial feeder with only a slack generator? Do DER-as-generators fix OPF mode?
- [ ] Does datakit produce enough reverse-flow and overload states? If not, perturb the DER generators too.
- [ ] Can datakit take a **custom aggregate load profile** (for example the Fluvius aggregate)?
- [ ] Do the Team-Nando Excel files include MV and LV line ampacities? If not, assign standard conductor ratings and state that assumption.
- [ ] Fluvius export: are offtake and injection separate columns? What is the licence?
- [ ] ScienceDirect paper S0378779626010291 was not accessible; get its abstract and check its relevance.

## Risks and cuts
- GENCO is poor zero-shot on high R/X feeders → that is a finding; show fine-tuning fixes it.
- Out of scope (put under "next steps"): an unbalanced GENCO, and gradient-based DOEs through GENCO as a differentiable surrogate.
- Exempt nothing from the reality check. Credibility comes from real test days and honest violation counts.

## Policy framing (one paragraph in the post; verify current rules first)
Position GENCO DOEs as the calculation engine for European congestion schemes: Germany's §14a EnWG (DSO dimming of heat pumps and wallboxes with a guaranteed floor), Dutch capacity-limitation contracts and GOPACS, and UK flexible connections.

## Related work to cite
- Unimelb (Ochoa et al.): model-free operating envelopes from smart meter data with neural networks.
- de Carvalho et al.: the DOE allocation depends on the power-flow model, the binding constraint, and import vs export (arXiv 2605.07989).
- Liu & Braslavsky (CSIRO): robust DOEs in unbalanced networks.
- CSIRO four-wire OPF DOE report (GPST Stage 4, Topic 8).
