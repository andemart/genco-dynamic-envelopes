# Work plan (14 → 25 October 2026)

| Day(s) | Task | Output |
|---|---|---|
| 14–15 Oct | Install datakit + graphkit on a GPU box; get case33bw into datakit (`configs/datakit_pf_case33bw.yaml`); make sure the training rows include net export (negative load) | ~50k PF samples |
| 15–16 Oct | Train GENCO Tiny/Small in PF mode; check voltage MAE against the AC sweep, especially near V<sub>max</sub> | trained checkpoint |
| 16–17 Oct | Implement `injections_to_datakit_format` and `read_predictions` in `genco.py`; test `GencoOracle` against `sweep_pf` | working oracle |
| 17–18 Oct | Conformal calibration on held-out envelope rows; per-bus coverage; choose α | q_v, q_l, coverage table |
| 18–19 Oct | Full day of half-hourly envelopes (both directions, three policies, three lead times) with AC, LinDistFlow, GENCO+conformal | results tables |
| 19–20 Oct | Runtime benchmark: k-section batch size vs GPU throughput; compare with pandapower NR | speed–accuracy plot |
| 21–22 Oct | Figures: envelope profile over the day per customer, hosting vs fairness per policy, violation probability vs ε, AC calls saved | figures |
| 23–24 Oct | Write-up, clean repo, publish data/model to Hugging Face | submission draft |
| 25 Oct | Post in the Community tab | submission |

## Open questions

- Does GENCO (trained on transmission-style pglib grids in the paper) reach usable voltage accuracy on a
  radial, high-R/X feeder? Train from scratch on feeder data. Fallback: a larger feeder (case141, or an
  LV network from SimBench) where the speed gain over AC is bigger.
- Balanced single-phase equivalent vs. unbalanced LV: real DOEs are per phase. State the limitation and
  keep the balanced model.
- Out-of-sample violation probability sits around 1 − ε ± 0.02 with 500 scenarios. Should the search use a
  scenario-approach margin, or tighten ε on a held-out set?
- Forecast error statistics: take load/PV error std from a public smart-meter + PV dataset (e.g. Ausgrid
  solar homes) instead of the placeholders in `configs/experiment.yaml`.
- Max-total LP: iterate the linearisation (re-solve around the AC point) so it keeps the 1 − ε target.

## Risks

- GENCO error is largest exactly at the binding constraint (voltage near V<sub>max</sub> with high reverse flow),
  outside the bulk of the datakit training distribution. Mitigations: oversample high-export rows, and
  per-element conformal quantiles.
- Conformal coverage is marginal, not per bus. Report per-bus coverage too.
- On a 33-bus feeder the AC sweep is already fast (~ms for 10k rows on CPU), so the speed case needs a
  bigger feeder or many feeders batched together.
