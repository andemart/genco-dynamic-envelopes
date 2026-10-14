# Work plan (14 → 25 October 2026)

| Day(s) | Task | Output |
|---|---|---|
| 14–15 Oct | Install datakit + graphkit on a GPU box; generate GOC 500 OPF data with `configs/datakit_opf_case500.yaml` | ~100k OPF samples |
| 15–16 Oct | Train GENCO Tiny/Small; check against the paper's OPFData numbers | trained checkpoint |
| 16–17 Oct | Tag generators as wind/PV, define regions; implement `scenarios_to_datakit_format` | scenarios as parquet |
| 17–18 Oct | Run AC-OPF (PowerModels) on a held-out scenario set as ground truth | reference set |
| 18–19 Oct | Batched GENCO inference; implement `line_loading_from_predictions` | predicted loadings, costs |
| 19–20 Oct | Conformal calibration, screening, metrics, runtime benchmark | results tables |
| 21–22 Oct | Figures: P(congestion) map, cost fan chart, screening confusion matrix, speed–accuracy plot | figures |
| 23–24 Oct | Write-up, clean repo, publish data/model to Hugging Face | submission draft |
| 25 Oct | Post in the Community tab | submission |

## Open questions

- Which grid: GOC 500 (native, fast) or a Benelux-scale model converted from PyPSA-Eur (more relevant, more work)?
- Real forecast error statistics: use Elia's published wind/solar forecast data to set `forecast_rel_std`?
- Redispatch cost definition: grid-constrained OPF minus copper-plate dispatch, or explicit up/down redispatch prices?
- GPU access for training.

## Risks

- GENCO OPF accuracy on RES-heavy, curtailment-prone scenarios outside the datakit training distribution → train on data that includes the re-tagged RES Pmax variation.
- Conformal coverage is marginal, not per-line → report per-line coverage too.
