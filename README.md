# Probabilistic congestion and redispatch screening with GENCO

An entry for the [GridFM GENCO Challenge](https://huggingface.co/spaces/gridfm/genco-challenge) (deadline 25 October 2026, AoE).

**Idea in one sentence:** instead of checking grid security for one deterministic forecast, push an
ensemble of probabilistic wind, solar and load forecast scenarios through GENCO's optimal power flow
(OPF) mode, and report the *distribution* of line loadings, congestion and redispatch cost hours ahead,
with a conformal-prediction safety layer that sends uncertain cases to a classical AC solver.

## Why GENCO

GENCO is a unified neural solver for power flow, OPF and state estimation
([paper](https://arxiv.org/abs/2608.09921)). Its OPF mode is reported as 16–85× faster than AC-OPF
(IPOPT) on grids from 14 to 2,000 buses, with fewer constraint violations than DC-OPF. Running thousands
of forecast scenarios per delivery hour is exactly the batched workload where that speed matters.

## Pipeline

```
probabilistic forecasts ──► scenario sampler ──► per-bus loads / RES injections
   (load, wind, PV)          (copula, N draws)            │
                                                          ▼
                                    GENCO OPF (batched, GPU) ──► dispatch, flows, cost
                                                          │
                                   conformal calibration ─┤
                                                          ▼
                     confident & safe ──► accept        uncertain / near-limit ──► AC-OPF (PowerModels)
                                                          │
                                                          ▼
                     P(congestion) per line · redispatch-cost distribution · solver calls saved
```

## Repository layout

| Path | What it holds |
|---|---|
| `src/probredispatch/scenarios.py` | Correlated forecast-scenario sampler (Gaussian copula over load, wind, PV errors) |
| `src/probredispatch/grid.py` | Maps scenarios onto bus loads and generator limits of a MATPOWER-style case |
| `src/probredispatch/conformal.py` | Split-conformal intervals on predicted line loading and the screen-then-verify rule |
| `src/probredispatch/metrics.py` | Congestion probabilities, redispatch-cost quantiles, screening confusion matrix |
| `src/probredispatch/genco.py` | Thin wrapper around `gridfm-graphkit` training and inference (to be completed) |
| `configs/` | `gridfm-datakit` and experiment configs |
| `docs/workplan.md` | 11-day plan to the deadline and open questions |
| `tests/` | Unit tests for the parts that run without GENCO |

## Status

Scaffold. Scenario sampling, conformal screening and metrics are implemented and tested.
GENCO training/inference and the AC-OPF verification step are stubbed with clear TODOs; see
`docs/workplan.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# GENCO toolkits (GPU recommended; see their READMEs for torch-scatter)
pip install gridfm-datakit gridfm-graphkit
gridfm_datakit setup_pm
gridfm_datakit generate configs/datakit_opf_case500.yaml
```

## Evaluation (what the write-up will report)

- **Accuracy:** line-loading MAE vs AC-OPF, per loading bin; redispatch-cost gap.
- **Screening:** recall on overloaded lines at the chosen conformal level; share of scenarios sent to AC-OPF.
- **Distribution quality:** CRPS / reliability of P(congestion) against AC-OPF on the same scenarios.
- **Runtime:** amortised per-scenario time, GENCO batched vs AC-OPF and DC-OPF.

## License

Apache-2.0, matching the OpenGridFM toolkits.
