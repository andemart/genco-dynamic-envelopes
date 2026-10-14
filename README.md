# Probabilistic dynamic operating envelopes with GENCO

An entry for the [GridFM GENCO Challenge](https://huggingface.co/spaces/gridfm/genco-challenge) (deadline 25 October 2026, AoE).

**Idea in one sentence:** compute dynamic operating envelopes (time-varying export and import limits
for every flexible customer on a distribution feeder) that hold under probabilistic load and PV
forecasts. GENCO's power flow mode evaluates thousands of (scenario × candidate envelope) cases in
one batch, and a conformal-prediction safety layer sends only the borderline cases to an exact AC
power flow.

## Dynamic operating envelopes in brief

A distribution network operator publishes, for each interval (say 30 minutes), a limit per connection
point: "you may export up to *E* kW / import up to *I* kW". If every customer uses its full limit at once,
voltages must stay within [V<sub>min</sub>, V<sub>max</sub>] and lines within their thermal ratings. The
limit has to hold across forecast uncertainty in the *passive* load and rooftop PV on the feeder, so we
treat it as a chance constraint: feasible in at least 1 − ε of the forecast scenarios.

Computing envelopes is a search over power flows: for every interval, every allocation policy and every
candidate limit you need to know whether all scenarios stay within limits. With 48 intervals,
1,000 scenarios and around 10 search steps, a single feeder needs about 500,000 power flows per day. That
batched, repeated workload is where a fast neural solver pays off.

## Why GENCO

GENCO is a unified neural solver for power flow, OPF and state estimation
([paper](https://arxiv.org/abs/2608.09921)). We use its power flow mode: bus injections in, voltage
phasors out, branch loadings computed from the predicted voltages. Every query in the envelope search
is an independent power flow, so they batch on a GPU. A plain surrogate is not enough for a safety
limit, so its errors are calibrated with split conformal prediction: GENCO decides only when its
prediction interval is clearly inside or outside the limits, and AC power flow decides the rest.

## Pipeline

```
probabilistic forecasts ──► scenario sampler ──► passive bus injections (n_s scenarios)
   (load, rooftop PV)        (Gaussian copula)              │
                                                            ▼
              candidate envelope levels ──► rows = scenarios × levels (one batch)
                                                            │
                                                GENCO power flow (batched, GPU)
                                                            │
                                     conformal screen (|error| quantiles q_v, q_l)
                                    ┌───────────────────────┼─────────────────────┐
                                  safe                  uncertain              violated
                                    │              AC power flow (sweep)          │
                                    └───────────────────────┼─────────────────────┘
                                                            ▼
                      P(feasible) per level ──► k-section search ──► envelope per customer
                                                            │
                          AC verification on fresh scenarios · hosting · fairness · AC calls saved
```

## Allocation policies

| Policy | Rule | Trade-off |
|---|---|---|
| `uniform` | same limit for everyone, capped at inverter rating | fair, leaves capacity unused |
| `proportional` | limit ∝ inverter rating | fair per kW installed |
| `max_total` | LP on LinDistFlow sensitivities maximising the summed limit | most hosting, favours customers near the substation |

Each direction only checks the constraints it makes worse. Export envelopes check voltage rise and
reverse-flow loading, and import envelopes check voltage drop and forward-flow loading. A base case that is
already undervoltage at peak load therefore does not shrink the export envelope.

## Repository layout

| Path | What it holds |
|---|---|
| `src/gencodoe/feeder.py` | Radial feeder model, exact AC power flow (vectorised backward/forward sweep), LinDistFlow baseline, pandapower import, flows from predicted voltages |
| `src/gencodoe/scenarios.py` | Correlated load/PV forecast-scenario sampler (Gaussian copula) |
| `src/gencodoe/envelopes.py` | Feasibility oracles, conformal screen-then-verify oracle, k-section envelope search, max-total LP |
| `src/gencodoe/conformal.py` | Split-conformal quantiles for voltage and loading errors, directional screening rule |
| `src/gencodoe/metrics.py` | Hosting, fairness (Jain), gap to the AC envelope, curtailment, screening confusion |
| `src/gencodoe/genco.py` | `GencoOracle`: graphkit training and inference behind the oracle interface (parquet I/O still TODO) |
| `src/gencodoe/demo.py` | End-to-end run without GENCO, using an emulated surrogate |
| `configs/` | `gridfm-datakit` PF config for case33bw and experiment settings |
| `docs/workplan.md` | Plan to the deadline, open questions, risks |

## Status

The envelope engine, scenarios, conformal layer, metrics and reference solvers are implemented and
tested. GENCO is not trained yet. The demo stands it in with exact AC power flow plus Gaussian noise
(σ<sub>V</sub> = 0.002 pu, σ<sub>loading</sub> = 0.01), which exercises the screening logic but says nothing
about GENCO's real accuracy. See `docs/workplan.md`.

Demo output, case33bw, noon interval, proportional export envelope, 500 scenarios, ε = 0.05, conformal α = 0.01:

| Method | Total export (pu) | Gap vs AC | P(ok), AC on fresh scenarios | AC power flows |
|---|---|---|---|---|
| AC power flow search | 5.949 | — | 0.958 | 11,000 |
| LinDistFlow search | 5.923 | −0.4% | 0.970 | 0 |
| Emulated surrogate + conformal + AC verify | 5.949 | 0.0% | 0.958 | 2,645 |
| Emulated surrogate + conformal only | 5.751 | −3.3% | 1.000 | 0 |
| Max-total LP | 6.559 | +10.3% | **0.856** | 500 |

Screen-then-verify gives the same envelope as AC power flow with 76% fewer AC power flows. The max-total LP
breaks the 0.95 target once checked with AC power flow, which is why every envelope gets an AC check.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,feeders]"
pytest
python -m gencodoe.demo --case33bw

# GENCO toolkits (GPU recommended; see their READMEs for torch-scatter)
pip install -e ".[genco]"
gridfm_datakit setup_pm
gridfm_datakit generate configs/datakit_pf_case33bw.yaml
```

## Evaluation (what the write-up will report)

- **Envelope accuracy:** total and per-customer gap between GENCO-based and AC-based envelopes, per policy and direction.
- **Safety:** out-of-sample violation probability (AC power flow) against the 1 − ε target. For screening, missed-violation rate and share sent to AC.
- **Power flow accuracy:** voltage MAE near V<sub>max</sub>, loading MAE near 1.0, conformal coverage per bus and branch.
- **Value:** hosting capacity and curtailed PV export versus a static export limit (e.g. 5 kW per customer) and versus LinDistFlow.
- **Runtime:** wall-clock per interval for GENCO batched vs AC sweep vs pandapower Newton–Raphson, and AC calls avoided.

## License

Apache-2.0, matching the OpenGridFM toolkits.
