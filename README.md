# Probabilistic dynamic operating envelopes with GENCO

An entry for the [GridFM GENCO Challenge](https://huggingface.co/spaces/gridfm/genco-challenge) (deadline 25 October 2026, AoE).

**Idea in one sentence:** compute dynamic operating envelopes (time-varying export and import limits
for every flexible customer on a distribution feeder) that hold under probabilistic load and PV
forecasts. GENCO's power flow mode evaluates thousands of (scenario × candidate envelope) cases in
one batch, and a conformal-prediction safety layer sends only the borderline cases to an exact AC
power flow. With a GENCO model trained on case33bw, this reproduces the AC export envelopes exactly
with 79% fewer AC power flows.

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
phasors out. Branch loadings come from one backward sweep of the injections at the predicted voltages,
which is far better conditioned on short lines than voltage differences. Every query in the envelope search
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
| `src/gencodoe/feeder.py` | Radial feeder model, exact AC power flow (vectorised backward/forward sweep), LinDistFlow baseline, pandapower import, branch flows from predicted voltages or from injections |
| `src/gencodoe/scenarios.py` | Correlated load/PV forecast-scenario sampler (Gaussian copula) |
| `src/gencodoe/envelopes.py` | Feasibility oracles, conformal screen-then-verify oracle, k-section envelope search, max-total LP |
| `src/gencodoe/conformal.py` | Split-conformal quantiles for voltage and loading errors, directional screening rule |
| `src/gencodoe/metrics.py` | Hosting, fairness (Jain), gap to the AC envelope, curtailment, screening confusion |
| `src/gencodoe/genco.py` | `GencoOracle`: in-process, batched GENCO power flow behind the same interface as the AC sweep |
| `src/gencodoe/demo.py` | Feeder setup, scenario and envelope-row samplers, and a quick demo with an emulated surrogate (no torch needed) |
| `scripts/export_case33bw.py` | Writes `grids/case33bw.m` (per unit, with thermal ratings) for datakit |
| `scripts/check_sweep_vs_powermodels.py` | Checks the AC sweep against PowerModels on datakit's scenarios |
| `scripts/make_training_data.py` | GENCO training rows (net export and import) in datakit's parquet schema |
| `scripts/run_genco_doe.py` | The GENCO study: accuracy, conformal screening, 24 h of envelopes, runtime |
| `configs/` | datakit and graphkit configs for case33bw, experiment settings |
| `results/genco_case33bw.md` | Output of `run_genco_doe.py` for the trained model |

## Results with GENCO (case33bw)

Full output: [`results/genco_case33bw.md`](results/genco_case33bw.md). GENCO = gridfm-graphkit 0.9.0.3
`GNS_heterogeneous` (hidden 48, 12 layers, 20 M parameters) in PowerFlow mode, trained from scratch for
21 epochs (best at epoch 11) on 40,000 rows on an Apple M5 (MPS), about 70 minutes.

**Power flow accuracy** on 5,000 fresh rows (AC sweep as truth): Vm MAE 0.0065 pu (99th percentile 0.031 pu,
0.015 pu MAE in the 1.06–1.10 pu band), Va MAE 0.35°, branch-loading MAE 0.0027 with flows from injections.

**Envelopes**, 24 hourly intervals, export and import, proportional policy, 500 scenarios, ε = 0.05, conformal α = 0.01
(q<sub>V</sub> = 0.032 pu, q<sub>loading</sub> = 0.017; empirical coverage 0.990 / 0.991):

| Method | Export: mean gap vs AC | Export: min P(ok) | Export: AC power flows | Import: mean gap | Import: AC power flows |
|---|---|---|---|---|---|
| AC power flow search | — | 0.944 | 114,000 | — | 184,000 |
| LinDistFlow | 0.09% | 0.948 | 0 | 30% | 0 |
| GENCO, no margin | 0.05% | **0.938** | 0 | 40% | 0 |
| GENCO + conformal + AC verify | **0.00%** | 0.944 | **23,491 (−79%)** | **0.00%** | 166,682 (−9%) |
| GENCO + conformal only | −20% (conservative) | 1.000 | 0 | −67% | 0 |

P(ok) is the AC-checked share of fresh scenarios without a violation. Its target is 0.95, and the AC search
itself dips to 0.944 because of sampling error with 500 scenarios. Eight import intervals are excluded,
because the feeder is already below V<sub>min</sub> there with zero import.

**Runtime per power flow:** GENCO batched on MPS 2.0 ms; pandapower Newton–Raphson 28 ms (one at a time);
our vectorised numpy sweep 5 µs; LinDistFlow 0.6 µs.

What this shows:

- Screen-then-verify works. GENCO plus conformal margins reproduces the AC envelope exactly and never misses
  a violation, while sending only the uncertain 2% of export rows to AC. Raw GENCO without margins is close
  but can break the target (0.938), and margins without verification are safe but give up 20% of the hosting.
- Import envelopes bind at low voltage, where GENCO is least accurate, so about half the rows are still uncertain.
- On a 33-bus radial feeder, GENCO beats a general Newton–Raphson solver but not a vectorised sweep, which is
  the best case for classical methods. The speed case for GENCO needs meshed or much larger networks, where no
  sweep exists.
- GENCO had to be adapted to the feeder in three ways. Training rows were generated with net export, which
  datakit's load generator does not produce. The physics loss was dropped, because on short, stiff lines it
  dominated training. Branch flows are taken from injections, because flows from predicted voltage differences
  have a loading MAE of 1.2.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12
pip install -e ".[dev,feeders]"
pytest
python -m gencodoe.demo --case33bw                      # no GENCO needed
```

Reproducing the GENCO study (about 1.5 h on an Apple M-series GPU):

```bash
pip install "torch>=2.10,<2.14"                         # first: torch-scatter compiles against it
pip install --no-build-isolation torch-scatter torch-sparse
pip install -e ".[genco,feeders]"
export MLFLOW_ALLOW_FILE_STORE=true                     # graphkit logs to ./mlruns-style file stores
W=~/gencodoe_work                                       # keep generated data out of synced folders
gridfm_datakit setup_pm                                 # Julia + PowerModels
python scripts/export_case33bw.py grids/case33bw.m
gridfm_datakit generate configs/datakit_pf_case33bw.yaml && mv data_out $W/
python scripts/check_sweep_vs_powermodels.py $W/data_out/case33bw/raw
python scripts/make_training_data.py $W/data_out/case33bw/raw $W/data_doe 40000
gridfm_graphkit train --config configs/genco_pf_case33bw.yaml --data_path $W/data_doe \
    --log_dir $W/mlruns --exp_name genco-pf-case33bw
python scripts/run_genco_doe.py
```

## License

Apache-2.0, matching the OpenGridFM toolkits.
