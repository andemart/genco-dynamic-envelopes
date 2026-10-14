# Work plan (to 25 October 2026)

## Done (14 Oct)

- Envelope engine, scenarios, conformal layer, metrics, AC sweep and LinDistFlow, with tests.
- case33bw exported for datakit (`scripts/export_case33bw.py`); AC sweep matches PowerModels to ~1e-10 pu
  on datakit's scenarios (`scripts/check_sweep_vs_powermodels.py`).
- 40k envelope-like training rows in datakit's schema (`scripts/make_training_data.py`).
- GENCO (graphkit 0.9.0.3, GNS_heterogeneous, hidden 48) trained on Apple MPS. A 4-epoch ablation picked
  lr 1e-4 and MSE-only loss; checkpointing on validation loss, not the physics residual.
- In-process `GencoOracle`, checked against `gridfm_graphkit predict` (≤ 2e-4 pu).
- Full study in `results/genco_case33bw.md`.

## Next

| Day(s) | Task |
|---|---|
| 15–16 Oct | Improve Vm accuracy near the limits: more rows near V<sub>max</sub>/V<sub>min</sub>, longer training, per-bus conformal quantiles (the import side sends 52% of rows to AC) |
| 16–17 Oct | A larger or meshed feeder (case141, a SimBench LV/MV grid) where no sweep exists and GENCO's speed matters; multi-feeder batching |
| 18–19 Oct | Max-total and uniform policies with GENCO; iterate the LP linearisation and check it with GENCO + AC |
| 19–20 Oct | Lead-time study (1 h / 6 h / 24 h forecast errors) and comparison with a static 5 kW export limit (hosting, curtailment) |
| 21–22 Oct | Figures: envelope profile over the day, P(ok) vs ε, AC calls saved vs α, speed-accuracy |
| 23–24 Oct | Write-up; publish the trained checkpoint and training data on Hugging Face |
| 25 Oct | Post in the Community tab |

## Open questions

- graphkit 0.9 masks the slack voltage in PF mode, so the model is trained at a fixed 1.0 pu substation voltage.
  The GitHub main branch can unmask it (`data.mask_ref_vm: false`); use it if OLTC set-point variation matters.
- Balanced single-phase equivalent vs. unbalanced LV: real DOEs are per phase. State the limitation.
- With 500 scenarios the out-of-sample P(ok) sits at 1 − ε ± 0.02; add a scenario-approach margin or more scenarios.
- Forecast error statistics: take load/PV error std from a public smart-meter + PV dataset.

## Risks

- GENCO error is largest near the binding limits (Vm MAE 0.015 pu in 1.06–1.10 pu), which widens conformal margins.
- On radial feeders a vectorised sweep is ~400× faster than GENCO, so the speed claim must rest on meshed/large grids.
