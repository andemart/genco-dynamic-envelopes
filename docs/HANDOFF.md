# Handoff: GENCO dynamic envelopes (state on 18 Oct 2026)

For anyone picking this work up: a collaborator, a reviewer or a new Claude session. Start here, then read [`README.md`](../README.md), [`docs/claims.md`](claims.md) and [`docs/experiment_log.md`](experiment_log.md).

The detailed history of operating notes, incidents and run procedures is in [`docs/HANDOFF_2026-10-18.md`](HANDOFF_2026-10-18.md).

## What this project is

An entry for the **GridFM GENCO Challenge 2026** (deadline 25 Oct AoE).

**The work, step by step:**
- We compute four-wire-verified **dynamic operating envelopes** (per-household time-varying import/export limits) on the ENWL1 four-wire LV benchmark feeder.
- The feeder is populated with real Fluvius 2024 smart-meter households.
- We adapt GENCO to LV, embed it in an exact four-wire workflow, and extend a GENCO-style model to explicit four-wire states.
- Every GENCO role is benchmarked against strong classical methods.

**The bottom line:**
- Dynamic envelopes free 4.8 MWh of export and 2.5 MWh of import envelope-hours per household per year, where a static limit frees 0.
- A GENCO-style four-wire GNN cuts OpenDSS fallbacks from 35.8% to 5.8% relative to a one-step four-wire prior.
- Pretrained balanced-GENCO representations add no value, and a specialised four-wire sweep remains the accuracy leader.

## Where everything is

| What | Where |
|---|---|
| Overview and reproduction | `README.md` |
| Hugging Face post draft (story mapped to the judging criteria) | `docs/hf_post.md` |
| One-pager | `docs/genco_challenge_onepager.tex` / `.pdf` |
| Slides (private until shared by the owner) | https://claude.ai/artifact/3amyojT9KXZEMuipT38FiA |
| Figures (regenerate: `python scripts/make_figures.py`) | `figures/fig1–fig5` |
| What we claim and do not claim | `docs/claims.md` |
| Every experiment, result and incident | `docs/experiment_log.md` |
| Protocols and amendments (frozen before results) | `docs/preregistration*.md`, `docs/cplus_protocol.md`, `docs/weekend_protocols.md`, `docs/option2_protocol.md`, `docs/envpred_protocol.md`, `docs/unbalanced_genco_protocol.md` |
| Result files and run manifests (commits, job IDs, SHA-256) | `results/` |
| Byte-for-byte reproduction of every headline number | `python scripts/reproduce_check.py` |
| Environment snapshots (not installable locks) | `env-snapshot-laptop.txt`, `env-snapshot-hydra.txt` |

## Branches

- **`ander`** (pushed): every finished result, the four-wire extension, the audit fixes and the submission documents.
- **`submission`** (local): identical to `ander` at the time of the push.
- **`main`**: still far behind. Fast-forward it to `ander` only when the owner approves; judges land on `main`.
- **`ander-unbalanced-genco`** and **`ander-multifeeder`**: superseded working branches.

## Open items

1. **Post-hoc per-boundary analysis.** Defined in unbalanced protocol amendment 3 (`gencodoe.fourwire_posthoc`). Hydra job 13343893 was pending because the A100 partition was down.
   - When it runs, fetch `results/fourwire_posthoc_per_boundary.json`, check its union-mask consistency field, commit it unchanged, and add a short post-hoc note to the experiment log.
2. **Artifact bundle.** Not published.
   - Raw Fluvius data must not be redistributed. Derivatives with uncertain status (four-wire labels with per-customer P/Q, features, embeddings, checkpoints, DOE run files) wait for the owner to confirm Fluvius's licence and privacy terms.
   - Safe to publish now: code, configs, date splits, manifests and aggregate report outputs.
   - For strict-JSON consumers, export sanitized copies (non-finite values → null) **without** rewriting the historical result files.
3. **Links.** The `[repository link]` and `[dataset link]` placeholders in `docs/hf_post.md` and on the last slide.
4. **Release.** Fast-forward `main`, tag a release, and post on Hugging Face (target 24 Oct).

## Rules this project follows

- Protocols are committed before runs, and nothing is tuned on test.
- Every deviation is a numbered amendment, and negative results are reported.
- Wording: "four-wire-verified", not "certified"; "exact-call reduction", not "speed-up"; "envelope-hours", not "delivered energy"; "ENWL1 four-wire benchmark populated with Flemish profiles", not "a real Flemish feeder".
- Never `git add -A`; stage files explicitly. Raw Fluvius data and `data_out/` stay out of git.
