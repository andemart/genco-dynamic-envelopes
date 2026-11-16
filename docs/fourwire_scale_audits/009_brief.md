Audit round 9 (read-only). HEAD is a8a2612, one commit after the freeze commit a4068ef. The final runs of the freeze are in progress on the cluster; nothing in src/ or in the frozen runs has been changed. This commit adds a dated protocol note and one script:

- docs/fourwire_scale_protocol.md, last section: "Note on two weak PG-pre seeds and an exploratory diagnostic (2026-11-16, ...)".
- scripts/fourwire_scale_pgpre_diagnostic.py (modes `checkpoints` and `retrace`).

It follows your feedback on the weak fold-2 PG-pre seeds: keep every frozen run, report weak seeds as such, diagnose the selected checkpoints on train/val only before any test outcome is viewed. I have not opened any evaluation record, timing record or report of the final runs.

Please try to refute the following, from the files:

1. `git diff a4068ef..HEAD` touches only docs/fourwire_scale_protocol.md and scripts/fourwire_scale_pgpre_diagnostic.py; the source hash of the freeze is unaffected.

2. The script cannot read a calib or test label, and cannot write into the run folders it reads (it is given the final run folder of another worktree as --runs). Look at every file access, including those made by the functions of src/gencodoe/fourwire_scale_train.py that it calls (`prepare`, `fit`, `start_provenance`, `set_determinism`, `fs._case`, `fs._write_json`).

3. `checkpoint_diagnostic` computes the same pooled validation loss as `evaluate()` inside `fit` in src/gencodoe/fourwire_scale_train.py (same forward pass, same loss terms and weights, same batches of EVAL_BATCH, same weighting by converged states), so that comparing it with the record's `best_val_loss` is a valid fidelity check; and the four reported terms add up to the loss. Is any per-network number it reports misleading (for example the per-batch normalisation by the near-limit weights)?

4. `retrace` calls `tt.fit` with the arguments `final_fit` used for a PG-pre fit, except `record_steps=True` and the output folder. Does `record_steps=True`, or anything else in the call, change the trajectory compared with the final fit? Is the replay of the shuffling (one `rng.permutation` per pool network, then one over the batches, skipping batches without a converged state) exactly the order in which `fit` draws from its generator, for every epoch including after early stopping? Would the checks in the script catch a wrong replay?

5. The protocol note: is any statement in it wrong, stronger than the evidence, or in conflict with Amendments 7 to 9? In particular the statements that PG-zs and PG-pre-tc use PG-pre seed 0, that 12 of the 120 PG-ft fits start from the two weak seeds, and that the rerun in `retrace` "is not a fit of the study and replaces nothing".

Constraints as before: read-only; do not run anything that writes into the repository or data_out; never open data_out/fourwire_scale/labels/test or data_out/fourwire_scale/labels/calib.

End with a short verdict: findings that should stop the diagnostic from being used, and remarks.
