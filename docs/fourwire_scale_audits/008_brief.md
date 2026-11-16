Audit round 8 (read-only). Scope: the last check before the final job chain is submitted. Nothing final has run yet.

State of the repository: HEAD is a4068ef. It is two commits after e85164c, the commit you cleared in round 7 "once an A100 dry run of this source ends with DRY_RUN_OK and its output has been read":
- c818852 adds evidence files under results/ (A100 dry-run records, a determinism probe record, job logs).
- a4068ef adds results/fourwire_scale_freeze.json, written on the cluster by `python -m gencodoe.fourwire_scale_train freeze` in a worktree at e85164c.

Please try to refute each of the following, from the files, and report what you find. Do not accept my reading; check the artefacts.

1. Source identity. `git diff e85164c..HEAD` touches only results/. src/, scripts/, tests/ and docs/ are byte-identical to the commit you cleared.

2. Freeze file. results/fourwire_scale_freeze.json is what `freeze()` in src/gencodoe/fourwire_scale_train.py would write for this source: recompute, without writing anything, the source hash the way `checkout_state()` does, the per-file hashes in `files_sha256`, the job-script hash, the two input hashes and the folds, and compare. Compare its `software`, `source_sha256`, `label_manifest_sha256` and `identification_sha256` with the same fields in the dry-run records of item 3. Is there any field that `check_freeze`, `check_final_record` or `checkout_unchanged` reads and that the file lacks or holds in a form they would reject?

3. A100 dry runs (train and val states only; jobs 13588810 on network 17 and 13588811 on network 7). Records: results/fourwire_scale_dryrun_a100_network{17,7}_{evaluation,timing,envelopes}.json. Job logs and accounting lines: results/fourwire_scale_hydra_job_evidence.txt from the line "== sacct (job, state, start, end, node, submit line): determinism probe rerun and the A100 dry runs of e85164c" to the end. Read them and look for anything that should block the submission: wrong device, numerical mode or core count; a decision that disagrees with OpenDSS; an inconsistency between a record and its log; a field the final validators would reject; a sign that a code path the final runs depend on was not exercised. In particular, give your own explanation, from the code, of these three things I noticed, and say whether any of them is a defect:
   a. All eight dry-run envelopes on networks 17 and 7 (one val day, hours 0 and 1, both directions) are 0 kW for evaluator A and for evaluator B, with `A_share_feasible` below the acceptance share at level 0.
   b. On network 17 the two rating-variant screens (`screen_cu120_240`, `screen_cu16_89`) are identical to the primary screen for every method; on network 7 one of them differs.
   c. In the timing records, the `u1_*` and `pgnet_*` series report 0 OpenDSS solves at every batch size, while `sweep_cpu` and `sweep_gpu` report 85 (network 17) and 350 (network 7) at the largest batch.

4. Time limits. scripts/hydra_fourwire_scale_submit.sh gives 12 h to a PG-pre fit, 6 h to the target stage (PG-net and five PG-scratch fits of one network), 3 h to the PG-ft stage, 1.5 h to an evaluation, 8 h to a timing run and 4 h to an envelope run. Using the step times in the dry-run fit lines of the job logs (deterministic mode), results/fourwire_scale_smoke_pgpre_job13586516.json (default mode, full PG-pre pool) and the round times implied by the dry timing records (1 warm-up and 3 timed rounds there; 3 and 100 in a final run, on 4,096 states), is any limit likely to be exceeded? A timeout is a rerun reason under Amendment 7, so this is about waste and about the rerun limit, not validity.

5. Determinism probe rerun: results/fourwire_scale_determinism_job13587703.json. Does it support the statement "the two deterministic-mode runs are identical in every step loss, validation loss and final weight; the two default-mode runs differ from step 2"? Is anything in it inconsistent with Amendment 7?

Constraints, as before: read-only; do not run anything that writes into the repository or data_out; never open data_out/fourwire_scale/labels/test or data_out/fourwire_scale/labels/calib.

End with one of: "Ready to submit" or "Not ready to submit", followed by the blockers in order. Separate blockers from remarks that can wait until after the runs.
