Sixth audit round. Same rules: read-only, try to refute. Keep it short and narrow: only the blockers of your
fifth report. Do not re-read what you already cleared. Do not run the fixture test suite; reason from the
code, with small in-memory experiments if needed. Never read data_out/fourwire_scale/labels/test or /calib.

Branch ander-fourwire-scale, HEAD = 6013516 (tracked files = HEAD, stable while you read). Since ed17a1a:
- 5372b22 docs/fourwire_scale_protocol.md: appended "Amendments 7 to 9: second set of corrections", and
  results/fourwire_scale_dryrun_laptop_network1_{timing,evaluation}.json (the retained dry-run records).
- 88a5b01 tests/conftest.py (guard widened).
- d406941 src/gencodoe/fourwire_scale_train.py, fourwire_scale_eval.py and their tests.
- 33f72ad src/gencodoe/fourwire_scale_e2.py and its tests.
- 6013516 scripts/hydra_fourwire_scale_final.slurm, scripts/hydra_fourwire_scale_submit.sh.

For each item say resolved / not, with file:line, and give a concrete failing sequence if not:
1. C1 concurrent retries: claim_run_folder now takes a lock directory for the whole inspect / set-aside /
   claim transaction. Can two claimants still both succeed, or an active folder still be replaced?
2. C2 publication crash: tt.completed + tt.publish in evaluate and in e2._run_files. Can a run that
   completed still end without a published record, or a label be read twice because of it?
3. C3 skips: is exit code 10 now reached only after the completed record (and for fits the checkpoint, for
   evaluation the per-state file, for timing the decisions file, for envelopes the 576 intervals) validates?
4. B2 TestStore: _authorised at construction and at every load, with the fits_checked marker.
5. B3 record identity: canonical_fits / check_record / check_final_record / fit_record. Can a record of
   another fit, seed, budget, pool, source, software or input set still be accepted?
6. B4: bare CANCELLED, rerun limit -> halt, halt honoured by collect and both reports.
7. B7: end-of-run checks (job script, rating variants for evaluation, profiles for envelopes); software()
   now lists optional PyG backends; cuDNN is recorded per run and deliberately not pinned (protocol C9).
8. B8/D10: check_envelopes (576 unique intervals, settings, recomputed summary).
9. Disclosure and protocol text ("second set of corrections", parts A-D): accurate? Any statement the
   code does not implement? Compare part B's numbers with the two committed dry-run records.
10. Anything new that these changes broke.

Then one line: ready to freeze once an A100 dry run of this source ends with DRY_RUN_OK and has been read,
or the remaining blockers in order.
