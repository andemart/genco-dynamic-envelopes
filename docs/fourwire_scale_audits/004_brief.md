Fourth audit round, same rules: read-only, try to refute, derive numbers yourself. Be concise.

Branch ander-fourwire-scale, HEAD = 73442a2 (working tree = HEAD for tracked files; src/ and tests/ are
stable while you read). Since your third report:
- 2356f51 docs/fourwire_scale_protocol.md: Amendments 7, 8 and 9, revised after your third report.
- babf88a, d079cf6 src/gencodoe/fourwire_scale_train.py (fit, final_fit, attempts, freeze, collect),
  fourwire_scale_data.py (served splits as a class attribute).
- 83c3454, ace87c8 src/gencodoe/fourwire_scale_eval.py (TestStore, DeviceNet/Pipeline, margins, metrics,
  evaluate, report).
- 8192d2e, e68aa9e src/gencodoe/fourwire_scale_e2.py (OpenDSS pool, timed series, timing, envelopes, report).
- 0d78749 scripts/hydra_fourwire_scale_final.slurm and hydra_fourwire_scale_dryrun.slurm.
- 06203c2, 73442a2 scripts/fourwire_scale_val_sweep_vs_opendss.py and its results JSON.
- tests/test_fourwire_scale_train.py, test_fourwire_scale_eval.py, test_fourwire_scale_e2.py.
Nothing is frozen. No final fit, calibration, test read or timing has been run. Dry runs so far used train
and val labels only (a laptop run on network 25; A100 runs are queued).

A. Your third report, items C1-C3, D1-D4, E1-E7, F1-F2: for each say resolved / partly / not, against the
   committed Amendments 7-9 AND the committed code. Where the text promises something, check that the code
   does it (file:line), and where the code does something the text does not say, flag it.

B. Code review, highest risk first. Try to break:
   1. Test isolation: any path by which fits, dry runs, calibration, timing or envelopes can read a test
      label, or by which the final evaluate can run twice, run before all fits are valid, or run with
      models from another freeze. Is TestStore reachable from anything except the final evaluate?
   2. final_fit / claim_run_folder / fit: can a completed fit be rerun or replaced; can an attempt be
      hidden; does a rerun really get compared with the earlier attempt; PG-ft start weights and scalers;
      scratch statistics from B states only; epoch 0 not eligible; terminal records.
   3. Pipeline (device path) against the training path in fit: U1, features, scaling, residual, KCL
      currents, the four quantities. Any train/deploy mismatch beyond FP32 rounding?
   4. evaluate: calibration sets (target calib; PG-zs pool calib), margins, decision_metrics,
      state_metrics, rating variants, sweep_agreement. Any wrong index, mask or direction? Does anything
      from the test labels feed back into a decision?
   5. timing: is every series' timed region complete and comparable (transfers, screen, fallback, OpenDSS)?
      Anything that favours one side? The union rule, the reset per chunk, the rotation, split_inputs.
   6. report (both modules): the E3 statistics and the E2 criterion against sections 7 and Amendments 8-9.
      The bootstrap, the delivered-decision scoring, the break-even definition.
   7. envelopes: scenario construction against Amendment 9 item 7, sign conventions for import/export,
      three-phase customers, the feasibility functions of evaluators A and B.
   8. check_freeze / freeze: what can still differ between a frozen state and a final run without being
      caught?
   9. The Slurm script: stages, skip logic, dependencies, anything that can corrupt or silently skip a fit.
   You may run the unit tests of these three test files if your sandbox allows (they need no test labels),
   and small experiments on train/val labels under data_out/fourwire_scale/labels. Do not read anything
   under data_out/fourwire_scale/labels/test.

C. Tests: which promised properties are not tested, or tested more weakly than the test name says?

D. Is anything in Amendments 7-9 still ambiguous enough to change a reported number, or inconsistent with
   sections 4-8 and Amendments 3, 5, 6?

E. Verdict: ready to freeze, or the list of blockers in order.
