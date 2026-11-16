Fifth audit round, same rules: read-only, try to refute, derive numbers yourself. Be concise.

Branch ander-fourwire-scale, HEAD = ed17a1a (tracked files in the working tree = HEAD; src/, tests/ and
scripts/ are stable while you read). Since your fourth report:
- c48b592 docs/fourwire_scale_protocol.md: appended "Amendments 7 to 9: a disclosure and corrections".
- cdf0b1d tests/conftest.py: the generator refuses the calib and test splits in every unit test.
- c9ef1ad src/gencodoe/fourwire_scale_train.py, fourwire_scale_eval.py and their tests.
- 9a3a68f src/gencodoe/fourwire_scale_e2.py and its tests.
- ed17a1a scripts/hydra_fourwire_scale_final.slurm, scripts/hydra_fourwire_scale_submit.sh.
Nothing is frozen. No final fit, final evaluation or timing has been run.

IMPORTANT about tests: your B1 was right. The fixture did generate 12 real test states of network_25 and the
tests evaluated throwaway models on them; this is disclosed in the protocol (section A of the new text). The
fixtures now draw what plays calib/test from val days (tests/conftest.py, pretend_held_out). You may run the
unit tests; they no longer generate or read a real calib or test state. Do not read anything under
data_out/fourwire_scale/labels/test or data_out/fourwire_scale/labels/calib.

A. The disclosure (protocol, new section, part A): is it accurate and complete against the repository
   history (commits 83c3454..73442a2, the old fixture, what evaluate computed with 12 calibration states)?
   Is anything understated? Verify that no unit test can now generate or read a real calib/test state:
   try to find a path around tests/conftest.py.

B. Your fourth report, B1-B8, C and D: for each, resolved / partly / not against the committed code and
   the new protocol text (file:line). In particular try to break:
   1. TestStore: construct it or read a test label without a valid claim + freeze; evaluate twice;
      evaluate concurrently; evaluate after a crash. (Reason about it or use the unit-test fixtures;
      do not touch real test labels.)
   2. check_record / load_run / evaluation_record / e2_record: accept a record of the wrong fit, of
      another freeze, with other inputs, or a per-state file that does not match its hash.
   3. claim_run_folder / rerun_reason / halting: rerun without an infrastructure reason; two owners;
      a folder without an owner; a defect that does not halt; a halt that a final command ignores.
   4. freeze / check_freeze: replace a freeze; run a final command with an uncommitted freeze, on the
      CPU or another GPU, with other library versions, or with an input changed during the run.
   5. timing / report: an unsafe clearance in a later repeat that the report misses; a no-model PG-net;
      core count; worker readiness; what is inside the timed region for each series.
   6. envelopes / report: per-direction fallback share; incomplete envelope records accepted?
   7. The Slurm scripts: skip logic (exit code 10), failure propagation, the dependency chain in
      hydra_fourwire_scale_submit.sh, anything that silently skips or repeats a run.

C. New defects introduced by these changes (any module). Highest risk first.

D. The new protocol text, parts B-D: any statement the code does not implement, or any behaviour of the
   code the text does not state? Anything still ambiguous enough to change a reported number?

E. Verdict: ready to freeze once the A100 dry run passes, or the list of blockers in order.
