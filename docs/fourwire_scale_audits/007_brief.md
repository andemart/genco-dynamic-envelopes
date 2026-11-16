Seventh audit round, very narrow. Read-only. Do not run the fixture tests; do not read labels/test or
labels/calib. Only this:

HEAD = e85164c (tracked files = HEAD, stable while you read). Since 6013516:
- d060d77 docs/fourwire_scale_protocol.md: appended "third note on the run claims".
- e85164c src/gencodoe/fourwire_scale_train.py (claim_run_folder, final_fit), fourwire_scale_eval.py
  (evaluate), fourwire_scale_e2.py (_run_files), tests/test_fourwire_scale_eval.py.

Check your sixth report's items 1, 2, 3, 9 and 10 against this commit:
1. Is the owner's liveness now checked before anything of a folder is set aside, validated or counted
   (invalid record of a live owner; rerun-limit halt with a live third attempt)?
2/3. Does every skip (exit 10) of fit, evaluate, timing and envelopes now pass through validation of the
   record and the files it names, and publication, inside the claim's lock? Any remaining path to exit 10
   or to an unpublished completed run?
9. Does the "third note" state what the code does (including the declared-deviation exception)?
10. Did this change break anything else in claim_run_folder or its three callers?

Answer per item in one or two lines with file:line, then one line: ready to freeze once an A100 dry run
of this source ends with DRY_RUN_OK and has been read, or the remaining blockers.
