Third audit round, same rules: read-only, try to refute, derive numbers yourself. Be concise.

State of the repository (branch ander-fourwire-scale):
- Commit ee2479f: code changes answering your second report (A1, A4, A5, C2, C3, C4, B5).
- Commit 58dd0da: results/fourwire_scale_hydra_job_evidence.txt and results/fourwire_scale_determinism_job13587453.json.
- Commit b2fb9a4: docs/fourwire_scale_protocol.md, "Amendment 6, corrections".
- NOT committed, draft in the working tree: Amendments 7, 8 and 9 at the end of docs/fourwire_scale_protocol.md,
  plus results/fourwire_scale_val_sweep_vs_opendss.json and scripts/fourwire_scale_val_sweep_vs_opendss.py.
- Files under src/ and tests/ are being edited while you read. For code, read the committed state only
  (git show ee2479f:<path>). Read the protocol from the working tree.

A. ee2479f against your second report: for A1, A4, A5, C2, C3, C4 and B5 say resolved / partly / not, with
   file:line evidence from the committed state. Anything new that is wrong?

B. "Amendment 6, corrections": check every factual statement against the repository, the evidence file and
   the two smoke reports. Does it resolve your B1-B4? Anything still overstated?

C. Draft Amendment 7 (deterministic final fits, what counts as a record). Does it resolve the same-seed
   nondeterminism prospectively? Is any rule ambiguous, unenforceable or open to choosing a favourable run?

D. Draft Amendment 8 (final fits, screen, test evaluation, freeze). It is meant to close your D1-D4.
   - List anything still open or ambiguous that could change a reported number.
   - List any rule that favours a method: in particular PG-ft against PG-scratch (the primary E3 criterion),
     epoch 0 not being eligible, scalers, the calibration sets, and the PG-zs pooled calibration.
   - Is any statement inconsistent with sections 4-8, Amendments 5-6, or with the ENWL1 code it says it
     inherits (src/gencodoe/fourwire_train.py, fourwire_eval.py: GlobalMargins, state_metrics)?
   - Check the claim "fewer than 399 converged calibration states gives an infinite margin" against
     GlobalMargins.

E. Draft Amendment 9 (E2 execution, envelope check). It is meant to close your D5-D6.
   - Recompute the val comparison numbers it quotes from results/fourwire_scale_val_sweep_vs_opendss.json,
     and, if you can, spot-check one network with the script (val labels only; do not read test labels).
   - Does any execution choice favour GENCO or the sweep? Look at: both sides taking the faster of two
     variants, the bootstrap, the numerical mode, the reduced repeats for OpenDSS and the NumPy sweep, the
     round-robin order, the union rule for fallback states, and the OpenDSS worker lifecycle.
   - The amendment says in advance that any speed-up would come from the section-3 rule that sends
     non-converged sweep states to OpenDSS, and adds a sensitivity series without those solves. Is that
     disclosure and the claim-wording rule sufficient? What would you require the paper to say?
   - Envelope check: is the departure from section 3 (non-converged = infeasible, no OpenDSS) defensible
     given the quoted val numbers? Is the scenario construction fully determined by the text?
   - List departures from the frozen sections 5-7 and Amendment 3 that the text does not flag as departures.

F. Anything else that must be fixed before code for the final fits is frozen.

Output: verdict per item for A and B, numbered issues for C, D, E, F, ordered by risk.
