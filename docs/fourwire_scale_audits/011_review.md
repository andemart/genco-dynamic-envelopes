1. **Equivalence gate: resolved.** [start:93](scripts/fourwire_scale_pgpre_stabilised.py:93) enforces the frozen environment and committed script without `--dry`; [equivalence:388](scripts/fourwire_scale_pgpre_stabilised.py:388) requires equality, completion, no numerical failure and unchanged provenance. The prescribed Slurm chain propagates failure and uses `afterok`.

2. **Output isolation: partly resolved.** Exclusive creation and the outside-work-folder rule fix both previous examples; full fits are restricted to this checkout’s stabilised root. However, [guard:68](scripts/fourwire_scale_pgpre_stabilised.py:68) checks only **resolved** path components. If another checkout’s `runs/dry` is a symlink to `/scratch/held_runs`, passing that `runs/dry` as a dry fit’s `--runs` loses the protected name during resolution and passes. The command can then create files there. The same problem applies to symlinked foreign `runs/final`.

3. **Attempts: partly resolved; a HALT-writing race remains.** Normal claims validate completed records, check owners and infrastructure reasons, and collection rejects ordinary hand-renamed attempts. But the [attempt-count pre-check:340](scripts/fourwire_scale_pgpre_stabilised.py:340) is outside the claim lock.

   Concrete sequence: A and B both see one earlier attempt; A claims the third attempt, creating the second earlier folder, then crashes without a record; delayed B enters the claim after A ends. The helper sees two earlier attempts and writes `runs/final/HALT.json` at [trainer:835](src/gencodoe/fourwire_scale_train.py:835). An in-memory execution of these functions reproduced that write.

   Separately, a completed third attempt cannot return the promised exit 10: the same pre-check rejects it before record validation.

4. **Collection: resolved for the prescribed chain.** [check_record:295](scripts/fourwire_scale_pgpre_stabilised.py:295) checks identity, configuration and frozen provenance; collection binds all twelve fits and the gate to one commit/script and checks states, scalers and artifact hashes. Empty-step results now produce no gradient summary at [collect:454](scripts/fourwire_scale_pgpre_stabilised.py:454).

   Remark: collection does not enforce the gate’s prescribed fold, seed, sizes or epoch count. The current Slurm command supplies them correctly.

5. **Protocol corrections: partly resolved.** The retained records support 162 equivalence steps and eight dry epochs of 36 steps. Clipping counts are **34, 34, 36, 35, 34, 34, 33, 33**: **91.67–100%**, so “92%” is rounded. Both recorded script hashes equal the committed script’s SHA-256: `74bd3a149013aa8719d6b70d27fa668f7207933aabba6ccec7406f1e7ef30053`.

   Item 7 overstates isolation because of item 2 above. Item 6 also implies immediate set-aside of an invalid run; [fit:360](scripts/fourwire_scale_pgpre_stabilised.py:360) leaves its invalid record in place until a subsequent claim.

The training body after the folder precondition is AST-identical to round 10. The retained dry record passes the script’s `check_record`; the writer supplies the fields required by its full-size branch.

**Not ready to submit.** Remaining blockers: **(1)** make the retry-limit safeguard atomic and prevent the shared helper’s HALT write; **(2)** close the symlinked foreign-run-directory bypass.