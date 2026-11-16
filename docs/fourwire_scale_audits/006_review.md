Checked `6013516` read-only, using small in-memory probes. No fixtures or prohibited labels were read.

1. **C1 — Not resolved completely.** The lock fixes the original concurrent-retry race. But an **invalid record bypasses owner-liveness checking**. Sequence: A writes `valid:false`, pauses before raising/HALT; B, already past preflight, archives A’s folder and claims it while A remains alive. [train:834–849](src/gencodoe/fourwire_scale_train.py:834).

2. **C2 — Not resolved under concurrent completion.** Sequential crash recovery works. Remaining sequence: A’s `completed()` returns false; B completes and dies before publication; A reaches `claim_run_folder`, which raises `AlreadyDone` without publishing. The job can report a successful skip with no public result. This sequence does **not** reread labels. [eval:375](src/gencodoe/fourwire_scale_eval.py:375), [train:834](src/gencodoe/fourwire_scale_train.py:834). E2 has the same sequence.

3. **C3 — Not resolved under that race.** The normal skip paths validate the requested artifacts. The separate `AlreadyDone` inside `claim_run_folder` checks only `record["valid"]`, bypassing those validators and still reaching exit 10. Reproduced in memory. [train:834](src/gencodoe/fourwire_scale_train.py:834).

4. **B2 — Resolved within the stated guards.** Construction and every load check freeze, current ownership, absence of a record and the expected `fits_checked` list. [eval:71](src/gencodoe/fourwire_scale_eval.py:71).

5. **B3 — Resolved.** Canonical identity checks now reject the wrong seed, budget, pool, source, software and inputs; checkpoint presence/hash is checked. My in-memory mutations were rejected. Earlier-freeze acceptance follows the disclosed policy. [train:671](src/gencodoe/fourwire_scale_train.py:671), [train:736](src/gencodoe/fourwire_scale_train.py:736).

6. **B4 — Resolved for the requested checks, with one wording exception.** Bare `CANCELLED` is refused **unless `GENCODOE_RERUN_DEVIATION` is set**. Rerun-limit HALT and collect/report refusal are implemented. See item 10 for a new ordering defect. [train:794](src/gencodoe/fourwire_scale_train.py:794), [train:939](src/gencodoe/fourwire_scale_train.py:939).

7. **B7 — Resolved.** End checks include the job script and stage-specific variants/profiles; optional backends are recorded. The implementation follows the disclosed cuDNN policy. [train:154](src/gencodoe/fourwire_scale_train.py:154), [train:527](src/gencodoe/fourwire_scale_train.py:527).

8. **B8/D10 — Resolved.** The validator checks all **576 unique intervals**, settings and recomputed summary. [E2:312](src/gencodoe/fourwire_scale_e2.py:312).

9. **Disclosure/protocol — Not entirely resolved.** Parts A/B address the disclosure gaps. Independently recomputed committed medians are **0.083153, 3.116456, 0.198738 and 6.026336 seconds**, with **55** sweep fallback solves; fallback shares span **0.350333–0.397**. These support the new rounded numbers. However, C1–C3 overstate the remaining guarantees above, and **C8 should state the declared-deviation exception**. [protocol:680](docs/fourwire_scale_protocol.md:680).

10. **New defect — premature global HALT.** With two archived attempts and the third **still running**, a duplicate claimant writes HALT before checking that owner’s liveness. No third failure has occurred. Reproduced in memory. Move the liveness check before the rerun-limit check and invalid-record archival. [train:838](src/gencodoe/fourwire_scale_train.py:838).

**Remaining blockers, in order: put completion validation/publication inside the claim transaction; check owner liveness before archival or rerun-limit HALT; correct the corresponding protocol guarantees and cancellation exception.**