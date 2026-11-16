**Not ready to freeze.** Several promised guards are absent from the implementation.

I ran **10 non-fixture test functions** under a write-blocking guard; all passed. Three-state CPU train/deploy checks matched exactly on networks **25 and 8**, with U1 differences below **2.4e−16 pu**. I derived **276 fits: 12/24/120/120**. No test-label file was read by this audit.

**A. Status against the third report**

| Item | Verdict | Evidence |
|---|---|---|
| C1: attempts/retries | **Partly** | Policy tightened; retry eligibility is not enforced. `fourwire_scale_train.py:664–749` |
| C2: post-freeze corrections | **Partly** | Rerun scopes are explicit, but records lack compatible freeze lineage. `train.py:579`, `eval.py:321` |
| C3: numerical mode/terminal records | **Partly** | Mode checks improved; terminal outcomes fail downstream. `train.py:692`, `e2.py:232` |
| D1: directional confidence bounds | **Resolved** | Directional denominators and CP bounds implemented. `eval.py:180–195` |
| D2: procedure/scaler comparison | **Resolved** | FT preserves source scales; scratch fits B-only statistics. `train.py:277–302` |
| D3: numerical edge cases | **Partly** | Defined in text; missing-model/empty-metric handling incomplete. `eval.py:216,464`; `e2.py:232` |
| D4: inherited conventions | **Resolved, except wording** | Epoch zero excluded; metric changes disclosed. `train.py:347–405`; protocol:500 |
| E1: delivered-decision safety | **Partly** | Only the first timed repeat is scored. `e2.py:297–307,529–552` |
| E2: envelope departure | **Resolved** | Explicitly sweep-relative, without OpenDSS safety claims. Protocol:624 |
| E3: worker history | **Resolved** | Reset occurs inside each timed chunk. `e2.py:73–84` |
| E4: bootstrap/order/repeats | **Resolved** | Both bootstrap schemes and rotating order implemented. `e2.py:288–315,473–493` |
| E5: execution departures | **Partly** | Mostly disclosed; input format, screening location and resource enforcement need closing. |
| E6: envelope construction | **Partly** | Construction/check precedence match; directional fallback reporting is missing. `e2.py:348–447` |
| E7: claim wording | **Resolved in text** | Appropriate rescue-dependent wording. Protocol:608–610 |
| F1: freeze coverage | **Partly** | Broader hashes added, but enforcement and upstream-record binding remain incomplete. `train.py:530–608` |
| F2: implementation/exposure controls | **Not resolved** | See B1–B6 below. |

Here, `train.py`, `eval.py`, and `e2.py` denote the corresponding `fourwire_scale_*` modules.

**B. Findings, highest risk first**

1. **The fixture suite is not test-data-free.** The `study` fixture generates genuine calibration states and **12 genuine network_25 test states**, using the frozen test seed **25003**. Those states then enter model evaluation. Temporary directories and mocked provenance do not change their split identity.

   **If these fixtures have been executed, the claim that model code has not evaluated test labels needs correction.** Establish and disclose the execution history; do not silently replace or drop exposed states. Future fixtures should derive their pretend held-out data from train/val.

   Evidence: [test fixture:32–41](tests/test_fourwire_scale_eval.py:32), [evaluation test:288–335](tests/test_fourwire_scale_eval.py:288), [label generation:470](src/gencodoe/fourwire_scale.py:470).

2. **Test access and “evaluate once” are not enforced at the reader boundary.** `TestStore(root)` is publicly constructible without checking freeze, fit completion or evaluation authorization. I constructed it without those prerequisites, obtaining its **24 test-manifest entries**, without opening labels.

   `evaluate()` checks whether its output exists, but does not reserve the evaluation before reading labels. Concurrent calls—or a crash before writing the record—permit another test read. Timing/envelopes also open individual evaluation records without requiring all 24 records first.

   Evidence: [TestStore:50](src/gencodoe/fourwire_scale_eval.py:50), [evaluate:301–334](src/gencodoe/fourwire_scale_eval.py:301), [E2 model loading:218](src/gencodoe/fourwire_scale_e2.py:218).

3. **Records are insufficiently bound to model identity and freeze.** `load_run()` checks `valid` and checkpoint bytes, but not expected method/network/budget/seed, input hashes or freeze identity. FT accepts a valid PG-pre record without validating its freeze. Evaluation checks source SHA only; timing’s `_models()` does not even require that match.

   Consequently, stale records from another freeze with identical source—or misplaced compatible checkpoints/records—can be accepted. PG-pre seeds 1–2 receive weaker validation through the FT-chain check than seed 0 does.

   Recorded NPZ hashes are also **not verified by the E2 report** before scoring their contents.

   Evidence: [train:611–617](src/gencodoe/fourwire_scale_train.py:611), [train:710–717](src/gencodoe/fourwire_scale_train.py:710), [eval:318–328](src/gencodoe/fourwire_scale_eval.py:318), [E2 report:526](src/gencodoe/fourwire_scale_e2.py:526).

4. **Retry authorization is missing, and attempt ownership is racy.** `claim_run_folder()` archives an incomplete attempt and starts another without checking whether accounting permits it. Accounting is queried **after the rerun**, only for reporting. I reproduced acceptance without any accounting authorization.

   There is a creation window between `mkdir()` and `attempt.json`: another claimant treats the ownerless directory as stale and can rename it. Queue-query failures are not fail-closed. Array-task identifiers are stripped when querying accounting.

   The Slurm script compounds this: **any existing `record.json` is skipped**, including invalid/stale records, and the stage can exit successfully. Unsupported-operation/repeated-OOM defects do not trigger the promised global halt.

   Evidence: [claim:623–689](src/gencodoe/fourwire_scale_train.py:623), [accounting after fitting:735](src/gencodoe/fourwire_scale_train.py:735), [Slurm:41–46](scripts/hydra_fourwire_scale_final.slurm:41).

5. **The E2 safety criterion can miss unsafe later repeats.** Timing saves the first full-batch decisions; later repeats contribute only a “different decisions” counter. The report scores that first saved array.

   My synthetic reproduction returned **`genco_faster=true` despite a recorded later decision change**. The implementation cannot determine whether that change introduced a false-safe or cleared-indeterminate outcome. Store/score every repeat, or sufficient state-level aggregates to detect any unsafe clearance.

   Evidence: [timing:297–307](src/gencodoe/fourwire_scale_e2.py:297), [scoring:529–552](src/gencodoe/fourwire_scale_e2.py:529).

6. **Promised terminal outcomes break later stages.** A valid PG-net record without a model cannot reach timing/envelopes: `_models()` calls `load_run()`, which requires a checkpoint. It should implement the specified all-deferred procedure.

   Separately, `state_metrics()` returns counts without a `vpn` key when no prediction is finite. The E3 report then raises **`KeyError: 'vpn'`**; I reproduced this in memory. Missing metrics must remain missing throughout reporting.

   Evidence: [E2:232–236](src/gencodoe/fourwire_scale_e2.py:232), [state metrics:216](src/gencodoe/fourwire_scale_eval.py:216), [E3 lookup:464](src/gencodoe/fourwire_scale_eval.py:464).

7. **Freeze validation has material holes.** `freeze()` can replace an existing freeze and does not require its file to be committed. `input_hashes()` returns the manifest’s expected constant rather than hashing its actual bytes. Final CPU execution is accepted—I reproduced this—and the GPU check accepts any A100 variant.

   Input hashes/software are read again for the final record without checking them against the starting freeze. An identification record changed during fitting can therefore be **mislabelled as the input used**, while `valid` remains true.

   Evidence: [input hashes:530](src/gencodoe/fourwire_scale_train.py:530), [freeze/check:562–608](src/gencodoe/fourwire_scale_train.py:562), [record:741–749](src/gencodoe/fourwire_scale_train.py:741).

8. **Remaining execution/reporting discrepancies.** Timing starts from prepacked complex `S`, with P/Q packing outside the timed region. Screening actually runs on CPU after summary transfer. GPU sweep output additionally transfers iteration counts. Document these choices or align them with the stated boundary.

   The required **16 threads/processes are recorded but not enforced**. Worker-startup mapping does not guarantee every worker finished initialization before proceeding. Envelope fallback share is stored globally, not per direction. E2 reporting accepts absent or unchecked envelope summaries.

   Evidence: `e2.py:87–98,169–182,197–209,258,445–446,562–563`.

For ordinary correctly matched records, I found no structural train/deploy mismatch beyond rounding. FT weights/scalers, B-only scratch statistics, calibration masks, rating flags and directional metrics follow the specification. Test truth enters `exact_safe` as an **agreement reference**; it does not feed normal screen calibration or actual timed decisions. The ordinary E3 formula recovered a synthetic **3-point effect over 23 fresh networks** correctly; bootstrap and break-even formulas also look correct.

**C. Tests remain weaker than the promises**

- Isolation tests exercise serial normal calls, not direct `TestStore` access, concurrent evaluation or crash recovery.
- Retry tests do not test accounting eligibility, ownership races, cancellation attribution or stale freezes.
- Report tests do not cover unsafe later repeats, sidecar corruption or empty finite-accuracy results.
- No-model testing covers failed PG-pre, not terminal PG-net through E2.
- “Every series delivers the same exact decisions” does not compare every returned decision with OpenDSS; several assertions check internal consistency.
- GPU parity, worker readiness and measured-region completeness remain untested by these CPU fixtures.

**D. Remaining protocol issues**

- “Epoch 0 … cannot favour PG-ft” is too categorical. Exclusion can harm scratch more than FT and thereby favour FT relatively. The inherited rule is clear; the neutrality claim is not justified. Protocol:500.
- Limited rerun scopes conflict with whole-tree source matching: an evaluation/report-only fix changes the global source SHA, making unchanged fits unacceptable under `evaluate()`. Define versioned, stage-compatible reuse.
- “Every delivered decision” must explicitly include all measured repeats and match B5’s implementation.
- Required per-direction envelope fallback shares and complete/valid envelope-report prerequisites need implementation.

**E. Verdict**

**Do not freeze yet.** Close B1–B7 first, then the execution/reporting omissions and protocol wording. Rerun corrected fixtures using train/val-derived data, and review the A100 dry-run evidence before freezing.