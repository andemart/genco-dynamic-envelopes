**Not ready to freeze.** The fixes close several failures, but concurrent retries can still start two owners, completion can be silently skipped after a crash, and record validation remains incomplete.

**A. Disclosure — partly complete.**

The history supports **12 unique network-25 test states and 24 calibration states** across networks 10 and 25. The percentages are **0.292969%** and **0.012207%** of the network’s and combined eligible test sets.

With at most 12 converged calibration states, every margin is infinite and every screen defers. However, evaluation also computed and saved **state-accuracy metrics, rating sensitivity, sweep agreement and per-state outputs** in temporary files (`73442a2:src/gencodoe/fourwire_scale_eval.py:373,389,406`). The disclosure should explicitly enumerate those computations. Scratch/FT used **12 training states**, while PG-net/PG-pre used 24. The statements about approximately ten executions and what was displayed, retained or used cannot be established from git.

Current fixtures use val-day states and check their calendar membership ([test_fourwire_scale_eval.py:64](tests/test_fourwire_scale_eval.py:64)). But **conftest does not establish universal isolation**: it patches two `customer_states` bindings, without guarding label-file reads. `FluviusScenarios.__call__` uses `split_steps`/`draw_state` directly and bypasses that wrapper ([conftest.py:17](tests/conftest.py:17), [datakit_fluvius.py:181](src/gencodoe/datakit_fluvius.py:181)). I found no current fixture using these bypasses; I did not generate held-out states or open prohibited labels.

**B. Previous findings**

| Finding | Verdict | Evidence |
|---|---|---|
| **B1: test exposure** | **Partly resolved** | Exposure is disclosed and the offending fixtures now use val days. The disclosure and universal guard claim need the qualifications above. |
| **B2: TestStore/isolation** | **Partly resolved** | Construction checks freeze and ownership. `load()` checks only network identity: an existing reader remains usable after its record is written, ownership changes or the freeze changes. An owned claim also permits construction before fit validation. [eval:65](src/gencodoe/fourwire_scale_eval.py:65), [eval:77](src/gencodoe/fourwire_scale_eval.py:77). |
| **B3: record identity/provenance** | **Partly resolved** | Checkpoint and consumed sidecar hashes are checked. But I reproduced `check_record` accepting PG-net with **seed 2, budget 200, wrong pool, source and software**, under the expected name/freeze. `run_name("pgnet",…)` discards seed/budget; pool/source/software are unchecked. Evaluation/E2 record readers also omit input validation. [train:659](src/gencodoe/fourwire_scale_train.py:659), [eval:338](src/gencodoe/fourwire_scale_eval.py:338), [E2:289](src/gencodoe/fourwire_scale_e2.py:289). |
| **B4: attempts, halting, scripts** | **Partly resolved** | Fresh claims are published with an owner; accounting and HALT checks exist. Retry concurrency and completion publication remain broken—C1–C2 below. Bare **`CANCELLED`** is accepted without identifying who cancelled it. The rerun-limit exception occurs before `halting`. [train:728](src/gencodoe/fourwire_scale_train.py:728), [train:764](src/gencodoe/fourwire_scale_train.py:764). |
| **B5: later-repeat unsafe clearance** | **Resolved** | `ever_safe`/`always_safe` aggregate all measured whole-split repeats; the report uses them for safety. [E2:338](src/gencodoe/fourwire_scale_e2.py:338), [E2:612](src/gencodoe/fourwire_scale_e2.py:612). |
| **B6: terminal/no-model cases** | **Resolved functionally** | No-model PG-net defers in timing and envelopes; missing nested accuracy metrics remain missing. [E2:185](src/gencodoe/fourwire_scale_e2.py:185), [E2:484](src/gencodoe/fourwire_scale_e2.py:484), [eval:507](src/gencodoe/fourwire_scale_eval.py:507). |
| **B7: freeze enforcement** | **Partly resolved** | Existing freeze, uncommitted freeze, CPU/wrong GPU and listed-version changes are refused. End checks omit **job-script, rating-variant and profile hashes**. The software dictionary also omits recorded cuDNN and optional native backend versions. [train:154](src/gencodoe/fourwire_scale_train.py:154), [train:521](src/gencodoe/fourwire_scale_train.py:521), [train:619](src/gencodoe/fourwire_scale_train.py:619). |
| **B8: timing/envelope completeness** | **Partly resolved** | Packing/host screening are disclosed; GPU sweep transfers, 16-core enforcement and worker readiness are fixed. Directional fallback shares exist. Envelope completeness still trusts a flag rather than validating the required intervals/configuration. [E2:95](src/gencodoe/fourwire_scale_e2.py:95), [E2:523](src/gencodoe/fourwire_scale_e2.py:523), [E2:581](src/gencodoe/fourwire_scale_e2.py:581). |
| **C: tests** | **Partly resolved** | New checks cover several formerly missing cases. They still omit concurrent retries, reader revocation, publication crashes and complete record schemas. The synthetic “complete” envelope records contain **no intervals**, yet pass the report test. [test_e2:225](tests/test_fourwire_scale_e2.py:225). |
| **D: protocol wording** | **Partly resolved** | Epoch-0 neutrality is withdrawn and timing choices are disclosed. Enforcement claims remain stronger than the code, as detailed below. |

The Slurm dependency graph is correct, and Python failures propagate to the job’s exit status. However, exit **10** does not consistently mean “validated completed run.” The optional dry-run dependency also does not prove that the complete probe passed on the same commit ([final.slurm:44](scripts/hydra_fourwire_scale_final.slurm:44), [submit.sh:21](scripts/hydra_fourwire_scale_submit.sh:21)).

**C. New implementation defects, highest risk first**

1. **Concurrent retries can corrupt an active run.** Both callers can read the old owner before either archives it. Caller A archives and starts; caller B then writes the old ownership into A’s new folder, archives A, and starts too. I reproduced **both calls returning success** with an in-memory filesystem. Both fits subsequently address the same active pathname. Lock the entire inspection/archive/claim transaction. [train:758](src/gencodoe/fourwire_scale_train.py:758), [train:776](src/gencodoe/fourwire_scale_train.py:776).

2. **A publication crash becomes a successful skip.** Evaluation/E2 first write the folder’s valid record, then copy it to `results/`. A crash between those operations leaves no public result; retry encounters the valid folder record and exits 10. A partial public copy also triggers an existence-only skip. Recover by validating and atomically publishing the completed record, without reevaluating labels. [eval:482](src/gencodoe/fourwire_scale_eval.py:482), [E2:280](src/gencodoe/fourwire_scale_e2.py:280), [train:760](src/gencodoe/fourwire_scale_train.py:760).

3. **Skip checks bypass record validation.** Fit skipping trusts `record["valid"]`; evaluation/timing/envelope skipping trusts output existence. Wrong-freeze records, damaged checkpoints or truncated output can therefore produce a successful skip. Apply the complete record validator before returning 10.

**D. New protocol text, parts B–D**

1. **D3/D4 overstate authorization and validation.** Test authorization expires only for newly constructed readers. Validators need canonical fit tuples, expected pools/configuration, stage inputs and stage provenance—not merely a matching declared freeze hash.

2. **C2/C5 overstate enforcement.** `collect` and both reports ignore HALT; exceeding the rerun limit does not create one. Stage-specific inputs are not rechecked at run end. Narrow the statements or implement them.

3. **D10 completeness is insufficiently checked.** Validate the exact **576 unique `(day, step, direction)`** entries, configuration and recomputed summary. Counting rows alone also permits six days × 48 hours to look complete through the Python API.

4. **Part B’s timings lack matching retained evidence.** The available network-1 dry record gives medians **0.08315 s / 3.11646 s** for sweep without/with OpenDSS, **0.19874 s** for U1 and **6.02634 s** for PG-net. It records **55** OpenDSS solves. Fallback shares **0.35033–0.397** corroborate that range. The quoted 0.07/2.6/0.17/5.7 may describe an earlier execution, but need its record or an explicit unarchived-measurement qualification. Evidence: `timing_network_1_20261115T234624.json → batches.3000.*`; `network_1_20261115T234431.json → methods.*.screen.fallback_share`.

**E. Blockers before freezing, in order**

1. Serialize retries and prevent active-folder replacement.
2. Validate completed records before skipping; make publication recoverable and atomic.
3. Recheck TestStore authorization on reads and strengthen held-out test guards.
4. Complete record schemas, end-of-run freeze checks, halt enforcement and envelope validation.
5. Add regression tests for these failures and finish the disclosure corrections.

Five read-only, fixture-free function checks passed. The file-writing fixture suite and CUDA execution were not run.