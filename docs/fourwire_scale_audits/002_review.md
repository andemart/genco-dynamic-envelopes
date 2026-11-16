**A1 — Partly resolved: provenance and overwrite protection.** Start-time commit capture and initial/final weight hashes are added. However:

- `checkout_unchanged()` compares HEAD and a **dirty boolean**, not file contents. Different dirty source trees pass; changes reverted before the final check also escape detection. Git command failures are unchecked. Evidence: [fourwire_scale_train.py:59](src/gencodoe/fourwire_scale_train.py:59).
- Run-folder creation correctly refuses an existing folder. Report protection is weaker: concurrent folds sharing a job tag use different run folders but the **same report filename**. `determinism()` unconditionally replaces its output. Evidence: [trainer:352](src/gencodoe/fourwire_scale_train.py:352), [trainer:375](src/gencodoe/fourwire_scale_train.py:375).
- `label_integrity()` still stamps checker provenance at the **end**, preserving the original misattribution risk. Evidence: [fourwire_scale.py:614](src/gencodoe/fourwire_scale.py:614).

**A2 — Resolved as a claim correction.** Amendment 6 explicitly withdraws label-free deployment and discloses target-labelled Thevenin identification, including B=200. This sufficiently resolves the framing finding; the dependence remains in the method. Qualify “first 500” as their **converged subset**. Evidence: [protocol:412](docs/fourwire_scale_protocol.md:412).

**A3 — Resolved prospectively.** Numba elapsed time is captured before reference comparison. Historical reports retain the contaminated measurements and now have an explicit erratum. They remain unsuitable as solve-time measurements. Evidence: [fourwire_scale.py:388](src/gencodoe/fourwire_scale.py:388).

**A4 — Partly resolved: loader isolation; shard-provenance checks resolved.**

- Constructor manifest/commit overrides are gone. Descendant symlinks and filenames containing traversal are rejected.
- The **root itself is resolved and trusted**: root aliases and root-level `..` are accepted. Hard links pass `resolve()==path`; there is also a check/open race. Hash checking protects returned contents under the trusted manifest, but cannot prevent reading redirected bytes first.
- The manifest is a mutable file without a pinned digest. A filename naming another network is rejected; normal paths remain bound to the requested network. Replacing manifest identities/content is not independently authenticated.
- I checked all **720 currently served shards**: none has multiple hard links. I separately read all **936 shard provenances**: all are clean, at `6ad2af6`. The checker now validates these and enforces one commit across units.

Evidence: [fourwire_scale_data.py:31](src/gencodoe/fourwire_scale_data.py:31), [fourwire_scale.py:568](src/gencodoe/fourwire_scale.py:568), [fourwire_scale.py:612](src/gencodoe/fourwire_scale.py:612).

**A5 — Partly resolved through disclosure.** The batch-dependent validation score is retained and accurately described, rather than corrected. It is acceptable as a declared checkpoint-selection score. The shortened smoke cosine schedule is now disclosed correctly. Effective `T_max` is still absent from configuration metadata, which continues to say `max_epochs=60` even for one-epoch probes. Evidence: [trainer:199](src/gencodoe/fourwire_scale_train.py:199), [trainer:227](src/gencodoe/fourwire_scale_train.py:227), [protocol:413](docs/fourwire_scale_protocol.md:413).

**A6 — Resolved as a disclosed numerical difference.** My fresh network_25 calculation gives ENWL1 VPN scale **0.0001481742801843**, pooled scale **0.0001481734584990**, relative difference **−5.5454e−6**. Amendment 6’s “about 6e−6” is supported. Evidence: [trainer:143](src/gencodoe/fourwire_scale_train.py:143), [protocol:415](docs/fourwire_scale_protocol.md:415).

**A7 — Resolved.** Independent reconstruction again gives **276 fits**, projected **34.238465/34.072424 GPU-hours**, with training-time regression R² **0.848116/0.846627**. Calling these empirical fit-only projections accurately states their limitations. Evidence: [trainer:317](src/gencodoe/fourwire_scale_train.py:317).

For B, both archived smoke files are byte-identical to the originals and their stated SHA-256 hashes match. I obtain epoch means **110.8807/110.3011 s**, peak allocation **2,008.8247 MiB**, host peaks **15.44994/15.45108 GiB**, and preparation **143.1946/141.5071 s**. The rounded resource figures and loss curves are supported.

**B1. The opening disclosure overstates the audit.** “No val … value entering training” omits validation’s prescribed influence on checkpoint selection and stopping. Say **no validation labels in gradient losses or scaler fitting**. Also, generator/checker exclusivity is false: the first independent audit directly inspected test-label voltages for integrity diagnostics. Evidence: [protocol:393](docs/fourwire_scale_protocol.md:393).

**B2. Timeline and node assertions lack archived evidence.** The two reports contain no hostname or job-start timestamps; corresponding job logs are absent. Thus “eight minutes,” “four minutes,” the second pin, and “same node” cannot be independently established from these artifacts. Identical committed `src/` is verified; identical runtime dependencies are not. Evidence: [protocol:396](docs/fourwire_scale_protocol.md:396).

**B3. The blanket protection claims are too strong.** “Run fails if checkout moved,” “never overwritten,” and unconditional non-redirection exceed the guarantees in A1/A4. An immutable checkout is currently a stated operating requirement, not something these functions enforce. Evidence: [protocol:406](docs/fourwire_scale_protocol.md:406).

**B4. Two quantitative qualifications are missing.**

- Actual Thevenin fitting subsets contain **385–500 states** across eligible networks; **7/24** use fewer than 500. The disclosure should distinguish candidate solves from fitted observations.
- The old laptop timing fields are **0.46951547 ms** for network_17 and **0.01416551 ms** for network_10. No committed measurement artifact substantiates the new solve-only **0.144/0.004 ms** readings. “About threefold” is consequently an unarchived laptop measurement, not an established correction factor for CUDA-node timings.

Evidence: identification `networks[*].fit.n_states`, [protocol:409](docs/fourwire_scale_protocol.md:409).

**B5. New tests establish narrower properties than their names suggest.**

- Redirect tests cover three constructed symlink/traversal cases, not hard links, root aliases, replacement manifests or races.
- The CPU reproducibility test makes **two fits in one process**, with **two optimizer steps each**, in default mode. It does not test two processes or both modes.
- The VPN comparison uses NumPy sample SD with `rtol=5e-3`; it does not execute ENWL1’s Torch FP32 formula.
- The provenance-tampering test leaves an earlier input corruption unrepaired. Its explicit commit assertions are useful, but overall gate failure is not isolated. No test exercises disagreement **across separate units**.

Evidence: [training tests:112](tests/test_fourwire_scale_train.py:112), [training tests:204](tests/test_fourwire_scale_train.py:204), [label tests:69](tests/test_fourwire_scale_labels.py:69).

**C1. This is a useful falsification probe, not certification of full training reproducibility.** The supplied data do produce **32 steps**: both networks have 16 batches and none is entirely indeterminate. But one epoch on two networks, with scales fitted from 500 states, does not reproduce three epochs on eighteen networks or final training. Two matching executions can miss nondeterminism. Compare **within each mode**; differences between default and deterministic modes do not establish non-repeatability.

**C2. Effective numerical settings are insufficiently controlled.** `start_provenance()` queries CUDA before `pretrain()` sets `CUBLAS_WORKSPACE_CONFIG`. `setdefault` preserves inherited values, and `deterministic=False` never disables previously enabled deterministic algorithms. Recorded mode reflects the request, not queried effective state. GPU properties are always read from device **0**, even for `device="cuda:1"`. Evidence: [trainer:65](src/gencodoe/fourwire_scale_train.py:65), [trainer:169](src/gencodoe/fourwire_scale_train.py:169).

**C3. The summary can validate invalid runs.** It ignores checkout-validity flags, validation losses, effective settings and the required step count. I exercised its comparison in memory: two records with `checkout_unchanged_during_run=false` still produce all three identity flags as true. Every `RuntimeError`, including OOM, is caught and converted into a successful command exit; that must not be interpreted as a deterministic-operation diagnosis. Evidence: [trainer:375](src/gencodoe/fourwire_scale_train.py:375), [probe script:38](scripts/hydra_fourwire_scale_determinism.slurm:38).

**C4. Requeues/repeats overwrite evidence.** The script uses `mkdir -p`, fixed filenames and unconditional report writes. Preserve attempts separately and require valid provenance, finite results and complete comparisons before issuing any reproducibility verdict. Evidence: [probe script:26](scripts/hydra_fourwire_scale_determinism.slurm:26).

D, highest risk first:

1. **Budget and scaling boundaries:** explicitly restrict scratch fitting-derived statistics to B states; fix every input/output/physics scale carried into fine-tuning, including whether any target-derived replacement is allowed.
2. **Fine-tuning and checkpoint selection:** best versus last pretrained checkpoint, trainable layers, fresh versus restored optimizer state, epoch-zero eligibility, tie rules, effective scheduler horizons, deterministic settings, and failure/resume/retry policy.
3. **Screen implementation:** pin the inherited ENWL1 implementation; specify pooled zero-shot calibration assembly and treatment of nonfinite calibration predictions or unusable margins. Keep the already-frozen α and decision rules.
4. **Test evaluation contract:** define error aggregation, complex versus magnitude errors, nonfinite-prediction accounting, and authoritative transformer truth. The old evaluator computes a **proxy-to-proxy** transformer error. Freeze the primary/sensitivity/envelope evaluation bundle and its test-isolation workflow.
5. **E2 execution choices:** freeze input order, CPU allocations, microbatches, engine-selection procedure, OpenDSS initialization/worker lifecycle, joint reuse of fallback solves across directions, and permitted caching. Include all prescribed transfers/features/fallbacks.
6. **Envelope construction:** freeze exact dates, scenario/profile seeds, technology assignments, caps, scenario reuse between methods/search levels, and zero-envelope/failure handling. These details can materially change the reported agreement and unsafe-envelope counts.