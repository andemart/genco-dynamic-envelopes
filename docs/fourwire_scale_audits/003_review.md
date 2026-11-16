**Not ready to freeze:** the remaining material issues concern run selection, matching safety assessment to timed execution, and the envelope experiment.

Code citations below refer exclusively to `ee2479f`; protocol citations refer to the working draft.

| A item | Verdict | Evidence / remaining issue |
|---|---|---|
| A1 | **Partly resolved** | Source hashing and checked Git calls fix the earlier dirty-boolean problem for training. Initial dirty source is still accepted by `start_provenance`; final commands must enforce Amendment 7’s stricter rule. The integrity checker compares provenance dictionaries without a source hash. `fourwire_scale_train.py:63–118`; `fourwire_scale.py:608–622`. |
| A4 | **Resolved within the stated limits** | Manifest digest matches **5625aa…b7dc9**. Descendant links, hard links and traversal are rejected. Trusted-root and check/open-race limitations are now disclosed accurately. `fourwire_scale_data.py:39–59`. |
| A5 | **Resolved as documentation/metadata** | Effective `scheduler_t_max` is recorded. The inherited batch-dependent validation score remains deliberately unchanged. `fourwire_scale_train.py:331–333`; protocol:413,403. |
| C2 | **Resolved for fresh command processes** | Settings precede CUDA queries; deterministic mode is explicitly enabled/disabled; device properties use the requested device. `fourwire_scale_train.py:80–111`. |
| C3 | **Partly resolved** | Invalid checkout, errors and incomplete/nonfinite training losses block verdicts. However, validation-loss finiteness/completeness is unchecked, and build matching omits NumPy/Python and numerical settings. I demonstrated that two records containing **infinite validation losses** receive `valid=true` and `identical_val_losses=true`. `fourwire_scale_train.py:443–471`. |
| C4 | **Resolved for the scheduled single-writer workflow** | Fold enters smoke filenames; probe folders refuse reuse; requeue is disabled. Check-then-write still permits concurrent replacement, consistent with the newly stated limitation. Trainer:385–430; determinism script:13,27–37. |
| B5 | **Resolved for the specific deficiencies identified** | Tests now cover hard links, actual Torch SD operations, isolated provenance corruption and cross-unit commit disagreement; CPU-test names accurately state their scope. Training tests:123–147,223–245; label tests:95–112. |

A remaining device bug: memory reset/query calls still use the **current CUDA device**, rather than `dev`. A `cuda:1` run can report device-0 memory. `fourwire_scale_train.py:280,317–318`.

| B item | Verdict |
|---|---|
| B1 | **Resolved.** Corrections accurately distinguish gradient/scaler isolation from validation selection, and disclose the auditor’s test-label integrity reads. Protocol:420–422. |
| B2 | **Resolved by archived evidence.** I derive **473 seconds = 7m53s** after Amendment 5 and **206 seconds = 3m26s** before checkout movement. Both jobs identify node401; log curves match the smoke JSONs. Minor wording: the reports record Torch **2.11.0+cu126**, although not the complete dependency stack. Evidence file:2–24; protocol:424–428. |
| B3 | **Resolved as a qualified operating guarantee**, with A1’s implementation limits. Protocol:430–434 now acknowledges restored checkouts, trusted roots and competing writers. |
| B4 | **Resolved.** Identification records give **385–500 fitted states**, fewer than 500 in exactly **7/24** eligible networks, matching the listed networks/counts. The unsupported timing correction factor is withdrawn. Protocol:436–438. |

The archived probe’s SHA-256 matches. Independently comparing its arrays gives identical initial weights, default-mode divergence at **step 3**, maximum loss difference **6.6757202148e−5**, and identical deterministic-mode weights, training losses and validation losses across **32 steps**. The limited interpretation in corrections item 5 is appropriate.

**C1 — High: cancellation and retry rules permit outcome-dependent selection.** “Cancelled job” is classified as infrastructure failure without distinguishing scheduler/site cancellation from voluntary cancellation after inspecting losses. Require an attempt registry, one active attempt per tuple, and externally documented failure reasons. Define treatment of completed **invalid** records, unsupported deterministic operations, nonfinite validation losses and repeated OOMs. Protocol:450–453.

**C2 — High: reconcile “never rerun” with later corrections.** Amendment 8 permits rerunning affected fits after defects. Define objective dependency-based invalidation across methods/networks/seeds, preserving every attempt and freeze version. Otherwise “affected” can become a favourable-run selection mechanism. Protocol:450,534.

**C3 — Medium: deterministic settings do not pin the software experiment.** The prospective rule appropriately avoids promising full-fit bit equality. It should nevertheless pin one software stack and verify effective settings at completion. Define terminal failure records, including absent checkpoints and failed pretraining dependencies, so those cases cannot be reclassified as missing-record retries. Protocol:449–458.

**D1 — High: pooled Clopper–Pearson intervals are unsupported.** Item 9 first promises directional metrics, then pools `(state,direction)` pairs for a binomial confidence bound. The two directions share the same electrical state and are correlated. Keep directional rates/intervals; aggregate counts may be descriptive. A pooled interval needs an explicitly justified state-level uncertainty procedure. Protocol:506–511.

**D2 — Medium: this compares method packages, not pretrained weights alone.** PG-ft retains source input **and loss-normalization scales**, while scratch estimates its own scales from B states. Thus their effective objectives differ despite identical nominal loss coefficients. Conversely, scratch gets target-adapted normalization. Neither guarantees an advantage. Primary FT/scratch calibration sets are matched; PG-zs uses up to **36,000** source calibration candidates and has different resources and calibration-population assumptions. State these limitations when interpreting E3. Protocol:473–479,501.

**D3 — Medium: several numerical cases remain ambiguous.**

- “Indeterminate states enter no target statistic” must distinguish **output-target scales** from input-scaler statistics: the inherited input scaler includes their inputs.
- Specify whether **any single nonfinite calibration entry** disables the whole screen; the present sentence also admits an “all entries unusable” reading.
- Define empty finite-prediction metrics, nonfinite validation losses, zero scales and absent pretrained checkpoints.
- PG-pre-tc represents FT’s starting checkpoint only for **seed 0**, not FT seeds 1–2.

Protocol:477–491,503–520; trainer:212–216.

**D4 — Lower risk: inheritance needs precise qualification.** Epoch-zero exclusion does match ENWL1 and can penalize FT when its starting model is strongest; it does not secretly favour FT. The new transformer metric intentionally changes ENWL1’s proxy-to-proxy metric to proxy-versus-authoritative-OpenDSS. Finite-only accuracy also changes the evaluated population and requires prominent coverage reporting. Protocol:490,514–519; `fourwire_train.py:203–235`; `fourwire_eval.py:105–112`.

The **399 threshold is correct**: I executed committed `GlobalMargins`. At **398**, bounds are infinite; at **399**, they are finite for finite calibration values. `fourwire_eval.py:146–156`.

I reran the val script’s calculations on **all 24 networks**, using committed modules with cache/output writes disabled and no test-label reads. Every JSON row matched exactly:

- **1,577** sweep failures; **1,547** OpenDSS-indeterminate states; **30** sweep-failed/OpenDSS-converged states; **0** reverse cases.
- Criterion disagreements: **0/0/0/73**; transformer false-negative flags: **58**, false-positive flags: **15**.
- Crucially, **58 is not the number of directionally false-safe states**. My added complete-decision check found **zero false clears**, **two false rejections for import**, and **zero export disagreements** when sweep failures are rejected.
- All **30** rescued states were unsafe in **both** directions.

**E1 — High: safety and speed concern different execution paths.** Safety comes from deterministic screening; timing uses default algorithms and merely reports differences. It also excludes fallback decisions from the primary safety metrics. Assess the actual winning timed variant’s complete decisions—including fallback replacements—before claiming zero observed pipeline false-safe outcomes. Preserve the preregistered screen metric separately. Protocol:512,526,563,566,573.

**E2 — High: the envelope departure is not validated by these val results.** The val evidence is reassuring: omitting OpenDSS rescue introduced no additional directional false rejections there. But capped random states do not establish equivalence on uncapped envelope-search scenarios. This is defensible as a disclosed sweep-relative experiment, not proof of OpenDSS envelope safety. “Can only lower an envelope” additionally assumes the search’s monotonicity; screened feasibility need not satisfy it. Protocol:586–591.

**E3 — High: worker history contradicts other execution statements.** Persistent OpenDSS solutions are reused across batches and potentially series, despite “nothing computed … is reused.” This also needs an explicit exception to §3’s no-warm-start wording. Worker assignment to chunks, series-specific engine ownership and resets must be fixed: history affects both timing and tolerance-level answers. Protocol:551,565.

**E4 — Medium: selection is symmetric, uncertainty/order remain problematic.** Taking the minimum of two variants on **both** sides and recomputing minima inside bootstrap replicates is reasonable. Independent resampling discards round-level correlation from round-robin execution. Prespecify a paired-round sensitivity or justify the independence assumption. Enumerate the actual series order; fixed position can produce systematic thermal/cache bias. Reduced OpenDSS repeats and omitted NumPy batch sizes are disclosed departures, but do not support comparable tail-latency precision or batch-independent extrapolation of fallback-inclusive NumPy time. Protocol:552–555,568–572.

**E5 — Medium: additional departures are not flagged.** Explicitly identify:

- prohibition of sweep microbatching while allowing model microbatching, contrary to Amendment 3’s allowance for any method;
- changed numerical path between calibration/safety evaluation and timing;
- replacement of both directional screen outcomes after a fallback requested by only one direction;
- rotating repeat inputs and the narrower screen-only interpretation of safety.

The union rule is computationally sensible—one physical solve answers both directions—but its overrides require complete-pipeline accounting. Protocol:547,553,560–566,573.

**E6 — Medium: envelope text disagrees with its inherited function.** `scalar_envelope` returns a feasible **cap before checking the zero-level failure**. The draft promises zero whenever the zero level fails. Choose and disclose the precedence. Also, **576** envelopes means **288 per direction**: specify whether the 95% criterion is combined or directional. Fix day-of-year indexing, customer/group iteration and meter-draw order; the seeds alone do not settle the modified sampler. Dates and the **0.98** acceptance threshold check out. Protocol:581–592; `envelopes.py:175–179`.

**E7 — Claim wording remains insufficiently specific.** The predicted cause of any speed-up is a hypothesis, not established by mismatched informational timings. Always report the rescue-free comparison and its decision differences. If gains disappear there, the paper should say **“faster than the sweep pipeline with mandatory OpenDSS rescue”**, explain that observed val rescues changed no safe decisions, and avoid attributing the gain to faster power-flow prediction or general solver superiority.

**F1 — High: freeze more than Python source and label hashes.** Bind the identification record, network files, profile inputs, configurations, scripts and software environment. Verify regenerated timing inputs against committed input hashes. Define the convergence/status field for model-screened states that receive no exact solve. Protocol:495,529.

**F2 — High: close implementation and exposure controls before freezing.** Require valid terminal records for failures and dependencies, finite validation checks, and parity tests for device U1/features and all timed fallback variants. Define how post-freeze corrections invalidate prior fits/evaluations across the study; per-network test release must not enable selective corrections to later networks. The concurrently edited implementation was outside this committed-code audit.