**A. Verified.** The diff against `a4068ef` is empty for `src/`, the freeze JSON and the final Slurm script. Recomputed source SHA is `661c1d265b870c57a51df58312930b78d5b0c81c259d24434dd30ceac1c4be87`, matching the freeze. The new analysis script is outside that hash.

**B. Loop computation matches in frozen mode; the gate is insufficient.** RNG draws, batch ordering, loss, scheduler, selection, patience and checkpoint writing match the PG-pre path. Measuring the gradient norm only reads gradients. A synthetic CPU comparison of the extracted loss expressions gave identical losses and gradients in 100 cases.

However, [equivalence:284](scripts/fourwire_scale_pgpre_stabilised.py:284) accepts matching truncated failures: both loops can select epoch 1, fail during epoch 2, and return success. It also ignores `unchanged_during_run:false` when deciding its exit code. An in-memory probe confirmed both omissions. Require four completed epochs, no numerical failure, unchanged provenance and the frozen A100/software environment.

The reduced check cannot exercise patience stopping, later cosine steps, full-size statistics beyond the first 2,000 converged states, the final eight-state training batch, the final 56-state validation batch, or the stabilised switches.

**C. Schedule and clipping are implemented correctly, with two wording qualifications.** At full size, epoch \(e=6,\ldots,60\) trains at
`3e-4 × (1 + cos(π(e−6)/55))/2`; zero is reached after epoch 60. Manual warm-up changes do not corrupt the recursive cosine schedule. Clipping follows the finiteness check and precedes `opt.step`; its returned norm is the norm before clipping. [Loop:181](scripts/fourwire_scale_pgpre_stabilised.py:181)

The first actual update uses `peak/(5 × epoch_1_steps)`, rather than zero. Warm-up is linear in fractional epoch; if retained step counts vary, its slope per update varies.

Round robin constructs 157 candidate rounds and 2,826 candidate batches per full epoch. Filtering entirely non-converged batches leaves holes in rounds; those states receive no forward pass. Qualify “every state” and “one batch of every network” accordingly. [Loop:167](scripts/fourwire_scale_pgpre_stabilised.py:167)

**D. Default isolation holds; the broader protection does not.** Only train and val labels are requested. Full fits perform `check_freeze` initially, and ordinary failures create no frozen-study HALT, claim or lock. But the end check merely records failure and still exits successfully. [Fit:254](scripts/fourwire_scale_pgpre_stabilised.py:254)

The [guard:54](scripts/fourwire_scale_pgpre_stabilised.py:54) protects only the imported worktree’s final/dry directories:

- `collect --frozen-runs OTHER_WORKTREE/.../runs/final --out OTHER_WORKTREE/.../runs/final/HALT.json` is permitted.
- `equivalence --out WORK/frozen_routine/best.pt` passes the initial existence check, then replaces the checkpoint it just validated.

The attempts rule is also unenforced: rename a completed or numerical-failure folder to `.attempt1`, rerun, and collect accepts the replacement while merely listing the earlier folder. Infrastructure reasons and first-completed status are not checked. This falls short of [Amendment 10:779](docs/fourwire_scale_protocol.md:779).

**E. Collection checks are incomplete.** Recipe, dry status, freeze hash, state counts, scalers and artifact hashes are checked. Analysis-code identity, source/input/software provenance, numerical mode and training configuration are not. Records from different analysis implementations can therefore be mixed despite sharing the frozen source. [Collect:314](scripts/fourwire_scale_pgpre_stabilised.py:314)

A valid numerical failure before the first successful update produces empty norm arrays; collection then crashes at `quantile`/`max`, instead of reporting the no-checkpoint weak fit. [Collect:339](scripts/fourwire_scale_pgpre_stabilised.py:339)

From the claimed table, I independently obtain median **0.0212** and cutoff **0.106**; the underlying twelve losses remain unverified. Norm summaries cover all successful updates, including those after selection. The median ratio excludes no-checkpoint fits; report that denominator. Per-network losses retain batch-local near-limit normalisation, rather than being one globally weighted network mean.

**F. Slurm wiring is correct, conditional on fixing B.** Tasks 0–11 map to four folds × three seeds. The documented `afterok` dependency and Python exit-code propagation prevent fits following a genuinely failed equivalence command. Currently B can incorrectly make that command successful. [Slurm:7](scripts/hydra_fourwire_scale_stabilised.slurm:7)

**G. The exploratory framing is sound; several assurances exceed implementation.** The calibre-reader correction is accurate. The common threshold, paired comparisons, joint four-change interpretation and absence of significance testing are reasonable exploratory choices. Five warm-up epochs, round robin and no EMA need no performance-driven revision.

The CPU equality and “over 90% clipped” observations lack retained evidence here. The latter cannot establish clipping prevalence in full fits, nor the asserted explanation of AdamW update size. Restrict that statement to the observed dry run. [Protocol:791](docs/fourwire_scale_protocol.md:791)

**H. Not ready to submit.** Blockers, in order: validate equivalence completion/environment/provenance; close output-isolation and overwrite holes; enforce the declared attempt rule; bind collection to the analysis implementation and full provenance; handle empty-step terminal results. These fixes belong to the exploratory analysis and need not alter the frozen study.