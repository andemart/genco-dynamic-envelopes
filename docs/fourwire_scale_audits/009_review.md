**The main blocker is output protection: the script can overwrite a frozen run.** The frozen loss formula and replay order otherwise look faithful.

1. **Source identity — supported.** The diff contains only the protocol note and new script. Independently recomputed source SHA-256 remains `661c1d265b870c57a51df58312930b78d5b0c81c259d24434dd30ceac1c4be87`, matching the freeze.

2. **File access — label restriction supported; write protection false.** The intended call graph loads only `train` and `val`. Provenance reads metadata; `_case` can materialise network files in the imported worktree’s `data_out`, without reading held-out labels.

   However, only `--work` is checked. Either mode accepts, for example, `--out "$RUNS/pgpre_fold2_seed1/record.json"` and `_write_json` replaces that frozen record. Output can also collide with replay artifacts. Protect all resolved output destinations and reject existing files before starting. [script:133](scripts/fourwire_scale_pgpre_diagnostic.py:133), [script:181](scripts/fourwire_scale_pgpre_diagnostic.py:181), [script:205](scripts/fourwire_scale_pgpre_diagnostic.py:205), [writer:449](src/gencodoe/fourwire_scale.py:449).

3. **Validation-loss fidelity — supported, conditional on matching inputs/configuration/software.** Forward pass, scalers, four weighted terms, 128-state batches and converged-state pooling match `fit.evaluate`. The reported terms sum to the loss **up to FP32 rounding**. [script:80](scripts/fourwire_scale_pgpre_diagnostic.py:80), [trainer:325](src/gencodoe/fourwire_scale_train.py:325).

   Per-network loss is a converged-count-weighted mean of **separately normalised batch losses**, not one globally near-weighted mean. A synthetic two-batch example gives **8.75 versus 8.3333**, respectively. Correction RMS is separately unweighted over complex conductor voltages.

   `read_record` does not verify the original freeze/source/software/input identity, and there is no end-of-run provenance check. Those comparisons should precede interpreting a fidelity failure. [script:34](scripts/fourwire_scale_pgpre_diagnostic.py:34).

4. **Retrace trajectory/order — supported for completed epochs.** It retains the original requested epoch maximum, hence the same cosine schedule despite early stopping. `prepared=False` changes directory creation only; `record_steps=True` adds another scalar read after the existing synchronisation, without changing optimisation or RNG draws.

   Source-extracted in-memory checks matched all 15 completed epochs, including skipped batches. But the replay checks validate **counts only**: a wrong replay seed preserved every count while changing network order. Curve/weight comparisons cannot independently validate labels assigned afterward. Partial failed epochs would trigger the length check rather than receive a complete mapping. [script:144](scripts/fourwire_scale_pgpre_diagnostic.py:144), [script:151](scripts/fourwire_scale_pgpre_diagnostic.py:151).

5. **Protocol note — mostly sound, with qualifications.**
   - PG-zs/PG-pre-tc use seed 0. The affected count is correctly **6 networks × 2 seeds = 12/120 PG-ft fits**. For the primary fresh-network comparison, these are **12/69 fine-tuned fits across 6/23 networks**.
   - Separate, disclosed train/val exploratory retraining can coexist with the frozen results. “No run … run again” should explicitly mean **no frozen study run is retried or replaced**.
   - Seed 2 was still running; describing its completed record or selected checkpoint as weak is premature. I could not independently verify the live-loss snapshot from the supplied artifacts.
   - `checkpoints` silently processes whichever records exist—even zero—and its glob also matches archived attempt folders. It does not ensure twelve completed canonical fits. Require completeness before interpreting a twelve-fit comparison. [script:128](scripts/fourwire_scale_pgpre_diagnostic.py:128).

**Verdict:** fix output protection before running the diagnostic. Before interpreting results, verify record provenance and completeness. Replay-check strength and loss/RMS interpretation are remarks; these findings provide no reason to replace the frozen fits.