Second audit round, same rules: read-only, try to refute, derive numbers yourself.

Since your first report, commits 42f61bc, 0756673, b23b286, a8efca1 and cbb0a02 were added on branch
ander-fourwire-scale (HEAD = cbb0a02). They claim to resolve your findings 1-7.

A. For each of your seven findings: resolved, partly resolved or not resolved, with file:line evidence.
   Look in particular at:
   - src/gencodoe/fourwire_scale_data.py: can a read still be redirected (symbolic links, "..", hard links,
     the label root itself, a manifest that names another network)? Is anything still overridable?
   - src/gencodoe/fourwire_scale_train.py: start_provenance, checkout_unchanged, smoke, determinism, the
     refusal to overwrite a run folder. Anything that can still mislabel or overwrite a run?
   - src/gencodoe/fourwire_scale.py: integrity_split and label_integrity (provenance from shard files, one
     commit across units), and the Numba informational timer in validate_sweeps. Note that label_integrity
     still reads its own commit at the end of the run.
   - tests/test_fourwire_scale_train.py and tests/test_fourwire_scale_labels.py: do the new tests prove what
     their names say?
B. docs/fourwire_scale_protocol.md, Amendment 6: check every factual statement against the repository and the
   two smoke reports in results/. Flag anything wrong, overstated or missing. Is the disclosure about target
   labels in the Thevenin prior sufficient for your finding 2?
C. scripts/hydra_fourwire_scale_determinism.slurm and the determinism command: will this probe answer whether
   training is reproducible on the A100, with and without torch.use_deterministic_algorithms? What would make
   its answer misleading? (On CPU, two processes were bit-identical in both modes.)
D. Before any final fit, the following still has to be written and frozen: PG-net, PG-scratch and PG-ft
   training, screen calibration, test evaluation, and the E2 timing harness. List the design decisions that the
   protocol (sections 4-8 and Amendments 5-6) leaves open and that must be fixed in an amendment before fitting,
   in order of risk.

Output: verdict per finding (A), numbered issues for B and C, and a short ordered list for D. Be concise.
