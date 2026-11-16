You are the independent auditor of a preregistered study in this repository (branch ander-fourwire-scale).
Work read-only: do not modify files, do not commit, do not submit jobs, do not contact any cluster.
Another agent produced the outputs below. Your job is to try to refute them, not to confirm them.
Derive every number yourself from the files before you compare it with a claim.

Read first:
- docs/fourwire_scale_protocol.md (frozen protocol, Amendments 1-5 and the Amendment 4 correction)
- results/fourwire_scale_*.json and results/fourwire_scale_labels_SHA256SUMS.txt
- src/gencodoe/fourwire_scale.py, fourwire_scale_data.py, fourwire_scale_train.py, fourwire_sweep_opt.py,
  and the ENWL1 trainer src/gencodoe/fourwire_train.py with src/gencodoe/fourwire_model.py
- tests/test_fourwire_scale_labels.py, tests/test_fourwire_scale_train.py
- Two gate-5 smoke reports, not committed:
  <scratchpad>/smoke.json            (Hydra job 13586491, started at commit b377e92)
  <scratchpad>/smoke_13586516.json   (Hydra job 13586516, pinned to commit a0b5e19)

A. Code audit (commits b377e92 and dde2bfc).
 1. fourwire_scale_data.LabelStore: can training code reach test labels by any path? Are states verified
    against the committed manifest before they are returned? Any gap?
 2. fourwire_scale_train.pretrain against fourwire_train.run_fit: the protocol (section 4, Amendment 5) says
    the ENWL1 configuration and loss are unchanged. List every deviation, bug, or place where val, calib or
    test information could enter training or the scalers.
 3. fourwire_scale_train.projection: does it count the fits of protocol section 8 correctly?
 4. smoke() records the git commit at the end of the run. Consequences and the minimal fix.

B. Claims made to the user. For each: supported, not supported or overstated, with your own numbers.
 1. The label-integrity gate passes on 96 of 96 network-splits; the corrected constant-power diagnostic is 0
    (first report 113); no other per-unit field differs between commits f0fbbdd and 642f431.
 2. CUDA sweep validation passes on 24 networks: worst voltage difference about 6.9e-16 pu, all 1,577
    reference-nonconverged states classified the same.
 3. Smoke: about 111 s per epoch, 2.0 GB peak GPU memory allocated, 15.4 GB host memory, about 34 GPU-hours
    as an upper bound for all section 8 fits. The two jobs agree on compute but differ in validation loss after
    epochs 1-3 (0.61/0.37/0.22 against 0.84/0.56/0.29) with the same seed, code, data and node. The other
    agent attributes this, unverified, to non-deterministic GPU operations. Is there another explanation in
    the code?
 4. Informational speed reading (explicitly not E2 timings): per state, the model's validation pass (batch 128,
    loss included; per_network.val_ms_per_state in the smoke report) costs about as much as the complete
    Numba sweep on the same node (numba.informational_ms_per_state in
    results/fourwire_scale_sweep_validation_cuda.json). The model/sweep ratio falls with network size mostly
    because the model's fixed overhead matters less; it is closest to 1 on networks where some states do not
    converge in the sweep, not on the largest networks as such. Marginal cost per bus is about 0.36 us for
    the model and about 0.34 us for the sweep (about 0.25 us on networks without non-converging states).
    Is this comparison fair? What biases it in either direction?
 5. Under the protocol, E2 end-to-end time includes the OpenDSS fallback for sweep non-convergence, so the
    GENCO pipeline could be faster end to end on stressed networks even if it is slower per solve.

C. Paper framing. The user wants a paper showing that GENCO was successfully adapted to four-wire LV networks
   and has better performance. Given protocol section 10 (claim discipline) and what has and has not been run,
   which claims are permissible, which are not, and what must not be done to protect the preregistration?

D. Anything else that would not survive peer review: protocol violations, undisclosed post-hoc choices,
   test-set exposure, provenance gaps.

Output: findings numbered by severity, each with file:line or JSON-key evidence; then a verdict per claim in B;
then C in at most ten lines. Be concise and concrete.
