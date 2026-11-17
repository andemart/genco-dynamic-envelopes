Audit round 13 (read-only), narrow. HEAD is 9aadb1d. Tracked files equal HEAD and stay stable while you read.

Since 740aca4 (round-12 files and note), 9aadb1d adds the output of the exploratory PG-pre diagnostic you asked for in round 9:
- results/fourwire_scale_pgpre_diagnostic.json (job 13595549, `checkpoints`, script of e34d242)
- results/fourwire_scale_pgpre_retrace_fold2_seed1.json (job 13595548, `retrace`)
- 49 lines appended to results/fourwire_scale_hydra_job_evidence.txt (accounting lines of the four diagnostic jobs and the logs of the two corrected ones)
- docs/fourwire_scale_protocol.md, last section: "Output of the exploratory PG-pre diagnostic (2026-11-17, before any evaluation record is opened)".

State: all final jobs ended (96 array tasks COMPLETED). No evaluation record, timing record or report of the final runs has been opened yet; that follows this commit. The equivalence job and the twelve stabilised fits of Amendment 10 were submitted from a worktree at cdb4e83 after your round 12 and the author's confirmation (jobs 13600689 and 13600690); no output yet.

Same rules: read-only; never read data_out/fourwire_scale/labels/test or /calib; no cluster. Be brief.

1. Provenance of the two JSON files: made by the script of e34d242 on the frozen source, libraries and GPU model, checkout unchanged, twelve fits expected and read, every recorded loss reproduced, retrace equal to the record, replayed order equal to the observed one? Anything in the files that makes them unusable?
2. Recompute every figure of the new protocol section from the two files: the table of fold 2 (per network, three seeds, and the range of the other twelve), the ranges of item 1, the error sizes of item 3 (and "at most 0.0011"), the term shares of item 4, the weight norms of item 5, and the epoch-8 paragraph (0.60 to 1.91, 68% against 51%, the median step losses of networks 8, 5 and 7, the median over all steps, the range 0.55 to 1.81 afterwards). Report each mismatch with the value you get.
3. Wording: any statement stronger than the data ("the checkpoint has learned little beyond the prior", "a shift of whole networks, not a few isolated steps", "the largest values are always on the networks where the correction is largest")? Anything that could be read as selecting or explaining more than a descriptive diagnostic can?
4. The account of the jobs (first version not read and not copied; times; order relative to the timing stage) against the evidence file. You cannot check the cluster; say what the committed evidence supports.

Verdict: is this section sound as the committed context for the E3 results, or what must be corrected.
