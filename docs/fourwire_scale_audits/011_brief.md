Audit round 11 (read-only), narrow. HEAD is the commit after 73f37af (it only adds your round-10 brief and report under docs/fourwire_scale_audits/). Tracked files equal HEAD and stay stable while you read.

73f37af answers your round-10 blockers. It changes scripts/fourwire_scale_pgpre_stabilised.py, scripts/hydra_fourwire_scale_stabilised.slurm, appends "Corrections to Amendment 10" to docs/fourwire_scale_protocol.md, and adds two laptop records under results/ (fourwire_scale_pgpre_stabilised_equivalence_laptop.json, fourwire_scale_pgpre_stabilised_dryrun_laptop.json). src/ is untouched. Nothing of Amendment 10 has been submitted; no evaluation record, timing record or report has been opened; the diagnostic jobs are still queued.

Same rules: read-only; never read data_out/fourwire_scale/labels/test or /calib; no fixture test suite; no cluster. Do not re-audit what you cleared in round 10 (source identity, the frozen-mode computation, the schedule and clipping). Small in-memory probes are fine. Be concise.

For each round-10 blocker say resolved, partly resolved or not resolved, with file:line evidence:

1. Equivalence gate: does it now fail unless every item is equal, both fits completed every epoch, there was no numerical failure, and nothing changed during the run? Without --dry, does it run only in the frozen environment (tt.check_freeze, the script equal to HEAD)? Can the job script still let the fits start after a gate that did not pass?
2. Output isolation and overwriting: can any command still write into a runs/final or runs/dry folder of any checkout, into --frozen-runs, or replace an existing file (your two examples included)? Full-size fits: can they go anywhere but this checkout's runs/stabilised?
3. Attempts: the fit now claims its folder with fourwire_scale_train.claim_run_folder(final=True for full size). Is the declared rule enforced (a valid record is never run again; a folder without a record is set aside only for an infrastructure end state; at most two earlier attempts; an invalid fit fails)? Can claim_run_folder with final=True write a HALT file or anything else under runs/final here, given the pre-check on the number of earlier attempts? Does collect refuse hand-renamed attempt folders?
4. collect: is it bound to one commit and one script for the twelve fits and the equivalence check, and to the full provenance and training configuration? Does it handle a fit that ended before its first step?
5. The corrections in the protocol (items 1 to 9): any statement stronger than the code or the two retained laptop records? Does the SHA-256 in those two records equal the committed script?

Anything new that these changes broke: in particular, is train() still the frozen computation when the switches are off (the only change there is the folder precondition), and is the record that fit writes sufficient for check_record?

Verdict: ready to submit the equivalence job and the twelve fits, or the remaining blockers in order.
