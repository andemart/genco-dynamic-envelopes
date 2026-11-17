# Audits of the four-wire scale study

The study of `docs/fourwire_scale_protocol.md` is audited by a second model (Codex, run headless and read-only in this repository). Each round has two files: the brief it was given (`NNN_brief.md`) and its report, unedited (`NNN_review.md`). The auditor never reads the calibration or test labels, never modifies a file and never contacts the cluster.

The files are copied as written, with three mechanical replacements: absolute paths of the local checkout are made relative to the repository root, the path of a local scratch folder reads `<scratchpad>`, and cluster account identifiers are replaced by `N`.

| Round | Finished | Audited state | Outcome |
|---|---|---|---|
| 001 | 2026-11-15 19:51 | `b377e92`, `dde2bfc` and two smoke reports | Seven findings on provenance, loader and integrity checks |
| 002 | 2026-11-15 20:21 | `cbb0a02` | Findings partly resolved; more required |
| 003 | 2026-11-15 22:28 | `ee2479f` and the draft of Amendments 7 to 9 | Not ready to freeze |
| 004 | 2026-11-15 23:23 | `73442a2` | Not ready to freeze |
| 005 | 2026-11-16 00:38 | `ed17a1a` | Not ready to freeze |
| 006 | 2026-11-16 01:05 | `6013516` | One blocker left on run claims |
| 007 | 2026-11-16 01:13 | `e85164c` | Ready to freeze once an A100 dry run of this source has ended and been read |
| 008 | 2026-11-16 09:26 | `a4068ef` (the freeze) | Ready to submit, no blockers |
| 009 | 2026-11-16 14:35 | `a8a2612` (weak-seed note and diagnostic script) | One blocker in the script, corrected in `e34d242` |
| 010 | 2026-11-16 20:48 | `bd1cc9c` (Amendment 10: stabilised-recipe analysis) | Not ready to submit: five blockers in the checks around the analysis, corrected in `73f37af` |
| 011 | 2026-11-16 21:09 | `745ce0b` (the corrections of round 10) | Gate and collection resolved; two blockers left (attempt limit outside the lock, links in the path guard), corrected in `bb89f92` |
| 012 | 2026-11-16 21:17 | `cdb4e83` (the corrections of round 11) | Ready to submit the equivalence job and the twelve fits; one wording note |
| 013 | 2026-11-17 08:03 | `9aadb1d` (output of the PG-pre diagnostic) | Every figure reproduced; wording to qualify and job evidence to add, done in the commit that follows |
| 014 | 2026-11-17 08:36 | `11bcf1f` (final outputs and the outcome section) | E3 and E2 numbers recomputed and confirmed; reporting corrections required, first of all two unsafe fallback decisions on network 18, done in the commit that follows |
| 015 | 2026-11-17 08:42 | `4d71551` (the corrections of round 14) | Resolved and the new figures confirmed; three wording corrections, done in the commit that follows. No record, report or criterion changed |
| 016 | 2026-11-17 12:44 | `9d2af25` (Amendment 11: E4, balanced-model envelopes on the 24 networks) | Not ready to submit: the models and checks are what the amendment states; blockers in the isolation of dry runs, the publishing of records, the report's record checks and open edge rules, corrected in `8bb4061` |
| 017 | 2026-11-17 13:18 | `4d74a1f` (the corrections of round 16; the output of the stabilised-recipe analysis) | E4 not yet ready: two report checks and three statements left, corrected in `78e9d1d`. Stabilised output: every figure reproduced, one sentence corrected in the same commit |
| 018 | 2026-11-17 13:24 | `0f5cb80` (the corrections of round 17) | All five points resolved, no new blocker; ready to submit the 24 E4 runs and the report from a worktree of `0f5cb80`. The corrected sentence of the stabilised output is within the records |
| 019 | 2026-11-17 15:52 | `f377217` (the output of E4: 24 records, ledger, report, job evidence, outcome section) | Criterion met and records usable: the report, all 960 table cells and the E2 comparison recomputed with independent code; frozen files and scripts unchanged. Two corrections (an inference on network 18 that was too strong, one ratio) and five clarifications, made in `a6e6393`. No rerun |
| 020 | 2026-11-17 15:54 | `bb98792` (the corrections of round 19) | All seven points resolved, no new error; the 49 E4 files hashed against the cluster-side listing. E4's outcome with its corrections is accurate and within Amendment 11: criterion met, 21 of 23. No remaining findings |
| 021 | 2026-11-17 17:11 | `8e46433` (Amendment 12: E5, the positive-sequence model and two further technology mixes; the specification table for E4) | Not ready to submit E5: the reduction, mixes, fresh check, records and job script are what the amendment states; two gaps in the code (the positive-sequence check, the comparison with the E4 records) and one in the disclosure, corrected in `3bac869`. The specification table can be computed as written |
| 022 | 2026-11-17 17:27 | `61b38f3` (the corrections of round 21) | The positive-sequence check and the disclosure resolved; the comparison with the E4 records still compared results only at exactly equal levels, corrected in `e5392c4`. Not yet ready to submit E5; no other blocker |
| 023 | 2026-11-17 17:37 | `07738d3` (the correction of round 22; the output of the specification table for E4) | The gap of round 22 resolved, no other executable change: ready to submit the 72 E5 runs and the report from a worktree of `07738d3`. Every number of the specification table reproduced; one sentence of its reading corrected in the commit that adds this row |
| 024 | 2026-11-17 23:38 | `6764d7b` (the 72 E5 records, the report, the job evidence and the protocol section "Outcome of E5") | No correction required. The report recomputed from the 72 records with independent code (all six cells, 21 of 23 in each; the comparison with the E4 records); every figure and table cell of the outcome section reproduced; the claims found within Amendment 12. Noted: on network 18 under the half mix the four-wire-verified envelopes reaching 0.95 on the three OpenDSS days drop from 123 to 106 of 128 when OpenDSS judges, which the section states |

How each finding was answered is recorded in the protocol's amendments and notes and in the commits that follow each round.
