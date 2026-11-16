1. **Resolved.** Owner liveness now precedes validation, archival and the rerun-limit check. In-memory probes rejected both a live invalid-record owner and a live third attempt without writing HALT. [fourwire_scale_train.py:827](src/gencodoe/fourwire_scale_train.py:827).

2/3. **Resolved.** The sole `AlreadyDone` path validates and publishes under the lock. All three callers supply the required validators; evaluation and E2 supply publication targets. No remaining exit-10 bypass or publication-recovery gap found. [train:829](src/gencodoe/fourwire_scale_train.py:829), [eval:403](src/gencodoe/fourwire_scale_eval.py:403), [E2:286](src/gencodoe/fourwire_scale_e2.py:286).

9. **Resolved.** The third note matches the ordering, validation/publication transaction and declared-deviation exception for own or unattributed cancellations. [protocol:702](docs/fourwire_scale_protocol.md:702), [train:788](src/gencodoe/fourwire_scale_train.py:788).

10. **No new blocker found** in the helper or its three callers. The lock is released on rejection, validation failure and successful skip. [train:868](src/gencodoe/fourwire_scale_train.py:868).

**Ready to freeze once an A100 dry run of this source ends with `DRY_RUN_OK` and its output has been read.**