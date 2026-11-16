1. **Resolved.** The outer lock covers counting and claiming, closing the round-11 race and preventing the helper’s HALT write. A completed, validated fit with two earlier attempts reaches `AlreadyDone` and exit 10. `finally` releases the lock on ordinary exceptions and skips; in-memory probes confirmed this. Abrupt process death can leave stale locks, which block further claims until verified and removed as a recorded decision. [fit:345](scripts/fourwire_scale_pgpre_stabilised.py:345)

2. **Resolved within the disclosed limit.** Checking both absolute and resolved spellings closes my symlink example. An unnamed alias to another checkout’s run directory remains unrecognisable, as correction 2 accurately states. [guard:70](scripts/fourwire_scale_pgpre_stabilised.py:70)

3. **Resolved.** Collection enforces fold 2, seed 1, 648 train, 312 val and four epochs. These exactly match the Slurm command. [collect:458](scripts/fourwire_scale_pgpre_stabilised.py:458), [Slurm:45](scripts/hydra_fourwire_scale_stabilised.slurm:45)

4. **Corrections accurate, with one minor wording qualification.** Item 1’s refusal applies when the current fit folder exists. If it is absent, two earlier folders can still precede a newly claimed third attempt; this respects the attempt limit. Items 2–6 match the code and retained evidence. The laptop hash matches the script at `73f37af`, and `train()` is unchanged. [protocol:817](docs/fourwire_scale_protocol.md:817)

**Ready to submit the equivalence job and the twelve fits through the documented `afterok` dependency.**