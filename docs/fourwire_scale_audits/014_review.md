The numerical conclusions reproduce. The most important omission is that the exact fallback makes two unsafe decisions on network 18, despite every target-calibrated **screen** having zero false-safe decisions.

**A. Records.** All **276 fits, 24 evaluations and 12 E2 records** pass the recorded provenance checks: correct freeze, source, libraries, A100, pinned commit, unchanged checkout, no earlier attempts and no numerical failures. All canonical fits exist; checkpoint links and PG-ft starting weights/scalers match. The 96 job-array tasks match the accounting evidence.

E3’s 24 evaluation hashes match. Timing and envelope records also correctly hash their evaluation inputs. **E2’s report lacks hashes identifying its timing/envelope input records**; add an input-hash manifest alongside it.

Checkpoints and per-state/decision NPZ files are unavailable locally. Consequently, I could recompute the JSON arithmetic, timings and envelopes, but could not independently replay state-level accuracy or delivered-decision counts.

**B. E3.** Recomputed using the prescribed network bootstrap:

| Comparison | Difference, percentage points | 95% interval |
|---|---:|---:|
| Primary, B=1,000 | **0.321452** | **−1.205170 to 1.921095** |
| Including network 1 | 0.881788 | −0.883009 to 2.876773 |
| Secondary B=200, seed 0 | 5.363133 | 2.734813 to 8.566523 |
| Secondary B=5,000, seed 0 | 0.520656 | 0.162393 to 0.950079 |

False-safe and cleared-indeterminate sums are zero for both compared methods in every comparison. **“Criterion not met” is correct:** the primary estimate is below two points and its interval includes zero.

The seed/fold table, method table, listed network effects and **1.929429-point** post-hoc mean over seventeen networks all reproduce.

**C. E2.** Independently recomputed ratios and bootstrap intervals agree with the report:

| Network | Speed-up [95% interval] | Without OpenDSS solves |
|---|---:|---:|
| 25 | 0.08172 [0.08169, 0.08176] | 0.08167 |
| 10 | 0.09229 [0.09225, 0.09232] | 0.09226 |
| 1 | 5.23776 [5.22772, 5.25651] | 0.21021 |
| 7 | 23.11758 [23.05367, 23.15513] | 0.28838 |
| 2 | 11.42804 [11.41330, 11.43973] | 0.22323 |
| 17 | 14.17040 [14.15297, 14.20798] | 0.15056 |

The reported safety counts make the criterion hold on **1, 7, 2 and 17**, with sampled break-even **397 buses**. Without OpenDSS solves it holds nowhere. U1 passes the same rule on all six, in both modes. Paired-round intervals, latency summaries, throughput and exponents reproduce; exponent differences are only floating-point rounding.

All envelope summaries reproduce: **570/576** within tolerance on network 10, **576/576** elsewhere. Its six unsafe intervals are all **import**, with maximum excess **0.02368 kW per customer**. Zero-envelope counts are correct.

**D–F. Corrections and qualifications, in importance order:**

1. **Disclose network 18’s unsafe exact fallbacks.** Its transformer flags differ on two states, producing **two unsafe state/direction decisions**. Every method defers both affected pairs to that engine. This does not change the screen-based E3 criterion, but rules out extending the zero-error claim to complete pipelines across all 24 networks. Evidence: [network_18.json](results/fourwire_scale_eval/network_18.json), `exact_sweep_vs_opendss` and every method’s `fallback_exact_engine`.

2. **Correct the E2 comparator description.** At [protocol:925](docs/fourwire_scale_protocol.md:925), “sweep on the GPU” is wrong for the speed-up table. The selected baseline is **CPU for 25, 10 and 1; GPU for 7, 2 and 17**. It is the preregistered faster-of-two comparison.

3. **Apply the two known corrections**, including the denominator wording. Counts **211 and 92** are nonzero envelopes on networks 7 and 17. If excluding only immediate infeasible-baseline returns, the remaining counts are **212 and 93**. “Agreement says nothing about the screen” is also too absolute; say it provides limited evidence about positive-envelope accuracy.

4. **One additional numerical correction:** network 7’s batch-256 speed-up is **15.647865×**, rounding to **15.6×**, not 15.7×. Network 11’s scratch value “1.4%” is the three-seed mean; seed 0 is **1.47705%**, rounding to **1.5%**.

5. **Qualify recovery and matching claims.** All twelve affected PG-ft fits have more fallbacks than their matched scratch fits; that supports “did not recover to scratch performance”, rather than an unrestricted statement about recovery. Likewise, B=5,000 “matches PG-net” concerns the **seed-0 network mean**, not every network or established equivalence. The post-hoc seed/fold and concentration descriptions are otherwise clearly fenced.

6. **Surface the required reporting details.** The records contain them, but the section should explicitly reference:
   - false-safe rates and one-sided bounds: PG-zs’s error is **1/2,100 export violations**, upper bound **0.2257%**; zero errors can still have wide bounds—**23.84%** on network 10;
   - rating sensitivities, other accuracy metrics, paired-round intervals and envelope fallback shares per direction;
   - budgets as **model-fitting states**, alongside fixed validation/calibration and the **500-state target Thevenin identification**, including for zero-shot models;
   - fallback shares as direction-averaged; actual timed solves use the union across directions.

**Verdict:** the primary numerical readings are sound. Correct the outcome’s reporting—especially network 18’s fallback errors, comparator identity and provenance limitations—while retaining the frozen criteria and all runs.