**Mostly resolved; three wording corrections remain.**

- **Network 18 disclosure: partly resolved.** The two unsafe fallback decisions are correctly disclosed. However, the transformer-flag range on the other 23 networks is **0–13**, not 1–13: networks **10 and 25 have zero differences**. [Protocol:962](docs/fourwire_scale_protocol.md:962)
- **E2 comparator, known corrections and envelope denominators: resolved.**
- **The two figures, recovery qualification and mean-matching qualification: resolved.**
- **Reporting details: partly resolved.** The interval-precision statement needs correction below.
- **Input hashes and provenance limits: resolved.** All **39 unique files** are present, cover the intended outputs exactly, and match their manifest hashes.

The newly checked figures reproduce:

| Quantity | Recomputed value |
|---|---:|
| Unsafe exact decisions | **2/8,192** on network 18; **2/196,608** overall |
| Envelope counts excluding immediate baseline rejection | **360 / 212 / 90 / 93** |
| Network 11 scratch fallback percentages | **1.47705 / 1.36719 / 1.42822** |
| PG-zs false-safe rate / upper bound | **0.047619% / 0.225697%** |
| PG-net smallest upper bound | **0.104290%**, 2,871 violating pairs |
| PG-net largest upper bound | **23.840419%**, 11 violating pairs |
| Networks with bounds above 5% | **10 and 25 only** |
| PG-net aggregate false-safe count | **0/86,588 violating pairs** |
| Envelope fallback range, import | **0.010992–24.124284%** |
| Envelope fallback range, export | **0–9.219473%** |

The rounded figures in items 4, 5, 8 and 11 are correct.

Two further qualifications:

1. **“Complete pipelines on 23 networks” needs an explicit scope:** say **“their complete pipelines”**, referring to the target-calibrated screens. Do not silently include PG-zs. Its network-7 screen error is known; whether union fallback corrects that error cannot be determined from the committed aggregate records.

2. **Item 11’s “third decimal” is false across every speed-up.** For U1 on network 7, the upper endpoint changes from **397.025730 to 396.977162**, a difference of **0.048569**. For PG-net, the largest endpoint change is **0.002484**. These changes do not alter the criteria. [Protocol:972](docs/fourwire_scale_protocol.md:972)

**Verdict:** sound after correcting **0–13**, explicitly limiting the complete-pipeline claim to target-calibrated methods, and removing the universal third-decimal assertion. No record, report or criterion needs changing.