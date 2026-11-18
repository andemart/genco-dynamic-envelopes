1. **The quoted replication definition omits the OpenDSS counting rule.** Below 50% correctly means *classified as balanced unsafe*. Counting toward the 12-of-23 criterion additionally excludes networks where the engines classify the OpenDSS days differently, subject to the documented empty-day exception. State that rule and use “classified” for the dotted line. All networks agree here, so **21/23 remains correct**. [Protocol:1307](docs/fourwire_scale_protocol.md:1307), [implementation:389](scripts/fourwire_scale_balanced_envelopes.py:389).

2. **Name the judging engine in the caption:** add “judged by the four-wire sweep.” The paragraph elsewhere provides the qualification, but the caption is incomplete when read alone. The distinction matters: on network 18’s half-mix OpenDSS days, the reference reaches the target on **123/128 envelopes by sweep versus 106/128 by OpenDSS**. The figure shows sweep-based shares over all 36 days.

3. **The 0.12-point bound is correct; “overlap” is more precise than “coincide.”** The largest difference is **0.11574074 percentage points**, network 18, half mix: **6.597222% versus 6.481481%**. The other nonzero difference is **0.05787037 points**, network 18, pv mix. These differences are visually unresolved by the overlapping markers. Scope the statement explicitly to the *plotted shares*, rather than the models generally. [Drawing order:50](scripts/paper_figure_replication.py:50).

4. **No numerical, selection or clipping defect found.** Independently reconstructed all **216 shares** from report counts; all match the stored shares and protocol tables.

   - The series use the correct cells and `summary["both"]` keys; the reference is identical between the two cells of each mix.
   - All 24 networks appear in customer-count order. Only network 1—the ninth column—is shaded. Tick labels are correctly **42, 171, 238, 330, 471, 883**.
   - Whiskers appear on **25, 10 and 18 in every mix**. All nine bounds match the protocol. For network 18 they are **2.3–5.3%, 4.7–8.5%, 1.4–4.2%**.
   - “Bootstrap interval over days” accurately describes the percentile day-bootstrap interval. Showing only positive-sequence intervals is adequately disclosed.
   - Shares span **0–99.652778%**; interval endpoints span **0–97.569444%**. Nothing is clipped by −5 to 105.
   - Both models count the same 21 fresh networks in every mix. “Either balanced model” is correct.

5. **The PNG matches the numbers.** Networks 25 and 10 are the two design-model columns above one half; network 18 is at **3.70%, 6.60%, 2.66%** for positive sequence; the other 20 fresh networks are at zero. The reference bands span **92.65–99.48%, 92.26–99.54%, 91.42–99.65%**.

The benchmark legend explicitly says “not counted”; the adjacent statement specifies the same 21 networks, so neither suggests additional independent network replications. The axis label, read with the caption, correctly denotes the share of successful envelopes—not mean state feasibility.

**Verdict: correction required to the caption and quoted counting definition. No change to the plotted data or script is required.**