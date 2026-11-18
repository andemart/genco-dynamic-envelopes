Audit round 25 (read-only). HEAD is ae83979. Tracked files equal HEAD and stay stable while you read, except scripts/paper_figures.py, which is modified in the working tree by unrelated work (two label strings in fig_surrogates) and is out of scope, as are docs/hf_post_*.

Since 02ca173 (round 24, the outcome of E5 confirmed):
- ae83979 adds scripts/paper_figure_replication.py, a plotting script for a figure of the E5 result. It reads results/fourwire_scale_e5.json only and imports its style constants and the helpers load, save and panel from scripts/paper_figures.py.
- Its output, not tracked: paper/figures/replication.pdf and paper/figures/replication.png.
- The figure is included in the paper draft with the caption quoted under C, and the sentence "The criterion is met in all six combinations of model and mix, each time on the same 21 of 23 networks" now refers to it. Nothing else in the paper was changed for this round. The paper source stays out of scope as before: do not open paper/main.tex or any other file under paper/ except the two figure files named above; everything you need from the paper is quoted in this brief.

Rules as before: read-only; never read data_out/fourwire_scale/labels/test or /calib; do not run the fixture test suite; no cluster access; do not compute any envelope, scenario or sweep state on a held-out day. Do not run scripts/paper_figure_replication.py as a program (it writes two files): read it, and reproduce what it plots with your own code in memory.

Please try to refute the following. Give artefacts and numbers, not agreement.

A. The script. Does it plot stored values of the report without recomputation or selection?
 1. Keys: in a cell "<mix>/<model>" the design model sits under summary[scope]["balanced"] and the four-wire-verified envelopes under summary[scope]["fourwire"]. Is the positive-sequence series taken from the cells "<mix>/posseq" and the balanced-load series from "<mix>/balanced", scope "both", and the four-wire series the same in both cells of a mix?
 2. Order: networks by number of customers, then by number. Is the shaded column network 1 and no other, and are the tick labels the customer numbers of the columns they sit under?
 3. Whiskers: by_scope["both"]["balanced_share_ci95_days"]["ci95"] of the positive-sequence cell, drawn only where the upper bound exceeds the lower. Which networks get a whisker in each mix, and do the bounds equal those of the protocol section "Outcome of E5"?
 4. The text in each panel, "fewer than half on N of M fresh networks (either balanced model)", takes N and M from the report's count and of. Is that what the criterion of Amendment 11 item 4 / Amendment 12 counts, and is "either balanced model" right for all three mixes?
 5. Anything the script hides: a value clipped by the axis limits (-5 to 105), a network left out, a marker drawn over another so that a difference cannot be seen where one exists.

B. The plotted values. Recompute the 24 x 3 x 3 shares (four-wire-verified, positive-sequence, balanced-load; three mixes) from the report and compare them with the per-network tables of the protocol section "Outcome of E5". If you can open paper/figures/replication.png, check the picture against these numbers (the two networks above one half, network 18, the band of the four-wire-verified shares); if you cannot open images, say so.

C. The caption, statement by statement. In full it reads: "Balanced-model envelopes on the 24 networks, by technology mix: share of a network's positive envelopes that reach 0.95 four-wire feasibility on fresh draws (36 test days, both directions). One column per network, ordered by number of customers. The two balanced models differ by at most 0.12 percentage points on any network, so their markers coincide. Whiskers: 95% bootstrap interval over days for the positive-sequence model. The dotted line marks one half, below which a network counts as balanced unsafe." The text of the paper defines the term just before: "a network counts as balanced unsafe if fewer than half of its positive balanced-model envelopes reach 0.95 four-wire feasibility on 200 fresh draws, and the result replicates if at least 12 of the 23 fresh networks do."
 1. "share of a network's positive envelopes that reach 0.95 four-wire feasibility on fresh draws (36 test days, both directions)".
 2. "One column per network, ordered by number of customers."
 3. "The two balanced models differ by at most 0.12 percentage points on any network, so their markers coincide." Give the largest difference you find and where.
 4. "Whiskers: 95% bootstrap interval over days for the positive-sequence model." Is this the interval computed by day_bootstrap in scripts/fourwire_scale_balanced_envelopes.py, and is "bootstrap interval over days" a fair name for it?
 5. "The dotted line marks one half, below which a network counts as balanced unsafe." Is that the definition, given that a network is counted only if the two engines classify it alike on the OpenDSS days?

D. Claims. Does the figure, with its caption, show or suggest anything that the outcome section of E5 does not support or that Amendment 12 item 9 does not allow? Consider at least:
 - The benchmark network is drawn with the same markers as the fresh networks and marked only by shading and a legend entry. Could a reader take it as counted?
 - Feasibility on the 36 days is judged by the sweep; the caption does not say so (a paragraph of the paper on the limits of the result does: "Feasibility over the 36 days is judged by the sweep. On the three days also solved by OpenDSS the two engines classify every network alike in every combination"). Must the caption say it?
 - The three panels show the same 21 networks below one half. Does the figure invite reading them as three independent replications?
 - Whiskers are drawn for one of the two models only.
 - The y axis is the share of envelopes reaching 0.95, not the feasibility itself. Is the axis label clear enough?

E. Anything else you find wrong in the script, the figure or the caption.

Answer with findings ordered by severity, each with the evidence, and end with a verdict: correction required or not.
