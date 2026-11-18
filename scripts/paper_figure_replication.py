"""Figure for the paper: balanced-model envelopes on the 24 networks (paper/figures/replication.pdf).

    .venv/bin/python scripts/paper_figure_replication.py

Input: results/fourwire_scale_e5.json only (the report of family E5, Amendment 12). Nothing is recomputed:
every plotted value is a share or an interval stored in the report. Style and palette are those of
scripts/paper_figures.py.

One panel per technology mix. One column per network, ordered by number of customers. Height: the share of
a network's positive envelopes that reach 0.95 four-wire feasibility on 200 fresh draws (36 test days, both
directions), for the four-wire-verified envelopes and for the envelopes of the two balanced design models.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_figures import CORAL, DCOL, GRID, INK2, MUTED, TEAL, VIOLET, load, panel, save

MIXES = (("base", "Base mix"), ("half", "Every share halved"), ("pv", "PV share doubled"))
TICK_EVERY = 5                                     # customer numbers under every fifth network and the last


def fig_replication():
    cells = load("results/fourwire_scale_e5.json")["cells"]

    def share(cell, n, arm):                       # in a cell the design model sits under the key 'balanced'
        return 100 * cells[cell]["per_network"][n]["summary"]["both"][arm]["share_reaching_target"]

    ref = cells["base/posseq"]["per_network"]
    order = sorted(ref, key=lambda n: (ref[n]["customers"], int(n)))
    x = list(range(len(order)))
    bench = [i for i, n in enumerate(order) if not ref[n]["fresh"]]
    assert [order[i] for i in bench] == ["1"]
    ticks = x[::TICK_EVERY] + [x[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (mix, title), letter in zip(axes, MIXES, "abc"):
        ps, bl = cells[f"{mix}/posseq"], cells[f"{mix}/balanced"]
        assert ps["criterion_met"] and bl["criterion_met"] and (ps["count"], ps["of"]) == (bl["count"], bl["of"])
        fw = [share(f"{mix}/posseq", n, "fourwire") for n in order]
        assert fw == [share(f"{mix}/balanced", n, "fourwire") for n in order]
        yp = [share(f"{mix}/posseq", n, "balanced") for n in order]
        yb = [share(f"{mix}/balanced", n, "balanced") for n in order]
        ax.axvspan(bench[0] - 0.5, bench[0] + 0.5, color=GRID, lw=0, zorder=0)
        ax.axhline(50, color=INK2, lw=0.9, ls=":", zorder=1)
        for i, n in enumerate(order):              # interval over days of the positive-sequence share
            ci = ps["per_network"][n]["by_scope"]["both"]["balanced_share_ci95_days"]["ci95"]
            if ci and ci[1] > ci[0]:
                ax.plot([i, i], [100 * ci[0], 100 * ci[1]], color=CORAL, lw=0.9, solid_capstyle="butt", zorder=2)
        ax.scatter(x, fw, s=15, color=TEAL, edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter(x, yb, s=30, facecolor="none", edgecolor=VIOLET, linewidth=0.9, zorder=3)
        ax.scatter(x, yp, s=8, color=CORAL, linewidth=0, zorder=5)
        ax.text(x[-1] + 0.4, 66, f"fewer than half on\n{ps['count']} of {ps['of']} fresh networks\n"
                "(either balanced model)", ha="right", va="center", fontsize=6.5, color=INK2)
        ax.set_xlim(-0.9, x[-1] + 0.9)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(ticks, [str(ref[order[i]]["customers"]) for i in ticks])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        panel(ax, letter)
        gap = max(abs(a - b) for a, b in zip(yp, yb))
        low = [i for i in x if yp[i] < 50 and i not in bench]
        print(f"replication {mix}: fewer than half on {ps['count']} of {ps['of']}; four-wire-verified "
              f"{min(fw):.1f} to {max(fw):.1f}%; largest gap between the two balanced models {gap:.2f} points; "
              f"of the {len(low)} fresh networks below one half, {sum(yp[i] == 0 and yb[i] == 0 for i in low)} at "
              f"zero for both models; not at zero: "
              + ", ".join(f"network {order[i]} ({ref[order[i]]['customers']} customers) {yp[i]:.1f} / {yb[i]:.1f}"
                          for i in x if yp[i] > 0 or yb[i] > 0))

    a = axes[0]
    a.set_ylabel("Envelopes reaching 0.95\nfour-wire feasibility (%)")
    axes[1].set_xlabel("Network, ordered by number of customers (tick labels: customers)")
    a.text(x[-1] + 0.4, 46, "one half", ha="right", va="top", fontsize=6.5, color=INK2)
    small = [ref[n]["customers"] for n in order[:2]]
    a.annotate(f"{small[0]} and {small[1]}\ncustomers", xy=(0.6, 62), xytext=(1.5, 30), fontsize=6.5, color=INK2,
               ha="left", va="center",
               arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkA": 1, "shrinkB": 2})
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TEAL, markeredgecolor="white", markeredgewidth=0.4, markersize=4.6,
               label="four-wire-verified"),
        Line2D([], [], marker="o", ls="", color=CORAL, markeredgewidth=0, markersize=3.4,
               label="positive-sequence model"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=VIOLET, markeredgewidth=0.9,
               markersize=5.6, label="balanced-load model"),
        Line2D([], [], color=CORAL, lw=0.9, label="95% interval over days"),
        Patch(facecolor=GRID, edgecolor="none", label="benchmark network, not counted"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, handletextpad=0.4, columnspacing=1.3,
        handlelength=1.2)
    save(fig, "replication")


if __name__ == "__main__":
    fig_replication()
