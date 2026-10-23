"""Figures for the IEEE paper (paper/figures/*.pdf), in print style at IEEE column widths.

    .venv/bin/python scripts/paper_figures.py           # all figures
    .venv/bin/python scripts/paper_figures.py feeder    # one of: feeder, heatmap, safety, surrogates

Inputs: committed result files, the per-day annual DOE runs in data_out/doe/annual (36 test days, B and the
balanced 'ac' arm with their four-wire checks) and the validation-split four-wire labels (never test labels).
Palette: the categorical order of scripts/make_figures.py (blue, orange, aqua), one sequential hue.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, "src")
OUT = Path("paper/figures")
TEAL, CORAL, VIOLET, YELLOW = "#1a9e8a", "#e05a47", "#5b3fa6", "#c9a227"   # validated categorical set (dataviz validator, light mode)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PHASE = (VIOLET, CORAL, TEAL)                     # phases a, b, c
SEQ = LinearSegmentedColormap.from_list("seq_teal", ["#eaf6f3", "#8fd0c2", "#1a9e8a", "#0b4f45"])
COL, DCOL = 3.5, 7.16                                    # IEEE single / double column width (in)
VMIN_PU, VMAX_PU = 216.2 / 240.18, 253.0 / 240.18

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.dpi": 300})


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold",
            color=INK)


# ------------------------------------------------------------------------------------------ feeder
def _coords():
    xy = {}
    for line in Path("data/feeder_topology/ENWL1/Buscoords.dss").read_text().splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) == 3:
            xy[parts[0].lower()] = (float(parts[1]), float(parts[2]))
    segs = []
    for line in Path("data/feeder_topology/ENWL1/Lines.dss").read_text().splitlines():
        if line.lower().startswith("new"):
            kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
            a, b = kv["Bus1"].split(".")[0].lower(), kv["Bus2"].split(".")[0].lower()
            if a in xy and b in xy:
                segs.append((xy[a], xy[b]))
    return xy, segs


def fig_feeder():
    from gencodoe import feeder as fdr
    from gencodoe.enwl import load_enwl1
    from gencodoe.fourwire_labels import load_split
    from gencodoe.fourwire_net import load_fourwire, phase_neutral_pu

    fw, bal = load_fourwire(), load_enwl1()
    xy, segs = _coords()
    d = load_split("val")
    ok = d["converged"]
    V, p_kw, q_kvar = d["V"][ok], d["p_kw"][ok].astype(float), d["q_kvar"][ok].astype(float)
    vpn = phase_neutral_pu(fw, V)                                         # (S, 200)
    # balanced positive-sequence model of the same states (customers lumped at their bus)
    feeder = bal.to_feeder()
    p = np.zeros((len(p_kw), bal.n_bus))
    q = np.zeros_like(p)
    np.add.at(p, (slice(None), bal.customer_bus), -p_kw / 1e3)
    np.add.at(q, (slice(None), bal.customer_bus), -q_kvar / 1e3)
    vm_bal = fdr.sweep_pf(feeder, p, q).vm[:, bal.customer_bus]           # (S, 200)
    # illustrative state: the balanced model sees no undervoltage, four-wire does; largest gap
    gap = vm_bal.min(1) - vpn.min(1)
    # restricted to plausible operating states (four-wire minimum above 0.85 pu, about 204 V)
    cand = np.flatnonzero((vm_bal.min(1) >= VMIN_PU) & (vpn.min(1) < VMIN_PU) & (vpn.min(1) > 0.85))
    s = cand[np.argmax(gap[cand])] if len(cand) else int(np.argmax(gap))
    print(f"feeder: state {s} of {len(gap)} val states; {len(cand)} with balanced-ok/four-wire-under; "
          f"min V balanced {vm_bal[s].min():.4f}, four-wire {vpn[s].min():.4f} pu; "
          f"feeder net load {p_kw[s].sum():.0f} kW")

    dist = np.zeros(fw.n_bus)
    for i in fw.order[1:]:
        dist[i] = dist[fw.parent[i]] + fw.length_m[i]
    cdist = dist[fw.customer_bus]
    cxy = np.array([xy[fw.bus_names[b]] for b in fw.customer_bus])

    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(DCOL, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[3.2, 1], wspace=0.08, hspace=0.25)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[:, 2])
    for ax in (ax1, ax2):
        ax.add_collection(LineCollection(segs, colors=GRID, linewidths=1.0, zorder=1))
        tr = xy["trbus"]
        ax.scatter(*tr, marker="s", s=30, color=INK, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ph, name in enumerate("abc"):
        m = fw.customer_phase == ph
        ax1.scatter(cxy[m, 0], cxy[m, 1], s=11, color=PHASE[ph], edgecolor="white", linewidth=0.5, zorder=3,
                    label=f"phase {name} ({m.sum()})")
    # technologies per phase in the headline household placement (seed 101, doe_day.DayScenarios: one uniform draw
    # per customer and technology against the 30/30/40 mix, Loads.dss order)
    tech = np.random.default_rng(101).uniform(size=(len(fw.customer_phase), 3)) < np.array([0.3, 0.3, 0.4])
    ins = fig.add_subplot(gs[1, 0])
    w = 0.26
    for k, (name, shade) in enumerate((("HP", INK2), ("EV", MUTED), ("PV", "#c9c8c3"))):
        n = [int(tech[fw.customer_phase == ph, k].sum()) for ph in range(3)]
        ins.bar(np.arange(3) + (k - 1) * w, n, width=w, color=shade, label=name)
    ins.set_xticks(range(3), [f"phase {'abc'[ph]} ({(fw.customer_phase == ph).sum()})" for ph in range(3)], fontsize=6)
    for t, c in zip(ins.get_xticklabels(), PHASE):
        t.set_color(c)
    ins.tick_params(axis="y", labelsize=5, length=2)
    ins.tick_params(axis="x", length=0, pad=1)
    ins.set_ylabel("homes", fontsize=6)
    ins.legend(fontsize=5.5, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.35), handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
    ins.spines[["top", "right"]].set_visible(False)
    ins.grid(False)
    ins.set_facecolor("none")
    ax1.set_title("Customer phase")
    print("feeder: technologies per phase (seed 101):",
          {"abc"[ph]: tech[fw.customer_phase == ph].sum(0).tolist() for ph in range(3)})
    panel(ax1, "a")
    volts = vpn[s] * 240.18
    order = np.argsort(np.abs(volts - 234.6))                            # out-of-limit customers drawn on top
    sc = ax2.scatter(cxy[order, 0], cxy[order, 1], c=volts[order], s=12, edgecolor="white", linewidth=0.4,
                     zorder=3, cmap=LinearSegmentedColormap.from_list("div", [CORAL, "#e6e5e1", VIOLET]),
                     norm=TwoSlopeNorm(vcenter=234.6, vmin=200, vmax=270))
    cax = ax2.inset_axes([0.1, -0.12, 0.8, 0.045])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([216.2, 253], labels=["216.2 V", "253 V"])
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax2.set_title("Four-wire voltage")
    panel(ax2, "b")

    for ph in range(3):
        m = fw.customer_phase == ph
        ax3.scatter(cdist[m], volts[m], s=9, color=PHASE[ph], edgecolor="white", linewidth=0.3, zorder=3,
                    label=f"four-wire, phase {'abc'[ph]}")
    ax3.scatter(cdist, vm_bal[s] * 240.18, s=9, facecolor="none", edgecolor=MUTED, linewidth=0.6, zorder=2,
                label="balanced model")
    net = [p_kw[s][fw.customer_phase == ph].sum() for ph in range(3)]
    ax3.text(0.02, 0.02, "net load per phase (kW): " + ", ".join(f"{'abc'[ph]} {net[ph]:+.0f}" for ph in range(3)),
             transform=ax3.transAxes, fontsize=6, color=INK2)
    print("feeder: net load per phase of the shown state (kW, + = import):", [round(x, 1) for x in net])
    for v, t in ((216.2, "216.2 V minimum"), (253.0, "253 V maximum")):
        ax3.axhline(v, color=INK2, lw=0.8, ls="--", zorder=2)
        ax3.text(cdist.max(), v + 0.8, t, ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax3.set_xlabel("Distance from transformer (m)")
    ax3.set_ylabel("Customer phase–neutral voltage (V)")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(True)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.1, columnspacing=0.8,
               borderaxespad=0, markerscale=1.5, fontsize=6.5)
    ax3.set_title("Same state along the feeder")
    panel(ax3, "c")
    save(fig, "feeder")


# ----------------------------------------------------------------------------------------- annual rows
def _annual():
    import pandas as pd
    rows = []
    for f in sorted(Path("data_out/doe/annual").glob("*_annual_4w_b_opendss_exact_*.json")):
        d = load(f)
        for r in d["rows"]:
            rows.append({k: r[k] for k in ("time", "direction", "method", "limit_kw", "prob_ok_unbalanced",
                                           "prob_ok_ac_fresh", "unbalanced_v_min", "unbalanced_v_max",
                                           "unbalanced_line_loading_max")} | {"day": d["meta"]["day"]})
    df = pd.DataFrame(rows)
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    return df


def fig_heatmap():
    df = _annual()
    b = df[df.method == "b_opendss_exact"]
    days = sorted(b.day.unique())
    vmax = b.limit_kw.max()
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 3.3), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, d, cap, letter in zip(axes, ("export", "import"), (5.0, 7.4), "ab"):
        m = b[b.direction == d].pivot(index="day", columns="hour", values="limit_kw").reindex(days)
        a = m.to_numpy()
        im = ax.imshow(np.where(a > 0, a, np.nan), aspect="auto", cmap=SEQ, vmin=0, vmax=vmax,
                       interpolation="nearest")
        zero = np.where(a == 0, 1.0, np.nan)
        ax.imshow(zero, aspect="auto", cmap=LinearSegmentedColormap.from_list("z", ["#d9d8d4"] * 2),
                  interpolation="nearest")
        ax.set_xticks(range(0, 24, 3), [f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlabel("Hour of day")
        ax.grid(False)
        ax.set_title(f"{d.capitalize()} envelope (cap {cap:g} kW)")
        panel(ax, letter)
        print(f"heatmap {d}: zero intervals {np.mean(a == 0):.1%}, mean {np.nanmean(a):.2f} kW")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("Envelope (kW per household)")
    cb.outline.set_visible(False)
    month = [dy[5:7] for dy in days]
    ticks = [i for i in range(len(days)) if i == 0 or month[i] != month[i - 1]]
    axes[0].set_yticks(ticks, [days[i][:7] for i in ticks])
    axes[0].set_ylabel("Test day (2024)")
    fig.text(0.13, -0.02, "grey = no positive envelope verified",
             fontsize=6.5, color=INK2)
    save(fig, "heatmap")


def fig_safety():
    df = _annual()
    key = ["day", "hour", "direction"]
    w = df.pivot_table(index=key, columns="method", values=["limit_kw", "prob_ok_unbalanced"]).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.25],
                                                                      "wspace": 0.35})
    for ax, d, letter in zip(axes[:2], ("export", "import"), "ab"):
        x = w.xs(d, level="direction")
        lb, lx = x[("limit_kw", "ac")], x[("limit_kw", "b_opendss_exact")]
        meets = (x[("prob_ok_unbalanced", "ac")] >= 0.95) & (lb > 0)
        hi = max(lb.max(), lx.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(hi * 0.97, hi * 0.9, "equal", ha="right", fontsize=6.5, color=MUTED, rotation=45)
        ax.scatter(lx, lb, s=9, color=CORAL, edgecolor="white", linewidth=0.3, zorder=3)
        n_pos, n_zero = int((lb > 0).sum()), int(((lb > 0) & (lx == 0)).sum())
        ax.text(0.97, 0.05, f"{len(x)} intervals\n{int(meets.sum())} of {n_pos} balanced envelopes\nmeet 0.95 "
                f"four-wire\n{n_zero} where four-wire allows 0", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, color=INK2)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_xlabel("Four-wire-verified envelope (kW)")
        ax.set_ylabel("Balanced-model envelope (kW)" if d == "export" else "")
        ax.set_title(d.capitalize())
        panel(ax, letter)
        nz = lx > 0
        print(f"safety {d}: n={len(x)}, balanced>0 {n_pos}, meets 0.95 {int(meets.sum())}, B=0 & balanced>0 {n_zero}, mean four-wire feasibility "
              f"of balanced {x[('prob_ok_unbalanced', 'ac')].mean():.3f}, median ratio balanced/B (B>0) "
              f"{np.median(lb[nz] / lx[nz]):.2f}")
    ax = axes[2]
    for d, ls in (("export", "-"), ("import", "--")):
        x = w.xs(d, level="direction")
        for meth, c, lab in (("b_opendss_exact", TEAL, "four-wire-verified"), ("ac", CORAL, "balanced")):
            v = x[("prob_ok_unbalanced", meth)][x[("limit_kw", meth)] > 0].to_numpy()
            v = np.sort(v)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, ls=ls, lw=1.3)
            if meth == "b_opendss_exact":
                print(f"safety {d}: B non-zero mean four-wire fresh feasibility {v.mean():.3f}, "
                      f"share >= 0.95 {np.mean(v >= 0.95):.1%}")
    ax.axvline(0.95, color=INK2, lw=0.9, ls=":")
    ax.text(0.94, 0.03, "0.95", ha="right", fontsize=6.5, color=INK2)
    ax.set_xlabel("Four-wire feasibility on fresh draws")
    ax.set_ylabel("Cumulative share of envelopes")
    ax.set_title("Out-of-sample feasibility")
    for xt, yt, t, ha in ((0.13, 0.55, "balanced,\nexport", "left"), (0.47, 0.42, "balanced,\nimport", "left"),
                          (0.9, 0.6, "four-wire-\nverified\n(both)", "right")):
        ax.text(xt, yt, t, ha=ha, fontsize=6.5, color=INK2)
    ax.plot([], [], color=INK2, ls="-", label="export")
    ax.plot([], [], color=INK2, ls="--", label="import")
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.1)
    panel(ax, "c")
    save(fig, "safety")


# --------------------------------------------------------------------------------------- surrogates
def fig_surrogates():
    e = load("results/fourwire_eval_test.json")
    f = load("results/envpred3_frontier_report.json")
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.45), gridspec_kw={"width_ratios": [1.05, 1, 1.15],
                                                                      "wspace": 0.75})
    names = {"U1": "one-step sweep", "PhaseGraph": "four-wire GNN", "PhaseGENCO": "GNN + GENCO emb.",
             "U3": "iterated sweep"}
    ax = axes[0]
    for k, (m, lab) in enumerate(names.items()):
        st = e["state"][m]
        vals = [st["vpn_mae"], st["vpn_p95"], st["vpn_p99"]]
        ax.plot(vals, [k] * 3, color=MUTED, lw=1, zorder=1)
        for v, mk, c in zip(vals, ("o", "s", "^"), (TEAL, CORAL, VIOLET)):
            ax.scatter(v, k, marker=mk, s=22, color=c, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(4), list(names.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Phase–neutral voltage error (pu)")
    for mk, c, lab in (("o", TEAL, "MAE"), ("s", CORAL, "p95"), ("^", VIOLET, "p99")):
        ax.scatter([], [], marker=mk, s=22, color=c, label=lab)
    ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1)
    ax.set_title("State accuracy")
    panel(ax, "a")

    ax = axes[1]
    order = ["BS", "U1", "PhaseGENCO", "PhaseGraph", "U3"]
    lab = {"BS": "balanced screen", **names}
    yy = np.arange(len(order))
    for d, c, off in (("import", CORAL, 0.17), ("export", TEAL, -0.17)):
        v = [e["screen"][m][d]["fallback"] * 100 for m in order]
        ax.barh(yy + off, v, height=0.32, color=c, label=d, zorder=2)
    for y, m in zip(yy, order):
        top = max(e["screen"][m][d]["fallback"] for d in ("import", "export")) * 100
        ax.text(top + 2, y, f"mean {e['screen'][m]['fallback_mean']:.1%}",
                va="center", fontsize=6.5, color=INK)
    ax.set_yticks(yy, [lab[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 95)
    ax.set_xlabel("States sent to OpenDSS (%)")
    ax.legend(loc="lower right", borderaxespad=0.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Exact-solver fallback")
    panel(ax, "b")
    print("surrogates: false-safe totals", {m: e["screen"][m]["false_safe_total"] for m in order})

    ax = axes[2]
    b_calls = f["cost_test"]["B_exact_calls"]
    arms = {"P": ("phase features", TEAL, "o"), "PC": ("+ classical", VIOLET, "s"),
            "PG": ("+ GENCO", CORAL, "^"), "PCG": ("+ both", YELLOW, "D")}
    for a, (lab, c, mk) in arms.items():
        pts = np.array(f["frontier"]["points_calls_retention"][a])
        ax.plot(pts[:, 0] / b_calls * 100, pts[:, 1] * 100, color=c, marker=mk, ms=4, lw=1.2,
                markeredgecolor="white", markeredgewidth=0.5, label=lab)
    ax.scatter([100], [100], color=INK, s=18, zorder=4)
    ax.annotate("exact search B", (100, 100), xytext=(-4, -9), textcoords="offset points", ha="right",
                fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Exact OpenDSS calls (% of B)")
    ax.set_ylabel("Envelope-hours retained (% of B)")
    ax.set_ylim(45, 103)
    ax.legend(loc="lower right", borderaxespad=0.1)
    ax.set_title("Proposal + one exact check")
    panel(ax, "c")
    save(fig, "surrogates")


# ------------------------------------------------------------------------------------------ concept
def fig_concept():
    """(a) the four-wire feeder and what a balanced model sees; (b) one day of verified envelopes (2024-07-28)."""
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DCOL, 2.7), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.32})
    a1.set_xlim(0, 10), a1.set_ylim(0, 7.2), a1.axis("off")
    names, ys = ("a", "b", "c", "n"), (5.6, 5.0, 4.4, 3.8)
    a1.add_patch(Rectangle((0.2, 3.5), 0.9, 2.4, fc="white", ec=INK, lw=1))
    a1.text(0.65, 4.7, "trafo", ha="center", va="center", fontsize=6, color=INK)
    for name, y, c in zip(names, ys, (*PHASE, MUTED)):
        a1.plot([1.1, 9.6], [y, y], color=c, lw=1.6 if name != "n" else 1.2, ls="-" if name != "n" else "--")
        a1.text(9.7, y, name, va="center", fontsize=7, color=c, fontweight="bold")
    def house(x, y, c):
        a1.add_patch(Polygon([(x - 0.28, y), (x + 0.28, y), (x + 0.28, y + 0.38), (x, y + 0.62), (x - 0.28, y + 0.38)],
                             closed=True, fc=c, ec="white", lw=0.6, zorder=3))
    for k, (x, ph) in enumerate(zip((2.2, 3.4, 4.6, 5.8, 7.0, 8.2), (0, 2, 0, 1, 0, 2))):
        top = ys[ph]
        a1.plot([x, x], [top, ys[3]], color=PHASE[ph], lw=0.9)
        house(x, 6.25, PHASE[ph])
        a1.plot([x, x], [top, 6.25], color=PHASE[ph], lw=0.9)
        a1.scatter([x, x], [top, ys[3]], s=7, color=[PHASE[ph], MUTED], zorder=4)
    a1.text(0.2, 7.0, "Four-wire: each home on one phase + neutral", fontsize=7, color=INK, fontweight="bold")
    a1.add_patch(FancyBboxPatch((0.2, 0.25), 9.6, 2.65, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f3f3f1", ec="none"))
    a1.plot([1.1, 9.6], [1.35, 1.35], color=INK2, lw=2.2)
    a1.add_patch(Rectangle((0.35, 0.95), 0.75, 0.8, fc="white", ec=INK2, lw=1))
    for x in (2.8, 5.0, 7.2, 9.0):
        a1.plot([x, x], [1.35, 1.85], color=INK2, lw=0.9)
        a1.add_patch(Rectangle((x - 0.32, 1.85), 0.64, 0.4, fc=MUTED, ec="white", lw=0.6))
    a1.text(0.4, 2.5, "Balanced model (GENCO's view)", fontsize=7, color=INK2, fontweight="bold")
    a1.text(0.4, 0.45, "neutral Kron-reduced: per-phase voltages averaged away", fontsize=6.3, color=INK2, style="italic")
    panel(a1, "a")

    d = load("data_out/doe/annual/2024-07-28_annual_4w_b_opendss_exact_sph1_seed101_uk_esqcr.json")
    lim = {dr: [r["limit_kw"] for r in d["rows"] if r["method"] == "b_opendss_exact" and r["direction"] == dr]
           for dr in ("import", "export")}
    h = np.arange(24)
    a2.bar(h, lim["import"], width=0.8, color=TEAL, label="import allowance")
    a2.bar(h, [-v for v in lim["export"]], width=0.8, color=VIOLET, label="export allowance")
    a2.axhline(0, color=INK, lw=0.8)
    a2.axvspan(8.5, 17.5, color=GRID, alpha=0.5, zorder=0)
    a2.text(13, -0.6, "no export\nallowance\nat midday", ha="center", fontsize=6.3, color=INK2)
    a2.set_xticks(range(0, 24, 3), [f"{x:02d}" for x in range(0, 24, 3)])
    a2.set_xlim(-0.6, 23.6), a2.set_ylim(-1.5, 1.3)
    a2.set_xlabel("Hour, 28 July 2024")
    a2.set_ylabel("Allowance per household (kW)", labelpad=8)
    a2.set_yticks([-1, -0.5, 0, 0.5, 1], ["1.0", "0.5", "0", "0.5", "1.0"])
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.3, ncol=2, borderaxespad=0)
    a2.set_title("A dynamic operating envelope, one day")
    panel(a2, "b")
    save(fig, "concept")


FIGS = {"concept": fig_concept, "feeder": fig_feeder, "heatmap": fig_heatmap, "safety": fig_safety, "surrogates": fig_surrogates}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
