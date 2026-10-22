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
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PHASE = (BLUE, ORANGE, AQUA)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc0ee", "#2a78d6", "#123f7a"])
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

    fig = plt.figure(figsize=(DCOL, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.08)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, k]) for k in range(3))
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
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, handletextpad=0.0, columnspacing=0.3,
               borderaxespad=0, fontsize=6)
    ax1.set_title("Customer phase")
    panel(ax1, "a")
    volts = vpn[s] * 240.18
    order = np.argsort(np.abs(volts - 234.6))                            # out-of-limit customers drawn on top
    sc = ax2.scatter(cxy[order, 0], cxy[order, 1], c=volts[order], s=12, edgecolor="white", linewidth=0.4,
                     zorder=3, cmap=LinearSegmentedColormap.from_list("div", [ORANGE, "#e6e5e1", BLUE]),
                     norm=TwoSlopeNorm(vcenter=234.6, vmin=200, vmax=270))
    cax = ax2.inset_axes([0.1, -0.08, 0.8, 0.04])
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
    ax3.scatter(cdist, vm_bal[s] * 240.18, s=5, color=INK, zorder=4, label="balanced model")
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
        ax.scatter(lx, lb, s=9, color=ORANGE, edgecolor="white", linewidth=0.3, zorder=3)
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
        for meth, c, lab in (("b_opendss_exact", BLUE, "four-wire-verified"), ("ac", ORANGE, "balanced")):
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
        for v, mk, c in zip(vals, ("o", "s", "^"), (BLUE, ORANGE, AQUA)):
            ax.scatter(v, k, marker=mk, s=22, color=c, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(4), list(names.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Phase–neutral voltage error (pu)")
    for mk, c, lab in (("o", BLUE, "MAE"), ("s", ORANGE, "p95"), ("^", AQUA, "p99")):
        ax.scatter([], [], marker=mk, s=22, color=c, label=lab)
    ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1)
    ax.set_title("State accuracy")
    panel(ax, "a")

    ax = axes[1]
    order = ["BS", "U1", "PhaseGENCO", "PhaseGraph", "U3"]
    lab = {"BS": "balanced screen", **names}
    yy = np.arange(len(order))
    for d, c, off in (("import", ORANGE, 0.17), ("export", BLUE, -0.17)):
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
    arms = {"P": ("phase features", BLUE, "o"), "PC": ("+ classical", AQUA, "s"),
            "PG": ("+ GENCO", ORANGE, "^"), "PCG": ("+ both", YELLOW, "D")}
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


FIGS = {"feeder": fig_feeder, "heatmap": fig_heatmap, "safety": fig_safety, "surrogates": fig_surrogates}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
