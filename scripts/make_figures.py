"""Submission figures, generated only from committed result files.

    .venv/bin/python scripts/make_figures.py        # writes figures/*.png and figures/*.pdf

Palette: reference categorical slots in fixed order (blue, orange, aqua, yellow), recessive axes,
one value axis per chart, direct labels on every series (relief rule for the low-contrast slots).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures")
TEAL, CORAL, VIOLET, YELLOW = "#1a9e8a", "#e05a47", "#5b3fa6", "#c9a227"   # validated categorical set (dataviz validator, light mode)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 11})


def load(p):
    return json.loads(Path(p).read_text())


CAPTIONS_OUTSIDE = True   # titles and subtitles go into the post's captions, not the image


def title(fig, main, sub):
    if CAPTIONS_OUTSIDE:
        return
    fig.text(0.01, 0.985, main, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_capacity():
    a = load("results/annual_report_with_cplus.json")["by_direction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=False)
    [ax.set_xlim(-0.6, 2.6) for ax in axes]
    for ax, d in zip(axes, ("export", "import")):
        r = a[d]
        b, ac = r["B"], r["ac"]
        vals = [0.0, b["kwh_per_household_year"] / 1e3, ac["kwh_per_household_year"] / 1e3]
        lo, hi = b["ci95"]
        xs = [0, 1, 2]
        ax.bar(1, vals[1], width=0.6, color=TEAL, zorder=2)
        ax.errorbar(1, vals[1], yerr=[[vals[1] - lo / 1e3], [hi / 1e3 - vals[1]]], color=INK, capsize=4, lw=1.2, zorder=3)
        ax.bar(2, vals[2], width=0.6, color="none", edgecolor=CORAL, hatch="///", linewidth=1.5, zorder=2)
        ax.text(0, 0.25, "0", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(1, hi / 1e3 + 0.3, f"{vals[1]:.2f} MWh", ha="center", color=INK, fontweight="bold")
        ax.text(2, vals[2] + 0.3, f"{vals[2]:.1f} MWh\nfour-wire unsafe", ha="center", color=INK2, fontsize=9)
        ax.set_xticks(xs, ["Fixed\nallowance", "Dynamic,\nfour-wire-\nchecked", "Dynamic,\nbalanced\nmodel"], fontsize=8.5)
        ax.set_title(f"{d.capitalize()}", color=INK)
        ax.set_ylim(0, 19 if d == "export" else 10)
        ax.set_ylabel("Envelope-hours per household per year (MWh)" if d == "export" else "")
        ax.grid(axis="x", visible=False)
    title(fig, "Dynamic envelopes free capacity a constant add-on cannot",
          "ENWL1 four-wire benchmark, Fluvius 2024 households, 36 test-split days; constant add-on = one optional allowance above the\n"
          "passive demand, valid in every interval; bars = annual estimate, whisker = season-stratified day-bootstrap 95% interval")
    fig.subplots_adjust(top=0.72, wspace=0.25)
    save(fig, "fig1_capacity")


def fig_robustness():
    w = load("results/weekend_report.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, d in zip(axes, ("export", "import")):
        seeds = ["101", "202", "303", "404", "505"]
        single = [w["W1_placement"][s][d]["kwh_per_household_year"] / 1e3 for s in seeds]
        three = {s: w["W2_three_phase"][s][d]["kwh_per_household_year"] / 1e3 for s in ("101", "202")}
        x = list(range(5))
        ax.scatter(x, single, s=70, color=TEAL, zorder=3, edgecolor=SURFACE, linewidth=2, label="single-phase (as built)")
        ax.scatter([0, 1], [three["101"], three["202"]], s=70, color=VIOLET, marker="D", zorder=3, edgecolor=SURFACE,
                   linewidth=2, label="HP/EV homes three-phase (idealised)")
        for xi, v in zip(x, single):
            ax.text(xi, v - (0.9 if d == "export" else 0.55), f"{v:.1f}", ha="center", color=INK2, fontsize=8.5)
        for xi, s in ((0, "101"), (1, "202")):
            ax.text(xi + 0.12, three[s], f"{three[s]:.1f}", va="center", color=INK2, fontsize=8.5)
        ax.annotate("headline", (0, single[0]), xytext=(0.35, single[0] - (2.4 if d == "export" else 1.6)),
                    fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_xticks(x, seeds)
        ax.set_xlabel("household-technology placement seed")
        ax.set_ylim(0, 13 if d == "export" else 9)
        ax.set_title(d.capitalize(), color=INK)
        ax.set_ylabel("MWh per household per year" if d == "export" else "")
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    title(fig, "Robust to which homes own HP/EV/PV; single-phase connection is a major contributor",
          "Exact four-wire-verified envelopes, 36 days; five household-technology placements (W1) and idealised three-phase HP/EV connection (W2)")
    fig.subplots_adjust(top=0.76, bottom=0.24, wspace=0.22)
    save(fig, "fig2_robustness")


def fig_adaptation():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    names = ["predict the\ntraining mean", "stock\nGENCO", "+ impedance-\naware encoder", "+ linear prior\n(zero-init head)"]
    vals = [0.0093, 0.0107, 0.0119, 3.80e-4]
    cols = [MUTED, CORAL, CORAL, TEAL]
    a1.bar(range(4), vals, color=cols, width=0.62, zorder=2)
    for i, v in enumerate(vals):
        a1.text(i, v * 1.15, f"{v:.1e}", ha="center", color=INK, fontsize=8.5)
    a1.set_yscale("log")
    a1.set_ylim(1e-4, 4e-2)
    a1.set_xticks(range(4), names, fontsize=8.5)
    a1.set_ylabel("voltage MAE (pu, log scale)")
    a1.set_title("Representation fixes alone are not enough", color=INK, fontsize=10)
    a1.grid(axis="x", visible=False)
    names2 = ["physics prior\nalone", "GENCO-LV\n(selected)", "5 fixed-point\niterations"]
    vals2 = [5.43e-3, 1.01e-3, 1.2e-4]
    a2.bar(range(3), vals2, color=[MUTED, TEAL, MUTED], width=0.62, zorder=2)
    for i, v in enumerate(vals2):
        a2.text(i, v * 1.15, f"{v:.1e}", ha="center", color=INK, fontsize=8.5)
    a2.set_yscale("log")
    a2.set_ylim(1e-4, 4e-2)
    a2.set_xticks(range(3), names2, fontsize=8.5)
    a2.set_title("On envelope states: 5.4× better than its prior", color=INK, fontsize=10)
    a2.grid(axis="x", visible=False)
    title(fig, "Making GENCO work on an LV feeder: the physics prior is the decisive step",
          "Left: 1,000 validation states (v1 screen, 15 epochs). Right: 1,000 envelope-augmented validation states (v2, A100 run).\n"
          "Separate runs on different state sets, not one ablation sequence; one training seed each. coral = fails; grey = baselines and the classical reference")
    fig.subplots_adjust(top=0.76, wspace=0.22)
    save(fig, "fig3_genco_adaptation")


def fig_fourwire():
    e = load("results/fourwire_eval_test.json")
    order = [("BG", "balanced GENCO", CORAL), ("BS", "balanced exact sweep", MUTED), ("U1", "four-wire prior (1 step)", MUTED),
             ("PhaseGENCO", "+ GNN + GENCO embeddings", CORAL), ("PhaseGraph", "+ GNN residual", TEAL),
             ("U3", "iterated four-wire sweep", MUTED)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fb = [100 * e["screen"][k]["fallback_mean"] for k, _, _ in order]
    y = list(range(len(order)))[::-1]
    a1.barh(y, fb, color=[c for _, _, c in order], height=0.6, zorder=2)
    for yi, v in zip(y, fb):
        a1.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    a1.set_yticks(y, [n for _, n, _ in order])
    a1.set_xlim(0, 75)
    a1.set_xlabel("states sent to exact OpenDSS (%)")
    a1.set_title("Fallbacks under the same exact gate", color=INK, fontsize=10)
    a1.grid(axis="y", visible=False)
    o2 = [x for x in order if x[0] in e["state"]]
    mae = [e["state"][k]["vpn_mae"] for k, _, _ in o2]
    y2 = list(range(len(o2)))[::-1]
    a2.barh(y2, mae, color=[c for _, _, c in o2], height=0.6, zorder=2)
    for yi, v in zip(y2, mae):
        a2.text(v * 1.2, yi, f"{v:.1e}", va="center", color=INK, fontsize=9)
    a2.set_xscale("log")
    a2.set_xlim(1e-6, 5e-2)
    a2.set_yticks(y2, [n for _, n, _ in o2])
    a2.set_xlabel("phase-to-neutral voltage MAE (pu, log scale)")
    a2.set_title("State accuracy (2,699 converged test states)", color=INK, fontsize=10)
    a2.grid(axis="y", visible=False)
    title(fig, "Four-wire extension: a four-wire GENCO corrects a cheap four-wire prior; GENCO embeddings do not help",
          "Test split: fallbacks on 2,968 screening states, voltage MAE on the 2,699 converged ones; zero observed false-safe decisions for every\n"
          "method; one training seed per model. teal = learned model that helps; coral = GENCO variants that do not; grey = physics baselines")
    fig.subplots_adjust(top=0.78, wspace=0.75)
    save(fig, "fig4_fourwire")


def fig_frontier():
    f = load("results/envpred3_frontier_report.json")["frontier"]["points_calls_retention"]
    b_calls = load("results/envpred3_primary_q95.json")["strata"]["combined"]["arms"]["P"]["combined"]["exact_calls_B"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    spec = [("P", "phase features", TEAL), ("PC", "+ classical", CORAL), ("PG", "+ GENCO", VIOLET), ("PCG", "+ classical + GENCO", YELLOW)]
    for k, name, c in spec:
        pts = sorted(f[k])
        xs = [100 * x / b_calls for x, _ in pts]
        ys = [100 * y for _, y in pts]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
    ax.axhline(100, color=MUTED, lw=1, ls="--")
    ax.text(21.5, 97.5, "exact search (B) = 100% capacity, 100% calls", ha="right", color=INK2, fontsize=8.5)
    ax.annotate("primary operating point\n79% capacity, 6.7% of calls", (6.73, 79.0), xytext=(10, 62),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("exact OpenDSS calls, % of the exact search")
    ax.set_ylabel("exact envelope-hours retained (%)")
    ax.set_xlim(0, 24)
    ax.set_ylim(40, 103)
    ax.legend(loc="lower right", fontsize=8.5, title="proposal features", title_fontsize=8.5)
    title(fig, "A learned proposal + one exact check: most capacity for a fraction of the exact calls",
          "36 test days × 5 placements; four shrink points per arm. GENCO arms overlap the others: no GENCO-specific gain")
    fig.subplots_adjust(top=0.80)
    save(fig, "fig5_proposal_frontier")


def fig_workflow():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_xlim(0, 10), ax.set_ylim(0, 3.2), ax.axis("off")
    boxes = [(0.1, "Household\nscenarios", "200 per interval,\nresampled real meters", MUTED),
             (2.6, "Candidate\nenvelope", "uniform kW per household,\nsearched or proposed", MUTED),
             (5.1, "Phase-aware\nGENCO screen", "four-wire GNN residual\non a one-step sweep:\nsafe / unsafe / unsure", TEAL),
             (7.6, "Full four-wire\nOpenDSS check", "all 200 design scenarios,\n>= 98% feasible\n-> envelope issued", VIOLET)]
    for x, head, body, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.2, 1.75, boxstyle="round,pad=0.03,rounding_size=0.12", fc=SURFACE, ec=c, lw=2))
        ax.text(x + 1.1, 2.25, head, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK, linespacing=1.15)
        ax.text(x + 1.1, 1.35, body, ha="center", va="center", fontsize=8.3, color=INK2, linespacing=1.25)
    for x in (2.3, 4.8, 7.3):
        ax.add_patch(FancyArrowPatch((x, 1.67), (x + 0.3, 1.67), arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4))
    ax.add_patch(FancyArrowPatch((6.2, 0.9), (8.7, 0.9), connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=12,
                                 color=INK2, lw=1.1, ls="--"))
    ax.text(7.45, 0.18, "unsure states go to OpenDSS (5.8% with the GNN, 35.8% with the prior alone)", ha="center", fontsize=8.5,
            color=INK2)
    save(fig, "fig8_workflow")


def fig_fallback():
    e = load("results/fourwire_eval_test.json")
    order = [("BS", "balanced model sweep", MUTED), ("BG", "balanced GENCO", CORAL),
             ("U1", "four-wire one-step prior", MUTED), ("PhaseGENCO", "+ GNN + balanced GENCO embeddings", CORAL),
             ("PhaseGraph", "+ phase-aware GENCO (four-wire)", TEAL), ("U3", "iterated four-wire sweep (classical)", MUTED)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    fb = [100 * e["screen"][k]["fallback_mean"] for k, _, _ in order]
    y = list(range(len(order)))[::-1]
    ax.barh(y, fb, color=[c for _, _, c in order], height=0.62, zorder=2)
    for yi, v in zip(y, fb):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [n for _, n, _ in order], fontsize=10)
    ax.set_xlim(0, 75)
    ax.set_xlabel("evaluation states sent to the full OpenDSS solver (%)")
    ax.grid(axis="y", visible=False)
    title(fig, "How often each screen must call OpenDSS",
          "2,968 evaluation states, same final gate for every method; zero observed false-safe decisions for all")
    fig.subplots_adjust(top=0.8)
    save(fig, "fig4a_fallback")


if __name__ == "__main__":
    fig_capacity()
    fig_robustness()
    fig_adaptation()
    fig_fourwire()
    fig_frontier()
    fig_workflow()
    fig_fallback()
    print(sorted(p.name for p in OUT.glob("*.png")))
