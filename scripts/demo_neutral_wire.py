"""Public, runnable demo (no Fluvius data): the same household injections on ENWL1, balanced model versus four-wire.

    .venv/bin/python scripts/demo_neutral_wire.py            # prints a summary, writes figures/demo_neutral_wire.png

One synthetic afternoon state with a fixed seed: rooftop-PV homes export, EV homes charge, the rest draw a small load.
The balanced positive-sequence model lumps each customer at its bus; OpenDSS solves the four-wire feeder with every
customer on its own phase and the neutral as a conductor. Only the public ENWL1 files in data/feeder_topology are used.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from gencodoe import feeder as fdr  # noqa: E402
from gencodoe.enwl import load_enwl1  # noqa: E402
from gencodoe.fourwire_net import load_fourwire  # noqa: E402
from gencodoe.unbalanced import UnbalancedENWL1, diagnose  # noqa: E402

V_BASE, VMIN, VMAX = 240.18, 216.2, 253.0
BLUE, ORANGE, AQUA, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"


def state(n: int = 200, seed: int = 0):
    """Customer net consumption (kW, negative = export) and kvar: 35% PV homes exporting, 25% EV homes charging."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=n, p=[0.35, 0.25, 0.40])                  # 0 PV export, 1 EV charging, 2 base load
    p = np.where(kind == 0, -rng.uniform(3.0, 5.0, n), np.where(kind == 1, rng.uniform(5.0, 7.4, n), rng.uniform(0.2, 1.0, n)))
    q = np.where(p > 0, p * np.tan(np.arccos(0.95)), 0.0)
    return p, q


def main():
    p_kw, q_kvar = state()
    fw = diagnose(UnbalancedENWL1(), p_kw, q_kvar)
    v4 = fw["v_pn"] * V_BASE
    net = load_enwl1()
    pb = np.zeros((1, net.n_bus))
    qb = np.zeros_like(pb)
    np.add.at(pb, (0, net.customer_bus), -p_kw / 1e3)
    np.add.at(qb, (0, net.customer_bus), -q_kvar / 1e3)
    vb = fdr.sweep_pf(net.to_feeder(), pb, qb).vm[0, net.customer_bus] * V_BASE

    four = load_fourwire()
    dist = np.zeros(four.n_bus)
    for i in four.order[1:]:
        dist[i] = dist[four.parent[i]] + four.length_m[i]
    cd = dist[four.customer_bus]
    ph = fw["customer_phase"] - 1

    out = lambda v: int(((v < VMIN) | (v > VMAX)).sum())  # noqa: E731
    print(f"net feeder load {p_kw.sum():.0f} kW; four-wire converged: {fw['converged']}")
    print(f"balanced model : {vb.min():.1f}-{vb.max():.1f} V, customers outside {VMIN}-{VMAX} V: {out(vb)}")
    print(f"four-wire      : {v4.min():.1f}-{v4.max():.1f} V, customers outside {VMIN}-{VMAX} V: {out(v4)}")
    print(f"transformer LV neutral current: {fw['tr_lv_neutral_current_a']:.0f} A")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, c in enumerate((BLUE, ORANGE, AQUA)):
        m = ph == k
        ax.scatter(cd[m], v4[m], s=14, color=c, edgecolor="white", linewidth=0.4, label=f"four-wire, phase {'abc'[k]}", zorder=3)
    ax.scatter(cd, vb, s=8, color=INK, label="balanced model", zorder=4)
    for v in (VMIN, VMAX):
        ax.axhline(v, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Distance from transformer (m)")
    ax.set_ylabel("Customer voltage (V)")
    ax.set_title("Same injections, two models (synthetic state, public ENWL1 files)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), handletextpad=0.1)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/demo_neutral_wire.png", dpi=200, bbox_inches="tight")
    print("wrote figures/demo_neutral_wire.png")


if __name__ == "__main__":
    main()
