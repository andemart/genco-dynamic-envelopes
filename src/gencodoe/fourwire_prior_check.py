"""Smoke gate 2 (docs/unbalanced_genco_protocol.md): U1 prior and iterated sweep U3 against OpenDSS on
train/val states, by state category. No test data."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .fourwire_labels import load_split
from .fourwire_net import V_PN_NOM, Thevenin, load_fourwire, phase_neutral_pu, sweep


def categories(p_kw, phase):
    tot = p_kw.sum(1)
    ph = np.stack([p_kw[:, phase == k].sum(1) for k in range(3)], 1)
    spread = ph.max(1) - ph.min(1)
    cat = np.full(len(tot), "normal", dtype=object)
    cat[spread > 40] = "imbalanced"
    cat[tot > 150] = "import_heavy"
    cat[tot < -100] = "export_heavy"
    return cat


def main(split="val", n=1500):
    net = load_fourwire()
    z = np.load("data_out/fourwire/thevenin.npz")
    th = Thevenin(e=z["e"], zth=z["zth"])
    d = load_split(split)
    c = np.flatnonzero(d["converged"])[:n]
    s = (d["p_kw"][c] + 1j * d["q_kvar"][c]).astype(complex)
    vt = d["V"][c].astype(complex)
    it_t = d["I"][c].astype(complex)
    t = time.perf_counter()
    v1, i1, _ = sweep(net, th, s, iterations=1)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    v3, i3, its = sweep(net, th, s, iterations=50, tol=1e-8)
    t3 = time.perf_counter() - t
    vpn_t = phase_neutral_pu(net, vt)
    cat = categories(d["p_kw"][c], net.customer_phase)
    rep = {"split": split, "states": int(len(c)), "u3_iterations": int(its),
           "ms_per_state": {"u1": 1e3 * t1 / len(c), "u3": 1e3 * t3 / len(c)}, "by_category": {}}
    rat = net.rating_a[1:] / 1e3
    for name, v, i in (("u1", v1, i1), ("u3", v3, i3)):
        vpn = phase_neutral_pu(net, v)
        for k in ["all"] + sorted(set(cat)):
            m = np.ones(len(c), bool) if k == "all" else cat == k
            e = np.abs(vpn[m] - vpn_t[m])
            rep["by_category"].setdefault(k, {"states": int(m.sum())})[name] = {
                "vpn_mae": float(e.mean()), "vpn_p99": float(np.percentile(e, 99)), "vpn_max": float(e.max()),
                "worst_customer_err": float(np.abs(vpn[m].min(1) - vpn_t[m].min(1)).max()),
                "neutral_v_mae_pu": float(np.abs(np.abs(v[m][:, :, 3]) - np.abs(vt[m][:, :, 3])).mean() / V_PN_NOM),
                "phase_loading_mae": float(np.abs(np.abs(i[m][:, 1:, :3]) - np.abs(it_t[m][:, 1:, :3])).mean(axis=(0, 2)).__truediv__(rat).mean()),
                "neutral_current_mae_a": float(np.abs(np.abs(i[m][:, 1:, 3]) - np.abs(it_t[m][:, 1:, 3])).mean() * 1e3)}
    out = Path(f"results/fourwire_prior_check_{split}.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
