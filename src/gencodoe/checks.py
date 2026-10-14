"""Gate 1: data correctness and coverage of a datakit dataset.

Checks, on every state:
  * our exact sweep (feeder.sweep_pf) against datakit's PowerModels solution:
    Vm, Va and branch loading;
  * datakit branch flows pf/qf against S = V conj(Y V) rebuilt from datakit's own
    Yff/Yft and solved voltages (units, angle convention, branch orientation);
  * power balance: slack Pg = sum(Pd) + losses (sign convention: Pd > 0 is load);
  * coverage: net import/export, voltage range, loading, share of states with
    each kind of violation, per calendar-day split.

Usage: python -m gencodoe.checks --data data_out/datakit data_out/datakit_calib data_out/datakit_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import feeder as fdr
from .datakit_fluvius import day_split
from .enwl import load_enwl1
from .evaluate import VMAX, VMIN, _sweep_voltage, load_truth


def check_dataset(data_dir: Path, chunk: int = 2000) -> dict:
    net = load_enwl1()
    feeder = net.to_feeder()
    truth = load_truth(data_dir)
    bus, gen, branch = truth["frames"]
    n, n_bus = truth["vm"].shape
    p, q = -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva

    dvm = dva = dload = 0.0
    loading = np.empty_like(truth["vm"])
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        v = _sweep_voltage(feeder, p[sl], q[sl])
        dvm = max(dvm, float(np.abs(np.abs(v) - truth["vm"][sl]).max()))
        dva = max(dva, float(np.rad2deg(np.abs(np.angle(v) - truth["va"][sl])).max()))
        t = fdr.branch_flows_from_voltages(feeder, truth["vm"][sl], truth["va"][sl]).loading
        ac = fdr.branch_flows_from_voltages(feeder, np.abs(v), np.angle(v)).loading
        dload = max(dload, float(np.abs(t - ac).max()))
        loading[sl] = t

    # branch flows from datakit's own admittances
    nbr = len(net.from_bus)
    vc = truth["vm"] * np.exp(1j * truth["va"])
    f = branch.from_bus.to_numpy().reshape(n, nbr)
    t_ = branch.to_bus.to_numpy().reshape(n, nbr)
    if not ((f == net.from_bus).all() and (t_ == net.to_bus).all()):
        raise AssertionError("datakit branch order/orientation differs from the network")
    col = lambda c: branch[c].to_numpy().reshape(n, nbr)
    yff, yft = col("Yff_r") + 1j * col("Yff_i"), col("Yft_r") + 1j * col("Yft_i")
    vf, vt = vc[:, net.from_bus], vc[:, net.to_bus]
    sf = vf * np.conj(yff * vf + yft * vt)                      # pu on the case base
    s_dk = col("pf") + 1j * col("qf")                            # MW / MVAr
    base_mva = feeder.base_mva
    dflow = float(np.abs(sf * base_mva - s_dk).max())

    # power balance: slack generation = load + losses (losses >= 0)
    pg = gen.groupby("scenario").p_mw.sum().to_numpy()
    losses = pg - truth["pd"].sum(1)

    # coverage
    draws = truth["draws"]
    split = draws["split"].to_numpy() if "split" in draws else day_split(draws["step"].to_numpy())
    vmin, vmax = truth["vm"][:, 1:].min(1), truth["vm"][:, 1:].max(1)
    lmax = loading[:, 1:].max(1)
    net_mw = truth["pd"].sum(1)
    cov = {}
    for s in sorted(set(split)):
        m = split == s
        cov[s] = {"states": int(m.sum()),
                  "net_import_mw_p0_p50_p100": np.percentile(net_mw[m], [0, 50, 100]).round(4).tolist(),
                  "share_reverse_flow": float((net_mw[m] < 0).mean()),
                  "vm_min_p0_p50": np.percentile(vmin[m], [0, 50]).round(4).tolist(),
                  "vm_max_p50_p100": np.percentile(vmax[m], [50, 100]).round(4).tolist(),
                  "share_undervoltage": float((vmin[m] < VMIN).mean()),
                  "share_overvoltage": float((vmax[m] > VMAX).mean()),
                  "share_line_overload": float((lmax[m] > 1).mean()),
                  "share_transformer_overload": float((loading[m, 1] > 1).mean()),
                  "loading_max_p50_p100": np.percentile(lmax[m], [50, 100]).round(3).tolist()}
    return {"data": str(data_dir), "states": int(n), "buses": int(n_bus),
            "max_abs_dvm_sweep_vs_datakit_pu": dvm, "max_abs_dva_sweep_vs_datakit_deg": dva,
            "max_abs_dloading_sweep_vs_datakit": dload,
            "max_abs_branch_flow_rebuilt_vs_datakit_mva": dflow,
            "losses_mw_min_max": [float(losses.min()), float(losses.max())],
            "coverage_by_split": cov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data_out/eval/data_checks.json"))
    a = ap.parse_args()
    res = [check_dataset(d) for d in a.data]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
