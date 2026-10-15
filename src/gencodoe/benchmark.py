"""Stage G: end-to-end power-flow throughput for the DOE search on ENWL1.

Times what one envelope round costs per method, for batches of injection rows
(scenarios x candidate levels):
  ac_sweep        exact backward/forward sweep, numpy (CPU)
  fixed_point_1   one sweep iteration (the linear prior alone)
  genco           GencoOracle end to end: injections written into a cached
                  normalized batch, model, copy back, KCL branch flows
  genco_model     model forward only, inputs already on the device (synchronised)
  pandapower_nr   pandapower Newton-Raphson per state: a general-purpose
                  balanced AC solver (timed on --generic-rows rows)
  opendss_unbal   OpenDSS snapshot of the four-wire model per state: the
                  general-purpose unbalanced solver (timed on --generic-rows rows)
Cold = first call in a fresh process, including model load; warm = median of
repeated calls. Rows come from real held-out states plus envelope offsets.

Usage:
  python -m gencodoe.benchmark --run mlruns/<exp>/<run> --config configs/... --device mps \
      --batch 256 1024 4096 --out data_out/eval/benchmark_mps.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import feeder as fdr
from .enwl import load_enwl1
from .evaluate import _sweep_voltage, load_truth


def _time(fn, repeats: int):
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        warm.append(time.perf_counter() - t0)
    return cold, float(np.median(warm))


def pandapower_net(net):
    """The reduced balanced ENWL1 network as a pandapower net (lines on the 0.416 kV base)."""
    import pandapower as pp
    from .enwl import KV_LV
    pn = pp.create_empty_network(sn_mva=1.0)
    for i in range(net.n_bus):
        pp.create_bus(pn, vn_kv=KV_LV, index=i)
    pp.create_ext_grid(pn, 0, vm_pu=1.0)
    for f, t, r, x, s in zip(net.from_bus, net.to_bus, net.r_ohm, net.x_ohm, net.rate_mva):
        pp.create_line_from_parameters(pn, int(f), int(t), length_km=1.0, r_ohm_per_km=max(r, 1e-6),
                                       x_ohm_per_km=max(x, 1e-6), c_nf_per_km=0.0,
                                       max_i_ka=s / (np.sqrt(3) * KV_LV))
    for b in range(1, net.n_bus):
        pp.create_load(pn, b, p_mw=0.0, q_mvar=0.0)
    return pn


def time_generic(net, feeder, p, q, rows: int) -> dict:
    """ms/state of pandapower NR and OpenDSS (unbalanced) on the first `rows` states."""
    import pandapower as pp
    out = {}
    pn = pandapower_net(net)
    lb = pn.load.bus.to_numpy()
    t0 = time.perf_counter()
    vm_err = 0.0
    ref = fdr.sweep_pf(feeder, p[:rows], q[:rows]).vm
    for i in range(rows):
        pn.load["p_mw"] = -p[i, lb] * feeder.base_mva
        pn.load["q_mvar"] = -q[i, lb] * feeder.base_mva
        pp.runpp(pn, algorithm="nr", init="dc" if i == 0 else "results", numba=True)
        vm_err = max(vm_err, float(np.abs(pn.res_bus.vm_pu.to_numpy() - ref[i]).max()))
    out["pandapower_nr"] = {"ms_per_row": 1e3 * (time.perf_counter() - t0) / rows, "max_dvm_vs_sweep": vm_err}
    try:
        from .unbalanced import UnbalancedENWL1
        u = UnbalancedENWL1()
        pc, qc = -p[:rows, net.customer_bus] * 1e3, -q[:rows, net.customer_bus] * 1e3
        t0 = time.perf_counter()
        for i in range(rows):
            u.solve(pc[i], qc[i])
        out["opendss_unbal"] = {"ms_per_row": 1e3 * (time.perf_counter() - t0) / rows}
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data_out/datakit_v2_calib"))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--model-batch", type=int, default=1024, help="graphs per forward pass")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generic-rows", type=int, default=0, help="states for pandapower/OpenDSS timing (0 = skip)")
    ap.add_argument("--tf32", action="store_true")
    a = ap.parse_args()
    if a.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    net = load_enwl1()
    feeder = net.to_feeder()
    truth = load_truth(a.data, range(max(a.batch)))
    p_all, q_all = -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva
    rows = []
    oracle = None
    if a.run:
        t0 = time.perf_counter()
        from .genco import GencoModel, GencoOracle, read_raw
        model = GencoModel(a.config, a.run / "artifacts/model/best_model_state_dict.pt",
                           a.run / "artifacts/stats/normalizer_stats.pt", device=a.device)
        oracle = GencoOracle(model, read_raw(a.data / "enwl1_fluvius" / "raw", scenarios=[0]), a.model_batch)
        load_s = time.perf_counter() - t0
    for n in a.batch:
        reps = int(np.ceil(n / len(p_all)))
        p, q = np.tile(p_all, (reps, 1))[:n], np.tile(q_all, (reps, 1))[:n]
        methods = {"ac_sweep": lambda: fdr.sweep_pf(feeder, p, q),
                   "fixed_point_1": lambda: fdr.branch_flows_kcl(
                       feeder, *(lambda v: (np.abs(v), np.angle(v)))(_sweep_voltage(feeder, p, q, max_iter=1)), p, q)}
        if oracle is not None:
            import torch
            methods["genco"] = lambda: oracle(feeder, p, q)
            mb = min(n, a.model_batch)
            model.prepare_template(oracle.template_graph, mb)

            def model_only():
                for i in range(0, n, mb):
                    model.predict_injections(-p[i:i + mb], -q[i:i + mb], sync=False)
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                elif model.device.type == "mps":
                    torch.mps.synchronize()
            methods["genco_model"] = model_only
        for name, fn in methods.items():
            cold, warm = _time(fn, a.repeats)
            r = {"method": name, "rows": n, "cold_s": cold, "warm_s": warm, "warm_ms_per_row": 1e3 * warm / n,
                 "device": a.device if name.startswith("genco") else "cpu"}
            if name == "genco":
                r["model_load_s"] = load_s
            rows.append(r)
            print(f"{name:14s} rows={n:6d} cold {cold:8.3f} s  warm {warm:8.3f} s  ({r['warm_ms_per_row']:.3f} ms/row)")
    generic = time_generic(net, feeder, p_all, q_all, a.generic_rows) if a.generic_rows else {}
    for k, v in generic.items():
        print(f"{k:14s} {v}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"machine": platform.platform(), "processor": platform.processor(),
                                 "run": str(a.run), "tf32": a.tf32, "model_batch": a.model_batch,
                                 "rows": rows, "generic": generic}, indent=1))


if __name__ == "__main__":
    main()
