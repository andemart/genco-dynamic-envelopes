"""Stage G: end-to-end power-flow throughput for the DOE search on ENWL1.

Times what one envelope round costs per method, for batches of injection rows
(scenarios x candidate levels):
  ac_sweep        exact backward/forward sweep, numpy (CPU)
  fixed_point_1   one sweep iteration (the linear prior alone)
  genco           GencoOracle end to end: injections -> graphs (conversion),
                  normalizer + masking, model, KCL branch flows (post-processing)
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
    a = ap.parse_args()

    feeder = load_enwl1().to_feeder()
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
            methods["genco"] = lambda: oracle(feeder, p, q)
        for name, fn in methods.items():
            cold, warm = _time(fn, a.repeats)
            r = {"method": name, "rows": n, "cold_s": cold, "warm_s": warm, "warm_ms_per_row": 1e3 * warm / n,
                 "device": a.device if name == "genco" else "cpu"}
            if name == "genco":
                r["model_load_s"] = load_s
            rows.append(r)
            print(f"{name:14s} rows={n:6d} cold {cold:8.3f} s  warm {warm:8.3f} s  ({r['warm_ms_per_row']:.3f} ms/row)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"machine": platform.platform(), "processor": platform.processor(),
                                 "run": str(a.run), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
