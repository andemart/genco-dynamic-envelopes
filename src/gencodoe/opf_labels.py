"""Multi-scenario OPF labels and graph features for the GENCO-OPF layer (protocol amendment 2).

    python -m gencodoe.opf_labels --days 0 1 2 --workers 8      # one npz per day under data_out/opf_case33/labels

Features per interval and direction (node = bus): 7 statistics (mean, min, p02, p10, p90, p98, max) over the 200
design scenarios of net P and of Q, cap of the direction, PV flag, root flag, depth, direction flag (19).
Label: the multi-scenario OPF shared vector as a fraction of cap (zero sentinel -> zeros, flagged).
Training and validation days only; test days are never labelled here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import opf_case33 as oc
from . import opf_envelopes as oe

OUT = Path("data_out/opf_case33/labels")
STATS = (lambda x: x.mean(0), lambda x: x.min(0), lambda x: np.percentile(x, 2, axis=0),
         lambda x: np.percentile(x, 10, axis=0), lambda x: np.percentile(x, 90, axis=0),
         lambda x: np.percentile(x, 98, axis=0), lambda x: x.max(0))
N_FEAT = 19


def node_features(case: oc.Case33, p: np.ndarray, q: np.ndarray, direction: str) -> np.ndarray:
    n = case.fd.n_bus
    f = [s(p) for s in STATS] + [s(q) for s in STATS]
    cap = np.zeros(n)
    cap[case.unit_bus] = case.cap[direction]
    pv = np.zeros(n)
    pv[case.pv_bus] = 1.0
    root = (case.fd.parent < 0).astype(float)
    depth = np.zeros(n)
    for k in case.fd.order[1:] if hasattr(case.fd, "order") else range(1, n):
        depth[k] = depth[case.fd.parent[k]] + 1
    dirf = np.full(n, 1.0 if direction == "export" else -1.0)
    return np.column_stack(f + [cap, pv, root, depth, dirf]).astype(np.float32)


def label_day(day: int) -> Path:
    if day in oc.SPLIT["test"]:
        raise SystemExit("test days are never labelled for training")
    out = OUT / f"day{day:03d}.npz"
    if out.exists():
        return out
    from .opf_multi import MultiOPF, multi_envelope
    case = oc.build()
    sc = oc.Scenarios(case, 0)
    cust = oe.customers(case)
    m = MultiOPF()
    X, Y, meta = [], [], []
    for h in range(24):
        p, q, _, _ = oe.interval_draws(sc, day, h)
        for d in ("export", "import"):
            t = time.perf_counter()
            x, info = multi_envelope(m, case.fd, cust, p, q, d)
            X.append(node_features(case, p, q, d))
            Y.append((x / case.cap[d]).astype(np.float32))
            meta.append({"day": day, "hour": h, "direction": d, "wall_s": time.perf_counter() - t, **info})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".day{day:03d}.partial.npz"
    np.savez(tmp, X=np.array(X), Y=np.array(Y), meta=json.dumps(meta))
    os.replace(tmp, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.workers == 1:
        for d in a.days:
            print(label_day(d), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(label_day, a.days):
                print(r, flush=True)


if __name__ == "__main__":
    main()
