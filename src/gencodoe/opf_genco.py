"""The actual GENCO on case33bw (docs/genco_opf_protocol.md, amendment 3): gate G3b for the graphkit PF checkpoint,
frozen final-layer bus embeddings of the mean design scenario per interval, and the GENCO-conditioned envelope model.

    python -m gencodoe.opf_genco gate  --run mlruns/<exp>/<run>                      # results/genco_case33_gate.json
    python -m gencodoe.opf_genco embed --run mlruns/<exp>/<run> --days 0 ... 99      # data_out/opf_case33/emb/dayNNN.npz
    python -m gencodoe.opf_gnn fit --model gnn_genco --tag final                     # node features + embeddings

The backbone is never trained and never pooled (the PhaseGENCO recipe of the four-wire extension).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import feeder as fdr
from . import opf_case33 as oc
from .datakit_case33 import NAME

CONFIG = Path("configs/genco_pf_case33_lv_prior.yaml")
DATAKIT = Path("data_out/datakit_case33")
GATE_DIR = Path("data_out/datakit_case33_gate")
EMB = Path("data_out/opf_case33/emb")
EMB_WIDTH = 96
GATE = {"vm_mae": 0.002, "vm_p99": 0.01}                     # workplan G3b


def find_run(exp_name: str, log_dir: Path = Path("mlruns")) -> Path:
    """Newest mlflow run of the experiment that has a best-model export."""
    import yaml
    runs = []
    for meta in log_dir.glob("*/meta.yaml"):
        if yaml.safe_load(meta.read_text()).get("name") == exp_name:
            runs += [p.parents[2] for p in meta.parent.glob("*/artifacts/model/best_model_state_dict.pt")]
    if not runs:
        raise FileNotFoundError(f"no finished run of {exp_name} under {log_dir}")
    return max(runs, key=lambda r: r.stat().st_mtime)


def load_oracle(run: Path, device: str = "cpu"):
    import torch
    from .genco import GencoModel, GencoOracle, read_raw
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    model = GencoModel(CONFIG, run / "artifacts/model/best_model_state_dict.pt",
                       run / "artifacts/stats/normalizer_stats.pt", device=device, network=NAME)
    return GencoOracle(model, read_raw(DATAKIT / NAME / "raw", scenarios=[0]), batch_size=512)


def embed_states(oracle, fd: fdr.Feeder, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Final-layer bus embeddings (n, n_bus, 96) of states (p, q injections, pu, consumption negative)."""
    import torch
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    n = len(p)
    if n == 1:                                             # graphkit squeezes the gen output of a lone graph
        p, q = np.repeat(p, 2, 0), np.repeat(q, 2, 0)
    captured = []
    hook = oracle.model.model.mlp_bus.register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))
    try:
        oracle(fd, p, q)
    finally:
        hook.remove()
    n_chunks = -(-len(p) // oracle.batch_size)
    per = len(captured) // n_chunks                        # the bus head runs after every layer; keep the last
    h = torch.cat(captured[per - 1::per]).reshape(len(p), fd.n_bus, -1)
    if h.shape[-1] != EMB_WIDTH:
        raise AssertionError(f"embedding width {h.shape[-1]} != {EMB_WIDTH}")
    return h[:n].cpu().numpy().astype(np.float32)


def embed_interval(oracle, fd: fdr.Feeder, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(n_bus, 96): embedding of the per-bus mean of the design scenarios."""
    return embed_states(oracle, fd, p.mean(0, keepdims=True), q.mean(0, keepdims=True))[0]


def gate(run: Path, out: Path = Path("results/genco_case33_gate.json")) -> dict:
    """G3b on the gate states: GENCO voltages against the exact sweep."""
    import pandas as pd
    case = oc.build()
    bus = pd.read_parquet(GATE_DIR / NAME / "raw" / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd"])
    bus = bus.sort_values(["scenario", "bus"])
    n = bus.scenario.nunique()
    p = -bus.Pd.to_numpy().reshape(n, case.fd.n_bus)
    q = -bus.Qd.to_numpy().reshape(n, case.fd.n_bus)
    exact = fdr.sweep_pf(case.fd, p, q)
    oracle = load_oracle(run)
    pred = oracle(case.fd, p, q)
    dv = np.abs(pred.vm[:, 1:] - exact.vm[:, 1:])
    rated = np.isfinite(case.fd.rate)
    dl = np.abs(pred.loading[:, rated] - exact.loading[:, rated])
    near = (exact.vm[:, 1:].min(1) < oc.VMIN + 0.02) | (exact.vm[:, 1:].max(1) > oc.VMAX - 0.02)
    res = {"run": str(run), "n_states": int(n), "vm_mae": float(dv.mean()), "vm_p99": float(np.percentile(dv, 99)),
           "vm_max": float(dv.max()), "vm_p99_near_limits": float(np.percentile(dv[near], 99)) if near.any() else None,
           "n_near_limits": int(near.sum()), "loading_mae": float(dl.mean()), "gate": GATE}
    res["pass"] = bool(res["vm_mae"] < GATE["vm_mae"] and res["vm_p99"] < GATE["vm_p99"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def embed_days(run: Path, days) -> list[Path]:
    """Per label day: E (48, n_bus, 96) in the row order of opf_labels (hour, then export/import)."""
    from .opf_envelopes import interval_draws
    case = oc.build()
    sc = oc.Scenarios(case, 0)
    oracle = load_oracle(run)
    EMB.mkdir(parents=True, exist_ok=True)
    out = []
    for day in days:
        f = EMB / f"day{day:03d}.npz"
        if f.exists():
            out.append(f)
            continue
        E = []
        for h in range(24):
            p, q, _, _ = interval_draws(sc, day, h)
            e = embed_interval(oracle, case.fd, p, q)
            E += [e, e]                                   # the mean state does not depend on the direction
        tmp = EMB / f".day{day:03d}.partial.npz"
        np.savez(tmp, E=np.array(E, np.float32), run=str(run))
        os.replace(tmp, f)
        out.append(f)
    return out


class GencoConditioned:
    """Predictor for gnn_genco: node features + frozen GENCO embeddings of the interval's mean state."""

    def __init__(self, run_dir: Path, case: oc.Case33, genco_run: Path, device: str = "cpu"):
        from .opf_gnn import Predictor
        self.pred = Predictor(run_dir, case, device)
        self.oracle = load_oracle(genco_run, device)
        self.fd = case.fd

    def augment(self, feats: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        e = embed_interval(self.oracle, self.fd, p, q)[None]
        return np.concatenate([feats, np.repeat(e, len(feats), 0)], axis=-1)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return self.pred.predict(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "embed"])
    ap.add_argument("--run", type=Path, default=None, help="mlflow run dir; default: newest genco_pf_case33_lv_prior run")
    ap.add_argument("--days", type=int, nargs="+", default=list(range(100)))
    a = ap.parse_args()
    run = a.run or find_run("genco_pf_case33_lv_prior")
    if a.cmd == "gate":
        r = gate(run)
        print(json.dumps(r, indent=1))
        raise SystemExit(0 if r["pass"] else 5)
    if any(d >= 100 for d in a.days):
        raise SystemExit("embeddings are only precomputed for train/val days; test days are embedded at evaluation")
    for f in embed_days(run, a.days):
        print(f, flush=True)


if __name__ == "__main__":
    main()
