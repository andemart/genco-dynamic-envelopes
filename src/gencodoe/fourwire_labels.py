"""Four-wire OpenDSS labels for the unbalanced GENCO extension (docs/unbalanced_genco_protocol.md).

    python -m gencodoe.fourwire_labels generate --split train --shard i --n-shards k
    python -m gencodoe.fourwire_labels integrity --split train
    python -m gencodoe.fourwire_labels thevenin            # fixed root equivalent from TRAIN labels only

Per state: complex node voltages (397 buses x a,b,c,n; kV, OpenDSS frame), complex branch currents into
each retained bus from its parent (kA; root row = total out of trbus), customer P/Q, the OpenDSS summary
(V_pn min/max, worst cable loading, transformer loading), convergence and the four violation labels.
Test shards go to their own directory and are never read by training, debugging or calibration code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .fourwire_net import CONDUCTORS, load_fourwire

OUT = Path("data_out/fourwire/labels")
SPLITS = {"train": (Path("data_out/datakit_v2"), "train.pt"), "val": (Path("data_out/datakit_v2"), "val.pt"),
          "calib": (Path("data_out/datakit_v2_calib"), None), "test": (Path("data_out/datakit_v2_test"), None)}
SCHEMA = {"version": 1, "V": "complex64 (S, 397, 4) kV a,b,c,n", "I": "complex64 (S, 397, 4) kA into bus from parent",
          "p_kw/q_kvar": "float32 (S, 200) customer net consumption, Loads.dss order",
          "summary": ["v_pn_min", "v_pn_max", "line_loading_max", "transformer_loading"],
          "flags": ["undervoltage", "overvoltage", "line_overload", "transformer_overload"]}
SCHEMA_HASH = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def split_states(split: str):
    """(scenario ids, pd (S, n_bus) MW, qd) for one split; test is only loaded by the generator."""
    import torch

    from .evaluate import load_truth
    data, sf = SPLITS[split]
    t = load_truth(data)
    keep = np.ones(len(t["scenario"]), bool)
    if sf:
        keep = np.isin(t["scenario"], torch.load(data / "splits" / sf, weights_only=False).numpy())
    return t["scenario"][keep], t["pd"][keep], t["qd"][keep]


class Extractor:
    def __init__(self):
        from .unbalanced import UnbalancedENWL1
        self.net = load_fourwire()
        self.ub = UnbalancedENWL1()
        if self.ub.load_names != self.net.customer_names:
            raise AssertionError("customer order differs between the four-wire net and OpenDSS")
        names = [n.lower() for n in self.ub.dss.Circuit.AllNodeNames()]
        pos = {n: i for i, n in enumerate(names)}
        self.node_idx = np.array([pos[n] for n in self.net.node_names()])   # complete mapping or KeyError
        self.children_of_root = np.flatnonzero(self.net.parent == 0)

    def state(self, p_kw, q_kvar) -> dict:
        dss = self.ub.dss
        r = self.ub.solve(p_kw, q_kvar)
        v = np.array(dss.Circuit.AllBusVolts())
        v = (v[0::2] + 1j * v[1::2])[self.node_idx].reshape(self.net.n_bus, 4) / 1e3
        cur = np.zeros((self.net.n_bus, 4), complex)
        for i in range(1, self.net.n_bus):
            dss.Lines.Name(self.net.first_line[i])
            c = np.array(dss.CktElement.Currents())
            c = c[0::2] + 1j * c[1::2]                                       # terminal 1: 0..3, terminal 2: 4..7
            cur[i] = c[:4] if self.net.first_line_parent_terminal[i] == 1 else c[4:8]
        cur /= 1e3
        cur[0] = cur[self.children_of_root].sum(0)
        return {"V": v, "I": cur, **{k: r[k] for k in ("v_pn_min", "v_pn_max", "line_loading_max",
                                                        "transformer_loading", "converged", "undervoltage",
                                                        "overvoltage", "line_overload", "transformer_overload")}}


def generate(split: str, shard: int, n_shards: int) -> Path:
    from .enwl import load_enwl1
    out = OUT / split / f"shard_{shard:03d}_of_{n_shards:03d}.npz"
    if out.exists():
        return out
    sc, pd_, qd = split_states(split)
    sel = np.arange(len(sc))[shard::n_shards]
    bal = load_enwl1()
    ex = Extractor()
    if [bal.bus_names[i] for i in bal.customer_bus] != [ex.net.bus_names[i] for i in ex.net.customer_bus]:
        raise AssertionError("customer bus alignment")
    n = len(sel)
    V = np.zeros((n, ex.net.n_bus, 4), np.complex64)
    I = np.zeros_like(V)
    P = (pd_[sel][:, bal.customer_bus] * 1e3).astype(np.float32)
    Q = (qd[sel][:, bal.customer_bus] * 1e3).astype(np.float32)
    summ = np.zeros((n, 4), np.float32)
    flags = np.zeros((n, 4), bool)
    conv = np.zeros(n, bool)
    max_cast_err = 0.0
    t0 = time.perf_counter()
    for k in range(n):
        s = ex.state(P[k].astype(float), Q[k].astype(float))
        V[k], I[k] = s["V"], s["I"]
        max_cast_err = max(max_cast_err, float(np.abs(V[k] - s["V"]).max() / 0.24018))
        summ[k] = [s["v_pn_min"], s["v_pn_max"], s["line_loading_max"], s["transformer_loading"]]
        flags[k] = [s["undervoltage"], s["overvoltage"], s["line_overload"], s["transformer_overload"]]
        conv[k] = s["converged"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name.replace(".npz", ".partial.npz"))
    np.savez(tmp, scenario=sc[sel], V=V, I=I, p_kw=P, q_kvar=Q, summary=summ, flags=flags, converged=conv,
             schema_hash=SCHEMA_HASH, engine_health=json.dumps(ex.ub.health()), split=split,
             float32_cast_max_err_pu=max_cast_err, seconds=time.perf_counter() - t0,
             code_commit=os.environ.get("GENCODOE_COMMIT", ""))
    os.replace(tmp, out)
    return out


def load_split(split: str, allow_test: bool = False) -> dict:
    """Concatenate a split's shards. Test can only be loaded with allow_test (frozen evaluation)."""
    if split == "test" and not allow_test:
        raise PermissionError("test labels are isolated until the evaluation is frozen")
    files = sorted((OUT / split).glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(split)
    parts = [np.load(f) for f in files]
    d = {k: np.concatenate([z[k] for z in parts]) for k in ("scenario", "V", "I", "p_kw", "q_kvar", "summary",
                                                             "flags", "converged")}
    o = np.argsort(d["scenario"])
    return {k: v[o] for k, v in d.items()}


def semantic_check(d: dict, net, tol_kcl_ka: float = 2e-4) -> dict:
    """Physics/semantic checks on converged states (integrity only; no model metric)."""
    from .fourwire_net import customer_currents, phase_neutral_pu
    from .limits import VMAX_PU, VMIN_PU
    c = d["converged"]
    V, I = d["V"][c].astype(complex), d["I"][c].astype(complex)
    s = (d["p_kw"][c] + 1j * d["q_kvar"][c]).astype(complex)
    inj = customer_currents(net, V, s)
    # KCL at each retained bus: current in = customer injection + currents into children
    kcl = I - inj
    for i in net.order[1:]:
        kcl[:, net.parent[i]] -= I[:, i]
    kcl[:, 0] = 0.0                                                        # root row is defined as the total
    vpn = phase_neutral_pu(net, V)
    # worst cable loading recomputed from stored branch currents: a merged chain carries one current and its
    # rating is the chain minimum, so the max over retained branches equals OpenDSS's worst line
    line_load = (np.abs(I[:, 1:, :3]) / (net.rating_a[1:, None] / 1e3)).max((1, 2))
    rep = {"states": int(len(c)), "converged": int(c.sum()),
           "scenarios_unique": bool(len(np.unique(d["scenario"])) == len(d["scenario"])),
           "line_loading_recomputed_max_abs_diff": float(np.abs(line_load - d["summary"][c, 2]).max()) if c.any() else 0.0,
           "cable_flag_consistent": bool(((d["summary"][:, 2] > 1.0) == d["flags"][:, 2]).all()),
           "transformer_flag_consistent": bool(((d["summary"][:, 3] > 1.0) == d["flags"][:, 3]).all()),
           "kcl_max_ka": float(np.abs(kcl).max()) if c.any() else 0.0,
           "root_neutral_max_v": float(np.abs(V[:, 0, 3]).max() * 1e3) if c.any() else 0.0,
           "vpn_min_recomputed_max_abs_diff": float(np.abs(vpn.min(1) - d["summary"][c, 0]).max()) if c.any() else 0.0,
           "vpn_max_recomputed_max_abs_diff": float(np.abs(vpn.max(1) - d["summary"][c, 1]).max()) if c.any() else 0.0,
           "undervoltage_flag_consistent": bool(((vpn.min(1) < VMIN_PU) == d["flags"][c, 0]).all()),
           "overvoltage_flag_consistent": bool(((vpn.max(1) > VMAX_PU) == d["flags"][c, 1]).all()),
           "finite": bool(np.isfinite(V).all() and np.isfinite(I).all())}
    rep["ok"] = (rep["kcl_max_ka"] < tol_kcl_ka and rep["root_neutral_max_v"] < 1.0 and rep["finite"]
                 and rep["vpn_min_recomputed_max_abs_diff"] < 1e-4 and rep["vpn_max_recomputed_max_abs_diff"] < 1e-4
                 and rep["undervoltage_flag_consistent"] and rep["overvoltage_flag_consistent"]
                 and rep["scenarios_unique"] and rep["cable_flag_consistent"] and rep["transformer_flag_consistent"]
                 and rep["line_loading_recomputed_max_abs_diff"] < 1e-3)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "integrity", "thevenin"])
    ap.add_argument("--split", choices=list(SPLITS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "generate":
        print(generate(a.split, a.shard, a.n_shards))
    elif a.cmd == "integrity":
        net = load_fourwire()
        files = sorted((OUT / a.split).glob("shard_*.npz"))
        rep = {"split": a.split, "files": {}, "schema_ok": True, "health": []}
        for f in files:
            z = np.load(f)
            rep["schema_ok"] &= str(z["schema_hash"]) == SCHEMA_HASH
            rep["health"].append(json.loads(str(z["engine_health"])))
            rep["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        # test labels may be checked semantically only after the frozen evaluation (amendment 2)
        rep["semantic"] = semantic_check(load_split(a.split, allow_test=a.split == "test"), net)
        out = Path(f"results/fourwire_label_integrity_{a.split}.json")
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=1))
    else:
        from .fourwire_net import fit_thevenin
        d = load_split("train")
        c = d["converged"]
        th = fit_thevenin(d["V"][c, 0, :3].astype(complex), d["I"][c, 0, :3].astype(complex))
        np.savez("data_out/fourwire/thevenin.npz", e=th.e, zth=th.zth)
        print("E (kV)", np.round(th.e, 5), "\nZth (ohm)\n", np.round(th.zth, 6))


if __name__ == "__main__":
    main()
