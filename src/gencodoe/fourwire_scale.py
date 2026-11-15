"""Four-wire GENCO across 25 ENWL networks (docs/fourwire_scale_protocol.md).

    python -m gencodoe.fourwire_scale identify [--networks 1 2 ...] [--processes k]

identify (Amendment 1): per network, the first 500 training states (frozen generator order), OpenDSS
labels under the §3 failure rule, and the Thevenin root equivalent fitted on the converged ones, with
fit diagnostics. Writes data_out/fourwire_scale/ (networks, labels, Thevenin) and
results/fourwire_scale_identification.json (hashes, diagnostics, frozen E and Z_th).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import csiro_convert as cc
from .datakit_fluvius import customer_states

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_out/fourwire_scale"
RESULTS = ROOT / "results"
SPLIT_INDEX = {"train": 0, "val": 1, "calib": 2, "test": 3}
AUGMENT = 0.5            # §2.5, passed explicitly; the generator default stays 0.0
N_IDENT = 500            # Amendment 1
OPENDSS = {"tolerance": 1e-6, "retry_algorithm": "newton"}   # §3


def seed(network: int, split: str) -> int:
    return 1000 * network + SPLIT_INDEX[split]


def network_folder(network: int, variant: str = "primary") -> Path:
    return OUT / "networks" / variant / f"network_{network}"


def connection_states(st: dict, idx: np.ndarray, share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer p/q (S, n_cust) -> per phase-connection (S, n_conn) in Loads.dss order (§2.2 item 5)."""
    return st["p_kw"][:, idx] * share, st["q_kvar"][:, idx] * share


def meter_reuse(st: dict) -> dict:
    """§2.5 report per technology group: share of repeated meter assignments and unique-meter fraction,
    pooled over states (within a state, a meter repeated k times counts k - 1 repeats)."""
    out = {}
    key = st["groups"].astype(int) @ np.array([4, 2, 1])               # (S, n_cust) group code hp/ev/pv
    for code in range(8):
        n_assign = n_unique = 0
        for s in range(len(key)):
            m = st["meters"][s][key[s] == code]
            n_assign += m.size
            n_unique += np.unique(m).size
        if n_assign:
            out[f"hp{code >> 2 & 1}_ev{code >> 1 & 1}_pv{code & 1}"] = {
                "assignments": int(n_assign), "repeated_share": 1 - n_unique / n_assign,
                "unique_fraction": n_unique / n_assign}
    return out


def fit_diagnostics(v_root: np.ndarray, i_root: np.ndarray) -> dict:
    """Design matrix A = [1, -I_a, -I_b, -I_c] (kA) of fourwire_net.fit_thevenin: rank, singular values,
    cond2(A) and the condition number after scaling each column to unit 2-norm (scale-independent)."""
    a = np.hstack([np.ones((len(i_root), 1)), -i_root])
    sv = np.linalg.svd(a, compute_uv=False)
    sv_n = np.linalg.svd(a / np.linalg.norm(a, axis=0), compute_uv=False)
    return {"n_states": int(len(a)), "rank": int(np.linalg.matrix_rank(a)), "singular_values": sv.tolist(),
            "cond2": float(sv[0] / sv[-1]), "cond2_column_normalized": float(sv_n[0] / sv_n[-1])}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_block(ex, p_kw: np.ndarray, q_kvar: np.ndarray) -> dict:
    n = len(p_kw)
    d = {"V": np.zeros((n, ex.net.n_bus, 4), complex), "I": np.zeros((n, ex.net.n_bus, 4), complex),
         "summary": np.zeros((n, 4)), "flags": np.zeros((n, 4), bool), "converged": np.zeros(n, bool),
         "retried": np.zeros(n, bool), "rescued": np.zeros(n, bool)}
    for k in range(n):
        s = ex.state(p_kw[k], q_kvar[k])
        d["V"][k], d["I"][k] = s["V"], s["I"]
        d["summary"][k] = [s["v_pn_min"], s["v_pn_max"], s["line_loading_max"], s["transformer_loading"]]
        d["flags"][k] = [s["undervoltage"], s["overvoltage"], s["line_overload"], s["transformer_overload"]]
        d["converged"][k], d["retried"][k], d["rescued"][k] = s["converged"], ex.ub.last_retried, ex.ub.last_rescued
    return d


def identify_network(network: int) -> dict:
    from .fourwire_labels import Extractor
    from .fourwire_net import fit_thevenin
    t0 = time.perf_counter()
    folder = network_folder(network)
    counts = cc.convert(cc.network_dirs()[network], folder)
    ex = Extractor(folder, **OPENDSS)
    names, idx, share = cc.customers(ex.net.customer_names)
    rec = {"network": network, "conversion": counts, "seed": seed(network, "train"),
           "network_files_sha256": {f.name: _sha(f) for f in sorted(folder.glob("*.dss"))}, "blocks": []}
    st = {}
    for block in range(4):                                      # Amendment 1 rank rule: 500 .. 2,000 states
        n = N_IDENT * (block + 1)
        st = customer_states(len(names), "train", n, seed(network, "train"), AUGMENT)
        p, q = connection_states(st, idx, share)
        lab = label_block(ex, p[n - N_IDENT:], q[n - N_IDENT:])
        prev = rec.get("_labels")
        lab = lab if prev is None else {k: np.concatenate([prev[k], lab[k]]) for k in lab}
        rec["_labels"] = lab
        c = lab["converged"]
        diag = fit_diagnostics(lab["V"][c, 0, :3], lab["I"][c, 0, :3])
        rec["blocks"].append(diag)
        if diag["rank"] == 4:
            break
    lab = rec.pop("_labels")
    c = lab["converged"]
    rec["fit"] = rec["blocks"][-1]
    rec["frozen"] = rec["fit"]["rank"] == 4
    th = fit_thevenin(lab["V"][c, 0, :3], lab["I"][c, 0, :3])
    rec["thevenin"] = {"e_kv": [[z.real, z.imag] for z in th.e],
                       "zth_ohm": [[[z.real, z.imag] for z in row] for row in th.zth]}
    rec["opendss"] = {"states": int(len(c)), "converged": int(c.sum()), "newton_rescued": int(lab["rescued"].sum()),
                      "solver_indeterminate": int((~c).sum()), "retried": int(lab["retried"].sum())}
    rec["meter_reuse"] = meter_reuse(st)
    out = OUT / "identification" / f"network_{network}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, step=st["step"], mix=st["mix"], flex_kw=st["flex_kw"], groups=st["groups"],
             meters=st["meters"], p_kw_customer=st["p_kw"], q_kvar_customer=st["q_kvar"],
             customer_names=np.array(names), connection_customer=idx, connection_share=share,
             e=th.e, zth=th.zth, **lab)
    rec["labels_file"] = str(out.relative_to(ROOT))
    rec["labels_sha256"] = _sha(out)
    rec["seconds"] = time.perf_counter() - t0
    return rec


def identify(networks: list[int], processes: int) -> Path:
    with Pool(processes) as pool:
        recs = pool.map(identify_network, networks, chunksize=1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    out = RESULTS / "fourwire_scale_identification.json"
    out.write_text(json.dumps({"protocol": "docs/fourwire_scale_protocol.md, Amendment 1", "code_commit": commit,
                               "src_dirty": dirty, "augment": AUGMENT, "opendss": OPENDSS,
                               "networks": recs}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identify"])
    ap.add_argument("--networks", type=int, nargs="*", default=list(range(1, 26)))
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    print(identify(a.networks, a.processes))


if __name__ == "__main__":
    main()
