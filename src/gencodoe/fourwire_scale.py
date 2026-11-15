"""Four-wire GENCO across 25 ENWL networks (docs/fourwire_scale_protocol.md).

    python -m gencodoe.fourwire_scale identify [--networks 1 2 ...] [--processes k]

identify (Amendments 1, 2): per network, the first 500 training states (frozen generator order, capped
augmentation), OpenDSS
labels under the §3 failure rule, and the Thevenin root equivalent fitted on the converged ones, with
fit diagnostics. Writes data_out/fourwire_scale/ (networks, labels, Thevenin) and
results/fourwire_scale_identification.json (hashes, diagnostics, frozen E and Z_th).

    python -m gencodoe.fourwire_scale e1

e1 (§7): per network, the first 200 val states; OpenDSS under the failure rule vs the NumPy sweep with the
frozen Thevenin equivalent. Writes data_out/fourwire_scale/e1/ and results/fourwire_scale_e1.json.
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
from .fourwire_net import V_PN_NOM as V_PN_NOM_KV

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_out/fourwire_scale"
RESULTS = ROOT / "results"
SPLIT_INDEX = {"train": 0, "val": 1, "calib": 2, "test": 3}
AUGMENT = 0.5            # §2.5, passed explicitly; the generator default stays 0.0
N_IDENT = 500            # Amendment 1
OPENDSS = {"tolerance": 1e-6, "retry_algorithm": "newton"}   # §3
CAP_KVA = 880.0          # Amendment 2: 1.10 x 800 kVA, aggregate customer apparent power cap for the augmentation


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


def augmentation_report(st: dict) -> dict:
    """Amendment 2 report: share of augmented states (drawn f != 0) with gamma < 1 and with gamma = 0."""
    aug = st["flex_kw_drawn"] != 0
    g = st["gamma"][aug]
    return {"states": int(len(aug)), "augmented": int(aug.sum()),
            "share_gamma_below_1": float((g < 1).mean()) if aug.any() else 0.0,
            "share_gamma_0": float((g == 0).mean()) if aug.any() else 0.0}


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


def content_sha(arrays: dict) -> str:
    """SHA-256 over array names, dtypes, shapes and bytes: reproducible across runs (an .npz file hash is
    not, because the zip container stores timestamps)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


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
        st = customer_states(len(names), "train", n, seed(network, "train"), AUGMENT, CAP_KVA)
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
    rec["augmentation_cap"] = augmentation_report(st)
    out = OUT / "identification" / f"network_{network}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(step=st["step"], mix=st["mix"], flex_kw=st["flex_kw"], flex_kw_drawn=st["flex_kw_drawn"],
                  gamma=st["gamma"], groups=st["groups"], meters=st["meters"], p_kw_customer=st["p_kw"],
                  q_kvar_customer=st["q_kvar"], customer_names=np.array(names), connection_customer=idx,
                  connection_share=share, e=th.e, zth=th.zth, **lab)
    np.savez(out, **arrays)
    rec["labels_file"] = str(out.relative_to(ROOT))
    rec["labels_sha256"] = _sha(out)
    rec["labels_content_sha256"] = content_sha(arrays)
    rec["seconds"] = time.perf_counter() - t0
    return rec


N_E1 = 200               # §7 E1: first 200 val states per network
E1_V_TOL_PU, E1_I_TOL_REL = 1e-4, 1e-3


def frozen_thevenin(network: int):
    """The frozen Amendment 1 equivalent (identification run), checked against its recorded hash."""
    from .fourwire_net import Thevenin
    rec = {r["network"]: r for r in json.loads((RESULTS / "fourwire_scale_identification.json").read_text())["networks"]}
    f = ROOT / rec[network]["labels_file"]
    if _sha(f) != rec[network]["labels_sha256"]:
        raise AssertionError(f"identification file of network {network} differs from the frozen record")
    z = np.load(f)
    return Thevenin(e=z["e"], zth=z["zth"])


SWEEP_TOL, SWEEP_MAX_IT = 1e-8, 100     # §3 common sweep settings


def reference_sweep_state(net, th, s_kva: np.ndarray):
    """The NumPy sweep on one state (flat start): (V (n_bus, 4) kV, I (n_bus, 4) kA, iterations, converged).
    converged: the voltage update of the last performed iteration is below SWEEP_TOL within SWEEP_MAX_IT."""
    from .fourwire_net import sweep
    v, i, it = sweep(net, th, s_kva[None], iterations=SWEEP_MAX_IT, tol=SWEEP_TOL)
    conv = True
    if it == SWEEP_MAX_IT:                                  # no early exit: check the update of iteration 100
        v_prev, _, _ = sweep(net, th, s_kva[None], iterations=SWEEP_MAX_IT - 1, tol=SWEEP_TOL)
        conv = bool(np.abs(v - v_prev).max() / V_PN_NOM_KV < SWEEP_TOL)
    return v[0], i[0], it, conv


def e1_network(network: int, split: str = "val", n_states: int = N_E1, write: bool = True) -> dict:
    """E1 (§7): 200 val states, OpenDSS (failure rule) vs the NumPy sweep with the frozen Thevenin root.
    split/n_states/write exist only for smoke tests of this code on training states."""
    from .fourwire_labels import Extractor
    from .fourwire_net import load_fourwire, phase_neutral_pu
    from .limits import VMAX_PU, VMIN_PU
    t0 = time.perf_counter()
    folder = network_folder(network)
    if any(_sha(folder / k) != v for k, v in _identification(network)["network_files_sha256"].items()):
        raise AssertionError(f"network {network} files differ from the identification run")
    th = frozen_thevenin(network)
    ex = Extractor(folder, **OPENDSS)
    net = ex.net
    names, idx, share = cc.customers(net.customer_names)
    st = customer_states(len(names), split, n_states, seed(network, split), AUGMENT, CAP_KVA)
    p, q = connection_states(st, idx, share)
    lab = label_block(ex, p, q)
    s = p + 1j * q
    v_sw, i_sw, its, sw_conv = [], [], np.zeros(n_states, int), np.zeros(n_states, bool)
    for k in range(n_states):                                  # one state at a time: per-state iteration counts
        v, i, its[k], sw_conv[k] = reference_sweep_state(net, th, s[k])
        v_sw.append(v); i_sw.append(i)
    v_sw, i_sw = np.array(v_sw), np.array(i_sw)
    c = lab["converged"]                                       # §7: every state where OpenDSS converges
    vpn_sw = phase_neutral_pu(net, v_sw)
    vpn_dss = phase_neutral_pu(net, lab["V"])
    rating = net.rating_a[1:] / 1e3                                          # kA, root row excluded
    v_err = float(np.abs(vpn_sw - vpn_dss)[c].max()) if c.any() else float("nan")
    i_err = float((np.abs(i_sw - lab["I"])[:, 1:, :] / rating[None, :, None])[c].max()) if c.any() \
        else float("nan")
    # violation shares on OpenDSS-converged states; cable overload also under the rating variants
    vpn_c = vpn_dss[c]
    shares = {"undervoltage": float((vpn_c.min(1) < VMIN_PU).mean()), "overvoltage": float((vpn_c.max(1) > VMAX_PU).mean()),
              "transformer_overload": float(lab["flags"][c, 3].mean())}
    loading = {}
    for variant in cc.RATING_VARIANTS:
        if variant == "primary":
            r = rating
        else:
            vf = network_folder(network, variant)
            cc.convert(cc.network_dirs()[network], vf, variant)
            r = load_fourwire(vf).rating_a[1:] / 1e3
        loading[variant] = (np.abs(lab["I"][c, 1:, :3]) / r[None, :, None]).max((1, 2))
        shares[f"cable_overload_{variant}"] = float((loading[variant] > 1.0).mean())
    arrays = dict(step=st["step"], flex_kw=st["flex_kw"], flex_kw_drawn=st["flex_kw_drawn"], gamma=st["gamma"],
                  meters=st["meters"], p_kw_customer=st["p_kw"], q_kvar_customer=st["q_kvar"], V_sweep=v_sw,
                  I_sweep=i_sw, sweep_iterations=its, sweep_converged=sw_conv, **lab)
    out = OUT / "e1" / f"network_{network}.npz"
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
    return {"network": network, "states": n_states, "opendss_converged": int(c.sum()),
            "newton_rescued": int(lab["rescued"].sum()), "solver_indeterminate": int((~c).sum()),
            "sweep_nonconverged": int((~sw_conv).sum()),
            "opendss_converged_sweep_nonconverged": int((c & ~sw_conv).sum()), "compared": int(c.sum()),
            "max_vpn_err_pu": v_err, "max_branch_current_err_rel_rating": i_err,
            "pass": bool(c.any() and v_err <= E1_V_TOL_PU and i_err <= E1_I_TOL_REL),
            "sweep_iterations_median": float(np.median(its)), "sweep_iterations_p95": float(np.percentile(its, 95)),
            "opendss_line_loading_check": float(np.abs(loading["primary"] - lab["summary"][c, 2]).max()),
            "violation_shares": shares, "meter_reuse": meter_reuse(st), "augmentation_cap": augmentation_report(st),
            "file": str(out.relative_to(ROOT)) if write else None, "sha256": _sha(out) if write else None,
            "content_sha256": content_sha(arrays), "seconds": time.perf_counter() - t0}


def _identification(network: int) -> dict:
    d = json.loads((RESULTS / "fourwire_scale_identification.json").read_text())
    return {r["network"]: r for r in d["networks"]}[network]


def e1(networks: list[int], processes: int) -> Path:
    with Pool(processes) as pool:
        recs = pool.map(e1_network, networks, chunksize=1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    out = RESULTS / "fourwire_scale_e1.json"
    out.write_text(json.dumps({"protocol": "docs/fourwire_scale_protocol.md §7 E1, Amendments 1 and 2",
                               "code_commit": commit, "src_dirty": dirty, "criteria": {
                                   "max_vpn_err_pu": E1_V_TOL_PU, "max_branch_current_err_rel_rating": E1_I_TOL_REL},
                               "networks": recs}, indent=1))
    return out


def identify(networks: list[int], processes: int) -> Path:
    with Pool(processes) as pool:
        recs = pool.map(identify_network, networks, chunksize=1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    out = RESULTS / "fourwire_scale_identification.json"
    out.write_text(json.dumps({"protocol": "docs/fourwire_scale_protocol.md, Amendments 1 and 2",
                               "cap_kva": CAP_KVA, "code_commit": commit,
                               "src_dirty": dirty, "augment": AUGMENT, "opendss": OPENDSS,
                               "networks": recs}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identify", "e1"])
    ap.add_argument("--networks", type=int, nargs="*", default=list(range(1, 26)))
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    print((identify if a.cmd == "identify" else e1)(a.networks, a.processes))


if __name__ == "__main__":
    main()
