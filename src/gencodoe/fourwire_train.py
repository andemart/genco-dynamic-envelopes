"""Train PhaseGraph / PhaseGENCO (docs/unbalanced_genco_protocol.md).

    python -m gencodoe.fourwire_train embeddings --split train --device cuda   # PhaseGENCO inputs (frozen GENCO)
    python -m gencodoe.fourwire_train fit --model phasegraph --epochs 3 --tag smoke --device cuda
    python -m gencodoe.fourwire_train fit --model phasegenco --epochs 60 --tag final --device cuda

Training reads train and val only. Calibration and test are evaluated elsewhere, after freezing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import fourwire_model as fm
from .fourwire_labels import load_split, split_states
from .fourwire_net import V_PN_NOM, Thevenin, load_fourwire

OUT = Path("data_out/fourwire")
EMB = OUT / "embeddings"
VMIN, VMAX = 0.9001667898951713, 1.0533866690262643
CONFIG = {"hidden": 64, "layers": 8, "heads": 4, "lr": 1e-3, "weight_decay": 1e-4, "batch": 32, "max_epochs": 60,
          "patience": 8, "seed": 0, "loss_weights": {"v": 1.0, "vpn": 1.0, "cur": 0.5, "kvl": 0.1}, "near_limit_weight": 3.0}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def thevenin() -> Thevenin:
    z = np.load(OUT / "thevenin.npz")
    return Thevenin(e=z["e"], zth=z["zth"])


# ------------------------------------------------------------------------------- PhaseGENCO embeddings
def bus_alignment():
    """Four-wire bus i -> balanced bus index; asserts a complete one-to-one map (sourcebus excluded)."""
    from .enwl import load_enwl1
    bal, net = load_enwl1(), load_fourwire()
    pos = {b: i for i, b in enumerate(bal.bus_names)}
    idx = np.array([pos[b] for b in net.bus_names])
    if len(set(idx)) != net.n_bus or set(bal.bus_names) - {net.bus_names[i] for i in range(net.n_bus)} != {"sourcebus"}:
        raise AssertionError("balanced <-> four-wire bus alignment is not one-to-one")
    return idx


def embeddings(split: str, device: str):
    """Node-aligned final-layer bus embeddings of the frozen balanced GENCO, float16, (S, 397, 96).
    Never pooled. The backbone runs under no_grad and is never trained."""
    from .enwl import load_enwl1
    from .envpred3 import RUN_HYDRA, RUN_LOCAL, load_genco
    run = RUN_LOCAL if RUN_LOCAL.exists() else RUN_HYDRA
    oracle, _ = load_genco(device, run)
    for p_ in oracle.model.model.parameters():
        p_.requires_grad_(False)
    feeder = load_enwl1().to_feeder()
    align = bus_alignment()
    sc, pd_, qd = split_states(split)
    out = np.zeros((len(sc), len(align), fm.EMB_WIDTH), np.float16)
    captured = []
    hook = oracle.model.model.mlp_bus.register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))
    try:
        with torch.no_grad():
            for k in range(0, len(sc), 512):
                captured.clear()
                p, q = -pd_[k:k + 512] / feeder.base_mva, -qd[k:k + 512] / feeder.base_mva
                oracle(feeder, p, q)
                n_chunks = -(-len(p) // oracle.batch_size)
                per = len(captured) // n_chunks
                h = torch.cat(captured[per - 1::per]).reshape(len(p), feeder.n_bus, -1)
                if h.shape[-1] != fm.EMB_WIDTH:
                    raise AssertionError(f"embedding width {h.shape[-1]}")
                out[k:k + len(p)] = h[:, torch.as_tensor(align, device=h.device)].cpu().numpy().astype(np.float16)
    finally:
        hook.remove()
    EMB.mkdir(parents=True, exist_ok=True)
    f = EMB / f"{split}.npz"
    tmp = EMB / f".{split}.partial.npz"
    np.savez(tmp, scenario=sc, emb=out, align=align)
    os.replace(tmp, f)
    return f


# ------------------------------------------------------------------------------- data
def prepare(split: str, st: fm.Static, with_emb: bool, allow_test: bool = False) -> dict:
    d = load_split(split, allow_test=allow_test)
    v1 = fm.prior_u1(st, d["p_kw"].astype(float), d["q_kvar"].astype(float))
    x = fm.node_features(st, d["p_kw"], d["q_kvar"], v1)
    out = {"scenario": d["scenario"], "x": x, "v1": v1.astype(np.complex64), "V": d["V"], "I": d["I"],
           "s": (d["p_kw"] + 1j * d["q_kvar"]).astype(np.complex64), "conv": d["converged"], "summary": d["summary"],
           "flags": d["flags"]}
    sm = d["summary"]
    near = (np.abs(sm[:, 0] - VMIN) < 0.02) | (np.abs(sm[:, 1] - VMAX) < 0.02) | ((sm[:, 2] > 0.9) & (sm[:, 2] < 1.1)) \
        | ((sm[:, 3] > 0.9) & (sm[:, 3] < 1.1))
    out["near"] = near
    if with_emb:
        z = np.load(EMB / f"{split}.npz")
        if not np.array_equal(z["scenario"], d["scenario"]):
            raise AssertionError("embedding / label state order differs")
        if not np.array_equal(z["align"], bus_alignment()):
            raise AssertionError("bus alignment changed")
        if z["emb"].shape[1:] != (st.net.n_bus, fm.EMB_WIDTH):
            raise AssertionError("embedding must be per bus (no pooled representation)")
        out["emb"] = z["emb"]
    return out


def residual_scale(tr: dict) -> np.ndarray:
    r = ((tr["V"] - tr["v1"]) / V_PN_NOM)[tr["conv"]]
    return np.concatenate([r.real.reshape(-1, 4).std(0), r.imag.reshape(-1, 4).std(0)]).astype(np.float32)


# ------------------------------------------------------------------------------- training
def run_fit(model_kind: str, epochs: int, tag: str, device: str, max_train: int | None = None):
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    fm.register()
    net = load_fourwire()
    st = fm.Static(net, thevenin())
    with_emb = model_kind == "phasegenco"
    tr, va = prepare("train", st, with_emb), prepare("val", st, with_emb)
    if max_train:
        tr = {k: v[:max_train] for k, v in tr.items()}
    xs = fm.Scaler(tr["x"])                                        # training statistics only
    es = fm.Scaler(st.edge_raw)
    rs = residual_scale(tr)
    vpn_tr = fm.Physics(st, torch.device("cpu"))
    with torch.no_grad():
        d_vpn = (vpn_tr.vpn(torch.as_tensor(tr["V"][tr["conv"]][:2000])) -
                 vpn_tr.vpn(torch.as_tensor(tr["v1"][tr["conv"]][:2000])))
    vpn_scale = float(d_vpn.std())
    drop = (tr["V"][tr["conv"]][:2000][:, net.parent[1:]] - tr["V"][tr["conv"]][:2000][:, 1:]) / V_PN_NOM
    kvl_scale = float(np.abs(drop).std())
    node_in = fm.N_NODE_BASE + (fm.EMB_WIDTH if with_emb else 0)
    model = fm.GNS_heterogeneous_4wire(node_in, hidden=CONFIG["hidden"], layers=CONFIG["layers"],
                                       heads=CONFIG["heads"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    batcher = fm.Batcher(st, es(st.edge_raw), dev)
    phys = fm.Physics(st, dev)
    rs_t = torch.as_tensor(rs, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    W = CONFIG["loss_weights"]

    def feats(d, idx):
        x = xs(d["x"][idx])
        if with_emb:
            x = np.concatenate([x, d["emb"][idx].astype(np.float32)], axis=2)
        return torch.as_tensor(x, device=dev)

    def forward(d, idx):
        b = len(idx)
        x = feats(d, idx).reshape(b * net.n_bus, -1)
        ei, ea = batcher.graph(b)
        out = model(x, ei, ea, b).view(b, net.n_bus, 8) * rs_t
        v1 = torch.as_tensor(d["v1"][idx], device=dev)
        return v1 + torch.complex(out[..., :4], out[..., 4:]) * V_PN_NOM

    def loss_fn(d, idx, v):
        vt = torch.as_tensor(d["V"][idx], device=dev)
        s = torch.as_tensor(d["s"][idx], device=dev)
        it = torch.as_tensor(d["I"][idx], device=dev)
        conv = torch.as_tensor(d["conv"][idx], device=dev).float()
        w = conv * torch.as_tensor(np.where(d["near"][idx], CONFIG["near_limit_weight"], 1.0), device=dev).float()
        dv = (v - vt) / V_PN_NOM
        lv = ((dv.real / rs_t[:4]) ** 2 + (dv.imag / rs_t[4:]) ** 2).mean((1, 2))
        lvpn = ((phys.vpn(v) - phys.vpn(vt)) / vpn_scale).pow(2).mean(1)
        cur = phys.currents(v, s)
        rat = phys.rating[1:].view(1, -1, 1)
        lcur = ((cur[:, 1:] - it[:, 1:]).abs() / rat).pow(2).mean((1, 2))
        lkvl = (phys.kvl(v, cur).abs() / kvl_scale).pow(2).mean((1, 2))
        per = W["v"] * lv + W["vpn"] * lvpn + W["cur"] * lcur + W["kvl"] * lkvl
        return (per * w).sum() / w.sum().clamp(min=1.0), {"v": float((lv * w).sum() / w.sum()), "vpn": float((lvpn * w).sum() / w.sum())}

    def evaluate(d):
        model.eval()
        tot, n, comp = 0.0, 0, {}
        with torch.no_grad():
            for k in range(0, len(d["x"]), 128):
                idx = np.arange(k, min(k + 128, len(d["x"])))
                l, c = loss_fn(d, idx, forward(d, idx))
                m = float(d["conv"][idx].sum())
                tot += float(l) * m
                n += m
        model.train()
        return tot / max(n, 1)

    rng = np.random.default_rng(CONFIG["seed"])
    curve, best, best_ep, bad = [], np.inf, -1, 0
    ckdir = OUT / "runs" / f"{model_kind}_{tag}"
    ckdir.mkdir(parents=True, exist_ok=True)
    v0 = evaluate(va)
    curve.append({"epoch": 0, "val_loss": v0, "note": "untrained = U1 prior"})
    print(json.dumps(curve[-1]), flush=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(len(tr["x"]))
        tl, nb = 0.0, 0
        for k in range(0, len(perm), CONFIG["batch"]):
            idx = np.sort(perm[k:k + CONFIG["batch"]])
            if not tr["conv"][idx].any():
                continue
            l, _ = loss_fn(tr, idx, forward(tr, idx))
            opt.zero_grad()
            l.backward()
            g = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(x).all() for x in g):
                raise FloatingPointError("non-finite gradient")
            opt.step()
            tl += float(l)
            nb += 1
        sched.step()
        vl = evaluate(va)
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "epoch_s": time.perf_counter() - t0,
               "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None}
        curve.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            tmp = ckdir / ".best.partial.pt"
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, ckdir / "best.pt")
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    meta = {"model": model_kind, "tag": tag, "config": CONFIG, "n_params": n_params, "node_in": node_in,
            "best_epoch": best_ep, "best_val_loss": best, "curve": curve,
            "scalers": {"x": xs.state(), "edge": es.state(), "residual": rs.tolist(), "vpn": vpn_scale, "kvl": kvl_scale},
            "states": {"train": int(len(tr["x"])), "val": int(len(va["x"])), "train_converged": int(tr["conv"].sum())},
            "thevenin_sha256": sha(OUT / "thevenin.npz"), "code_commit": os.environ.get("GENCODOE_COMMIT"),
            "checkpoint_sha256": sha(ckdir / "best.pt") if (ckdir / "best.pt").exists() else None}
    tmp = ckdir / ".meta.partial.json"
    tmp.write_text(json.dumps(meta, indent=1, default=float))
    os.replace(tmp, ckdir / "meta.json")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embeddings", "fit"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", choices=["phasegraph", "phasegenco"], default="phasegraph")
    ap.add_argument("--epochs", type=int, default=CONFIG["max_epochs"])
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "embeddings":
        print(embeddings(a.split, a.device))
    else:
        print(json.dumps({k: v for k, v in run_fit(a.model, a.epochs, a.tag, a.device, a.max_train).items()
                          if k not in ("scalers", "curve")}, indent=1, default=float))


if __name__ == "__main__":
    main()
