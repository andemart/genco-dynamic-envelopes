"""GENCO-style graph model (and an MLP baseline) that amortise the multi-scenario OPF envelope
(docs/genco_opf_protocol.md, amendment 2). Trained from scratch on case33bw.

    python -m gencodoe.opf_gnn fit --model gnn --tag smoke --epochs 5
    python -m gencodoe.opf_gnn fit --model mlp --tag final

Output: the shared envelope as a fraction of each unit's cap (sigmoid). Loss: cap-weighted MSE over units.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import opf_case33 as oc
from .opf_labels import OUT as LABELS

RUNS = Path("data_out/opf_case33/runs")
CONFIG = {"hidden": 64, "layers": 6, "heads": 4, "lr": 1e-3, "weight_decay": 1e-4, "batch": 32, "max_epochs": 200,
          "patience": 15, "seed": 0}


class Scaler:
    def __init__(self, x):
        y = np.arcsinh(x.reshape(-1, x.shape[-1]).astype(np.float64))
        self.mu, self.sd = y.mean(0), y.std(0)
        self.sd[self.sd < 1e-8] = 1.0

    def __call__(self, x):
        return ((np.arcsinh(x.astype(np.float64)) - self.mu) / self.sd).astype(np.float32)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def load(cls, s):
        o = cls.__new__(cls)
        o.mu, o.sd = np.array(s["mu"]), np.array(s["sd"])
        return o


class EnvGNN(nn.Module):
    """Encode-process-decode with edge-conditioned TransformerConv (GENCO's processor family); encoders have
    no final LayerNorm (the stock edge encoder's LayerNorm cancels impedance magnitude)."""

    def __init__(self, node_in, edge_in, hidden=64, layers=6, heads=4):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        self.enc_n = nn.Sequential(nn.Linear(node_in, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden))
        self.enc_e = nn.Sequential(nn.Linear(edge_in, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ctx = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))

    def forward(self, x, ei, ea, b):
        h, e = self.enc_n(x), self.enc_e(ea)
        n = x.shape[0] // b
        for conv, norm, ctx in zip(self.convs, self.norms, self.ctx):
            m = conv(h, ei, e)
            g = m.view(b, n, -1).mean(1, keepdim=True).expand(-1, n, -1).reshape(-1, m.shape[-1])
            h = norm(h + m + ctx(g))
        return torch.sigmoid(self.dec(h)).view(b, n)


class EnvMLP(nn.Module):
    def __init__(self, n_bus, node_in, hidden=256):
        super().__init__()
        self.n = n_bus
        self.net = nn.Sequential(nn.Linear(n_bus * node_in, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden),
                                 nn.LeakyReLU(), nn.Linear(hidden, n_bus))

    def forward(self, x, ei, ea, b):
        return torch.sigmoid(self.net(x.view(b, -1))).view(b, self.n)


def graph(case: oc.Case33):
    fd = case.fd
    child = np.flatnonzero(fd.parent >= 0)
    par = fd.parent[child]
    ei = np.vstack([np.concatenate([par, child]), np.concatenate([child, par])])
    ef = np.column_stack([fd.r[child], fd.x[child], fd.rate[child]])
    ea = np.vstack([np.hstack([ef, np.ones((len(child), 1))]), np.hstack([ef, -np.ones((len(child), 1))])])
    return ei, ea


def load_days(days):
    X, Y, M = [], [], []
    for d in days:
        z = np.load(LABELS / f"day{d:03d}.npz")
        X.append(z["X"])
        Y.append(z["Y"])
        M += json.loads(str(z["meta"]))
    return np.concatenate(X), np.concatenate(Y), M


def build_model(kind, n_bus, node_in, edge_in):
    torch.manual_seed(CONFIG["seed"])
    if kind == "gnn":
        return EnvGNN(node_in, edge_in, CONFIG["hidden"], CONFIG["layers"], CONFIG["heads"])
    return EnvMLP(n_bus, node_in)


class Predictor:
    """Loads a trained run; predict(features (B, n, F)) -> fraction of cap per bus."""

    def __init__(self, run_dir: Path, case: oc.Case33, device="cpu"):
        self.meta = json.loads((run_dir / "meta.json").read_text())
        self.xs = Scaler.load(self.meta["scalers"]["x"])
        es = Scaler.load(self.meta["scalers"]["edge"])
        ei, ea = graph(case)
        self.n = case.fd.n_bus
        self.dev = torch.device(device)
        self.model = build_model(self.meta["model"], self.n, self.meta["node_in"], ea.shape[1]).to(self.dev)
        self.model.load_state_dict(torch.load(run_dir / "best.pt", map_location=self.dev))
        self.model.eval()
        self.ei0 = torch.as_tensor(ei, dtype=torch.long, device=self.dev)
        self.ea0 = torch.as_tensor(es(ea), device=self.dev)

    def batch_graph(self, b):
        off = (torch.arange(b, device=self.dev) * self.n).repeat_interleave(self.ei0.shape[1])
        return self.ei0.repeat(1, b) + off, self.ea0.repeat(b, 1)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        b = feats.shape[0]
        with torch.no_grad():
            x = torch.as_tensor(self.xs(feats), device=self.dev).reshape(b * self.n, -1)
            ei, ea = self.batch_graph(b)
            return self.model(x, ei, ea, b).cpu().numpy()


def fit(kind, tag, epochs, device="cpu"):
    case = oc.build()
    Xtr, Ytr, Mtr = load_days(oc.SPLIT["train"])
    Xva, Yva, Mva = load_days(oc.SPLIT["val"])
    ei, ea = graph(case)
    xs, es = Scaler(Xtr), Scaler(ea)                                   # training statistics only
    cap_w = np.zeros(case.fd.n_bus, np.float32)                        # loss weight: unit caps (hosting-aligned)
    dev = torch.device(device)
    model = build_model(kind, case.fd.n_bus, Xtr.shape[-1], ea.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    ei0, ea0 = torch.as_tensor(ei, dtype=torch.long, device=dev), torch.as_tensor(es(ea), device=dev)
    n = case.fd.n_bus

    def batch_graph(b):
        off = (torch.arange(b, device=dev) * n).repeat_interleave(ei0.shape[1])
        return ei0.repeat(1, b) + off, ea0.repeat(b, 1)

    def weights(M):
        w = np.zeros((len(M), n), np.float32)
        for k, m in enumerate(M):
            w[k, case.unit_bus] = case.cap[m["direction"]]
        return w
    Wtr, Wva = weights(Mtr), weights(Mva)
    Xtr_s, Xva_s = xs(Xtr), xs(Xva)
    del cap_w

    def loss_on(X, Y, W, idx):
        b = len(idx)
        x = torch.as_tensor(X[idx], device=dev).reshape(b * n, -1)
        y, w = torch.as_tensor(Y[idx], device=dev), torch.as_tensor(W[idx], device=dev)
        e1, e2 = batch_graph(b)
        yh = model(x, e1, e2, b)
        return ((yh - y) ** 2 * w).sum() / w.sum()

    rng = np.random.default_rng(CONFIG["seed"])
    best, bad, curve = np.inf, 0, []
    run = RUNS / f"{kind}_{tag}"
    run.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(len(Xtr_s))
        tl = 0.0
        for k in range(0, len(perm), CONFIG["batch"]):
            idx = perm[k:k + CONFIG["batch"]]
            l = loss_on(Xtr_s, Ytr, Wtr, idx)
            opt.zero_grad()
            l.backward()
            opt.step()
            tl += float(l) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(np.sum([float(loss_on(Xva_s, Yva, Wva, np.arange(k, min(k + 256, len(Xva_s))))) *
                               len(range(k, min(k + 256, len(Xva_s)))) for k in range(0, len(Xva_s), 256)]) / len(Xva_s))
        curve.append({"epoch": ep, "train": tl / len(perm), "val": vl, "s": time.perf_counter() - t0})
        if vl < best:
            best, bad = vl, 0
            tmp = run / ".best.partial.pt"
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, run / "best.pt")
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    meta = {"model": kind, "tag": tag, "config": CONFIG, "node_in": int(Xtr.shape[-1]), "best_val": best,
            "best_epoch": int(np.argmin([c["val"] for c in curve]) + 1), "curve": curve,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "scalers": {"x": xs.state(), "edge": es.state()}, "train_intervals": int(len(Xtr)), "val_intervals": int(len(Xva))}
    tmp = run / ".meta.partial.json"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, run / "meta.json")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--model", choices=["gnn", "mlp"], required=True)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--epochs", type=int, default=CONFIG["max_epochs"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m = fit(a.model, a.tag, a.epochs, a.device)
    print(json.dumps({k: v for k, v in m.items() if k not in ("curve", "scalers")}, indent=1))


if __name__ == "__main__":
    main()
