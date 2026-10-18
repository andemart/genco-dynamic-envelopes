"""Unbalanced GENCO (`GNS_heterogeneous_4wire`), data assembly and physics-consistent losses
(docs/unbalanced_genco_protocol.md).

GENCO-style encode-process-decode on the 397 physical bus nodes of the four-wire ENWL1 graph with
conductor-vector states. Processor: edge-conditioned TransformerConv (GENCO's processor family) on
parent->child and child->parent edges, residual updates, per-layer LayerNorm on node states and a
per-graph global context channel. Encoders have no final LayerNorm (the stock encoder's LayerNorm
cancelled |Y|). The decoder predicts dRe/dIm of Va, Vb, Vc, Vn added to the one-iteration four-wire
sweep prior (U1); its last layer is zero-initialised so the untrained model reproduces U1 exactly.
Branch currents are derived by KCL from the predicted voltages (not Z^-1 dV, which is ill-conditioned
on short LV segments); KVL enters as a consistency loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .fourwire_net import V_PN_NOM, FourWireNet, Thevenin, sweep

N_NODE_BASE = 20          # Pa..Qc (6), customers per phase (3), root, depth, distance, prior Re/Im a,b,c,n (8)
N_EDGE = 35               # R (16), X (16), rating, length, direction
EMB_WIDTH = 96            # balanced GENCO final-layer bus embedding (PhaseGENCO only)


# ------------------------------------------------------------------------------------------- data
class Static:
    """Topology tensors shared by every state."""

    def __init__(self, net: FourWireNet, th: Thevenin):
        self.net, self.th = net, th
        n = net.n_bus
        child = np.arange(1, n)
        par = net.parent[1:]
        self.edge_index = np.vstack([np.concatenate([par, child]), np.concatenate([child, par])])
        z = net.z[1:]
        ef = np.hstack([z.real.reshape(-1, 16), z.imag.reshape(-1, 16), net.rating_a[1:, None], net.length_m[1:, None]])
        self.edge_raw = np.vstack([np.hstack([ef, np.ones((n - 1, 1))]), np.hstack([ef, -np.ones((n - 1, 1))])])
        cnt = np.zeros((n, 3))
        np.add.at(cnt, (net.customer_bus, net.customer_phase), 1)
        dist = np.zeros(n)
        for i in net.order[1:]:
            dist[i] = dist[net.parent[i]] + net.z[i][0, 0].real
        root = np.zeros(n)
        root[0] = 1
        self.node_static = np.hstack([cnt, root[:, None], net.depth[:, None], dist[:, None]])
        # subtree matrix: A[j, i] = 1 if i is in the subtree of j (branch current into j = A @ injections)
        a = np.eye(n)
        for i in net.order[::-1][:-1]:
            a[net.parent[i]] += a[i]
        self.subtree = a
        self.rating_ka = np.where(np.isfinite(net.rating_a), net.rating_a, np.inf) / 1e3


def bus_injections(net: FourWireNet, p_kw: np.ndarray, q_kvar: np.ndarray) -> np.ndarray:
    """(S, n_bus, 6) Pa, Qa, Pb, Qb, Pc, Qc per bus (kW, kvar)."""
    out = np.zeros((p_kw.shape[0], net.n_bus, 6), np.float32)
    for k, (b, ph) in enumerate(zip(net.customer_bus, net.customer_phase)):
        out[:, b, 2 * ph] += p_kw[:, k]
        out[:, b, 2 * ph + 1] += q_kvar[:, k]
    return out


def prior_u1(st: Static, p_kw, q_kvar, chunk: int = 2048) -> np.ndarray:
    """U1: one sweep iteration from nominal, (S, n_bus, 4) complex kV."""
    out = []
    for i in range(0, len(p_kw), chunk):
        s = p_kw[i:i + chunk].astype(complex) + 1j * q_kvar[i:i + chunk]
        out.append(sweep(st.net, st.th, s, iterations=1)[0])
    return np.concatenate(out)


def node_features(st: Static, p_kw, q_kvar, v1: np.ndarray) -> np.ndarray:
    """(S, n_bus, 20) raw node features (scaled later with training statistics)."""
    s = len(p_kw)
    pv = v1 / V_PN_NOM
    pri = np.concatenate([pv.real, pv.imag], axis=2)                      # (S, n, 8)
    return np.concatenate([bus_injections(st.net, p_kw, q_kvar), np.broadcast_to(st.node_static, (s,) + st.node_static.shape),
                           pri], axis=2).astype(np.float32)


class Scaler:
    """asinh then standardise; statistics from training data only."""

    def __init__(self, x: np.ndarray):
        y = np.arcsinh(x.reshape(-1, x.shape[-1]).astype(np.float64))
        self.mu, self.sd = y.mean(0), y.std(0)
        self.sd[self.sd < 1e-8] = 1.0

    def __call__(self, x):
        return ((np.arcsinh(x.astype(np.float64)) - self.mu) / self.sd).astype(np.float32)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


# ------------------------------------------------------------------------------------------- model
class GNS_heterogeneous_4wire(nn.Module):
    def __init__(self, node_in: int, edge_in: int = N_EDGE, hidden: int = 64, layers: int = 8, heads: int = 4):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        self.node_in = node_in
        self.enc_node = nn.Sequential(nn.Linear(node_in, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden))
        self.enc_edge = nn.Sequential(nn.Linear(edge_in, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, concat=True)
                                   for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ctx = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 8))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x, edge_index, edge_attr, n_graphs: int):
        h = self.enc_node(x)
        e = self.enc_edge(edge_attr)
        n = x.shape[0] // n_graphs
        for conv, norm, ctx in zip(self.convs, self.norms, self.ctx):
            m = conv(h, edge_index, e)
            g = m.view(n_graphs, n, -1).mean(1, keepdim=True).expand(-1, n, -1).reshape(-1, m.shape[-1])
            h = norm(h + m + ctx(g))
        return self.dec(h)                                                 # (B*n, 8) normalised residual


def register():
    """Register in graphkit's model registry (construction only; training uses fourwire_train)."""
    try:
        from gridfm_graphkit.io.registries import MODELS_REGISTRY
    except Exception:
        return
    if "GNS_heterogeneous_4wire" not in getattr(MODELS_REGISTRY, "_registry", {}):
        try:
            MODELS_REGISTRY.register("GNS_heterogeneous_4wire")(GNS_heterogeneous_4wire)
        except Exception:
            pass


class Batcher:
    """Batched fixed-topology graphs (every state has the same 397-bus graph)."""

    def __init__(self, st: Static, edge_attr: np.ndarray, device):
        self.n = st.net.n_bus
        self.ei = torch.as_tensor(st.edge_index, dtype=torch.long, device=device)
        self.ea = torch.as_tensor(edge_attr, dtype=torch.float32, device=device)
        self.device = device
        self._cache = {}

    def graph(self, b: int):
        if b not in self._cache:
            off = (torch.arange(b, device=self.device) * self.n).repeat_interleave(self.ei.shape[1])
            self._cache[b] = (self.ei.repeat(1, b) + off, self.ea.repeat(b, 1))
        return self._cache[b]


# ------------------------------------------------------------------------------------------- physics in torch
class Physics(nn.Module):
    """Torch versions of V_pn, KCL branch currents and the KVL residual (complex64)."""

    def __init__(self, st: Static, device):
        super().__init__()
        net = st.net
        self.cb = torch.as_tensor(net.customer_bus, device=device)
        self.cp = torch.as_tensor(net.customer_phase, device=device)
        self.sub = torch.as_tensor(st.subtree, dtype=torch.complex64, device=device)
        self.par = torch.as_tensor(net.parent[1:], device=device)
        self.z = torch.as_tensor(net.z[1:], dtype=torch.complex64, device=device)          # ohm
        self.rating = torch.as_tensor(st.rating_ka, dtype=torch.float32, device=device)
        self.n = net.n_bus

    def vpn(self, v):                                                      # v (B, n, 4) complex kV
        vb = v[:, self.cb]
        vph = vb.gather(2, self.cp.view(1, -1, 1).expand(v.shape[0], -1, 1)).squeeze(2)
        return (vph - vb[..., 3]).abs() / V_PN_NOM

    def currents(self, v, s_kva):                                          # s (B, n_c) complex kVA
        vb = v[:, self.cb]
        vph = vb.gather(2, self.cp.view(1, -1, 1).expand(v.shape[0], -1, 1)).squeeze(2)
        vpn = vph - vb[..., 3]
        ic = torch.conj(s_kva / 1e3 / torch.where(vpn.abs() > 1e-9, vpn, torch.full_like(vpn, 1e-9)))
        inj = torch.zeros(v.shape[0], self.n, 4, dtype=v.dtype, device=v.device)
        idx = (self.cb * 4 + self.cp)
        inj.view(v.shape[0], -1).index_add_(1, idx, ic)
        inj.view(v.shape[0], -1).index_add_(1, self.cb * 4 + 3, -ic)
        return torch.einsum("jk,bkc->bjc", self.sub, inj)                   # (B, n, 4) into each bus

    def kvl(self, v, cur):                                                 # (B, n-1, 4) pu
        drop = torch.einsum("eij,bej->bei", self.z, cur[:, 1:])
        return (v[:, self.par] - v[:, 1:] - drop) / V_PN_NOM


def summaries(phys: Physics, v, cur):
    """State-level worst cases: min V_pn, max V_pn, max phase loading (non-root branches), root loading proxy."""
    vpn = phys.vpn(v)
    ph = cur[:, 1:, :3].abs() / phys.rating[1:].view(1, -1, 1)
    s_root = (v[:, 0, :3] * torch.conj(cur[:, 0, :3])).abs().sum(1) * 1e3 / 800.0     # kVA / 800 kVA
    return torch.stack([vpn.min(1).values, vpn.max(1).values, ph.amax((1, 2)), s_root], 1)
