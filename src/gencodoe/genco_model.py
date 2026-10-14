"""GENCO variant for low-voltage feeders, registered in graphkit's model registry.

Why: on ENWL1 the normalized admittance edge features are ~1e3-3e5 (short LV
segments, normalizer baseMVA ~2 kW). graphkit's edge encoder
(Linear -> LeakyReLU -> Linear -> LayerNorm) is then positively homogeneous in
its input, so LayerNorm cancels the admittance magnitude: the stock model sees
each line's R/X direction but not its impedance (scaling every Y by 10 leaves
the edge embedding unchanged, cosine similarity 1.000000). The per-layer power
residual fed back through physics_mlp has the same scale problem.

GNS_heterogeneous_lv keeps graphkit's architecture and forward pass and only
inserts fixed asinh compressions in front of the two encoders that see these
quantities. The physics layers still receive the raw admittances, so branch
flows and residuals are unchanged.

Option model.linear_prior (default false): add a fixed linear power-flow
estimate to the bus head, so the network learns the nonlinear correction. The
estimate is the first fixed-point iterate from a flat start,
V0 = Vs + Z conj(S) / conj(Vs), Z = inverse of the bus admittance matrix
without the slack row/column, built from the same (normalized) admittance and
injection features the model receives. All states of one feeder share Z, so it
is computed once per topology (float64, CPU) and cached; per batch the prior
costs one matmul. The head adds model.prior_scale (default 1e-3) times its
output, starting from zero. Supports REF + PQ buses only (no PV buses). graphkit's PF
mask hides the slack Vm (input 0); the prior then assumes 1.0 pu, which is how
every ENWL1 case is generated (write_matpower pins the source bus).

Option model.global_context (default false): after every layer, add a learned
projection of the graph-mean bus state (a virtual node). A radial feeder's
voltage at one bus depends on the load anywhere downstream of its path to the
slack; ENWL1 has depth 38, so a 12-hop model cannot see the total load without
a global channel.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter_mean

from gridfm_graphkit.datasets.globals import (BS, GS, PD_H, QD_H, REF_H, PV_H, VA_H, VA_OUT, VM_H, VM_OUT,
                                              YFF_TT_I, YFF_TT_R, YFT_TF_I, YFT_TF_R)
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous


class Asinh(nn.Module):
    """asinh on all columns, or only on columns[lo:hi]."""

    def __init__(self, lo: int | None = None, hi: int | None = None):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        if self.lo is None:
            return torch.asinh(x)
        x = x.clone()
        x[:, self.lo:self.hi] = torch.asinh(x[:, self.lo:self.hi])
        return x


class _BatchIndex:
    """Holds the bus-to-graph index (and linear prior) of the current batch."""
    batch: torch.Tensor | None = None
    prior: torch.Tensor | None = None


class GlobalContextNorm(nn.Module):
    """LayerNorm followed by + W * mean over the bus nodes of the same graph."""

    def __init__(self, norm: nn.LayerNorm, holder: _BatchIndex):
        super().__init__()
        self.norm = norm
        dim = norm.normalized_shape[0]
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self._holder = [holder]   # list: keep it out of the module tree

    def forward(self, x):
        y = self.norm(x)
        batch = self._holder[0].batch
        if batch is None:
            batch = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
        g = scatter_mean(y, batch, dim=0)
        return y + self.proj(g)[batch]


class PriorHead(nn.Module):
    """Wraps the bus head: prior + scale * head(h) for Vm and Va.

    The head's last layer starts at zero, so training starts exactly at the
    prior, and `scale` (pu, rad) lets the head work in O(1) units although the
    needed correction is ~1e-4 pu. Other output columns are left unscaled."""

    def __init__(self, head: nn.Module, holder: "_BatchIndex", scale: float):
        super().__init__()
        self.head = head
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        self.scale = scale
        self._holder = [holder]

    def forward(self, h):
        out = self.head(h)
        prior = self._holder[0].prior
        if prior is None:
            return out
        add = torch.zeros_like(out)
        add[:, VM_OUT] = prior[:, 0]
        add[:, VA_OUT] = prior[:, 1]
        mul = torch.ones(out.shape[1], device=out.device, dtype=out.dtype)
        mul[VM_OUT] = mul[VA_OUT] = self.scale
        return out * mul + add


def linear_prior(x_bus: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor | None,
                 cache: dict) -> torch.Tensor:
    """(n_nodes, 2) first fixed-point iterate (vm, va rad) for a batch of graphs that share one topology."""
    n_graphs = 1 if batch is None else int(batch.max()) + 1
    n = x_bus.shape[0] // n_graphs
    if n * n_graphs != x_bus.shape[0] or (x_bus[:, PV_H] != 0).any():
        raise ValueError("linear_prior needs equal-size graphs with REF and PQ buses only")
    e0 = edge_index[0] < n                                    # edges of the first graph
    ei, ea = edge_index[:, e0].cpu(), edge_attr[e0].detach().cpu().double()
    key = (n, ei.numpy().tobytes(), ea[:, YFF_TT_R:YFT_TF_I + 1].numpy().tobytes())
    if key not in cache:
        y = torch.zeros(n, n, dtype=torch.complex128)
        yaa = torch.complex(ea[:, YFF_TT_R], ea[:, YFF_TT_I])
        yab = torch.complex(ea[:, YFT_TF_R], ea[:, YFT_TF_I])
        y.index_put_((ei[0], ei[0]), yaa, accumulate=True)
        y.index_put_((ei[0], ei[1]), yab, accumulate=True)
        xb = x_bus[:n].detach().cpu().double()
        y += torch.diag(torch.complex(xb[:, GS], xb[:, BS]))
        ref = torch.nonzero(xb[:, REF_H] == 1).flatten()
        keep = torch.nonzero(xb[:, REF_H] != 1).flatten()
        z = torch.linalg.inv(y[keep][:, keep])
        cache.clear()
        cache[key] = (ref, keep, z)
    ref, keep, z = cache[key]
    # all graphs of the batch must share the admittances of the first one
    ea_all = edge_attr[:, YFF_TT_R:YFT_TF_I + 1].detach().cpu().double().reshape(n_graphs, -1)
    if not torch.allclose(ea_all, ea_all[:1].expand_as(ea_all), rtol=1e-6, atol=0):
        raise ValueError("linear_prior: graphs in one batch have different admittances")
    xb = x_bus.detach().cpu().double().reshape(n_graphs, n, -1)
    vm_s = xb[:, ref, VM_H].squeeze(1)
    vm_s = torch.where(vm_s > 0, vm_s, torch.ones_like(vm_s))                         # masked slack Vm -> 1 pu
    vs = torch.polar(vm_s, xb[:, ref, VA_H].squeeze(1))                               # (G,)
    s = -torch.complex(xb[:, :, PD_H], xb[:, :, QD_H])                                 # injections, (G, n)
    i_red = torch.conj(s[:, keep]) / torch.conj(vs)[:, None]
    v = vs[:, None].expand(n_graphs, n).clone()
    v[:, keep] = vs[:, None] + i_red @ z.T
    prior = torch.stack([v.abs(), v.angle()], dim=-1).reshape(n_graphs * n, 2)
    return prior.to(device=x_bus.device, dtype=x_bus.dtype)


@MODELS_REGISTRY.register("GNS_heterogeneous_lv")
class GNS_heterogeneous_lv(GNS_heterogeneous):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.input_proj_edge = nn.Sequential(Asinh(YFF_TT_R, YFT_TF_I + 1), *self.input_proj_edge)
        self.physics_mlp = nn.Sequential(Asinh(), *self.physics_mlp)
        self._ctx = _BatchIndex()
        self.global_context = bool(getattr(args.model, "global_context", False))
        if self.global_context:
            self.norms_bus = nn.ModuleList(GlobalContextNorm(n, self._ctx) for n in self.norms_bus)
        self.use_prior = bool(getattr(args.model, "linear_prior", False))
        if self.use_prior:
            self.mlp_bus = PriorHead(self.mlp_bus, self._ctx, float(getattr(args.model, "prior_scale", 1e-3)))
        self._z_cache: dict = {}

    def forward(self, batch, return_embeddings: bool = False):
        bus = batch["bus"]
        self._ctx.batch = bus.batch if hasattr(bus, "batch") else None
        if self.use_prior:
            et = ("bus", "connects", "bus")
            self._ctx.prior = linear_prior(batch.x_dict["bus"], batch.edge_index_dict[et], batch.edge_attr_dict[et],
                                           self._ctx.batch, self._z_cache)
        try:
            return super().forward(batch, return_embeddings=return_embeddings)
        finally:
            self._ctx.batch = None
            self._ctx.prior = None
