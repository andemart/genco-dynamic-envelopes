"""Radial distribution feeder and the power-flow oracles used to test envelopes.

A feeder is stored in per unit with bus 0 as the substation (slack). Every other
bus k has exactly one parent, and branch k is the line parent[k] -> k, so
branches and non-root buses share an index (branch 0 is unused).

All oracles take batched net injections p, q of shape (n, n_bus) in pu
(positive = export into the grid) and return a PFResult:

  * sweep_pf      exact balanced AC power flow (backward/forward sweep), the
                  ground truth that GENCO is compared against
  * lindistflow   linearised power flow, the cheap baseline (the distribution
                  analogue of DC power flow)
  * GENCO         see genco.py; same interface, so it drops into envelopes.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Feeder:
    parent: np.ndarray   # (n_bus,) parent bus, -1 for the root
    r: np.ndarray        # (n_bus,) series resistance of branch parent[k] -> k, pu
    x: np.ndarray        # (n_bus,) series reactance, pu
    rate: np.ndarray     # (n_bus,) thermal limit, pu MVA (inf = unconstrained)
    base_mva: float = 1.0
    v_slack: float = 1.0
    vmin: float = 0.94
    vmax: float = 1.10

    def __post_init__(self):
        n = len(self.parent)
        self.order = _bfs_order(self.parent)
        # T[i, k] = 1 if bus i lies in the subtree rooted at bus k (branch k feeds i)
        T = np.zeros((n, n))
        for k in range(n):
            i = k
            while i != -1:
                T[k, i] = 1.0
                i = self.parent[i]
        T[:, 0] = 0.0  # no branch into the root
        self.T = T

    @property
    def n_bus(self) -> int:
        return len(self.parent)

    @property
    def z(self) -> np.ndarray:
        return self.r + 1j * self.x

    @classmethod
    def from_branches(cls, from_bus, to_bus, r, x, rate=None, root: int = 0, **kw) -> Feeder:
        """Build from an undirected radial branch list (bus ids 0..n-1, root = substation)."""
        from_bus, to_bus = np.asarray(from_bus), np.asarray(to_bus)
        n = int(max(from_bus.max(), to_bus.max())) + 1
        if len(from_bus) != n - 1:
            raise ValueError(f"radial feeder with {n} buses needs {n - 1} branches, got {len(from_bus)}")
        if root != 0:
            raise ValueError("relabel buses so that the substation is bus 0")
        rate = np.full(len(from_bus), np.inf) if rate is None else np.asarray(rate, float)
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
        for b, (f, t) in enumerate(zip(from_bus, to_bus)):
            adj[int(f)].append((int(t), b))
            adj[int(t)].append((int(f), b))
        parent = np.full(n, -1)
        rr, xx, rt = np.zeros(n), np.zeros(n), np.full(n, np.inf)
        seen, queue = {0}, deque([0])
        while queue:
            u = queue.popleft()
            for v, b in adj[u]:
                if v not in seen:
                    seen.add(v)
                    parent[v], rr[v], xx[v], rt[v] = u, r[b], x[b], rate[b]
                    queue.append(v)
        if len(seen) != n:
            raise ValueError("feeder is not connected")
        return cls(parent=parent, r=rr, x=xx, rate=rt, **kw)


def _bfs_order(parent: np.ndarray) -> np.ndarray:
    children: dict[int, list[int]] = {}
    for k, p in enumerate(parent):
        children.setdefault(int(p), []).append(k)
    order, queue = [], deque(children.get(-1, []))
    while queue:
        u = queue.popleft()
        order.append(u)
        queue.extend(children.get(u, []))
    return np.array(order)


def synthetic_feeder(n_trunk: int = 12, n_laterals: int = 3, lateral_len: int = 4, r: float = 0.012,
                     x: float = 0.008, rate: float = 4.0, **kw) -> Feeder:
    """A trunk with evenly spaced laterals; impedances in pu on base_mva."""
    f, t = [], []
    for k in range(1, n_trunk + 1):
        f.append(k - 1), t.append(k)
    nxt = n_trunk + 1
    taps = np.linspace(2, n_trunk - 1, n_laterals).astype(int) if n_laterals else []
    for tap in taps:
        prev = int(tap)
        for _ in range(lateral_len):
            f.append(prev), t.append(nxt)
            prev, nxt = nxt, nxt + 1
    nb = len(f)
    return Feeder.from_branches(f, t, np.full(nb, r), np.full(nb, x), np.full(nb, rate), **kw)


def from_pandapower(net, base_mva: float = 1.0, **kw) -> tuple[Feeder, np.ndarray, np.ndarray]:
    """Convert a radial pandapower net (lines only, one ext_grid at bus 0).

    Returns (feeder, pd, qd) with the nominal bus loads in pu. Ratings of
    max_i_ka >= 1000 (the 99999 placeholder of MATPOWER conversions) become inf.
    """
    line = net.line[net.line.in_service]
    vn = net.bus.vn_kv.to_numpy()
    zbase = vn[line.from_bus.to_numpy()] ** 2 / base_mva
    length = line.length_km.to_numpy() / line.parallel.to_numpy()
    r = line.r_ohm_per_km.to_numpy() * length / zbase
    x = line.x_ohm_per_km.to_numpy() * length / zbase
    imax = line.max_i_ka.to_numpy() * line.parallel.to_numpy()
    rate = np.where(imax >= 1000, np.inf, np.sqrt(3) * vn[line.from_bus.to_numpy()] * imax / base_mva)
    ext = net.ext_grid[net.ext_grid.in_service]
    if len(ext) != 1 or int(ext.bus.iloc[0]) != 0:
        raise ValueError("expected one in-service ext_grid at bus 0")
    feeder = Feeder.from_branches(line.from_bus.to_numpy(), line.to_bus.to_numpy(), r, x, rate,
                                  base_mva=base_mva, v_slack=float(ext.vm_pu.iloc[0]), **kw)
    pd, qd = np.zeros(feeder.n_bus), np.zeros(feeder.n_bus)
    load = net.load[net.load.in_service]
    s = load.scaling.to_numpy()
    np.add.at(pd, load.bus.to_numpy(), load.p_mw.to_numpy() * s / base_mva)
    np.add.at(qd, load.bus.to_numpy(), load.q_mvar.to_numpy() * s / base_mva)
    return feeder, pd, qd


@dataclass
class PFResult:
    vm: np.ndarray        # (n, n_bus) voltage magnitude, pu
    loading: np.ndarray   # (n, n_bus) |S| / rate at the sending end of branch k (column 0 = 0)
    p_branch: np.ndarray  # (n, n_bus) active flow parent -> child, pu
    converged: np.ndarray | None = None  # (n,) bool, None for non-iterative oracles


def _loading(feeder: Feeder, s_branch: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(s_branch) / feeder.rate[None, :]
    out[:, 0] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0)


def sweep_pf(feeder: Feeder, p: np.ndarray, q: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> PFResult:
    """Exact AC power flow on a radial feeder, vectorised over scenarios.

    Backward sweep: branch current J = sum of load currents in the subtree.
    Forward sweep:  V = V_slack - sum of z * J along the path from the root.
    Both are a single matmul with the subtree matrix T.
    """
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    s = p + 1j * q
    T, z = feeder.T, feeder.z
    v = np.full(s.shape, feeder.v_slack, dtype=complex)
    converged = np.zeros(s.shape[0], dtype=bool)
    for _ in range(max_iter):
        i_load = -np.conj(s / v)          # current drawn by each bus
        i_load[:, 0] = 0.0
        j = i_load @ T                    # j[:, k] = current in branch parent[k] -> k
        v_new = feeder.v_slack - (j * z[None, :]) @ T.T
        dv = np.abs(v_new - v).max(axis=1)
        v = v_new
        converged = dv < tol
        if converged.all():
            break
    s_branch = v[:, feeder.parent] * np.conj(j)
    s_branch[:, 0] = 0.0
    return PFResult(vm=np.abs(v), loading=_loading(feeder, s_branch), p_branch=s_branch.real, converged=converged)


def lindistflow(feeder: Feeder, p: np.ndarray, q: np.ndarray) -> PFResult:
    """Lossless linearised power flow (LinDistFlow, |V| ~ 1)."""
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    R, X = sensitivity(feeder)
    vm = feeder.v_slack + p @ R + q @ X
    s_branch = -(p + 1j * q) @ feeder.T
    s_branch[:, 0] = 0.0
    return PFResult(vm=vm, loading=_loading(feeder, s_branch), p_branch=s_branch.real)


def sensitivity(feeder: Feeder) -> tuple[np.ndarray, np.ndarray]:
    """dV_i/dP_j and dV_i/dQ_j: summed r (x) over the path shared by buses i and j (symmetric)."""
    R = (feeder.T * feeder.r[None, :]) @ feeder.T.T
    X = (feeder.T * feeder.x[None, :]) @ feeder.T.T
    return R, X


def branch_flows_kcl(feeder: Feeder, vm: np.ndarray, va: np.ndarray, p: np.ndarray, q: np.ndarray) -> PFResult:
    """Branch flows from predicted voltages via Kirchhoff's current law (one backward sweep).

    Branch current = sum of the injection currents conj(s_i / v_i) in its subtree;
    |S| at the sending end = |v_parent| |J|. Voltage errors enter only through
    the current-to-power conversion, so the flow error is O(dV) relative. Use this
    for surrogate voltages on short LV segments, where the voltage-difference form
    (branch_flows_from_voltages) multiplies a voltage error by 1/|z|.
    """
    v = vm * np.exp(1j * va)
    i_inj = -np.conj((p + 1j * q) / v)
    i_inj[:, 0] = 0.0
    j = i_inj @ feeder.T
    s_branch = v[:, feeder.parent] * np.conj(j)
    s_branch[:, 0] = 0.0
    return PFResult(vm=np.abs(v), loading=_loading(feeder, s_branch), p_branch=s_branch.real)


def branch_flows_from_voltages(feeder: Feeder, vm: np.ndarray, va: np.ndarray) -> PFResult:
    """Branch flows from a voltage phasor field, as (v_parent - v) / z. va in radians.

    Exact for exact voltages, but a voltage error dV becomes a flow error dV / |z|;
    for surrogate predictions prefer branch_flows_kcl."""
    v = vm * np.exp(1j * va)
    vp = v[:, feeder.parent]
    z = np.where(feeder.z == 0, 1.0, feeder.z)  # branch 0 is a placeholder with z = 0
    s_branch = vp * np.conj((vp - v) / z[None, :])
    s_branch[:, 0] = 0.0
    return PFResult(vm=np.abs(v), loading=_loading(feeder, s_branch), p_branch=s_branch.real)
