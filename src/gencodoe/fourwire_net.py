"""Four-wire (a, b, c, n) ENWL1 network and physics prior (docs/unbalanced_genco_protocol.md).

Topology: full 4x4 R and X per line code, no Kron reduction, rooted at the transformer secondary
`trbus`. Leaves without customers are pruned and series chains without retained taps are merged by
adding their 4x4 impedance matrices (exact: ENWL1 lines have zero capacitance). The keep-set equals the
balanced reduction's (customers + trbus), so the retained buses are the balanced network's minus
`sourcebus`.

Prior: backward/forward sweep with customers as constant-power phase-to-neutral loads,
I = conj(S / (V_phase - V_neutral)), and the source + Dyn transformer as a fixed 3x3 phase-domain
Thevenin equivalent at the root (neutral grounded there).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .enwl import ENWL1_DIR

V_PN_NOM = 416.0 / np.sqrt(3) / 1e3          # kV phase-to-neutral base (matches limits.V_BASE)
ROOT = "trbus"
CONDUCTORS = ("a", "b", "c", "n")


def _matrix(text: str) -> np.ndarray:
    rows = [r.split() for r in text.strip().strip("[]").split("|")]
    return np.array([[float(x) for x in r] for r in rows])


def read_linecodes_4x4(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    """name -> (Z 4x4 complex in ohm/km, NormAmps)."""
    text = path.read_text()
    out = {}
    for block in re.split(r"(?=New LineCode\.)", text):
        m = re.match(r"New LineCode\.(\w+).*?NormAmps=([\d.]+)", block, re.S | re.I)
        if not m:
            continue
        r = _matrix(re.search(r"rmatrix=\s*(\[[^\]]*\])", block).group(1))
        x = _matrix(re.search(r"xmatrix=\s*(\[[^\]]*\])", block).group(1))
        assert r.shape == x.shape == (4, 4), m.group(1)
        out[m.group(1).lower()] = (r + 1j * x, float(m.group(2)))
    return out


@dataclass
class FourWireNet:
    bus_names: list[str]                 # retained buses; index 0 is the root (trbus)
    parent: np.ndarray                   # (n_bus,) parent index, -1 at the root
    order: np.ndarray                    # topological order root -> leaves
    z: np.ndarray                        # (n_bus, 4, 4) complex ohm of the branch into each bus (zeros at root)
    rating_a: np.ndarray                 # (n_bus,) min NormAmps along the merged chain (inf at root)
    length_m: np.ndarray                 # (n_bus,) total length of the merged chain
    first_line: list[str | None]         # OpenDSS line whose terminal current is the branch current
    first_line_parent_terminal: np.ndarray   # 1 or 2: the terminal of first_line on the parent side
    customer_names: list[str]            # Loads.dss order
    customer_bus: np.ndarray             # (n_c,) retained-bus index
    customer_phase: np.ndarray           # (n_c,) 0, 1, 2 for a, b, c
    depth: np.ndarray = field(default=None)

    @property
    def n_bus(self) -> int:
        return len(self.bus_names)

    def node_names(self) -> list[str]:
        """OpenDSS node names in (bus, conductor) order: bus.1 .. bus.4."""
        return [f"{b}.{k}" for b in self.bus_names for k in (1, 2, 3, 4)]


def load_fourwire(folder: Path = ENWL1_DIR) -> FourWireNet:
    codes = read_linecodes_4x4(folder / "LineCode_3f4w.dss")
    line_re = re.compile(r"New\s+Line\.(\w+)\s+Bus1=(\w+)\.1\.2\.3\.4\s+Bus2=(\w+)\.1\.2\.3\.4.*?Linecode=(\w+)\s+"
                         r"Length=([\d.eE+-]+)\s+Units=(\w+)", re.I)
    edges = []                                   # (bus1, bus2, Z 4x4, amps, length m, line name)
    for line in (folder / "Lines.dss").read_text().splitlines():
        m = line_re.search(line)
        if not m:
            continue
        name, b1, b2, code, length, units = m.groups()
        meters = float(length) * {"m": 1.0, "km": 1e3}[units.lower()]
        zkm, amps = codes[code.lower()]
        edges.append((b1.lower(), b2.lower(), zkm * meters / 1e3, amps, meters, name))
    load_re = re.compile(r"New Load\.(\w+).*?Bus1=(\w+)\.([123])\.4", re.I)
    cust = [(m.group(1), m.group(2).lower(), int(m.group(3)) - 1)
            for m in map(load_re.search, (folder / "Loads.dss").read_text().splitlines()) if m]
    keep = {b for _, b, _ in cust} | {ROOT}

    # adjacency, then orient from the root
    adj: dict[str, list[int]] = {}
    for i, (b1, b2, *_ ) in enumerate(edges):
        adj.setdefault(b1, []).append(i)
        adj.setdefault(b2, []).append(i)
    par_edge, par, order = {ROOT: None}, {ROOT: None}, [ROOT]
    k = 0
    while k < len(order):
        u = order[k]
        k += 1
        for i in adj.get(u, []):
            b1, b2 = edges[i][:2]
            v = b2 if b1 == u else b1
            if v not in par:
                par[v], par_edge[v] = u, i
                order.append(v)
    if len(order) != len(adj):
        raise ValueError("network is not a single radial tree rooted at trbus")
    children: dict[str, list[str]] = {}
    for v in order[1:]:
        children.setdefault(par[v], []).append(v)

    # prune subtrees without customers
    has = {}
    for v in reversed(order):
        has[v] = v in keep or any(has[c] for c in children.get(v, []))
    alive = [v for v in order if has[v]]
    # retained buses: keep-set, plus every bus with >= 2 alive children (taps)
    alive_children = {v: [c for c in children.get(v, []) if has[c]] for v in alive}
    retained = [v for v in alive if v in keep or len(alive_children[v]) >= 2]
    rset = set(retained)
    idx = {v: i for i, v in enumerate(retained)}
    n = len(retained)
    parent = np.full(n, -1)
    z = np.zeros((n, 4, 4), complex)
    rating = np.full(n, np.inf)
    length = np.zeros(n)
    first_line, first_term = [None] * n, np.zeros(n, int)
    for v in retained[1:]:
        u, zsum, amps, meters, chain = v, np.zeros((4, 4), complex), np.inf, 0.0, []
        while True:                               # climb to the nearest retained ancestor
            e = edges[par_edge[u]]
            zsum += e[2]
            amps = min(amps, e[3])
            meters += e[4]
            chain.append((e, par[u]))
            u = par[u]
            if u in rset:
                break
        i = idx[v]
        parent[i], z[i], rating[i], length[i] = idx[u], zsum, amps, meters
        e_top, upper = chain[-1]                  # segment adjacent to the retained parent
        first_line[i] = e_top[5]
        first_term[i] = 1 if e_top[0] == upper else 2
    ordr = np.array([idx[v] for v in retained])  # retained is already in BFS (root-first) order
    depth = np.zeros(n, int)
    for i in ordr[1:]:
        depth[i] = depth[parent[i]] + 1
    return FourWireNet(bus_names=retained, parent=parent, order=ordr, z=z, rating_a=rating, length_m=length,
                       first_line=first_line, first_line_parent_terminal=first_term,
                       customer_names=[c for c, _, _ in cust], customer_bus=np.array([idx[b] for _, b, _ in cust]),
                       customer_phase=np.array([ph for _, _, ph in cust]), depth=depth)


# ------------------------------------------------------------------------------------------- sweep
@dataclass
class Thevenin:
    """Source + transformer seen from the LV root: V_abc = E - Zth @ I_abc (kV, kA, ohm)."""
    e: np.ndarray        # (3,) complex kV phase-to-ground
    zth: np.ndarray      # (3, 3) complex ohm


def customer_currents(net: FourWireNet, v: np.ndarray, s_kva: np.ndarray) -> np.ndarray:
    """Per-bus conductor injections (kA, flowing out of the network into the customer) for node voltages
    v (B, n_bus, 4) kV and customer complex consumption s_kva (B, n_c) kVA. Returns (B, n_bus, 4)."""
    b = v.shape[0]
    vb = v[:, net.customer_bus]                                   # (B, n_c, 4)
    vph = np.take_along_axis(vb, net.customer_phase[None, :, None].repeat(b, 0), axis=2)[..., 0]
    vpn = vph - vb[..., 3]
    i_c = np.conj(s_kva / 1e3 / np.where(np.abs(vpn) > 1e-9, vpn, 1e-9))   # kA (S in MVA / kV)
    out = np.zeros((b, net.n_bus, 4), complex)
    np.add.at(out, (slice(None), net.customer_bus, net.customer_phase), i_c)
    np.add.at(out, (slice(None), net.customer_bus, 3), -i_c)     # return current on the neutral
    return out


def branch_currents_kcl(net: FourWireNet, inj: np.ndarray) -> np.ndarray:
    """Branch current into each bus (B, n_bus, 4) = sum of injections in its subtree (root row = total)."""
    cur = inj.copy()
    for i in net.order[::-1][:-1]:
        cur[:, net.parent[i]] += cur[:, i]
    return cur


def sweep(net: FourWireNet, th: Thevenin, s_kva: np.ndarray, iterations: int = 1, tol: float = 1e-8,
          v0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Backward/forward sweep. s_kva (B, n_c) complex customer consumption. Returns node voltages
    (B, n_bus, 4) kV, branch currents (B, n_bus, 4) kA, iterations used. iterations=1 is the U1 prior."""
    b = s_kva.shape[0]
    if v0 is None:
        v = np.zeros((b, net.n_bus, 4), complex)
        v[:, :, :3] = th.e[None, None, :]
    else:
        v = v0.copy()
    it = 0
    for it in range(1, iterations + 1):
        cur = branch_currents_kcl(net, customer_currents(net, v, s_kva))
        vn = np.empty_like(v)
        vn[:, 0, :3] = th.e[None, :] - cur[:, 0, :3] @ th.zth.T
        vn[:, 0, 3] = 0.0
        for i in net.order[1:]:
            vn[:, i] = vn[:, net.parent[i]] - cur[:, i] @ net.z[i].T
        delta = np.abs(vn - v).max() / V_PN_NOM
        v = vn
        if delta < tol:
            break
    cur = branch_currents_kcl(net, customer_currents(net, v, s_kva))
    return v, cur, it


def phase_neutral_pu(net: FourWireNet, v: np.ndarray) -> np.ndarray:
    """Customer phase-to-neutral voltage magnitude in pu of 240.18 V, (B, n_c)."""
    vb = v[:, net.customer_bus]
    vph = np.take_along_axis(vb, net.customer_phase[None, :, None].repeat(v.shape[0], 0), axis=2)[..., 0]
    return np.abs(vph - vb[..., 3]) / V_PN_NOM


def fit_thevenin(v_root_abc: np.ndarray, i_root_abc: np.ndarray) -> Thevenin:
    """Least-squares E and Zth from root voltages (S, 3) and outgoing root currents (S, 3)."""
    a = np.hstack([np.ones((len(v_root_abc), 1)), -i_root_abc])        # [1, -I] @ [E; Zth^T] = V
    sol, *_ = np.linalg.lstsq(a, v_root_abc, rcond=None)
    return Thevenin(e=sol[0], zth=sol[1:].T)
