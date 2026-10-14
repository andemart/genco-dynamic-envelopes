"""ENWL1 LV feeder (OpenDSS, four-wire) -> balanced positive-sequence model.

The OpenDSS model has one 800 kVA 11/0.416 kV transformer, ~3,140 short
four-wire line segments and 200 single-phase customers. GENCO is a balanced
solver, so we reduce it to a positive-sequence radial network:

  * line impedance: Kron-reduce the neutral out of the 4x4 linecode matrix,
    then z1 = mean(self) - mean(mutual) of the 3x3 phase matrix;
  * thermal rating: NormAmps of the linecode, as 3-phase MVA at 0.416 kV;
  * customers: each single-phase load is lumped as a balanced load at its bus;
  * dead-end segments without customers are dropped, and series chains with
    no customer and no branching are merged (impedances add, rating = the
    smallest in the chain). Voltages stay exact for a balanced radial network
    without shunts. The current is the same along a chain, but MVA loading is
    taken at the chain's sending end, so it differs from the worst segment by
    the voltage ratio along the chain (<1% on ENWL1, in either direction).

Bus 0 is the 11 kV source (slack), bus 1 the transformer LV terminal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .feeder import Feeder

ENWL1_DIR = Path(__file__).resolve().parents[2] / "data" / "feeder_topology" / "ENWL1"
KV_LV = 0.416
KV_MV = 11.0
TR_KVA = 800.0
TR_R_PCT = 0.4   # OpenDSS default %R = 0.2 per winding (Master.dss sets none)
TR_X_PCT = 1.0   # XHL=1 in Master.dss


@dataclass
class Network:
    """Reduced balanced network in physical units (ohm, MVA)."""
    bus_names: list[str]        # bus 0 = source, bus 1 = transformer LV terminal
    from_bus: np.ndarray        # (n_branch,)
    to_bus: np.ndarray
    r_ohm: np.ndarray           # referred to the LV side for every branch
    x_ohm: np.ndarray
    rate_mva: np.ndarray
    customer_bus: np.ndarray    # (200,) bus index of each customer, in Loads.dss order
    customer_names: list[str]

    @property
    def n_bus(self) -> int:
        return len(self.bus_names)

    def to_feeder(self, base_mva: float = 1.0, **kw) -> Feeder:
        zbase = KV_LV**2 / base_mva
        return Feeder.from_branches(self.from_bus, self.to_bus, self.r_ohm / zbase, self.x_ohm / zbase,
                                    self.rate_mva / base_mva, base_mva=base_mva, **kw)


def _matrix(text: str) -> np.ndarray:
    rows = [r.split() for r in text.split("|")]
    return np.array(rows, dtype=float)


def read_linecodes(path: Path) -> dict[str, tuple[complex, float]]:
    """Linecode name -> (positive-sequence z in ohm/km, NormAmps)."""
    text = path.read_text()
    out = {}
    for block in re.split(r"(?=New LineCode\.)", text):
        m = re.match(r"New LineCode\.(\w+).*?NormAmps=([\d.]+)", block)
        if not m:
            continue
        r = _matrix(re.search(r"rmatrix=\s*\[(.*?)\]", block).group(1))
        x = _matrix(re.search(r"xmatrix=\s*\[(.*?)\]", block).group(1))
        z = r + 1j * x
        zp = z[:3, :3] - np.outer(z[:3, 3], z[3, :3]) / z[3, 3]   # Kron-reduce the neutral
        z1 = np.mean(np.diag(zp)) - np.mean(zp[~np.eye(3, dtype=bool)])
        out[m.group(1).lower()] = (complex(z1), float(m.group(2)))
    return out


def load_enwl1(folder: Path = ENWL1_DIR, reduce: bool = True) -> Network:
    codes = read_linecodes(folder / "LineCode_3f4w.dss")
    line_re = re.compile(r"Bus1=([\w]+)\S*\s+Bus2=([\w]+)\S*.*?Linecode=(\w+)\s+Length=([\d.eE+-]+)\s+Units=(\w+)", re.I)
    edges = []
    for line in (folder / "Lines.dss").read_text().splitlines():
        m = line_re.search(line)
        if not m:
            continue
        b1, b2, code, length, units = m.groups()
        km = float(length) * {"m": 1e-3, "km": 1.0}[units.lower()]
        z1, amps = codes[code.lower()]
        edges.append((b1.lower(), b2.lower(), z1 * km, np.sqrt(3) * KV_LV * amps / 1e3))

    load_re = re.compile(r"New Load\.(\w+).*?Bus1=([\w]+)", re.I)
    customers = [(m.group(1), m.group(2).lower()) for m in map(load_re.search, (folder / "Loads.dss").read_text().splitlines()) if m]

    # transformer, referred to the LV side
    zbase_lv = KV_LV**2 / (TR_KVA / 1e3)
    edges.insert(0, ("sourcebus", "trbus", (TR_R_PCT + 1j * TR_X_PCT) / 100 * zbase_lv, TR_KVA / 1e3))

    if reduce:
        keep = {b for _, b in customers} | {"sourcebus", "trbus"}
        edges = _merge_series(_prune_leaves(edges, keep), keep)

    names = ["sourcebus", "trbus"]
    names += sorted({b for e in edges for b in e[:2]} - set(names))
    idx = {b: i for i, b in enumerate(names)}
    f = np.array([idx[e[0]] for e in edges])
    t = np.array([idx[e[1]] for e in edges])
    z = np.array([e[2] for e in edges])
    return Network(bus_names=names, from_bus=f, to_bus=t, r_ohm=z.real, x_ohm=z.imag,
                   rate_mva=np.array([e[3] for e in edges]),
                   customer_bus=np.array([idx[b] for _, b in customers]),
                   customer_names=[c for c, _ in customers])


def _prune_leaves(edges, keep: set[str]):
    """Drop dead-end segments that lead to no customer (they carry no current)."""
    edges = list(edges)
    while True:
        deg: dict[str, int] = {}
        for a, b, _, _ in edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        kept = [e for e in edges if not any(deg[n] == 1 and n not in keep for n in e[:2])]
        if len(kept) == len(edges):
            return edges
        edges = kept


def _merge_series(edges, keep: set[str]):
    """Merge chains through buses of degree 2 that are not in `keep`."""
    adj: dict[str, list[int]] = {}
    for k, (a, b, _, _) in enumerate(edges):
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)
    alive = dict(enumerate(edges))
    for bus, ks in adj.items():
        if bus in keep or len(ks) != 2:
            continue
        k1, k2 = ks
        a1, b1, z1, r1 = alive.pop(k1)
        a2, b2, z2, r2 = alive.pop(k2)
        u = a1 if b1 == bus else b1
        w = a2 if b2 == bus else b2
        alive[k1] = (u, w, z1 + z2, min(r1, r2))
        adj[w] = [k1 if k == k2 else k for k in adj[w]]
    return list(alive.values())


def write_matpower(net: Network, path: Path, base_mva: float = 1.0, vmin: float = 0.5, vmax: float = 1.5) -> Path:
    """Write the reduced network as a MATPOWER case for gridfm-datakit.

    Bus loads are zero placeholders (the scenario generator sets them). Limits
    are loose on purpose: datakit's PF mode first solves an OPF, and real
    ratings and voltage bounds would make the overloaded and over/undervoltage
    states we need for training infeasible. Real ratings are kept in
    Network.rate_mva and checked from GENCO's predicted voltages.
    Branch impedances are referred to the LV base, so the transformer is
    modelled as a line with ratio 0 and bus 0 also has base 0.416 kV.
    """
    zbase = KV_LV**2 / base_mva
    lines = [f"function mpc = {path.stem}", "mpc.version = '2';", f"mpc.baseMVA = {base_mva};", "mpc.bus = ["]
    for i in range(net.n_bus):
        # the source bus is pinned at 1 pu, else the OPF raises it to vmax to cut losses
        btype, hi, lo = (3, 1.0, 1.0) if i == 0 else (1, vmax, vmin)
        lines.append(f"\t{i + 1}\t{btype}\t0\t0\t0\t0\t1\t1.0\t0\t{KV_LV}\t1\t{hi}\t{lo};")
    lines += ["];", "mpc.gen = ["]
    big = 100 * base_mva
    lines.append(f"\t1\t0\t0\t{big}\t{-big}\t1.0\t{base_mva}\t1\t{big}\t{-big}\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;")
    lines += ["];", "mpc.branch = ["]
    for f, t, r, x in zip(net.from_bus, net.to_bus, net.r_ohm / zbase, net.x_ohm / zbase):
        lines.append(f"\t{f + 1}\t{t + 1}\t{r:.10g}\t{x:.10g}\t0\t0\t0\t0\t0\t0\t1\t-360\t360;")
    lines += ["];", "mpc.gencost = [", "\t2\t0\t0\t3\t0\t1\t0;", "];", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
