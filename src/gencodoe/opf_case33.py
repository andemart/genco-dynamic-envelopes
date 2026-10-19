"""Stage A of the GENCO-OPF layer (docs/genco_opf_protocol.md, section 6): case33bw with 32 flexible units,
the synthetic scenario generator, and a per-scenario AC-OPF (PowerModels + IPOPT through juliacall).

Network: pandapower case33bw on a 1 MVA base, slack pinned at 1.0 pu, voltage limits 0.95-1.05 pu.
case33bw has no finite line ratings, so (protocol rule) each line is rated 1.5x its base-case flow.
Flexible unit at every load bus, cap = 1.5x the bus's nominal active load in both directions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import feeder as fdr

VMIN, VMAX = 0.95, 1.05
CAP_FACTOR = {"export": 3.0, "import": 1.5}   # per direction (protocol amendment 1: binding check on training days)
RATING_FACTOR = 1.5
N_DAYS, SPLIT = 120, {"train": range(0, 80), "val": range(80, 100), "test": range(100, 120)}
# fixed 24-hour shapes (hour 0..23)
LOAD_SHAPE = np.array([0.55, 0.50, 0.48, 0.47, 0.48, 0.55, 0.70, 0.85, 0.80, 0.72, 0.68, 0.66,
                       0.65, 0.64, 0.65, 0.70, 0.80, 0.95, 1.00, 0.98, 0.92, 0.82, 0.70, 0.60])
_h = np.arange(24)
PV_SHAPE = np.clip(np.sin(np.pi * (_h - 6) / 14), 0, None) ** 1.3          # 06:00-20:00, peak at 13:00


@dataclass
class Case33:
    fd: fdr.Feeder
    pd: np.ndarray          # nominal active load per bus, MW (= pu on 1 MVA)
    qd: np.ndarray
    unit_bus: np.ndarray    # buses with a flexible unit (all load buses)
    cap: dict               # direction -> per-unit cap, MW
    pv_bus: np.ndarray      # every second load bus
    pv_cap: np.ndarray


def build() -> Case33:
    import pandapower.networks as pn
    fd, pd, qd = fdr.from_pandapower(pn.case33bw(), base_mva=1.0)
    fd.vmin, fd.vmax, fd.v_slack = VMIN, VMAX, 1.0
    # case33bw has no finite ratings: rate each branch at 1.5x its base-case apparent flow
    # (sweep with unit ratings, so loading = |S| in MVA on the 1 MVA base)
    fd.rate = np.where(fd.parent < 0, np.inf, 1.0)
    s_branch = fdr.sweep_pf(fd, -pd[None, :], -qd[None, :]).loading[0]
    fd.rate = np.where(fd.parent < 0, np.inf, RATING_FACTOR * np.maximum(s_branch, 1e-6))
    unit_bus = np.flatnonzero(pd > 0)
    pv_bus = unit_bus[::2]
    return Case33(fd, pd, qd, unit_bus, {d: f * pd[unit_bus] for d, f in CAP_FACTOR.items()}, pv_bus, 1.0 * pd[pv_bus])


class Scenarios:
    """Protocol section 6: load = nominal x profile x lognormal day factor x copula noise; PV on every second
    load bus with a clear-sky shape x day cloud factor x noise. Deterministic per (seed, day)."""

    def __init__(self, case: Case33, seed: int = 0):
        self.c, self.seed = case, seed
        rng = np.random.default_rng(seed)
        self.day_factor = np.exp(rng.normal(0.0, 0.15, N_DAYS))
        self.cloud = rng.uniform(0.3, 1.0, N_DAYS)
        n = len(case.unit_bus)
        corr = 0.6 * np.ones((n, n)) + 0.4 * np.eye(n)
        self.L = np.linalg.cholesky(corr)

    def sample(self, day: int, hour: int, n: int, rng: np.random.Generator):
        """(p, q) net injections per bus, (n, n_bus) pu (consumption negative), PV included."""
        c = self.c
        z = rng.standard_normal((n, len(c.unit_bus))) @ self.L.T
        load = c.pd[c.unit_bus] * LOAD_SHAPE[hour] * self.day_factor[day] * (1 + 0.10 * z)
        pv = c.pv_cap * PV_SHAPE[hour] * self.cloud[day] * np.clip(1 + 0.20 * rng.standard_normal((n, len(c.pv_bus))), 0, None)
        p = np.zeros((n, c.fd.n_bus))
        q = np.zeros_like(p)
        p[:, c.unit_bus] = -load
        q[:, c.unit_bus] = -load * (c.qd[c.unit_bus] / c.pd[c.unit_bus])
        p[:, c.pv_bus] += pv
        return p, q


class OPF:
    """Per-scenario AC-OPF with flexible units (PowerModels.solve_ac_opf + IPOPT), one Julia process.
    Export: cost on slack import, units Pg in [0, cap] at zero cost. Import: units Pg in [-cap, 0] with cost +1
    (minimising sum Pg maximises import). Losses enter the export objective (disclosed in the protocol)."""

    def __init__(self, case: Case33, workdir: Path):
        from juliacall import Main as jl
        self.c, self.jl = case, jl
        jl.seval("using PowerModels, Ipopt; PowerModels.silence()")
        jl.seval('const SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)')
        self.data = {}
        workdir.mkdir(parents=True, exist_ok=True)
        for d in ("export", "import"):
            path = workdir / f"case33_units_{d}.m"
            path.write_text(self._matpower(d))
            jl.seval(f'DATA_{d} = PowerModels.parse_file("{path}")')
        # one Julia call per scenario: set every load's pd/qd from bus-indexed vectors
        jl.seval("""function set_loads!(data, pd, qd)
            for (k, l) in data["load"]; b = l["load_bus"]; l["pd"] = pd[b]; l["qd"] = qd[b]; end
        end""")
        jl.seval("""function solve_units(data, pd, qd)
            set_loads!(data, pd, qd)
            t = time()
            res = solve_ac_opf(data, SOLVER)
            dt = time() - t
            st = string(res["termination_status"])
            n = length(data["gen"])
            pg = st in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED") ?
                 [res["solution"]["gen"][string(i)]["pg"] for i in 2:n] : fill(NaN, n - 1)
            return pg, st, dt
        end""")
        self._solve = jl.seval("solve_units")

    def _matpower(self, direction: str) -> str:
        c, fd = self.c, self.c.fd
        L = ["function mpc = c", "mpc.version = '2';", "mpc.baseMVA = 1;", "mpc.bus = ["]
        for k in range(fd.n_bus):
            vmax, vmin = (1.0, 1.0) if fd.parent[k] < 0 else (VMAX, VMIN)   # slack pinned at 1.0 pu
            # nominal loads present at every load bus so PowerModels creates a load per bus (overwritten per scenario)
            L.append(f"{k+1} {3 if fd.parent[k] < 0 else 1} {max(c.pd[k], 0):.10g} {max(c.qd[k], 0):.10g} 0 0 1 1 0 12.66 1 {vmax} {vmin};")
        L += ["];", "mpc.gen = [", "1 0 0 100 -100 1 1 1 100 -100;"]
        for b, cap in zip(c.unit_bus, c.cap[direction]):
            pmax, pmin = (cap, 0.0) if direction == "export" else (0.0, -cap)
            L.append(f"{b+1} 0 0 {0.3*cap:.10g} {-0.3*cap:.10g} 1 1 1 {pmax:.10g} {pmin:.10g};")
        L += ["];", "mpc.branch = ["]
        for k in range(1, fd.n_bus):
            L.append(f"{fd.parent[k]+1} {k+1} {fd.r[k]:.10g} {fd.x[k]:.10g} 0 {fd.rate[k]:.10g} {fd.rate[k]:.10g} {fd.rate[k]:.10g} 0 0 1 -60 60;")
        L += ["];", "mpc.gencost = [", f"2 0 0 3 0 {1 if direction == 'export' else 0} 0;"]
        L += [f"2 0 0 3 0 {0 if direction == 'export' else 1} 0;" for _ in c.unit_bus] + ["];"]
        return "\n".join(L)

    def solve(self, p: np.ndarray, q: np.ndarray, direction: str):
        """One scenario (bus injections p, q in pu, consumption negative).
        Returns (unit Pg MW, NaN if not solved; ok; IPOPT solve seconds; total seconds incl. data transfer)."""
        t = time.perf_counter()
        pg, st, dt = self._solve(self.jl.seval(f"DATA_{direction}"), -np.asarray(p, float), -np.asarray(q, float))
        pg = np.asarray(pg, float)
        ok = str(st) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
        return pg, ok, float(dt), time.perf_counter() - t
