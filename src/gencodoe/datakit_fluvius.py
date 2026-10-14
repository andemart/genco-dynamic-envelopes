"""GENCO PF training data for ENWL1 driven by real Fluvius meter profiles.

datakit's own load generators scale one aggregate profile per bus. We replace
it with a generator that builds every scenario from real meters:

  1. draw a 15-min step t from the training days (every 5th day is held out
     for testing);
  2. draw a technology mix: shares of customers with heat pump, EV and PV,
     each uniform in [0, 1], so training covers low to full penetration;
  3. give each of the 200 customers a group from that mix and a random meter
     of that group (no meter used twice), and read its net kW at step t.
     All customers share the same calendar step, which keeps weather and
     coincident peaks.

Offtake has PF 0.95 (lagging); injection is at unity power factor.
The per-scenario draw (step, mix, meters) is saved next to datakit's output so
test states can be rebuilt.

Usage: python -m gencodoe.datakit_fluvius --scenarios 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .enwl import load_enwl1, write_matpower
from .fluvius import GROUPS, STEPS, load_group

ROOT = Path(__file__).resolve().parents[2]
NAME = "enwl1_fluvius"
TAN_PHI = np.tan(np.arccos(0.95))


def is_test_step(step: np.ndarray) -> np.ndarray:
    return (np.asarray(step) // 96) % 5 == 4


class FluviusScenarios:
    """datakit load-scenario generator: returns (n_bus, n_scenarios, 2) in MW / MVAr."""

    def __init__(self, customer_bus: np.ndarray, meta_path: Path):
        self.customer_bus = customer_bus
        self.meta_path = meta_path

    def __call__(self, net, n_scenarios, scenario_log, max_iter, seed):
        rng = np.random.default_rng(seed)
        groups = {name: load_group(name) for name in GROUPS}
        by_flags = {(g.hp, g.ev, g.pv): g for g in groups.values()}
        n_cust = len(self.customer_bus)
        train_steps = np.flatnonzero(~is_test_step(np.arange(STEPS)))

        out = np.zeros((net.Pd.size, n_scenarios, 2))
        rows = []
        for s in range(n_scenarios):
            t = int(rng.choice(train_steps))
            mix = rng.uniform(0, 1, 3)
            flags = rng.uniform(size=(n_cust, 3)) < mix
            p_kw = np.empty(n_cust)
            q_kvar = np.empty(n_cust)
            meters = np.empty(n_cust, dtype=np.int64)
            for key in {tuple(int(v) for v in f) for f in flags}:
                g = by_flags[key]
                who = np.flatnonzero((flags == key).all(axis=1))
                pick = rng.choice(len(g.ean), size=len(who), replace=False)
                off, inj = g.offtake_kw[pick, t], g.injection_kw[pick, t]
                p_kw[who], q_kvar[who], meters[who] = off - inj, off * TAN_PHI, g.ean[pick]
            np.add.at(out[:, s, 0], self.customer_bus, p_kw / 1e3)
            np.add.at(out[:, s, 1], self.customer_bus, q_kvar / 1e3)
            rows.append({"scenario": s, "step": t, "time": groups["geen_ZP"].time[t],
                         "share_hp": mix[0], "share_ev": mix[1], "share_pv": mix[2],
                         "meters": meters.tolist(), "groups": flags.astype(np.int8).tolist()})
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(self.meta_path)
        return out


def config(scenarios: int, processes: int, data_dir: Path) -> dict:
    return {
        "network": {"name": NAME, "source": "file", "network_dir": str(ROOT / "grids")},
        "load": {"generator": "fluvius", "scenarios": scenarios},
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {"num_processes": processes, "data_dir": str(data_dir), "large_chunk_size": 1000,
                     "overwrite": True, "mode": "pf", "include_dc_res": False, "enable_solver_logs": False,
                     "pf_fast": True, "dcpf_fast": True, "max_iter": 200, "seed": 42},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=20000)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data_out" / "datakit")
    a = ap.parse_args()

    import gridfm_datakit.generate as gen

    net = load_enwl1()
    write_matpower(net, ROOT / "grids" / f"{NAME}.m")
    meta = a.data_dir / NAME / "raw" / "fluvius_draws.parquet"
    gen.get_load_scenario_generator = lambda _args: FluviusScenarios(net.customer_bus, meta)
    paths = gen.generate_power_flow_data_distributed(config(a.scenarios, a.processes, a.data_dir))
    print(paths["bus_data"])


if __name__ == "__main__":
    main()
