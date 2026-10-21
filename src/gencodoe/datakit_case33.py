"""GENCO PF training states for case33bw (docs/genco_opf_protocol.md, amendment 3): the stage A scenario generator
plugged into gridfm-datakit, so the actual GENCO (graphkit) can be trained in PF mode on this network.

    python -m gencodoe.datakit_case33 --scenarios 20000 --processes 8        # days 0-99 -> data_out/datakit_case33
    python -m gencodoe.datakit_case33 --split gate --scenarios 2000          # val days only -> data_out/datakit_case33_gate

Each state: one (day, hour) uniform over the split's days and 24 hours, one scenario draw from opf_case33.Scenarios
(load profile x day factor x copula noise, PV on every second load bus); with probability `augment` an envelope
injection row on top: direction export or import (50/50), every flexible unit at one level uniform in [0, cap].
Splits by day as in stage A: 0-79 train (weights, normalizer), 80-99 val (early stopping); test days 100-119 are
never drawn. The draw log (day, hour, direction, level) is saved next to datakit's output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import opf_case33 as oc

ROOT = Path(__file__).resolve().parents[2]
NAME = "case33bw_opf"
KV = 12.66
SEED = {"trainval": 42 + 3000, "gate": 42 + 4000}
DAYS = {"trainval": range(0, 100), "gate": range(80, 100)}


def write_matpower(case: oc.Case33, path: Path) -> Path:
    """case33bw for datakit: zero load placeholders, loose limits (0.5-1.5 pu, no ratings), slack pinned at 1.0 pu.
    Stage A limits and ratings are applied by the exact check only (as for ENWL1, enwl.write_matpower)."""
    fd = case.fd
    L = [f"function mpc = {path.stem}", "mpc.version = '2';", "mpc.baseMVA = 1;", "mpc.bus = ["]
    for k in range(fd.n_bus):
        btype, hi, lo = (3, 1.0, 1.0) if fd.parent[k] < 0 else (1, 1.5, 0.5)
        L.append(f"\t{k + 1}\t{btype}\t0\t0\t0\t0\t1\t1.0\t0\t{KV}\t1\t{hi}\t{lo};")
    L += ["];", "mpc.gen = [", "\t1\t0\t0\t100\t-100\t1.0\t1\t1\t100\t-100\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;", "];", "mpc.branch = ["]
    for k in range(1, fd.n_bus):
        L.append(f"\t{fd.parent[k] + 1}\t{k + 1}\t{fd.r[k]:.10g}\t{fd.x[k]:.10g}\t0\t0\t0\t0\t0\t0\t1\t-360\t360;")
    L += ["];", "mpc.gencost = [", "\t2\t0\t0\t3\t0\t1\t0;", "];", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path


class Case33States:
    """datakit load-scenario generator: (n_bus, n_scenarios, 2) in MW / MVAr, positive = consumption."""

    def __init__(self, case: oc.Case33, meta_path: Path, split: str, augment: float):
        self.case, self.meta_path, self.split, self.augment = case, meta_path, split, augment

    def __call__(self, net, n_scenarios, scenario_log, max_iter, seed):
        c = self.case
        if net.Pd.size != c.fd.n_bus:
            raise ValueError(f"datakit network has {net.Pd.size} buses, case has {c.fd.n_bus}")
        rng, rng_flex = np.random.default_rng(seed), np.random.default_rng(seed + 7)
        sc = oc.Scenarios(c, 0)
        days = np.asarray(DAYS[self.split])
        out = np.zeros((c.fd.n_bus, n_scenarios, 2))
        rows = []
        inc = np.zeros(c.fd.n_bus)
        for s in range(n_scenarios):
            day, hour = int(rng.choice(days)), int(rng.integers(24))
            p, q = sc.sample(day, hour, 1, rng)
            direction, level = "none", 0.0
            if self.augment > 0 and rng_flex.uniform() < self.augment:
                direction = "export" if rng_flex.uniform() < 0.5 else "import"
                level = float(rng_flex.uniform())                              # share of cap, same for every unit
                inc[:] = 0.0
                inc[c.unit_bus] = level * c.cap[direction]
                p = p + (inc if direction == "export" else -inc)
            out[:, s, 0], out[:, s, 1] = -p[0], -q[0]                          # injections -> consumption, MW on 1 MVA
            rows.append({"scenario": s, "day": day, "hour": hour, "direction": direction, "level": level})
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(self.meta_path)
        return out


def config(scenarios: int, processes: int, data_dir: Path, seed: int) -> dict:
    return {
        "network": {"name": NAME, "source": "file", "network_dir": str(ROOT / "grids")},
        "load": {"generator": "case33", "scenarios": scenarios},
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {"num_processes": processes, "data_dir": str(data_dir), "large_chunk_size": 1000,
                     "overwrite": True, "mode": "pf", "include_dc_res": False, "enable_solver_logs": False,
                     "pf_fast": True, "dcpf_fast": True, "max_iter": 200, "seed": seed},
    }


def solved_draws(draws: Path) -> pd.DataFrame:
    """Draw log joined to datakit's (renumbered) scenario ids; one row per solved state."""
    d = pd.read_parquet(draws).rename(columns={"scenario": "draw"})
    bus = pd.read_parquet(Path(draws).parent / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
    ids = bus.drop_duplicates("scenario").rename(columns={"load_scenario_idx": "draw"})
    return ids.merge(d, on="draw", how="left", validate="one_to_one").sort_values("scenario").reset_index(drop=True)


def write_graphkit_splits(draws: Path, out_dir: Path) -> dict[str, int]:
    """train.pt / val.pt / test.pt for graphkit from the draw log: days 0-79 train, 80-99 val (also 'test')."""
    import torch
    d = solved_draws(draws)
    if (d.day >= 100).any():
        raise ValueError("training draws contain test days")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, sel in (("train", d.day < 80), ("val", d.day >= 80), ("test", d.day >= 80)):
        ids = torch.tensor(np.sort(d.scenario.to_numpy()[sel.to_numpy()]), dtype=torch.long)
        torch.save(ids, out_dir / f"{split}.pt")
        sizes[split] = len(ids)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=20000)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--split", choices=list(SEED), default="trainval")
    ap.add_argument("--augment", type=float, default=0.5)
    ap.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.data_dir is None:
        a.data_dir = ROOT / "data_out" / ("datakit_case33" if a.split == "trainval" else f"datakit_case33_{a.split}")
    import gridfm_datakit.generate as gen
    case = oc.build()
    write_matpower(case, ROOT / "grids" / f"{NAME}.m")
    meta = a.data_dir / NAME / "raw" / "case33_draws.parquet"
    gen.get_load_scenario_generator = lambda _args: Case33States(case, meta, a.split, a.augment)
    paths = gen.generate_power_flow_data_distributed(config(a.scenarios, a.processes, a.data_dir, SEED[a.split]))
    if a.split == "trainval":
        print("splits", write_graphkit_splits(meta, a.data_dir / "splits"))
    print(f"solved {len(solved_draws(meta))} of {a.scenarios} states")
    print(paths["bus_data"])


if __name__ == "__main__":
    main()
