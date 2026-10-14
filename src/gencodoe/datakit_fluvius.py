"""GENCO PF training data for ENWL1 driven by real Fluvius meter profiles.

datakit's own load generators scale one aggregate profile per bus. We replace
it with a generator that builds every scenario from real meters:

  1. draw a 15-min step t from the days of the requested split (day_split:
     training/validation days for the training set, or the held-out
     calibration or test days);
  2. draw a technology mix: shares of customers with heat pump, EV and PV,
     each uniform in [0, 1], so training covers low to full penetration;
  3. give each of the 200 customers a group from that mix and a random meter
     of that group (no meter used twice), and read its net kW at step t.
     All customers share the same calendar step, which keeps weather and
     coincident peaks.

Offtake has PF 0.95 (lagging); injection is at unity power factor.

Envelope augmentation (augment > 0): the DOE search evaluates states where
every customer adds a flexible injection at one candidate level on top of the
passive meter state, which pushes the feeder into overvoltage, reverse-flow
and transformer overload that real meter states rarely reach. With probability
`augment` a state gets such a row: direction export or import (50/50), level
uniform in [0, cap] for all customers, unity power factor, with caps
FLEX_EXPORT_KW (single-phase inverter) and FLEX_IMPORT_KW (32 A single-phase
charger). These draws use their own random stream, so augment=0 reproduces
the original dataset exactly.
The per-scenario draw (step, mix, meters) is saved next to datakit's output so
test states can be rebuilt.

Calendar-day splits (day of year d = step // 96, 2024 has 366 days):
  d % 5 in {0, 1, 2}  train   (fit weights and the normalizer)
  d % 5 == 3          val     (early stopping and model selection)
  d % 5 == 4          held out: calib if (d // 5) is even, else test
The training set is drawn from train + val days; write_graphkit_splits() then
assigns its scenarios by day, so no held-out day reaches training.

Usage:
  python -m gencodoe.datakit_fluvius --scenarios 20000              # train+val days
  python -m gencodoe.datakit_fluvius --split calib --scenarios 3000
  python -m gencodoe.datakit_fluvius --split test --scenarios 3000
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
FLEX_EXPORT_KW = 5.0   # assumption: single-phase inverter export limit
FLEX_IMPORT_KW = 7.4   # assumption: 32 A single-phase EV charger


SPLITS = ("train", "val", "calib", "test")
SEED_OFFSET = {"trainval": 0, "calib": 1000, "test": 2000}


def day_split(step: np.ndarray) -> np.ndarray:
    """Split name of each 15-min step, by calendar day."""
    day = np.asarray(step) // 96
    out = np.where(day % 5 == 3, "val", "train").astype(object)
    held = day % 5 == 4
    out[held & ((day // 5) % 2 == 0)] = "calib"
    out[held & ((day // 5) % 2 == 1)] = "test"
    return out


def is_test_step(step: np.ndarray) -> np.ndarray:
    """True for held-out days (calib or test), which the training set never uses."""
    return (np.asarray(step) // 96) % 5 == 4


def split_steps(split: str) -> np.ndarray:
    """Steps a dataset may draw from: 'trainval' (train + val days), 'calib' or 'test'."""
    names = day_split(np.arange(STEPS))
    if split == "trainval":
        return np.flatnonzero((names == "train") | (names == "val"))
    if split in ("calib", "test"):
        return np.flatnonzero(names == split)
    raise ValueError(f"unknown split {split!r}")


class FluviusScenarios:
    """datakit load-scenario generator: returns (n_bus, n_scenarios, 2) in MW / MVAr."""

    def __init__(self, customer_bus: np.ndarray, meta_path: Path, split: str = "trainval", augment: float = 0.0):
        self.customer_bus = customer_bus
        self.meta_path = meta_path
        self.split = split
        self.augment = augment

    def __call__(self, net, n_scenarios, scenario_log, max_iter, seed):
        rng = np.random.default_rng(seed)
        rng_flex = np.random.default_rng(seed + 7)
        groups = {name: load_group(name) for name in GROUPS}
        by_flags = {(g.hp, g.ev, g.pv): g for g in groups.values()}
        n_cust = len(self.customer_bus)
        train_steps = split_steps(self.split)

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
            flex_kw = 0.0   # signed flexible injection per customer, + = export
            if self.augment > 0 and rng_flex.uniform() < self.augment:
                if rng_flex.uniform() < 0.5:
                    flex_kw = rng_flex.uniform(0, FLEX_EXPORT_KW)
                else:
                    flex_kw = -rng_flex.uniform(0, FLEX_IMPORT_KW)
                p_kw = p_kw - flex_kw
            np.add.at(out[:, s, 0], self.customer_bus, p_kw / 1e3)
            np.add.at(out[:, s, 1], self.customer_bus, q_kvar / 1e3)
            rows.append({"scenario": s, "step": t, "time": groups["geen_ZP"].time[t], "split": day_split(t).item(),
                         "flex_kw": flex_kw, "share_hp": mix[0], "share_ev": mix[1], "share_pv": mix[2],
                         "meters": meters.tolist(), "groups": flags.astype(np.int8).tolist()})
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(self.meta_path)
        return out


def config(scenarios: int, processes: int, data_dir: Path, seed: int = 42) -> dict:
    return {
        "network": {"name": NAME, "source": "file", "network_dir": str(ROOT / "grids")},
        "load": {"generator": "fluvius", "scenarios": scenarios},
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {"num_processes": processes, "data_dir": str(data_dir), "large_chunk_size": 1000,
                     "overwrite": True, "mode": "pf", "include_dc_res": False, "enable_solver_logs": False,
                     "pf_fast": True, "dcpf_fast": True, "max_iter": 200, "seed": seed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=20000)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--split", choices=list(SEED_OFFSET), default="trainval")
    ap.add_argument("--augment", type=float, default=0.0, help="share of states with an envelope injection row")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="default: data_out/datakit (trainval) or data_out/datakit_<split>")
    a = ap.parse_args()
    if a.data_dir is None:
        a.data_dir = ROOT / "data_out" / ("datakit" if a.split == "trainval" else f"datakit_{a.split}")

    import gridfm_datakit.generate as gen

    net = load_enwl1()
    write_matpower(net, ROOT / "grids" / f"{NAME}.m")
    meta = a.data_dir / NAME / "raw" / "fluvius_draws.parquet"
    gen.get_load_scenario_generator = lambda _args: FluviusScenarios(net.customer_bus, meta, a.split, a.augment)
    paths = gen.generate_power_flow_data_distributed(
        config(a.scenarios, a.processes, a.data_dir, seed=42 + SEED_OFFSET[a.split]))
    if a.split == "trainval":
        write_graphkit_splits(meta, a.data_dir / "splits")
    print(f"solved {len(solved_draws(meta))} of {a.scenarios} states")
    print(paths["bus_data"])


def solved_draws(draws: Path) -> pd.DataFrame:
    """Draw log joined to datakit's scenario ids.

    datakit drops states whose OPF/PF fails and renumbers `scenario` to stay
    contiguous; `load_scenario_idx` keeps the draw index. The draw log's own
    `scenario` column is the draw index. Returns one row per solved state, with
    `scenario` = datakit's id and `draw` = the draw index."""
    d = pd.read_parquet(draws).rename(columns={"scenario": "draw"})
    bus = pd.read_parquet(Path(draws).parent / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
    ids = bus.drop_duplicates("scenario").rename(columns={"load_scenario_idx": "draw"})
    return ids.merge(d, on="draw", how="left", validate="one_to_one").sort_values("scenario").reset_index(drop=True)


def write_graphkit_splits(draws: Path, out_dir: Path) -> dict[str, int]:
    """graphkit split files from the draw log: scenarios on train days -> train,
    on val days -> val (also used as graphkit's 'test', which is only a
    sanity check; the real test days live in a separate dataset)."""
    import torch

    d = solved_draws(draws)
    names = day_split(d.step.to_numpy())
    if np.isin(names, ["calib", "test"]).any():
        raise ValueError("training draws contain held-out days")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, which in (("train", "train"), ("val", "val"), ("test", "val")):
        ids = torch.tensor(np.sort(d.scenario.to_numpy()[names == which]), dtype=torch.long)
        torch.save(ids, out_dir / f"{split}.pt")
        sizes[split] = len(ids)
    return sizes


if __name__ == "__main__":
    main()
