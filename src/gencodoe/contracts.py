"""Case-study households and their contracted capacity.

Each of the 200 ENWL1 customers gets a fixed identity for the whole year: a
technology group (heat pump / EV / PV, drawn once from a penetration mix) and
one real Fluvius meter (EAN) of that group.

Contracted capacity follows the Flemish capacity tariff (capaciteitstarief):
the average of the 12 monthly peaks of 15-minute offtake (kW), where each
monthly peak is floored at 2.5 kW (VREG / Fluvius rule). The unfloored average
is kept as well. Injection is not part of the capacity tariff; its monthly-peak
average is stored for reference only.

Usage: python -m gencodoe.contracts --out data_out/case_study/households.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .enwl import load_enwl1
from .fluvius import GROUPS, load_group

MIX = {"hp": 0.3, "ev": 0.3, "pv": 0.4}   # assumed penetration for the case study
TARIFF_FLOOR_KW = 2.5


def monthly_average_peak(kw: np.ndarray, month: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Average over the 12 months of each row's monthly 15-min peak, each peak floored at `floor`.
    kw: (n, steps) power in kW; month: (steps,) month number 1..12."""
    peaks = np.stack([kw[:, month == m].max(axis=1) for m in range(1, 13)], axis=1)
    return np.maximum(peaks, floor).mean(axis=1)


@dataclass
class Households:
    customer: list[str]          # ENWL1 load names, Loads.dss order
    bus: np.ndarray              # (n,) bus index in the reduced network
    flags: np.ndarray            # (n, 3) heat pump, EV, PV
    group: list[str]             # Fluvius group name
    ean: np.ndarray              # (n,) meter id
    offtake_kw: np.ndarray       # (n, steps)
    injection_kw: np.ndarray     # (n, steps)
    time: np.ndarray             # (steps,) UTC labels from the files

    @property
    def month(self) -> np.ndarray:
        return pd.DatetimeIndex(self.time).month.to_numpy()

    def contracted_capacity(self, floor: float = TARIFF_FLOOR_KW) -> np.ndarray:
        """kW per household: capacity-tariff average of monthly offtake peaks."""
        return monthly_average_peak(self.offtake_kw, self.month, floor)

    def table(self) -> pd.DataFrame:
        names = [k for k in GROUPS]
        return pd.DataFrame({
            "customer": self.customer, "bus": self.bus, "group": self.group, "ean": self.ean,
            "heat_pump": self.flags[:, 0], "ev": self.flags[:, 1], "pv": self.flags[:, 2],
            "contracted_kw": self.contracted_capacity(),
            "contracted_kw_unfloored": self.contracted_capacity(floor=0.0),
            "annual_peak_offtake_kw": self.offtake_kw.max(axis=1),
            "injection_monthly_avg_peak_kw": monthly_average_peak(self.injection_kw, self.month),
            "annual_offtake_kwh": self.offtake_kw.sum(axis=1) / 4,
            "annual_injection_kwh": self.injection_kw.sum(axis=1) / 4,
        }).assign(group_known=lambda d: d.group.isin(names))


def make_households(seed: int = 0, mix: dict = MIX) -> Households:
    """Fixed technology group and meter per ENWL1 customer (no meter used twice)."""
    net = load_enwl1()
    n = len(net.customer_bus)
    rng = np.random.default_rng(seed)
    flags = (rng.uniform(size=(n, 3)) < np.array([mix["hp"], mix["ev"], mix["pv"]])).astype(int)
    by_flags = {}
    for name in GROUPS:
        g = load_group(name)
        by_flags[(g.hp, g.ev, g.pv)] = g
    group = [""] * n
    ean = np.empty(n, dtype=np.int64)
    off = np.empty((n, by_flags[(0, 0, 0)].offtake_kw.shape[1]), dtype=np.float32)
    inj = np.empty_like(off)
    for key in sorted(by_flags):
        g = by_flags[key]
        who = np.flatnonzero((flags == np.array(key)).all(axis=1))
        pick = rng.choice(len(g.ean), size=len(who), replace=False)
        ean[who], off[who], inj[who] = g.ean[pick], g.offtake_kw[pick], g.injection_kw[pick]
        for i in who:
            group[i] = g.name
    return Households(net.customer_names, net.customer_bus, flags, group, ean, off, inj,
                      by_flags[(0, 0, 0)].time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data_out/case_study/households.csv"))
    a = ap.parse_args()
    h = make_households(a.seed)
    t = h.table().drop(columns="group_known")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    by = t.groupby("group").agg(households=("customer", "size"), contracted_kw=("contracted_kw", "mean"),
                                unfloored_kw=("contracted_kw_unfloored", "mean"),
                                annual_peak_kw=("annual_peak_offtake_kw", "mean"))
    print(by.round(2).to_string())
    print(f"\n{len(t)} households: contracted capacity mean {t.contracted_kw.mean():.2f} kW "
          f"(median {t.contracted_kw.median():.2f}, min {t.contracted_kw.min():.2f}, max {t.contracted_kw.max():.2f}); "
          f"sum {t.contracted_kw.sum():.0f} kW")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
