"""Fluvius 2024 digital-meter sample: 8 groups x 300 residential meters, 15-min kWh.

Each CSV holds one group (with/without PV x heat pump x EV). We convert them
once to compact arrays: offtake and injection in kW, shape (300, 35136).
Every meter has 35,136 rows (366 days x 96), in time order; a few timestamps
repeat around the DST change, so steps are indexed by row position and the
first meter's timestamps are kept as labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "load_profiles"
CACHE_DIR = ROOT / "data_out" / "fluvius"
STEPS = 366 * 96

# file suffix -> (heat pump, EV, PV)
GROUPS = {
    "geen_ZP": (0, 0, 0),
    "enkel_ZP": (0, 0, 1),
    "WP_geen_ZP": (1, 0, 0),
    "WP_met_ZP": (1, 0, 1),
    "EV_geen_ZP": (0, 1, 0),
    "EV_met_ZP": (0, 1, 1),
    "WP_EV_geen_ZP": (1, 1, 0),
    "WP_EV_met_ZP": (1, 1, 1),
}


@dataclass
class Group:
    name: str
    hp: int
    ev: int
    pv: int
    ean: np.ndarray        # (n_meters,)
    offtake_kw: np.ndarray  # (n_meters, STEPS)
    injection_kw: np.ndarray
    time: np.ndarray       # (STEPS,) datetime64, UTC as given in the file

    @property
    def net_kw(self) -> np.ndarray:
        """Net consumption at the connection point (negative = export)."""
        return self.offtake_kw - self.injection_kw


def convert(name: str) -> Path:
    """Read one group CSV and cache it as .npz."""
    import pyarrow.csv as pc

    src = next(RAW_DIR.glob(f"*/P6269_Open_Data_{name}.csv"))
    t = pc.read_csv(src, convert_options=pc.ConvertOptions(include_columns=[
        "EAN_ID", "Datum_Startuur", "Volume_Afname_KWh", "Volume_Injectie_KWh",
        "Warmtepomp_Indicator", "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator"]))
    df = t.to_pandas()
    df = df.sort_values("EAN_ID", kind="stable")
    ean = df.EAN_ID.unique()
    counts = df.groupby("EAN_ID", sort=True).size()
    if not (counts == STEPS).all():
        raise ValueError(f"{name}: expected {STEPS} rows per meter, got {counts.unique()}")
    hp, ev, pv = GROUPS[name]
    flags = df[["Warmtepomp_Indicator", "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator"]].drop_duplicates()
    if len(flags) != 1 or tuple(flags.iloc[0]) != (hp, ev, pv):
        raise ValueError(f"{name}: indicator columns {flags.values.tolist()} do not match {(hp, ev, pv)}")
    shape = (len(ean), STEPS)
    out = CACHE_DIR / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, ean=ean,
        offtake_kw=(df.Volume_Afname_KWh.to_numpy() * 4).reshape(shape).astype(np.float32),
        injection_kw=(df.Volume_Injectie_KWh.to_numpy() * 4).reshape(shape).astype(np.float32),
        time=df.Datum_Startuur.to_numpy()[:STEPS].astype("datetime64[m]"),
    )
    return out


def load_group(name: str) -> Group:
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        convert(name)
    d = np.load(path)
    hp, ev, pv = GROUPS[name]
    return Group(name, hp, ev, pv, d["ean"], d["offtake_kw"], d["injection_kw"], d["time"])


if __name__ == "__main__":
    for g in GROUPS:
        print(g, convert(g))
