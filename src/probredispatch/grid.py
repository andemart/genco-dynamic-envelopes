"""Map regional forecast scenarios onto a MATPOWER-style case.

Each bus belongs to one region. Regional load is distributed over buses in
proportion to the nominal case load; regional wind/PV is distributed over the
generators tagged as that technology, by nominal capacity, and enters as an
upper limit (Pmax) so OPF may curtail it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CaseMapping:
    bus_region: np.ndarray        # (n_bus,) region index per bus
    bus_pd_nominal: np.ndarray    # (n_bus,) nominal active load, MW
    bus_qd_nominal: np.ndarray    # (n_bus,) nominal reactive load, MVAr
    gen_region: np.ndarray        # (n_gen,) region index per generator
    gen_tech: np.ndarray          # (n_gen,) "wind", "pv" or "other"
    gen_pmax_nominal: np.ndarray  # (n_gen,) nominal Pmax, MW


def bus_loads(m: CaseMapping, regional_load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """regional_load (n, n_regions) -> (pd, qd), each (n, n_bus), constant power factor."""
    n_regions = regional_load.shape[1]
    share = np.zeros_like(m.bus_pd_nominal, dtype=float)
    for r in range(n_regions):
        idx = m.bus_region == r
        tot = m.bus_pd_nominal[idx].sum()
        if tot > 0:
            share[idx] = m.bus_pd_nominal[idx] / tot
    pd = regional_load[:, m.bus_region] * share[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(m.bus_pd_nominal > 0, m.bus_qd_nominal / m.bus_pd_nominal, 0.0)
    return pd, pd * ratio[None, :]


def res_pmax(m: CaseMapping, regional: dict[str, np.ndarray]) -> np.ndarray:
    """Return per-scenario Pmax (n, n_gen); non-RES generators keep nominal Pmax."""
    n = next(iter(regional.values())).shape[0]
    pmax = np.tile(m.gen_pmax_nominal.astype(float), (n, 1))
    for tech in ("wind", "pv"):
        for r in np.unique(m.gen_region):
            idx = (m.gen_tech == tech) & (m.gen_region == r)
            cap = m.gen_pmax_nominal[idx].sum()
            if cap > 0:
                pmax[:, idx] = regional[tech][:, [r]] * (m.gen_pmax_nominal[idx] / cap)[None, :]
    return pmax
