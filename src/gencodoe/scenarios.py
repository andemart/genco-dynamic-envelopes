"""Correlated probabilistic forecast scenarios for passive load and rooftop PV.

Forecast errors are drawn from a Gaussian copula over (source, bus) pairs so
that errors are correlated along the feeder (one cloud front affects every PV
system on it) and, optionally, between sources. Marginals are relative errors
applied to point forecasts and clipped to physical bounds.

The scenarios describe the *uncontrolled* part of each bus: the net injection
the envelope has to be robust against. The customers' flexible export/import
is what the envelope limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("load", "pv")


@dataclass
class ForecastSpec:
    """Point forecast and relative error std per source, per bus (or zone).

    point: dict source -> array (n_groups,) in pu
    rel_std: dict source -> relative standard deviation (e.g. 0.05 for load)
    corr: (len(sources)*n_groups,)^2 correlation matrix of the stacked errors
    capacity: dict source -> array (n_groups,) upper bound in pu (None = unbounded)
    """

    point: dict[str, np.ndarray]
    rel_std: dict[str, float]
    corr: np.ndarray
    capacity: dict[str, np.ndarray | None]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.point)

    @property
    def n_groups(self) -> int:
        return len(next(iter(self.point.values())))


def default_corr(n_groups: int, rho_within: dict[str, float] | None = None, rho_cross: float = 0.0,
                 sources: tuple[str, ...] = SOURCES) -> np.ndarray:
    """Block correlation: same source across groups = rho_within[source], across sources = rho_cross."""
    rho_within = rho_within or {"load": 0.5, "pv": 0.9}
    k, g = len(sources), n_groups
    c = np.full((k * g, k * g), rho_cross)
    for a, s in enumerate(sources):
        block = np.full((g, g), rho_within.get(s, 0.0))
        np.fill_diagonal(block, 1.0)
        c[a * g:(a + 1) * g, a * g:(a + 1) * g] = block
    # project to nearest PSD by clipping eigenvalues
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def sample(spec: ForecastSpec, n: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Draw n scenarios. Returns source -> array (n, n_groups)."""
    rng = rng or np.random.default_rng()
    g = spec.n_groups
    z = rng.multivariate_normal(np.zeros(len(spec.sources) * g), spec.corr, size=n)
    out: dict[str, np.ndarray] = {}
    for i, s in enumerate(spec.sources):
        eps = z[:, i * g:(i + 1) * g] * spec.rel_std[s]
        val = spec.point[s][None, :] * (1.0 + eps)
        cap = spec.capacity.get(s)
        out[s] = np.clip(val, 0.0, None if cap is None else cap[None, :])
    return out


def net_injection(draws: dict[str, np.ndarray], qd_over_pd: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Passive bus injections (p, q) = (pv - load, -load * tan(phi)); PV at unity power factor."""
    load = draws["load"]
    pv = draws.get("pv", np.zeros_like(load))
    return pv - load, -load * qd_over_pd
