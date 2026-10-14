"""Correlated probabilistic forecast scenarios for load, wind and PV.

Forecast errors are drawn from a Gaussian copula so that errors are correlated
across sources (e.g. wind and PV both short in a cloudy, calm hour) and across
regions. Marginals are relative errors applied to point forecasts and clipped
to physical bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("load", "wind", "pv")


@dataclass
class ForecastSpec:
    """Point forecast and relative error std per source, per region.

    point: dict source -> array of shape (n_regions,) in MW
    rel_std: dict source -> float relative standard deviation (e.g. 0.03 for load)
    corr: (3*n_regions, 3*n_regions) correlation matrix of the stacked errors
    capacity: dict source -> array (n_regions,) upper bound in MW (None = unbounded)
    """

    point: dict[str, np.ndarray]
    rel_std: dict[str, float]
    corr: np.ndarray
    capacity: dict[str, np.ndarray | None]

    @property
    def n_regions(self) -> int:
        return len(self.point["load"])


def default_corr(n_regions: int, rho_regions: float = 0.6, rho_wind_pv: float = -0.1) -> np.ndarray:
    """Block correlation: same source across regions = rho_regions, wind-PV = rho_wind_pv."""
    k = len(SOURCES)
    n = k * n_regions
    c = np.zeros((n, n))
    for a in range(k):
        for b in range(k):
            if a == b:
                block = np.full((n_regions, n_regions), rho_regions)
                np.fill_diagonal(block, 1.0)
            elif {SOURCES[a], SOURCES[b]} == {"wind", "pv"}:
                block = np.full((n_regions, n_regions), rho_wind_pv)
            else:
                block = np.zeros((n_regions, n_regions))
            c[a * n_regions:(a + 1) * n_regions, b * n_regions:(b + 1) * n_regions] = block
    # project to nearest PSD by clipping eigenvalues
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def sample(spec: ForecastSpec, n: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Draw n scenarios. Returns source -> array (n, n_regions) in MW."""
    rng = rng or np.random.default_rng()
    r = spec.n_regions
    z = rng.multivariate_normal(np.zeros(3 * r), spec.corr, size=n)
    out: dict[str, np.ndarray] = {}
    for i, s in enumerate(SOURCES):
        eps = z[:, i * r:(i + 1) * r] * spec.rel_std[s]
        val = spec.point[s][None, :] * (1.0 + eps)
        cap = spec.capacity.get(s)
        val = np.clip(val, 0.0, None if cap is None else cap[None, :])
        out[s] = val
    return out
