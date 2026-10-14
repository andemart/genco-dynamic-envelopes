"""Split-conformal safety layer for GENCO line-loading predictions.

Calibrate on scenarios where both GENCO and AC-OPF were run. The nonconformity
score is the absolute loading error |pred - true| per line (loading = flow /
thermal limit). At level alpha the upper bound pred + q covers the true loading
with probability >= 1 - alpha (marginally, under exchangeability).

Screen-then-verify rule per scenario:
  * every line has pred + q < 1           -> "safe", accept GENCO
  * some line has pred - q > 1            -> "congested", accept GENCO
  * otherwise (interval straddles limit)  -> "uncertain", send to AC-OPF
"""

from __future__ import annotations

import numpy as np


def calibrate(pred: np.ndarray, true: np.ndarray, alpha: float = 0.05, per_line: bool = False) -> np.ndarray | float:
    """Return conformal quantile q. pred/true shape (n_scenarios, n_lines).

    per_line=True gives a separate quantile per line (needs more calibration data).
    """
    scores = np.abs(pred - true)
    if per_line:
        n = scores.shape[0]
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(scores, level, axis=0, method="higher")
    flat = scores.ravel()
    n = flat.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(flat, level, method="higher"))


def screen(pred: np.ndarray, q: np.ndarray | float, limit: float = 1.0) -> np.ndarray:
    """Classify scenarios. Returns array of "safe" | "congested" | "uncertain"."""
    upper = pred + q
    lower = pred - q
    congested = (lower > limit).any(axis=1)
    safe = (upper < limit).all(axis=1)
    out = np.full(pred.shape[0], "uncertain", dtype=object)
    out[safe] = "safe"
    out[congested] = "congested"
    return out


def coverage(pred: np.ndarray, true: np.ndarray, q: np.ndarray | float) -> float:
    """Empirical share of line-scenario pairs with true <= pred + q."""
    return float((true <= pred + q).mean())
