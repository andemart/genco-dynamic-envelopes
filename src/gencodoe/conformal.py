"""Split-conformal safety layer for GENCO power flow predictions.

Calibrate on injections where both GENCO and AC power flow were run. The
nonconformity scores are |vm_pred - vm_true| per bus and |loading_pred -
loading_true| per branch. At level alpha, pred +/- q covers the true value with
probability >= 1 - alpha (marginally per element, under exchangeability).

Screen-then-verify rule per row of injections:
  * all intervals inside the limits        -> "safe", accept GENCO
  * some interval entirely outside a limit -> "violated", accept GENCO
  * otherwise (an interval straddles one)  -> "uncertain", run AC power flow
"""

from __future__ import annotations

import numpy as np

from .feeder import Feeder, PFResult


def calibrate(pred: np.ndarray, true: np.ndarray, alpha: float = 0.05, per_element: bool = False) -> np.ndarray | float:
    """Return conformal quantile q of |pred - true|. pred/true shape (n_samples, n_elements).

    per_element=True gives a separate quantile per bus/branch (needs more calibration data).
    """
    scores = np.abs(pred - true)
    if per_element:
        n = scores.shape[0]
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(scores, level, axis=0, method="higher")
    flat = scores.ravel()
    n = flat.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(flat, level, method="higher"))


VM_BIN_EDGES = (-np.inf, 0.90, 0.93, 0.95, 0.98, 1.02, 1.05, 1.08, 1.11, 1.15, np.inf)


def calibrate_binned(pred: np.ndarray, true: np.ndarray, alpha: float, edges=VM_BIN_EDGES) -> np.ndarray:
    """Mondrian (group-conditional) split conformal: one quantile of |pred - true| per bin of
    the *predicted* value, so coverage holds within each bin (e.g. near 0.94 / 1.10 pu), not
    only on average. Bins with too few samples for the level fall back to the largest score."""
    p, t = np.ravel(pred), np.ravel(true)
    scores = np.abs(p - t)
    b = np.digitize(p, edges[1:-1])
    q = np.empty(len(edges) - 1)
    for k in range(len(q)):
        sk = scores[b == k]
        if sk.size == 0:
            q[k] = scores.max()
            continue
        level = np.ceil((sk.size + 1) * (1 - alpha)) / sk.size
        q[k] = sk.max() if level > 1 else np.quantile(sk, level, method="higher")
    return q


def binned_margin(pred: np.ndarray, q_bins: np.ndarray, edges=VM_BIN_EDGES) -> np.ndarray:
    """Per-element margin for predictions `pred` from the per-bin quantiles."""
    return q_bins[np.digitize(pred, edges[1:-1])]


def calibrate_pf(feeder: Feeder, pred: PFResult, true: PFResult, alpha: float = 0.05) -> tuple[float, float]:
    """Conformal quantiles (q_v, q_l) for voltage magnitude (non-slack buses) and loading (rated branches)."""
    q_v = calibrate(pred.vm[:, 1:], true.vm[:, 1:], alpha)
    rated = np.isfinite(feeder.rate)
    rated[0] = False
    q_l = calibrate(pred.loading[:, rated], true.loading[:, rated], alpha) if rated.any() else 0.0
    return q_v, q_l


def directional_limits(feeder: Feeder, res: PFResult, direction: str | None) -> tuple[float, float, np.ndarray]:
    """(vmax, vmin, loading) restricted to the constraints a direction can make worse.

    Exports raise voltage and drive reverse flow; imports lower voltage and drive
    forward flow. A base case that is already undervoltage under heavy load should
    not shrink the export envelope (exports relieve it), and vice versa.
    direction=None checks everything.
    """
    vmax = feeder.vmax if direction in (None, "export") else np.inf
    vmin = feeder.vmin if direction in (None, "import") else -np.inf
    ld = res.loading[:, 1:]
    if direction == "export":
        ld = np.where(res.p_branch[:, 1:] < 0, ld, 0.0)
    elif direction == "import":
        ld = np.where(res.p_branch[:, 1:] > 0, ld, 0.0)
    return vmax, vmin, ld


def screen(feeder: Feeder, pred: PFResult, q_v: float, q_l: float, direction: str | None = None) -> np.ndarray:
    """Classify rows. Returns array of "safe" | "violated" | "uncertain"."""
    vmax, vmin, ld = directional_limits(feeder, pred, direction)
    vm = pred.vm[:, 1:]
    safe = ((vm + q_v <= vmax) & (vm - q_v >= vmin)).all(axis=1) & (ld + q_l <= 1.0).all(axis=1)
    violated = ((vm - q_v > vmax) | (vm + q_v < vmin)).any(axis=1) | (ld - q_l > 1.0).any(axis=1)
    out = np.full(vm.shape[0], "uncertain", dtype=object)
    out[safe] = "safe"
    out[violated] = "violated"
    return out


def coverage(pred: np.ndarray, true: np.ndarray, q: np.ndarray | float) -> float:
    """Empirical share of elements with |pred - true| <= q."""
    return float((np.abs(pred - true) <= q).mean())
