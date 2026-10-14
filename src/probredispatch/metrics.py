"""Metrics for the probabilistic congestion and redispatch write-up."""

from __future__ import annotations

import numpy as np


def congestion_probability(loading: np.ndarray, limit: float = 1.0) -> np.ndarray:
    """P(loading > limit) per line across scenarios. loading (n_scenarios, n_lines)."""
    return (loading > limit).mean(axis=0)


def cost_quantiles(cost: np.ndarray, qs=(0.05, 0.5, 0.95)) -> dict[float, float]:
    """Quantiles of total dispatch / redispatch cost across scenarios."""
    return {q: float(np.quantile(cost, q)) for q in qs}


def redispatch_cost(cost_constrained: np.ndarray, cost_unconstrained: np.ndarray) -> np.ndarray:
    """Redispatch cost per scenario = grid-constrained OPF cost - copper-plate cost."""
    return cost_constrained - cost_unconstrained


def loading_mae_by_bin(pred: np.ndarray, true: np.ndarray, edges=(0, 0.5, 0.8, 0.9, 1.0, 1.1, np.inf)) -> list[tuple[str, float, int]]:
    """MAE of loading per true-loading bin: [(label, mae, count), ...]."""
    p, t = pred.ravel(), true.ravel()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (t >= lo) & (t < hi)
        label = f"[{lo}, {hi})"
        out.append((label, float(np.abs(p[idx] - t[idx]).mean()) if idx.any() else float("nan"), int(idx.sum())))
    return out


def screening_report(decision: np.ndarray, true_loading: np.ndarray, limit: float = 1.0) -> dict[str, float]:
    """Compare screen() decisions against AC-OPF truth at scenario level."""
    truly_congested = (true_loading > limit).any(axis=1)
    accepted = decision != "uncertain"
    pred_congested = decision == "congested"
    missed = accepted & truly_congested & ~pred_congested   # labelled safe but congested
    false_alarm = accepted & ~truly_congested & pred_congested
    return {
        "share_sent_to_ac": float((~accepted).mean()),
        "missed_congestion_rate": float(missed.sum() / max(truly_congested.sum(), 1)),
        "false_alarm_rate": float(false_alarm.sum() / max((~truly_congested).sum(), 1)),
    }


def crps_ensemble(samples: np.ndarray, obs: float) -> float:
    """CRPS of an ensemble forecast against a scalar observation."""
    s = np.asarray(samples, dtype=float)
    return float(np.abs(s - obs).mean() - 0.5 * np.abs(s[:, None] - s[None, :]).mean())
