"""Metrics for the dynamic operating envelope write-up."""

from __future__ import annotations

import numpy as np


def jain_index(x: np.ndarray) -> float:
    """Jain's fairness index: 1 = all equal, 1/n = one customer gets everything."""
    x = np.asarray(x, float)
    s2 = (x ** 2).sum()
    return float(x.sum() ** 2 / (len(x) * s2)) if s2 > 0 else 1.0


def envelope_summary(limit: np.ndarray) -> dict[str, float]:
    """Hosting (sum of limits), worst-off customer and fairness of one envelope."""
    return {"total": float(limit.sum()), "min": float(limit.min()), "mean": float(limit.mean()),
            "jain": jain_index(limit)}


def envelope_gap(limit: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compare an envelope against the reference (AC power flow) envelope."""
    tot = ref.sum()
    return {"total_rel_gap": float((limit.sum() - tot) / tot) if tot > 0 else float("nan"),
            "mae": float(np.abs(limit - ref).mean())}


def curtailment(limit: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Export lost to the envelope per scenario. available (n_s, n_c) flexible export, limit (n_c,)."""
    return np.clip(available - limit[None, :], 0.0, None).sum(axis=1)


def mae_by_bin(pred: np.ndarray, true: np.ndarray, edges) -> list[tuple[str, float, int]]:
    """MAE per bin of the true value: [(label, mae, count), ...]. E.g. voltage bins near vmax."""
    p, t = pred.ravel(), true.ravel()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (t >= lo) & (t < hi)
        out.append((f"[{lo}, {hi})", float(np.abs(p[idx] - t[idx]).mean()) if idx.any() else float("nan"), int(idx.sum())))
    return out


def screening_report(decision: np.ndarray, true_ok: np.ndarray) -> dict[str, float]:
    """Compare conformal.screen() decisions against AC power flow feasibility per row."""
    accepted = decision != "uncertain"
    missed = accepted & (decision == "safe") & ~true_ok        # labelled safe, actually violates
    false_alarm = accepted & (decision == "violated") & true_ok
    return {
        "share_sent_to_ac": float((~accepted).mean()),
        "missed_violation_rate": float(missed.sum() / max((~true_ok).sum(), 1)),
        "false_alarm_rate": float(false_alarm.sum() / max(true_ok.sum(), 1)),
    }


def crps_ensemble(samples: np.ndarray, obs: float) -> float:
    """CRPS of an ensemble forecast against a scalar observation."""
    s = np.asarray(samples, dtype=float)
    return float(np.abs(s - obs).mean() - 0.5 * np.abs(s[:, None] - s[None, :]).mean())
