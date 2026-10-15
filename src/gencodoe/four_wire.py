"""Option C: certify a balanced surrogate against the four-wire (unbalanced) feeder.

GENCO predicts the balanced positive-sequence state. The real ENWL1 feeder is
four-wire with single-phase customers, where imports drop customer voltages by
~0.07 pu more and phase loading is ~1.2-1.4x higher (see the experiment log).
Instead of correcting the model, we calibrate the *combined* error -- surrogate
error plus balanced-approximation error -- against OpenDSS on calibration days,
with Mondrian (binned) split conformal on state-level worst cases:

  undervoltage  s = pred_vmin - fw_vmin          binned by predicted min V
  overvoltage   s = fw_vmax - pred_vmax          binned by predicted max V
  loading       s = fw_loading / pred_loading    binned by predicted loading

A four-wire state that did not converge (voltage collapse under heavy import)
scores +inf for undervoltage and loading -- the failure modes of a collapse --
and is left out of the overvoltage calibration: its maximum voltage sits near
the slack, in the same bin as nearly every normal state, and an infinite score
there would block every export decision. (All collapsed calibration states lie
in the highest predicted-loading bin, so they can never be screened safe.) For
each bin we keep the (1 - alpha) quantile (to declare a row safe) and the alpha
quantile (to declare it certainly violating); everything else is checked with
OpenDSS. Directions only check what they can worsen: imports the minimum
voltage, exports the maximum voltage; loading (phase current, any direction)
is checked for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VMIN, VMAX = 0.94, 1.10
V_EDGES = (-np.inf, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.07, 1.09, 1.12, np.inf)
L_EDGES = (-np.inf, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 2.0, np.inf)


def _bin_quantiles(pred, score, edges, alpha):
    """Per bin of `pred`: (alpha, 1 - alpha) split-conformal quantiles of `score`."""
    b = np.digitize(pred, edges[1:-1])
    lo, hi = np.empty(len(edges) - 1), np.empty(len(edges) - 1)
    for k in range(len(lo)):
        s = np.sort(score[b == k])
        n = s.size
        if n == 0:                                  # no calibration data: never decide without OpenDSS
            lo[k], hi[k] = -np.inf, np.inf
            continue
        i_hi = int(np.ceil((n + 1) * (1 - alpha))) - 1
        i_lo = int(np.floor((n + 1) * alpha)) - 1
        hi[k] = s[i_hi] if i_hi < n else np.inf
        lo[k] = s[i_lo] if i_lo >= 0 else -np.inf
    return lo, hi


@dataclass
class FourWireMargins:
    alpha: float
    vmin_lo: np.ndarray
    vmin_hi: np.ndarray
    vmax_lo: np.ndarray
    vmax_hi: np.ndarray
    load_lo: np.ndarray
    load_hi: np.ndarray

    @classmethod
    def fit(cls, pred_vmin, pred_vmax, pred_load, fw_vmin, fw_vmax, fw_load, converged, alpha: float = 0.01):
        bad = ~np.asarray(converged, bool)
        s_vmin = np.where(bad, np.inf, pred_vmin - fw_vmin)
        s_vmax = fw_vmax - pred_vmax
        s_load = np.where(bad, np.inf, fw_load / np.maximum(pred_load, 1e-9))
        return cls(alpha, *_bin_quantiles(pred_vmin, s_vmin, V_EDGES, alpha),
                   *_bin_quantiles(np.asarray(pred_vmax)[~bad], np.asarray(s_vmax)[~bad], V_EDGES, alpha),
                   *_bin_quantiles(pred_load, s_load, L_EDGES, alpha))

    def bounds(self, pred_vmin, pred_vmax, pred_load):
        """Pessimistic and optimistic four-wire worst cases implied by the margins."""
        bv = np.digitize(pred_vmin, V_EDGES[1:-1])
        bx = np.digitize(pred_vmax, V_EDGES[1:-1])
        bl = np.digitize(pred_load, L_EDGES[1:-1])
        with np.errstate(invalid="ignore"):
            pess = (pred_vmin - self.vmin_hi[bv], pred_vmax + self.vmax_hi[bx], pred_load * self.load_hi[bl])
            opt = (pred_vmin - self.vmin_lo[bv], pred_vmax + self.vmax_lo[bx], pred_load * self.load_lo[bl])
        return pess, opt

    def screen(self, pred_vmin, pred_vmax, pred_load, direction: str | None = None) -> np.ndarray:
        """'safe' | 'violated' | 'uncertain' per row."""
        (pv, px, pl), (ov, ox, ol) = self.bounds(pred_vmin, pred_vmax, pred_load)
        chk_v = direction in (None, "import")      # plain bools: use `not`, never `~` (~True == -2)
        chk_x = direction in (None, "export")
        safe = (pl <= 1.0) & ((pv >= VMIN) if chk_v else True) & ((px <= VMAX) if chk_x else True)
        violated = (ol > 1.0) | ((ov < VMIN) if chk_v else False) | ((ox > VMAX) if chk_x else False)
        out = np.full(len(safe), "uncertain", dtype=object)
        out[safe & ~violated] = "safe"
        out[violated] = "violated"
        return out


def four_wire_ok(res: dict, direction: str | None = None) -> bool:
    """Feasibility of one OpenDSS result (UnbalancedENWL1.solve) for a direction."""
    if not res["converged"]:
        return False
    ok = not res["line_overload"] and not res["transformer_overload"]
    if direction in (None, "import"):
        ok &= not res["undervoltage"]
    if direction in (None, "export"):
        ok &= not res["overvoltage"]
    return bool(ok)


class FourWireScreened:
    """Feasibility oracle for envelopes.scalar_envelope: GENCO screen with four-wire margins,
    OpenDSS for the uncertain rows (and for every row when surrogate is None: option B)."""

    def __init__(self, feeder, customer_bus, unbalanced, surrogate=None, margins: FourWireMargins | None = None):
        self.feeder, self.customer_bus, self.unbalanced = feeder, customer_bus, unbalanced
        self.surrogate, self.margins = surrogate, margins
        self.n_surrogate = self.n_verified = 0

    def _opendss(self, p, q, direction):
        pc = -p[:, self.customer_bus] * 1e3 * self.feeder.base_mva
        qc = -q[:, self.customer_bus] * 1e3 * self.feeder.base_mva
        return np.array([four_wire_ok(self.unbalanced.solve(pc[i], qc[i]), direction) for i in range(len(p))])

    def __call__(self, p, q, direction=None):
        p, q = np.atleast_2d(p), np.atleast_2d(q)
        if self.surrogate is None:
            self.n_verified += len(p)
            return self._opendss(p, q, direction)
        pred = self.surrogate(self.feeder, p, q)
        decision = self.margins.screen(pred.vm[:, 1:].min(1), pred.vm[:, 1:].max(1),
                                       pred.loading[:, 1:].max(1), direction)
        self.n_surrogate += len(p)
        ok = decision == "safe"
        unsure = decision == "uncertain"
        if unsure.any():
            ok[unsure] = self._opendss(p[unsure], q[unsure], direction)
            self.n_verified += int(unsure.sum())
        return ok
