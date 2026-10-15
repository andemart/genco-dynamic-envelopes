"""Dynamic operating envelopes (DOEs) under forecast uncertainty.

A DOE gives each flexible customer an export (or import) limit for one time
interval such that, if every customer uses its full limit at once, voltage and
thermal limits hold in at least a share 1 - eps of the forecast scenarios for
the passive load and PV on the feeder (a chance constraint over the scenario set).

Envelope search is written against a *feasibility oracle*: a function
(p, q, direction) -> bool per row, where each row is one (scenario, candidate
envelope) pair of bus injections, checked against the constraints the direction
can worsen (is_feasible). That is where the solvers plug in:

  * exact(sweep_pf)          ground truth AC power flow
  * exact(lindistflow)       linear baseline
  * ScreenedFeasibility      GENCO + conformal margins, AC power flow only for
                             rows whose prediction is too close to a limit

Allocation policies:
  * uniform       same limit for all customers (capped at inverter rating)
  * proportional  limit proportional to inverter rating
  * max_total     maximise the summed limit with linear sensitivities (an LP);
                  efficient but favours customers close to the substation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import conformal
from .feeder import Feeder, PFResult, sensitivity

Oracle = Callable[[Feeder, np.ndarray, np.ndarray], PFResult]
Feasibility = Callable[[np.ndarray, np.ndarray, str], np.ndarray]

DIRECTIONS = {"export": 1.0, "import": -1.0}


@dataclass
class Customers:
    bus: np.ndarray         # (n_c,) connection bus
    export_cap: np.ndarray  # (n_c,) inverter export rating, pu
    import_cap: np.ndarray  # (n_c,) import rating (e.g. EV charger + battery), pu

    def __len__(self) -> int:
        return len(self.bus)

    def incidence(self, n_bus: int) -> np.ndarray:
        C = np.zeros((len(self), n_bus))
        C[np.arange(len(self)), self.bus] = 1.0
        return C

    def cap(self, direction: str) -> np.ndarray:
        return self.export_cap if direction == "export" else self.import_cap


@dataclass
class Envelope:
    limit: np.ndarray   # (n_c,) magnitude of the export (or import) limit, pu
    direction: str
    policy: str
    level: float        # scalar level for uniform/proportional, nan for max_total
    prob_ok: float      # share of scenarios feasible at the limit, as judged by the oracle used
    n_evals: int        # power flow evaluations spent (scenarios x candidate levels)


def is_feasible(feeder: Feeder, res: PFResult, direction: str | None = None) -> np.ndarray:
    """True per row if voltages and branch loadings are within limits.

    With a direction, only the constraints that direction can worsen are checked
    (see conformal.directional_limits).
    """
    vmax, vmin, ld = conformal.directional_limits(feeder, res, direction)
    vm = res.vm[:, 1:]
    ok = (vm <= vmax).all(axis=1) & (vm >= vmin).all(axis=1) & (ld <= 1.0).all(axis=1)
    if res.converged is not None:
        ok &= res.converged
    return ok


def exact(oracle: Oracle, feeder: Feeder) -> Feasibility:
    return lambda p, q, direction=None: is_feasible(feeder, oracle(feeder, p, q), direction)


class ScreenedFeasibility:
    """Screen with a fast surrogate (GENCO), verify only uncertain rows with the exact oracle.

    q_v, q_l are conformal quantiles of the surrogate's |error| in voltage and
    loading (conformal.calibrate_pf). q_v may also be a callable mapping the
    predicted non-slack voltages (n, n_bus - 1) to per-element margins, e.g.
    binned (Mondrian) quantiles from conformal.calibrate_binned. With
    verifier=None uncertain rows count as infeasible, which gives a conservative
    envelope with no AC calls at all.
    """

    def __init__(self, feeder: Feeder, surrogate: Oracle, q_v: float, q_l: float, verifier: Oracle | None = None):
        self.feeder, self.surrogate, self.verifier = feeder, surrogate, verifier
        self.q_v, self.q_l = q_v, q_l
        self.n_surrogate = 0
        self.n_verified = 0

    def __call__(self, p: np.ndarray, q: np.ndarray, direction: str | None = None) -> np.ndarray:
        pred = self.surrogate(self.feeder, p, q)
        q_v = self.q_v(pred.vm[:, 1:]) if callable(self.q_v) else self.q_v
        decision = conformal.screen(self.feeder, pred, q_v, self.q_l, direction)
        ok = decision == "safe"
        unsure = decision == "uncertain"
        self.n_surrogate += len(p)
        if self.verifier is not None and unsure.any():
            ok[unsure] = is_feasible(self.feeder, self.verifier(self.feeder, p[unsure], q[unsure]), direction)
            self.n_verified += int(unsure.sum())
        return ok


def allocate(customers: Customers, policy: str, level: float | np.ndarray, direction: str = "export") -> np.ndarray:
    """Limits for scalar level(s): min(level * w, cap). Returns (n_c,) or (k, n_c)."""
    cap = customers.cap(direction)
    w = np.ones_like(cap) if policy == "uniform" else cap
    level = np.asarray(level, float)
    return np.minimum(level[..., None] * w, cap)


def acceptance_threshold(n: int, eps: float, beta: float | None) -> float:
    """Share of the n scenarios that must be feasible to accept a level.

    beta=None: the plain empirical rule, 1 - eps. Otherwise the smallest k/n with
    P[Binomial(n, 1 - eps) >= k] <= beta. Feasibility is monotone in the level, so
    the search can only end above the true (1 - eps)-level L* if the count at a
    level just above L* -- a fixed level, P(feasible) < 1 - eps -- reaches k; hence
    P(true violation probability of the envelope > eps) <= beta.
    """
    if beta is None:
        return 1.0 - eps
    from scipy.stats import binom
    k = np.arange(n + 1)
    tail = binom.sf(k - 1, n, 1.0 - eps)          # P[X >= k]
    ok = np.flatnonzero(tail <= beta)
    return float(ok[0] / n) if len(ok) else 1.0 + 1e-9   # unreachable: no level certifiable with n scenarios


def scalar_envelope(feasible: Feasibility, n_bus: int, customers: Customers, p0: np.ndarray, q0: np.ndarray,
                    direction: str = "export", policy: str = "uniform", eps: float = 0.05,
                    levels_per_round: int = 1, rel_tol: float = 1e-3, max_rounds: int = 60,
                    beta: float | None = None) -> Envelope:
    """Largest level such that P(feasible) >= 1 - eps over the scenarios p0, q0 (n_s, n_bus).

    With beta, the empirical share must reach acceptance_threshold(n_s, eps, beta),
    which certifies P(violation) <= eps with confidence 1 - beta (finite-sample).

    k-section search: each round evaluates `levels_per_round` candidate levels for
    all scenarios in one batch (k=1 is bisection; larger k means fewer, bigger
    batches, which suits GPU inference). Assumes the feasible levels form an
    interval [0, level*], which holds for voltage-rise and reverse-flow limits.
    """
    if policy not in ("uniform", "proportional"):
        raise ValueError(policy)
    sign, cap = DIRECTIONS[direction], customers.cap(direction)
    C = customers.incidence(n_bus)
    n_s = p0.shape[0]
    target = acceptance_threshold(n_s, eps, beta)
    n_evals = 0

    def prob(levels: np.ndarray) -> np.ndarray:
        nonlocal n_evals
        inj = sign * allocate(customers, policy, levels, direction) @ C        # (k, n_bus)
        p = (p0[None, :, :] + inj[:, None, :]).reshape(-1, n_bus)
        q = np.tile(q0, (len(levels), 1))
        n_evals += len(p)
        return feasible(p, q, direction).reshape(len(levels), n_s).mean(axis=1)

    w = np.ones_like(cap) if policy == "uniform" else cap
    top = float(np.max(cap / np.where(w > 0, w, np.inf)))
    p_top, p_zero = prob(np.array([top, 0.0]))
    if p_top >= target:
        return Envelope(allocate(customers, policy, top, direction), direction, policy, top, float(p_top), n_evals)
    if p_zero < target:
        return Envelope(np.zeros(len(customers)), direction, policy, 0.0, float(p_zero), n_evals)

    lo, hi, p_lo = 0.0, top, float(p_zero)
    for _ in range(max_rounds):
        if hi - lo <= rel_tol * top:
            break
        levels = np.linspace(lo, hi, levels_per_round + 2)[1:-1]
        pr = prob(levels)
        ok = pr >= target
        if ok.all():
            lo, p_lo = float(levels[-1]), float(pr[-1])
        else:
            i = int(np.argmin(ok))
            hi = float(levels[i])
            if i > 0:
                lo, p_lo = float(levels[i - 1]), float(pr[i - 1])
    return Envelope(allocate(customers, policy, lo, direction), direction, policy, lo, p_lo, n_evals)


def max_total_envelope(feeder: Feeder, customers: Customers, base: PFResult, direction: str = "export",
                       eps: float = 0.05, v_margin: float = 0.0, l_margin: float = 0.0,
                       floor: float = 0.0) -> Envelope:
    """Maximise the summed limit with LinDistFlow sensitivities around each scenario's base point.

    base: oracle result for every scenario with all flexible customers at zero
    (from GENCO or AC). Each linear constraint is enforced at its own (1 - eps)
    scenario quantile, which is a per-constraint, not a joint, chance constraint;
    verify the result with the exact oracle. `floor` is a minimum limit per
    customer (pu) to temper the policy's bias towards the substation end.
    """
    from scipy.optimize import linprog

    sign, cap = DIRECTIONS[direction], customers.cap(direction)
    R, _ = sensitivity(feeder)
    buses = np.arange(1, feeder.n_bus)
    # Only constraints the direction worsens (as in is_feasible): voltage rise and
    # reverse flow for exports, voltage drop and forward flow for imports.
    rows = [R[np.ix_(buses, customers.bus)]]                       # |dV| per unit of limit
    vm = base.vm[:, buses]
    head_v = feeder.vmax - v_margin - vm if direction == "export" else vm - feeder.vmin - v_margin
    rhs = [np.quantile(head_v, eps, axis=0)]

    rated = buses[np.isfinite(feeder.rate[buses])]
    if len(rated):
        rows.append(feeder.T[np.ix_(customers.bus, rated)].T)      # |dP_branch| per unit of limit
        rate = feeder.rate[rated] * (1.0 - l_margin)
        s0 = base.loading[:, rated] * feeder.rate[rated]
        p0 = base.p_branch[:, rated]
        q0_sq = np.clip(s0 ** 2 - p0 ** 2, 0.0, None)
        p_lim = np.sqrt(np.clip(rate ** 2 - q0_sq, 0.0, None))
        rhs.append(np.quantile(p_lim + sign * p0, eps, axis=0))

    A, b = np.vstack(rows), np.concatenate(rhs)
    lower = np.minimum(floor, cap)
    res = linprog(-np.ones(len(customers)), A_ub=A, b_ub=b, bounds=list(zip(lower, cap)), method="highs")
    limit = res.x if res.success else np.zeros(len(customers))
    return Envelope(limit, direction, "max_total", float("nan"), float("nan"), 0)
