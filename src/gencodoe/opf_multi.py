"""Multi-scenario AC-OPF envelope (GENCO-OPF layer, option 1): one AC-OPF per interval and direction with a
SHARED limit vector x over the design scenarios (scenario approach), solved with JuMP + IPOPT.

  max sum_i x_i   s.t. for every kept scenario s: polar AC power flow with unit injection sign*x_i (P only),
                       direction-relevant voltage bound (export: V <= Vmax; import: V >= Vmin),
                       sending-end branch apparent power <= rating,  0 <= x_i <= cap_i.
Scenarios infeasible at x = 0 (exact sweep) are discarded first; if more than the allowed number
(S - ceil(0.98 S)) are infeasible the envelope is 0. Remaining discards are spent greedily: drop the kept
scenario with the largest summed constraint dual and re-solve (sampling-and-discarding).
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import envelopes
from . import feeder as fdr

JULIA = r"""
using PowerModels, Ipopt; import PowerModels: JuMP; using .JuMP
function multi_env(par::AbstractVector{<:Integer}, r::AbstractVector{Float64}, x_::AbstractVector{Float64}, rate::AbstractVector{Float64},
                   P::AbstractMatrix{Float64}, Q::AbstractMatrix{Float64}, ubus::AbstractVector{<:Integer}, cap::AbstractVector{Float64},
                   sgn::Float64, vmin::Float64, vmax::Float64, export_dir::Bool)
    S, n = size(P)
    m = Model(Ipopt.Optimizer); set_silent(m); set_optimizer_attribute(m, "max_iter", 3000)
    nu = length(ubus)
    @variable(m, 0 <= xu[i=1:nu] <= cap[i])
    vlo = export_dir ? 0.5 : vmin; vhi = export_dir ? vmax : 1.5
    @variable(m, vlo <= v[s=1:S, k=1:n] <= vhi, start = 1.0)
    @variable(m, th[s=1:S, k=1:n], start = 0.0)
    @variable(m, pf[s=1:S, k=1:n]); @variable(m, qf[s=1:S, k=1:n])      # flow parent -> k at the sending end
    root = findfirst(==(0), par)
    for s in 1:S
        @constraint(m, v[s, root] == 1.0); @constraint(m, th[s, root] == 0.0)
    end
    inj = zeros(AffExpr, n)
    for (j, b) in enumerate(ubus); inj[b] += sgn * xu[j]; end
    children = [Int[] for _ in 1:n]
    for k in 1:n; par[k] > 0 && push!(children[par[k]], k); end
    for s in 1:S, k in 1:n
        k == root && continue
        i = par[k]; z2 = r[k]^2 + x_[k]^2; g = r[k] / z2; b = -x_[k] / z2
        # sending-end flow i -> k of a series branch (no shunt)
        @NLconstraint(m, pf[s, k] == g * v[s, i]^2 - v[s, i] * v[s, k] * (g * cos(th[s, i] - th[s, k]) + b * sin(th[s, i] - th[s, k])))
        @NLconstraint(m, qf[s, k] == -b * v[s, i]^2 - v[s, i] * v[s, k] * (g * sin(th[s, i] - th[s, k]) - b * cos(th[s, i] - th[s, k])))
        isfinite(rate[k]) && @constraint(m, pf[s, k]^2 + qf[s, k]^2 <= rate[k]^2)
        @NLconstraint(m, v[s, k]^2 * g - v[s, k] * v[s, i] * (g * cos(th[s, k] - th[s, i]) + b * sin(th[s, k] - th[s, i]))
                         + sum(pf[s, c] for c in children[k]; init = 0.0) == inj[k] - P[s, k])
        @NLconstraint(m, -v[s, k]^2 * b - v[s, k] * v[s, i] * (g * sin(th[s, k] - th[s, i]) - b * cos(th[s, k] - th[s, i]))
                         + sum(qf[s, c] for c in children[k]; init = 0.0) == -Q[s, k])
    end
    @objective(m, Max, sum(xu))
    optimize!(m)
    st = string(termination_status(m))
    ok = st in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
    return ok ? value.(xu) : fill(NaN, nu), st, solve_time(m)
end
"""
# KCL at bus k: (flow k -> parent) + sum(flows k -> children) = unit injection - demand  (P and Q).


class MultiOPF:
    def __init__(self):
        from juliacall import Main as jl
        self.jl = jl
        jl.seval(JULIA)
        self._f = jl.seval("multi_env")

    def solve(self, fd: fdr.Feeder, P: np.ndarray, Q: np.ndarray, ubus: np.ndarray, cap: np.ndarray, direction: str):
        """P, Q: (S, n) bus CONSUMPTION in pu (positive = demand). Returns (x, ok, solve_s)."""
        par = np.where(fd.parent < 0, 0, fd.parent + 1).astype(np.int64)
        x, st, dt = self._f(par, fd.r.astype(float), fd.x.astype(float),
                            np.where(np.isfinite(fd.rate), fd.rate, np.inf).astype(float),
                            np.ascontiguousarray(P, float), np.ascontiguousarray(Q, float),
                            (ubus + 1).astype(np.int64), cap.astype(float), 1.0 if direction == "export" else -1.0,
                            float(fd.vmin), float(fd.vmax), direction == "export")
        return np.asarray(x, float), str(st) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED"), float(dt)


def multi_envelope(mopf: MultiOPF, fd, cust, p, q, direction, eps=0.05, beta=0.05):
    """Shared-x envelope over the design scenarios (p, q injections, consumption negative)."""
    S = len(p)
    allowed = S - math.ceil(envelopes.acceptance_threshold(S, eps, beta) * S - 1e-9)
    feas = envelopes.exact(fdr.sweep_pf, fd)
    base_ok = feas(p, q, direction)
    info = {"infeasible_at_zero": int((~base_ok).sum()), "allowed_discards": int(allowed)}
    if (~base_ok).sum() > allowed:
        return np.zeros(len(cust)), {**info, "zero_sentinel": True, "opf_solve_s": 0.0}
    keep = base_ok.copy()
    cap = cust.cap(direction)
    x, ok, dt = mopf.solve(fd, -p[keep], -q[keep], cust.bus, cap, direction)
    solve_s, n_solves = dt, 1
    if not ok:
        return np.zeros(len(cust)), {**info, "zero_sentinel": False, "opf_ok": False, "opf_solve_s": solve_s}
    # greedy sampling-and-discarding: spend the remaining allowance on the most binding kept scenario
    sign = 1.0 if direction == "export" else -1.0
    for _ in range(allowed - int((~base_ok).sum())):
        inj = sign * np.clip(x, 0, cap) @ cust.incidence(fd.n_bus)
        res = fdr.sweep_pf(fd, p + inj, q)
        vm = res.vm[:, 1:]
        head_v = (fd.vmax - vm.max(1)) if direction == "export" else (vm.min(1) - fd.vmin)
        rated = np.isfinite(fd.rate)
        head_l = 1.0 - res.loading[:, rated].max(1) if rated.any() else np.full(len(p), np.inf)
        head = np.minimum(head_v, head_l)
        head[~keep] = np.inf
        drop = int(np.argmin(head))
        keep2 = keep.copy()
        keep2[drop] = False
        x2, ok2, dt2 = mopf.solve(fd, -p[keep2], -q[keep2], cust.bus, cap, direction)
        solve_s += dt2
        n_solves += 1
        if not ok2 or np.nansum(x2) <= np.nansum(x) * (1 + 1e-4):
            break                                      # no gain from discarding: stop
        x, keep = x2, keep2
    info.update({"zero_sentinel": False, "opf_ok": True, "opf_solve_s": solve_s, "opf_solves": n_solves,
                 "kept_scenarios": int(keep.sum())})
    return np.clip(np.nan_to_num(x), 0, cap), info
