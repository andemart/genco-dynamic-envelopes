"""C+ (post-hoc safety enhancement): C plus a final exact gate. See docs/cplus_protocol.md.

For every stored C result (interval x direction):
  * L_C = 0 -> kept; classified zero_infeasible_sentinel / zero_feasible from C's
    own (screened) design share;
  * L_C > 0 -> gate: exact four-wire share of the replayed 200 design scenarios at
    L_C (B-compatible OpenDSS configuration and scenario order);
      - pass: keep L_C;
      - fail: diagnostic canonical re-solve (every scenario from the no-load
        state) classifies the failure as solver-history noise or screen error;
        exact downward repair on [0, L_C] (same rule, levels per round,
        tolerance as the envelope search); a repaired level is re-verified on
        the run's fresh draws (four-wire and balanced metrics recomputed).
Search cost = C's + gate + repair; verification and diagnostic are counted
separately. Source files are hashed before and after (must not change).

Usage:
  python -m gencodoe.cplus --runs data_out/doe/annual/*c_genco_4w_opendss*.json \
      --b-dir data_out/doe/annual --out-dir data_out/doe/cplus_annual
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

CAPS_KW = {"export": 5.0, "import": 7.4}
SIGN = {"export": 1.0, "import": -1.0}
C_METHOD, CPLUS_METHOD, B_METHOD = "c_genco_4w_opendss", "cplus_genco_4w_gate", "b_opendss_exact"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Counter:
    """Calls and seconds per accounting category."""

    def __init__(self):
        self.calls, self.seconds = {}, {}

    def add(self, cat: str, calls: int, seconds: float):
        self.calls[cat] = self.calls.get(cat, 0) + calls
        self.seconds[cat] = self.seconds.get(cat, 0.0) + seconds


class FourWireOracle:
    """Exact four-wire evaluation of rows with every customer at one level.

    canonical=False: B-compatible (solves in scenario order, each starting from the previous
    solution, retry from no-load on non-convergence). canonical=True: every solve starts from
    the no-load state (diagnostic only)."""

    def __init__(self, net, unbalanced):
        from .four_wire import four_wire_ok
        self.net, self.u, self._ok = net, unbalanced, four_wire_ok

    def results(self, p, q, level_kw, direction, canonical=False):
        cb = self.net.customer_bus
        pc = -(p[:, cb] + SIGN[direction] * level_kw * 1e-3) * 1e3
        qc = -q[:, cb] * 1e3
        out = []
        for i in range(len(p)):
            if canonical:
                self.u._reset()
            out.append(self.u.solve(pc[i], qc[i]))
        return out

    def share(self, p, q, level_kw, direction, canonical=False):
        res = self.results(p, q, level_kw, direction, canonical)
        return float(np.mean([self._ok(r, direction) for r in res])), len(res)


def repair(oracle, p, q, hi_kw: float, direction: str, thr: float, k: int = 4, rel_tol: float = 1e-3):
    """Largest level in [0, hi_kw] with exact share >= thr. Returns (level, share_at_level, calls, zero_infeasible)."""
    calls = 0
    s0, n = oracle.share(p, q, 0.0, direction)
    calls += n
    if s0 < thr:
        return 0.0, s0, calls, True
    lo, s_lo = 0.0, s0
    tol = rel_tol * CAPS_KW[direction]
    while hi_kw - lo > tol:
        levels = np.linspace(lo, hi_kw, k + 2)[1:-1]
        shares = []
        for L in levels:
            s, n = oracle.share(p, q, float(L), direction)
            calls += n
            shares.append(s)
        ok = [s >= thr for s in shares]
        if all(ok):
            lo, s_lo = float(levels[-1]), shares[-1]
        else:
            i = ok.index(False)
            hi_kw = float(levels[i])
            if i > 0:
                lo, s_lo = float(levels[i - 1]), shares[i - 1]
    return lo, s_lo, calls, False


def gate_interval(oracle, p, q, row: dict, direction: str, thr: float, acc: Counter, b_limit_kw: float | None = None):
    """C+ decision for one stored C row (design scenarios p, q). Returns the C+ fields (no verification)."""
    L_c = float(row["limit_kw"])
    out = {"c_limit_kw": L_c, "gated": False, "repaired": False, "gate_calls": 0, "repair_calls": 0,
           "gate_s": 0.0, "repair_s": 0.0, "diagnostic_calls": 0, "diagnostic_s": 0.0,
           "b_limit_kw": b_limit_kw, "limit_kw": L_c, "prob_ok_oracle": row["prob_ok_oracle"],
           "failure_class": None, "exact_design_share_at_c": None, "canonical_design_share_at_c": None}
    if L_c <= 1e-9:
        zero_bad = row["prob_ok_oracle"] < thr
        out.update({"zero_infeasible_sentinel": bool(zero_bad), "zero_feasible": not zero_bad,
                    "zero_class_source": "C-screened design share"})
        return out
    out["gated"] = True
    t = time.perf_counter()
    share, n = oracle.share(p, q, L_c, direction)
    out.update({"gate_calls": n, "gate_s": time.perf_counter() - t, "exact_design_share_at_c": share})
    acc.add("gate", n, out["gate_s"])
    if share >= thr:
        out.update({"prob_ok_oracle": share, "zero_infeasible_sentinel": False, "zero_feasible": False})
        return out
    # diagnostic (never changes the envelope)
    t = time.perf_counter()
    canon, n = oracle.share(p, q, L_c, direction, canonical=True)
    out.update({"diagnostic_calls": n, "diagnostic_s": time.perf_counter() - t, "canonical_design_share_at_c": canon,
                "failure_class": "solver_history_noise" if canon >= thr else "screen_error"})
    acc.add("diagnostic", n, out["diagnostic_s"])
    # exact downward repair
    t = time.perf_counter()
    level, s_level, n, zero_bad = repair(oracle, p, q, L_c, direction, thr)
    out.update({"repaired": True, "limit_kw": level, "prob_ok_oracle": s_level, "repair_calls": n,
                "repair_s": time.perf_counter() - t, "zero_infeasible_sentinel": bool(zero_bad),
                "zero_feasible": bool(level <= 1e-9 and not zero_bad)})
    acc.add("repair", n, out["repair_s"])
    return out


def verify_repaired(oracle, feeder, customers_bus, pv, qv, n_unbalanced: int, level_kw: float, direction: str,
                    acc: Counter) -> dict:
    """Held-out metrics of a repaired level on the run's fresh draws (as doe_day.run_day)."""
    from . import envelopes
    from . import feeder as fdr
    from .four_wire import four_wire_ok
    t = time.perf_counter()
    res = oracle.results(pv[:n_unbalanced], qv[:n_unbalanced], level_kw, direction)
    sign = SIGN[direction]
    inj = np.zeros(feeder.n_bus)
    inj[customers_bus] = sign * level_kw * 1e-3
    p_bal = float(envelopes.exact(fdr.sweep_pf, feeder)(pv + inj, qv, direction).mean())
    acc.add("verification", len(res), time.perf_counter() - t)
    return {"prob_ok_unbalanced": 1.0 - float(np.mean([not four_wire_ok(r, direction) for r in res])),
            "unbalanced_v_min": min(r["v_pn_min"] for r in res), "unbalanced_v_max": max(r["v_pn_max"] for r in res),
            "unbalanced_line_loading_max": max(r["line_loading_max"] for r in res), "unbalanced_draws": len(res),
            "prob_ok_ac_fresh": p_bal, "verification_calls": len(res)}


def gate_run(path: Path, out_dir: Path, b_dir: Path | None = None, oracle=None) -> dict:
    from .doe_day import DayScenarios
    from .enwl import load_enwl1
    from .envelopes import acceptance_threshold
    from .unbalanced import UnbalancedENWL1

    hashes_before = {str(path): sha256(path)}
    doc = json.loads(path.read_text())
    m = doc["meta"]
    b_path = None
    if b_dir is not None:
        b_path = Path(b_dir) / path.name.replace(C_METHOD, B_METHOD)
        hashes_before[str(b_path)] = sha256(b_path)
    b_rows = {}
    if b_path is not None:
        b_rows = {(r["step"], r["direction"]): r for r in json.loads(b_path.read_text())["rows"]
                  if r["method"] == B_METHOD}
    sph = m.get("steps_per_hour", 4)
    seed = m.get("scenario_seed", 101)
    hh = m.get("household_seed", seed)
    n_s, n_v, n_u = m["scenarios"], m.get("verify_draws") or m["scenarios"], m.get("unbalanced_draws") or 200
    thr = acceptance_threshold(n_s, m["eps"], m["beta"])
    net = load_enwl1()
    fd = net.to_feeder()
    scen = DayScenarios(net.customer_bus, net.n_bus, seed=hh)
    rng = np.random.default_rng(seed)
    oracle = oracle or FourWireOracle(net, UnbalancedENWL1())
    acc = Counter()
    t0 = int((np.datetime64(m["day"]) - np.datetime64("2024-01-01")).astype(int)) * 96
    rows_c = {(r["step"], r["direction"]): r for r in doc["rows"] if r["method"] == C_METHOD}
    out_rows = []
    for k in range(0, 96, 4 // sph):
        step = t0 + k
        p, q = scen.sample(step, n_s, rng, fd.base_mva)        # replay, same order as run_day
        pv, qv = scen.sample(step, n_v, rng, fd.base_mva)
        for direction in ("export", "import"):
            c = rows_c[(step, direction)]
            b = b_rows.get((step, direction))
            g = gate_interval(oracle, p, q, c, direction, thr, acc, b["limit_kw"] if b else None)
            r = dict(c)
            r.update(g)
            if g["repaired"]:
                r.update(verify_repaired(oracle, fd, net.customer_bus, pv, qv, n_u, g["limit_kw"], direction, acc))
                r["total_kw"] = g["limit_kw"] * len(net.customer_bus)
            r["method"] = CPLUS_METHOD
            r["c_ac_calls"], r["c_runtime_s"] = c["ac_calls"], c["runtime_s"]
            r["ac_calls"] = c["ac_calls"] + g["gate_calls"] + g["repair_calls"]          # search cost only
            r["runtime_s"] = c["runtime_s"] + g["gate_s"] + g["repair_s"]
            out_rows.append(r)
    hashes_after = {p_: sha256(Path(p_)) for p_ in hashes_before}
    if hashes_after != hashes_before:
        raise SystemExit(f"input files changed during C+ processing: {hashes_before} -> {hashes_after}")
    meta = {**m, "methods_requested": [CPLUS_METHOD], "post_hoc": "C+ final exact gate (docs/cplus_protocol.md)",
            "source_run": str(path), "source_b_run": str(b_path) if b_path else None,
            "input_sha256": hashes_before, "acceptance_threshold": thr,
            "accounting": {"calls": acc.calls, "seconds": acc.seconds},
            "gated_intervals": int(sum(r["gated"] for r in out_rows)),
            "repaired_intervals": int(sum(r["repaired"] for r in out_rows)),
            "screen_errors": int(sum(r["failure_class"] == "screen_error" for r in out_rows)),
            "solver_noise": int(sum(r["failure_class"] == "solver_history_noise" for r in out_rows))}
    ac_rows = [r for r in doc["rows"] if r["method"] == "ac"]
    out = Path(out_dir) / path.name.replace(C_METHOD, CPLUS_METHOD)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": ac_rows + out_rows}, indent=1))
    return {"file": str(out), **{k: meta[k] for k in ("gated_intervals", "repaired_intervals", "screen_errors",
                                                      "solver_noise", "accounting")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="stored C runs")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--b-dir", type=Path, default=None, help="directory of the matching B runs (for |L_C - L_B|)")
    a = ap.parse_args()
    for path in a.runs:
        print(json.dumps(gate_run(path, a.out_dir, a.b_dir)), flush=True)


if __name__ == "__main__":
    main()
