"""GENCO-OPF layer, shared envelope pipeline (docs/genco_opf_protocol.md, sections 3-4 and amendment 1).

Per interval and direction:
  proposal: per-scenario optimal allocation x^s (method-specific)
  -> x_i = eps-quantile over scenarios of |x_i^s|
  -> exact check of the vector on all S design scenarios and scale-down lambda in [0, 1] (k-section, same rule)
  -> held-out verification on fresh draws.
Methods: uniform (exact uniform search), lp (LinDistFlow LP per scenario), ipopt (AC-OPF per scenario),
genco (GENCO-OPF per scenario, added once trained). The exact model is the balanced radial AC sweep (stage A).

    python -m gencodoe.opf_envelopes --split val --days 80 81 --methods uniform lp ipopt --out data_out/opf_case33/dev.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import envelopes
from . import feeder as fdr
from . import opf_case33 as oc

S_DESIGN, S_FRESH, EPS, BETA, LEVELS = 200, 1000, 0.05, 0.05, 4
SCENARIO_SEED = 101
SIGN = {"export": 1.0, "import": -1.0}


def customers(case: oc.Case33) -> envelopes.Customers:
    return envelopes.Customers(bus=case.unit_bus, export_cap=case.cap["export"], import_cap=case.cap["import"])


def interval_draws(sc: oc.Scenarios, day: int, hour: int):
    rng = np.random.default_rng([SCENARIO_SEED, day, hour])
    p, q = sc.sample(day, hour, S_DESIGN, rng)
    pv, qv = sc.sample(day, hour, S_FRESH, rng)
    return p, q, pv, qv


def proposal_lp(case, cust, p, q, direction):
    """LinDistFlow LP per scenario around that scenario's exact AC base point."""
    base = fdr.sweep_pf(case.fd, p, q)
    out = np.zeros((len(p), len(cust)))
    for s in range(len(p)):
        b = fdr.PFResult(vm=base.vm[s:s + 1], loading=base.loading[s:s + 1], p_branch=base.p_branch[s:s + 1])
        out[s] = envelopes.max_total_envelope(case.fd, cust, b, direction, eps=0.0).limit
    return out


def proposal_ipopt(opf: oc.OPF, p, q, direction):
    x, solve_s, total_s, infeasible = np.zeros((len(p), len(opf.c.unit_bus))), 0.0, 0.0, 0
    for s in range(len(p)):
        pg, ok, dt, tt = opf.solve(p[s], q[s], direction)
        solve_s += dt
        total_s += tt
        if ok:
            x[s] = np.abs(pg)
        else:
            infeasible += 1                          # amendment 1: x^s = 0 for every unit
    return x, {"ipopt_solve_s": solve_s, "ipopt_total_s": total_s, "infeasible_scenarios": infeasible}


def scale_down(case, cust, x, p, q, direction):
    """Exact joint check of the vector x and the largest lambda in [0,1] that passes the design-sample rule."""
    feas = envelopes.exact(fdr.sweep_pf, case.fd)
    if not (x > 0).any():
        base_ok = float(feas(p, q, direction).mean())
        return np.zeros(len(cust)), 0.0, base_ok, len(p)
    keep = x > 0
    sub = envelopes.Customers(bus=cust.bus[keep], export_cap=x[keep], import_cap=x[keep])
    env = envelopes.scalar_envelope(feas, case.fd.n_bus, sub, p, q, direction, "proportional", EPS, LEVELS, beta=BETA)
    lim = np.zeros(len(cust))
    lim[keep] = env.limit
    return lim, float(env.level), float(env.prob_ok), int(env.n_evals)


def fresh_ok(case, cust, limit, pv, qv, direction):
    inj = SIGN[direction] * limit @ cust.incidence(case.fd.n_bus)
    return float(envelopes.exact(fdr.sweep_pf, case.fd)(pv + inj, qv, direction).mean())


def run_interval(case, sc, opf, day, hour, direction, methods, genco=None, mopf=None, predictors=None):
    cust = customers(case)
    p, q, pv, qv = interval_draws(sc, day, hour)
    rows = []
    for m in methods:
        t = time.perf_counter()
        extra = {}
        if m == "uniform":
            env = envelopes.scalar_envelope(envelopes.exact(fdr.sweep_pf, case.fd), case.fd.n_bus, cust, p, q, direction,
                                            "uniform", EPS, LEVELS, beta=BETA)
            prop_s = time.perf_counter() - t
            limit, lam, share, evals = env.limit, float("nan"), float(env.prob_ok), int(env.n_evals)
            check_s = 0.0
        elif m in ("multi", "gnn", "mlp", "gnn_genco"):                 # direct shared-vector proposals
            if m == "multi":
                from .opf_multi import multi_envelope
                x, extra = multi_envelope(mopf, case.fd, cust, p, q, direction)
            else:
                from .opf_labels import node_features
                t1 = time.perf_counter()
                feats = node_features(case, p, q, direction)[None]
                if m == "gnn_genco":                                    # amendment 3: + frozen GENCO embeddings
                    feats = predictors[m].augment(feats, p, q)
                frac = predictors[m].predict(feats)[0]
                extra = {"infer_s": time.perf_counter() - t1}
                x = np.clip(frac[case.unit_bus], 0, 1) * case.cap[direction]
            prop_s = time.perf_counter() - t
            t2 = time.perf_counter()
            limit, lam, share, evals = scale_down(case, cust, x, p, q, direction)
            check_s = time.perf_counter() - t2
            extra["proposal_total_mw"] = float(x.sum())
        else:
            if m == "lp":
                xs = proposal_lp(case, cust, p, q, direction)
            elif m == "ipopt":
                xs, extra = proposal_ipopt(opf, p, q, direction)
            elif m == "genco":
                xs, extra = genco(p, q, direction)
            else:
                raise ValueError(m)
            prop_s = time.perf_counter() - t
            x = np.quantile(xs, EPS, axis=0)
            t2 = time.perf_counter()
            limit, lam, share, evals = scale_down(case, cust, x, p, q, direction)
            check_s = time.perf_counter() - t2
            extra["proposal_total_mw"] = float(x.sum())
        rows.append({"day": day, "hour": hour, "direction": direction, "method": m,
                     "total_mw": float(limit.sum()), "limit_mw": limit.round(6).tolist(), "lambda": lam,
                     "design_share": share, "fresh_ok": fresh_ok(case, cust, limit, pv, qv, direction),
                     "proposal_s": prop_s, "check_s": check_s, "exact_evals": evals, **extra})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--days", type=int, nargs="+", required=True)
    ap.add_argument("--hours", type=int, nargs="+", default=list(range(24)))
    ap.add_argument("--methods", nargs="+", default=["uniform", "lp", "ipopt"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="final", help="trained run tag for gnn/mlp/gnn_genco")
    ap.add_argument("--genco-run", type=Path, default=None, help="graphkit run dir for gnn_genco (default: newest)")
    a = ap.parse_args()
    if any(d not in oc.SPLIT[a.split] for d in a.days):
        raise SystemExit(f"days {a.days} are not all in split {a.split}")
    case = oc.build()
    sc = oc.Scenarios(case, seed=0)
    opf = oc.OPF(case, Path("data_out/opf_case33")) if "ipopt" in a.methods else None
    mopf = None
    if "multi" in a.methods:
        from .opf_multi import MultiOPF
        mopf = MultiOPF()
    predictors = {}
    for m in ("gnn", "mlp"):
        if m in a.methods:
            from .opf_gnn import RUNS, Predictor
            predictors[m] = Predictor(RUNS / f"{m}_{a.tag}", case)
    if "gnn_genco" in a.methods:
        from .opf_genco import GencoConditioned, find_run
        from .opf_gnn import RUNS
        predictors["gnn_genco"] = GencoConditioned(RUNS / f"gnn_genco_{a.tag}", case, a.genco_run or find_run("genco_pf_case33_lv_prior"))
    rows = []
    for day in a.days:
        for h in a.hours:
            for d in ("export", "import"):
                rows += run_interval(case, sc, opf, day, h, d, a.methods, mopf=mopf, predictors=predictors)
        print(f"day {day} done", flush=True)
    import os
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps({"meta": {"split": a.split, "days": a.days, "methods": a.methods, "S": S_DESIGN,
                                        "fresh": S_FRESH, "eps": EPS, "beta": BETA, "caps": oc.CAP_FACTOR,
                                        "scenario_seed": SCENARIO_SEED}, "rows": rows}, indent=1))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
