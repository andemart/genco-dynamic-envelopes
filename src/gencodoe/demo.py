"""End-to-end demo without GENCO: python -m gencodoe.demo [--case33bw]

Computes a day of export envelopes on a feeder and compares, for the noon
interval, envelope search with exact AC power flow, the LinDistFlow baseline,
an *emulated* surrogate (AC + Gaussian noise, standing in for GENCO until the
trained model is wired up) with conformal screening, and the max-total LP.
Every envelope is verified with AC power flow on fresh scenarios.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import conformal, envelopes, feeder, metrics, scenarios

LOAD_SHAPE = np.array([.55, .5, .48, .47, .48, .55, .7, .8, .75, .65, .6, .58,
                       .57, .57, .6, .68, .8, .95, 1., .97, .9, .8, .7, .6])
PV_SHAPE = np.clip(np.sin(np.pi * (np.arange(24) - 6) / 13), 0, None)


def build(case33bw: bool):
    if case33bw:
        import pandapower.networks as pn
        fd, pd, qd = feeder.from_pandapower(pn.case33bw())
    else:
        fd = feeder.synthetic_feeder(n_trunk=14, n_laterals=3, lateral_len=5, r=0.004, x=0.003, rate=np.inf)
        pd = np.full(fd.n_bus, 0.08)
        pd[0] = 0.0
        qd = 0.3 * pd
    # thermal ratings where the case has none: 1.5x the branch |S| at peak load
    given, fd.rate = fd.rate, np.ones(fd.n_bus)
    s_peak = feeder.sweep_pf(fd, -pd[None], -qd[None]).loading[0]
    fd.rate = np.where(np.isfinite(given), given, 1.5 * s_peak)
    fd.rate[0] = np.inf
    loads = np.flatnonzero(pd > 0)
    cust = envelopes.Customers(bus=loads, export_cap=2.0 * pd[loads], import_cap=1.0 * pd[loads])
    return fd, pd, qd, cust


def scenario_set(fd, pd, qd, hour, n, rng, pv_ratio=0.5):
    g = fd.n_bus
    spec = scenarios.ForecastSpec(
        point={"load": pd * LOAD_SHAPE[hour], "pv": pv_ratio * pd * PV_SHAPE[hour]},
        rel_std={"load": 0.08, "pv": 0.25},
        corr=scenarios.default_corr(g),
        capacity={"load": None, "pv": pv_ratio * pd},
    )
    tan_phi = np.divide(qd, pd, out=np.zeros_like(pd), where=pd > 0)
    return scenarios.net_injection(scenarios.sample(spec, n, rng), tan_phi)


def ac_prob_ok(fd, cust, limit, p, q, direction="export"):
    sign = envelopes.DIRECTIONS[direction]
    return float(envelopes.is_feasible(fd, feeder.sweep_pf(fd, p + sign * limit @ cust.incidence(fd.n_bus), q), direction).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case33bw", action="store_true", help="use pandapower's case33bw instead of the synthetic feeder")
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01, help="conformal miscoverage level")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    fd, pd, qd, cust = build(a.case33bw)
    ac = envelopes.exact(feeder.sweep_pf, fd)
    print(f"feeder: {fd.n_bus} buses, {len(cust)} flexible connection points, "
          f"{a.scenarios} scenarios/interval, eps={a.eps}\n")

    print("hour  export level  total export  import level  P(ok, AC, fresh)")
    for h in range(0, 24, 2):
        p, q = scenario_set(fd, pd, qd, h, a.scenarios, rng)
        exp = envelopes.scalar_envelope(ac, fd.n_bus, cust, p, q, "export", "proportional", a.eps, levels_per_round=4)
        imp = envelopes.scalar_envelope(ac, fd.n_bus, cust, p, q, "import", "proportional", a.eps, levels_per_round=4)
        pt, qt = scenario_set(fd, pd, qd, h, a.scenarios, rng)
        print(f"{h:4d}  {exp.level:12.3f}  {exp.limit.sum():12.3f}  {imp.level:12.3f}  "
              f"{ac_prob_ok(fd, cust, exp.limit, pt, qt):.3f}")

    hour = 12
    p, q = scenario_set(fd, pd, qd, hour, a.scenarios, rng)
    pt, qt = scenario_set(fd, pd, qd, hour, a.scenarios, rng)

    # emulated GENCO: AC + noise; calibrate on random envelope levels from a separate scenario set
    noise_rng = np.random.default_rng(a.seed + 1)

    def surrogate(f, pp, qq):
        r = feeder.sweep_pf(f, pp, qq)
        r.vm = r.vm + noise_rng.normal(0, 0.002, r.vm.shape)
        r.loading = r.loading + noise_rng.normal(0, 0.01, r.loading.shape)
        return r

    pc, qc = scenario_set(fd, pd, qd, hour, 2000, rng)
    pc = pc + envelopes.allocate(cust, "proportional", rng.uniform(0, 1, len(pc))) @ cust.incidence(fd.n_bus)
    q_v, q_l = conformal.calibrate_pf(fd, surrogate(fd, pc, qc), feeder.sweep_pf(fd, pc, qc), a.alpha)
    print(f"\nnoon interval; conformal margins q_v={q_v:.4f} pu, q_l={q_l:.4f}\n")

    rows = []

    def run(name, fn):
        t0 = time.perf_counter()
        env, n_ac = fn()
        dt = time.perf_counter() - t0
        rows.append((name, env, n_ac, dt))

    run("AC power flow", lambda: (lambda e: (e, e.n_evals))(
        envelopes.scalar_envelope(ac, fd.n_bus, cust, p, q, "export", "proportional", a.eps, 4)))
    run("LinDistFlow", lambda: (envelopes.scalar_envelope(
        envelopes.exact(feeder.lindistflow, fd), fd.n_bus, cust, p, q, "export", "proportional", a.eps, 4), 0))
    for verify in (True, False):
        def screened(verify=verify):
            sf = envelopes.ScreenedFeasibility(fd, surrogate, q_v, q_l, feeder.sweep_pf if verify else None)
            return envelopes.scalar_envelope(sf, fd.n_bus, cust, p, q, "export", "proportional", a.eps, 4), sf.n_verified
        run("surrogate+conformal" + (" +AC verify" if verify else " only"), screened)
    run("max-total LP (AC base)", lambda: (envelopes.max_total_envelope(
        fd, cust, feeder.sweep_pf(fd, p, q), "export", a.eps), len(p)))

    ref = rows[0][1].limit
    print(f"{'method':32s} {'total':>7s} {'min':>7s} {'jain':>5s} {'gap':>7s} {'P(ok)':>6s} {'AC calls':>9s} {'time s':>7s}")
    for name, env, n_ac, dt in rows:
        s, g = metrics.envelope_summary(env.limit), metrics.envelope_gap(env.limit, ref)
        print(f"{name:32s} {s['total']:7.3f} {s['min']:7.4f} {s['jain']:5.2f} {g['total_rel_gap']:+7.1%} "
              f"{ac_prob_ok(fd, cust, env.limit, pt, qt):6.3f} {n_ac:9d} {dt:7.2f}")
    print("\nP(ok) is checked with AC power flow on fresh scenarios; target is >= 1 - eps.")


if __name__ == "__main__":
    main()
