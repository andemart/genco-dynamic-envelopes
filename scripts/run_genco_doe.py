"""Dynamic operating envelopes on case33bw with a trained GENCO model.

    python scripts/run_genco_doe.py [--run <mlflow run dir>] [--out results/genco_case33bw.md]

1. GENCO power flow accuracy against the AC sweep on fresh envelope-like rows
2. Conformal calibration (separate rows) and screening quality
3. Hourly export and import envelopes for one day: AC, LinDistFlow, raw GENCO,
   GENCO + conformal + AC verification, GENCO + conformal only; every envelope
   checked with AC power flow on fresh scenarios
4. Runtime per power flow: GENCO batched, AC sweep, LinDistFlow, pandapower Newton-Raphson
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from gencodoe import conformal, demo, envelopes, feeder, metrics
from gencodoe.genco import GencoOracle, latest_run

# generated data and MLflow runs live outside the repo (tens of thousands of small files)
WORK = Path(os.environ.get("GENCODOE_WORK", Path.home() / "gencodoe_work"))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def accuracy(fd, g, t):
    pq = np.arange(1, fd.n_bus)
    rated = pq[np.isfinite(fd.rate[pq])]
    lines = ["| Quantity | MAE | 99th pct abs error | max abs error |", "|---|---|---|---|"]
    for name, a, b in [("Vm (pu)", g.vm[:, pq], t.vm[:, pq]),
                       ("Va (deg)", np.degrees(g.va[:, pq]), np.degrees(t.va[:, pq])),
                       ("Branch loading", g.loading[:, rated], t.loading[:, rated])]:
        e = np.abs(a - b)
        lines.append(f"| {name} | {e.mean():.5f} | {np.quantile(e, 0.99):.5f} | {e.max():.5f} |")
    gv = feeder.branch_flows_from_voltages(fd, g.vm, g.va)
    e = np.abs(gv.loading[:, rated] - t.loading[:, rated])
    lines.append(f"| Branch loading, flows from voltage differences | {e.mean():.5f} | {np.quantile(e, 0.99):.5f} | {e.max():.5f} |")
    lines += ["", "| True Vm bin | Vm MAE | count |", "|---|---|---|"]
    for lab, mae, cnt in metrics.mae_by_bin(g.vm[:, pq], t.vm[:, pq], (0.8, 0.94, 0.97, 1.0, 1.03, 1.06, 1.1, 1.2)):
        lines.append(f"| {lab} | {mae:.5f} | {cnt} |")
    lines += ["", "| True loading bin | Loading MAE | count |", "|---|---|---|"]
    for lab, mae, cnt in metrics.mae_by_bin(g.loading[:, rated], t.loading[:, rated], (0, 0.5, 0.9, 1.0, 1.1, np.inf)):
        lines.append(f"| {lab} | {mae:.5f} | {cnt} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="MLflow run dir (default: latest genco-pf-case33bw run)")
    ap.add_argument("--config", default="configs/genco_pf_case33bw.yaml")
    ap.add_argument("--raw", default=str(WORK / "data_doe/case33bw/raw"))
    ap.add_argument("--log_dir", default=str(WORK / "mlruns"))
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--cal", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/genco_case33bw.md")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--flows", default="injections", choices=["injections", "voltages"])
    ap.add_argument("--bench", type=int, default=20000)
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else latest_run(a.log_dir)
    rng = np.random.default_rng(a.seed)
    fd, pd, qd, cust = demo.build(case33bw=True)
    genco = GencoOracle(a.config, run, a.raw, device=a.device, flows=a.flows)
    out = [f"# GENCO dynamic operating envelopes, case33bw", "",
           f"Model run: `{str(run).replace(str(Path.home()), '~')}`; device: {genco.device}; {a.scenarios} scenarios per interval; "
           f"eps = {a.eps}; conformal alpha = {a.alpha}; GENCO branch flows from {a.flows}.", ""]

    # 1. accuracy on fresh rows (different seed from the training data)
    pt, qt = demo.envelope_rows(fd, pd, qd, cust, a.test, rng, batch=50)   # ~100 distinct hours
    truth, g = feeder.sweep_pf(fd, pt, qt), genco(fd, pt, qt)
    out += ["## 1. Power flow accuracy on fresh envelope-like rows", ""] + accuracy(fd, g, truth) + [""]

    # 2. conformal calibration and screening
    pc, qc = demo.envelope_rows(fd, pd, qd, cust, a.cal, rng, batch=50)
    q_v, q_l = conformal.calibrate_pf(fd, genco(fd, pc, qc), feeder.sweep_pf(fd, pc, qc), a.alpha)
    rated = np.isfinite(fd.rate)
    rated[0] = False
    out += ["## 2. Conformal margins and screening", "",
            f"q_v = {q_v:.4f} pu, q_l = {q_l:.4f}. Coverage on test rows: Vm "
            f"{conformal.coverage(g.vm[:, 1:], truth.vm[:, 1:], q_v):.4f}, loading "
            f"{conformal.coverage(g.loading[:, rated], truth.loading[:, rated], q_l):.4f} (target {1 - a.alpha}).", "",
            "| Direction | Sent to AC | Missed violations | False alarms |", "|---|---|---|---|"]
    for d in ("export", "import"):
        rep = metrics.screening_report(conformal.screen(fd, g, q_v, q_l, d), envelopes.is_feasible(fd, truth, d))
        out.append(f"| {d} | {rep['share_sent_to_ac']:.3f} | {rep['missed_violation_rate']:.4f} | {rep['false_alarm_rate']:.4f} |")
    out.append("")

    # 3. hourly envelopes
    methods = {
        "AC power flow": lambda: (envelopes.exact(feeder.sweep_pf, fd), None),
        "LinDistFlow": lambda: (envelopes.exact(feeder.lindistflow, fd), None),
        "GENCO raw": lambda: (envelopes.exact(genco, fd), None),
        "GENCO + conformal + AC verify": lambda: (lambda s: (s, s))(
            envelopes.ScreenedFeasibility(fd, genco, q_v, q_l, feeder.sweep_pf)),
        "GENCO + conformal only": lambda: (envelopes.ScreenedFeasibility(fd, genco, q_v, q_l, None), None),
    }
    agg = {(m, d): {"gap": [], "ok": [], "ac": 0, "t": 0.0} for m in methods for d in ("export", "import")}
    base_bad = {"export": 0, "import": 0}
    hourly = []
    for h in np.linspace(0, 23, a.hours).round().astype(int):
        p, q = demo.scenario_set(fd, pd, qd, h, a.scenarios, rng)
        pf, qf = demo.scenario_set(fd, pd, qd, h, a.scenarios, rng)
        for d in ("export", "import"):
            # intervals where the feeder already violates with zero flexible exchange get a zero
            # envelope from every method; their P(ok) describes the base case, not the envelope
            bad = demo.ac_prob_ok(fd, cust, np.zeros(len(cust)), pf, qf, d) < 1 - a.eps
            base_bad[d] += bad
            ref = None
            for m, make in methods.items():
                feas, screened = make()
                env, dt = timed(lambda: envelopes.scalar_envelope(feas, fd.n_bus, cust, p, q, d, "proportional", a.eps, 4))
                ref = env.limit if ref is None else ref
                gap = metrics.envelope_gap(env.limit, ref)["total_rel_gap"] if ref.sum() > 0 else 0.0
                ok = np.nan if bad else demo.ac_prob_ok(fd, cust, env.limit, pf, qf, d)
                n_ac = env.n_evals if m == "AC power flow" else (screened.n_verified if screened else 0)
                r = agg[(m, d)]
                r["gap"].append(gap), r["ok"].append(ok)
                r["ac"] += n_ac
                r["t"] += dt
                hourly.append((h, d, m, env.limit.sum(), gap, ok))

    out += [f"## 3. Envelopes over {a.hours} hourly intervals (proportional policy)", "",
            "| Method | Direction | Mean abs gap vs AC (total) | Worst gap | Mean P(ok) | Min P(ok) | AC power flows | Search time s |",
            "|---|---|---|---|---|---|---|---|"]
    for (m, d), r in agg.items():
        gp = np.abs(r["gap"])
        out.append(f"| {m} | {d} | {gp.mean():.2%} | {np.array(r['gap'])[gp.argmax()]:+.2%} | {np.nanmean(r['ok']):.3f} | "
                   f"{np.nanmin(r['ok']):.3f} | {r['ac']:,} | {r['t']:.1f} |")
    out += ["", "P(ok): share of fresh scenarios in which AC power flow finds no violation at the envelope "
            f"(target >= {1 - a.eps}). Intervals where the feeder already violates with zero flexible exchange "
            f"(envelope 0 for every method) are excluded from P(ok): {base_bad['export']} export and "
            f"{base_bad['import']} import intervals.", "", "<details><summary>Per-hour totals (pu)</summary>", "",
            "| Hour | Direction | Method | Total | Gap | P(ok) |", "|---|---|---|---|---|---|"]
    out += [f"| {h} | {d} | {m} | {tot:.3f} | {gap:+.2%} | {ok:.3f} |" for h, d, m, tot, gap, ok in hourly]
    out += ["", "</details>", ""]

    # 4. runtime
    n = a.bench
    pb, qb = demo.envelope_rows(fd, pd, qd, cust, n, rng, batch=50)
    genco(fd, pb[:min(n, 4096)], qb[:min(n, 4096)])                                   # warm-up
    _, t_g = timed(lambda: genco(fd, pb, qb))
    _, t_s = timed(lambda: feeder.sweep_pf(fd, pb, qb))
    _, t_l = timed(lambda: feeder.lindistflow(fd, pb, qb))
    import pandapower as pp
    import pandapower.networks as pn
    net = pn.case33bw()
    m_pp = 100
    t0 = time.perf_counter()
    for i in range(m_pp):
        net.load.p_mw = -pb[i, net.load.bus.to_numpy()]
        net.load.q_mvar = -qb[i, net.load.bus.to_numpy()]
        pp.runpp(net)
    t_pp = (time.perf_counter() - t0) / m_pp
    out += ["## 4. Runtime per power flow", "", "| Solver | µs per power flow |", "|---|---|",
            f"| GENCO ({genco.device}, batched) | {1e6 * t_g / n:.1f} |",
            f"| AC backward/forward sweep (numpy, batched) | {1e6 * t_s / n:.1f} |",
            f"| LinDistFlow (numpy, batched) | {1e6 * t_l / n:.2f} |",
            f"| pandapower Newton-Raphson (one at a time) | {1e6 * t_pp:.0f} |", ""]

    text = "\n".join(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
