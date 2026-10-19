"""GENCO OPF vs IPOPT for max-hosting operating envelopes on case118_flex.

    python scripts/run_opf_envelopes.py [--run <mlflow run dir>] [--out results/genco_opf_case118.md]

On held-out test scenarios of the training run (graphkit's saved split):
1. envelope accuracy: total and per-unit flexible export, GENCO vs IPOPT (datakit's
   labels), next to IPOPT's own per-unit spread between two starting points
2. AC feasibility of GENCO's full solution (exact power flow, reactive limits enforced)
3. runtime: GENCO batched vs IPOPT per scenario
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import juliacall  # noqa: F401  (import before torch: juliacall warns of segfaults otherwise)
import numpy as np

from gencodoe.genco import latest_run
from gencodoe.opf_envelopes import GencoOPF, PowerModelsBaseline, flex_mask, read_tables

WORK = Path(os.environ.get("GENCODOE_WORK", Path.home() / "gencodoe_work"))


def pct(x, q):
    return float(np.quantile(x, q))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--config", default="configs/genco_opf_case118_flex.yaml")
    ap.add_argument("--raw", default=str(WORK / "data_out/case118_flex/raw"))
    ap.add_argument("--case", default="grids/case118_flex.m")
    ap.add_argument("--log_dir", default=str(WORK / "mlruns"))
    ap.add_argument("--n", type=int, default=500, help="test scenarios evaluated")
    ap.add_argument("--n_ipopt", type=int, default=60, help="scenarios re-solved by IPOPT for timing and spread")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/genco_opf_case118.md")
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else latest_run(a.log_dir, "genco-opf-case118-flex")
    split = json.loads((run / "artifacts/stats/case118_flex_scenario_splits.json").read_text())
    test = np.sort(np.asarray(split["test"]))[: a.n]

    t = read_tables(a.raw, scenarios=test)
    gen0 = t["gen"][t["gen"].scenario == test[0]].reset_index(drop=True)
    nb, ng = t["bus"].bus.nunique(), len(gen0)
    n = len(test)
    PD, QD = t["bus"].Pd.to_numpy().reshape(n, nb), t["bus"].Qd.to_numpy().reshape(n, nb)
    PG_true = t["gen"].p_mw.to_numpy().reshape(n, ng)
    VM_true = t["bus"].Vm.to_numpy().reshape(n, nb)
    fl = flex_mask(gen0)
    cap = gen0.max_p_mw.to_numpy()[fl]

    genco = GencoOPF(a.config, run, a.raw, device=a.device)
    genco(PD[:64], QD[:64])                                         # warm-up
    t0 = time.perf_counter()
    pred = genco(PD, QD)
    t_genco = (time.perf_counter() - t0) / n
    F_true, F_pred = PG_true[:, fl], np.clip(pred["pg_mw"][:, fl], 0, cap)

    pm = PowerModelsBaseline(a.case)
    m = min(a.n_ipopt, n)
    pm.opf(PD[:2], QD[:2])                                          # compile
    pg_cold, ok_cold, s_cold = pm.opf(PD[:m], QD[:m])
    rng = np.random.default_rng(0)
    pg_alt, ok_alt, _ = pm.opf(PD[:m], QD[:m], start={"pg_mw": pg_cold * rng.uniform(0.7, 1.3, pg_cold.shape),
                                                         "vm": np.ones((m, nb)), "va": np.zeros((m, nb))})
    both = ok_cold & ok_alt
    spread_unit = np.abs(pg_cold[both][:, fl] - pg_alt[both][:, fl]).mean()
    spread_total = np.abs(pg_cold[both][:, fl].sum(1) / pg_alt[both][:, fl].sum(1) - 1)
    label_gap = np.abs(pg_cold[ok_cold][:, fl] - F_true[:m][ok_cold]).mean()

    pm.check(PD[:2], QD[:2], PG_true[:2], VM_true[:2], fl)          # compile
    pg_check = pred["pg_mw"].copy()
    pg_check[:, fl] = F_pred
    viol, s_check = pm.check(PD, QD, pg_check, pred["vm"], fl)
    viol_true, _ = pm.check(PD[:m], QD[:m], PG_true[:m], VM_true[:m], fl)

    tot_err = F_pred.sum(1) / F_true.sum(1) - 1
    unit_err = np.abs(F_pred - F_true)
    out = ["# GENCO OPF envelopes vs IPOPT, case118_flex", "",
           f"Model run: `{str(run).replace(str(Path.home()), '~')}`; device {genco.device}; {n} held-out test scenarios "
           f"(graphkit split); IPOPT re-solves on {m}.", "",
           "Each scenario's envelope = AC-OPF dispatch of 99 flexible export units (cap 2 x bus load, cost -100 $/MWh "
           "+ 0.01 $/MW^2h) on case118 with the slack as interconnection. Mean total flexible export: "
           f"{F_true.sum(1).mean():,.0f} MW of {cap.sum():,.0f} MW installed.", "",
           "## 1. Envelope accuracy", "",
           "| Quantity | GENCO vs IPOPT | IPOPT vs IPOPT (other start) |", "|---|---|---|",
           f"| Total hosting, mean abs relative error | {np.abs(tot_err).mean():.2%} | {spread_total.mean():.2%} |",
           f"| Total hosting, mean signed error | {tot_err.mean():+.2%} | — |",
           f"| Total hosting, 95th pct abs error | {pct(np.abs(tot_err), .95):.2%} | {pct(spread_total, .95):.2%} |",
           f"| Per-unit export MAE (MW) | {unit_err.mean():.2f} | {spread_unit:.2f} |",
           f"| Per-unit export MAE / mean unit export | {unit_err.mean() / F_true.mean():.1%} | {spread_unit / F_true.mean():.1%} |",
           f"| Units at cap: share correctly classified | {((F_pred >= 0.99 * cap) == (F_true >= 0.99 * cap)).mean():.1%} | — |",
           "", f"Datakit labels vs our cold IPOPT re-solve: per-unit MAE {label_gap:.3f} MW (same solver, same start).", "",
           "## 2. AC feasibility of GENCO's full solution", "",
           "Exact AC power flow with GENCO's dispatch and voltage set-points, reactive limits enforced; worst violation "
           "in pu of voltage (a 5% branch overload counts as 0.05).", "",
           "| Solutions | Feasible (<= 1e-4) | <= 0.01 | <= 0.02 | <= 0.05 | median worst | 95th pct |", "|---|---|---|---|---|---|---|"]
    for name, v in (("IPOPT (labels)", viol_true), ("GENCO", viol)):
        fin = np.where(np.isfinite(v), v, 1.0)
        out.append(f"| {name} | {(fin <= 1e-4).mean():.1%} | {(fin <= .01).mean():.1%} | {(fin <= .02).mean():.1%} | "
                   f"{(fin <= .05).mean():.1%} | {np.median(fin):.4f} | {pct(fin, .95):.4f} |")
    out += ["", "## 3. Runtime per scenario", "", "| Step | ms |", "|---|---|",
            f"| GENCO OPF, batched ({genco.device}) | {1e3 * t_genco:.2f} |",
            f"| IPOPT AC-OPF (PowerModels), median | {1e3 * np.median(s_cold):.0f} |",
            f"| Exact AC check (PowerModels compute_ac_pf + Q limits), median | {1e3 * np.median(s_check):.1f} |",
            "", f"IPOPT / GENCO: {np.median(s_cold) / t_genco:,.0f}x. IPOPT solved {ok_cold.mean():.0%} cold.", ""]
    text = "\n".join(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
