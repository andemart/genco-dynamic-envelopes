"""Partial metering and state estimation on case33bw (docs/genco_opf_protocol.md, amendment 4).

    python -m gencodoe.opf_se calibrate --days 80 ... 99 --estimators wls genco --out results/se_calibration.json
    python -m gencodoe.opf_se run --split test --days 100 --calibration results/se_calibration.json \
        --out data_out/opf_case33/se/day100.json
    python -m gencodoe.opf_se report --dir data_out/opf_case33/se --out results/genco_opf_amendment4_report.json

Per interval: one realised state (seed 303); a share f of the 32 households metered (P, Q, 1% noise; subset per
(day, f), seed 404); |V| at the substation and four fixed buses (0.2% noise). Estimators of the unmeasured homes:
prior (unchanged design draws), WLS (Gauss-Newton on the voltages with the prior mean as pseudo-measurements) and
GENCO-SE (graphkit StateEstimation checkpoint). Each gives 200 design scenarios (metered homes fixed at their
readings; WLS/GENCO: Gaussian around the estimate with the per-bus std calibrated on validation days). Envelope:
LP proposal + stage A exact check and scale-down. Metrics: hosting after the check, realised feasibility at the
realised state, estimation MAE, cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from . import envelopes
from . import feeder as fdr
from . import opf_case33 as oc
from . import opf_envelopes as oe

SEED_REAL, SEED_METER, SEED_NOISE, SEED_SCEN = 303, 404, 405, 505
FRACTIONS = (0.1, 0.3, 0.5, 0.7, 1.0)
V_BUSES = np.array([0, 17, 21, 24, 32])            # substation, feeder end, lateral ends (1-based 1, 18, 22, 25, 33)
NOISE_P, NOISE_V = 0.01, 0.002                     # relative std
ESTIMATORS = ("prior", "wls", "genco")
MIN_STD_MW = 1e-4
SE_CONFIG = Path("configs/genco_se_case33_lv.yaml")


# ----------------------------------------------------------------------------------------- realised state, metering
def realised_state(sc: oc.Scenarios, day: int, hour: int):
    rng = np.random.default_rng([SEED_REAL, day, hour])
    p, q = sc.sample(day, hour, 1, rng)
    return p[0], q[0]


def metered_units(case: oc.Case33, day: int, f: float) -> np.ndarray:
    """Indices (into case.unit_bus) of the metered households for (day, f); sorted."""
    n = len(case.unit_bus)
    k = int(round(f * n))
    rng = np.random.default_rng([SEED_METER, day, int(round(100 * f))])
    return np.sort(rng.choice(n, size=k, replace=False))


def measure(case: oc.Case33, p_real, q_real, metered, day, hour, f) -> dict:
    """Noisy readings: P, Q at the metered homes' buses; |V| at V_BUSES. Values in pu (injection convention)."""
    rng = np.random.default_rng([SEED_NOISE, day, hour, int(round(100 * f))])
    bus = case.unit_bus[metered]
    pm = p_real[bus] * (1 + NOISE_P * rng.standard_normal(len(bus)))
    qm = q_real[bus] * (1 + NOISE_P * rng.standard_normal(len(bus)))
    vm = fdr.sweep_pf(case.fd, p_real[None], q_real[None]).vm[0, V_BUSES]
    vm = vm * (1 + NOISE_V * rng.standard_normal(len(V_BUSES)))
    return {"bus": bus, "p": pm, "q": qm, "v_bus": V_BUSES, "vm": vm,
            "unmeasured": np.setdiff1d(case.unit_bus, bus)}


# ------------------------------------------------------------------------------------------------------ estimators
def prior_moments(p_design, q_design, bus):
    return p_design[:, bus].mean(0), q_design[:, bus].mean(0), p_design[:, bus].std(0), q_design[:, bus].std(0)


def estimate_wls(case: oc.Case33, p_design, q_design, meas, iters: int = 6):
    """Gauss-Newton WLS of the unmeasured homes' (p, q) from the voltage measurements with the prior mean as
    pseudo-measurements (weight 1 / prior variance). Sensitivities by finite differences of the exact sweep.
    Returns (p_hat, q_hat) at meas['unmeasured']."""
    fd = case.fd
    U = meas["unmeasured"]
    if len(U) == 0:
        return np.zeros(0), np.zeros(0)
    mp, mq, sp, sq = prior_moments(p_design, q_design, U)
    sp, sq = np.maximum(sp, MIN_STD_MW), np.maximum(sq, MIN_STD_MW)
    vb = meas["v_bus"][1:]                                       # the substation is pinned: no information
    w_v = 1.0 / (NOISE_V * np.maximum(meas["vm"][1:], 0.5)) ** 2
    x = np.concatenate([mp, mq])
    x0, w_x = x.copy(), 1.0 / np.concatenate([sp, sq]) ** 2
    base_p, base_q = np.zeros(fd.n_bus), np.zeros(fd.n_bus)
    base_p[meas["bus"]], base_q[meas["bus"]] = meas["p"], meas["q"]
    m = len(U)
    delta = 1e-3

    def states(xv):
        p, q = base_p.copy(), base_q.copy()
        p[U], q[U] = xv[:m], xv[m:]
        return p, q

    for _ in range(iters):
        p, q = states(x)
        P = np.repeat(p[None], 2 * m + 1, 0)
        Q = np.repeat(q[None], 2 * m + 1, 0)
        for j in range(m):
            P[1 + j, U[j]] += delta
            Q[1 + m + j, U[j]] += delta
        vm = fdr.sweep_pf(fd, P, Q).vm[:, vb]
        h, J = vm[0], (vm[1:] - vm[0]) / delta                   # (2m, n_v)
        J = J.T                                                  # (n_v, 2m)
        r = meas["vm"][1:] - h
        A = J.T @ (w_v[:, None] * J) + np.diag(w_x)
        b = J.T @ (w_v * r) + w_x * (x0 - x)
        dx = np.linalg.solve(A, b)
        x = x + dx
        if np.abs(dx).max() < 1e-7:
            break
    return x[:m], x[m:]


class GencoSE:
    """graphkit StateEstimation checkpoint: measured P, Q at the metered buses and |V| at V_BUSES in, voltages and
    physics-decoded net injections out. Mirrors graphkit's SimulateMeasurements + ApplyMasking with our pattern."""

    def __init__(self, run: Path, device: str = "cpu"):
        import torch
        from .datakit_case33 import NAME
        from .genco import GencoModel, frames_to_graphs, read_raw
        from .opf_genco import DATAKIT
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        self.model = GencoModel(SE_CONFIG, run / "artifacts/model/best_model_state_dict.pt",
                                run / "artifacts/stats/normalizer_stats.pt", device=device, network=NAME)
        from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches, RemoveInactiveGenerators
        g = frames_to_graphs(*read_raw(DATAKIT / NAME / "raw", scenarios=[0]))[0]
        self.template = RemoveInactiveGenerators()(RemoveInactiveBranches()(g))   # graphkit's SE preprocessing
        self.n_bus = self.template["bus"].num_nodes

    def graph(self, meas):
        import torch
        from gridfm_graphkit.datasets.globals import PD_H, PG_H, QD_H, QG_H, VA_H, VM_H
        g = self.template.clone()
        x = g["bus"].x
        n_y = g["bus"].y.size(1)
        x[:, :n_y] = 0.0                                          # nothing known but the measurements
        x[meas["bus"], PD_H] = torch.as_tensor(-meas["p"], dtype=x.dtype)      # consumption, MW on 1 MVA
        x[meas["bus"], QD_H] = torch.as_tensor(-meas["q"], dtype=x.dtype)
        x[meas["v_bus"], VM_H] = torch.as_tensor(meas["vm"], dtype=x.dtype)
        mask = torch.ones(self.n_bus, n_y, dtype=torch.bool)
        mask[meas["bus"], PD_H] = False
        mask[meas["bus"], QD_H] = False
        mask[meas["bus"], QG_H] = False                           # no units with Q: QG = 0 is known there
        mask[meas["v_bus"], VM_H] = False
        mask[:, VA_H] = True
        gx = g["gen"].x
        gmask = torch.ones(gx.size(0), g["gen"].y.size(1), dtype=torch.bool)
        gx[:, :g["gen"].y.size(1)] = 0.0
        ea = g[("bus", "connects", "bus")].edge_attr
        bmask = torch.ones(ea.size(0), 2, dtype=torch.bool)
        ea[:, :2] = 0.0
        self.model.normalizer.transform(g)
        g.mask_dict = {"bus": torch.nn.functional.pad(mask, (0, x.size(1) - n_y)),
                       "gen": torch.nn.functional.pad(gmask, (0, gx.size(1) - g["gen"].y.size(1))),
                       "branch": torch.nn.functional.pad(bmask, (0, ea.size(1) - 2))}
        return g

    def estimate(self, meas):
        """(p_hat, q_hat) net injections (pu, injection convention) at meas['unmeasured'], and vm at every bus."""
        import torch
        from torch_geometric.data import Batch
        g = self.graph(meas)
        batch = Batch.from_data_list([g, g]).to(self.model.device)
        with torch.no_grad():
            out = self.model.model(batch)
        self.model.normalizer.inverse_output(out, batch)
        bus = out["bus"].reshape(2, self.n_bus, -1)[0].cpu().numpy()
        U = meas["unmeasured"]
        return bus[U, 2].astype(float), bus[U, 3].astype(float), bus[:, 0].astype(float)


# ------------------------------------------------------------------------------------------------------ scenarios
def scenarios_prior(p_design, q_design, meas):
    P, Q = p_design.copy(), q_design.copy()
    P[:, meas["bus"]], Q[:, meas["bus"]] = meas["p"], meas["q"]
    return P, Q


def scenarios_from_estimate(p_design, q_design, meas, p_hat, q_hat, std_p, std_q, rng):
    """Metered homes fixed; unmeasured homes Gaussian around the estimate with the calibrated per-bus std,
    truncated to the sign of the estimate."""
    P, Q = scenarios_prior(p_design, q_design, meas)
    U = meas["unmeasured"]
    S = len(P)
    zp = p_hat + std_p * rng.standard_normal((S, len(U)))
    zq = q_hat + std_q * rng.standard_normal((S, len(U)))
    P[:, U] = np.where(p_hat >= 0, np.maximum(zp, 0), np.minimum(zp, 0))
    Q[:, U] = np.where(q_hat >= 0, np.maximum(zq, 0), np.minimum(zq, 0))
    return P, Q


# ----------------------------------------------------------------------------------------------------- evaluation
def envelope(case, cust, P, Q, direction):
    xs = oe.proposal_lp(case, cust, P, Q, direction)
    x = np.quantile(xs, oe.EPS, axis=0)
    limit, lam, share, evals = oe.scale_down(case, cust, x, P, Q, direction)
    return limit, lam, share, evals


def realised_ok(case, cust, limit, p_real, q_real, direction) -> bool:
    inj = oe.SIGN[direction] * limit @ cust.incidence(case.fd.n_bus)
    return bool(envelopes.exact(fdr.sweep_pf, case.fd)(p_real[None] + inj, q_real[None], direction)[0])


class Study:
    def __init__(self, genco_run: Path | None = None, calibration: dict | None = None):
        self.case = oc.build()
        self.sc = oc.Scenarios(self.case, 0)
        self.cust = oe.customers(self.case)
        self.cal = calibration or {}
        self.genco = GencoSE(genco_run) if genco_run is not None else None

    def estimates(self, est, p_design, q_design, meas):
        """(p_hat, q_hat) at the unmeasured homes for one estimator, plus the estimator's seconds."""
        t = time.perf_counter()
        U = meas["unmeasured"]
        if est == "prior":
            mp, mq, _, _ = prior_moments(p_design, q_design, U)
            out = (mp, mq)
        elif est == "wls":
            out = estimate_wls(self.case, p_design, q_design, meas)
        elif est == "genco":
            out = self.genco.estimate(meas)[:2]
        else:
            raise ValueError(est)
        return out, time.perf_counter() - t

    def std_for(self, est, f, U):
        """Calibrated per-bus std (p, q) for the unmeasured homes; index by unit position."""
        c = self.cal[est][f"{f:.1f}"]
        pos = np.searchsorted(self.case.unit_bus, U)
        return np.array(c["std_p"])[pos], np.array(c["std_q"])[pos]

    def interval(self, day, hour, f, estimators, directions=("export", "import")):
        case, cust = self.case, self.cust
        p_real, q_real = realised_state(self.sc, day, hour)
        p_design, q_design, _, _ = oe.interval_draws(self.sc, day, hour)
        metered = metered_units(case, day, f)
        meas = measure(case, p_real, q_real, metered, day, hour, f)
        U = meas["unmeasured"]
        rows = []
        for est in estimators:
            (p_hat, q_hat), est_s = self.estimates(est, p_design, q_design, meas)
            mae = float(np.abs(p_hat - p_real[U]).mean() * 1e3) if len(U) else 0.0   # kW
            if est == "prior" or len(U) == 0:
                P, Q = scenarios_prior(p_design, q_design, meas)
            else:
                sp, sq = self.std_for(est, f, U)
                rng = np.random.default_rng([SEED_SCEN, day, hour, int(round(100 * f))])
                P, Q = scenarios_from_estimate(p_design, q_design, meas, p_hat, q_hat, sp, sq, rng)
            for d in directions:
                t = time.perf_counter()
                limit, lam, share, evals = envelope(case, cust, P, Q, d)
                rows.append({"day": day, "hour": hour, "f": f, "estimator": est, "direction": d,
                             "total_mw": float(limit.sum()), "limit_mw": limit.round(6).tolist(), "lambda": lam,
                             "design_share": share, "realised_ok": realised_ok(case, cust, limit, p_real, q_real, d),
                             "n_unmeasured": int(len(U)), "mae_unmeasured_kw": mae, "estimate_s": est_s,
                             "envelope_s": time.perf_counter() - t, "exact_evals": evals})
        return rows

    def calibrate(self, days, estimators, fractions=FRACTIONS):
        """Residual std of each estimator per unit, per f, over the intervals where the unit was unmeasured."""
        case = self.case
        n = len(case.unit_bus)
        out = {est: {} for est in estimators}
        for f in fractions:
            res = {est: ([[] for _ in range(n)], [[] for _ in range(n)]) for est in estimators}
            for day in days:
                metered = metered_units(case, day, f)
                pos = np.setdiff1d(np.arange(n), metered)
                if len(pos) == 0:
                    continue
                for hour in range(24):
                    p_real, q_real = realised_state(self.sc, day, hour)
                    p_design, q_design, _, _ = oe.interval_draws(self.sc, day, hour)
                    meas = measure(case, p_real, q_real, metered, day, hour, f)
                    U = meas["unmeasured"]
                    for est in estimators:
                        (p_hat, q_hat), _ = self.estimates(est, p_design, q_design, meas)
                        for k, u in zip(pos, U):
                            res[est][0][k].append(p_hat[list(U).index(u)] - p_real[u])
                            res[est][1][k].append(q_hat[list(U).index(u)] - q_real[u])
            for est in estimators:
                sp = [float(np.sqrt(np.mean(np.square(r)))) if r else float("nan") for r in res[est][0]]
                sq = [float(np.sqrt(np.mean(np.square(r)))) if r else float("nan") for r in res[est][1]]
                sp = np.where(np.isnan(sp), np.nanmean(sp) if np.isfinite(np.nanmean(sp)) else MIN_STD_MW, sp)
                sq = np.where(np.isnan(sq), np.nanmean(sq) if np.isfinite(np.nanmean(sq)) else MIN_STD_MW, sq)
                out[est][f"{f:.1f}"] = {"std_p": np.maximum(sp, MIN_STD_MW).tolist(),
                                        "std_q": np.maximum(sq, MIN_STD_MW).tolist(),
                                        "n_intervals": int(sum(len(r) for r in res[est][0]))}
        return out


# ------------------------------------------------------------------------------------------------------------ gate
SEED_GATE, N_GATE, GATE = 606, 2000, {"vm_mae": 2e-3}


def gate(genco_run: Path, days=tuple(oc.SPLIT["val"]), out: Path = Path("results/genco_se_case33_gate.json")) -> dict:
    """Amendment 4 gate: on N_GATE states from the validation days' scenario distribution with a random metered half
    (f = 0.5) and noisy measurements, GENCO-SE must reach Vm MAE < 2e-3 pu (all buses) and an MAE of the unmeasured
    active loads below the prior mean's."""
    case = oc.build()
    sc = oc.Scenarios(case, 0)
    model = GencoSE(genco_run)
    rng = np.random.default_rng(SEED_GATE)
    dv, e_g, e_p = [], [], []
    for k in range(N_GATE):
        day, hour = int(rng.choice(days)), int(rng.integers(24))
        p, q = sc.sample(day, hour, 1, rng)
        p, q = p[0], q[0]
        metered = np.sort(rng.choice(len(case.unit_bus), size=len(case.unit_bus) // 2, replace=False))
        meas = measure(case, p, q, metered, day, hour + 100 * k, 0.5)       # distinct noise stream per state
        U = meas["unmeasured"]
        ph, _, vm = model.estimate(meas)
        exact = fdr.sweep_pf(case.fd, p[None], q[None]).vm[0]
        dv.append(np.abs(vm - exact))
        e_g.append(np.abs(ph - p[U]))
        pd_, _, _, _ = oe.interval_draws(sc, day, hour)
        e_p.append(np.abs(pd_[:, U].mean(0) - p[U]))
    dv = np.concatenate(dv)
    res = {"run": str(genco_run), "n_states": N_GATE, "vm_mae": float(dv.mean()), "vm_p99": float(np.percentile(dv, 99)),
           "vm_max": float(dv.max()), "mae_unmeasured_kw_genco": float(np.concatenate(e_g).mean() * 1e3),
           "mae_unmeasured_kw_prior": float(np.concatenate(e_p).mean() * 1e3), "gate": GATE}
    res["pass"] = bool(res["vm_mae"] < GATE["vm_mae"] and res["mae_unmeasured_kw_genco"] < res["mae_unmeasured_kw_prior"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------------------------------------- report
def report(se_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    rows = []
    for f in sorted(glob.glob(str(Path(se_dir) / "day*.json"))):
        rows += json.loads(Path(f).read_text())["rows"]
    days = sorted({r["day"] for r in rows})
    ests = [e for e in ESTIMATORS if any(r["estimator"] == e for r in rows)]
    fr = sorted({r["f"] for r in rows})
    rng = np.random.default_rng(seed)

    def tot(est, f, direction=None):
        t = {d: 0.0 for d in days}
        for r in rows:
            if r["estimator"] == est and r["f"] == f and (direction is None or r["direction"] == direction):
                t[r["day"]] += r["total_mw"]
        return np.array([t[d] for d in days])

    def boot(num, den):
        idx = rng.integers(0, len(num), size=(n_boot, len(num)))
        q = num[idx].sum(1) / den[idx].sum(1)
        return {"point": float(num.sum() / den.sum()), "lo": float(np.percentile(q, 2.5)), "hi": float(np.percentile(q, 97.5))}

    summary = {}
    for est in ests:
        summary[est] = {}
        for f in fr:
            sel = [r for r in rows if r["estimator"] == est and r["f"] == f]
            full = tot(ests[0], 1.0) if 1.0 in fr else None
            summary[est][f"{f:.1f}"] = {
                "hosting_mwh_per_day": float(tot(est, f).sum() / len(days)),
                "export_mwh_per_day": float(tot(est, f, "export").sum() / len(days)),
                "import_mwh_per_day": float(tot(est, f, "import").sum() / len(days)),
                "ratio_to_full_metering": boot(tot(est, f), full) if full is not None else None,
                "ratio_to_prior": boot(tot(est, f), tot("prior", f)) if "prior" in ests else None,
                "realised_feasibility": float(np.mean([r["realised_ok"] for r in sel])),
                "mae_unmeasured_kw": float(np.mean([r["mae_unmeasured_kw"] for r in sel])),
                "estimate_s": float(np.mean([r["estimate_s"] for r in sel])),
                "envelope_s": float(np.mean([r["envelope_s"] for r in sel]))}
    out = {"days": days, "n_rows": len(rows), "estimators": ests, "fractions": fr, "summary": summary,
           "bootstrap": {"replicates": n_boot, "seed": seed, "block": "day"}}
    if "genco" in ests and "wls" in ests:
        c = {}
        for f in (0.3, 0.5):
            if f in fr:
                r = boot(tot("genco", f), tot("wls", f))
                c[f"{f:.1f}"] = {"ratio_genco_to_wls": r, "genco_realised_feasibility": summary["genco"][f"{f:.1f}"]["realised_feasibility"],
                                 "pass": r["point"] >= 1.02 and r["lo"] > 1.0 and summary["genco"][f"{f:.1f}"]["realised_feasibility"] >= 0.95}
        c["pass"] = bool(c) and all(v["pass"] for k, v in c.items() if k != "pass")
        out["criterion_2_genco_beats_wls"] = c
    out["unsafe"] = [(est, f) for est in ests for f in fr if summary[est][f"{f:.1f}"]["realised_feasibility"] < 0.95]
    return out


def table(out: dict) -> str:
    L = [f"{'est':6} {'f':>4} {'MWh/d':>7} {'exp':>7} {'imp':>6} {'/full':>6} {'/prior':>7} {'real.ok':>7} {'MAE kW':>7} {'est s':>6}"]
    for est, per in out["summary"].items():
        for f, s in per.items():
            rf = s["ratio_to_full_metering"]["point"] if s["ratio_to_full_metering"] else float("nan")
            rp = s["ratio_to_prior"]["point"] if s["ratio_to_prior"] else float("nan")
            L.append(f"{est:6} {f:>4} {s['hosting_mwh_per_day']:7.2f} {s['export_mwh_per_day']:7.2f} {s['import_mwh_per_day']:6.2f} "
                     f"{rf:6.3f} {rp:7.3f} {s['realised_feasibility']:7.3f} {s['mae_unmeasured_kw']:7.2f} {s['estimate_s']:6.3f}")
    if "criterion_2_genco_beats_wls" in out:
        L.append(f"criterion 2 (GENCO-SE beats WLS at f=0.3 and 0.5): {'PASS' if out['criterion_2_genco_beats_wls']['pass'] else 'FAIL'}")
    L.append(f"unsafe (realised feasibility < 0.95): {out['unsafe']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "calibrate", "run", "report"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--days", type=int, nargs="+", default=[])
    ap.add_argument("--hours", type=int, nargs="+", default=list(range(24)))
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATORS))
    ap.add_argument("--genco-run", type=Path, default=None)
    ap.add_argument("--calibration", type=Path, default=Path("results/se_calibration.json"))
    ap.add_argument("--dir", type=Path, default=Path("data_out/opf_case33/se"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "report":
        out = report(a.dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print(table(out))
        return
    if any(d not in oc.SPLIT[a.split] for d in a.days):
        raise SystemExit(f"days {a.days} are not all in split {a.split}")
    genco_run = None
    if "genco" in a.estimators:
        from .opf_genco import find_run
        genco_run = a.genco_run or find_run("genco_se_case33_lv")
    if a.cmd == "gate":
        r = gate(genco_run, out=a.out)
        print(json.dumps(r, indent=1))
        raise SystemExit(0 if r["pass"] else 5)
    if a.cmd == "calibrate":
        if a.split != "val":
            raise SystemExit("calibration uses validation days only")
        st = Study(genco_run)
        cal = st.calibrate(a.days, [e for e in a.estimators if e != "prior"], a.fractions)
        cal["meta"] = {"days": a.days, "genco_run": str(genco_run)}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(cal, indent=1))
        print(f"wrote {a.out}")
        return
    cal = json.loads(a.calibration.read_text()) if a.calibration.exists() else {}
    if any(e not in cal for e in a.estimators if e != "prior"):
        raise SystemExit(f"calibration {a.calibration} is missing an estimator")
    st = Study(genco_run, cal)
    rows = []
    for day in a.days:
        for h in a.hours:
            for f in a.fractions:
                rows += st.interval(day, h, f, a.estimators)
        print(f"day {day} done", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps({"meta": {"split": a.split, "days": a.days, "fractions": a.fractions, "estimators": a.estimators,
                                        "genco_run": str(genco_run), "calibration": str(a.calibration), "v_buses": V_BUSES.tolist(),
                                        "noise": {"p_rel": NOISE_P, "v_rel": NOISE_V}, "seeds": [SEED_REAL, SEED_METER, SEED_NOISE, SEED_SCEN]},
                               "rows": rows}, indent=1))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
