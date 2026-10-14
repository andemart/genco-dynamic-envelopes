"""Full-day import/export DOEs on ENWL1 for held-out Fluvius days (Stage F).

Per 15-min interval t of a test day:
  * forecast scenarios: every customer keeps one technology group for the day
    (drawn once from a fixed penetration mix); the uncertainty is which real
    meter of that group the customer behaves like at step t. n_s draws of the
    meters at the same step form the scenario set (weather/coincidence kept);
  * envelope: uniform policy (same limit for every connection point, capped at
    FLEX_EXPORT_KW / FLEX_IMPORT_KW), P(feasible) >= 1 - eps over the scenarios;
  * verification: exact AC on a fresh, independent set of n_s meter draws.

Methods: exact AC sweep, LinDistFlow, GENCO alone, GENCO + conformal screen
(uncertain rows rejected), GENCO + conformal screen + AC verify. Conformal
margins come from the calibration days only (conformal_margins()).

Usage:
  python -m gencodoe.doe_day --day 2024-07-03 --run mlruns/<exp>/<run> --config configs/... \
      --calib data_out/datakit_v2_calib --out results/doe_2024-07-03.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import conformal, envelopes
from . import feeder as fdr
from .datakit_fluvius import FLEX_EXPORT_KW, FLEX_IMPORT_KW, TAN_PHI, day_split
from .enwl import load_enwl1
from .fluvius import GROUPS, load_group

MIX = {"hp": 0.3, "ev": 0.3, "pv": 0.4}   # assumed penetration for the DOE study


class DayScenarios:
    """Real-meter scenario sets for one feeder-day."""

    def __init__(self, customer_bus: np.ndarray, n_bus: int, seed: int = 0, mix: dict = MIX):
        rng = np.random.default_rng(seed)
        groups = [load_group(g) for g in GROUPS]
        self.by_flags = {(g.hp, g.ev, g.pv): g for g in groups}
        self.time = groups[0].time
        self.flags = rng.uniform(size=(len(customer_bus), 3)) < np.array([mix["hp"], mix["ev"], mix["pv"]])
        self.keys = [tuple(int(v) for v in f) for f in self.flags]
        self.customer_bus, self.n_bus = customer_bus, n_bus

    def sample(self, step: int, n: int, rng: np.random.Generator, base_mva: float = 1.0):
        """(p, q) of shape (n, n_bus), pu, positive = export."""
        p_kw = np.empty((n, len(self.keys)))
        q_kvar = np.empty_like(p_kw)
        for key in set(self.keys):
            g = self.by_flags[key]
            who = np.flatnonzero([k == key for k in self.keys])
            for s in range(n):
                pick = rng.choice(len(g.ean), size=len(who), replace=False)
                off, inj = g.offtake_kw[pick, step], g.injection_kw[pick, step]
                p_kw[s, who], q_kvar[s, who] = off - inj, off * TAN_PHI
        p = np.zeros((n, self.n_bus))
        q = np.zeros((n, self.n_bus))
        np.add.at(p, (slice(None), self.customer_bus), -p_kw / 1e3 / base_mva)
        np.add.at(q, (slice(None), self.customer_bus), -q_kvar / 1e3 / base_mva)
        return p, q


def conformal_margins(model, feeder: fdr.Feeder, calib_dir: Path, alpha: float, max_states: int | None = None):
    """Global conformal quantiles (q_v pu, q_l loading) of GENCO on the calibration days."""
    from .evaluate import _loading, load_truth
    from .genco import frames_to_graphs

    truth = load_truth(calib_dir, None if max_states is None else range(max_states))
    names = set(truth["draws"]["split"]) if "split" in truth["draws"] else None
    if names is not None and names != {"calib"}:
        raise ValueError(f"calibration data contains splits {names}")
    vm, va = model.predict_graphs(frames_to_graphs(*truth["frames"]))
    pred = fdr.PFResult(vm=vm, loading=_loading(feeder, truth, vm, va), p_branch=np.zeros_like(vm))
    true = fdr.PFResult(vm=truth["vm"], loading=_loading(feeder, truth, truth["vm"], truth["va"]),
                        p_branch=np.zeros_like(vm))
    q_v, q_l = conformal.calibrate_pf(feeder, pred, true, alpha)
    return float(q_v), float(q_l), int(len(vm))


def run_day(day: str, feeder, customers, scen: DayScenarios, methods: dict, n_s: int, eps: float,
            levels_per_round: int, seed: int, steps_per_hour: int = 4) -> list[dict]:
    """Envelopes for every interval of `day` with every method; AC-verified on fresh draws."""
    rng = np.random.default_rng(seed)
    # days are indexed by row position (96 steps per local day; file timestamps are UTC)
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
    if day_split(t0) != "test":
        raise ValueError(f"{day} is a {day_split(t0)} day, not a test day")
    ac_feas = envelopes.exact(fdr.sweep_pf, feeder)
    rows = []
    for k in range(0, 96, 4 // steps_per_hour):
        step = t0 + k
        p, q = scen.sample(step, n_s, rng, feeder.base_mva)
        pv, qv = scen.sample(step, n_s, rng, feeder.base_mva)       # fresh draws for verification
        for direction in ("export", "import"):
            for name, make in methods.items():
                feas, counter = make()
                t = time.perf_counter()
                env = envelopes.scalar_envelope(feas, feeder.n_bus, customers, p, q, direction, "uniform", eps,
                                                levels_per_round)
                dt = time.perf_counter() - t
                sign = envelopes.DIRECTIONS[direction]
                inj = sign * env.limit @ customers.incidence(feeder.n_bus)
                p_ok = float(ac_feas(pv + inj, qv, direction).mean())
                rows.append({"step": step, "time": str(scen.time[step]), "direction": direction, "method": name,
                             "limit_kw": float(env.level * 1e3 * feeder.base_mva),
                             "total_kw": float(env.limit.sum() * 1e3 * feeder.base_mva),
                             "prob_ok_oracle": env.prob_ok, "prob_ok_ac_fresh": p_ok,
                             "power_flows": env.n_evals, "ac_calls": counter(env), "runtime_s": dt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="a test day, e.g. 2024-07-03")
    ap.add_argument("--run", type=Path, help="GENCO run dir; omit to run only AC and LinDistFlow")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--calib", type=Path, default=Path("data_out/datakit_v2_calib"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--levels-per-round", type=int, default=4)
    ap.add_argument("--steps-per-hour", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    net = load_enwl1()
    feeder = net.to_feeder()
    n_c = len(net.customer_bus)
    customers = envelopes.Customers(bus=net.customer_bus, export_cap=np.full(n_c, FLEX_EXPORT_KW / 1e3),
                                    import_cap=np.full(n_c, FLEX_IMPORT_KW / 1e3))
    scen = DayScenarios(net.customer_bus, net.n_bus, seed=a.seed)

    ac = envelopes.exact(fdr.sweep_pf, feeder)
    lin = envelopes.exact(fdr.lindistflow, feeder)
    methods = {"ac": lambda: (ac, lambda env: env.n_evals),
               "lindistflow": lambda: (lin, lambda env: 0)}
    meta = {"day": a.day, "scenarios": a.scenarios, "eps": a.eps, "mix": MIX,
            "export_cap_kw": FLEX_EXPORT_KW, "import_cap_kw": FLEX_IMPORT_KW}
    if a.run:
        from .genco import GencoModel, GencoOracle, read_raw

        model = GencoModel(a.config, a.run / "artifacts/model/best_model_state_dict.pt",
                           a.run / "artifacts/stats/normalizer_stats.pt", device=a.device)
        oracle = GencoOracle(model, read_raw(a.calib / "enwl1_fluvius" / "raw", scenarios=[0]))
        q_v, q_l, n_cal = conformal_margins(model, feeder, a.calib, a.alpha)
        meta.update({"run": str(a.run), "q_v": q_v, "q_l": q_l, "calib_states": n_cal, "alpha": a.alpha})
        print(f"conformal margins on {n_cal} calibration-day states: q_v={q_v:.5f} pu, q_l={q_l:.5f}")
        methods["genco"] = lambda: (envelopes.exact(oracle, feeder), lambda env: 0)

        def screened(verify):
            sf = envelopes.ScreenedFeasibility(feeder, oracle, q_v, q_l, fdr.sweep_pf if verify else None)
            return sf, lambda env: sf.n_verified
        methods["genco_conformal"] = lambda: screened(False)
        methods["genco_conformal_ac"] = lambda: screened(True)

    rows = run_day(a.day, feeder, customers, scen, methods, a.scenarios, a.eps, a.levels_per_round, a.seed,
                   a.steps_per_hour)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(a.out.with_suffix(".csv"), index=False)
    ref = df[df.method == "ac"].set_index(["step", "direction"]).total_kw
    df["gap_vs_ac_kw"] = df.total_kw - df.set_index(["step", "direction"]).index.map(ref).to_numpy()
    summ = df.groupby(["direction", "method"]).agg(
        total_kw_mean=("total_kw", "mean"), gap_kw_mean=("gap_vs_ac_kw", "mean"),
        gap_kw_maxabs=("gap_vs_ac_kw", lambda x: x.abs().max()), p_ok_min=("prob_ok_ac_fresh", "min"),
        p_ok_mean=("prob_ok_ac_fresh", "mean"), ac_calls=("ac_calls", "sum"), runtime_s=("runtime_s", "sum"))
    print(summ.to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
