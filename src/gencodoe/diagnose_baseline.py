"""Baseline (no envelope) four-wire diagnosis of one day: why is no envelope certifiable?

For every interval, draws fresh baseline states of the case-study households
(same technology groups as the DOE runs, DayScenarios seed 0; draw seed given)
and records, per draw, on the ENWL1 four-wire benchmark:
  * violation causes: undervoltage, overvoltage, line overload, transformer
    overload, non-convergence, and their overlaps;
  * offending phase and line segment, number of overloaded segments;
  * per-phase transformer LV current and power, neutral current;
  * the balanced-model verdict for the same draw.
Optionally adds per-interval Clopper-Pearson intervals of the four-wire P(ok)
stored in a DOE run JSON.

Usage:
  python -m gencodoe.diagnose_baseline --day 2024-12-15 --draws 50 --seed 202 \
      --doe data_out/doe/hydra/2024-12-15_certified_4w_b_opendss_exact_sph4_seed101.json \
      --out results/winter_2024-12-15_baseline_diagnosis.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import feeder as fdr
from .doe_day import DayScenarios
from .enwl import load_enwl1
from .limits import VMIN_PU, record as limits_record
from .unbalanced import UnbalancedENWL1, diagnose


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta
    a = 1 - conf
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def run(day: str, draws: int, seed: int) -> pd.DataFrame:
    net = load_enwl1()
    fd = net.to_feeder()
    scen = DayScenarios(net.customer_bus, net.n_bus, seed=0)
    u = UnbalancedENWL1()
    rng = np.random.default_rng(seed)
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
    rows = []
    for k in range(96):
        p, q = scen.sample(t0 + k, draws, rng, fd.base_mva)
        bal = fdr.sweep_pf(fd, p, q)
        for i in range(draws):
            d = diagnose(u, -p[i, net.customer_bus] * 1e3, -q[i, net.customer_bus] * 1e3)
            ia, ib, ic = d["tr_lv_phase_current_a"]
            rows.append({
                "interval": k, "draw": i, "converged": d["converged"],
                "undervoltage": d["undervoltage"] and d["converged"], "overvoltage": d["overvoltage"] and d["converged"],
                "line_overload": d["line_overload"] and d["converged"],
                "transformer_overload": d["transformer_overload"] and d["converged"],
                "v_pn_min": d["v_pn_min"], "v_pn_max": d["v_pn_max"], "line_loading_max": d["line_loading_max"],
                "transformer_loading": d["transformer_loading"], "worst_line": d["worst_line"],
                "worst_line_phase": d["worst_line_phase"], "n_overloaded_lines": d["n_overloaded_lines"],
                "uv_customers_ph1": d["undervoltage_customers_by_phase"][1],
                "uv_customers_ph2": d["undervoltage_customers_by_phase"][2],
                "uv_customers_ph3": d["undervoltage_customers_by_phase"][3],
                "neutral_over_rating_max": d["neutral_over_rating_max"],
                "tr_ia": ia, "tr_ib": ib, "tr_ic": ic, "tr_in": d["tr_lv_neutral_current_a"],
                "tr_kw_a": d["tr_lv_phase_kw"][0], "tr_kw_b": d["tr_lv_phase_kw"][1], "tr_kw_c": d["tr_lv_phase_kw"][2],
                "feeder_kw": float(-p[i].sum() * 1e3 * fd.base_mva),
                "balanced_vmin": float(bal.vm[i, 1:].min()), "balanced_loading_max": float(bal.loading[i, 1:].max()),
            })
    df = pd.DataFrame(rows)
    df["balanced_violation"] = (df.balanced_vmin < VMIN_PU) | (df.balanced_loading_max > 1.0)
    df["any_four_wire"] = ~df.converged | df.undervoltage | df.overvoltage | df.line_overload | df.transformer_overload
    return df


def summarise(df: pd.DataFrame, doe: dict | None) -> dict:
    causes = ["undervoltage", "overvoltage", "line_overload", "transformer_overload"]
    combo = df[causes].astype(int).astype(str).agg("".join, axis=1)
    overlaps = {"+".join(c for c, f in zip(causes, key) if f == "1") or "none": int(n)
                for key, n in combo[df.converged].value_counts().items()}
    imb = df[["tr_ia", "tr_ib", "tr_ic"]].to_numpy()
    out = {
        "draws": int(len(df)), "intervals": int(df.interval.nunique()),
        "share_any_four_wire_violation": float(df.any_four_wire.mean()),
        "share_balanced_violation": float(df.balanced_violation.mean()),
        "share_non_converged": float((~df.converged).mean()),
        "share_by_cause": {c: float(df[c].mean()) for c in causes},
        "cause_overlaps_converged_draws": overlaps,
        "worst_line_phase_counts": {int(k): int(v) for k, v in df[df.converged].worst_line_phase.value_counts().items()},
        "top_worst_lines": {k: int(v) for k, v in collections.Counter(df[df.line_overload].worst_line).most_common(8)},
        "undervoltage_customers_by_phase_mean": {f"phase_{i}": float(df[f"uv_customers_ph{i}"].mean()) for i in (1, 2, 3)},
        "customers_per_phase": None,
        "distributions": {c: {q: float(df[c].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
                          for c in ("v_pn_min", "line_loading_max", "transformer_loading", "n_overloaded_lines",
                                    "neutral_over_rating_max", "tr_in", "feeder_kw", "balanced_vmin")},
        "transformer_phase_current_imbalance_max_over_mean": {q: float(np.quantile(imb.max(1) / imb.mean(1), q))
                                                              for q in (0.5, 0.95)},
        "by_interval": df.groupby("interval").agg(any_four_wire=("any_four_wire", "mean"),
                                                 balanced=("balanced_violation", "mean"),
                                                 undervoltage=("undervoltage", "mean"),
                                                 line_overload=("line_overload", "mean"),
                                                 feeder_kw=("feeder_kw", "median")).round(4).reset_index().to_dict("records"),
    }
    if doe is not None:
        rows = pd.DataFrame(doe["rows"])
        ci = []
        for (step, direction, method), g in rows.groupby(["step", "direction", "method"]):
            r = g.iloc[0]
            n = int(r.get("unbalanced_draws", 0) or 0)
            if n:
                k = int(round(r["prob_ok_unbalanced"] * n))
                lo, hi = clopper_pearson(k, n)
                ci.append({"step": int(step), "direction": direction, "method": method, "p_ok": k / n,
                           "ci95_lo": lo, "ci95_hi": hi, "n": n, "limit_kw": float(r["limit_kw"])})
        ci = pd.DataFrame(ci)
        out["doe_four_wire_ci"] = {
            "file": doe.get("_file"),
            "intervals_with_ci_hi_below_0.95": ci[ci.ci95_hi < 0.95].groupby(["direction", "method"]).size().to_dict()
            if len(ci) else {},
            "per_interval": ci.round(4).to_dict("records")}
        out["doe_four_wire_ci"]["intervals_with_ci_hi_below_0.95"] = {
            f"{d}|{m}": int(v) for (d, m), v in out["doe_four_wire_ci"]["intervals_with_ci_hi_below_0.95"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--doe", type=Path, help="DOE run JSON for per-interval confidence intervals")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = run(a.day, a.draws, a.seed)
    doe = None
    if a.doe:
        doe = json.loads(a.doe.read_text())
        doe["_file"] = str(a.doe)
    s = summarise(df, doe)
    s["customers_per_phase"] = {int(k): int(v) for k, v in
                                collections.Counter(int(b.split(".")[1]) for b in UnbalancedENWL1().load_bus).items()}
    s.update({"day": a.day, "draw_seed": a.seed, "households_seed": 0, "voltage_limits": limits_record()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(s, indent=1, default=float))
    df.to_parquet(a.out.with_suffix(".parquet"))
    brief = {k: s[k] for k in ("share_any_four_wire_violation", "share_balanced_violation", "share_non_converged",
                               "share_by_cause", "cause_overlaps_converged_draws", "worst_line_phase_counts",
                               "top_worst_lines", "undervoltage_customers_by_phase_mean", "customers_per_phase",
                               "transformer_phase_current_imbalance_max_over_mean")}
    print(json.dumps(brief, indent=1))
    print("distributions:", json.dumps(s["distributions"], indent=1))
    if "doe_four_wire_ci" in s:
        print("intervals with 95% CI upper bound below 0.95:", s["doe_four_wire_ci"]["intervals_with_ci_hi_below_0.95"])


if __name__ == "__main__":
    main()
