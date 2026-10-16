"""Annual report of the pre-registered 36-test-day extension
(docs/preregistration_annual_extension.md), written before any annual result was opened.

Per day it applies the gencodoe.report_final definitions (Δt = 1 h, 24 intervals);
across days it reports, per direction and method:
  * annual estimate of capacity = mean daily kW·h per household x 366, with a
    season-stratified bootstrap 95% CI over days (days resampled within season);
  * seasonal means;
  * the annual static limit L_static = min over all days and intervals of L_B,
    its annual capacity, and relative uplift only where it is > 0;
  * safety: share of intervals with one-sided 95% lower bound >= 0.95 (four-wire);
  * B-C agreement over all intervals, and total exact calls / search time.
Days missing a B or C run are listed and excluded from paired quantities.

Usage:
  python -m gencodoe.report_annual --runs data_out/doe/annual/*.json --out results/annual_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import limits
from .report_final import feasibility, load_runs

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
DAYS_IN_YEAR = 366
DT_H = 1.0


def season(day: str) -> str:
    return SEASON[int(day[5:7])]


def stratified_bootstrap(daily: pd.Series, seasons: pd.Series, n_boot: int = 2000, seed: int = 0):
    """95% CI of mean(daily) x 366, resampling days within each season."""
    rng = np.random.default_rng(seed)
    groups = [daily[seasons == s].to_numpy() for s in sorted(seasons.unique())]
    weights = np.array([len(g) for g in groups], float) / len(daily)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = sum(w * rng.choice(g, len(g)).mean() for w, g in zip(weights, groups) if len(g))
    return float(np.percentile(stats, 2.5) * DAYS_IN_YEAR), float(np.percentile(stats, 97.5) * DAYS_IN_YEAR)


def annual(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["season"] = df.day.map(season)
    out = {"days_per_method": df.groupby("method").day.nunique().to_dict(), "by_direction": {}}
    paired_days = sorted(set(df[df.method == "B"].day) & set(df[df.method == "C"].day))
    out["paired_days"] = len(paired_days)
    out["unpaired_days"] = sorted(set(df[df.method.isin(["B", "C"])].day) - set(paired_days))
    for direction, g in df.groupby("direction"):
        res = {}
        daily = g.groupby(["method", "day"]).limit_kw.sum().mul(DT_H).rename("kwh").reset_index()
        daily["season"] = daily.day.map(season)
        static = float(g[g.method == "B"].limit_kw.min()) if (g.method == "B").any() else np.nan
        res["annual_static_limit_kw"] = static
        res["static_kwh_per_household_year"] = static * 24 * DT_H * DAYS_IN_YEAR
        for m, gm in daily.groupby("method"):
            est = float(gm.kwh.mean() * DAYS_IN_YEAR)
            lo, hi = stratified_bootstrap(gm.kwh, gm.season)
            row = {"days": int(len(gm)), "kwh_per_household_year": est, "ci95": [lo, hi],
                   "kwh_feeder_year": est * 200,
                   "seasonal_mean_kwh_per_day": gm.groupby("season").kwh.mean().round(4).to_dict()}
            if static > 0:
                row["relative_uplift_vs_annual_static"] = float(est / res["static_kwh_per_household_year"] - 1)
            sub = g[g.method == m]
            row["share_intervals_lb4_ge_0.95"] = float((sub.lb4 >= 0.95).mean())
            row["share_intervals_p4_ge_0.95"] = float((sub.p4 >= 0.95).mean())
            row["share_intervals_nonzero_envelope"] = float((sub.limit_kw > 1e-9).mean())
            if m in ("B", "C", "C+"):
                row["exact_calls"] = int(sub.ac_calls.sum())
                row["search_s"] = float(sub.runtime_s.sum())
            res[m] = row
        pb = g[(g.method == "B") & g.day.isin(paired_days)].set_index(["day", "step"]).limit_kw
        pc = g[(g.method == "C") & g.day.isin(paired_days)].set_index(["day", "step"]).limit_kw
        if len(pb) and len(pc):
            b, c = pb.align(pc, join="inner")
            tol = 1e-3 * {"export": 5.0, "import": 7.4}[direction]
            d = c - b
            res["agreement_paired"] = {"intervals": int(len(d)), "share_within_tol": float((d.abs() <= tol).mean()),
                                       "mean_abs_diff_kw": float(d.abs().mean()), "max_abs_diff_kw": float(d.abs().max()),
                                       "intervals_c_exceeds_b": int((d > tol).sum()),
                                       "intervals_both_zero": int(((b <= 1e-9) & (c <= 1e-9)).sum()),
                                       "intervals_both_nonzero": int(((b > 1e-9) & (c > 1e-9)).sum())}
            cb = g[(g.method == "B") & g.day.isin(paired_days)]
            cc = g[(g.method == "C") & g.day.isin(paired_days)]
            res["paired_cost"] = {"exact_calls_B": int(cb.ac_calls.sum()), "exact_calls_C": int(cc.ac_calls.sum()),
                                  "search_s_B": float(cb.runtime_s.sum()), "search_s_C": float(cc.runtime_s.sum())}
        cp_days = sorted(set(g[g.method == "B"].day) & set(g[g.method == "C+"].day))   # post-hoc C+, new keys only
        if cp_days:
            pb2 = g[(g.method == "B") & g.day.isin(cp_days)].set_index(["day", "step"]).limit_kw
            pp = g[(g.method == "C+") & g.day.isin(cp_days)].set_index(["day", "step"]).limit_kw
            b, c = pb2.align(pp, join="inner")
            tol = 1e-3 * {"export": 5.0, "import": 7.4}[direction]
            d = c - b
            res["agreement_paired_cplus"] = {"intervals": int(len(d)), "share_within_tol": float((d.abs() <= tol).mean()),
                                             "mean_abs_diff_kw": float(d.abs().mean()), "max_abs_diff_kw": float(d.abs().max()),
                                             "intervals_cplus_exceeds_b": int((d > tol).sum()),
                                             "intervals_both_zero": int(((b <= 1e-9) & (c <= 1e-9)).sum()),
                                             "intervals_both_nonzero": int(((b > 1e-9) & (c > 1e-9)).sum())}
            cb = g[(g.method == "B") & g.day.isin(cp_days)]
            cc = g[(g.method == "C+") & g.day.isin(cp_days)]
            res["paired_cost_cplus"] = {"exact_calls_B": int(cb.ac_calls.sum()), "exact_calls_Cplus": int(cc.ac_calls.sum()),
                                        "search_s_B": float(cb.runtime_s.sum()), "search_s_Cplus": float(cc.runtime_s.sum())}
            for col in ("gate_calls", "repair_calls", "verification_calls", "diagnostic_calls"):
                if col in cc:
                    res["paired_cost_cplus"][col] = int(cc[col].fillna(0).sum())
            res["paired_cost_cplus"]["repaired_intervals"] = int(cc["repaired"].fillna(False).astype(bool).sum()) if "repaired" in cc else 0
        out["by_direction"][direction] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    a = ap.parse_args()
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": 101, "manifest": a.manifest, "steps_per_hour": 1}
    df, inputs = load_runs(a.runs, expect)
    res = annual(feasibility(df))
    doc = {"plan": "docs/preregistration_annual_extension.md", "voltage_limits": limits.record(),
           "n_inputs": len(inputs), "inputs": [{"file": i["file"], "sha256": i["sha256"]} for i in inputs], **res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({k: v for k, v in doc.items() if k != "inputs"}, indent=1, default=float))


if __name__ == "__main__":
    main()
