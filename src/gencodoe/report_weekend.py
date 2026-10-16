"""Weekend analyses W1-W3 (docs/weekend_protocols.md; post-hoc, exploratory).

Applies the gencodoe.report_annual definitions per (track, household seed); W3 also pairs the
spatial envelopes with uniform B of the same day (seed 101) interval by interval.

    python -m gencodoe.report_weekend --b-dir data_out/doe/annual --weekend data_out/doe/weekend \
        --out results/weekend_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import limits
from .report_annual import annual
from .envelopes import acceptance_threshold
from .report_final import feasibility, load_runs

THRESHOLD = acceptance_threshold(200, 0.05, 0.05)


def _load(paths, household_seed):
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": None, "steps_per_hour": 1}
    df, inputs = load_runs(sorted(paths), expect)
    for i in inputs:
        m = json.loads(Path(i["file"]).read_text())["meta"]
        if m.get("scenario_seed") != 101 or m.get("household_seed") != household_seed:
            raise SystemExit(f"{i['file']}: seeds {m.get('scenario_seed')}/{m.get('household_seed')}, "
                             f"expected 101/{household_seed}")
    return feasibility(df), inputs


def summary(df: pd.DataFrame, method: str) -> dict:
    a = annual(df)
    out = {"days": int(df[df.method == method].day.nunique())}
    for d, res in a["by_direction"].items():
        r = res[method]
        g = df[(df.method == method) & (df.direction == d)]
        nz = g[g.limit_kw > 1e-9]
        out[d] = {"kwh_per_household_year": r["kwh_per_household_year"], "ci95": r["ci95"],
                  "share_intervals_nonzero": r["share_intervals_nonzero_envelope"],
                  # zero sentinel: level 0 and design share at 0 below the 0.98 acceptance threshold
                  "share_intervals_baseline_infeasible": float(((g.limit_kw <= 1e-9) & (g.prob_ok_oracle < THRESHOLD)).mean()),
                  "nonzero_mean_p4": float(nz.p4.mean()) if len(nz) else np.nan,
                  "nonzero_share_p4_ge_0.95": float((nz.p4 >= 0.95).mean()) if len(nz) else np.nan,
                  "exact_calls": int(g.ac_calls.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", type=Path, default=Path("data_out/doe/annual"))
    ap.add_argument("--weekend", type=Path, default=Path("data_out/doe/weekend"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    doc = {"protocol": "docs/weekend_protocols.md", "status": "post-hoc, exploratory", "voltage_limits": limits.record(),
           "W1_placement": {}, "W2_three_phase": {}, "inputs": []}
    b101 = list(a.b_dir.glob("*_annual_4w_b_opendss_exact_sph1_seed101_uk_esqcr.json"))
    single = {101: b101}
    for hh in (202, 303, 404, 505):
        single[hh] = list((a.weekend / "w1_placement").glob(f"*_b_opendss_exact_sph1_seed101_hh{hh}_uk_esqcr.json"))
    for hh, paths in single.items():
        if paths:
            df, inp = _load(paths, hh)
            doc["W1_placement"][str(hh)] = summary(df, "B")
            doc["inputs"] += inp
    spread = {}
    for d in ("export", "import"):
        v = [s[d]["kwh_per_household_year"] for s in doc["W1_placement"].values()]
        spread[d] = {"min": min(v), "max": max(v), "mean": float(np.mean(v)), "seeds": len(v)}
    doc["W1_placement"]["across_seeds"] = spread
    for hh in (101, 202):
        paths = list((a.weekend / "w2_three_phase").glob(f"*_hh{hh}_uk_esqcr.json"))
        if paths:
            df, inp = _load(paths, hh)
            doc["W2_three_phase"][str(hh)] = summary(df, "B")
            doc["inputs"] += inp

    paths = list((a.weekend / "w3_spatial").glob("*_s_spatial_exact_*_hh101_uk_esqcr.json"))
    if paths:
        days = {p.name[:10] for p in paths}
        df, inp = _load(paths + [p for p in b101 if p.name[:10] in days], 101)
        doc["inputs"] += inp
        w3 = {"days": len(days), "complete": len(days) == 36}
        for d in ("export", "import"):
            g = df[df.direction == d]
            b = g[g.method == "B"].set_index(["day", "step"])
            s = g[g.method == "s_spatial_exact"].set_index(["day", "step"])
            b, s = b.align(s, join="inner", axis=0)
            better = worse = n_cust = 0
            for (_, rb), (_, rs) in zip(b.iterrows(), s.iterrows()):
                if rs.limit_kw > 1e-9 or rb.limit_kw > 1e-9:
                    c = np.asarray(rs.customer_limit_kw)
                    better += int((c > rb.limit_kw + 1e-3).sum())
                    worse += int((c < rb.limit_kw - 1e-3).sum())
                    n_cust += len(c)
            nz = s[s.limit_kw > 1e-9]
            w3[d] = {"intervals": int(len(s)), "kwh_per_household_day_B": float(b.limit_kw.sum() / len(days)),
                     "kwh_per_household_day_spatial": float(s.limit_kw.sum() / len(days)),
                     "ratio_spatial_over_B": float(s.limit_kw.sum() / max(b.limit_kw.sum(), 1e-12)),
                     "intervals_spatial_nonzero_B_zero": int(((s.limit_kw > 1e-9) & (b.limit_kw <= 1e-9)).sum()),
                     "share_customer_intervals_better_than_uniform": better / max(n_cust, 1),
                     "share_customer_intervals_worse_than_uniform": worse / max(n_cust, 1),
                     "lambda_eq_1_share": float((nz["lambda"] >= 1 - 1e-9).mean()) if len(nz) else np.nan,
                     "nonzero_mean_p4": float(nz.p4.mean()) if len(nz) else np.nan,
                     "nonzero_share_p4_ge_0.95": float((nz.p4 >= 0.95).mean()) if len(nz) else np.nan,
                     "exact_calls_B": int(b.ac_calls.sum()), "exact_calls_spatial": int(s.ac_calls.sum())}
        doc["W3_spatial"] = w3
    doc["inputs"] = [{"file": i["file"], "sha256": i["sha256"]} for i in doc["inputs"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({k: v for k, v in doc.items() if k != "inputs"}, indent=1, default=float))


if __name__ == "__main__":
    main()
