"""Final report of the amended pre-registered evaluation (docs/reporting_plan.md).

Reads the B (exact four-wire) and C (GENCO + four-wire screen) DOE runs of each
pre-registered day and writes every reported number to one JSON file plus a
markdown table rendering. Integrity checks fail loudly.

Usage:
  python -m gencodoe.report_final --runs data_out/doe/*_uk_esqcr.json --out results/final_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from . import limits

DT_H = 0.25
N_HOUSEHOLDS = 200
CAPS_KW = {"export": 5.0, "import": 7.4}
METHOD_ALIAS = {"ac": "ac", "b_opendss_exact": "B", "c_genco_4w_opendss": "C"}


def cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound."""
    return 0.0 if k == 0 else float(beta.ppf(1 - conf, k, n - k + 1))


def load_runs(paths: list[Path], expect: dict) -> tuple[pd.DataFrame, list[dict]]:
    frames, inputs = [], []
    for path in paths:
        raw = path.read_bytes()
        doc = json.loads(raw)
        m = doc["meta"]
        vl = m.get("voltage_limits", {})
        for key in ("jurisdiction",):
            if vl.get(key) != expect[key]:
                raise SystemExit(f"{path}: voltage_limits.{key}={vl.get(key)!r}, expected {expect[key]!r}")
        for key in ("vmin_pu", "vmax_pu"):
            if abs(vl.get(key, np.nan) - expect[key]) > 1e-9:
                raise SystemExit(f"{path}: voltage_limits.{key}={vl.get(key)}, expected {expect[key]}")
        if expect.get("seed") is not None:
            for key in ("scenario_seed", "household_seed"):
                if m.get(key, m.get("seed")) != expect["seed"]:
                    raise SystemExit(f"{path}: {key}={m.get(key)}, expected {expect['seed']}")
        if m.get("steps_per_hour", 4) != 4:
            raise SystemExit(f"{path}: steps_per_hour={m.get('steps_per_hour')}, expected 4")
        df = pd.DataFrame(doc["rows"])
        df["day"] = m["day"]
        df["verify_draws_n"] = int(m.get("verify_draws") or m["scenarios"])
        df["method"] = df.method.map(METHOD_ALIAS).fillna(df.method)
        df["file"] = path.name
        frames.append(df)
        inputs.append({"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "meta": m})
    df = pd.concat(frames, ignore_index=True)
    # the balanced reference runs in both the B and the C job; keep one copy per day
    ac = df[df.method == "ac"].drop_duplicates(["day", "direction", "step"])
    df = pd.concat([df[df.method != "ac"], ac], ignore_index=True)
    counts = df.groupby(["day", "direction", "method"]).step.nunique()
    if (counts != 96).any():
        raise SystemExit(f"expected 96 intervals per day/direction/method, got:\n{counts}")
    return df, inputs


def feasibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n4 = out.unbalanced_draws.astype(int)
    k4 = np.round(out.prob_ok_unbalanced * n4).astype(int)
    out["p4"], out["lb4"] = k4 / n4, [cp_lower(k, n) for k, n in zip(k4, n4)]
    nb = out.get("verify_draws_n", pd.Series(1000, index=out.index)).astype(int)
    kb = np.round(out.prob_ok_ac_fresh * nb).astype(int)
    out["pb"], out["lbb"] = kb / nb, [cp_lower(k, n) for k, n in zip(kb, nb)]
    return out


def report(df: pd.DataFrame) -> dict:
    res = {"safety": [], "capacity": [], "agreement": [], "runtime": []}
    for (day, direction), g in df.groupby(["day", "direction"]):
        L = {m: gm.set_index("step").limit_kw.sort_index() for m, gm in g.groupby("method")}
        static = float(L["B"].min()) if "B" in L else np.nan
        for m, gm in g.groupby("method"):
            res["safety"].append({"day": day, "direction": direction, "method": m,
                                  "intervals_lb4_ge_0.95": int((gm.lb4 >= 0.95).sum()),
                                  "intervals_p4_ge_0.95": int((gm.p4 >= 0.95).sum()),
                                  "min_p4": float(gm.p4.min()), "mean_p4": float(gm.p4.mean()),
                                  "intervals_lbb_ge_0.95": int((gm.lbb >= 0.95).sum()), "intervals": int(len(gm))})
            cap = float((gm.limit_kw * DT_H).sum())
            res["capacity"].append({"day": day, "direction": direction, "method": m,
                                    "kwh_per_household": cap, "kwh_feeder": cap * N_HOUSEHOLDS,
                                    "mean_limit_kw": float(gm.limit_kw.mean())})
            if m in ("B", "C"):
                rt = gm.runtime_s
                res["runtime"].append({"day": day, "direction": direction, "method": m,
                                       "search_s_total": float(rt.sum()), "search_s_median": float(rt.median()),
                                       "search_s_p95": float(rt.quantile(0.95)), "search_s_max": float(rt.max()),
                                       "exact_calls_total": int(gm.ac_calls.sum()),
                                       "exact_calls_per_interval_median": float(gm.ac_calls.median())})
        static_cap = static * DT_H * 96
        res["capacity"].append({"day": day, "direction": direction, "method": "static_from_B",
                                "static_limit_kw": static, "kwh_per_household": static_cap,
                                "kwh_feeder": static_cap * N_HOUSEHOLDS, "mean_limit_kw": static})
        if "B" in L and "C" in L:
            b, c = L["B"].align(L["C"], join="inner")
            tol = 1e-3 * CAPS_KW[direction]
            d = (c - b)
            res["agreement"].append({"day": day, "direction": direction, "intervals": int(len(d)),
                                     "mean_abs_diff_kw": float(d.abs().mean()), "max_abs_diff_kw": float(d.abs().max()),
                                     "share_within_tol": float((d.abs() <= tol).mean()), "tol_kw": tol,
                                     "intervals_c_exceeds_b": int((d > tol).sum()),
                                     "max_c_excess_kw": float(max(d.max(), 0.0)),
                                     "intervals_both_zero": int(((b <= 1e-9) & (c <= 1e-9)).sum())})
    # uplift only where the static baseline is positive
    cap = pd.DataFrame(res["capacity"])
    for (day, direction), g in cap.groupby(["day", "direction"]):
        s = g[g.method == "static_from_B"].kwh_per_household.iloc[0]
        for m in ("B", "C"):
            row = g[g.method == m]
            if len(row) and s > 0:
                res.setdefault("uplift_vs_static", []).append(
                    {"day": day, "direction": direction, "method": m,
                     "relative_uplift": float((row.kwh_per_household.iloc[0] - s) / s)})
    return res


def markdown(res: dict) -> str:
    parts = []
    for key in ("safety", "capacity", "agreement", "runtime", "uplift_vs_static"):
        if res.get(key):
            parts.append(f"### {key}\n\n" + pd.DataFrame(res[key]).to_markdown(index=False, floatfmt=".4g"))
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=101, help="expected scenario and household seed (None to skip)")
    a = ap.parse_args()
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": a.seed}
    df, inputs = load_runs(a.runs, expect)
    res = report(feasibility(df))
    doc = {"plan": "docs/reporting_plan.md", "voltage_limits": limits.record(), "inputs": inputs, **res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    a.out.with_suffix(".md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
