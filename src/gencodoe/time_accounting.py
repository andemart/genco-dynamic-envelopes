"""Wall-clock accounting of DOE workflows from measured counts and per-row costs.

A DOE run (doe_day) records, per interval, direction and method, the rows the
oracle evaluated (power_flows) and the exact AC calls it made (ac_calls). A
benchmark (benchmark.py) measures warm ms/row for each solver. Time of a
workflow = surrogate rows x t_surrogate + AC calls x t_ac, for any choice of
AC engine; this isolates the solver costs from the Python search loop, which is
identical across methods. The fastest per-interval and per-day times are
reported for each (surrogate, AC engine) pair.

Usage:
  python -m gencodoe.time_accounting --doe data_out/doe/2024-07-08_hourly_all.json \
      --bench data_out/eval/benchmark_m4_cpu_v2.json data_out/eval/benchmark_a100_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def costs(bench_files: list[Path]) -> dict[str, float]:
    """Best measured warm ms/row per solver label across benchmark files."""
    best: dict[str, float] = {}
    for f in bench_files:
        b = json.loads(Path(f).read_text())
        tag = Path(f).stem.replace("benchmark_", "")
        for r in b["rows"]:
            key = f"{r['method']}@{r['device']}" if r["method"].startswith("genco") else r["method"]
            if r["method"].startswith("genco"):
                key += f" [{tag}]"
            best[key] = min(best.get(key, float("inf")), r["warm_ms_per_row"])
        for k, v in b.get("generic", {}).items():
            best[k] = min(best.get(k, float("inf")), v["ms_per_row"])
    return best


def account(doe: Path, c: dict[str, float], ac_engines: list[str], surrogates: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(json.loads(Path(doe).read_text())["rows"])
    out = []
    for engine in ac_engines:
        if engine not in c:
            continue
        t_ac = c[engine]
        ac = df[df.method == "ac"]
        rows = [("AC only", engine, ac.power_flows, ac.ac_calls * 0 + ac.power_flows, None)]
        for method, sur in surrogates.items():
            m = df[df.method == method]
            if m.empty:
                continue
            for key in [k for k in c if k == sur or k.startswith(sur + " [") or k.startswith(sur + "@")]:
                rows.append((f"{method} ({key})", engine, m.power_flows, m.ac_calls, key))
        for name, eng, pf, acc, key in rows:
            ms = (pf * c[key] if key else 0) + acc * t_ac if key else pf * t_ac
            per = pd.Series(ms.to_numpy() / 1e3)
            out.append({"workflow": name, "ac_engine": eng, "day_s": per.sum(), "worst_interval_dir_s": per.max(),
                        "ac_calls": int(acc.sum()), "surrogate_rows": int(pf.sum()) if key else 0})
    res = pd.DataFrame(out)
    base = res[res.workflow == "AC only"].set_index("ac_engine").day_s
    res["speedup_vs_ac_only"] = res.ac_engine.map(base) / res.day_s
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe", type=Path, required=True)
    ap.add_argument("--bench", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    c = costs(a.bench)
    print("measured warm ms/row:", {k: round(v, 4) for k, v in sorted(c.items())})
    res = account(a.doe, c, ["ac_sweep", "pandapower_nr", "opendss_unbal"],
                  {"genco_conformal_ac": "genco", "fixed_point_conformal_ac": "fixed_point_1"})
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
