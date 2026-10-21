"""Stage A report for the GENCO-OPF layer (docs/genco_opf_protocol.md, sections 7 and 9, amendment 2 criteria).

    python -m gencodoe.opf_report --test-dir data_out/opf_case33/test --out results/genco_opf_stageA_report.json

Reads the per-day evaluation files written by opf_envelopes (one per test day), and reports per method and direction
(and combined): hosting, gain over uniform, gap to the multi-scenario reference, held-out P(ok) mean and share >= 0.95,
design share, Jain's index, share of zero limits, proposal / check / end-to-end wall-clock. Criteria 1-4 and the
graph-structure claim are evaluated for the GENCO-style model with a day-block bootstrap (2,000 replicates, seed 0)
of the hosting ratios.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REF, LEARNED, BASELINE_LP, BASELINE_MLP = "multi", "gnn", "lp", "mlp"
N_BOOT, BOOT_SEED = 2000, 0


GENCO = "gnn_genco"                                      # amendment 3 arm, evaluated in a separate directory


def load_rows(test_dir: Path, extra_dirs=()):
    rows, days = [], []
    for f in sorted(glob.glob(str(test_dir / "day*.json"))):
        d = json.loads(Path(f).read_text())
        rows += d["rows"]
        days += d["meta"]["days"]
    for x in extra_dirs:
        for f in sorted(glob.glob(str(Path(x) / "day*.json"))):
            d = json.loads(Path(f).read_text())
            if any(dd not in days for dd in d["meta"]["days"]):
                raise ValueError(f"{f}: days not in the main evaluation")
            rows += d["rows"]
    return rows, sorted(days)


def jain(x):
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if (x > 0).any() else float("nan")


def summarise(rows, method, direction):
    r = [x for x in rows if x["method"] == method and (direction == "combined" or x["direction"] == direction)]
    if not r:
        return None
    uni = {(x["day"], x["hour"], x["direction"]): x["total_mw"] for x in rows if x["method"] == "uniform"}
    ref = {(x["day"], x["hour"], x["direction"]): x["total_mw"] for x in rows if x["method"] == REF}
    host = sum(x["total_mw"] for x in r)
    host_u = sum(uni[(x["day"], x["hour"], x["direction"])] for x in r)
    host_ref = sum(ref[(x["day"], x["hour"], x["direction"])] for x in r) if ref else float("nan")
    fresh = np.array([x["fresh_ok"] for x in r])
    j = np.array([jain(x["limit_mw"]) for x in r])
    days = sorted({x["day"] for x in r})
    return {"n_intervals": len(r), "hosting_mwh": host, "hosting_mwh_per_day": host / len(days),
            "gain_over_uniform_pct": 100.0 * (host / host_u - 1) if host_u > 0 else float("nan"),
            "gap_to_reference_pct": 100.0 * (1 - host / host_ref) if host_ref > 0 else float("nan"),
            "design_share_min": float(min(x["design_share"] for x in r)),
            "fresh_ok_mean": float(fresh.mean()), "fresh_ok_share_ge_095": float((fresh >= 0.95).mean()),
            "jain_mean": float(np.nanmean(j)) if np.isfinite(j).any() else float("nan"),
            "zero_limit_customer_share": float(np.mean([np.mean(np.asarray(x["limit_mw"]) <= 0) for x in r])),
            "zero_interval_share": float(np.mean([x["total_mw"] <= 0 for x in r])),
            "proposal_s_mean": float(np.mean([x["proposal_s"] for x in r])),
            "check_s_mean": float(np.mean([x["check_s"] for x in r])),
            "end_to_end_s_mean": float(np.mean([x["proposal_s"] + x["check_s"] for x in r])),
            "exact_evals_mean": float(np.mean([x["exact_evals"] for x in r]))}


def day_totals(rows, method, days):
    """Hosting per day (combined over directions and hours) for one method, aligned with `days`."""
    t = {d: 0.0 for d in days}
    for x in rows:
        if x["method"] == method:
            t[x["day"]] += x["total_mw"]
    return np.array([t[d] for d in days])


def boot_ratio(num, den, rng, n=N_BOOT):
    """Day-block bootstrap of sum(num)/sum(den): point estimate and 2.5/97.5 percentiles."""
    idx = rng.integers(0, len(num), size=(n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return {"point": float(num.sum() / den.sum()), "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def boot_diff_ratio(a, b, den, rng, n=N_BOOT):
    """Day-block bootstrap of (sum(a) - sum(b)) / sum(den)."""
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = (a[idx].sum(1) - b[idx].sum(1)) / den[idx].sum(1)
    return {"point": float((a.sum() - b.sum()) / den.sum()), "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def criteria(rows, days, summary):
    rng = np.random.default_rng(BOOT_SEED)
    g, ref, lp, mlp = (day_totals(rows, m, days) for m in (LEARNED, REF, BASELINE_LP, BASELINE_MLP))
    sg, sref = summary[LEARNED]["combined"], summary[REF]["combined"]
    r_ref, r_lp = boot_ratio(g, ref, rng), boot_ratio(g, lp, rng)
    graph = boot_diff_ratio(g, mlp, ref, rng)
    c = {"1_near_optimal": {"ratio_to_reference": r_ref, "pass": r_ref["point"] >= 0.95},
         "2_beats_lp": {"ratio_to_lp": r_lp, "pass": r_lp["point"] >= 1.05},
         "3_safe": {"fresh_mean": sg["fresh_ok_mean"], "reference_fresh_mean": sref["fresh_ok_mean"],
                    "fresh_share": sg["fresh_ok_share_ge_095"], "reference_fresh_share": sref["fresh_ok_share_ge_095"],
                    "pass": sg["fresh_ok_mean"] >= sref["fresh_ok_mean"] - 0.005
                    and sg["fresh_ok_share_ge_095"] >= sref["fresh_ok_share_ge_095"] - 0.01},
         "4_cheaper": {"end_to_end_s": sg["end_to_end_s_mean"], "reference_end_to_end_s": sref["end_to_end_s_mean"],
                       "pass": sg["end_to_end_s_mean"] < sref["end_to_end_s_mean"]},
         "graph_structure": {"gnn_minus_mlp_over_reference": graph, "pass": graph["point"] >= 0.02 and graph["lo"] > 0}}
    c["genco_opf_positive"] = all(c[k]["pass"] for k in ("1_near_optimal", "2_beats_lp", "3_safe", "4_cheaper"))
    if GENCO in summary:                                                # amendment 3: the actual GENCO as backbone
        gg = day_totals(rows, GENCO, days)
        sg2 = summary[GENCO]["combined"]
        helps = boot_diff_ratio(gg, g, ref, rng)
        r2, l2 = boot_ratio(gg, ref, rng), boot_ratio(gg, lp, rng)
        c["amendment3_genco_helps"] = {
            "genco_minus_gnn_over_reference": helps, "ratio_to_reference": r2, "ratio_to_lp": l2,
            "fresh_mean": sg2["fresh_ok_mean"], "fresh_share": sg2["fresh_ok_share_ge_095"],
            "end_to_end_s": sg2["end_to_end_s_mean"],
            "criterion_1": r2["point"] >= 0.95, "criterion_2": l2["point"] >= 1.05,
            "criterion_3": sg2["fresh_ok_mean"] >= sref["fresh_ok_mean"] - 0.005
            and sg2["fresh_ok_share_ge_095"] >= sref["fresh_ok_share_ge_095"] - 0.01,
            "criterion_4": sg2["end_to_end_s_mean"] < sref["end_to_end_s_mean"]}
        a3 = c["amendment3_genco_helps"]
        a3["pass"] = bool(helps["point"] >= 0.02 and helps["lo"] > 0 and a3["criterion_1"] and a3["criterion_3"])
    c["near_optimal_and_safe_not_faster"] = all(c[k]["pass"] for k in ("1_near_optimal", "2_beats_lp", "3_safe")) and not c["4_cheaper"]["pass"]
    return c


def report(test_dir: Path, extra_dirs=()):
    rows, days = load_rows(test_dir, extra_dirs)
    methods = sorted({x["method"] for x in rows}, key=["uniform", "lp", "ipopt", REF, LEARNED, BASELINE_MLP, GENCO].index)
    summary = {m: {d: summarise(rows, m, d) for d in ("export", "import", "combined")} for m in methods}
    out = {"days": days, "n_days": len(days), "n_rows": len(rows), "methods": methods, "summary": summary,
           "bootstrap": {"replicates": N_BOOT, "seed": BOOT_SEED, "block": "day"}}
    if all(m in methods for m in (LEARNED, REF, BASELINE_LP, BASELINE_MLP)):
        out["criteria"] = criteria(rows, days, summary)
    return out


def table(out):
    lines = [f"{'method':8} {'dir':8} {'MWh/day':>8} {'vs uni%':>8} {'gap ref%':>9} {'fresh':>6} {'>=.95':>6} {'jain':>5} {'zero%':>6} {'e2e s':>7}"]
    for m, per in out["summary"].items():
        for d, s in per.items():
            if s:
                lines.append(f"{m:8} {d:8} {s['hosting_mwh_per_day']:8.2f} {s['gain_over_uniform_pct']:8.1f} {s['gap_to_reference_pct']:9.1f} "
                             f"{s['fresh_ok_mean']:6.3f} {s['fresh_ok_share_ge_095']:6.2f} {s['jain_mean']:5.2f} "
                             f"{100 * s['zero_interval_share']:6.1f} {s['end_to_end_s_mean']:7.1f}")
    if "criteria" in out:
        for k, v in out["criteria"].items():
            lines.append(f"{k}: {v if isinstance(v, bool) else ('PASS' if v['pass'] else 'FAIL')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=Path("data_out/opf_case33/test"))
    ap.add_argument("--extra-dir", type=Path, nargs="*", default=[], help="e.g. data_out/opf_case33/test_genco")
    ap.add_argument("--out", type=Path, default=Path("results/genco_opf_stageA_report.json"))
    a = ap.parse_args()
    out = report(a.test_dir, a.extra_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(table(out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
