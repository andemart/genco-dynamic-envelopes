"""Envelope prediction v3: A100 nondeterminism and CPU-vs-A100 diagnostic on the 12 fixed smoke items.

Features only: no B label is opened and no prediction quality is evaluated. Uses the production inference
path (envpred3.load_genco / ga_block) and wraps the oracle to record every GENCO evaluation, so that
per-scenario balanced feasibility decisions and per-constraint classifications can be compared.

    python -m gencodoe.envpred3_diag --out results/envpred3_determinism.json      (on an A100 node)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from . import envpred3 as e3
from .envpred import replay

REPS = 5
UNEVEN_BATCH = 333                 # deliberately different, uneven chunking (production: 1024)


class Recorder:
    """Wraps oracle / embed; keeps every PFResult with its row count, in call order."""

    def __init__(self, oracle, embed):
        self.oracle, self.embed_fn, self.log = oracle, embed, []

    def o(self, feeder, p, q):
        r = self.oracle(feeder, p, q)
        self.log.append(("oracle", r))
        return r

    def e(self, feeder, p, q):
        r, pooled = self.embed_fn(feeder, p, q)
        self.log.append(("embed", r))
        return r, pooled


def _flags(fd, r) -> np.ndarray:
    """(n, 4) per-scenario constraint classification: undervoltage, overvoltage, cable, transformer."""
    s = e3._summary(r)
    return np.column_stack([s[:, 0] < fd.vmin, s[:, 1] > fd.vmax, s[:, 2] > 1.0, s[:, 3] > 1.0])


def run_item(day, hh, hour, d, oracle, embed, fd, cust):
    from . import envelopes
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
    step = t0 + 4 * hour
    for st, p, q, _, _ in replay(day, hh):
        if st == step:
            break
    rec = Recorder(oracle, embed)
    pr0, emb0 = rec.e(fd, p, q)
    s0 = e3._summary(pr0)
    share0 = float(envelopes.is_feasible(fd, pr0, d).mean())
    G = np.concatenate([e3.stats(s0), emb0.mean(0)])
    GA = e3.ga_block(rec.o, rec.e, fd, cust, p, q, d, s0, emb0, share0, {})
    feas = [envelopes.is_feasible(fd, r, d) for _, r in rec.log]           # every GENCO decision, call order
    flags = [_flags(fd, r) for _, r in rec.log]
    return {"features": np.concatenate([G, GA]), "feasible": feas, "flags": flags,
            "calls": [len(f) for f in feas]}


def categories(names):
    cat = []
    for n in names:
        if n == "GA:L_G":
            c = "1_L_G"
        elif n in ("GA:share_LG", "GA:LG_is_zero"):
            c = "2_share_LG_and_zero_indicator"
        elif n.startswith("GA:share_c"):
            c = "3_fixed_grid_shares"
        elif n.startswith("G:emb0_"):
            c = "7_zero_state_embeddings"
        elif n.startswith("GA:embLG_") or n.startswith("GA:dembLG_"):
            c = "8_LG_embeddings_and_deltas"
        elif ":dLG_" in n or ":hLG_" in n:
            c = "6_headroom_and_change"
        elif n.endswith(("_vmin", "_vmax")):
            c = "4_voltage_summaries"
        else:
            c = "5_loading_summaries"
        cat.append(c)
    return cat


def block_of(n):
    if n.startswith("G:"):
        return "G zero-state" if not n.startswith("G:emb0_") else "embedding (zero)"
    for key, b in ((":L_G", "L_G"), ("share_c", "feasible share (grid)"), ("share_LG", "feasible share"),
                   ("LG_is_zero", "zero indicator"), (":dLG_", "response change"), (":hLG_", "headroom"),
                   ("embLG_", "embedding (L_G)"), (":LG_", "proposal response"), (":c0", "response curve")):
        if key in n:
            return b
    return "other"


def compare(a, b):
    """Decision-level comparison of two runs of one item."""
    same_calls = a["calls"] == b["calls"]
    out = {"same_call_sequence": same_calls}
    if same_calls:
        out["feasibility_decisions_changed"] = int(sum((x != y).sum() for x, y in zip(a["feasible"], b["feasible"])))
        out["constraint_flags_changed"] = int(sum((x != y).sum() for x, y in zip(a["flags"], b["flags"])))
    return out


def versions():
    import torch
    v = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ("torch_geometric", "torch_scatter"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:
            v[mod] = f"not importable: {e!r}"
    try:
        v["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        v["driver"] = None
    return v


def main():
    import torch

    from . import envelopes
    from .doe_day import FLEX_EXPORT_KW, FLEX_IMPORT_KW
    from .enwl import load_enwl1
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/envpred3_determinism.json"))
    a = ap.parse_args()
    run = e3.RUN_LOCAL if e3.RUN_LOCAL.exists() else e3.RUN_HYDRA
    net = load_enwl1()
    fd = net.to_feeder()
    n_c = len(net.customer_bus)
    cust = envelopes.Customers(net.customer_bus, np.full(n_c, FLEX_EXPORT_KW / 1e3), np.full(n_c, FLEX_IMPORT_KW / 1e3))
    gpu_o, gpu_e = e3.load_genco("cuda", run)
    cpu_o, cpu_e = e3.load_genco("cpu", run)
    names = [n for n in e3.block_names()["G"]] + e3.block_names()["GA"]
    cats = categories(names)
    reps, uneven, cpu, det_attempt = [], [], [], None
    for it in e3.SMOKE:
        torch.manual_seed(0)
        reps.append([run_item(*it, gpu_o, gpu_e, fd, cust) for _ in range(REPS)])
        gpu_o.batch_size = UNEVEN_BATCH
        uneven.append(run_item(*it, gpu_o, gpu_e, fd, cust))
        gpu_o.batch_size = 1024
        cpu.append(run_item(*it, cpu_o, cpu_e, fd, cust))
        print(it, "done", flush=True)
    try:                                                     # deterministic algorithms, once, as a diagnostic
        torch.use_deterministic_algorithms(True)
        run_item(*e3.SMOKE[0], gpu_o, gpu_e, fd, cust)
        det_attempt = {"ok": True}
    except Exception as e:
        det_attempt = {"ok": False, "error": repr(e)[:2000]}
    finally:
        torch.use_deterministic_algorithms(False)

    F = np.array([[r["features"] for r in rr] for rr in reps])           # (12, 5, 808)
    diff = np.abs(F - F[:, :1])                                            # vs repetition 1
    scale = np.nanstd(F[:, 0], axis=0)
    per_feature = []
    for j, n in enumerate(names):
        mx = float(np.nanmax(diff[:, :, j]))
        per_feature.append({"index": j, "name": n, "block": block_of(n), "category": cats[j], "max_abs_diff": mx,
                            "mean_abs_diff": float(np.nanmean(diff[:, 1:, j])),
                            "smoke_std": float(scale[j]), "mean_abs_value": float(np.nanmean(np.abs(F[:, 0, j]))),
                            "max_diff_over_std": float(mx / scale[j]) if scale[j] > 0 else None})
    by_cat = {}
    for c in sorted(set(cats)):
        js = [j for j in range(len(names)) if cats[j] == c]
        worst = max(js, key=lambda j: per_feature[j]["max_abs_diff"])
        by_cat[c] = {"max_abs_diff": per_feature[worst]["max_abs_diff"], "worst_feature": names[worst],
                     "worst_max_diff_over_std": per_feature[worst]["max_diff_over_std"], "n_features": len(js)}
    iL = names.index("GA:L_G")
    iz = names.index("GA:LG_is_zero")
    ishare = [j for j, n in enumerate(names) if "share" in n]
    decisions = []
    for k, it in enumerate(e3.SMOKE):
        rr = reps[k]
        cmp_reps = [compare(rr[0], r) for r in rr[1:]]
        decisions.append({"item": list(it), "L_G_kw_runs": [float(r["features"][iL]) for r in rr],
                          "L_G_identical": len({float(r["features"][iL]) for r in rr}) == 1,
                          "zero_indicator_identical": len({float(r["features"][iz]) for r in rr}) == 1,
                          "max_share_change": float(max(np.abs(r["features"][ishare] - rr[0]["features"][ishare]).max()
                                                        for r in rr)),
                          "repeat_vs_first": cmp_reps,
                          "uneven_chunking_vs_first": {**compare(rr[0], uneven[k]),
                                                       "L_G_kw": float(uneven[k]["features"][iL]),
                                                       "max_share_change": float(np.abs(uneven[k]["features"][ishare]
                                                                                        - rr[0]["features"][ishare]).max()),
                                                       "max_abs_feature_diff": float(np.nanmax(np.abs(
                                                           uneven[k]["features"] - rr[0]["features"])))},
                          "cpu_vs_a100": {**compare(rr[0], cpu[k]), "L_G_kw_cpu": float(cpu[k]["features"][iL]),
                                          "L_G_kw_a100": float(rr[0]["features"][iL]),
                                          "share_diffs": {names[j]: float(cpu[k]["features"][j] - rr[0]["features"][j])
                                                          for j in ishare},
                                          "max_abs_diff_by_category": {
                                              c: float(np.nanmax(np.abs(cpu[k]["features"][[j for j in range(len(names))
                                                                                             if cats[j] == c]]
                                                                        - rr[0]["features"][[j for j in range(len(names))
                                                                                             if cats[j] == c]])))
                                              for c in sorted(set(cats))}}})
    a100_stable = all(d["L_G_identical"] and d["zero_indicator_identical"] and d["max_share_change"] == 0
                      and all(c.get("feasibility_decisions_changed", 1) == 0 and c.get("constraint_flags_changed", 1) == 0
                              for c in d["repeat_vs_first"]) for d in decisions)
    out = {"purpose": "envpred v3 nondeterminism and device diagnostic (features only; no labels, no prediction quality)",
           "versions": versions(), "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                            "cudnn": bool(torch.backends.cudnn.allow_tf32)},
           "repetitions": REPS, "production_batch": 1024, "uneven_batch": UNEVEN_BATCH,
           "deterministic_algorithms_attempt": det_attempt,
           "a100_repetitions_decision_stable": bool(a100_stable),
           "by_category": by_cat, "decisions": decisions,
           "top_features_by_max_abs_diff": sorted(per_feature, key=lambda r: -r["max_abs_diff"])[:25],
           "per_feature": per_feature}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("versions", "deterministic_algorithms_attempt",
                                          "a100_repetitions_decision_stable", "by_category")}, indent=1, default=float))


if __name__ == "__main__":
    main()
