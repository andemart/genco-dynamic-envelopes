"""Envelope prediction with an exact gate (docs/envpred_protocol.md incl. amendment 1; post-hoc, exploratory).

    python -m gencodoe.envpred features --workers 8          # replay scenarios, features for all 180 files
    python -m gencodoe.envpred fit                           # heads, shrink, validation selection
    python -m gencodoe.envpred gate --workers 8              # exact gate + fallback on the test dates
    python -m gencodoe.envpred report --out results/envpred_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANNUAL = Path("data_out/doe/annual")
W1 = Path("data_out/doe/weekend/w1_placement")
LABELS = Path("data_out/doe/envpred_labels_corrected")   # amendment 3d: fresh paths (stale node caches of reused names)
OUT = Path("data_out/envpred")
SPLIT = Path("results/envpred_split_v2.json")              # amendment 2
RUN = Path("mlruns_hydra/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee")
CONFIG = Path("configs/genco_pf_enwl1_lv_prior_v2.yaml")
CALIB = Path("data_out/datakit_v2_calib")
B = "b_opendss_exact"
ARMS = {"P": ("P",), "PC": ("P", "C"), "PG": ("P", "G"), "PCG": ("P", "C", "G")}
FAMILY = {"non_genco": ("P", "PC"), "genco": ("PG", "PCG")}
CAPS_KW = {"export": 5.0, "import": 7.4}
N_S, N_V, N_U = 200, 1000, 200
SCENARIO_SEED = 101


def label_files() -> list[tuple[Path, int]]:
    files = [(p, 101) for p in sorted(ANNUAL.glob(f"*_annual_4w_{B}_sph1_seed101_uk_esqcr.json"))]
    for hh in (202, 303, 404, 505):
        files += [(p, hh) for p in sorted(W1.glob(f"*_{B}_sph1_seed101_hh{hh}_uk_esqcr.json"))]
    for hh in (101, 202):                                   # train/val/calib labels (amendment 2)
        files += [(p, hh) for p in sorted(LABELS.glob(f"*_{B}_sph1_seed101_hh{hh}_uk_esqcr.json"))]
    return files


def missing_labels() -> list[str]:
    """Label (date, placement) blocks of amendment 2 without a finished file."""
    days = [d for d in Path("results/envpred_label_days.txt").read_text().split() if d]
    return [f"{d}_hh{hh}" for hh in (101, 202) for d in days
            if not (LABELS / f"{d}_{B}_sph1_seed101_hh{hh}_uk_esqcr.json").exists()]


def stats(x: np.ndarray) -> np.ndarray:
    """(n_s, k) -> (7k,): mean, min, 2/10/90/98th percentile, max over scenarios (amendment 1)."""
    return np.concatenate([x.mean(0), x.min(0), *np.percentile(x, [2, 10, 90, 98], axis=0), x.max(0)])


def replay(day: str, hh: int):
    """Yield (step, p, q, pv, qv) exactly as doe_day.run_day drew them (hourly)."""
    from .doe_day import DayScenarios
    from .enwl import load_enwl1
    net = load_enwl1()
    fd = net.to_feeder()
    scen = DayScenarios(net.customer_bus, net.n_bus, seed=hh)
    rng = np.random.default_rng(SCENARIO_SEED)
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
    for k in range(0, 96, 4):
        p, q = scen.sample(t0 + k, N_S, rng, fd.base_mva)
        pv, qv = scen.sample(t0 + k, N_V, rng, fd.base_mva)
        yield t0 + k, p, q, pv, qv


_CTX = {}


def _ctx(genco: bool = True):
    if not _CTX:
        import torch
        torch.set_num_threads(1)
        from .doe_day import customer_groups
        from .enwl import load_enwl1
        from .unbalanced import UnbalancedENWL1
        net = load_enwl1()
        ub = UnbalancedENWL1()
        _CTX.update(net=net, fd=net.to_feeder(), ub=ub, groups=customer_groups(net, ub),
                    phase=np.array([int(lb.split(".")[1]) for lb in ub.load_bus]))
    if genco and "oracle" not in _CTX:                     # the gate needs no GENCO
        from .genco import GencoModel, GencoOracle, read_raw
        model = GencoModel(CONFIG, RUN / "artifacts/model/best_model_state_dict.pt",
                           RUN / "artifacts/stats/normalizer_stats.pt")
        _CTX["oracle"] = GencoOracle(model, read_raw(CALIB / "enwl1_fluvius" / "raw", scenarios=[0]))
    return _CTX


def file_features(path: Path, hh: int) -> Path:
    """Features (shared by both directions) and labels of one labelled B file."""
    from . import feeder as fdr
    from .four_wire import state_summary
    from .option2 import genco_features, phase_features
    out = OUT / "features" / (path.stem + ".npz")
    if out.exists():
        return out
    c = _ctx()
    net, fd = c["net"], c["fd"]
    doc = json.loads(path.read_text())
    m = doc["meta"]
    if m.get("household_seed", hh) != hh or m.get("scenario_seed", SCENARIO_SEED) != SCENARIO_SEED:
        raise SystemExit(f"{path}: seed mismatch")
    rows = {(r["step"], r["direction"], r["method"]): r for r in doc["rows"]}
    F = {"P": [], "C": [], "G": []}
    lab, t_blk = [], {"P": 0.0, "C": 0.0, "G": 0.0}
    for step, p, q, _, _ in replay(m["day"], hh):
        pd_mw, qd = -p * fd.base_mva, -q * fd.base_mva
        t = time.perf_counter()
        F["P"].append(stats(phase_features(pd_mw, qd, net.customer_bus, c["groups"], c["phase"])))
        t_blk["P"] += time.perf_counter() - t
        t = time.perf_counter()
        sw = np.column_stack(state_summary(fdr.sweep_pf(fd, p, q)))
        t_sw = time.perf_counter() - t
        t = time.perf_counter()
        gs, ge = genco_features(c["oracle"], fd, pd_mw, qd)
        F["G"].append(np.concatenate([stats(gs), ge.mean(0)]))
        t_blk["G"] += time.perf_counter() - t
        for d in ("export", "import"):
            b, ac = rows[(step, d, B)], rows[(step, d, "ac")]
            lab.append({"step": step, "direction": d, "L_B": b["limit_kw"], "L_ac": ac["limit_kw"],
                        "b_calls": b["ac_calls"], "b_runtime_s": b["runtime_s"],
                        "b_p4": b.get("prob_ok_unbalanced", np.nan), "b_oracle": b["prob_ok_oracle"]})
        F["C"].append(np.concatenate([[np.nan], stats(sw)]))      # L_ac filled per direction below
        t_blk["C"] += t_sw
    lab = pd.DataFrame(lab)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, P=np.array(F["P"]), C=np.array(F["C"]), G=np.array(F["G"]), steps=lab.step.unique(),
             labels=lab.to_json(orient="records"), day=m["day"], hh=hh, source=str(path),
             timing=json.dumps({k: v / 24 for k, v in t_blk.items()}))
    return out


def load_table() -> tuple[pd.DataFrame, dict]:
    """One row per (file, step, direction) with feature blocks and the split of its date."""
    split = json.loads(SPLIT.read_text())
    of = {d: s for s in ("train", "val", "calib", "test") for d in split[s]}
    rows, blocks = [], {"P": [], "C": [], "G": []}
    for f in sorted((OUT / "features").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        lab = pd.DataFrame(json.loads(str(z["labels"])))
        idx = {s: i for i, s in enumerate(z["steps"])}
        for r in lab.itertuples():
            i = idx[r.step]
            c = z["C"][i].copy()
            c[0] = r.L_ac
            blocks["P"].append(z["P"][i]), blocks["C"].append(c), blocks["G"].append(z["G"][i])
            rows.append({"file": str(z["source"]), "day": str(z["day"]), "hh": int(z["hh"]), "split": of[str(z["day"])],
                         **r._asdict()})
    df = pd.DataFrame(rows).drop(columns="Index")
    return df, {k: np.array(v) for k, v in blocks.items()}


def arm_X(blocks, arm, mask):
    return np.hstack([blocks[b][mask] for b in ARMS[arm]])


def proxy_retention(L_s, L_B, direction):
    tol = 1e-3 * np.vectorize(CAPS_KW.get)(direction)
    ok = L_s <= L_B + tol
    return float(np.minimum(L_s, L_B)[ok].sum() / max(L_B.sum(), 1e-12))


def fit(out_json: Path):
    from .option2 import fit_heads, predict
    df, blocks = load_table()
    pred = {}
    info = {"rows_per_split": df.split.value_counts().to_dict(), "delta_kw": {}, "val_proxy_retention": {},
            "iterations": {}, "mae_val_kw": {}, "mae_calib_kw": {}, "inference_s_per_interval": {}}
    df_out = df[["file", "day", "hh", "split", "step", "direction", "L_B"]].copy()
    for arm in ARMS:
        L_hat = np.full(len(df), np.nan)
        for d in ("export", "import"):
            m = {s: ((df.split == s) & (df.direction == d)).to_numpy() for s in ("train", "val", "calib", "test")}
            heads = fit_heads(arm_X(blocks, arm, m["train"]), df.L_B[m["train"]].to_numpy()[:, None],
                              arm_X(blocks, arm, m["val"]), df.L_B[m["val"]].to_numpy()[:, None])
            dm = (df.direction == d).to_numpy()
            t = time.perf_counter()
            L_hat[dm] = np.clip(predict(heads, arm_X(blocks, arm, dm))[:, 0], 0, CAPS_KW[d])
            info["inference_s_per_interval"][f"{arm}_{d}"] = (time.perf_counter() - t) / dm.sum()
            info["iterations"][f"{arm}_{d}"] = int(heads[0].n_iter_)
            r = np.sort(L_hat[m["calib"]] - df.L_B[m["calib"]].to_numpy())
            k = int(np.ceil((len(r) + 1) * 0.95)) - 1
            info["delta_kw"][f"{arm}_{d}"] = delta = max(float(r[min(k, len(r) - 1)]), 0.0)
            for s in ("val", "calib"):
                info[f"mae_{s}_kw"][f"{arm}_{d}"] = float(np.abs(L_hat[m[s]] - df.L_B[m[s]]).mean())
            df_out.loc[dm, f"Lhat_{arm}"] = L_hat[dm]
            df_out.loc[dm, f"Ls_{arm}"] = np.maximum(L_hat[dm] - delta, 0.0)
        v = (df.split == "val").to_numpy()
        info["val_proxy_retention"][arm] = proxy_retention(df_out[f"Ls_{arm}"][v].to_numpy(), df.L_B[v].to_numpy(),
                                                           df.direction[v].to_numpy())
    vr = info["val_proxy_retention"]
    info["missing_label_blocks"] = missing_labels()
    info["selected"] = {fam: max(arms, key=lambda a: (vr[a], -len(ARMS[a]))) for fam, arms in FAMILY.items()}
    df_out.to_parquet(OUT / "predictions.parquet")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def gate_file(path_str: str) -> Path:
    """Exact gate + fallback + fresh feasibility for every arm on one test file."""
    from .cplus import Counter, FourWireOracle, repair, verify_repaired
    from .envelopes import acceptance_threshold
    path = Path(path_str)
    out = OUT / "gate" / (path.stem + ".json")
    if out.exists():
        return out
    c = _ctx(genco=False)
    pr = pd.read_parquet(OUT / "predictions.parquet")
    pr = pr[pr.file == path_str]
    hh = int(pr.hh.iloc[0])
    thr = acceptance_threshold(N_S, 0.05, 0.05)
    oracle = FourWireOracle(c["net"], c["ub"])
    res = []
    for step, p, q, pv, qv in replay(pr.day.iloc[0], hh):
        for d in ("export", "import"):
            row = pr[(pr.step == step) & (pr.direction == d)].iloc[0]
            tol = 1e-3 * CAPS_KW[d]
            for arm in ARMS:
                L_s = float(row[f"Ls_{arm}"])
                r = {"step": int(step), "direction": d, "arm": arm, "L_B": float(row.L_B), "L_s": L_s,
                     "gate_calls": 0, "fallback_calls": 0, "gate_s": 0.0, "fallback_s": 0.0, "gate_pass": None,
                     "fallback": False, "predicted_zero": L_s <= tol, "zero_sentinel": False}
                issued = 0.0
                if not r["predicted_zero"]:
                    t = time.perf_counter()
                    share, n = oracle.share(p, q, L_s, d)
                    r.update(gate_calls=n, gate_s=time.perf_counter() - t, gate_share=share, gate_pass=share >= thr)
                    if share >= thr:
                        issued = L_s
                    else:
                        t = time.perf_counter()
                        issued, s_lo, calls, zinf = repair(oracle, p, q, L_s, d, thr)
                        r.update(fallback=True, fallback_calls=calls, fallback_s=time.perf_counter() - t,
                                 zero_sentinel=bool(zinf), fallback_share=s_lo)
                r["issued"] = float(issued)
                if issued > tol:
                    v = verify_repaired(oracle, c["fd"], c["net"].customer_bus, pv, qv, N_U, issued, d, Counter())
                    r.update(fresh_p4=v["prob_ok_unbalanced"], fresh_ac=v["prob_ok_ac_fresh"])
                res.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": path_str, "hh": hh, "threshold": thr, "rows": res}, indent=1))
    return out


def _pool(fn, items, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, *zip(*items)) if isinstance(items[0], tuple) else ex.map(fn, items)):
            print(f"[{i + 1}/{len(items)}] {r}", flush=True)


def _season_boot(days: np.ndarray, per_day: dict, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Season-stratified day-block bootstrap of a ratio statistic: per_day[day] = (num_G, num_N, den)."""
    from .report_annual import season
    rng = np.random.default_rng(seed)
    by = {}
    for d in days:
        by.setdefault(season(d), []).append(d)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = [x for ds in by.values() for x in rng.choice(ds, len(ds))]
        g, n, den = np.sum([per_day[x] for x in pick], axis=0)
        out[b] = (g - n) / den
    return out


def report(out_json: Path):
    from .cplus import sha256
    info = json.loads((OUT / "fit.json").read_text())
    rows = []
    for f in sorted((OUT / "gate").glob("*.json")):
        doc = json.loads(f.read_text())
        rows += [{**r, "file": doc["source"], "hh": doc["hh"]} for r in doc["rows"]]
    g = pd.DataFrame(rows)
    pr = pd.read_parquet(OUT / "predictions.parquet")
    test = pr[pr.split == "test"]
    if len(g) != 4 * len(test):
        raise SystemExit(f"gate rows {len(g)} != 4 x {len(test)} test rows: gate incomplete")
    df, _ = load_table()
    bmeta = df[df.split == "test"][["file", "step", "direction", "b_calls", "b_p4", "b_runtime_s"]]
    g = g.merge(bmeta, on=["file", "step", "direction"], how="left")
    g["day"] = g.file.map(lambda s: Path(s).name[:10])
    g["tol"] = g.direction.map(CAPS_KW) * 1e-3
    g["kept"] = np.minimum(g.issued, g.L_B)
    g["excess"] = np.maximum(g.issued - g.L_B, 0)
    g["calls"] = g.gate_calls + g.fallback_calls
    out = {"protocol": "docs/envpred_protocol.md (incl. amendments 1 and 2)", "missing_label_blocks": missing_labels(), "status": "post-hoc, exploratory",
           "split": json.loads(SPLIT.read_text()), "fit": info, "arms": {},
           "inputs_sha256": {str(p): sha256(p) for p, _ in label_files() if Path(p).name[:10] in set(g.day)}}
    for arm, ga in g.groupby("arm"):
        res = {}
        for scope, gs in [("combined", ga)] + list(ga.groupby("direction")):
            nz = gs[gs.issued > gs.tol]
            bnz = gs[gs.L_B > gs.tol]
            res[scope] = {
                "intervals": int(len(gs)), "retention": float(gs.kept.sum() / gs.L_B.sum()),
                "excess_over_B": float(gs.excess.sum() / gs.L_B.sum()),
                "intervals_issued_above_B": int((gs.issued > gs.L_B + gs.tol).sum()),
                "exact_calls": int(gs.calls.sum()), "exact_calls_B": int(gs.b_calls.sum()),
                "exact_call_ratio_vs_B": float(gs.calls.sum() / gs.b_calls.sum()),
                "gate_pass_rate": float(gs.gate_pass.dropna().astype(bool).mean()) if gs.gate_pass.notna().any() else None,
                "gated": int(gs.gate_pass.notna().sum()), "fallbacks": int(gs.fallback.sum()),
                "calls_per_fallback": float(gs.fallback_calls[gs.fallback].mean()) if gs.fallback.any() else None,
                "predicted_zero": int(gs.predicted_zero.sum()),
                "predicted_zero_where_B_positive": int((gs.predicted_zero & (gs.L_B > gs.tol)).sum()),
                "zero_sentinels_from_fallback": int(gs.zero_sentinel.sum()),
                "B_zero_intervals": int((gs.L_B <= gs.tol).sum()),
                "gate_s": float(gs.gate_s.sum()), "fallback_s": float(gs.fallback_s.sum()),
                "B_search_s": float(gs.b_runtime_s.sum()),
                "fresh_p4_mean": float(nz.fresh_p4.mean()) if len(nz) else None,
                "fresh_p4_share_ge_0.95": float((nz.fresh_p4 >= 0.95).mean()) if len(nz) else None,
                "fresh_ac_mean": float(nz.fresh_ac.mean()) if len(nz) else None,
                "B_fresh_p4_mean_on_B_nonzero": float(bnz.b_p4.mean()) if len(bnz) else None,
                "B_fresh_p4_share_ge_0.95": float((bnz.b_p4 >= 0.95).mean()) if len(bnz) else None,
                "mae_Lhat_kw": None}
        out["arms"][arm] = res
    for arm in ARMS:
        for d in ("export", "import"):
            t = test[test.direction == d]
            out["arms"][arm][d]["mae_Lhat_kw"] = float(np.abs(t[f"Lhat_{arm}"] - t.L_B).mean())
    sel = info["selected"]
    G, N = sel["genco"], sel["non_genco"]
    a, b = out["arms"][G]["combined"], out["arms"][N]["combined"]
    per_day = {}
    for day, gd in g.groupby("day"):
        per_day[day] = (gd[gd.arm == G].kept.sum(), gd[gd.arm == N].kept.sum(), gd[gd.arm == G].L_B.sum())
    boot = _season_boot(np.array(sorted(per_day)), per_day)
    crit = {"retention_diff": a["retention"] - b["retention"],
            "c1_retention_ge_3pp": a["retention"] - b["retention"] >= 0.03,
            "c2_exact_calls_not_more": a["exact_calls"] <= b["exact_calls"],
            "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "c3_bootstrap_lower_gt_0": float(np.percentile(boot, 2.5)) > 0,
            "c4_feasibility_not_worse": bool(a["fresh_p4_mean"] >= b["fresh_p4_mean"] - 0.005
                                             and a["fresh_p4_share_ge_0.95"] >= b["fresh_p4_share_ge_0.95"] - 0.01)}
    crit["genco_positive"] = bool(crit["c1_retention_ge_3pp"] and crit["c2_exact_calls_not_more"]
                                  and crit["c3_bootstrap_lower_gt_0"] and crit["c4_feasibility_not_worse"])
    out["decision"] = {"best_genco_arm": G, "best_non_genco_arm": N, **crit}
    out_json.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs_sha256", "split")}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "fit", "gate", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/envpred_report.json"))
    ap.add_argument("--file-index", type=int, default=None, help="gate: only this test file (Hydra array task)")
    a = ap.parse_args()
    if a.cmd == "features":
        _pool(file_features, label_files(), a.workers)
    elif a.cmd == "fit":
        fit(OUT / "fit.json")
    elif a.cmd == "gate":
        split = json.loads(SPLIT.read_text())
        files = [str(p) for p, _ in label_files() if p.name[:10] in split["test"]]
        if len(files) != 180:
            raise SystemExit(f"{len(files)} test files, expected 180")
        if a.file_index is not None:
            print(gate_file(files[a.file_index]))
        else:
            _pool(gate_file, files, a.workers)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
