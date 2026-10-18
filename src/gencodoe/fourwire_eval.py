"""Frozen evaluation of the unbalanced GENCO extension (docs/unbalanced_genco_protocol.md, amendment 1).

    python -m gencodoe.fourwire_eval --device cuda --out results/fourwire_eval_test.json

Calibration on calib only; test opened only here, after the final fits. Methods:
U1 prior, PhaseGraph (U2), PhaseGENCO, iterated four-wire sweep U3 (validated on val), balanced GENCO (BG)
and exact balanced sweep (BS) as contextual baselines, OpenDSS truth.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import fourwire_model as fm
from . import fourwire_train as ft
from .fourwire_labels import SPLITS, load_split
from .fourwire_net import V_PN_NOM, load_fourwire, phase_neutral_pu, sweep

VMIN, VMAX = ft.VMIN, ft.VMAX
METHODS4 = ("U1", "PhaseGraph", "PhaseGENCO", "U3")


def _scaler(s):
    x = fm.Scaler.__new__(fm.Scaler)
    x.mu, x.sd = np.array(s["mu"]), np.array(s["sd"])
    return x


def dates_of(split: str, scenarios: np.ndarray) -> np.ndarray:
    from .datakit_fluvius import solved_draws
    d = solved_draws(SPLITS[split][0] / "enwl1_fluvius" / "raw" / "fluvius_draws.parquet").set_index("scenario")
    return np.array([str(np.datetime64("2024-01-01") + int(s) // 96) for s in d.loc[scenarios, "step"]])


class Learned:
    def __init__(self, kind: str, tag: str, st: fm.Static, device):
        self.kind, self.dev = kind, device
        run = ft.OUT / "runs" / f"{kind.lower()}_{tag}"
        self.meta = json.loads((run / "meta.json").read_text())
        sc = self.meta["scalers"]
        self.xs, self.es = _scaler(sc["x"]), _scaler(sc["edge"])
        self.ems = _scaler(sc["emb"]) if sc.get("emb") else None
        self.model = fm.GNS_heterogeneous_4wire(self.meta["node_in"]).to(device)
        self.model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.b = fm.Batcher(st, self.es(st.edge_raw), device)
        self.rs = torch.as_tensor(np.array(sc["residual"], np.float32), device=device)
        self.n = st.net.n_bus
        self.ckpt_sha = ft.sha(run / "best.pt")

    def predict(self, x, v1, emb=None, bs=256):
        out, t_model = [], 0.0
        with torch.no_grad():
            for k in range(0, len(x), bs):
                xx = self.xs(x[k:k + bs])
                if self.ems is not None:
                    xx = np.concatenate([xx, self.ems(emb[k:k + bs].astype(np.float32))], axis=2)
                b = len(xx)
                xt = torch.as_tensor(xx, device=self.dev).reshape(b * self.n, -1)
                ei, ea = self.b.graph(b)
                if self.dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                r = self.model(xt, ei, ea, b).view(b, self.n, 8) * self.rs
                if self.dev.type == "cuda":
                    torch.cuda.synchronize()
                t_model += time.perf_counter() - t
                r = r.cpu().numpy()
                out.append(v1[k:k + bs] + (r[..., :4] + 1j * r[..., 4:]) * V_PN_NOM)
        return np.concatenate(out), t_model


def summaries4(st, v, s):
    """(S, 4): min V_pn, max V_pn, max phase loading, LV-root transformer loading proxy."""
    phys = fm.Physics(st, torch.device("cpu"))
    out = []
    for k in range(0, len(v), 512):
        vt = torch.as_tensor(v[k:k + 512], dtype=torch.complex64)
        cur = phys.currents(vt, torch.as_tensor(s[k:k + 512], dtype=torch.complex64))
        out.append(fm.summaries(phys, vt, cur).numpy())
    return np.concatenate(out)


def state_metrics(st, v, s, vt, it, conv):
    net = st.net
    c = conv
    vpn, vpt = phase_neutral_pu(net, v[c].astype(complex)), phase_neutral_pu(net, vt[c].astype(complex))
    e = np.abs(vpn - vpt)
    phys = fm.Physics(st, torch.device("cpu"))
    cur = []
    for k in range(0, int(c.sum()), 512):
        cur.append(phys.currents(torch.as_tensor(v[c][k:k + 512], dtype=torch.complex64),
                                 torch.as_tensor(s[c][k:k + 512], dtype=torch.complex64)).numpy())
    cur = np.concatenate(cur)
    rat = net.rating_a[1:] / 1e3
    sm, smt = summaries4(st, v[c], s[c]), summaries4(st, vt[c], s[c])
    return {"states": int(c.sum()), "vpn_mae": float(e.mean()), "vpn_p95": float(np.percentile(e, 95)),
            "vpn_p99": float(np.percentile(e, 99)), "worst_customer_err_mae": float(np.abs(vpn.min(1) - vpt.min(1)).mean()),
            "worst_customer_err_max": float(np.abs(vpn.min(1) - vpt.min(1)).max()),
            "neutral_v_mae_pu": float(np.abs(np.abs(v[c][:, :, 3]) - np.abs(vt[c][:, :, 3])).mean() / V_PN_NOM),
            "phase_loading_mae": float((np.abs(np.abs(cur[:, 1:, :3]) - np.abs(it[c][:, 1:, :3])) / rat[None, :, None]).mean()),
            "neutral_current_mae_a": float(np.abs(np.abs(cur[:, 1:, 3]) - np.abs(it[c][:, 1:, 3])).mean() * 1e3),
            "transformer_loading_proxy_mae": float(np.abs(sm[:, 3] - smt[:, 3]).mean())}


def classification(pred_sum, truth_sum, truth_flags, conv, near):
    """Raw (uncalibrated) violation classification from predicted summaries vs OpenDSS flags."""
    p = np.column_stack([pred_sum[:, 0] < VMIN, pred_sum[:, 1] > VMAX, pred_sum[:, 2] > 1.0, pred_sum[:, 3] > 1.0])
    out = {}
    for scope, m in (("all", conv), ("near_limit", conv & near)):
        res, bals = {}, []
        for j, name in enumerate(("undervoltage", "overvoltage", "cable_overload", "transformer_overload")):
            y, yh = truth_flags[m, j], p[m, j]
            tp, fp, fn, tn = (y & yh).sum(), (~y & yh).sum(), (y & ~yh).sum(), (~y & ~yh).sum()
            r = {"precision": float(tp / (tp + fp)) if tp + fp else None, "recall": float(tp / (tp + fn)) if tp + fn else None,
                 "positives": int(y.sum())}
            if y.any() and (~y).any():
                r["balanced_accuracy"] = float(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
                bals.append(r["balanced_accuracy"])
            res[name] = r
        res["mean_balanced_accuracy"] = float(np.mean(bals)) if bals else None
        out[scope] = res
    return out


def truth_ok(summ, flags, conv, direction):
    ok = conv & ~flags[:, 2] & ~flags[:, 3]
    return ok & (~flags[:, 0] if direction == "import" else ~flags[:, 1])


class GlobalMargins:
    """Amendment 1: one global (unbinned) split-conformal margin per criterion, fitted on CONVERGED calibration
    states; alpha split over 4 criteria (Bonferroni) as in FourWireMargins. Mondrian bins with alpha/4 = 0.0025
    need >= ~400 states per bin for a finite margin; with 2,953 calibration states most bins were infinite, so
    even an exact predictor could not clear them (dry run on calib/val)."""

    def __init__(self, pred, truth, conv, alpha=0.01):
        a = alpha / 4
        p, t = pred[conv], truth[conv]
        sc = [p[:, 0] - t[:, 0], t[:, 1] - p[:, 1], t[:, 2] / np.maximum(p[:, 2], 1e-9), t[:, 3] / np.maximum(p[:, 3], 1e-9)]
        self.lo, self.hi = [], []
        for x in sc:
            x = np.sort(x)
            n = len(x)
            i_hi, i_lo = int(np.ceil((n + 1) * (1 - a))) - 1, int(np.floor((n + 1) * a)) - 1
            self.hi.append(x[i_hi] if i_hi < n else np.inf)
            self.lo.append(x[i_lo] if i_lo >= 0 else -np.inf)

    def screen(self, pred, direction):
        pv, px, pl, pt = pred.T
        with np.errstate(invalid="ignore"):
            pess = (pv - self.hi[0], px + self.hi[1], pl * self.hi[2], pt * self.hi[3])
            opt = (pv - self.lo[0], px + self.lo[1], pl * self.lo[2], pt * self.lo[3])
        chk_v, chk_x = direction == "import", direction == "export"
        safe = (pess[2] <= 1.0) & (pess[3] <= 1.0) & ((pess[0] >= VMIN) if chk_v else True) & ((pess[1] <= VMAX) if chk_x else True)
        viol = (opt[2] > 1.0) | (opt[3] > 1.0) | ((opt[0] < VMIN) if chk_v else False) | ((opt[1] > VMAX) if chk_x else False)
        out = np.full(len(pred), "uncertain", dtype=object)
        out[safe & ~viol] = "safe"
        out[viol] = "violated"
        out[~np.isfinite(pred).all(1)] = "uncertain"                 # a method without a finite state defers
        return out


def screen(cal_pred, cal_truth, cal_conv, te_pred, te_truth_flags, te_conv, alpha=0.01):
    m = GlobalMargins(cal_pred, cal_truth, cal_conv, alpha)
    res = {"margins": {"hi": [float(x) for x in m.hi], "lo": [float(x) for x in m.lo]}}
    for d in ("import", "export"):
        s = m.screen(te_pred, d)
        ok = truth_ok(None, te_truth_flags, te_conv, d)
        res[d] = {"safe": float((s == "safe").mean()), "violated": float((s == "violated").mean()),
                  "fallback": float((s == "uncertain").mean()), "exact_calls": int((s == "uncertain").sum()),
                  "false_safe": int(((s == "safe") & ~ok).sum()), "false_violated": int(((s == "violated") & ok).sum()),
                  "_uncertain": (s == "uncertain")}
    res["fallback_mean"] = float(np.mean([res[d]["fallback"] for d in ("import", "export")]))
    res["false_safe_total"] = res["import"]["false_safe"] + res["export"]["false_safe"]
    return res


def boot_diff(unc_a, unc_b, dates, n=2000, seed=0):
    """Paired date-block bootstrap (within season) of the fallback-share difference A - B."""
    from .report_annual import season
    rng = np.random.default_rng(seed)
    ud = np.unique(dates)
    per = {d: (unc_a[dates == d].mean(), unc_b[dates == d].mean(), int((dates == d).sum())) for d in ud}
    by = {}
    for d in ud:
        by.setdefault(season(d), []).append(d)
    st = np.empty(n)
    for b in range(n):
        pick = [x for ds in by.values() for x in rng.choice(ds, len(ds))]
        w = np.array([per[x][2] for x in pick], float)
        st[b] = (np.array([per[x][0] for x in pick]) @ w - np.array([per[x][1] for x in pick]) @ w) / w.sum()
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def run_split(split, st, learned, device, allow_test):
    from .enwl import load_enwl1
    from .envpred3 import RUN_HYDRA, RUN_LOCAL, load_genco
    from .four_wire import state_summary
    from . import feeder as fdr
    d = load_split(split, allow_test=allow_test)
    s = (d["p_kw"] + 1j * d["q_kvar"]).astype(complex)
    times = {}
    t = time.perf_counter()
    v1 = fm.prior_u1(st, d["p_kw"].astype(float), d["q_kvar"].astype(float))
    times["U1_prior_cpu_s"] = time.perf_counter() - t
    t = time.perf_counter()
    x = fm.node_features(st, d["p_kw"], d["q_kvar"], v1)
    times["features_cpu_s"] = time.perf_counter() - t
    t = time.perf_counter()
    ft.embeddings(split, device.type) if not (ft.EMB / f"{split}.npz").exists() else None
    times["genco_embeddings_s_incl_io"] = time.perf_counter() - t
    z = np.load(ft.EMB / f"{split}.npz")
    if not np.array_equal(z["scenario"], d["scenario"]):
        raise AssertionError("embedding order")
    V = {"U1": v1}
    for name, lm in learned.items():
        V[name], times[f"{name}_model_s"] = lm.predict(x, v1, z["emb"] if lm.ems is not None else None)
    t = time.perf_counter()
    V["U3"] = np.concatenate([sweep(st.net, st.th, s[k:k + 2048], iterations=50, tol=1e-8)[0] for k in range(0, len(s), 2048)])
    times["U3_cpu_s"] = time.perf_counter() - t
    # balanced contextual baselines on the same states
    bal = load_enwl1()
    feeder = bal.to_feeder()
    pb = np.zeros((len(s), feeder.n_bus))
    qb = np.zeros_like(pb)
    pb[:, bal.customer_bus] = -d["p_kw"] / 1e3
    qb[:, bal.customer_bus] = -d["q_kvar"] / 1e3
    oracle, emb = load_genco(device.type, RUN_LOCAL if RUN_LOCAL.exists() else RUN_HYDRA)
    t = time.perf_counter()
    bg = oracle(feeder, pb, qb)
    times["BG_s"] = time.perf_counter() - t
    t = time.perf_counter()
    bs = fdr.sweep_pf(feeder, pb, qb)
    times["BS_cpu_s"] = time.perf_counter() - t
    summ = {k: summaries4(st, v, s) for k, v in V.items()}
    summ["BG"] = np.column_stack(state_summary(bg))
    summ["BS"] = np.column_stack(state_summary(bs))
    return d, s, V, summ, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out", type=Path, default=Path("results/fourwire_eval_test.json"))
    ap.add_argument("--eval-split", default="test", choices=["test", "val"], help="val = dry run before freezing")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    net = load_fourwire()
    st = fm.Static(net, ft.thevenin())
    learned = {"PhaseGraph": Learned("phasegraph", a.tag, st, dev), "PhaseGENCO": Learned("phasegenco", a.tag, st, dev)}
    cal = run_split("calib", st, learned, dev, allow_test=False)
    te = run_split(a.eval_split, st, learned, dev, allow_test=a.eval_split == "test")   # test opened only here
    dC, sC, VC, smC, _ = cal
    dT, sT, VT, smT, times = te
    dates = dates_of("val" if a.eval_split == "val" else "test", dT["scenario"])
    near = ((np.abs(dT["summary"][:, 0] - VMIN) < 0.02) | (np.abs(dT["summary"][:, 1] - VMAX) < 0.02)
            | ((dT["summary"][:, 2] > 0.9) & (dT["summary"][:, 2] < 1.1)) | ((dT["summary"][:, 3] > 0.9) & (dT["summary"][:, 3] < 1.1)))
    n = len(dT["scenario"])
    opendss_s = float(np.mean([float(np.load(f)["seconds"]) / len(np.load(f)["scenario"])
                               for f in sorted((Path("data_out/fourwire/labels/calib")).glob("shard_*.npz"))]))
    out = {"protocol": "docs/unbalanced_genco_protocol.md (+ amendment 1)", "status": "exploratory extension",
           "test_states": n, "test_converged": int(dT["converged"].sum()), "test_dates": int(len(np.unique(dates))),
           "checkpoints": {k: {"sha256": v.ckpt_sha, "best_epoch": v.meta["best_epoch"], "n_params": v.meta["n_params"]}
                           for k, v in learned.items()},
           "state": {}, "classification": {}, "screen": {}, "runtime": {}}
    for k in METHODS4:
        out["state"][k] = state_metrics(st, VT[k], sT, dT["V"], dT["I"], dT["converged"])
    for k in list(METHODS4) + ["BG", "BS"]:
        out["classification"][k] = classification(smT[k], dT["summary"], dT["flags"], dT["converged"], near)
        out["screen"][k] = screen(smC[k], dC["summary"], dC["converged"], smT[k], dT["flags"], dT["converged"])
    per_state = {"U1": times["U1_prior_cpu_s"] / n, "U3": times["U3_cpu_s"] / n, "BS": times["BS_cpu_s"] / n,
                 "BG": times["BG_s"] / n,
                 "PhaseGraph": (times["U1_prior_cpu_s"] + times["features_cpu_s"] + times["PhaseGraph_model_s"]) / n,
                 "PhaseGENCO": (times["U1_prior_cpu_s"] + times["features_cpu_s"] + times["PhaseGENCO_model_s"]
                                + times["BG_s"]) / n}
    for k in per_state:
        out["runtime"][k] = {"method_s_per_state": per_state[k],
                             "model_only_s_per_state": times.get(f"{k}_model_s", np.nan) / n if k.startswith("Phase") else None,
                             "end_to_end_s_per_state": per_state[k] + out["screen"][k]["fallback_mean"] * opendss_s}
    out["runtime"]["opendss_s_per_state"] = opendss_s
    out["runtime"]["note"] = ("method time on the hardware used (four-wire learned models: U1 + features on CPU, model on "
                              "A100; PhaseGENCO adds the balanced GENCO pass, measured as BG); end-to-end adds the mean "
                              "fallback share x measured OpenDSS seconds per state")
    u1, u2 = out["state"]["U1"], out["state"]["PhaseGraph"]
    cu1, cu2 = out["classification"]["U1"]["near_limit"]["mean_balanced_accuracy"], out["classification"]["PhaseGraph"]["near_limit"]["mean_balanced_accuracy"]
    f1, f2 = out["screen"]["U1"]["fallback_mean"], out["screen"]["PhaseGraph"]["fallback_mean"]
    out["primary_U2_vs_U1"] = {
        "vpn_mae_reduction": 1 - u2["vpn_mae"] / u1["vpn_mae"], "c1_ge_30pct": 1 - u2["vpn_mae"] / u1["vpn_mae"] >= 0.30,
        "near_limit_mean_balanced_accuracy": [cu1, cu2], "c2_improved": cu2 is not None and cu1 is not None and cu2 > cu1,
        "fallback_reduction_pp": 100 * (f1 - f2), "c3_ge_5pp": f1 - f2 >= 0.05,
        "false_safe": [out["screen"]["U1"]["false_safe_total"], out["screen"]["PhaseGraph"]["false_safe_total"]],
        "c4_not_more_false_safe": out["screen"]["PhaseGraph"]["false_safe_total"] <= out["screen"]["U1"]["false_safe_total"],
        "c5_runtime_reported": True}
    p = out["primary_U2_vs_U1"]
    p["success"] = bool(p["c1_ge_30pct"] and p["c2_improved"] and p["c3_ge_5pp"] and p["c4_not_more_false_safe"])
    ua = np.mean([out["screen"]["PhaseGENCO"][d]["_uncertain"] for d in ("import", "export")], axis=0)
    ub = np.mean([out["screen"]["PhaseGraph"][d]["_uncertain"] for d in ("import", "export")], axis=0)
    ci = boot_diff(ua, ub, dates)
    diff = out["screen"]["PhaseGENCO"]["fallback_mean"] - out["screen"]["PhaseGraph"]["fallback_mean"]
    out["genco_9b_PhaseGENCO_minus_PhaseGraph"] = {
        "fallback_diff_pp": 100 * diff, "bootstrap_ci95_pp": [100 * ci[0], 100 * ci[1]],
        "c_reduces_ge_5pp": -diff >= 0.05, "c_ci_excludes_zero": ci[1] < 0 or ci[0] > 0,
        "false_safe": [out["screen"]["PhaseGraph"]["false_safe_total"], out["screen"]["PhaseGENCO"]["false_safe_total"]],
        "c_not_more_false_safe": out["screen"]["PhaseGENCO"]["false_safe_total"] <= out["screen"]["PhaseGraph"]["false_safe_total"],
        "end_to_end_s_per_state": [out["runtime"]["PhaseGraph"]["end_to_end_s_per_state"],
                                   out["runtime"]["PhaseGENCO"]["end_to_end_s_per_state"]],
        "vpn_mae": [out["state"]["PhaseGraph"]["vpn_mae"], out["state"]["PhaseGENCO"]["vpn_mae"]],
        "neutral_current_mae_a": [out["state"]["PhaseGraph"]["neutral_current_mae_a"], out["state"]["PhaseGENCO"]["neutral_current_mae_a"]]}
    g = out["genco_9b_PhaseGENCO_minus_PhaseGraph"]
    g["genco_positive"] = bool(g["c_reduces_ge_5pp"] and ci[1] < 0 and g["c_not_more_false_safe"])
    for k in out["screen"]:
        for d in ("import", "export"):
            out["screen"][k][d].pop("_uncertain", None)
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, a.out)
    print(json.dumps({k: out[k] for k in ("primary_U2_vs_U1", "genco_9b_PhaseGENCO_minus_PhaseGraph")}, indent=1, default=float))


if __name__ == "__main__":
    main()
