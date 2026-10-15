"""Stage E: conformal margins on calibration days, checked on untouched test days.

Fits split-conformal quantiles of |Vm error| (non-slack buses) and |loading error|
(branches) on the calibration-day dataset only, globally and per element, then
on the test-day dataset reports:
  * coverage (share of elements inside +/- q) overall and near the limits;
  * screening (conformal.screen, all constraints): share of states labelled
    safe / violated / uncertain (= AC fallback), and false-safe states
    (labelled safe but violating under AC) -- the number that must be 0;
  * conservatism: share of truly feasible states not labelled safe.

Usage:
  python -m gencodoe.conformal_eval --run mlruns/<exp>/<run> --config configs/... \
      --calib data_out/datakit_v2_calib --test data_out/datakit_v2_test --out data_out/eval/conformal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import conformal
from . import feeder as fdr
from .enwl import load_enwl1
from .evaluate import VMAX, VMIN, _loading, _sweep_voltage, load_truth, violations


def predict(kind: str, feeder, truth, model=None):
    """(vm, va) predictions of a GENCO model, or of the fixed-point prior baseline."""
    if kind == "genco":
        from .genco import frames_to_graphs
        return model.predict_graphs(frames_to_graphs(*truth["frames"]))
    v = _sweep_voltage(feeder, -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva, max_iter=1)
    return np.abs(v), np.angle(v)


def pfres(feeder, truth, vm, va):
    p, q = -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva
    return fdr.branch_flows_kcl(feeder, vm, va, p, q)


def assess(feeder, truth, pred: fdr.PFResult, q_v, q_l) -> dict:
    """Coverage and screening on one dataset for margins q_v, q_l: scalars, per element
    (n_bus - 1,), or per state and element (n, n_bus - 1)."""
    true = pfres(feeder, truth, truth["vm"], truth["va"])
    ev, el = np.abs(pred.vm[:, 1:] - true.vm[:, 1:]), np.abs(pred.loading[:, 1:] - true.loading[:, 1:])
    qv = q_v if np.isscalar(q_v) else np.asarray(q_v)
    ql = q_l if np.isscalar(q_l) else np.asarray(q_l)
    qv = qv[None, :] if np.ndim(qv) == 1 else qv
    ql = ql[None, :] if np.ndim(ql) == 1 else ql
    near_v = (true.vm[:, 1:] < VMIN + 0.01) | (true.vm[:, 1:] > VMAX - 0.01)
    near_l = true.loading[:, 1:] > 0.9
    vm = pred.vm[:, 1:]
    lo = pred.loading[:, 1:]
    safe = ((vm + qv <= VMAX) & (vm - qv >= VMIN)).all(1) & (lo + ql <= 1.0).all(1)
    viol = ((vm - qv > VMAX) | (vm + qv < VMIN)).any(1) | (lo - ql > 1.0).any(1)
    unsure = ~safe & ~viol
    t_viol = violations(true.vm, true.loading)
    return {"coverage_vm": float((ev <= qv).mean()), "coverage_loading": float((el <= ql).mean()),
            "coverage_vm_near_limits": float((ev <= qv)[near_v].mean()) if near_v.any() else None,
            "coverage_loading_near_limit": float((el <= ql)[near_l].mean()) if near_l.any() else None,
            "share_safe": float(safe.mean()), "share_violated": float(viol.mean()),
            "share_uncertain_ac_fallback": float(unsure.mean()),
            "false_safe_states": int((safe & t_viol).sum()), "false_violated_states": int((viol & ~t_viol).sum()),
            "violating_states": int(t_viol.sum()), "states": int(len(t_viol)),
            "conservatism_feasible_not_safe": float((~safe & ~t_viol).sum() / max((~t_viol).sum(), 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.01, 0.001])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    feeder = load_enwl1().to_feeder()
    cal, test = load_truth(a.calib), load_truth(a.test)
    for name, t, want in (("calib", cal, "calib"), ("test", test, "test")):
        got = set(t["draws"]["split"]) if "split" in t["draws"] else None
        if got is not None and got != {want}:
            raise ValueError(f"{name} data contains splits {got}")
    model = None
    kinds = ["fixed_point_1"]
    if a.run:
        from .genco import GencoModel
        model = GencoModel(a.config, a.run / "artifacts/model/best_model_state_dict.pt",
                           a.run / "artifacts/stats/normalizer_stats.pt", device=a.device)
        kinds.append("genco")
    results = []
    for kind in kinds:
        pc = pfres(feeder, cal, *predict(kind, feeder, cal, model))
        pt = pfres(feeder, test, *predict(kind, feeder, test, model))
        tc = pfres(feeder, cal, cal["vm"], cal["va"])
        for alpha in a.alpha:
            for mode in ("global", "per-el", "binned"):
                q_l = conformal.calibrate(pc.loading[:, 1:], tc.loading[:, 1:], alpha, mode == "per-el")
                if mode == "binned":
                    qb = conformal.calibrate_binned(pc.vm[:, 1:], tc.vm[:, 1:], alpha)
                    q_v_cal = conformal.binned_margin(pc.vm[:, 1:], qb)
                    q_v_test = conformal.binned_margin(pt.vm[:, 1:], qb)
                    q_v_rep = {"q_v_bins": qb.tolist(), "q_v_bin_edges": [float(e) for e in conformal.VM_BIN_EDGES]}
                else:
                    q_v_cal = q_v_test = conformal.calibrate(pc.vm[:, 1:], tc.vm[:, 1:], alpha, mode == "per-el")
                    q_v_rep = {}
                r = {"predictor": kind, "alpha": alpha, "mode": mode,
                     "q_v": float(np.max(q_v_test)), "q_v_median": float(np.median(q_v_test)),
                     "q_l": float(np.max(q_l)), **q_v_rep,
                     "test": assess(feeder, test, pt, q_v_test, q_l), "calib": assess(feeder, cal, pc, q_v_cal, q_l)}
                results.append(r)
                s = r["test"]
                print(f"{kind:14s} alpha={alpha:<6} {mode:6s} q_v={r['q_v']:.4g} "
                      f"q_l={r['q_l']:.4g} | test cov_v {s['coverage_vm']:.4f} near {s['coverage_vm_near_limits']} "
                      f"| safe {s['share_safe']:.3f} AC-fallback {s['share_uncertain_ac_fallback']:.3f} "
                      f"false-safe {s['false_safe_states']}/{s['violating_states']} conservative {s['conservatism_feasible_not_safe']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    from .limits import record as limits_record
    a.out.write_text(json.dumps({"voltage_limits": limits_record(), "calib": str(a.calib), "test": str(a.test), "run": str(a.run), "results": results},
                                indent=1))


if __name__ == "__main__":
    main()
