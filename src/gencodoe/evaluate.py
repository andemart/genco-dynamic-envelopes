"""Physical-unit evaluation of power-flow predictors on a datakit dataset.

Ground truth is datakit's AC solution (PowerModels). Every predictor returns
voltages (vm pu, va rad) per state and bus; branch loadings are rebuilt from
those voltages and the known injections with feeder.branch_flows_kcl (Kirchhoff's
current law) for every predictor, so loading errors compare like with like. The
voltage-difference form (v_parent - v) / z is not used for predictions: on short
LV segments it turns a 1e-4 pu voltage error into a flow error of order 100%. Predictors:

  genco       a trained graphkit checkpoint (GencoModel)
  flat        Vm = 1 pu, Va = 0
  train_mean  per-bus mean Vm/Va over the training-day states
  lindistflow linearised power flow (no angles; loading from its own flows)
  fixed_point_1  one backward/forward sweep iteration from a flat start: the
              linear prior of GNS_heterogeneous_lv (linear_prior: true) on its own
  ac_sweep    our exact backward/forward sweep (should match datakit to ~1e-7)

Usage:
  python -m gencodoe.evaluate --data data_out/datakit_test --out results/test_baselines.json --baselines
  python -m gencodoe.evaluate --data data_out/datakit_test --run mlruns/<exp>/<run> \
      --config configs/genco_pf_enwl1_lv.yaml --out results/test_<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import feeder as fdr
from .enwl import load_enwl1
from .genco import NETWORK, read_raw

ROOT = Path(__file__).resolve().parents[2]
from .limits import VMAX_PU as VMAX, VMIN_PU as VMIN, record as limits_record  # noqa: E402


def load_truth(data_dir: Path, scenarios=None) -> dict:
    """Truth arrays from a datakit dataset dir ({data_dir}/enwl1_fluvius/raw)."""
    raw = Path(data_dir) / NETWORK / "raw"
    bus, gen, branch = read_raw(raw, scenarios)
    sc = np.sort(bus.scenario.unique())
    n_bus = bus[bus.scenario == sc[0]].shape[0]
    shape = (len(sc), n_bus)
    from .datakit_fluvius import solved_draws
    draws = solved_draws(raw / "fluvius_draws.parquet").set_index("scenario").loc[sc]
    return {
        "scenario": sc,
        "pd": bus.Pd.to_numpy().reshape(shape), "qd": bus.Qd.to_numpy().reshape(shape),
        "vm": bus.Vm.to_numpy().reshape(shape), "va": np.deg2rad(bus.Va.to_numpy().reshape(shape)),
        "frames": (bus, gen, branch), "draws": draws.reset_index(),
    }


def depth(feeder: fdr.Feeder) -> np.ndarray:
    d = np.zeros(feeder.n_bus, dtype=int)
    for k in feeder.order:
        if feeder.parent[k] >= 0:
            d[k] = d[feeder.parent[k]] + 1
    return d


def _stats(err: np.ndarray) -> dict:
    a = np.abs(err[np.isfinite(err)])
    if a.size == 0:
        return {}
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((a**2).mean())), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "n": int(a.size)}


def violations(vm: np.ndarray, loading: np.ndarray) -> np.ndarray:
    """(n,) True where any bus is outside [VMIN, VMAX] or any branch above 100%."""
    v = vm[:, 1:]
    return (v < VMIN).any(1) | (v > VMAX).any(1) | (loading[:, 1:] > 1.0).any(1)


def score(name: str, feeder: fdr.Feeder, truth: dict, vm: np.ndarray, va: np.ndarray | None,
          loading: np.ndarray, runtime_s: float | None = None) -> dict:
    """Metrics of one predictor against truth. Bus 0 (slack) and branch 0 (unused) are excluded."""
    tl, tv = _loading(feeder, truth, truth["vm"], truth["va"]), truth["vm"]
    dv = vm[:, 1:] - tv[:, 1:]
    dl = loading[:, 1:] - tl[:, 1:]
    dep = depth(feeder)[1:]
    net_import = truth["pd"].sum(1) > 0     # positive total Pd: the feeder imports
    out = {"name": name, "n_states": int(len(tv)),
           "vm_pu": _stats(dv),
           "va_deg": _stats(np.rad2deg(va[:, 1:] - truth["va"][:, 1:])) if va is not None else None,
           "loading": _stats(dl),
           "transformer_loading": _stats(loading[:, 1] - tl[:, 1]),
           "vm_pu_near_limits": _stats(dv[(tv[:, 1:] < VMIN + 0.01) | (tv[:, 1:] > VMAX - 0.01)]),
           "loading_near_limit": _stats(dl[tl[:, 1:] > 0.9]),
           "vm_pu_import_states": _stats(dv[net_import]),
           "vm_pu_export_states": _stats(dv[~net_import]),
           "vm_pu_by_depth": {f"{lo}-{hi}": _stats(dv[:, (dep >= lo) & (dep < hi)])
                              for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 40))},
           "runtime_s": runtime_s}
    tviol, pviol = violations(tv, tl), violations(vm, loading)
    tp, fp = int((tviol & pviol).sum()), int((~tviol & pviol).sum())
    fn, tn = int((tviol & ~pviol).sum()), int((~tviol & ~pviol).sum())
    out["violation"] = {"tp": tp, "fp": fp, "fn_false_safe": fn, "tn": tn,
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                        "false_safe_rate": fn / max(tp + fn, 1), "share_violating": float(tviol.mean())}
    return out


def _loading(feeder, truth, vm, va):
    p, q = -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva
    return fdr.branch_flows_kcl(feeder, vm, va, p, q).loading


def baselines(feeder: fdr.Feeder, truth: dict, train_truth: dict | None) -> list[dict]:
    p, q = -truth["pd"] / feeder.base_mva, -truth["qd"] / feeder.base_mva
    res = []
    vm = np.ones_like(truth["vm"]); vm[:, 0] = truth["vm"][:, 0]
    va = np.zeros_like(vm)
    res.append(score("flat", feeder, truth, vm, va, _loading(feeder, truth, vm, va)))
    if train_truth is not None:
        vm = np.broadcast_to(train_truth["vm"].mean(0), truth["vm"].shape).copy()
        va = np.broadcast_to(train_truth["va"].mean(0), truth["vm"].shape).copy()
        res.append(score("train_mean", feeder, truth, vm, va, _loading(feeder, truth, vm, va)))
    t0 = time.perf_counter()
    lin = fdr.lindistflow(feeder, p, q)
    res.append(score("lindistflow", feeder, truth, lin.vm, None, lin.loading, time.perf_counter() - t0))
    t0 = time.perf_counter()
    v1 = _sweep_voltage(feeder, p, q, max_iter=1)
    dt = time.perf_counter() - t0
    res.append(score("fixed_point_1", feeder, truth, np.abs(v1), np.angle(v1),
                     _loading(feeder, truth, np.abs(v1), np.angle(v1)), dt))
    t0 = time.perf_counter()
    ac = fdr.sweep_pf(feeder, p, q)
    dt = time.perf_counter() - t0
    va_ac = np.angle(_sweep_voltage(feeder, p, q))
    res.append(score("ac_sweep", feeder, truth, ac.vm, va_ac, ac.loading, dt))
    return res


def _sweep_voltage(feeder, p, q, tol=1e-10, max_iter=100):
    """Complex voltages of the backward/forward sweep (sweep_pf returns magnitudes).
    max_iter=1 gives the first fixed-point iterate from a flat start."""
    s = p + 1j * q
    v = np.full(s.shape, feeder.v_slack, dtype=complex)
    for _ in range(max_iter):
        i_load = -np.conj(s / v); i_load[:, 0] = 0.0
        j = i_load @ feeder.T
        v_new = feeder.v_slack - (j * feeder.z[None, :]) @ feeder.T.T
        if np.abs(v_new - v).max() < tol:
            return v_new
        v = v_new
    return v


def evaluate_genco(feeder, truth, run: Path, config: Path, device: str, batch_size: int,
                   checkpoint: str = "artifacts/model/best_model_state_dict.pt") -> dict:
    from .genco import GencoModel, frames_to_graphs

    model = GencoModel(config, run / checkpoint,
                       run / "artifacts/stats/normalizer_stats.pt", device=device)
    graphs = frames_to_graphs(*truth["frames"])
    model.predict_graphs(graphs[:batch_size], batch_size)          # warm-up
    t0 = time.perf_counter()
    vm, va = model.predict_graphs(graphs, batch_size)
    dt = time.perf_counter() - t0
    loading = _loading(feeder, truth, vm, va)
    out = score(f"genco:{run.name}", feeder, truth, vm, va, loading, dt)
    out["run"], out["config"], out["device"], out["checkpoint"] = str(run), str(config), device, checkpoint
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="datakit data dir (contains enwl1_fluvius/raw)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", type=Path, help="MLflow run dir with artifacts/model and artifacts/stats")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--train-data", type=Path, default=ROOT / "data_out" / "datakit",
                    help="training dataset, for the train_mean baseline (train-day states only)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default="artifacts/model/best_model_state_dict.pt",
                    help="path inside the run dir, e.g. checkpoints/last.ckpt")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-states", type=int)
    ap.add_argument("--scenario-file", type=Path, help="graphkit split .pt with the scenario ids to evaluate")
    a = ap.parse_args()

    feeder = load_enwl1().to_feeder()
    ids = None
    if a.scenario_file:
        import torch
        ids = np.sort(torch.load(a.scenario_file).numpy())
    if a.max_states is not None:
        ids = np.arange(a.max_states) if ids is None else ids[:a.max_states]
    truth = load_truth(a.data, ids)
    results = []
    if a.baselines:
        import torch
        tr_ids = torch.load(a.train_data / "splits" / "train.pt").numpy()[:2000]
        results += baselines(feeder, truth, load_truth(a.train_data, tr_ids))
    if a.run:
        results.append(evaluate_genco(feeder, truth, a.run, a.config, a.device, a.batch_size, a.checkpoint))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"voltage_limits": limits_record(), "data": str(a.data), "scenario_file": str(a.scenario_file) if a.scenario_file else None,
           "splits_in_data": sorted(set(truth["draws"]["split"])) if "split" in truth["draws"] else None,
           "results": results}
    a.out.write_text(json.dumps(doc, indent=1))
    rows = [{"name": r["name"], **{f"vm_{k}": v for k, v in r["vm_pu"].items() if k != "n"},
             "va_mae_deg": (r["va_deg"] or {}).get("mae"), "loading_mae": r["loading"].get("mae"),
             "loading_p99": r["loading"].get("p99"), "false_safe": r["violation"]["fn_false_safe"],
             "recall": r["violation"]["recall"], "runtime_s": r["runtime_s"]} for r in results]
    table = pd.DataFrame(rows)
    table.to_csv(a.out.with_suffix(".csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
