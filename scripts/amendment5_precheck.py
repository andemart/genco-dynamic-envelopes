"""Amendment 5, step 0: the official pretrained GENCO OPF checkpoint (gridfm/genco-opf-datakit case118_ieee/base/seed0)
on (a) plain case118 datakit OPF scenarios and (b) one case118_flex scenario (99 extra generators).

    python scripts/amendment5_precheck.py --hub <dir with last.pt, normalizer_stats.pt, yaml> \
        --plain <datakit raw dir, case118_ieee> --flex <datakit raw dir, case118_flex> --out results/genco_opf_amendment5_precheck.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from gencodoe.opf_envelopes import GencoOPF, flex_mask, read_tables


def run_dir_from_hub(hub: Path, out: Path) -> Path:
    """graphkit's run layout from the Hub files; last.pt (a Lightning checkpoint) -> plain state dict."""
    (out / "artifacts/model").mkdir(parents=True, exist_ok=True)
    (out / "artifacts/stats").mkdir(parents=True, exist_ok=True)
    state = torch.load(hub / "last.pt", map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = {k: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    state = {k.removeprefix("_orig_mod.").removeprefix("model.").removeprefix("_orig_mod."): v for k, v in state.items()}
    torch.save(state, out / "artifacts/model/best_model_state_dict.pt")
    shutil.copy(hub / "normalizer_stats.pt", out / "artifacts/stats/normalizer_stats.pt")
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(genco: GencoOPF, raw: Path, n: int | None) -> dict:
    t = read_tables(raw)
    scen = np.sort(t["bus"].scenario.unique())[:n]
    t = read_tables(raw, scenarios=scen)
    gen0 = t["gen"][t["gen"].scenario == scen[0]].reset_index(drop=True)
    nb, ng, m = t["bus"].bus.nunique(), len(gen0), len(scen)
    PD, QD = t["bus"].Pd.to_numpy().reshape(m, nb), t["bus"].Qd.to_numpy().reshape(m, nb)
    PG, VM = t["gen"].p_mw.to_numpy().reshape(m, ng), t["bus"].Vm.to_numpy().reshape(m, nb)
    t0 = time.perf_counter()
    pred = genco(PD, QD)
    secs = (time.perf_counter() - t0) / m
    pg, vm = pred["pg_mw"], pred["vm"]
    fl = flex_mask(gen0)
    out = {"n_scenarios": int(m), "n_bus": int(nb), "n_gen": int(ng), "n_flex": int(fl.sum()),
           "finite": bool(np.isfinite(pg).all() and np.isfinite(vm).all()),
           "pg_rmse_mw": float(np.sqrt(np.mean((pg - PG) ** 2))), "pg_mae_mw": float(np.abs(pg - PG).mean()),
           "pg_rmse_mw_rel_to_mean_pg": float(np.sqrt(np.mean((pg - PG) ** 2)) / np.abs(PG).mean()),
           "vm_mae_pu": float(np.abs(vm - VM).mean()), "vm_p99_pu": float(np.percentile(np.abs(vm - VM), 99)),
           "total_pg_rel_err_mean_abs": float(np.abs(pg.sum(1) / PG.sum(1) - 1).mean()), "inference_s_per_scenario": secs}
    if fl.any():
        cap = gen0.max_p_mw.to_numpy()[fl]
        f_pred, f_true = pg[:, fl], PG[:, fl]
        out.update({"flex_raw_min": float(f_pred.min()), "flex_raw_max_over_cap": float((f_pred / cap).max()),
                    "flex_total_rel_err": (f_pred.clip(0, cap).sum(1) / f_true.sum(1) - 1).tolist(),
                    "flex_unit_mae_mw": float(np.abs(f_pred.clip(0, cap) - f_true).mean()), "flex_cap_total_mw": float(cap.sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", type=Path, required=True)
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--flex", type=Path, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/genco_opf_amendment5_precheck.json"))
    a = ap.parse_args()
    cfg = next(a.hub.glob("*.yaml"))
    run = run_dir_from_hub(a.hub, a.hub / "run")
    res = {"checkpoint": {"repo": "gridfm/genco-opf-datakit", "path": "case118_ieee/base/seed0", "files": {f.name: sha(f) for f in a.hub.glob("*.*")},
                          "config": cfg.name},
           "graphkit_main_note": "installed gridfm-graphkit 0.9.0.3 (main); the model card warns main may not reproduce the paper branch",
           "card_metrics": {"pg_rmse_mw": 17.1376, "optimality_gap_pct": 1.6389, "vm_source": "metrics.csv of the checkpoint"}}
    genco = GencoOPF(cfg, run, a.plain, network="case118_ieee", device="cpu")
    res["a_plain_case118"] = evaluate(genco, a.plain, a.n)
    print("(a)", json.dumps(res["a_plain_case118"], indent=1))
    if a.flex is not None:
        try:
            genco_flex = GencoOPF(cfg, run, a.flex, network="case118_ieee", device="cpu")
            res["b_case118_flex"] = evaluate(genco_flex, a.flex, None)
            res["b_case118_flex"]["pass"] = bool(res["b_case118_flex"]["finite"])
        except Exception as e:  # noqa: BLE001  (the pre-check's whole point is to record a failure)
            res["b_case118_flex"] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        print("(b)", json.dumps(res["b_case118_flex"], indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
