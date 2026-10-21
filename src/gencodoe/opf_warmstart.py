"""Amendment 5, option 1: the official pretrained GENCO as a warm start for the exact AC-OPF on case118_flex.

    python -m gencodoe.opf_warmstart --raw <datakit raw dir> --case grids/case118_flex.m --hub <checkpoint dir> \
        --n 500 --out results/genco_opf_amendment5_warmstart.json

Per scenario, PowerModels/IPOPT AC-OPF from (i) a flat start, (ii) the previous scenario's solution, (iii) the pretrained
GENCO's dispatch and voltages. Reported: IPOPT iterations and wall-clock per solve (median, p90), convergence failures,
objective match to the flat-start optimum (1e-6 relative), and the preregistered criterion: (iii) cuts the median
wall-clock by >= 20% against (ii) with a scenario-bootstrap lower bound above 0. Hardware is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

JULIA = r"""
using PowerModels, Ipopt, Memento; import PowerModels: JuMP; const MOI5 = JuMP.MOI
Memento.setlevel!(Memento.getlogger(PowerModels), "error")
PowerModels.silence()
const OPT5 = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

function set_loads5!(d, pd, qd)
    for (k, l) in d["load"]
        b = l["load_bus"]
        l["pd"] = pd[b] / d["baseMVA"]; l["qd"] = qd[b] / d["baseMVA"]
    end
end

# One AC-OPF with optional start values; returns pg (MW), vm, va (rad), ok, iterations, seconds, objective.
function opf_one(d, pd, qd, pg0, vm0, va0, use_start)
    set_loads5!(d, pd, qd)
    ng, nb = length(d["gen"]), length(d["bus"])
    for j in 1:ng
        g = d["gen"][string(j)]
        if use_start; g["pg_start"] = pg0[j] / d["baseMVA"]; else; delete!(g, "pg_start"); end
    end
    for (k, b) in d["bus"]
        i = parse(Int, k)
        if use_start; b["vm_start"] = vm0[i]; b["va_start"] = va0[i]; else; delete!(b, "vm_start"); delete!(b, "va_start"); end
    end
    t = time()
    pm = instantiate_model(d, ACPPowerModel, PowerModels.build_opf)
    r = optimize_model!(pm, optimizer = OPT5)
    secs = time() - t
    it = try MOI5.get(pm.model, MOI5.BarrierIterations()) catch; -1 end
    ok = string(r["termination_status"]) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
    pg = fill(NaN, ng); vm = fill(NaN, nb); va = fill(NaN, nb)
    if ok
        for j in 1:ng; pg[j] = r["solution"]["gen"][string(j)]["pg"] * d["baseMVA"]; end
        for (k, b) in r["solution"]["bus"]; i = parse(Int, k); vm[i] = b["vm"]; va[i] = b["va"]; end
    end
    return pg, vm, va, ok, it, secs, ok ? r["objective"] : NaN
end
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(raw: Path, case: Path, hub: Path, n: int, seed: int = 0, n_boot: int = 2000, device: str = "cpu") -> dict:
    import juliacall  # noqa: F401  (before torch)
    from juliacall import Main as jl
    import torch
    from .amendment5_util import run_dir_from_hub
    from .opf_envelopes import GencoOPF, flex_mask, read_tables

    torch.set_num_threads(1)
    t = read_tables(raw)
    scen = np.sort(t["bus"].scenario.unique())[:n]
    t = read_tables(raw, scenarios=scen)
    gen0 = t["gen"][t["gen"].scenario == scen[0]].reset_index(drop=True)
    nb, ng, m = t["bus"].bus.nunique(), len(gen0), len(scen)
    PD, QD = t["bus"].Pd.to_numpy().reshape(m, nb), t["bus"].Qd.to_numpy().reshape(m, nb)
    fl = flex_mask(gen0)
    cfg = next(hub.glob("*.yaml"))
    genco = GencoOPF(cfg, run_dir_from_hub(hub, hub / "run"), raw, network="case118_ieee", device=device)
    genco(PD[:2], QD[:2])
    t0 = time.perf_counter()
    pred = genco(PD, QD)
    genco_s = (time.perf_counter() - t0) / m
    jl.seval(JULIA)
    d = jl.seval(f'PowerModels.parse_file("{case.resolve()}")')
    opf_one = jl.seval("opf_one")
    z = np.zeros(ng), np.ones(nb), np.zeros(nb)
    opf_one(d, PD[0], QD[0], *z, False)                                       # compile
    starts = ("flat", "previous", "genco")
    res = {s: {"ok": np.zeros(m, bool), "it": np.zeros(m, int), "secs": np.zeros(m), "obj": np.full(m, np.nan)} for s in starts}
    prev = None
    for s in range(m):
        for name in starts:
            if name == "flat":
                args = (*z, False)
            elif name == "previous":
                args = (*z, False) if prev is None else (*prev, True)
            else:
                args = (pred["pg_mw"][s], np.clip(pred["vm"][s], 0.9, 1.1), pred["va"][s], True)
            pg, vm, va, ok, it, secs, obj = opf_one(d, PD[s], QD[s], *args)
            r = res[name]
            r["ok"][s], r["it"][s], r["secs"][s], r["obj"][s] = bool(ok), int(it), float(secs), float(obj)
            if name == "flat" and ok:
                prev = (np.asarray(pg, float), np.asarray(vm, float), np.asarray(va, float))
        if s % 50 == 0:
            print(f"scenario {s}/{m}", flush=True)
    rng = np.random.default_rng(seed)
    both = res["previous"]["ok"] & res["genco"]["ok"]
    sp, sg = res["previous"]["secs"][both], res["genco"]["secs"][both]
    idx = rng.integers(0, both.sum(), size=(n_boot, both.sum()))
    cut = 1 - np.median(sg[idx], 1) / np.median(sp[idx], 1)
    out = {"n_scenarios": int(m), "n_gen": int(ng), "n_flex": int(fl.sum()), "case": str(case), "raw": str(raw),
           "checkpoint_files": {f.name: sha(f) for f in hub.glob("*.*")},
           "hardware": {"node": platform.node(), "cpu": platform.processor() or platform.machine(), "threads": 1,
                        "julia": str(jl.seval("string(VERSION)")), "powermodels": str(jl.seval('string(pkgversion(PowerModels))')),
                        "ipopt_jl": str(jl.seval('string(pkgversion(Ipopt))'))},
           "genco_inference_s_per_scenario_cpu": genco_s,
           "starts": {name: {"converged": int(r["ok"].sum()), "failures": int((~r["ok"]).sum()),
                             "iterations_median": float(np.median(r["it"][r["ok"]])), "iterations_p90": float(np.percentile(r["it"][r["ok"]], 90)),
                             "secs_median": float(np.median(r["secs"][r["ok"]])), "secs_p90": float(np.percentile(r["secs"][r["ok"]], 90)),
                             "objective_match_1e-6_share": float(np.mean(np.abs(r["obj"] / res["flat"]["obj"] - 1) <= 1e-6)) if name != "flat" else 1.0,
                             "objective_match_1e-4_share": float(np.mean(np.abs(r["obj"] / res["flat"]["obj"] - 1) <= 1e-4)) if name != "flat" else 1.0,
                             "objective_rel_diff_median": float(np.nanmedian(np.abs(r["obj"] / res["flat"]["obj"] - 1))) if name != "flat" else 0.0}
                      for name, r in res.items()},
           "criterion": {"median_secs_cut_genco_vs_previous": float(1 - np.median(sg) / np.median(sp)),
                         "bootstrap_lo": float(np.percentile(cut, 2.5)), "bootstrap_hi": float(np.percentile(cut, 97.5)),
                         "iterations_cut_genco_vs_previous": float(1 - np.median(res["genco"]["it"][both]) / np.median(res["previous"]["it"][both]))},
           "from_scratch_checkpoint": "not available on the branch (iv not run)",
           "per_scenario": {name: {k: (v.tolist() if k != "obj" else np.nan_to_num(v, nan=-1).tolist()) for k, v in r.items()} for name, r in res.items()}}
    c = out["criterion"]
    c["pass"] = bool(c["median_secs_cut_genco_vs_previous"] >= 0.20 and c["bootstrap_lo"] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--case", type=Path, default=Path("grids/case118_flex.m"))
    ap.add_argument("--hub", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("results/genco_opf_amendment5_warmstart.json"))
    a = ap.parse_args()
    out = run(a.raw, a.case, a.hub, a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_scenario"}, indent=1))


if __name__ == "__main__":
    main()
