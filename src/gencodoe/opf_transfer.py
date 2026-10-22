"""Amendment 5 option 2 evaluation: pretrained-fine-tuned versus from-scratch GENCO OPF models at scarce labels, as
envelope proposals on case118_flex, each through the same exact check and scale-down (amendment 5a).

    python -m gencodoe.opf_transfer --raw data_out/case118_flex/raw --splits data_out/case118_flex/splits \
        --runs a5_finetune_N100=<run dir> a5_scratch_N100=<run dir> ... --n 1000 --out results/genco_opf_amendment5_transfer.json

Per model and test scenario: proposal = predicted flexible dispatch clipped to [0, cap]; exact check = the flexible units
fixed at lambda * proposal and the AC-OPF re-solved by IPOPT from the scenario's plain OPF solution (conventional units
re-dispatched; feasible iff it converges); scale-down = the largest lambda on the grid {1, 0.75, 0.5, 0.25} that is
feasible, else 0 (zero export can itself be infeasible here, amendment 5b); hosting after the check = lambda * sum(flexible
dispatch); reference = the IPOPT label dispatch. The raw full-solution feasibility (pf_violation: GENCO's own
conventional dispatch and set-points held) is reported alongside. Scenario bootstrap of the
fine-tuned minus scratch hosting share per N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

TOL, LEVELS, N_BOOT, SEED = 1e-4, 4, 2000, 0

JULIA_SCALE = r"""
# Exact check of a flexible-export proposal with the conventional units RE-DISPATCHED (amendment 5b): the flexible
# units are fixed at lambda * proposal (pmin = pmax) and the AC-OPF is re-solved by IPOPT, started from the scenario's
# plain OPF solution (flexible units free); feasible iff it converges. In these scenarios zero flexible export can be
# infeasible (the load needs the export), so feasibility is not monotone in lambda: lambda is the largest value on the
# grid GRID with a feasible re-dispatch, else 0.
const GRID = [1.0, 0.75, 0.5, 0.25]

function plain_start!(d, pd, qd)
    set_loads!(d, pd, qd)
    r0 = solve_ac_opf(d, OPT)
    ok = string(r0["termination_status"]) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
    if ok
        for (k, g) in r0["solution"]["gen"]; d["gen"][k]["pg_start"] = g["pg"]; d["gen"][k]["qg_start"] = g["qg"]; end
        for (k, b) in r0["solution"]["bus"]; d["bus"][k]["vm_start"] = b["vm"]; d["bus"][k]["va_start"] = b["va"]; end
    end
    return ok
end

function redispatch_ok(d, pd, qd, pflex, flex, lambda)
    set_loads!(d, pd, qd)
    base = d["baseMVA"]; saved = Dict{String,Tuple{Float64,Float64}}()
    for j in 1:length(d["gen"])
        flex[j] || continue
        g = d["gen"][string(j)]; saved[string(j)] = (g["pmin"], g["pmax"])
        v = lambda * pflex[j] / base; g["pmin"] = v; g["pmax"] = v; g["pg_start"] = v
    end
    ok = false
    try
        r = solve_ac_opf(d, OPT)
        ok = string(r["termination_status"]) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
    finally
        for (k, (lo, hi)) in saved; d["gen"][k]["pmin"] = lo; d["gen"][k]["pmax"] = hi; end
    end
    return ok
end

# lambda per scenario on GRID (else 0), evaluations, seconds, plain-start convergence, and the raw full-solution
# violation at lambda = 1 (pf_violation: GENCO's own conventional dispatch and set-points held).
function scale_batch(d0, dpf, PD, QD, PG, VG, flex, levels)
    n = size(PD, 1); lam = zeros(n); viol1 = zeros(n); evals = zeros(Int, n); secs = zeros(n); plain_ok = falses(n)
    for s in 1:n
        t = time()
        d = deepcopy(d0)
        viol1[s] = pf_violation(dpf, PD[s, :], QD[s, :], PG[s, :], VG[s, :], 1.0, flex)
        plain_ok[s] = plain_start!(d, PD[s, :], QD[s, :])
        for g in GRID[1:min(levels, length(GRID))]
            evals[s] += 1
            if redispatch_ok(d, PD[s, :], QD[s, :], PG[s, :], flex, g); lam[s] = g; break; end
        end
        secs[s] = time() - t
    end
    return lam, viol1, evals, secs, plain_ok
end
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate_models(raw: Path, case: Path, test_ids: np.ndarray, runs: dict[str, Path], config_of: dict[str, Path],
                    device: str = "cpu") -> dict:
    import juliacall  # noqa: F401
    import torch
    from .opf_envelopes import GencoOPF, PowerModelsBaseline, flex_mask, read_tables
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    t = read_tables(raw, scenarios=test_ids)
    gen0 = t["gen"][t["gen"].scenario == test_ids[0]].reset_index(drop=True)
    nb, ng, m = t["bus"].bus.nunique(), len(gen0), len(test_ids)
    PD, QD = t["bus"].Pd.to_numpy().reshape(m, nb), t["bus"].Qd.to_numpy().reshape(m, nb)
    PG_true, VM_true = t["gen"].p_mw.to_numpy().reshape(m, ng), t["bus"].Vm.to_numpy().reshape(m, nb)
    fl = flex_mask(gen0)
    cap = gen0.max_p_mw.to_numpy()[fl]
    ref = PG_true[:, fl].sum(1)
    pm = PowerModelsBaseline(case)
    pm.jl.seval(JULIA_SCALE)
    pm.set_flex(fl)
    scale_batch = pm.jl.seval("scale_batch")
    out = {"n_test": int(m), "n_flex": int(fl.sum()), "flex_cap_total_mw": float(cap.sum()), "reference_mwh_sum": float(ref.sum()),
           "models": {}, "per_scenario": {"reference_flex_mw": ref.tolist()}}
    # the reference itself through the check (labels are IPOPT solutions; reports how exact the check is)
    lam_r, v_r, ev_r, s_r, pok_r = (np.asarray(x) for x in scale_batch(pm.d, pm.d_pf, PD, QD, PG_true, VM_true, fl, LEVELS))
    out["per_scenario"]["reference_lambda"] = lam_r.tolist()
    out["reference_check"] = {"raw_feasible_share": float((v_r <= TOL).mean()), "lambda_mean": float(lam_r.mean()),
                              "hosting_after_check_share": float((lam_r * ref).sum() / ref.sum()),
                              "plain_start_converged_share": float(pok_r.mean()), "check_s_per_scenario": float(s_r.mean())}
    for name, run in runs.items():
        genco = GencoOPF(config_of[name], run, raw, network="case118_flex", device=device)
        genco(PD[:2], QD[:2])
        t0 = time.perf_counter()
        pred = genco(PD, QD)
        prop_s = (time.perf_counter() - t0) / m
        pg = pred["pg_mw"].copy()
        pg[:, fl] = np.clip(pg[:, fl], 0, cap)
        lam, v1, ev, secs, _ = (np.asarray(x) for x in scale_batch(pm.d, pm.d_pf, PD, QD, pg, pred["vm"], fl, LEVELS))
        host = lam * pg[:, fl].sum(1)
        out["models"][name] = {"run": str(run), "best_pt_sha256": sha(Path(run) / "artifacts/model/best_model_state_dict.pt"),
                               "hosting_share_of_reference": float(host.sum() / ref.sum()),
                               "raw_proposal_feasible_share": float((v1 <= TOL).mean()), "raw_proposal_share_of_reference": float(pg[:, fl].sum() / ref.sum()),
                               "lambda_mean": float(lam.mean()), "lambda_zero_share": float((lam == 0).mean()),
                               "proposal_s_per_scenario": prop_s, "check_s_per_scenario": float(secs.mean()), "check_evals_mean": float(ev.mean()),
                               "flex_unit_mae_mw": float(np.abs(pg[:, fl] - PG_true[:, fl]).mean())}
        out["per_scenario"][name] = {"hosting_mw": host.tolist(), "lambda": lam.tolist(), "worst_violation_raw": v1.tolist()}
        print(name, json.dumps({k: v for k, v in out["models"][name].items() if k not in ("run", "best_pt_sha256")}), flush=True)
    return out


def criteria(out: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    ref = np.array(out["per_scenario"]["reference_flex_mw"])
    c = {}
    for n in (100, 1000, 10000):
        a, b = f"a5_finetune_N{n}", f"a5_scratch_N{n}"
        if a not in out["models"] or b not in out["models"]:
            continue
        ha, hb = np.array(out["per_scenario"][a]["hosting_mw"]), np.array(out["per_scenario"][b]["hosting_mw"])
        idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
        diff = (ha[idx].sum(1) - hb[idx].sum(1)) / ref[idx].sum(1)
        c[str(n)] = {"finetuned_share": out["models"][a]["hosting_share_of_reference"], "scratch_share": out["models"][b]["hosting_share_of_reference"],
                     "diff_points": float(100 * (ha.sum() - hb.sum()) / ref.sum()),
                     "diff_lo": float(100 * np.percentile(diff, 2.5)), "diff_hi": float(100 * np.percentile(diff, 97.5))}
    if "1000" in c:
        k = c["1000"]
        c["pass"] = bool(k["diff_points"] >= 2.0 and k["diff_lo"] > 0 and k["finetuned_share"] >= 0.95)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--case", type=Path, default=Path("grids/case118_flex.m"))
    ap.add_argument("--splits", type=Path, required=True, help="splits dir; the test split of N100 is used (identical for every N)")
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir (names a5_{finetune,scratch}_N{N}); config from configs/amendment5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/genco_opf_amendment5_transfer.json"))
    a = ap.parse_args()
    import torch
    test = np.sort(torch.load(a.splits / "N100/test.pt").numpy())[: a.n]
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in a.runs)}
    cfgs = {k: Path("configs/amendment5") / (k.removeprefix("a5_") + ".yaml") for k in runs}
    out = evaluate_models(a.raw, a.case, test, runs, cfgs)
    out["test_ids_sha256"] = hashlib.sha256(test.tobytes()).hexdigest()
    out["criterion"] = criteria(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)
    print(json.dumps(out["criterion"], indent=1))


if __name__ == "__main__":
    main()
