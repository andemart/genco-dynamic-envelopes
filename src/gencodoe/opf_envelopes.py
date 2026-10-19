"""OPF-based (max-hosting) operating envelopes with GENCO in OptimalPowerFlow mode.

Every load bus carries a flexible export unit (scripts/export_flex_case.py) with a
negative cost, so the AC-OPF dispatch of those units in a load scenario is the
largest export the network can host in that scenario: that scenario's envelope.

Three ways to get it:
  * IPOPT    PowerModels AC-OPF per scenario (exact, ~0.2 s on case118)
  * GENCO    batched OPF inference (GencoOPF), no feasibility guarantee
  * GENCO + AC check: GENCO's full solution (dispatch and voltage set-points) is
    checked with an exact AC power flow (PowerModels compute_ac_pf with reactive
    limits enforced; the slack absorbs the mismatch), and its violations reported
  * GENCO-warm-started IPOPT: exact envelopes, IPOPT started from GENCO's solution

Why no cheap repair: the OPF optimum sits on many binding limits (Vmax, branch
ratings, reactive limits) at once, so moving the flexible export in either
direction with the conventional dispatch and voltage set-points held fixed
violates some limit. Checked on case118_flex: exact solutions scaled by
0.95-1.15 all fail. The exact check of an envelope is an OPF (conventional units
re-dispatch), not a power flow.

Bus and generator order follow the datakit tables: bus i is MATPOWER bus row i,
generator j is MATPOWER gen row j (datakit's `idx`), which PowerModels keys as j + 1.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from gridfm_graphkit.datasets.globals import PD_H, PG_OUT_GEN, QD_H, VA_OUT, VM_OUT
from gridfm_graphkit.datasets.graph_builder import build_hetero_data
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer

from .genco import default_device


def read_tables(raw_dir: Path, scenarios=None) -> dict[str, pd.DataFrame]:
    """datakit bus/gen tables (sorted), optionally restricted to some scenarios."""
    out = {}
    for name in ("bus", "gen"):
        df = pd.read_parquet(Path(raw_dir) / f"{name}_data.parquet")
        if scenarios is not None:
            df = df[df.scenario.isin(scenarios)]
        out[name] = df.drop(columns=["scenario_partition"], errors="ignore").sort_values(
            ["scenario", "bus" if name == "bus" else "idx"]).reset_index(drop=True)
    return out


def flex_mask(gen0: pd.DataFrame) -> np.ndarray:
    """Flexible units: unity power factor (Qmin = Qmax = 0) with positive Pmax and negative cost."""
    return ((gen0.min_q_mvar == 0) & (gen0.max_q_mvar == 0) & (gen0.max_p_mw > 0)
            & (gen0.cp1_eur_per_mw < 0)).to_numpy()


class GencoOPF:
    """Batched GENCO OPF inference for one grid: loads (n, n_bus) MW/MVAr -> dispatch and voltages."""

    def __init__(self, config, run_dir, raw_dir, network: str = "case118_flex", device: str | None = None,
                 chunk: int = 1024):
        with open(config) as f:
            self.args = NestedNamespace(**yaml.safe_load(f))
        run_dir = Path(run_dir)
        self.device = device or default_device()
        self.chunk = chunk
        self.model = load_model(self.args)
        state = torch.load(run_dir / "artifacts/model/best_model_state_dict.pt", map_location="cpu")
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        self.model.eval().to(self.device)
        self.normalizer = load_normalizer(self.args)
        stats = torch.load(run_dir / "artifacts/stats/normalizer_stats.pt", map_location="cpu")
        self.normalizer.fit_from_dict(stats[network])

        t = read_tables(raw_dir, scenarios=[0])
        bus, gen = t["bus"], t["gen"]
        branch = pd.read_parquet(Path(raw_dir) / "branch_data.parquet")
        branch = branch[branch.scenario == 0].drop(columns=["scenario_partition"], errors="ignore").reset_index(drop=True)
        bus = bus.merge(gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index(), on="bus", how="left").fillna(0)
        if not gen.in_service.all():
            raise ValueError("template has out-of-service generators; gen order would not match")
        graph = build_hetero_data(bus, gen, branch)
        self.normalizer.transform(graph)
        self.template = get_task_transforms(self.args)(graph)
        self.n_bus, self.n_gen = len(bus), len(gen)
        self.flex = flex_mask(gen)
        self.gen_bus = gen.bus.to_numpy()
        self._batches: dict[int, Batch] = {}

    def _batch(self, k: int) -> Batch:
        if k not in self._batches:
            self._batches[k] = Batch.from_data_list([self.template] * k).to(self.device)
        return self._batches[k]

    @torch.inference_mode()
    def __call__(self, pd_mw: np.ndarray, qd_mvar: np.ndarray) -> dict[str, np.ndarray]:
        n = len(pd_mw)
        if n == 1:                       # the model squeezes per-graph tensors; pad a single row
            out = self(np.repeat(pd_mw, 2, 0), np.repeat(qd_mvar, 2, 0))
            return {k: v[:1] for k, v in out.items()}
        pg, vm, va = [], [], []
        for s in range(0, n, self.chunk):
            k = min(self.chunk, n - s)
            if k == 1:
                k, pad = 2, True
            else:
                pad = False
            rows = slice(s, s + k) if not pad else [s, s]
            batch = self._batch(k)
            x = batch["bus"].x.clone()
            x[:, PD_H] = torch.as_tensor(np.asarray(pd_mw[rows]).ravel() / self.normalizer.baseMVA, dtype=x.dtype, device=x.device)
            x[:, QD_H] = torch.as_tensor(np.asarray(qd_mvar[rows]).ravel() / self.normalizer.baseMVA, dtype=x.dtype, device=x.device)
            saved, batch["bus"].x = batch["bus"].x, x
            out = self.model(batch)
            batch["bus"].x = saved
            g = out["gen"][:, PG_OUT_GEN].float().cpu().numpy().reshape(k, self.n_gen) * self.normalizer.baseMVA
            b = out["bus"].float().cpu().numpy().reshape(k, self.n_bus, -1)
            take = 1 if pad else k
            pg.append(g[:take]), vm.append(b[:take, :, VM_OUT]), va.append(b[:take, :, VA_OUT])
        return {"pg_mw": np.vstack(pg), "vm": np.vstack(vm), "va": np.vstack(va)}


JULIA = r"""
using PowerModels, Ipopt, Memento
Memento.setlevel!(Memento.getlogger(PowerModels), "error")
PowerModels.silence()
const OPT = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# loads: one PowerModels load per bus with load (datakit Pd/Qd are per bus)
function set_loads!(d, pd, qd)
    for (k, l) in d["load"]
        b = l["load_bus"]
        l["pd"] = pd[b] / d["baseMVA"]; l["qd"] = qd[b] / d["baseMVA"]
    end
end

# AC-OPF per scenario. With start values (MW, pu, rad) IPOPT starts from them (PowerModels "*_start").
function opf_batch(d0, PD, QD; PG0 = nothing, VM0 = nothing, VA0 = nothing)   # PD, QD: (n, n_bus) MW / MVAr
    n, ng = size(PD, 1), length(d0["gen"])
    PG = fill(NaN, n, ng); ok = falses(n); secs = zeros(n)
    d = deepcopy(d0)
    for s in 1:n
        set_loads!(d, PD[s, :], QD[s, :])
        if PG0 !== nothing
            for j in 1:ng; d["gen"][string(j)]["pg_start"] = PG0[s, j] / d["baseMVA"]; end
            for (k, b) in d["bus"]
                b["vm_start"] = VM0[s, parse(Int, k)]; b["va_start"] = VA0[s, parse(Int, k)]
            end
        end
        t = time(); r = solve_ac_opf(d, OPT); secs[s] = time() - t
        ok[s] = string(r["termination_status"]) in ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
        if ok[s]
            for j in 1:ng; PG[s, j] = r["solution"]["gen"][string(j)]["pg"] * d["baseMVA"]; end
        end
    end
    return PG, ok, secs
end

# Power-flow copy of the case: flexible units (fixed Q = 0) are switched off and enter as negative
# load at their bus (compute_ac_pf allows active generators only at PV and slack buses), and buses
# PowerModels had made PV only because of a flexible unit go back to PQ.
function pq_flex_buses!(d, flex)
    conv = Set(d["gen"][string(j)]["gen_bus"] for j in 1:length(d["gen"]) if !flex[j])
    for j in 1:length(d["gen"])
        g = d["gen"][string(j)]
        flex[j] || continue
        g["gen_status"] = 0
        b = d["bus"][string(g["gen_bus"])]
        (!(g["gen_bus"] in conv) && b["bus_type"] == 2) && (b["bus_type"] = 1)
    end
end

# AC power flow with given dispatch (MW, all gens) and gen-bus voltage set-points; the slack absorbs.
# Reactive limits are enforced as in MATPOWER's enforce_q_lims: a PV bus whose generators exceed
# their Q range is switched to PQ with Q at the limit (its units enter as negative load, because
# compute_ac_pf allows active generators only at PV/slack buses) and the flow is re-solved.
# Returns the worst normalised violation (<= 0 means feasible) over bus voltages, branch ratings and
# the interconnection's absorb limit (its import side is unlimited: it stands for the wider grid).
# Units: 0.05 = 0.05 pu of voltage = 5% overload. All changes to `d` are undone before returning.
function pf_violation(d, pd, qd, pg, vg, flex_scale, flex; qiters = 8, qtol = 1e-4)
    base = d["baseMVA"]
    net, netq = copy(pd), copy(qd)
    for j in 1:length(d["gen"])
        g = d["gen"][string(j)]
        if flex[j]
            net[g["gen_bus"]] -= flex_scale * pg[j]                 # flexible export as negative load
        else
            g["pg"] = pg[j] / base
        end
    end
    bus_gens = Dict{String,Vector{String}}()
    for (k, g) in d["gen"]
        (g["gen_status"] != 0) && push!(get!(bus_gens, string(g["gen_bus"]), String[]), k)
    end
    switched_bus = String[]; switched_gen = String[]
    try
        for it in 1:qiters
            set_loads!(d, net, netq)
            for (k, b) in d["bus"]                      # PV/slack set-points live in bus "vm"; PQ start flat
                b["vm"] = b["bus_type"] in (2, 3) ? vg[parse(Int, k)] : 1.0
                b["va"] = 0.0
            end
            r = compute_ac_pf(d)
            r["termination_status"] || return Inf
            sol = r["solution"]
            hit = false
            for (bk, gens) in bus_gens
                d["bus"][bk]["bus_type"] == 2 || continue
                qtot = sum(sol["gen"][k]["qg"] for k in gens)
                qmin = sum(d["gen"][k]["qmin"] for k in gens); qmax = sum(d["gen"][k]["qmax"] for k in gens)
                if qtot > qmax + qtol || qtot < qmin - qtol
                    hit = true
                    b = parse(Int, bk)
                    d["bus"][bk]["bus_type"] = 1; push!(switched_bus, bk)
                    for k in gens
                        d["gen"][k]["gen_status"] = 0; push!(switched_gen, k)
                        net[b] -= d["gen"][k]["pg"] * base
                    end
                    netq[b] -= clamp(qtot, qmin, qmax) * base
                end
            end
            hit && continue
            update_data!(d, sol)
            flows = calc_branch_flow_ac(d)["branch"]
            worst = -Inf
            for (k, bb) in sol["bus"]
                bd = d["bus"][k]
                worst = max(worst, bb["vm"] - bd["vmax"], bd["vmin"] - bb["vm"])
            end
            for (k, br) in flows
                rate = d["branch"][k]["rate_a"]
                rate > 0 || continue
                sf = max(abs(complex(br["pf"], br["qf"])), abs(complex(br["pt"], br["qt"])))
                worst = max(worst, (sf - rate) / rate * 0.05)      # 5% overload counts like 0.05 pu of voltage
            end
            for (k, g) in sol["gen"]
                gd = d["gen"][k]
                d["bus"][string(gd["gen_bus"])]["bus_type"] == 3 && (worst = max(worst, (gd["pmin"] - g["pg"]) / 10))
            end
            return worst
        end
        return Inf
    finally
        for bk in switched_bus; d["bus"][bk]["bus_type"] = 2; end
        for k in switched_gen; d["gen"][k]["gen_status"] = 1; end
    end
end

# Worst AC violation of each scenario's full solution (dispatch + voltage set-points), see pf_violation.
function check_batch(d, PD, QD, PG, VG, flex)
    n = size(PD, 1); viol = zeros(n); secs = zeros(n)
    for s in 1:n
        t = time(); viol[s] = pf_violation(d, PD[s, :], QD[s, :], PG[s, :], VG[s, :], 1.0, flex); secs[s] = time() - t
    end
    return viol, secs
end
"""


class PowerModelsBaseline:
    """IPOPT AC-OPF and exact AC power-flow checks through juliacall (one Julia session)."""

    def __init__(self, case_file: str | Path):
        from juliacall import Main as jl
        self.jl = jl
        jl.seval(JULIA)
        self.d = jl.seval(f'PowerModels.parse_file("{Path(case_file).resolve()}")')
        self.n_gen = int(jl.seval("d -> length(d[\"gen\"])")(self.d))
        self.d_pf = None

    def set_flex(self, flex):
        """Power-flow copy of the case with flexible-only buses as PQ buses."""
        self.d_pf = self.jl.deepcopy(self.d)
        self.jl.seval("pq_flex_buses!")(self.d_pf, np.asarray(flex, bool))

    def opf(self, pd_mw, qd_mvar, start: dict | None = None):
        """IPOPT AC-OPF per row; `start` = GENCO output (pg_mw, vm, va[rad]) for a warm start."""
        kw = {} if start is None else {"PG0": np.asarray(start["pg_mw"], float), "VM0": np.asarray(start["vm"], float),
                                       "VA0": np.asarray(start["va"], float)}
        pg, ok, secs = self.jl.opf_batch(self.d, np.asarray(pd_mw, float), np.asarray(qd_mvar, float), **kw)
        return np.asarray(pg), np.asarray(ok, bool), np.asarray(secs)

    def check(self, pd_mw, qd_mvar, pg_mw, vm_bus, flex):
        """Worst AC violation per row of a full OPF solution (<= 0: feasible), and seconds per row."""
        if self.d_pf is None:
            self.set_flex(flex)
        viol, secs = self.jl.check_batch(self.d_pf, np.asarray(pd_mw, float), np.asarray(qd_mvar, float),
                                         np.asarray(pg_mw, float), np.asarray(vm_bus, float), np.asarray(flex, bool))
        return np.asarray(viol), np.asarray(secs)


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t
