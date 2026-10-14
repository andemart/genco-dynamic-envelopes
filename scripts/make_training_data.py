"""Build GENCO power flow training data for envelope work on case33bw.

datakit's load generator only scales loads up, so its rows never contain the net
export (negative load) and heavy import that envelope search evaluates. This
script keeps datakit's exact parquet schema, using one datakit scenario as a
template, and fills it with envelope-like rows labelled by our AC sweep
(checked against PowerModels to ~1e-10 pu on datakit's own scenarios):

  * passive load and rooftop PV at a random hour, as in gencodoe.demo
  * flexible customers at a random proportional level, export or import
    (sometimes with per-customer jitter, for the max-total LP)
  * slack voltage at the feeder setpoint (1.0 pu): gridfm-graphkit 0.9 masks the
    slack Vm in PF mode, so the model cannot see a varying slack voltage

    python scripts/make_training_data.py ~/gencodoe_work/data_out/case33bw/raw ~/gencodoe_work/data_doe 40000
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from gencodoe import demo, feeder


def sample_rows(n: int, rng: np.random.Generator):
    fd, pd_nom, qd_nom, cust = demo.build(case33bw=True)       # base 1 MVA -> pu == MW
    p, q = demo.envelope_rows(fd, pd_nom, qd_nom, cust, n, rng)
    return fd, p, q, np.full(len(p), fd.v_slack)


def solve(fd, p, q, v_slack):
    res = feeder.sweep_pf(fd, p, q, v_slack=v_slack)
    v = res.vm * np.exp(1j * res.va)
    ok = res.converged & (res.vm.min(axis=1) > 0.5)
    return v[ok], p[ok], q[ok]


def write(template_dir: str, out_dir: str, fd, v, p, q):
    def first(name):
        df = pd.read_parquet(os.path.join(template_dir, name))
        return df[df.scenario == 0].drop(columns=["scenario_partition"], errors="ignore").reset_index(drop=True)

    bus_t, gen_t, br_t = first("bus_data.parquet"), first("gen_data.parquet"), first("branch_data.parquet")
    bus_t = bus_t.sort_values("bus").reset_index(drop=True)
    n, nb = v.shape
    f, t = br_t.from_bus.to_numpy(), br_t.to_bus.to_numpy()
    assert (fd.parent[t] == f).all(), "template branches must be oriented parent -> child"

    i_br = (v[:, f] - v[:, t]) / fd.z[t][None, :]
    s_from, s_to = v[:, f] * np.conj(i_br), v[:, t] * np.conj(-i_br)      # pu == MW on base 1
    s_slack = s_from[:, f == 0].sum(axis=1)
    scen = np.arange(n)

    bus = pd.concat([bus_t] * n, ignore_index=True)
    bus["scenario"] = np.repeat(scen, nb)
    bus["load_scenario_idx"] = bus["scenario"]
    bus["Pd"], bus["Qd"] = (-p).ravel(), (-q).ravel()
    pg = np.zeros((n, nb)); qg = np.zeros((n, nb))
    pg[:, 0], qg[:, 0] = s_slack.real, s_slack.imag
    bus["Pg"], bus["Qg"] = pg.ravel(), qg.ravel()
    bus["Vm"], bus["Va"] = np.abs(v).ravel(), np.degrees(np.angle(v)).ravel()

    gen = pd.concat([gen_t] * n, ignore_index=True)
    gen["scenario"] = scen
    gen["load_scenario_idx"] = scen
    gen["p_mw"], gen["q_mvar"] = s_slack.real, s_slack.imag

    br = pd.concat([br_t] * n, ignore_index=True)
    br["scenario"] = np.repeat(scen, len(br_t))
    br["load_scenario_idx"] = br["scenario"]
    br["pf"], br["qf"] = s_from.real.ravel(), s_from.imag.ravel()
    br["pt"], br["qt"] = s_to.real.ravel(), s_to.imag.ravel()

    raw = os.path.join(out_dir, "case33bw", "raw")
    os.makedirs(raw, exist_ok=True)
    bus.to_parquet(os.path.join(raw, "bus_data.parquet"), index=False)
    gen.to_parquet(os.path.join(raw, "gen_data.parquet"), index=False)
    br.to_parquet(os.path.join(raw, "branch_data.parquet"), index=False)
    return raw


def main(template_dir: str, out_dir: str, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    fd, p, q, vs = sample_rows(n, rng)
    v, p, q = solve(fd, p, q, vs)
    raw = write(template_dir, out_dir, fd, v, p, q)
    vm = np.abs(v[:, 1:])
    print(f"wrote {len(v)} rows to {raw}; Vm range [{vm.min():.3f}, {vm.max():.3f}], "
          f"share of rows with Vm > 1.10: {(vm.max(axis=1) > 1.10).mean():.2f}, "
          f"with Vm < 0.94: {(vm.min(axis=1) < 0.94).mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40000)
