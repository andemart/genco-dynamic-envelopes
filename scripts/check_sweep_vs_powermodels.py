"""Check our AC backward/forward sweep against PowerModels on datakit's own scenarios.

    python scripts/check_sweep_vs_powermodels.py ~/gencodoe_work/data_out/case33bw/raw

datakit solves each scenario with PowerModels (AC PF after an OPF that sets the
slack voltage); we re-solve the same loads and slack voltage with feeder.sweep_pf.
"""

import sys

import numpy as np
import pandas as pd

from gencodoe import feeder


def main(raw: str, base_mva: float = 10.0):
    bus = pd.read_parquet(f"{raw}/bus_data.parquet").sort_values(["scenario", "bus"])
    br = pd.read_parquet(f"{raw}/branch_data.parquet").sort_values(["scenario", "idx"])
    t = br[br.scenario == 0]
    fd = feeder.Feeder.from_branches(t.from_bus.values, t.to_bus.values, t.r.values, t.x.values,
                                     t.rate_a.values / base_mva)
    n, nb = bus.scenario.nunique(), len(t) + 1
    P, Q = bus.Pd.values.reshape(n, nb) / base_mva, bus.Qd.values.reshape(n, nb) / base_mva
    vm, va = bus.Vm.values.reshape(n, nb), np.radians(bus.Va.values.reshape(n, nb))
    res = feeder.sweep_pf(fd, -P, -Q, v_slack=vm[:, 0])
    pf = br.pf.values.reshape(n, -1)
    print(f"{n} scenarios, converged: {res.converged.all()}")
    print(f"max |Vm| diff {np.abs(res.vm - vm).max():.2e} pu, max |Va| diff {np.abs(res.va - va).max():.2e} rad, "
          f"max |P_from| diff {np.abs(res.p_branch[:, t.to_bus.values] * base_mva - pf).max():.2e} MW")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_out/case33bw/raw")
