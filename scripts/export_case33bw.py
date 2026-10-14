"""Write case33bw as a plain MATPOWER .m file (per unit, no MATLAB post-processing)
that gridfm-datakit can read with network.source: file.

Thermal ratings (rate_a) are set to `headroom` x the branch |S| at nominal load,
the same rule as gencodoe.demo, because the original case carries none.

    python scripts/export_case33bw.py grids/case33bw.m
"""

import sys

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.converter.pypower import to_ppc

from gencodoe import feeder


def main(out: str, headroom: float = 1.5):
    net = pn.case33bw()
    pp.runpp(net)
    ppc = to_ppc(net, init="flat")
    base = float(ppc["baseMVA"])
    bus, branch, gen = ppc["bus"][:, :13].copy(), ppc["branch"][:, :13].real.copy(), ppc["gen"][:, :10].copy()
    branch = branch[branch[:, 10] > 0]                               # drop open tie lines

    fd, pd, qd = feeder.from_pandapower(net, base_mva=base)
    fd.rate = np.ones(fd.n_bus)
    s_peak = feeder.sweep_pf(fd, -pd[None], -qd[None]).loading[0]   # |S| in pu of base
    child = np.where(fd.parent[branch[:, 1].astype(int)] == branch[:, 0].astype(int),
                     branch[:, 1], branch[:, 0]).astype(int)
    branch[:, 5:8] = (headroom * s_peak[child] * base)[:, None]      # RATE_A/B/C in MVA
    bus[:, 11], bus[:, 12] = 1.10, 0.94                              # VMAX, VMIN
    bus[:, 0] += 1                                                   # MATPOWER is 1-based
    branch[:, :2] += 1
    gen[:, 0] += 1
    gencost = np.tile([2, 0, 0, 3, 0.0, 20.0, 0.0], (len(gen), 1))

    def block(name, arr, fmt="%.10g"):
        rows = "\n".join("\t" + "\t".join(fmt % v for v in r) + ";" for r in arr)
        return f"mpc.{name} = [\n{rows}\n];\n"

    with open(out, "w") as f:
        f.write("function mpc = case33bw\n% Baran & Wu 33-bus feeder, exported from pandapower in per unit\n")
        f.write("mpc.version = '2';\n")
        f.write(f"mpc.baseMVA = {base:g};\n")
        f.write(block("bus", bus))
        f.write(block("gen", gen))
        f.write(block("branch", branch))
        f.write(block("gencost", gencost))
    print(f"wrote {out}: {len(bus)} buses, {len(branch)} branches, baseMVA={base:g}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grids/case33bw.m")
