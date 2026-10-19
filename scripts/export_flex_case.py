"""Add flexible export units to a pglib case, for OPF-based operating envelopes.

Every bus with load gets one flexible unit: a PQ generator at unity power factor
(Qmin = Qmax = 0) with Pmax = share x the bus's nominal load and a linear cost
of `cost` $/MWh (negative) plus a small quadratic term `c2` $/MW^2h. With a
marginal cost below every conventional generator, the AC-OPF dispatches as much
flexible export as the network allows (voltage, branch and conventional-generator
limits), so the flexible units' optimal dispatch is the max-hosting operating
envelope of that scenario. The quadratic term makes the split between units
unique (with identical linear costs many allocations have the same total, and
IPOPT returned allocations up to 326 MW apart per unit) and spreads export more
evenly; the marginal cost stays <= -89 $/MWh at the largest cap on case118.

The slack generator can also absorb power (Pmin = -total load), standing in for
the interconnection to the wider grid: without it, flexible export is capped by
the system's own load and the envelope is set by energy balance, not by the network.
At share 2.0 on case118, 39 of 99 units are then curtailed by voltage and branch limits.

    python scripts/export_flex_case.py case118_ieee grids/case118_flex.m
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from gridfm_datakit.network import get_pglib_file_path


def block(text: str, name: str) -> tuple[int, int, list[str]]:
    """(start, end) character span of mpc.<name> = [ ... ]; and its data rows."""
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    rows = [r.split("%")[0].strip().rstrip(";").strip() for r in m.group(1).strip().splitlines()]
    return m.start(1), m.end(1), [r for r in rows if r]


def pglib_source(case: str) -> Path:
    """The original pglib file (datakit downloads it next to its corrected copy)."""
    corrected = Path(get_pglib_file_path(case))
    original = corrected.with_name(f"pglib_opf_{case}.m")
    return original if original.exists() else corrected


def main(case: str, out: str, share: float = 2.0, cost: float = -100.0, c2: float = 0.01, absorb: bool = True):
    text = pglib_source(case).read_text()
    _, _, bus = block(text, "bus")
    total_pd = sum(float(r.split()[2]) for r in bus)
    ref = [int(float(r.split()[0])) for r in bus if int(float(r.split()[1])) == 3]
    if absorb:                                     # slack generator(s) may absorb up to the total load
        start, end, rows = block(text, "gen")
        lines = text[start:end].splitlines()
        for i, line in enumerate(lines):
            v = line.split("%")[0].strip().rstrip(";").split()
            if v and int(float(v[0])) in ref:
                v[9] = f"{-total_pd:.1f}"
                lines[i] = "\t" + "\t ".join(v) + "; % SLACK, absorbs (interconnection)"
        text = text[:start] + "\n".join(lines) + text[end:]
    flex_gen, flex_cost = [], []
    for r in bus:
        v = r.split()
        bus_i, pd = int(float(v[0])), float(v[2])
        if pd > 0:
            # bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin
            flex_gen.append(f"\t{bus_i}\t 0.0\t 0.0\t 0.0\t 0.0\t 1.0\t 100.0\t 1\t {share * pd:.4f}\t 0.0; % FLEX")
            flex_cost.append(f"\t2\t 0.0\t 0.0\t 3\t   {c2:.6f}\t {cost:.6f}\t   0.000000; % FLEX")
    for name, extra in (("gencost", flex_cost), ("gen", flex_gen)):   # gencost first: it comes later in the file
        start, end, _ = block(text, name)
        text = text[:end].rstrip() + "\n" + "\n".join(extra) + "\n" + text[end:]
    text = re.sub(r"function mpc = \w+", f"function mpc = {Path(out).stem}", text, count=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(flex_gen)} flexible units (Pmax = {share} x load, cost {cost} $/MWh + {c2} $/MW^2h)"
          + (f"; slack bus {ref} may absorb {total_pd:.0f} MW" if absorb else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case118_ieee", sys.argv[2] if len(sys.argv) > 2 else "grids/case118_flex.m")
