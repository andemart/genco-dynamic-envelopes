"""Unbalanced reality check: apply balanced-model envelopes to the four-wire OpenDSS ENWL1.

The balanced model lumps each single-phase customer into a three-phase bus load.
Here every customer sits on its own phase (Loads.dss), with the neutral
modelled as a conductor and grounded at the transformer only, and we count
what the balanced envelopes miss:
  * customer phase-to-neutral voltage outside [0.94, 1.10] x 240.2 V (the
    balanced model's 1 pu);
  * line phase current above the linecode NormAmps;
  * transformer loading above its 800 kVA rating.

The ENWL1 OpenDSS files reference load-shape CSVs on another machine; a
snapshot study does not need them, so the circuit is built without them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .enwl import ENWL1_DIR, TR_KVA

# Same per-unit base as the balanced model: 1 pu = 0.416 kV line-to-line, i.e.
# 240.2 V phase-to-neutral (the transformer's no-load secondary voltage). The
# 0.94 / 1.10 limits therefore mean the same volts in both models.
V_PN = 416.0 / np.sqrt(3)


class UnbalancedENWL1:
    def __init__(self, folder: Path = ENWL1_DIR):
        import opendssdirect as dss

        self.dss = dss
        master = (folder / "Master.dss").read_text().splitlines()
        dss.Text.Command("clear")
        for line in master:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            low = s.lower()
            if low.startswith("redirect"):
                name = s.split(None, 1)[1].strip()
                if name.lower().startswith(("loadshapes", "loads")):
                    continue
                self._run_file(folder / name)
            elif low.startswith("buscoords") or low.startswith("new energymeter"):
                continue
            else:
                dss.Text.Command(s)
        self.load_names, self.load_bus = [], []
        for line in (folder / "Loads.dss").read_text().splitlines():
            if line.lower().startswith("new load"):
                dss.Text.Command(re.sub(r"\s+Daily=\S+", "", line))
                m = re.search(r"New Load\.(\w+).*?Bus1=(\S+)", line, re.I)
                self.load_names.append(m.group(1))
                self.load_bus.append(m.group(2).lower())
        dss.Text.Command("Set VoltageBases=[11, 0.416]")
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("Set mode=snapshot")
        dss.Text.Command("Set maxiterations=100")
        # customer bus/phase -> node names for phase and neutral voltage
        self.nodes = [n.lower() for n in dss.Circuit.AllNodeNames()]
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.cust_phase, self.cust_neutral = [], []
        for b in self.load_bus:
            bus, phase, neutral = b.split(".")[:3]
            self.cust_phase.append(self.node_idx[f"{bus}.{phase}"])
            self.cust_neutral.append(self.node_idx[f"{bus}.{neutral}"])
        self.cust_phase, self.cust_neutral = np.array(self.cust_phase), np.array(self.cust_neutral)
        self.line_amps = {}
        dss.Lines.First()
        while True:
            self.line_amps[dss.Lines.Name()] = dss.Lines.NormAmps()
            if not dss.Lines.Next():
                break

    def _run_file(self, path: Path):
        for line in path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("!"):
                self.dss.Text.Command(s)

    def solve(self, p_kw: np.ndarray, q_kvar: np.ndarray) -> dict:
        """Snapshot with customer net consumption p_kw/q_kvar (200,), negative = export."""
        dss = self.dss
        for name, p, q in zip(self.load_names, p_kw, q_kvar):
            dss.Loads.Name(name)
            dss.Loads.kW(float(p))
            dss.Loads.kvar(float(q))
        dss.Solution.Solve()
        v = np.array(dss.Circuit.AllBusVolts())
        v = v[0::2] + 1j * v[1::2]
        v_pn = np.abs(v[self.cust_phase] - v[self.cust_neutral]) / V_PN
        worst_line = 0.0
        dss.Lines.First()
        while True:
            amps = self.line_amps[dss.Lines.Name()]
            cur = np.array(dss.CktElement.CurrentsMagAng()[0:6:2])        # phases a, b, c at terminal 1
            if amps > 0:
                worst_line = max(worst_line, float(cur.max() / amps))
            if not dss.Lines.Next():
                break
        dss.Transformers.Name("TR1")
        s_tr = np.array(dss.CktElement.Powers()[:6]).reshape(3, 2)
        tr_loading = float(np.abs(s_tr[:, 0] + 1j * s_tr[:, 1]).sum() / TR_KVA)
        return {"converged": bool(dss.Solution.Converged()), "v_pn_min": float(v_pn.min()),
                "v_pn_max": float(v_pn.max()), "line_loading_max": worst_line, "transformer_loading": tr_loading,
                "undervoltage": bool(v_pn.min() < 0.94), "overvoltage": bool(v_pn.max() > 1.10),
                "line_overload": worst_line > 1.0, "transformer_overload": tr_loading > 1.0}
