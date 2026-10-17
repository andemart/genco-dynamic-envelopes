"""Unbalanced reality check: apply balanced-model envelopes to the four-wire OpenDSS ENWL1.

The balanced model lumps each single-phase customer into a three-phase bus load.
Here every customer sits on its own phase (Loads.dss), with the neutral
modelled as a conductor and grounded at the transformer only, and we count
what the balanced envelopes miss:
  * customer phase-to-neutral voltage outside the limits of gencodoe.limits
    (absolute volts, on the balanced model's 1 pu = 240.18 V base);
  * line phase current above the linecode NormAmps;
  * transformer loading above its 800 kVA rating.

The ENWL1 OpenDSS files reference load-shape CSVs on another machine; a
snapshot study does not need them, so the circuit is built without them.

Loads are constant power over 0.5-1.5 pu (Vminpu/Vmaxpu), as in the balanced
model and datakit. OpenDSS's default switches a load to constant impedance
below 0.95 pu and above 1.05 pu, which understates exactly the under- and
overvoltage this check is for. A solve that does not converge is retried once
from the no-load solution (OpenDSS starts from the previous solution).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .enwl import ENWL1_DIR, TR_KVA

# Same per-unit base as the balanced model: 1 pu = 0.416 kV line-to-line, i.e.
# 240.2 V phase-to-neutral (the transformer's no-load secondary voltage). The
# limits therefore mean the same volts in both models.
from .limits import V_BASE as V_PN, VMAX_PU, VMIN_PU  # noqa: E402


def voltage_flags(v_pn: np.ndarray) -> tuple[bool, bool]:
    """(undervoltage, overvoltage) of customer phase-neutral voltages in pu of V_PN."""
    return bool(np.min(v_pn) < VMIN_PU), bool(np.max(v_pn) > VMAX_PU)


class OpenDSSEngineError(RuntimeError):
    """The OpenDSS engine is unhealthy: a no-load (priming or recovery) solve did not converge."""


class UnbalancedENWL1:
    """four-wire ENWL1. three_phase: optional bool mask over customers (Loads.dss order); those
    customers are connected three-phase wye (balanced, same total kW/kvar) and the worst of their
    three phase-neutral voltages counts -- the W2 idealised multi-phase sensitivity."""

    def __init__(self, folder: Path = ENWL1_DIR, three_phase: np.ndarray | None = None):
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
                dss.Text.Command(re.sub(r"\s+Daily=\S+", "", line) + " Vminpu=0.5 Vmaxpu=1.5")
                m = re.search(r"New Load\.(\w+).*?Bus1=(\S+)", line, re.I)
                self.load_names.append(m.group(1))
                self.load_bus.append(m.group(2).lower())
        self.three_phase = (np.zeros(len(self.load_names), bool) if three_phase is None
                            else np.asarray(three_phase, bool))
        if len(self.three_phase) != len(self.load_names):
            raise ValueError("three_phase mask must have one entry per customer")
        for name, b, tp in zip(self.load_names, self.load_bus, self.three_phase):
            if tp:
                dss.Text.Command(f"Edit Load.{name} Phases=3 Bus1={b.split('.')[0]}.1.2.3.4 kV=0.416 Conn=wye "
                                 "Vminpu=0.5 Vmaxpu=1.5")
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
        # monitored (phase, neutral) node pairs: one per single-phase customer, three per three-phase one
        vp, vn = [], []
        for b, tp in zip(self.load_bus, self.three_phase):
            bus, phase, neutral = b.split(".")[:3]
            for ph in ((1, 2, 3) if tp else (int(phase),)):
                vp.append(self.node_idx[f"{bus}.{ph}"])
                vn.append(self.node_idx[f"{bus}.{neutral}"])
        self.mon_phase, self.mon_neutral = np.array(vp), np.array(vn)
        self.line_amps = {}
        dss.Lines.First()
        while True:
            self.line_amps[dss.Lines.Name()] = dss.Lines.NormAmps()
            if not dss.Lines.Next():
                break
        # prime the engine with the no-load solution: if the very first solve of a fresh engine diverges
        # (e.g. every customer at the 5 kW export cap), OpenDSS never recovers, not even through a reset,
        # and every later solve fails (envelope-prediction labels; docs/envpred_protocol.md amendment 3c)
        self.n_resets = self.n_nonconverged = 0
        self._reset()

    def health(self) -> dict:
        return {"resets": self.n_resets, "nonconverged_returned": self.n_nonconverged, "reset_failures": 0}

    def _reset(self):
        """Solve the no-load case so the next solve starts from a flat, feasible point. Raises
        OpenDSSEngineError if it does not converge: an unhealthy engine must never continue silently."""
        for name in self.load_names:
            self.dss.Loads.Name(name)
            self.dss.Loads.kW(0.0)
            self.dss.Loads.kvar(0.0)
        self.dss.Solution.Solve()
        self.n_resets += 1
        if not self.dss.Solution.Converged():
            raise OpenDSSEngineError("OpenDSS no-load solve did not converge: engine unhealthy")

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
        if not dss.Solution.Converged():
            self._reset()                                    # checked: raises if the engine is unhealthy
            for name, p, q in zip(self.load_names, p_kw, q_kvar):
                dss.Loads.Name(name)
                dss.Loads.kW(float(p))
                dss.Loads.kvar(float(q))
            dss.Solution.Solve()
            if not dss.Solution.Converged():                 # genuinely non-convergent after a healthy reset
                self.n_nonconverged += 1
        v = np.array(dss.Circuit.AllBusVolts())
        v = v[0::2] + 1j * v[1::2]
        v_pn = np.abs(v[self.mon_phase] - v[self.mon_neutral]) / V_PN
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
                "undervoltage": voltage_flags(v_pn)[0], "overvoltage": voltage_flags(v_pn)[1],
                "line_overload": worst_line > 1.0, "transformer_overload": tr_loading > 1.0}


def diagnose(u: "UnbalancedENWL1", p_kw: np.ndarray, q_kvar: np.ndarray) -> dict:
    """Detailed four-wire snapshot: per-customer voltage, worst line (name, phase, loading),
    neutral currents, and per-phase transformer current/power. Slower than solve()."""
    dss = u.dss
    r = u.solve(p_kw, q_kvar)
    v = np.array(dss.Circuit.AllBusVolts())
    v = v[0::2] + 1j * v[1::2]
    v_pn = np.abs(v[u.cust_phase] - v[u.cust_neutral]) / V_PN
    phase = np.array([int(b.split(".")[1]) for b in u.load_bus])
    worst = (0.0, "", 0)
    neutral_ratio = 0.0
    overloaded = []
    dss.Lines.First()
    while True:
        name, amps = dss.Lines.Name(), u.line_amps[dss.Lines.Name()]
        mags = np.array(dss.CktElement.CurrentsMagAng()[0:8:2])     # a, b, c, n at terminal 1
        if amps > 0:
            ld = mags[:3] / amps
            k = int(ld.argmax())
            if ld[k] > worst[0]:
                worst = (float(ld[k]), name, k + 1)
            if ld[k] > 1.0:
                overloaded.append(name)
            neutral_ratio = max(neutral_ratio, float(mags[3] / amps))
        if not dss.Lines.Next():
            break
    dss.Transformers.Name("TR1")
    cur = np.array(dss.CktElement.CurrentsMagAng())[::2]
    n_cond = dss.CktElement.NumConductors()
    lv = cur[n_cond:2 * n_cond]                                       # terminal 2 (LV): a, b, c, n
    pw = np.array(dss.CktElement.Powers())
    lv_pw = pw[2 * n_cond:2 * n_cond + 6].reshape(3, 2)               # kW, kvar per LV phase (into the transformer)
    return {**r, "v_pn": v_pn, "customer_phase": phase,
            "undervoltage_customers_by_phase": {int(ph): int(((v_pn < VMIN_PU) & (phase == ph)).sum()) for ph in (1, 2, 3)},
            "worst_line": worst[1], "worst_line_phase": worst[2], "worst_line_loading": worst[0],
            "n_overloaded_lines": len(overloaded), "neutral_over_rating_max": neutral_ratio,
            "tr_lv_phase_current_a": lv[:3].tolist(), "tr_lv_neutral_current_a": float(lv[3]) if len(lv) > 3 else float("nan"),
            "tr_lv_phase_kw": (-lv_pw[:, 0]).tolist()}


def four_wire_truth(data_dir: Path, out: Path | None = None) -> "pd.DataFrame":
    """Four-wire OpenDSS summary for every state of a datakit dataset.

    Each customer has its own bus in the reduced network, so the bus Pd/Qd of the
    datakit state is that customer's net consumption (MW/MVAr, + = consumption).
    Columns: scenario, v_pn_min, v_pn_max, line_loading_max, transformer_loading,
    converged."""
    import pandas as pd

    from .enwl import load_enwl1
    from .evaluate import load_truth

    net = load_enwl1()
    truth = load_truth(data_dir)
    u = UnbalancedENWL1()
    rows = []
    for i, sc in enumerate(truth["scenario"]):
        r = u.solve(truth["pd"][i, net.customer_bus] * 1e3, truth["qd"][i, net.customer_bus] * 1e3)
        rows.append({"scenario": int(sc), **{k: r[k] for k in ("v_pn_min", "v_pn_max", "line_loading_max",
                                                               "transformer_loading", "converged")}})
    df = pd.DataFrame(rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="four-wire OpenDSS truth for datakit datasets")
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    for d in a.data:
        df = four_wire_truth(d, d / "enwl1_fluvius" / "four_wire_truth.parquet")
        print(d, len(df), "states, converged", df.converged.mean())
