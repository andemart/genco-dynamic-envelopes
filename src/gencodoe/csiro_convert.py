"""CSIRO four-wire ENWL networks -> the ENWL1 four-wire layout (docs/fourwire_scale_protocol.md §2.2, §2.4).

Each network folder (Feeder_*/LineCode.txt, Lines.txt, Loads.txt) becomes one folder with
LineCode_3f4w.dss, Lines.dss, Loads.dss and Master.dss, readable by `fourwire_net.load_fourwire`
and `unbalanced.UnbalancedENWL1`:

- source and transformer: ENWL1's Master.dss (energy meters, bus coordinates and load shapes dropped);
- feeder bus `1` -> `TRbus`, other bus b of feeder k -> `k_b`; the stub LINE0 (sourcebus -> 1) is dropped;
- line codes merged by original cable name (CSIRO's LC1..LC9 are the same cable in every feeder and
  match ENWL1's lc1..lc9); NormAmps from the frozen table RATINGS;
- single-phase loads kept (same phase, neutral return); three-phase loads split into three
  single-phase phase-to-neutral loads named `<load>_3p1/_3p2/_3p3`, one customer (see `customers`).
"""
import re
from pathlib import Path

import numpy as np

CSIRO_DIR = Path(__file__).resolve().parents[2] / "data/feeder_topology/ENWL_CSIRO/Four-wire"
ENWL1_DIR = Path(__file__).resolve().parents[2] / "data/feeder_topology/ENWL1"

# §2.4, frozen. Keys are the dataset's original cable names.
RATINGS = {
    "BS 5467 1997 table 10 16 (50 Hz)": 110,
    "BS 5467 1997 table 10 25 (50 Hz)": 141,
    "BS 5467 1997 table 10 35 (50 Hz)": 174,
    "BS 5467 1997 table 10 70 (50 Hz)": 269,
    "BS 6480 1988 table 17 Cu 16 (50 Hz)": 101,
    "BS 6480 1988 table 17 Cu 120 (50 Hz)": 280,
    "Waveform 4-core 95 IEC 60287 (50 Hz)": 235,
    "Waveform 4-core 185 IEC 60287 (50 Hz)": 335,
    "Waveform 4-core 300 IEC 60287 (50 Hz)": 435,
}
# §2.4 rating sensitivities (frozen; no criterion uses them)
RATING_VARIANTS = {
    "primary": RATINGS,
    "cu120_240": {**RATINGS, "BS 6480 1988 table 17 Cu 120 (50 Hz)": 240},
    "cu16_89": {**RATINGS, "BS 6480 1988 table 17 Cu 16 (50 Hz)": 89},
}
THREE_PHASE_SUFFIX = "_3p"

_CODE_RE = re.compile(r"original name: (.*?)\n\s*New LineCode\.(\w+).*?Rmatrix=\(([^)]*)\).*?Xmatrix=\(([^)]*)\)",
                      re.S | re.I)
_LINE_RE = re.compile(r"New Line\.(\w+) Bus1=(\w+)\.1\.2\.3\.4 Bus2=(\w+)\.1\.2\.3\.4 phases=4 Linecode=(\w+) "
                      r"(Length=\S+ Units=\w+)\s*$", re.I)
_LOAD1_RE = re.compile(r"New Load\.(\w+) Phases=1 Bus1=(\w+)\.([123])\.4 kV=0\.23 kW=\S+\s+PF=0\.95\s*$", re.I)
_LOAD3_RE = re.compile(r"New Load\.(\w+) Phases=3 Bus1=(\w+)\.1\.2\.3\.4 kV=\S+ kW=\S+\s+PF=0\.95\s*$", re.I)


def network_dirs(src: Path = CSIRO_DIR) -> dict[int, Path]:
    """network number -> folder (the dataset mixes `network_N` and `Network_N`)."""
    return {int(p.name.split("_")[1]): p for p in src.iterdir() if p.name.lower().startswith("network_")}


def _feeders(net_dir: Path) -> list[tuple[int, Path]]:
    return sorted(((int(p.name.split("_")[1]), p) for p in net_dir.glob("Feeder_*")), key=lambda t: t[0])


def convert(net_dir: Path, out: Path, variant: str = "primary") -> dict:
    """Write the ENWL1-layout files for one CSIRO network into `out`. Returns counts."""
    ratings = RATING_VARIANTS[variant]
    out.mkdir(parents=True, exist_ok=True)
    codes: dict[str, tuple[str, str, str]] = {}      # original name -> (code name, R text, X text)
    lines, loads, n_cust, n_three = [], [], 0, 0
    for k, fd in _feeders(net_dir):
        for m in _CODE_RE.finditer((fd / "LineCode.txt").read_text()):
            orig, name, r, x = m.group(1).strip(), m.group(2).lower(), m.group(3), m.group(4)
            if orig not in ratings:
                raise ValueError(f"{fd}: cable '{orig}' has no frozen rating")
            if orig in codes and codes[orig] != (name, r, x):
                raise ValueError(f"{fd}: cable '{orig}' differs from an earlier feeder")
            codes[orig] = (name, r, x)
        known = {c[0] for c in codes.values()}

        def bus(b: str) -> str:
            return "TRbus" if b == "1" else f"{k}_{b}"

        for ln in (fd / "Lines.txt").read_text().splitlines():
            if not ln.lower().startswith("new line"):
                continue
            m = _LINE_RE.match(ln)
            if not m:
                raise ValueError(f"{fd}: unparsed line: {ln}")
            if m.group(2).lower() == "sourcebus":
                continue                                            # stub LINE0
            if m.group(4).lower() not in known:
                raise ValueError(f"{fd}: unknown line code {m.group(4)}")
            lines.append(f"New Line.L{k}_{m.group(1)} Bus1={bus(m.group(2))}.1.2.3.4 Bus2={bus(m.group(3))}.1.2.3.4 "
                         f"phases=4 Linecode={m.group(4).lower()} {m.group(5)}")
        for ln in (fd / "Loads.txt").read_text().splitlines():
            if not ln.lower().startswith("new load"):
                continue
            if m := _LOAD1_RE.match(ln):
                loads.append(f"New Load.F{k}_{m.group(1)} Phases=1 Bus1={bus(m.group(2))}.{m.group(3)}.4 "
                             "kV=0.23 kW=1 PF=0.95")
            elif m := _LOAD3_RE.match(ln):
                n_three += 1
                for ph in (1, 2, 3):
                    loads.append(f"New Load.F{k}_{m.group(1)}{THREE_PHASE_SUFFIX}{ph} Phases=1 "
                                 f"Bus1={bus(m.group(2))}.{ph}.4 kV=0.23 kW=1 PF=0.95")
            else:
                raise ValueError(f"{fd}: unparsed load: {ln}")
            n_cust += 1
    lc_txt = []
    for orig, (name, r, x) in sorted(codes.items(), key=lambda t: int(t[1][0][2:])):
        lc_txt += [f"! original name: {orig}",
                   f"New LineCode.{name} nphases=4 baseFreq=50 Units=km NormAmps={ratings[orig]}",
                   f"~ rmatrix= [{r}]", f"~ xmatrix= [{x}]", ""]
    (out / "LineCode_3f4w.dss").write_text("\n".join(lc_txt))
    (out / "Lines.dss").write_text("\n".join(lines) + "\n")
    (out / "Loads.dss").write_text("\n".join(loads) + "\n")
    master = [ln for ln in (ENWL1_DIR / "Master.dss").read_text().splitlines()
              if not ln.lower().startswith(("new energymeter", "buscoords", "redirect loadshapes"))]
    (out / "Master.dss").write_text("\n".join(master) + "\n")
    return {"lines": len(lines), "customers": n_cust, "three_phase": n_three, "phase_connections": len(loads)}


def customers(load_names: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Map phase-connections (Loads.dss order) to customers. Returns (customer names, customer index of
    each connection, share of the customer's power on each connection: 1 or 1/3)."""
    names, idx, share = [], [], []
    for ln in load_names:
        m = re.fullmatch(rf"(\w+){THREE_PHASE_SUFFIX}([123])", ln)
        base = m.group(1) if m else ln
        if not names or names[-1] != base:
            names.append(base)
        idx.append(len(names) - 1)
        share.append(1 / 3 if m else 1.0)
    return names, np.array(idx), np.array(share)


def convert_all(out_root: Path, variant: str = "primary", src: Path = CSIRO_DIR) -> dict[int, dict]:
    """Convert every network into out_root/network_N."""
    return {n: convert(d, out_root / f"network_{n}", variant) for n, d in sorted(network_dirs(src).items())}
