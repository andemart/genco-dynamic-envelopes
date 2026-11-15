"""CSIRO -> ENWL1-layout converter (docs/fourwire_scale_protocol.md §2.2, §2.4)."""
import os
import re

import numpy as np
import pytest

from gencodoe import csiro_convert as cc
from gencodoe import fourwire_net as fw


def test_all_networks_convert_with_frozen_counts(tmp_path):
    r = cc.convert_all(tmp_path)
    assert sorted(r) == list(range(1, 26))
    assert sum(v["customers"] for v in r.values()) == 7539
    assert {n: v["three_phase"] for n, v in r.items() if v["three_phase"]} == {8: 1, 12: 7, 16: 1}
    assert all(v["phase_connections"] == v["customers"] + 2 * v["three_phase"] for v in r.values())


def test_network1_reproduces_enwl1(tmp_path):
    cc.convert(cc.network_dirs()[1], tmp_path)
    a, b = fw.load_fourwire(tmp_path), fw.load_fourwire()
    assert a.n_bus == b.n_bus
    assert np.allclose(a.z, b.z) and np.array_equal(a.rating_a, b.rating_a)
    assert np.array_equal(a.customer_phase, b.customer_phase)
    assert np.array_equal(a.customer_bus, b.customer_bus)


def test_ratings_match_enwl1_and_variants_change_one_cable(tmp_path):
    enwl1 = {}
    for block in re.split(r"(?=! original name)", (cc.ENWL1_DIR / "LineCode_3f4w.dss").read_text()):
        m = re.search(r"original name: (.*?)\n.*?NormAmps=(\d+)", block, re.S)
        if m:
            enwl1[m.group(1).strip()] = int(m.group(2))
    assert {k: v for k, v in cc.RATINGS.items() if k in enwl1} == enwl1
    assert set(cc.RATINGS) - set(enwl1) == {"BS 6480 1988 table 17 Cu 120 (50 Hz)"}
    for variant, (cable, amps) in {"cu120_240": ("Cu 120", 240), "cu16_89": ("Cu 16", 89)}.items():
        diff = {k: v for k, v in cc.RATING_VARIANTS[variant].items() if cc.RATINGS[k] != v}
        assert diff == {f"BS 6480 1988 table 17 {cable} (50 Hz)": amps}


def test_three_phase_split_maps_to_one_customer(tmp_path):
    cc.convert(cc.network_dirs()[16], tmp_path)
    net = fw.load_fourwire(tmp_path)
    names, idx, share = cc.customers(net.customer_names)
    assert len(names) == 238 and len(net.customer_names) == 240
    split = [i for i, n in enumerate(net.customer_names) if cc.THREE_PHASE_SUFFIX in n]
    assert len(split) == 3 and len(set(idx[split])) == 1
    assert sorted(net.customer_phase[split]) == [0, 1, 2] and np.allclose(share[split], 1 / 3)
    assert np.isclose(share.sum(), len(names))


def _bus_volts(folder, loads_text):
    od = pytest.importorskip("opendssdirect")
    (folder / "Loads.dss").write_text(loads_text)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        od.Text.Command("clear")
        od.Text.Command(f"Redirect {folder / 'Master.dss'}")
        od.Text.Command("Set mode=snapshot")
        od.Text.Command("Set maxiterations=100")
        od.Text.Command("Set tolerance=1e-6")
        od.Solution.Solve()
        assert od.Solution.Converged()
        v = np.array(od.Circuit.AllBusVolts())
        return dict(zip(od.Circuit.AllNodeNames(), v[0::2] + 1j * v[1::2]))
    finally:
        os.chdir(cwd)


def test_split_equals_balanced_wye_load(tmp_path):
    """§2.2 item 5: three phase-to-neutral loads of S/3 == one balanced constant-power wye load of S.
    At the §2.3 light load (0.5 kW per customer); at 1 kW OpenDSS fails to converge with the wye load
    while the split converges, and where both converge they agree to ~2e-9 V."""
    cc.convert(cc.network_dirs()[16], tmp_path)
    split_text = (tmp_path / "Loads.dss").read_text().replace("kW=1 ", "kW=0.5 ")
    lines = split_text.splitlines()
    first = next(ln for ln in lines if cc.THREE_PHASE_SUFFIX in ln)
    name, bus = re.search(rf"Load\.(\w+){cc.THREE_PHASE_SUFFIX}1 .*?Bus1=(\w+)\.", first).groups()
    wye = [ln for ln in lines if cc.THREE_PHASE_SUFFIX not in ln]
    wye.append(f"New Load.{name} Phases=3 Bus1={bus}.1.2.3.4 kV=0.416 Conn=wye kW=1.5 PF=0.95")
    a = _bus_volts(tmp_path, split_text)
    b = _bus_volts(tmp_path, "\n".join(wye) + "\n")
    assert a.keys() == b.keys()
    assert max(abs(a[k] - b[k]) for k in a) < 1e-6           # volts


def _copy_network(src, dst):
    for fd in src.glob("Feeder_*"):
        (dst / fd.name).mkdir(parents=True)
        for f in ("LineCode.txt", "Lines.txt", "Loads.txt"):
            (dst / fd.name / f).write_text((fd / f).read_text())


@pytest.mark.parametrize("fname, old, new, msg", [
    ("Lines.txt", "phases=4", "phases=3", "unparsed line"),
    ("Loads.txt", "PF=0.95", "PF=0.9", "unparsed load"),
    ("LineCode.txt", "BS 5467 1997 table 10 16", "BS 5467 1997 table 10 10", "no frozen rating"),
])
def test_conversion_fails_closed(tmp_path, fname, old, new, msg):
    src = tmp_path / "net"
    _copy_network(cc.network_dirs()[25], src)
    f = src / "Feeder_1" / fname
    text = f.read_text()
    assert old in text
    f.write_text(text.replace(old, new, 1))
    with pytest.raises(ValueError, match=msg):
        cc.convert(src, tmp_path / "out")
