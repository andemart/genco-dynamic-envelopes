"""Four-wire network, labels and prior (docs/unbalanced_genco_protocol.md)."""
import re

import numpy as np
import pytest

from gencodoe import fourwire_net as fw


def test_linecode_4x4_parsing():
    codes = fw.read_linecodes_4x4(fw.ENWL1_DIR / "LineCode_3f4w.dss")
    z, amps = codes["lc1"]
    assert z.shape == (4, 4) and amps == 110
    assert np.isclose(z[0, 0].real, 1.3480250575502448) and np.isclose(z[0, 1].imag, 0.7013971531485135)
    assert all(np.allclose(zz, zz.T, atol=1e-6) for zz, _ in codes.values())


def test_reduction_preserves_summed_impedance_and_aligns_with_balanced():
    from gencodoe.enwl import load_enwl1
    net = fw.load_fourwire()
    codes = fw.read_linecodes_4x4(fw.ENWL1_DIR / "LineCode_3f4w.dss")
    raw = {}
    pat = re.compile(r"Bus1=(\w+)\.1\.2\.3\.4\s+Bus2=(\w+)\.1\.2\.3\.4.*?Linecode=(\w+)\s+Length=([\d.eE+-]+)", re.I)
    for line in (fw.ENWL1_DIR / "Lines.dss").read_text().splitlines():
        m = pat.search(line)
        if m:
            b1, b2, code, length = m.groups()
            z = codes[code.lower()][0] * float(length) / 1e3
            raw.setdefault(b1.lower(), []).append((b2.lower(), z))
            raw.setdefault(b2.lower(), []).append((b1.lower(), z))
    # root-to-bus path impedance in the raw network = sum of reduced branch matrices along the reduced path
    ztot, seen, stack = {fw.ROOT: np.zeros((4, 4), complex)}, {fw.ROOT}, [fw.ROOT]
    while stack:
        u = stack.pop()
        for v, z in raw[u]:
            if v not in seen:
                seen.add(v)
                ztot[v] = ztot[u] + z
                stack.append(v)
    for i in range(1, net.n_bus):
        acc, k = np.zeros((4, 4), complex), i
        while k > 0:
            acc += net.z[k]
            k = net.parent[k]
        assert np.allclose(acc, ztot[net.bus_names[i]], rtol=1e-10, atol=1e-12)
    bal = load_enwl1()
    assert set(net.bus_names) == set(bal.bus_names) - {"sourcebus"}
    assert [net.bus_names[i] for i in net.customer_bus] == [bal.bus_names[i] for i in bal.customer_bus]
    assert net.bus_names[0] == fw.ROOT and (net.parent[1:] >= 0).all() and np.isfinite(net.rating_a[1:]).all()


def _toy():
    """root -> 1 -> 2 (customers a, b, c at bus 2)."""
    zl = 0.02 * np.eye(4) + 0.005 * (np.ones((4, 4)) - np.eye(4)) + 1j * 0.01 * np.eye(4)
    return fw.FourWireNet(bus_names=["trbus", "x", "y"], parent=np.array([-1, 0, 1]), order=np.array([0, 1, 2]),
                          z=np.stack([np.zeros((4, 4)), zl, zl]).astype(complex), rating_a=np.array([np.inf, 100, 100]),
                          length_m=np.array([0, 10, 10.0]), first_line=[None, "l1", "l2"],
                          first_line_parent_terminal=np.array([0, 1, 1]), customer_names=["a", "b", "c"],
                          customer_bus=np.array([2, 2, 2]), customer_phase=np.array([0, 1, 2]), depth=np.array([0, 1, 2]))


def _th():
    e = fw.V_PN_NOM * np.exp(1j * np.deg2rad([0, -120, 120]))
    return fw.Thevenin(e=e, zth=0.001j * np.eye(3))


def test_sweep_balanced_inputs_small_neutral_current():
    net = _toy()
    s = np.full((1, 3), 3.0 + 1.0j)                          # 3 kW + 1 kvar on each phase
    v, cur, it = fw.sweep(net, _th(), s, iterations=50)
    assert it < 50 and abs(cur[0, 1, 3]) < 1e-9 * 1e3 + 1e-9
    vpn = fw.phase_neutral_pu(net, v)[0]
    assert np.ptp(vpn) < 1e-9 and vpn.max() < 1.0


def test_sweep_imbalanced_inputs_neutral_displacement_and_kcl():
    net = _toy()
    s = np.array([[6.0 + 2.0j, 0.0, 0.0]])                   # phase a only
    v, cur, _ = fw.sweep(net, _th(), s, iterations=50)
    assert np.isclose(cur[0, 2, 3], -cur[0, 2, 0], rtol=1e-12)     # neutral returns the phase current
    assert abs(v[0, 2, 3]) > 1e-4                             # neutral displaced at the customer bus
    inj = fw.customer_currents(net, v, s)
    assert np.allclose(cur[0, 2], inj[0, 2]) and np.allclose(cur[0, 1], inj[0, 2])
    one, _, _ = fw.sweep(net, _th(), s, iterations=1)
    assert not np.allclose(one, v)                           # U1 differs from the converged sweep


def test_extractor_node_mapping_complete():
    pytest.importorskip("opendssdirect")
    from gencodoe.fourwire_labels import Extractor
    ex = Extractor()
    assert len(ex.node_idx) == ex.net.n_bus * 4 == len(set(ex.node_idx))
    s = ex.state(np.full(200, 0.5), np.zeros(200))
    assert s["converged"] and abs(s["V"][0, 3]) < 1e-6 and np.isfinite(s["I"]).all()


def test_test_split_is_isolated():
    from gencodoe.fourwire_labels import load_split
    with pytest.raises(PermissionError):
        load_split("test")
