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


def _setup_model(node_in=None, device="cpu"):
    torch = pytest.importorskip("torch")
    from gencodoe import fourwire_model as fm
    net = fw.load_fourwire()
    e = fw.V_PN_NOM * np.exp(1j * np.deg2rad([30, -90, 150]))
    st = fm.Static(net, fw.Thevenin(e=e, zth=0.002j * np.eye(3)))
    rng = np.random.default_rng(0)
    p = rng.uniform(-3, 3, (3, 200)).astype(np.float32)
    q = (0.3 * p).astype(np.float32)
    v1 = fm.prior_u1(st, p.astype(float), q.astype(float))
    x = fm.node_features(st, p, q, v1)
    model = fm.GNS_heterogeneous_4wire(node_in or fm.N_NODE_BASE, hidden=16, layers=2, heads=2)
    b = fm.Batcher(st, fm.Scaler(st.edge_raw)(st.edge_raw), torch.device(device))
    return torch, fm, net, st, p, q, v1, x, model, b


def test_model_zero_init_reproduces_prior_and_shapes():
    torch, fm, net, st, p, q, v1, x, model, b = _setup_model()
    assert x.shape == (3, net.n_bus, fm.N_NODE_BASE) and st.edge_raw.shape == (2 * (net.n_bus - 1), fm.N_EDGE)
    xs = fm.Scaler(x)(x)
    ei, ea = b.graph(3)
    out = model(torch.as_tensor(xs).reshape(3 * net.n_bus, -1), ei, ea, 3)
    assert out.shape == (3 * net.n_bus, 8) and torch.count_nonzero(out) == 0        # residual 0 -> exactly U1


def test_model_batch_isolation_and_finite_gradients():
    torch, fm, net, st, p, q, v1, x, model, b = _setup_model()
    torch.manual_seed(0)
    torch.nn.init.normal_(model.dec[-1].weight, std=0.1)                # make the residual non-trivial
    xs = torch.as_tensor(fm.Scaler(x)(x))
    ei, ea = b.graph(3)
    full = model(xs.reshape(3 * net.n_bus, -1), ei, ea, 3).view(3, net.n_bus, 8)
    ei1, ea1 = b.graph(1)
    alone = model(xs[1:2].reshape(net.n_bus, -1), ei1, ea1, 1).view(1, net.n_bus, 8)
    assert torch.allclose(full[1], alone[0], atol=1e-5)                 # no leakage between graphs in a batch
    phys = fm.Physics(st, torch.device("cpu"))
    v = torch.as_tensor(v1, dtype=torch.complex64) + torch.complex(full[..., :4], full[..., 4:]) * 1e-3
    s = torch.as_tensor(p + 1j * q, dtype=torch.complex64)
    loss = phys.vpn(v).mean() + phys.currents(v, s).abs().mean() + phys.kvl(v, phys.currents(v, s)).abs().mean()
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)


def test_physics_torch_matches_numpy_sweep():
    torch, fm, net, st, p, q, v1, x, model, b = _setup_model()
    s = (p + 1j * q).astype(complex)
    v, cur, _ = fw.sweep(net, st.th, s, iterations=30)
    phys = fm.Physics(st, torch.device("cpu"))
    vt = torch.as_tensor(v, dtype=torch.complex64)
    ct = phys.currents(vt, torch.as_tensor(s, dtype=torch.complex64)).numpy()
    assert np.allclose(ct, cur, atol=1e-6)
    assert np.allclose(phys.vpn(vt).numpy(), fw.phase_neutral_pu(net, v), atol=1e-5)
    assert np.abs(phys.kvl(vt, torch.as_tensor(cur, dtype=torch.complex64)).numpy()).max() < 1e-4


def test_phasegenco_alignment_and_frozen_backbone():
    from gencodoe.fourwire_train import bus_alignment
    idx = bus_alignment()
    from gencodoe.enwl import load_enwl1
    bal, net = load_enwl1(), fw.load_fourwire()
    assert len(idx) == net.n_bus == len(set(idx)) and [bal.bus_names[i] for i in idx] == net.bus_names
    torch, fm, net, st, p, q, v1, x, _, b = _setup_model()
    model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE + fm.EMB_WIDTH, hidden=16, layers=2, heads=2)
    assert model.enc_node[0].in_features == fm.N_NODE_BASE + fm.EMB_WIDTH     # per-bus, not pooled
