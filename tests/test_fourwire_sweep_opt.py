"""Optimised FP64 sweeps against the one-state NumPy reference (fourwire_scale_protocol.md Amendment 3)."""
import numpy as np
import pytest

from gencodoe import fourwire_net as fw
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_sweep_opt as so

V_TOL_PU = 1e-9


@pytest.fixture(scope="module")
def case():
    net = fw.load_fourwire()                                      # ENWL1, in the repository
    ang = np.deg2rad([0.0, -120.0, 120.0])
    th = fw.Thevenin(e=fw.V_PN_NOM * np.exp(1j * ang), zth=(0.004 + 0.012j) * np.eye(3) + 0.001j)
    rng = np.random.default_rng(0)
    n_c = len(net.customer_names)
    p = np.clip(rng.normal(0.8, 2.0, (40, n_c)), -5, 7.4)
    p[30:36] += np.linspace(4, 9, 6)[:, None]                     # heavy import: some states do not converge
    p[36:] -= 6.0                                                 # heavy export
    s = p + 1j * np.abs(p) * 0.33
    ref = [fs.reference_sweep_state(net, th, row) for row in s]
    ref = {"V": np.array([r[0] for r in ref]), "I": np.array([r[1] for r in ref]),
           "iterations": np.array([r[2] for r in ref]), "converged": np.array([r[3] for r in ref])}
    assert ref["converged"].any() and (~ref["converged"]).any()   # the case covers both outcomes
    return net, th, s, ref


def _check(out, ref):
    c = ref["converged"]
    assert np.array_equal(out["converged"], c)
    assert np.array_equal(out["iterations"], ref["iterations"])
    assert np.abs(out["V"][c] - ref["V"][c]).max() / fw.V_PN_NOM <= V_TOL_PU
    assert np.abs(out["I"][c] - ref["I"][c]).max() <= 1e-9        # kA


def test_level_net_is_a_consistent_renumbering(case):
    net, th, _, _ = case
    ln = so.level_net(net, th)
    assert np.array_equal(ln.perm[ln.inv], np.arange(net.n_bus)) and ln.parent[0] == -1
    assert np.array_equal(ln.perm[ln.parent[1:]], net.parent[ln.perm[1:]])        # same tree
    assert (ln.parent[1:] < np.arange(1, net.n_bus)).all()                        # parents first
    assert np.array_equal(ln.perm[ln.cust_bus], net.customer_bus)
    for lvl in range(len(ln.level_ptr) - 1):
        assert (net.depth[ln.perm[ln.level_ptr[lvl]:ln.level_ptr[lvl + 1]]] == lvl).all()


def test_numba_sweep_matches_reference(case):
    net, th, s, ref = case
    _check(so.NumbaSweep(net, th).solve(s), ref)


def test_torch_sweep_matches_reference(case):
    net, th, s, ref = case
    _check(so.TorchSweep(net, th, "cpu").solve(s), ref)


@pytest.mark.parametrize("kind", ["numba", "torch"])
def test_per_state_semantics_and_summaries(case, kind):
    """A state's result does not depend on the batch it is solved in; summaries follow the ENWL1 definitions."""
    net, th, s, ref = case
    solver = so.NumbaSweep(net, th) if kind == "numba" else so.TorchSweep(net, th, "cpu")
    full = solver.solve(s)
    for sel in (slice(0, 1), slice(5, 6), slice(28, 40), slice(0, 40, 3)):
        part = solver.solve(s[sel])
        c = ref["converged"][sel]
        assert np.array_equal(part["iterations"], full["iterations"][sel])
        assert np.abs(part["V"][c] - full["V"][sel][c]).max() / fw.V_PN_NOM <= 1e-12
    light = solver.solve(s, return_state=False)
    assert "V" not in light and np.array_equal(light["summary"], full["summary"])
    c = ref["converged"]
    vpn = fw.phase_neutral_pu(net, ref["V"])
    load = (np.abs(ref["I"][:, 1:, :3]) / (net.rating_a[1:, None] / 1e3)).max((1, 2))
    s_root = np.abs(ref["V"][:, 0, :3] * np.conj(ref["I"][:, 0, :3])).sum(1) * 1e3 / 800.0
    expect = np.stack([vpn.min(1), vpn.max(1), load, s_root], 1)
    assert np.allclose(full["summary"][c], expect[c], rtol=0, atol=1e-8)
