"""GENCO-OPF layer (docs/genco_opf_protocol.md): case, multi-scenario OPF physics, pipeline, models."""
import numpy as np
import pytest

from gencodoe import envelopes
from gencodoe import feeder as fdr
from gencodoe import opf_case33 as oc
from gencodoe import opf_envelopes as oe


@pytest.fixture(scope="module")
def case():
    return oc.build()


def test_case_ratings_slack_and_caps(case):
    assert case.fd.n_bus == 33 and len(case.unit_bus) == 32
    assert np.isfinite(case.fd.rate[1:]).all() and case.fd.v_slack == 1.0
    assert np.allclose(case.cap["export"], 3.0 * case.pd[case.unit_bus])
    assert np.allclose(case.cap["import"], 1.5 * case.pd[case.unit_bus])


def test_scenarios_deterministic(case):
    sc = oc.Scenarios(case, 0)
    a = oe.interval_draws(sc, 80, 13)
    b = oe.interval_draws(sc, 80, 13)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and a[0].shape == (200, 33)


def test_multi_opf_solution_is_exact_feasible(case):
    """End-to-end check of the JuMP power-flow equations: the optimal shared vector must pass the exact sweep
    on (almost) every kept scenario, and must beat uniform on these scenarios."""
    pytest.importorskip("juliacall")
    from gencodoe.opf_multi import MultiOPF
    m = MultiOPF()
    sc = oc.Scenarios(case, 0)
    p, q, _, _ = oe.interval_draws(sc, 80, 13)
    p, q = p[:12], q[:12]
    cust = oe.customers(case)
    for d in ("export", "import"):
        x, ok, _ = m.solve(case.fd, -p, -q, cust.bus, cust.cap(d), d)
        assert ok and (x >= -1e-7).all() and (x <= cust.cap(d) + 1e-7).all()
        inj = oe.SIGN[d] * np.clip(x, 0, None) @ cust.incidence(case.fd.n_bus)
        res = fdr.sweep_pf(case.fd, p + inj, q)
        if d == "export":
            assert res.vm[:, 1:].max() <= case.fd.vmax + 1e-5
        else:
            assert res.vm[:, 1:].min() >= case.fd.vmin - 1e-5
        assert res.loading[:, 1:].max() <= 1.0 + 1e-5
        uni = envelopes.scalar_envelope(envelopes.exact(fdr.sweep_pf, case.fd), case.fd.n_bus, cust, p, q, d,
                                        "uniform", 0.0, 4)
        assert x.sum() >= uni.limit.sum() - 1e-6


def _multi_env_setup(case, n_bad):
    """40 design scenarios, scenario 0..n_bad-1 scaled x5 so they fail the exact check at x = 0 in both
    directions (undervoltage and branch overload). With beta=None, eps=0.05 the rule allows 40 - ceil(38) = 2
    discards."""
    pytest.importorskip("juliacall")
    from gencodoe.opf_multi import MultiOPF
    sc = oc.Scenarios(case, 0)
    p, q, _, _ = oe.interval_draws(sc, 80, 13)
    p, q = p[:40].copy(), q[:40].copy()
    p[:n_bad] *= 5.0
    q[:n_bad] *= 5.0
    feas = envelopes.exact(fdr.sweep_pf, case.fd)
    for d in ("export", "import"):
        assert (~feas(p, q, d)).sum() == n_bad
    return MultiOPF(), oe.customers(case), p, q


def test_multi_envelope_discards_infeasible_scenario(case):
    """One scenario infeasible at x = 0 is discarded, the OPF runs on the rest, and the result passes the exact
    check on every scenario except the discarded ones (at most 2 in total)."""
    from gencodoe.opf_multi import multi_envelope
    m, cust, p, q = _multi_env_setup(case, 1)
    for d in ("export", "import"):
        x, info = multi_envelope(m, case.fd, cust, p, q, d, eps=0.05, beta=None)
        assert info["infeasible_at_zero"] == 1 and info["allowed_discards"] == 2
        assert not info["zero_sentinel"] and info["opf_ok"]
        assert 38 <= info["kept_scenarios"] <= 39
        assert x.sum() > 0 and (x >= 0).all() and (x <= cust.cap(d) + 1e-7).all()
        # x is exact-feasible (to IPOPT's boundary tolerance) on every kept scenario; the real violators are
        # exactly the discarded ones: the bad scenario plus any greedy discard
        inj = oe.SIGN[d] * x @ cust.incidence(case.fd.n_bus)
        res = fdr.sweep_pf(case.fd, p + inj, q)
        vm = res.vm[:, 1:]
        head_v = (case.fd.vmax - vm.max(1)) if d == "export" else (vm.min(1) - case.fd.vmin)
        violated = (np.minimum(head_v, 1.0 - res.loading[:, 1:].max(1)) < -1e-5) | ~res.converged
        assert violated[0] and violated.sum() == 40 - info["kept_scenarios"]


def test_multi_envelope_zero_sentinel_when_too_many_infeasible(case):
    """Three infeasible scenarios exceed the 2-discard allowance: envelope 0, no OPF solve."""
    from gencodoe.opf_multi import multi_envelope
    m, cust, p, q = _multi_env_setup(case, 3)
    for d in ("export", "import"):
        x, info = multi_envelope(m, case.fd, cust, p, q, d, eps=0.05, beta=None)
        assert info["zero_sentinel"] and info["infeasible_at_zero"] == 3 and info["allowed_discards"] == 2
        assert not x.any() and len(x) == len(cust) and info["opf_solve_s"] == 0.0


def test_scale_down_zero_and_passthrough(case):
    sc = oc.Scenarios(case, 0)
    p, q, _, _ = oe.interval_draws(sc, 80, 13)
    cust = oe.customers(case)
    lim, lam, share, _ = oe.scale_down(case, cust, np.zeros(len(cust)), p, q, "export")
    assert not lim.any() and lam == 0.0
    small = 0.01 * case.cap["export"]
    lim, lam, share, _ = oe.scale_down(case, cust, small, p, q, "export")
    assert share >= envelopes.acceptance_threshold(200, 0.05, 0.05) and np.allclose(lim, small * lam)


def test_features_and_models(case):
    torch = pytest.importorskip("torch")
    from gencodoe import opf_gnn, opf_labels
    sc = oc.Scenarios(case, 0)
    p, q, _, _ = oe.interval_draws(sc, 80, 13)
    f = opf_labels.node_features(case, p, q, "import")
    assert f.shape == (33, opf_labels.N_FEAT) and np.isfinite(f).all()
    ei, ea = opf_gnn.graph(case)
    for kind in ("gnn", "mlp"):
        m = opf_gnn.build_model(kind, 33, f.shape[1], ea.shape[1])
        b = 3
        x = torch.as_tensor(np.tile(f, (b, 1)))
        off = (torch.arange(b) * 33).repeat_interleave(ei.shape[1])
        out = m(x, torch.as_tensor(ei).repeat(1, b) + off, torch.as_tensor(ea, dtype=torch.float32).repeat(b, 1), b)
        assert out.shape == (b, 33) and (out >= 0).all() and (out <= 1).all()


def test_fit_smoke_on_per_unit_labels(case, tmp_path, monkeypatch):
    """fit() consumes labels shaped (N, units) as opf_labels writes them, trains one epoch for both models and
    writes a loadable run; the Predictor returns one fraction per bus."""
    pytest.importorskip("torch")
    from gencodoe import opf_gnn, opf_labels
    sc = oc.Scenarios(case, 0)
    rng = np.random.default_rng(0)

    def fake_days(days):
        X, Y, M = [], [], []
        for d in days:
            for h in (0, 12):
                p, q, _, _ = oe.interval_draws(sc, d, h)
                for dr in ("export", "import"):
                    X.append(opf_labels.node_features(case, p, q, dr))
                    Y.append(rng.random(len(case.unit_bus)).astype(np.float32))
                    M.append({"day": d, "hour": h, "direction": dr})
        return np.array(X), np.array(Y), M
    monkeypatch.setattr(opf_gnn, "load_days", fake_days)
    monkeypatch.setattr(opf_gnn, "RUNS", tmp_path)
    for kind in ("gnn", "mlp"):
        meta = opf_gnn.fit(kind, "t", 1)
        assert meta["train_intervals"] == 4 * len(oc.SPLIT["train"]) and np.isfinite(meta["best_val"])
        pred = opf_gnn.Predictor(tmp_path / f"{kind}_t", case)
        out = pred.predict(fake_days([0])[0][:2])
        assert out.shape == (2, 33) and (out >= 0).all() and (out <= 1).all()


def test_test_days_never_labelled():
    from gencodoe.opf_labels import label_day
    with pytest.raises(SystemExit):
        label_day(110)
