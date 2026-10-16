import numpy as np
import pytest

from gencodoe import conformal, envelopes, feeder, metrics, scenarios


def _spec(g=3):
    return scenarios.ForecastSpec(
        point={"load": np.full(g, 0.10), "pv": np.full(g, 0.05)},
        rel_std={"load": 0.05, "pv": 0.2},
        corr=scenarios.default_corr(g),
        capacity={"load": None, "pv": np.full(g, 0.08)},
    )


def _setup(n_s=200, seed=0):
    fd = feeder.synthetic_feeder(n_trunk=8, n_laterals=2, lateral_len=3, rate=1.0)
    rng = np.random.default_rng(seed)
    load = 0.05 * (1 + 0.1 * rng.standard_normal((n_s, fd.n_bus)))
    load[:, 0] = 0.0
    p0, q0 = -load, -0.3 * load
    cust = envelopes.Customers(bus=np.arange(1, fd.n_bus), export_cap=np.full(fd.n_bus - 1, 0.3),
                               import_cap=np.full(fd.n_bus - 1, 0.3))
    return fd, p0, q0, cust


def test_sample_shapes_and_bounds():
    s = scenarios.sample(_spec(), 2000, np.random.default_rng(0))
    assert s["load"].shape == (2000, 3)
    assert (s["pv"] >= 0).all() and (s["pv"] <= 0.08).all()
    assert abs(s["load"].mean() - 0.10) < 0.002
    p, q = scenarios.net_injection(s, 0.3)
    assert np.allclose(p, s["pv"] - s["load"]) and np.allclose(q, -0.3 * s["load"])


def test_corr_is_valid():
    c = scenarios.default_corr(4)
    assert np.allclose(np.diag(c), 1)
    assert np.linalg.eigvalsh(c).min() > 0


def test_feeder_orientation_and_bad_input():
    fd = feeder.Feeder.from_branches([1, 1, 2], [0, 2, 3], [0.1, 0.2, 0.3], [0.1, 0.1, 0.1])
    assert list(fd.parent) == [-1, 0, 1, 2]
    assert np.allclose(fd.r, [0, 0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        feeder.Feeder.from_branches([0, 1], [1, 0], [0.1, 0.1], [0.1, 0.1])


def test_sweep_matches_flows_from_voltages_and_power_balance():
    fd, p0, q0, _ = _setup(n_s=5)
    res = feeder.sweep_pf(fd, p0, q0)
    assert res.converged.all()
    # recompute flows from voltage phasors via a second sweep with angles
    s = p0 + 1j * q0
    v = np.full(s.shape, 1.0, dtype=complex)
    for _ in range(50):
        j = (-np.conj(s / v)) @ fd.T
        v = fd.v_slack - (j * fd.z) @ fd.T.T
    again = feeder.branch_flows_from_voltages(fd, np.abs(v), np.angle(v))
    assert np.allclose(again.loading, res.loading, atol=1e-6)
    # substation supplies load + losses
    supply = res.p_branch[:, fd.parent == 0].sum(axis=1)
    losses = supply + p0.sum(axis=1)
    assert (losses > 0).all() and (losses < 0.05 * supply).all()


def test_lindistflow_close_to_ac_at_light_load():
    fd, p0, q0, _ = _setup(n_s=20)
    ac, lin = feeder.sweep_pf(fd, 0.2 * p0, 0.2 * q0), feeder.lindistflow(fd, 0.2 * p0, 0.2 * q0)
    assert np.abs(ac.vm - lin.vm).max() < 2e-3
    assert np.abs(ac.p_branch - lin.p_branch).max() < 5e-3


def test_uniform_envelope_is_tight():
    fd, p0, q0, cust = _setup()
    feas = envelopes.exact(feeder.sweep_pf, fd)
    env = envelopes.scalar_envelope(feas, fd.n_bus, cust, p0, q0, eps=0.05, levels_per_round=3)
    assert 0 < env.level < 0.3
    assert env.prob_ok >= 0.95
    C = cust.incidence(fd.n_bus)
    above = envelopes.allocate(cust, "uniform", env.level * 1.02) @ C
    assert feas(p0 + above, q0, "export").mean() < 0.95


def test_import_envelope_respects_vmin():
    fd, p0, q0, cust = _setup()
    env = envelopes.scalar_envelope(envelopes.exact(feeder.sweep_pf, fd), fd.n_bus, cust, p0, q0,
                                    direction="import", eps=0.0)
    vm = feeder.sweep_pf(fd, p0 - env.limit @ cust.incidence(fd.n_bus), q0).vm
    assert vm.min() >= fd.vmin - 1e-9


def test_max_total_beats_uniform_on_linear_model():
    fd, p0, q0, cust = _setup()
    lin = envelopes.exact(feeder.lindistflow, fd)
    uni = envelopes.scalar_envelope(lin, fd.n_bus, cust, p0, q0, eps=0.0)
    mt = envelopes.max_total_envelope(fd, cust, feeder.lindistflow(fd, p0, q0), eps=0.0)
    assert mt.limit.sum() >= uni.limit.sum() - 1e-6
    vm = feeder.lindistflow(fd, p0 + mt.limit @ cust.incidence(fd.n_bus), q0).vm
    assert vm.max() <= fd.vmax + 1e-6
    assert metrics.jain_index(mt.limit) <= metrics.jain_index(uni.limit) + 1e-9


def test_conformal_coverage_and_screen():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.3, 1.2, size=(4000, 20))
    pred = true + rng.normal(0, 0.03, size=true.shape)
    q = conformal.calibrate(pred[:2000], true[:2000], alpha=0.1)
    assert conformal.coverage(pred[2000:], true[2000:], q) >= 0.88


def test_screened_feasibility_matches_exact_with_noisy_surrogate():
    fd, p0, q0, cust = _setup()
    rng = np.random.default_rng(2)

    def noisy(f, p, q):
        r = feeder.sweep_pf(f, p, q)
        r.vm = r.vm + rng.normal(0, 0.002, r.vm.shape)
        return r

    C = cust.incidence(fd.n_bus)
    p_cal = p0 + rng.uniform(0, 0.3, (len(p0), 1)) * C.sum(axis=0)
    q_v, q_l = conformal.calibrate_pf(fd, noisy(fd, p_cal, q0), feeder.sweep_pf(fd, p_cal, q0), alpha=0.001)
    screened = envelopes.ScreenedFeasibility(fd, noisy, q_v, q_l, verifier=feeder.sweep_pf)
    truth = envelopes.is_feasible(fd, feeder.sweep_pf(fd, p_cal, q0))
    assert (screened(p_cal, q0) == truth).mean() > 0.99
    assert 0 < screened.n_verified < screened.n_surrogate
    dec = conformal.screen(fd, noisy(fd, p_cal, q0), q_v, q_l)
    assert metrics.screening_report(dec, truth)["missed_violation_rate"] < 0.02


def test_metrics():
    assert metrics.jain_index(np.ones(5)) == 1.0
    assert abs(metrics.jain_index(np.array([1.0, 0, 0, 0])) - 0.25) < 1e-12
    assert metrics.crps_ensemble(np.array([1.0, 1.0]), 1.0) == 0.0
    cur = metrics.curtailment(np.array([0.5, 1.0]), np.array([[1.0, 0.5], [0.2, 2.0]]))
    assert np.allclose(cur, [0.5, 1.0])


def test_export_envelope_ignores_base_case_undervoltage():
    fd, p0, q0, cust = _setup()
    heavy = 6 * p0                      # base case already below vmin
    assert not envelopes.is_feasible(fd, feeder.sweep_pf(fd, heavy, q0)).any()
    env = envelopes.scalar_envelope(envelopes.exact(feeder.sweep_pf, fd), fd.n_bus, cust, heavy, q0, "export")
    assert env.level > 0


def test_spatial_envelope_is_jointly_certified_and_not_below_uniform_total():
    fd, p0, q0, cust = _setup()
    feas = envelopes.exact(feeder.sweep_pf, fd)
    groups = (np.arange(len(cust)) >= len(cust) // 2).astype(int)
    env, solo, lam = envelopes.spatial_envelope(feas, fd.n_bus, cust, groups, p0, q0, "export", eps=0.05,
                                                levels_per_round=3, beta=0.05)
    uni = envelopes.scalar_envelope(feas, fd.n_bus, cust, p0, q0, "export", eps=0.05, levels_per_round=3, beta=0.05)
    target = envelopes.acceptance_threshold(len(p0), 0.05, 0.05)
    assert len(solo) == 2 and 0 <= lam <= 1
    assert np.allclose(env.limit, lam * solo[groups], atol=1e-12)
    assert feas(p0 + env.limit @ cust.incidence(fd.n_bus), q0, "export").mean() >= target
    assert solo.min() >= uni.level - 1e-3 * 0.3            # each group alone gets at least the uniform level
    assert env.limit.sum() >= uni.limit.sum() * (1 - 1e-2)


def test_spatial_envelope_zero_sentinel():
    fd, p0, q0, cust = _setup()
    never = lambda p, q, direction=None: np.zeros(len(p), bool)
    env, solo, lam = envelopes.spatial_envelope(never, fd.n_bus, cust, np.zeros(len(cust), int), p0, q0, "export")
    assert lam == 0 and not env.limit.any() and not solo.any()


def test_envpred_stats_proxy_and_split():
    import json
    from pathlib import Path
    from gencodoe import envpred
    from gencodoe.report_annual import season
    x = np.arange(200.0)[:, None] * np.ones((1, 3))
    s = envpred.stats(x)
    assert s.shape == (21,) and s[3] == 0 and s[-1] == 199          # mean | min ... max blocks of 3
    L_B = np.array([1.0, 2.0, 0.0])
    assert envpred.proxy_retention(np.array([0.5, 2.5, 0.0]), L_B, np.array(["export"] * 3)) == 0.5 / 3
    sp = json.loads(Path("results/envpred_split.json").read_text())
    days = [d for k in ("train", "val", "calib", "test") for d in sp[k]]
    assert len(days) == len(set(days)) == 36
    for k, n in (("train", 4), ("val", 1), ("calib", 2), ("test", 2)):
        assert all(sum(season(d) == s_ for d in sp[k]) == n for s_ in ("winter", "spring", "summer", "autumn"))
