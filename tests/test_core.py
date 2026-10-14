import numpy as np

from probredispatch import conformal, grid, metrics, scenarios


def _spec(r=3):
    return scenarios.ForecastSpec(
        point={"load": np.full(r, 100.0), "wind": np.full(r, 50.0), "pv": np.full(r, 30.0)},
        rel_std={"load": 0.03, "wind": 0.2, "pv": 0.15},
        corr=scenarios.default_corr(r),
        capacity={"load": None, "wind": np.full(r, 80.0), "pv": np.full(r, 40.0)},
    )


def test_sample_shapes_and_bounds():
    s = scenarios.sample(_spec(), 2000, np.random.default_rng(0))
    assert s["load"].shape == (2000, 3)
    assert (s["wind"] >= 0).all() and (s["wind"] <= 80).all()
    assert abs(s["load"].mean() - 100) < 1


def test_corr_is_valid():
    c = scenarios.default_corr(4)
    assert np.allclose(np.diag(c), 1)
    assert np.linalg.eigvalsh(c).min() > 0


def test_bus_loads_preserve_regional_totals():
    m = grid.CaseMapping(
        bus_region=np.array([0, 0, 1]), bus_pd_nominal=np.array([10.0, 30.0, 20.0]),
        bus_qd_nominal=np.array([2.0, 6.0, 4.0]), gen_region=np.array([0, 1]),
        gen_tech=np.array(["wind", "other"]), gen_pmax_nominal=np.array([50.0, 100.0]),
    )
    reg = np.array([[80.0, 40.0]])
    pd, qd = grid.bus_loads(m, reg)
    assert np.allclose(pd, [[20, 60, 40]])
    assert np.allclose(qd, pd * 0.2)
    pmax = grid.res_pmax(m, {"wind": np.array([[35.0, 0.0]]), "pv": np.zeros((1, 2))})
    assert np.allclose(pmax, [[35, 100]])


def test_conformal_coverage_and_screen():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.3, 1.2, size=(4000, 20))
    pred = true + rng.normal(0, 0.03, size=true.shape)
    q = conformal.calibrate(pred[:2000], true[:2000], alpha=0.1)
    cov = conformal.coverage(pred[2000:], true[2000:], q)
    assert cov >= 0.88
    d = conformal.screen(pred[2000:], q)
    rep = metrics.screening_report(d, true[2000:])
    assert rep["missed_congestion_rate"] < 0.01


def test_crps_zero_for_perfect_point():
    assert metrics.crps_ensemble(np.array([1.0, 1.0]), 1.0) == 0.0
