"""E2 timing series, statistics and the envelope check of the 25-network study
(docs/fourwire_scale_protocol.md §6, §7 E2, Amendment 9). Network 25, train and val states only."""
import numpy as np
import pytest

from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_e2 as e2
from gencodoe import fourwire_scale_eval as ev
from gencodoe.fourwire_train import VMAX, VMIN

pytest.importorskip("opendssdirect")
pytest.importorskip("torch_geometric")
pytestmark = pytest.mark.skipif(not (dk.ROOT / "data_out/fluvius/geen_ZP.npz").exists()
                                and not any((dk.ROOT / "data/load_profiles").glob("P6269_*")),
                                reason="Fluvius meter profiles not available")


@pytest.fixture(scope="module")
def series():
    """The timed series of network 25 on the CPU, with the U1 screen and hand-set margins that leave part
    of the states uncertain (minimum V_pn below 0.98 pu or maximum above 1.013 pu)."""
    m = ev.margins_from({"lo": [-0.08, -0.04, 0.5, 0.5], "hi": [0.08, 0.04, 1.5, 1.5]})
    ser = e2.Series(25, "cpu", 2, {"u1": (None, None, m)})
    yield ser
    ser.pool.close()


def states(n=40):
    """The first val states of network 25 and three import states far beyond the transformer rating."""
    from gencodoe import csiro_convert as cc
    net, _ = fs._case(25)
    names, idx, share = cc.customers(net.customer_names)
    st = dk.customer_states(len(names), "val", n, fs.seed(25, "val"), fs.AUGMENT, fs.CAP_KVA)
    p, q = fs.connection_states(st, idx, share)
    s = p + 1j * q
    s[:3] = 2000.0 + 0j                                               # 42 customers x 2 MW: no solution
    return s


def test_every_series_delivers_the_same_exact_decisions(series):
    s = states()
    ref = series.opendss(s)
    assert ref[0].shape == (40, 4) and ref[1].shape == (40, 2) and ref[2].dtype == bool
    assert not ref[2][:3].any() and not ref[1][:3].any() and ref[2][3:].all()         # indeterminate: unsafe both ways
    table = series.table()
    for name in ("sweep_cpu", "sweep_gpu", "numpy"):
        series.opendss_calls = 0
        summ, safe, conv = table[name](s)
        assert series.opendss_calls == 3 and (conv == ref[2]).all(), name               # §3: non-converged to OpenDSS
        assert np.abs(summ[3:, :3] - ref[0][3:, :3]).max() < 1e-3, name                 # E1 agreement
        assert (safe == e2.decide(summ, conv)).all(), name
    series.opendss_calls = 0
    summ, safe, conv = table["sweep_cpu_noopendss"](s)
    assert series.opendss_calls == 0 and not conv[:3].any() and not safe[:3].any()      # declared unsafe, no solve
    a, b = table["sweep_cpu"](s), table["sweep_gpu"](s)
    assert np.abs(a[0][3:] - b[0][3:]).max() < 1e-9 and (a[1] == b[1]).all()


def test_decision_of_an_exact_engine():
    summ = np.array([[VMIN + 1e-6, VMAX - 1e-6, 0.99, 0.99], [VMIN - 1e-6, 1.0, 0.5, 0.5], [0.95, VMAX + 1e-6, 0.5, 0.5],
                     [0.95, 1.0, 1.01, 0.5], [0.95, 1.0, 0.5, 1.01], [0.95, 1.0, 0.5, 0.5]])
    conv = np.array([1, 1, 1, 1, 1, 0], bool)
    assert e2.decide(summ, conv).tolist() == [[True, True], [False, True], [True, False], [False, False],
                                              [False, False], [False, False]]


def test_screen_pipeline_solves_uncertain_states_once_for_both_directions(series, monkeypatch):
    s = states()
    exact = series.sweep_cpu(s)
    for engine in e2.ENGINES:
        series.opendss_calls = 0
        summ, safe, conv, fb = series.screened("u1", engine, s)
        u1 = series.pipes["u1"][0].predict(s)[0].numpy().astype(float)
        codes = ev.screen_codes(series.pipes["u1"][1], u1)
        assert (fb == (codes == ev.UNCERTAIN).any(1)).all() and 0 < fb.sum() < len(s)   # the union of both directions
        assert np.allclose(summ[fb], exact[0][fb], atol=1e-9) and (safe[fb] == exact[1][fb]).all()
        assert (conv[fb] == exact[2][fb]).all()
        assert np.allclose(summ[~fb], u1[~fb]) and (safe[~fb] == (codes[~fb] == ev.SAFE)).all() and conv[~fb].all()
        assert series.opendss_calls == int((fb & ~series.numba.solve(s, return_state=False)["converged"]).sum())
    monkeypatch.setitem(series.pipes, "u1", (series.pipes["u1"][0], None))             # no usable margins
    none = series.screened("u1", "cpu", s)
    assert none[3].all() and (none[1] == exact[1]).all() and np.allclose(none[0], exact[0])   # everything deferred


def test_speedup_takes_the_faster_variant_on_both_sides_and_bootstraps_each_series():
    rng = np.random.default_rng(1)
    t = {"sweep_cpu": 2.0 + 0.01 * rng.normal(size=100), "sweep_gpu": 3.0 + 0.01 * rng.normal(size=100),
         "pgnet_cpu": 1.6 + 0.01 * rng.normal(size=100), "pgnet_gpu": 1.0 + 0.01 * rng.normal(size=100)}
    t = {k: v.tolist() for k, v in t.items()}
    sp = e2.speedup(t, ("sweep_cpu", "sweep_gpu"), ("pgnet_cpu", "pgnet_gpu"))
    assert sp["baseline"] == "sweep_cpu" and sp["pipeline"] == "pgnet_gpu"
    assert np.isclose(sp["speedup"], np.median(t["sweep_cpu"]) / np.median(t["pgnet_gpu"]))
    assert sp["ci95"][0] < sp["speedup"] < sp["ci95"][1] and 1.9 < sp["ci95"][0] and sp["ci95"][1] < 2.1
    g = np.random.default_rng(0)                                                      # the frozen bootstrap, redone
    boot = {k: np.median(np.array(t[k])[g.integers(0, 100, (2000, 100))], axis=1) for k in t}
    ratio = np.minimum(boot["sweep_cpu"], boot["sweep_gpu"]) / np.minimum(boot["pgnet_cpu"], boot["pgnet_gpu"])
    assert np.allclose(sp["ci95"], [np.percentile(ratio, 2.5), np.percentile(ratio, 97.5)])


def test_envelope_days_and_scenarios_follow_amendment_9():
    days = e2.envelope_days()
    assert len(days) == 12 and str(np.datetime64("2024-01-01") + days[0]) == "2024-01-10"
    assert str(np.datetime64("2024-01-01") + days[-1]) == "2024-12-05"
    assert set(dk.day_split(np.array(days) * 96)) == {"test"} and set(dk.day_split(np.array(e2.envelope_days(True)) * 96)) == {"val"}
    a = list(e2.day_scenarios(25, 42, 3, hours=2))
    b = list(e2.day_scenarios(25, 42, 3, hours=2))
    assert [x[0] for x in a] == [3 * 96, 3 * 96 + 4]                                  # the first quarter of each hour
    assert a[0][1].shape == (200, 42) and all(np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    assert not np.array_equal(a[0][1][0], a[0][1][1])                                 # meters are redrawn per scenario
    other = next(e2.day_scenarios(10, 42, 3, hours=1))
    assert not np.array_equal(other[1], a[0][1])                                      # the network enters the seed
    off = a[0][1].clip(min=0)                                                         # reactive power at PF 0.95 of the offtake
    assert (a[0][2] >= 0).all() and (a[0][2][a[0][1] > 0] >= off[a[0][1] > 0] * dk.TAN_PHI - 1e-12).all()


def test_envelope_summary_counts_agreement_and_unsafe_envelopes():
    rows = [{"direction": "export", "A_level_kw": 3.0, "B_level_kw": 3.0 + 0.004},   # within 1e-3 x 5 kW
            {"direction": "export", "A_level_kw": 3.0, "B_level_kw": 3.0 + 0.006},   # unsafe: B above A
            {"direction": "import", "A_level_kw": 2.0, "B_level_kw": 2.0 - 0.008},   # B below A by more than 7.4e-3
            {"direction": "import", "A_level_kw": 0.0, "B_level_kw": 0.0}]
    s = e2.envelope_summary(rows)
    assert s["export"]["share_within_tolerance"] == 0.5 and s["export"]["unsafe_envelopes"] == 1
    assert s["import"]["share_within_tolerance"] == 0.5 and s["import"]["unsafe_envelopes"] == 0
    assert s["import"]["zero_envelopes_a"] == 1 and s["both"]["intervals"] == 4
    assert s["both"]["share_within_tolerance"] == 0.5 and not s["agreement_criterion_met"]
    assert e2.envelope_summary([rows[0]] * 19 + [rows[1]])["agreement_criterion_met"]   # 95% of the intervals


def test_uniform_envelope_of_the_exact_evaluator_is_the_largest_feasible_level():
    """One interval of network 25 with the ENWL1 rule and the Numba sweep: the level found is accepted, and
    a level one tolerance above it is not (or it is the cap)."""
    from gencodoe import csiro_convert as cc
    from gencodoe import envelopes as env
    from gencodoe.fourwire_sweep_opt import NumbaSweep
    net, th = fs._case(25)
    names, idx, share = cc.customers(net.customer_names)
    step, p_kw, q_kvar = next(e2.day_scenarios(25, len(names), 3, hours=1))
    sweep = NumbaSweep(net, th)

    def feasible(p, q, direction):
        o = sweep.solve(-(p[:, idx] * share) - 1j * (q[:, idx] * share), return_state=False)
        return e2.decide(o["summary"], o["converged"])[:, ev.DIRECTIONS.index(direction)]

    cust = env.Customers(bus=np.arange(len(names)), export_cap=np.full(len(names), 5.0), import_cap=np.full(len(names), 7.4))
    thr = env.acceptance_threshold(e2.N_SCENARIOS, e2.EPS, e2.BETA)
    assert thr == 0.98
    for direction, sign in (("export", 1.0), ("import", -1.0)):
        e = env.scalar_envelope(feasible, len(names), cust, -p_kw, -q_kvar, direction, "uniform", e2.EPS, e2.LEVELS,
                                e2.REL_TOL, beta=e2.BETA)
        cap = e2.CAP_KW[direction]
        share_at = lambda lv: feasible(-p_kw + sign * lv, -q_kvar, direction).mean()      # noqa: E731
        assert 0 <= e.level <= cap and share_at(e.level) >= thr
        assert e.level == cap or share_at(e.level + 2 * e2.REL_TOL * cap) < thr


def test_report_applies_the_criterion_of_section_7(tmp_path, monkeypatch):
    """Synthetic records of the six networks. GENCO is 'faster' only with a speed-up of at least 1.10, an
    interval above 1, and no false-safe and no cleared-indeterminate decision of its screen."""
    import json
    monkeypatch.setattr(fs, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    for d in ("results/fourwire_scale_e2_timing", "results/fourwire_scale_eval", "out/e2", "out/eval"):
        (tmp_path / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    buses = {25: 79, 10: 94, 1: 397, 7: 876, 2: 1089, 17: 1668}
    #          sweep (with OpenDSS), sweep alone, pipeline (with OpenDSS), pipeline alone, false-safe
    plan = {25: (1.0, 1.0, 3.0, 3.0, 0), 10: (1.0, 1.0, 3.0, 3.0, 0), 1: (4.0, 1.0, 2.0, 2.0, 0),
            7: (4.0, 1.0, 3.8, 2.0, 0), 2: (4.0, 1.0, 2.0, 2.0, 1), 17: (6.0, 1.0, 2.0, 2.0, 0)}
    for n, (sw, sw0, pg, pg0, false_safe) in plan.items():
        t = lambda m: (m * (1 + 0.01 * rng.normal(size=100))).tolist()     # noqa: E731
        series = {"sweep_cpu": sw, "sweep_gpu": 2 * sw, "sweep_cpu_noopendss": sw0, "sweep_gpu_noopendss": 2 * sw0,
                  "pgnet_cpu": 2 * pg, "pgnet_gpu": pg, "pgnet_cpu_noopendss": 2 * pg0, "pgnet_gpu_noopendss": pg0,
                  "u1_cpu": pg, "u1_gpu": pg, "u1_cpu_noopendss": pg0, "u1_gpu_noopendss": pg0, "opendss": 50.0}
        batches = {str(b): {k: {"times_s": t(m), "p50_s": m, "p95_s": 1.1 * m, "states_per_s": b / m}
                            for k, m in series.items()} for b in e2.BATCHES}
        screen = {"fallback_share": 0.1, "false_safe": false_safe, "cleared_indeterminate": 0, "false_unsafe": 0}
        rec = {"valid": True, "dry_run": False, "per_state_file": f"network_{n}.npz", "exact_sweep_vs_opendss": {},
               "methods": {k: {"screen": screen, "fallback_exact_engine": {}} for k in ("PG-net", "U1")}}
        f = tmp_path / f"results/fourwire_scale_eval/network_{n}.json"
        f.write_text(json.dumps(rec))
        codes = np.array([[ev.SAFE, ev.SAFE], [ev.UNCERTAIN, ev.SAFE], [ev.VIOLATED, ev.VIOLATED]], np.uint8)
        exact = np.array([[True, True], [False, False], [True, True]])
        np.savez(tmp_path / f"out/eval/network_{n}.npz", **{"PG-net__codes": codes, "exact_safe": exact})
        want = np.array([[True, True], [False, False], [False, False]])                # one solve serves both directions
        np.savez(tmp_path / f"out/e2/timing_network_{n}_decisions.npz", pgnet_cpu__safe=want, pgnet_gpu__safe=~want)
        (tmp_path / f"results/fourwire_scale_e2_timing/network_{n}.json").write_text(json.dumps({
            "valid": True, "dry_run": False, "retained_buses": buses[n], "allocated_cores": 16,
            "evaluation_record_sha256": fs._sha(f), "decisions_file": f"timing_network_{n}_decisions.npz",
            "repeats_with_other_decisions": {}, "batches": batches}))
    rep = json.loads(e2.report().read_text())
    by = {r["network"]: r for r in rep["networks"]}
    sp = by[1]["batches"]["4096"]["speedup"]["pgnet"]
    assert sp["baseline"] == "sweep_cpu" and sp["pipeline"] == "pgnet_gpu" and 1.9 < sp["ci95"][0] < 2.0 < sp["ci95"][1] < 2.1
    assert by[1]["criterion"]["genco_faster"] and by[17]["criterion"]["genco_faster"]
    assert not by[25]["criterion"]["genco_faster"] and not by[25]["criterion"]["speedup_ge_1.10"]
    assert by[7]["criterion"]["ci_excludes_1"] and not by[7]["criterion"]["speedup_ge_1.10"]      # 1.05x is not enough
    assert by[2]["criterion"]["speedup_ge_1.10"] and not by[2]["criterion"]["genco_faster"]       # one false-safe
    assert rep["networks_where_genco_is_faster"] == [1, 17] and rep["break_even_retained_buses"] == 1668
    assert rep["networks_where_it_is_faster_without_the_opendss_solves"] == []                   # the §3 rule made it
    assert by[1]["timed_decisions_vs_evaluation_record"] == {"pgnet_cpu": 0, "pgnet_gpu": 6}
    assert rep["loglog_exponent_of_p50_time_against_retained_buses"]["sweep_cpu"] > 0
    (tmp_path / "results/fourwire_scale_eval/network_1.json").write_text(json.dumps({**rec, "valid": True, "x": 1}))
    with pytest.raises(AssertionError, match="another evaluation record"):
        e2.report()
