"""E2 timing series, statistics and the envelope check of the 25-network study
(docs/fourwire_scale_protocol.md §6, §7 E2, Amendment 9). Network 25, train and val states only."""
import numpy as np
import pytest

from test_fourwire_scale_eval import fit_all, study  # noqa: F401  (the tiny two-network study)

from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_e2 as e2
from gencodoe import fourwire_scale_eval as ev
from gencodoe import fourwire_scale_train as tt
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
    ser = e2.Series(25, "cpu", 2, {"u1": {"record": None, "state_dict": None, "margins": m, "model": True}})
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
        assert (safe == ref[1]).all(), name                                             # the decisions OpenDSS delivers
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
    for pipe in (series.pipes["u1"][0], None):                                         # no usable margins; no model
        monkeypatch.setitem(series.pipes, "u1", (pipe, None))
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
    rounds = np.random.default_rng(0).integers(0, 100, (2000, 100))                   # whole rounds resampled together
    boot = {k: np.median(np.array(t[k])[rounds], axis=1) for k in t}
    ratio = np.minimum(boot["sweep_cpu"], boot["sweep_gpu"]) / np.minimum(boot["pgnet_cpu"], boot["pgnet_gpu"])
    assert np.allclose(sp["ci95_paired_rounds"], [np.percentile(ratio, 2.5), np.percentile(ratio, 97.5)])
    assert sp["ci95_paired_rounds"] != sp["ci95"]


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


def _e2_records(tmp, plan, later_false_safe=()):
    """Synthetic final records of the six networks under one freeze: evaluation, timing and envelope records,
    each in results/ and in its run folder, with their per-state files and hashes.
    plan: network -> (sweep with OpenDSS, sweep alone, pipeline with OpenDSS, pipeline alone, screen false-safe,
    a false-safe in what the faster timed variant delivered in its first repeat)."""
    import hashlib
    import json
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()         # noqa: E731
    for d in ("results/fourwire_scale_e2_timing", "results/fourwire_scale_e2_envelopes", "results/fourwire_scale_eval"):
        (tmp / d).mkdir(parents=True)
    freeze = tmp / "results/fourwire_scale_freeze.json"
    freeze.write_text(json.dumps({"earlier_freezes_accepted_for": {"fits": [], "evaluation": []}}))
    rng = np.random.default_rng(0)
    buses = {25: 79, 10: 94, 1: 397, 7: 876, 2: 1089, 17: 1668}

    def put(kind_dir, folder, name, record):
        text = json.dumps(record)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "record.json").write_text(text)
        (tmp / "results" / kind_dir / name).write_text(text)

    for n, (sw, sw0, pg, pg0, false_safe, delivered_false_safe) in plan.items():
        t = lambda m: (m * (1 + 0.01 * rng.normal(size=100))).tolist()     # noqa: E731
        series = {"sweep_cpu": sw, "sweep_gpu": 2 * sw, "sweep_cpu_noopendss": sw0, "sweep_gpu_noopendss": 2 * sw0,
                  "pgnet_cpu": 2 * pg, "pgnet_gpu": pg, "pgnet_cpu_noopendss": 2 * pg0, "pgnet_gpu_noopendss": pg0,
                  "u1_cpu": pg, "u1_gpu": pg, "u1_cpu_noopendss": pg0, "u1_gpu_noopendss": pg0, "opendss": 50.0}
        batches = {str(b): {k: {"times_s": t(m), "p50_s": m, "p95_s": 1.1 * m, "states_per_s": b / m}
                            for k, m in series.items()} for b in e2.BATCHES}
        screen = {"fallback_share": 0.1, "false_safe": false_safe, "cleared_indeterminate": 0, "false_unsafe": 0}
        codes = np.array([[ev.SAFE, ev.SAFE], [ev.UNCERTAIN, ev.SAFE], [ev.VIOLATED, ev.VIOLATED]], np.uint8)
        exact = np.array([[True, True], [False, False], [True, True]])
        truth = np.array([[True, True], [False, False], [False, False]])               # the third state: not converged
        ef = tmp / f"out/eval/network_{n}"
        ef.mkdir(parents=True)
        np.savez(ef / "per_state.npz", **{"PG-net__codes": codes, "exact_safe": exact, "truth_safe": truth,
                                          "converged": np.array([1, 1, 0], bool)})
        put("fourwire_scale_eval", ef, f"network_{n}.json", {
            "network": n, "valid": True, "dry_run": False, "freeze_sha256": sha(freeze), "per_state_file": "per_state.npz",
            "per_state_sha256": sha(ef / "per_state.npz"), "exact_sweep_vs_opendss": {},
            "methods": {k: {"screen": screen, "fallback_exact_engine": {}} for k in ("PG-net", "U1")}})
        ev_sha = sha(tmp / f"results/fourwire_scale_eval/network_{n}.json")
        want = np.array([[True, True], [False, False], [False, False]])                # one solve serves both directions
        arrays = {}
        for k in series:
            first = ~truth if k.startswith("sweep") else want.copy()
            if k.startswith("pgnet_gpu"):
                first[1, 0] = bool(delivered_false_safe)                               # clears a state OpenDSS flags
            ever = first.copy()
            if k.startswith("pgnet_gpu") and n in later_false_safe:
                ever[1, 1] = True                                                      # only a later repeat cleared it
            arrays.update({f"{k}__safe": first, f"{k}__ever_safe": ever, f"{k}__always_safe": first})
        tf = tmp / f"out/e2/timing_network_{n}"
        tf.mkdir(parents=True)
        np.savez(tf / "decisions.npz", **arrays)
        put("fourwire_scale_e2_timing", tf, f"network_{n}.json", {
            "network": n, "valid": True, "dry_run": False, "freeze_sha256": sha(freeze), "retained_buses": buses[n],
            "allocated_cores": 16, "evaluation_record_sha256": ev_sha, "decisions_file": "decisions.npz",
            "decisions_sha256": sha(tf / "decisions.npz"), "repeats_with_other_decisions": {}, "batches": batches})
        put("fourwire_scale_e2_envelopes", tmp / f"out/e2/envelopes_network_{n}", f"network_{n}.json", {
            "network": n, "valid": True, "dry_run": False, "freeze_sha256": sha(freeze), "complete": True,
            "evaluation_record_sha256": ev_sha, "summary": {"agreement_criterion_met": True},
            "evaluator_b": {"import": {"fallback_share": 0.2}, "export": {"fallback_share": 0.1}}})


PLAN = {25: (1.0, 1.0, 3.0, 3.0, 0, 0), 10: (4.0, 1.0, 2.0, 2.0, 0, 1), 1: (4.0, 1.0, 2.0, 2.0, 0, 0),
        7: (4.0, 1.0, 3.8, 2.0, 0, 0), 2: (4.0, 1.0, 2.0, 2.0, 1, 0), 17: (6.0, 1.0, 2.0, 2.0, 0, 0)}


def test_report_applies_the_criterion_of_section_7(tmp_path, monkeypatch):
    """GENCO is 'faster' only with a speed-up of at least 1.10, an interval above 1, no false-safe and no
    cleared-indeterminate decision of its screen, and none in what the faster timed variant delivered."""
    import json
    monkeypatch.setattr(fs, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    _e2_records(tmp_path, PLAN)
    rep = json.loads(e2.report().read_text())
    by = {r["network"]: r for r in rep["networks"]}
    sp = by[1]["batches"]["4096"]["speedup"]["pgnet"]
    assert sp["baseline"] == "sweep_cpu" and sp["pipeline"] == "pgnet_gpu" and 1.9 < sp["ci95"][0] < 2.0 < sp["ci95"][1] < 2.1
    assert "ci95_paired_rounds" in sp
    assert by[1]["criterion"]["faster"] and by[17]["criterion"]["faster"]
    assert not by[25]["criterion"]["faster"] and not by[25]["criterion"]["speedup_ge_1.10"]
    assert by[7]["criterion"]["ci_excludes_1"] and not by[7]["criterion"]["speedup_ge_1.10"]      # 1.05x is not enough
    c = by[2]["criterion"]                                                                        # a screen false-safe
    assert c["speedup_ge_1.10"] and not c["screen_no_false_safe_no_cleared_indeterminate"] and not c["faster"]
    c = by[10]["criterion"]                                                 # a false-safe in what the timed variant delivered
    assert c["screen_no_false_safe_no_cleared_indeterminate"] and not c["delivered_no_false_safe_no_cleared_indeterminate"]
    assert c["speedup_ge_1.10"] and not c["faster"]
    d = by[10]["delivered_decisions_vs_opendss"]
    assert d["pgnet_gpu"]["false_safe"] == 1 and d["pgnet_gpu"]["by_direction"]["import"]["false_safe"] == 1
    assert d["pgnet_cpu"] == {**d["pgnet_cpu"], "false_safe": 0, "cleared_indeterminate": 0, "false_unsafe": 0}
    assert d["sweep_cpu"]["cleared_indeterminate"] == 2 and d["sweep_cpu"]["false_safe"] == 2 and d["sweep_cpu"]["false_unsafe"] == 2
    assert rep["networks_where_genco_is_faster"] == [1, 17] and rep["break_even_retained_buses"] == 1668
    assert rep["networks_where_it_is_faster_without_the_opendss_solves"] == []                   # the §3 rule made it
    assert rep["networks_where_the_u1_screen_is_faster_by_the_same_rule"] == [10, 1, 17]
    assert by[1]["first_timed_repeat_vs_evaluation_record"] == {"pgnet_cpu": 0, "pgnet_gpu": 0}
    assert by[10]["first_timed_repeat_vs_evaluation_record"] == {"pgnet_cpu": 0, "pgnet_gpu": 1}
    assert by[1]["envelopes"]["evaluator_b"]["import"]["fallback_share"] == 0.2
    assert rep["loglog_exponent_of_p50_time_against_retained_buses"]["sweep_cpu"] > 0


def test_report_counts_a_state_cleared_in_any_timed_repeat(tmp_path, monkeypatch):
    """Amendment 9, item 6: the delivered decisions are scored over every timed repeat, not the first one."""
    import json
    monkeypatch.setattr(fs, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    _e2_records(tmp_path, PLAN, later_false_safe=(17,))
    by = {r["network"]: r for r in json.loads(e2.report().read_text())["networks"]}
    assert by[17]["first_timed_repeat_vs_evaluation_record"]["pgnet_gpu"] == 0           # the first repeat was clean
    assert by[17]["delivered_decisions_vs_opendss"]["pgnet_gpu"]["by_direction"]["export"]["false_safe"] == 1
    assert not by[17]["criterion"]["delivered_no_false_safe_no_cleared_indeterminate"] and not by[17]["criterion"]["faster"]
    assert by[1]["criterion"]["faster"]


def test_report_refuses_records_and_files_that_are_not_the_recorded_ones(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(fs, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    _e2_records(tmp_path, PLAN)
    z = tmp_path / "out/e2/timing_network_7/decisions.npz"
    kept = z.read_bytes()
    np.savez(z, x=np.zeros(1))                                        # a per-state file that was replaced
    with pytest.raises(AssertionError, match="does not have the hash of its record"):
        e2.report()
    z.write_bytes(kept)
    f = tmp_path / "results/fourwire_scale_e2_envelopes/network_2.json"
    kept = f.read_text()
    f.write_text(kept.replace('"complete": true', '"complete": false'))   # an edited copy of a record
    with pytest.raises(AssertionError, match="differs from the record in its run folder"):
        e2.report()
    f.write_text(kept)
    g = tmp_path / "results/fourwire_scale_freeze.json"
    g.write_text(g.read_text() + " ")                                 # records of another freeze
    with pytest.raises(AssertionError, match="another freeze"):
        e2.report()


def test_timing_and_envelope_runs_end_to_end_on_val_states(study, monkeypatch):
    """The dry timing and envelope commands on the tiny study (val states and a val day only), and what a
    final run refuses."""
    import json
    root = study["root"]
    fit_all(root, {"n_train": 12, "n_val": 12, "epochs": 1})
    ev.evaluate(root, 25, "cpu", dry=True)
    out = e2.timing(25, "cpu", dry=True, warmup=1, repeats=3, batches=(1, 4, 4096), opendss_large=(0, 2))
    rec = json.loads(out.read_text())
    order = ["opendss", "numpy", "sweep_cpu", "sweep_gpu", "u1_cpu", "u1_gpu", "pgnet_cpu", "pgnet_gpu",
             "sweep_cpu_noopendss", "sweep_gpu_noopendss", "u1_cpu_noopendss", "u1_gpu_noopendss",
             "pgnet_cpu_noopendss", "pgnet_gpu_noopendss"]
    assert rec["series_order"] == order and rec["dry_run"] and rec["valid"] and rec["states_from"] == "val"
    assert set(rec["batches"]) == {"1", "4", "12"}                                    # the val split here has 12 states
    assert set(rec["batches"]["1"]) == set(order) and "numpy" not in rec["batches"]["4"]
    assert set(rec["batches"]["12"]) == set(order) - {"numpy"}
    assert all(v["repeats"] == 3 for k, v in rec["batches"]["12"].items() if k != "opendss")
    assert rec["batches"]["12"]["opendss"]["repeats"] == 2 and rec["deterministic_algorithms"] is False
    z = np.load(out.parent / rec["decisions_file"])
    for k in set(order) - {"numpy"}:                                                  # per state, over every timed repeat
        assert z[f"{k}__safe"].shape == z[f"{k}__ever_safe"].shape == z[f"{k}__always_safe"].shape == (12, 2)
        assert (z[f"{k}__always_safe"] <= z[f"{k}__safe"]).all() and (z[f"{k}__safe"] <= z[f"{k}__ever_safe"]).all()
    assert z["pgnet_cpu__fallback"].shape == (12,) and "sweep_cpu__fallback" not in z.files
    assert (z["sweep_cpu__safe"] == z["opendss__safe"]).all()
    assert rec["repeats_with_other_decisions"]["sweep_cpu"] == 0

    env_out = e2.envelopes(25, "cpu", dry=True, days=1, hours=1)
    envl = json.loads(env_out.read_text())
    assert [r["direction"] for r in envl["intervals"]] == ["import", "export"] and not envl["complete"]
    assert set(envl["evaluator_b"]) == {"import", "export"}                           # the fallback share, per direction
    assert all(0 <= d["fallback_share"] <= 1 and d["states"] > 0 for d in envl["evaluator_b"].values())
    assert envl["summary"]["both"]["intervals"] == 2 and envl["pgnet_has_model"]
    for r in envl["intervals"]:
        assert 0 <= r["A_level_kw"] <= e2.CAP_KW[r["direction"]] and 0 <= r["B_level_kw"] <= e2.CAP_KW[r["direction"]]

    with pytest.raises(RuntimeError, match="needs 16 cores"):                         # a final timing run: 16 cores
        e2.timing(25, "cpu")
    monkeypatch.setattr(e2, "E2_CORES", min(e2.allocated_cores(), 2))
    monkeypatch.setattr(e2, "allocated_cores", lambda: e2.E2_CORES)
    for run in (lambda: e2.timing(25, "cpu"), lambda: e2.envelopes(25, "cpu")):       # and only after all evaluations
        with pytest.raises(RuntimeError, match="no valid evaluation record yet"):
            run()
    assert not (fs.OUT / "e2").exists() and not tt.halt_file().exists()               # nothing claimed, nothing halted
    import torch
    torch.use_deterministic_algorithms(False)
