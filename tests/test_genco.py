from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gencodoe import datakit_fluvius as dkf
from gencodoe import evaluate, genco
from gencodoe.enwl import load_enwl1
from gencodoe.fluvius import STEPS

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data_out" / "datakit_calib"
needs_data = pytest.mark.skipif(not (CALIB / "enwl1_fluvius" / "raw").exists(),
                                reason="generated datakit data not present")


def test_day_splits_are_disjoint_whole_days():
    names = dkf.day_split(np.arange(STEPS)).reshape(366, 96)
    assert (names == names[:, :1]).all()                       # one split per calendar day
    trainval, calib, test = (set(dkf.split_steps(s)) for s in ("trainval", "calib", "test"))
    assert not (trainval & calib) and not (trainval & test) and not (calib & test)
    assert len(trainval) + len(calib) + len(test) == STEPS
    assert dkf.is_test_step(np.array(sorted(calib | test))).all()


def _fake_datakit(tmp_path, steps, solved):
    """Draw log over `steps` where only draws `solved` survived, renumbered like datakit."""
    pd.DataFrame({"scenario": np.arange(len(steps)), "step": steps}).to_parquet(tmp_path / "fluvius_draws.parquet")
    pd.DataFrame({"scenario": np.arange(len(solved)), "load_scenario_idx": solved}).to_parquet(
        tmp_path / "bus_data.parquet")
    return tmp_path / "fluvius_draws.parquet"


def test_graphkit_splits_reject_held_out_days(tmp_path):
    held = dkf.split_steps("test")[:3]
    with pytest.raises(ValueError):
        dkf.write_graphkit_splits(_fake_datakit(tmp_path, held, [0, 1, 2]), tmp_path / "splits")


def test_graphkit_splits_follow_datakit_renumbering(tmp_path):
    import torch
    tv = dkf.split_steps("trainval")
    names = dkf.day_split(tv)
    tr, va = tv[names == "train"][:2], tv[names == "val"][:2]
    steps = np.array([tr[0], va[0], tr[1], va[1]])          # draws 0..3
    draws = _fake_datakit(tmp_path, steps, solved=[0, 2, 3])  # draw 1 failed -> scenario ids 0, 1, 2
    sizes = dkf.write_graphkit_splits(draws, tmp_path / "splits")
    assert sizes == {"train": 2, "val": 1, "test": 1}
    assert torch.load(tmp_path / "splits" / "train.pt").tolist() == [0, 1]   # draws 0 and 2
    assert torch.load(tmp_path / "splits" / "val.pt").tolist() == [2]        # draw 3


def test_read_predictions_orders_by_scenario_and_bus(tmp_path):
    n, n_bus = 3, 4
    df = pd.DataFrame({"scenario": np.repeat(np.arange(n), n_bus), "bus": np.tile(np.arange(n_bus), n)})
    df["Vm_pred"] = 1 + 0.01 * df.scenario + 0.001 * df.bus
    df["Va_pred"] = -0.1 * df.bus
    df.sample(frac=1, random_state=0).to_parquet(tmp_path / "p.parquet")
    vm, va = genco.read_predictions(tmp_path / "p.parquet", n_bus)
    assert vm.shape == (n, n_bus)
    np.testing.assert_allclose(vm[2, 3], 1.023)
    np.testing.assert_allclose(va[1], -0.1 * np.arange(n_bus))


def test_score_is_zero_for_truth_and_flags_violations():
    fd = load_enwl1().to_feeder()
    rng = np.random.default_rng(0)
    p = np.zeros((4, fd.n_bus))
    p[:, 2:] = rng.uniform(-3e-3, 3e-3, (4, fd.n_bus - 2))
    v = evaluate._sweep_voltage(fd, p, 0.3 * p)
    truth = {"vm": np.abs(v), "va": np.angle(v), "pd": -p, "qd": -0.3 * p}
    loading = genco.branch_flows_kcl(fd, truth["vm"], truth["va"], p, 0.3 * p).loading
    r = evaluate.score("ac", fd, truth, truth["vm"], truth["va"], loading)
    assert r["vm_pu"]["max"] == 0 and r["loading"]["max"] == 0
    assert r["violation"]["fn_false_safe"] == 0
    flat = np.ones_like(truth["vm"])
    viol = evaluate.violations(flat * 0.9, loading)
    assert viol.all()


def _template():
    return genco.read_raw(CALIB / "enwl1_fluvius" / "raw", scenarios=[0, 1])


@needs_data
def test_fast_graph_path_matches_pandas_path():
    tmpl = _template()
    n_bus = tmpl[0].scenario.eq(0).sum()
    rng = np.random.default_rng(1)
    p, q = rng.normal(0, 2e-3, (3, n_bus)), rng.normal(0, 5e-4, (3, n_bus))
    slow = genco.frames_to_graphs(*genco.injections_to_datakit_format(tmpl, p, q))
    t0 = genco.frames_to_graphs(*(df[df.scenario == 0] for df in tmpl))[0]
    fast = genco.graphs_from_injections(t0, p, q)
    for a, b in zip(slow, fast):
        assert (a["bus"].x == b["bus"].x).all()
        et = ("bus", "connects", "bus")
        assert (a[et].edge_index == b[et].edge_index).all() and (a[et].edge_attr == b[et].edge_attr).all()


@needs_data
def test_template_graph_buses_and_branches_match_feeder():
    net = load_enwl1()
    bus, gen, branch = _template()
    b0, r0 = bus[bus.scenario == 0], branch[branch.scenario == 0]
    assert (b0.bus.to_numpy() == np.arange(net.n_bus)).all()
    assert (r0.from_bus.to_numpy() == net.from_bus).all() and (r0.to_bus.to_numpy() == net.to_bus).all()
    assert b0.REF.to_numpy()[0] == 1 and b0.REF.sum() == 1


@needs_data
def test_lv_edge_encoder_sees_impedance_magnitude_stock_does_not():
    import torch
    import yaml
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model

    from gencodoe import genco_model  # noqa: F401

    cfg = yaml.safe_load((ROOT / "configs" / "genco_pf_enwl1_tiny.yaml").read_text())
    g = genco.frames_to_graphs(*(df[df.scenario == 0] for df in _template()))[0]
    out = {}
    for typ in ("GNS_heterogeneous", "GNS_heterogeneous_lv"):
        cfg["model"]["type"] = typ
        args = NestedNamespace(**cfg)
        norm = HeteroDataMVANormalizer(args)
        norm.fit_from_dict({"baseMVA_orig": torch.tensor(1.0), "baseMVA": torch.tensor(0.0021),
                            "vn_kv_max": torch.tensor(0.416)})
        h = g.clone()
        norm.transform(h)
        h = get_task_transforms(args)(h)
        ea = h["bus", "connects", "bus"].edge_attr
        scale = torch.ones(ea.shape[1])
        scale[2:6] = 10.0
        torch.manual_seed(0)
        m = load_model(args).eval()
        with torch.no_grad():
            e1, e10 = m.input_proj_edge(ea), m.input_proj_edge(ea * scale)
            out[typ] = float((e1 - e10).abs().max() / e1.abs().max())
            from torch_geometric.data import Batch
            pred = m(Batch.from_data_list([h, h.clone()]))["bus"]
        assert torch.isfinite(pred).all()
    print(out)
    assert out["GNS_heterogeneous"] < 5e-3                               # LayerNorm cancels the magnitude
    assert out["GNS_heterogeneous_lv"] > 20 * out["GNS_heterogeneous"]


def _random_state(fd, n=4, seed=0, scale=3e-3):
    rng = np.random.default_rng(seed)
    p = np.zeros((n, fd.n_bus))
    p[:, 2:] = rng.uniform(-scale, scale, (n, fd.n_bus - 2))
    return p, 0.3 * p


def test_kcl_flows_exact_at_truth_and_robust_to_voltage_noise():
    from gencodoe import feeder
    fd = load_enwl1().to_feeder()
    p, q = _random_state(fd)
    ac = feeder.sweep_pf(fd, p, q)
    v = evaluate._sweep_voltage(fd, p, q)
    kcl = feeder.branch_flows_kcl(fd, np.abs(v), np.angle(v), p, q)
    np.testing.assert_allclose(kcl.loading, ac.loading, atol=1e-9)
    noisy = np.abs(v) + np.random.default_rng(1).normal(0, 1e-4, v.shape)
    err_kcl = np.abs(feeder.branch_flows_kcl(fd, noisy, np.angle(v), p, q).loading - ac.loading).max()
    err_dv = np.abs(feeder.branch_flows_from_voltages(fd, noisy, np.angle(v)).loading - ac.loading).max()
    assert err_kcl < 1e-3 < err_dv


def test_conformal_assess_counts_false_safe():
    from gencodoe import conformal_eval, feeder
    fd = load_enwl1().to_feeder()
    p, q = _random_state(fd, n=6, scale=8e-3)
    v = evaluate._sweep_voltage(fd, p, q)
    truth = {"vm": np.abs(v), "va": np.angle(v), "pd": -p, "qd": -q}
    pred = feeder.branch_flows_kcl(fd, truth["vm"], truth["va"], p, q)
    r = conformal_eval.assess(fd, truth, pred, 0.0, 0.0)              # perfect predictor, zero margins
    assert r["false_safe_states"] == 0 and r["coverage_vm"] == 1.0
    shifted = feeder.PFResult(vm=pred.vm + 0.2, loading=pred.loading * 0, p_branch=pred.p_branch)
    r = conformal_eval.assess(fd, truth, shifted, 0.0, 0.0)          # predicts overvoltage everywhere
    assert r["share_safe"] == 0.0


def test_doe_day_rejects_non_test_days():
    from gencodoe import doe_day
    with pytest.raises(ValueError):
        doe_day.run_day("2024-01-01", None, None, type("S", (), {"time": np.zeros(366 * 96)})(), {}, 1, 0.05, 1, 0)


@needs_data
def test_linear_prior_equals_first_sweep_iteration():
    import torch
    import yaml
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms
    from torch_geometric.data import Batch

    from gencodoe.genco_model import linear_prior

    truth = evaluate.load_truth(CALIB, range(8))
    args = NestedNamespace(**yaml.safe_load((ROOT / "configs" / "genco_pf_enwl1_tiny.yaml").read_text()))
    norm = HeteroDataMVANormalizer(args)
    norm.fit_from_dict({"baseMVA_orig": torch.tensor(1.0), "baseMVA": torch.tensor(0.0021),
                        "vn_kv_max": torch.tensor(0.416)})
    gs = []
    for g in genco.frames_to_graphs(*truth["frames"]):
        norm.transform(g)
        gs.append(get_task_transforms(args)(g))
    b = Batch.from_data_list(gs)
    et = ("bus", "connects", "bus")
    pr = linear_prior(b.x_dict["bus"], b.edge_index_dict[et], b.edge_attr_dict[et], b["bus"].batch, {})
    pr = pr.double().numpy().reshape(len(gs), -1, 2)
    fd = load_enwl1().to_feeder()
    v1 = evaluate._sweep_voltage(fd, -truth["pd"], -truth["qd"], max_iter=1)
    np.testing.assert_allclose(pr[:, :, 0], np.abs(v1), atol=2e-6)      # float32 features
    np.testing.assert_allclose(pr[:, :, 1], np.angle(v1), atol=2e-6)


def test_binned_conformal_covers_each_bin():
    from gencodoe import conformal
    rng = np.random.default_rng(0)
    pred = np.concatenate([rng.uniform(0.95, 1.05, 4000), rng.uniform(0.90, 0.94, 4000)])
    noise = np.where(pred < 0.94, 0.02, 0.002) * rng.standard_normal(pred.size)   # larger errors near the limit
    q = conformal.calibrate_binned(pred, pred + noise, alpha=0.05)
    test_pred = np.concatenate([rng.uniform(0.95, 1.05, 4000), rng.uniform(0.90, 0.94, 4000)])
    test_noise = np.where(test_pred < 0.94, 0.02, 0.002) * rng.standard_normal(test_pred.size)
    m = conformal.binned_margin(test_pred, q)
    near = test_pred < 0.94
    assert (np.abs(test_noise[near]) <= m[near]).mean() > 0.93           # a global quantile would under-cover here
    assert (np.abs(test_noise[~near]) <= m[~near]).mean() > 0.93


def test_best_model_callback_keeps_float_score():
    import torch
    from gencodoe import graphkit_cli  # noqa: F401  (applies the fix)
    from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

    cb = SaveBestModelStateDict(monitor="Validation loss")
    metric = torch.tensor(1.0)

    class _T:
        sanity_checking = False
        callback_metrics = {"Validation loss": metric}
        logger = type("L", (), {"save_dir": None})()

    saved = []
    import gridfm_graphkit.training.callbacks as cbm
    orig_save, cbm_os_makedirs = cbm.torch.save, cbm.os.makedirs
    cbm.torch.save = lambda obj, path: saved.append(path)
    cbm.os.makedirs = lambda *a, **k: None
    try:
        pl = type("M", (), {"state_dict": lambda self: {}})()
        _T.logger.save_dir = "/nonexistent"
        cb.on_validation_end(_T, pl)
        metric.fill_(0.5)                    # Lightning updates the tensor in place
        cb.on_validation_end(_T, pl)
    finally:
        cbm.torch.save, cbm.os.makedirs = orig_save, cbm_os_makedirs
    assert len(saved) == 2 and isinstance(cb.best_score, float) and cb.best_score == 0.5


def test_acceptance_threshold_bounds_false_acceptance():
    from gencodoe.envelopes import acceptance_threshold
    assert acceptance_threshold(200, 0.05, None) == 0.95
    t = acceptance_threshold(200, 0.05, 0.05)
    rng = np.random.default_rng(0)
    # a level whose true feasibility is just below target (0.949) must be accepted <= beta of the time
    acc = (rng.binomial(200, 0.949, 200_000) / 200 >= t).mean()
    assert acc <= 0.05
    assert acceptance_threshold(10, 0.05, 0.01) > 1.0      # nothing certifiable with 10 scenarios


@needs_data
def test_fast_injection_path_matches_graph_path_in_float64(tmp_path):
    import torch
    import yaml
    from gridfm_graphkit.io.param_handler import NestedNamespace, load_model

    from gencodoe import genco_model  # noqa: F401

    cfg = yaml.safe_load((ROOT / "configs" / "genco_pf_enwl1_lv_prior_v2.yaml").read_text())
    torch.manual_seed(0)
    m = load_model(NestedNamespace(**cfg))
    for prm in m.parameters():                      # non-trivial head so the correction is exercised
        prm.data.add_(0.01 * torch.randn_like(prm))
    torch.save(m.state_dict(), tmp_path / "m.pt")
    torch.save({"enwl1_fluvius": {"baseMVA_orig": torch.tensor(1.0), "baseMVA": torch.tensor(0.0021),
                                  "vn_kv_max": torch.tensor(0.416)}}, tmp_path / "s.pt")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    gm = genco.GencoModel(tmp_path / "c.yaml", tmp_path / "m.pt", tmp_path / "s.pt")
    gm.model.double()
    oracle = genco.GencoOracle(gm, _template(), batch_size=3)
    truth = evaluate.load_truth(CALIB, range(3))
    p, q = -truth["pd"], -truth["qd"]
    ch = []
    for g in genco.graphs_from_injections(oracle.template_graph, p, q):
        gm.normalizer.transform(g)
        g = gm.transform(g)
        for store in g.stores:
            for k, v in list(store.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    store[k] = v.double()
        ch.append(g)
    from torch_geometric.data import Batch
    with torch.no_grad():
        slow = gm.model(Batch.from_data_list(ch))["bus"][:, 0].reshape(3, -1).numpy()
    gm.prepare_template(oracle.template_graph, 3)
    b = gm._template[0]
    for store in b.stores:
        for k, v in list(store.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                store[k] = v.double()
    fast, _ = gm.predict_injections(-p, -q)
    np.testing.assert_allclose(fast, slow, atol=1e-5)


def test_monthly_average_peak_applies_floor_per_month():
    from gencodoe.contracts import monthly_average_peak
    month = np.repeat(np.arange(1, 13), 10)
    kw = np.zeros((1, 120))
    kw[0, month == 1] = 6.0       # one month at 6 kW, the rest near zero
    kw[0, month == 2] = 1.0
    assert np.isclose(monthly_average_peak(kw, month, 0.0)[0], 7.0 / 12)
    assert np.isclose(monthly_average_peak(kw, month, 2.5)[0], (6.0 + 11 * 2.5) / 12)


def test_four_wire_screen_directional_logic():
    from gencodoe.four_wire import FourWireMargins
    rng = np.random.default_rng(0)
    n = 4000
    pv, px, pl = rng.uniform(0.89, 1.0, n), rng.uniform(1.0, 1.05, n), rng.uniform(0.1, 0.5, n)
    pt = rng.uniform(0.1, 0.5, n)
    m = FourWireMargins.fit(pv, px, pl, pt, pv - 0.005, px + 0.005, pl * 1.1, pt * 1.05, np.ones(n, bool), alpha=0.05)
    from gencodoe.limits import VMIN_PU
    a = np.array
    low = VMIN_PU + 0.002          # within the 0.005 calibrated offset of the lower limit
    # a comfortably feasible state is safe in both directions; a near-undervoltage one is not for imports
    for d in ("import", "export", None):
        assert m.screen(a([0.99]), a([1.01]), a([0.2]), a([0.2]), d)[0] == "safe"
    assert m.screen(a([low]), a([1.01]), a([0.2]), a([0.2]), "import")[0] != "safe"
    assert m.screen(a([low]), a([1.01]), a([0.2]), a([0.2]), "export")[0] == "safe"   # export ignores Vmin
    # transformer loading near its rating blocks "safe" in every direction, even with good voltages and lines
    for d in ("import", "export"):
        assert m.screen(a([0.99]), a([1.01]), a([0.2]), a([0.98]), d)[0] != "safe"



def test_voltage_limits_are_absolute_uk_values_on_the_416v_base():
    from gencodoe import limits
    assert abs(limits.V_BASE - 240.177) < 1e-3
    if limits.JURISDICTION == "uk_esqcr":
        assert (limits.VMIN_V, limits.VMAX_V) == (216.2, 253.0)
        assert abs(limits.VMIN_PU - 0.9002) < 1e-4 and abs(limits.VMAX_PU - 1.0534) < 1e-4


def test_every_check_uses_the_same_voltage_limits():
    from gencodoe import conformal_eval, diagnose_baseline, envelopes, evaluate, four_wire, limits, unbalanced
    assert evaluate.VMIN == four_wire.VMIN == unbalanced.VMIN_PU == diagnose_baseline.VMIN_PU == limits.VMIN_PU
    assert evaluate.VMAX == four_wire.VMAX == unbalanced.VMAX_PU == limits.VMAX_PU
    assert conformal_eval.VMIN == limits.VMIN_PU and conformal_eval.VMAX == limits.VMAX_PU
    assert unbalanced.V_PN == limits.V_BASE
    fd = load_enwl1().to_feeder()
    assert fd.vmin == limits.VMIN_PU and fd.vmax == limits.VMAX_PU
    assert envelopes.is_feasible.__module__ == "gencodoe.envelopes"


def test_balanced_checks_at_the_voltage_limits():
    from gencodoe import envelopes, feeder, limits
    fd = load_enwl1().to_feeder()
    zero = np.zeros((1, fd.n_bus))
    for v, ok in ((limits.VMIN_PU + 1e-6, True), (limits.VMIN_PU - 1e-6, False),
                  (limits.VMAX_PU - 1e-6, True), (limits.VMAX_PU + 1e-6, False)):
        vm = np.ones((1, fd.n_bus))
        vm[0, 5] = v
        res = feeder.PFResult(vm=vm, loading=zero.copy(), p_branch=zero.copy())
        assert bool(envelopes.is_feasible(fd, res)[0]) is ok
        assert bool(evaluate.violations(vm, zero)[0]) is (not ok)


def test_four_wire_flags_at_the_voltage_limits():
    from gencodoe import limits
    from gencodoe.four_wire import four_wire_ok
    from gencodoe.unbalanced import voltage_flags
    base = np.ones(200)
    for v, uv, ov in ((limits.VMIN_PU + 1e-6, False, False), (limits.VMIN_PU - 1e-6, True, False),
                      (limits.VMAX_PU - 1e-6, False, False), (limits.VMAX_PU + 1e-6, False, True)):
        x = base.copy()
        x[7] = v
        assert voltage_flags(x) == (uv, ov)
        res = {"converged": True, "undervoltage": uv, "overvoltage": ov, "line_overload": False,
               "transformer_overload": False}
        assert four_wire_ok(res, "import") is (not uv)
        assert four_wire_ok(res, "export") is (not ov)


def _fake_run(tmp_path, name, method, limit_b, limit_c=None, jurisdiction=None, seed=101):
    from gencodoe import limits
    rows = []
    for direction in ("export", "import"):
        for step in range(96):
            for m, lim in (("ac", 2.0), (method, limit_b if method == "b_opendss_exact" else limit_c)):
                rows.append({"step": step, "time": "x", "direction": direction, "method": m, "limit_kw": lim,
                             "total_kw": 200 * lim, "prob_ok_oracle": 1.0, "prob_ok_ac_fresh": 0.99,
                             "prob_ok_unbalanced": 0.99, "unbalanced_draws": 200, "power_flows": 10,
                             "ac_calls": 7, "runtime_s": 1.0})
    vl = limits.record()
    if jurisdiction:
        vl["jurisdiction"] = jurisdiction
    meta = {"voltage_limits": vl, "day": "2024-06-28", "scenarios": 200, "verify_draws": 1000,
            "scenario_seed": seed, "household_seed": seed, "steps_per_hour": 4}
    path = tmp_path / name
    path.write_text(__import__("json").dumps({"meta": meta, "rows": rows}))
    return path


def test_report_final_metrics_and_integrity(tmp_path):
    from gencodoe import report_final
    from gencodoe import limits
    b = _fake_run(tmp_path, "b.json", "b_opendss_exact", 1.0)
    c = _fake_run(tmp_path, "c.json", "c_genco_4w_opendss", None, limit_c=1.0)
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU, "seed": 101}
    df, inputs = report_final.load_runs([b, c], expect)
    res = report_final.report(report_final.feasibility(df))
    agree = res["agreement"][0]
    assert agree["share_within_tol"] == 1.0 and agree["intervals_c_exceeds_b"] == 0
    cap = {(r["direction"], r["method"]): r for r in res["capacity"]}
    assert abs(cap[("export", "B")]["kwh_per_household"] - 1.0 * 0.25 * 96) < 1e-9
    # p = 0.99 with n = 200 -> one-sided 95% lower bound ~0.968 >= 0.95: all intervals certified
    assert res["safety"][0]["intervals_lb4_ge_0.95"] == 96
    assert abs(report_final.cp_lower(198, 200) - 0.9686) < 1e-3
    bad = _fake_run(tmp_path, "bad.json", "b_opendss_exact", 1.0, jurisdiction="old_0.94_1.10")
    with pytest.raises(SystemExit):
        report_final.load_runs([bad], expect)
    wrong_seed = _fake_run(tmp_path, "seed.json", "b_opendss_exact", 1.0, seed=0)
    with pytest.raises(SystemExit):
        report_final.load_runs([wrong_seed], expect)



def test_report_final_seed_fallback_requires_listing_in_manifest(tmp_path):
    import json
    from gencodoe import limits, report_final
    run = _fake_run(tmp_path, "2024-06-28_certified_4w_b_opendss_exact_sph4_seed101_uk_esqcr.json", "b_opendss_exact", 1.0)
    doc = json.loads(run.read_text())
    for k in ("scenario_seed", "household_seed"):
        doc["meta"].pop(k)
    run.write_text(json.dumps(doc))
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"jobs": {"1": {"command": f"x --out data_out/doe/{run.name}"}},
                               "seeds": {"scenario_and_verification_seed": 101,
                                         "household_technology_assignment_seed": 101}}))
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": 101, "manifest": man}
    df, inputs = report_final.load_runs([run], expect)
    assert inputs[0]["meta"]["_seed_source"] == "run manifest"
    other = tmp_path / "unlisted.json"
    other.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        report_final.load_runs([other], expect)


def test_report_annual_on_synthetic_days(tmp_path):
    import json
    from gencodoe import limits, report_annual
    from gencodoe.report_final import feasibility, load_runs
    files = []
    for day, lim in (("2024-01-10", 0.0), ("2024-04-09", 1.0), ("2024-07-08", 2.0), ("2024-10-17", 1.0)):
        for method in ("b_opendss_exact", "c_genco_4w_opendss"):
            rows = []
            for direction in ("export", "import"):
                for step in range(24):
                    for m, l in (("ac", 3.0), (method, lim)):
                        rows.append({"step": step, "time": "x", "direction": direction, "method": m, "limit_kw": l,
                                     "total_kw": 200 * l, "prob_ok_oracle": 1.0, "prob_ok_ac_fresh": 0.99,
                                     "prob_ok_unbalanced": 0.99, "unbalanced_draws": 200, "power_flows": 10,
                                     "ac_calls": 7 if m.startswith("c") else 10, "runtime_s": 1.0})
            meta = {"voltage_limits": limits.record(), "day": day, "scenarios": 200, "verify_draws": 1000,
                    "scenario_seed": 101, "household_seed": 101, "steps_per_hour": 1}
            f = tmp_path / f"{day}_{method}.json"
            f.write_text(json.dumps({"meta": meta, "rows": rows}))
            files.append(f)
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": 101, "manifest": None, "steps_per_hour": 1}
    df, _ = load_runs(files, expect)
    res = report_annual.annual(feasibility(df))
    exp = res["by_direction"]["export"]
    # daily kWh/household = 24 h x limit; mean over (0, 1, 2, 1) = 24 kWh/day -> x366
    assert abs(exp["B"]["kwh_per_household_year"] - 24 * 366) < 1e-6
    assert exp["annual_static_limit_kw"] == 0.0 and "relative_uplift_vs_annual_static" not in exp["B"]
    lo, hi = exp["B"]["ci95"]
    assert lo <= 24 * 366 <= hi
    assert exp["agreement_paired"]["share_within_tol"] == 1.0 and exp["agreement_paired"]["intervals_both_zero"] == 24
    assert exp["paired_cost"]["exact_calls_C"] < exp["paired_cost"]["exact_calls_B"]


# ----------------------------------------------------------------------------- C+ (post-hoc final gate)

class _MockOracle:
    """Exact-share stand-in: share(level) from a given function; counts calls like FourWireOracle."""

    def __init__(self, share_fn, canonical_fn=None):
        self.f, self.g, self.n = share_fn, canonical_fn or share_fn, 0

    def share(self, p, q, level_kw, direction, canonical=False):
        self.n += len(p)
        return float((self.g if canonical else self.f)(level_kw)), len(p)


def _crow(limit, oracle_share=0.98):
    return {"limit_kw": limit, "prob_ok_oracle": oracle_share, "ac_calls": 1000, "runtime_s": 5.0}


def test_cplus_gate_passes_and_keeps_c_level():
    from gencodoe import cplus
    p = q = np.zeros((200, 3))
    acc = cplus.Counter()
    out = cplus.gate_interval(_MockOracle(lambda L: 0.99), p, q, _crow(1.0), "import", 0.98, acc)
    assert out["gated"] and not out["repaired"] and out["limit_kw"] == 1.0
    assert out["gate_calls"] == 200 and out["repair_calls"] == 0 and out["diagnostic_calls"] == 0
    assert acc.calls == {"gate": 200}


def test_cplus_gate_fails_and_repairs_downward():
    from gencodoe import cplus
    p = q = np.zeros((200, 3))
    acc = cplus.Counter()
    share = lambda L: 0.99 if L <= 0.4 else 0.97          # largest acceptable level is 0.4
    out = cplus.gate_interval(_MockOracle(share), p, q, _crow(1.0), "import", 0.98, acc)
    assert out["repaired"] and out["failure_class"] == "screen_error"
    assert 0.4 - 1e-3 * 7.4 <= out["limit_kw"] <= 0.4 and out["prob_ok_oracle"] >= 0.98
    assert out["repair_calls"] > 0 and out["diagnostic_calls"] == 200
    assert acc.calls["gate"] == 200 and acc.calls["repair"] == out["repair_calls"] and acc.calls["diagnostic"] == 200
    assert not out["zero_infeasible_sentinel"]


def test_cplus_solver_noise_is_diagnostic_only():
    from gencodoe import cplus
    p = q = np.zeros((200, 3))
    out = cplus.gate_interval(_MockOracle(lambda L: 0.975 if L > 0.5 else 0.99, canonical_fn=lambda L: 0.98),
                              p, q, _crow(1.0), "export", 0.98, cplus.Counter())
    assert out["failure_class"] == "solver_history_noise" and out["repaired"] and out["limit_kw"] <= 0.5


def test_cplus_zero_infeasible_sentinel_and_zero_classes():
    from gencodoe import cplus
    p = q = np.zeros((200, 3))
    out = cplus.gate_interval(_MockOracle(lambda L: 0.90), p, q, _crow(1.0), "import", 0.98, cplus.Counter())
    assert out["limit_kw"] == 0.0 and out["zero_infeasible_sentinel"] and not out["zero_feasible"]
    z1 = cplus.gate_interval(_MockOracle(lambda L: 1.0), p, q, _crow(0.0, oracle_share=0.90), "import", 0.98, cplus.Counter())
    z2 = cplus.gate_interval(_MockOracle(lambda L: 1.0), p, q, _crow(0.0, oracle_share=0.99), "import", 0.98, cplus.Counter())
    assert not z1["gated"] and z1["zero_infeasible_sentinel"] and z1["zero_class_source"] == "C-screened design share"
    assert z2["zero_feasible"] and not z2["zero_infeasible_sentinel"]


def test_report_schema_for_b_and_c_unchanged_by_cplus(tmp_path):
    import json
    from gencodoe import limits, report_final
    b = _fake_run(tmp_path, "b.json", "b_opendss_exact", 1.0)
    c = _fake_run(tmp_path, "c.json", "c_genco_4w_opendss", None, limit_c=1.0)
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU, "seed": 101}
    base = report_final.report(report_final.feasibility(report_final.load_runs([b, c], expect)[0]))
    doc = json.loads(c.read_text())
    for r in doc["rows"]:
        if r["method"] == "c_genco_4w_opendss":
            r["method"] = "cplus_genco_4w_gate"
            r["limit_kw"] = 0.9
    cp = tmp_path / "cp.json"
    cp.write_text(json.dumps(doc))
    ext = report_final.report(report_final.feasibility(report_final.load_runs([b, c, cp], expect)[0]))
    assert ext["agreement"] == base["agreement"]                       # B-C agreement unchanged
    for key in ("safety", "capacity", "runtime"):
        old = [r for r in ext[key] if r["method"] in ("B", "C", "ac", "static_from_B")]
        assert old == base[key]
    assert "agreement_cplus" in ext and ext["agreement_cplus"][0]["intervals_cplus_exceeds_b"] == 0


@pytest.mark.skipif(not (ROOT / "data_out/doe/annual/2024-04-29_annual_4w_c_genco_4w_opendss_sph1_seed101_uk_esqcr.json").exists(),
                    reason="stored annual C run not present")
def test_cplus_catches_the_known_false_acceptance(tmp_path):
    """Integration (exact OpenDSS, ~1-2 min): 2024-04-29 13:00 import, C = 0.3647 kW vs B = 0.0308 kW."""
    import json
    from gencodoe import cplus
    src = ROOT / "data_out/doe/annual/2024-04-29_annual_4w_c_genco_4w_opendss_sph1_seed101_uk_esqcr.json"
    before = cplus.sha256(src)
    summary = cplus.gate_run(src, tmp_path, b_dir=ROOT / "data_out/doe/annual")
    assert cplus.sha256(src) == before
    rows = [r for r in json.loads(Path(summary["file"]).read_text())["rows"] if r["method"] == "cplus_genco_4w_gate"]
    bad = [r for r in rows if r["step"] == 11476 and r["direction"] == "import"][0]
    assert bad["repaired"] and bad["failure_class"] == "screen_error"
    assert abs(bad["limit_kw"] - bad["b_limit_kw"]) <= 1e-3 * 7.4 and bad["prob_ok_oracle"] >= 0.98
    for r in rows:                                           # consistency of every C+ row
        assert r["ac_calls"] == r["c_ac_calls"] + r["gate_calls"] + r["repair_calls"]
        assert abs(r["runtime_s"] - (r["c_runtime_s"] + r["gate_s"] + r["repair_s"])) < 1e-9
        assert r["limit_kw"] <= r["c_limit_kw"] + 1e-12
        if not r["repaired"]:
            assert r["limit_kw"] == r["c_limit_kw"]
        if r["gated"]:
            assert r["gate_calls"] == 200



def test_report_final_seed_fallback_follows_cplus_provenance(tmp_path):
    import json
    from gencodoe import limits, report_final
    src_name = "2024-06-28_certified_4w_c_genco_4w_opendss_sph4_seed101_uk_esqcr.json"
    run = _fake_run(tmp_path, "derived_cplus.json", "b_opendss_exact", 1.0)
    doc = json.loads(run.read_text())
    for k in ("scenario_seed", "household_seed"):
        doc["meta"].pop(k)
    doc["meta"]["source_run"] = f"data_out/doe/{src_name}"
    run.write_text(json.dumps(doc))
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"jobs": {"1": {"command": f"x --out data_out/doe/{src_name}"}},
                               "seeds": {"scenario_and_verification_seed": 101, "household_technology_assignment_seed": 101}}))
    expect = {"jurisdiction": limits.JURISDICTION, "vmin_pu": limits.VMIN_PU, "vmax_pu": limits.VMAX_PU,
              "seed": 101, "manifest": man}
    _, inputs = report_final.load_runs([run], expect)
    assert inputs[0]["meta"]["_seed_source"] == "run manifest"
    doc["meta"]["source_run"] = "data_out/doe/unlisted.json"
    run.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        report_final.load_runs([run], expect)


def test_three_phase_mask_and_spatial_groups():
    pytest.importorskip("opendssdirect")
    from gencodoe.doe_day import customer_groups
    from gencodoe.enwl import load_enwl1
    from gencodoe.unbalanced import UnbalancedENWL1
    net = load_enwl1()
    single = UnbalancedENWL1()
    p = np.full(len(single.load_bus), 1.7)
    q = np.zeros_like(p)
    base = single.solve(p, q)
    same = UnbalancedENWL1(three_phase=np.zeros(len(p), bool)).solve(p, q)
    assert np.isclose(base["v_pn_min"], same["v_pn_min"]) and np.isclose(base["line_loading_max"],
                                                                      same["line_loading_max"])
    three = UnbalancedENWL1(three_phase=np.ones(len(p), bool)).solve(p, q)
    assert three["line_loading_max"] < base["line_loading_max"] and three["v_pn_min"] > base["v_pn_min"]
    g = customer_groups(net, single)
    assert len(np.unique(g)) == 12 and len(g) == len(net.customer_bus)


def test_fresh_four_wire_engine_survives_divergent_first_solve():
    """Regression: a fresh engine whose first solve diverges used to stay broken (all later solves infeasible)."""
    pytest.importorskip("opendssdirect")
    from gencodoe.unbalanced import UnbalancedENWL1
    u = UnbalancedENWL1()
    n = len(u.load_names)
    first = u.solve(np.full(n, -5.0), np.zeros(n))          # every customer exporting 5 kW
    normal = u.solve(np.full(n, 0.7), np.zeros(n))          # an ordinary evening load
    assert normal["converged"] and np.isfinite(normal["transformer_loading"])
    assert 0.9 < normal["v_pn_min"] < 1.06 and normal["line_loading_max"] < 1.0
    del first
