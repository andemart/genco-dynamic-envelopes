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
