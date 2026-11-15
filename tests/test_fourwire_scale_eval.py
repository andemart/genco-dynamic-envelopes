"""Final fits, the device pipeline, the screen and the evaluation of the 25-network study
(docs/fourwire_scale_protocol.md §4-§8, Amendments 7 and 8). Tiny label sets of networks 10 and 25."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_eval as fe1
from gencodoe import fourwire_model as fm
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_data as fd
from gencodoe import fourwire_scale_eval as ev
from gencodoe import fourwire_scale_train as tt
from gencodoe.fourwire_net import V_PN_NOM

pytest.importorskip("opendssdirect")
pytest.importorskip("torch_geometric")
pytestmark = pytest.mark.skipif(not (dk.ROOT / "data_out/fluvius/geen_ZP.npz").exists()
                                and not any((dk.ROOT / "data/load_profiles").glob("P6269_*")),
                                reason="Fluvius meter profiles not available")
COMMIT = "0" * 40
HEAD = "a" * 40
STATES = {"train": 24, "val": 12, "calib": 12, "test": 12}


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A two-network study: network 10 is the pretraining pool of fold 0, network 25 its target. Labels of
    train, val and calib for both, the held-out split for network 25, a frozen clean checkout, one epoch."""
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    monkeypatch.setattr(fs, "N_STATES", STATES)
    monkeypatch.setattr(fs, "SHARD", 12)
    monkeypatch.setattr(fs, "_provenance", lambda: {"code_commit": COMMIT, "src_dirty": False})
    units = [(n, s) for n in (10, 25) for s in ("train", "val", "calib")] + [(25, "test")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"states": STATES, "shard_states": 12, "label_code_commits": [COMMIT],
                                    "splits": [fs.label_split(u) for u in units]}))
    monkeypatch.setattr(fd, "MANIFEST", manifest)
    monkeypatch.setattr(fd, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(fd, "LABEL_COMMIT", COMMIT)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/fourwire_scale_identification.json").write_bytes(
        (fs.RESULTS / "fourwire_scale_identification.json").read_bytes())      # the frozen Thevenin record
    monkeypatch.setattr(fs, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fs, "ELIGIBLE", [10, 25])
    monkeypatch.setattr(ev, "FRESH", [25])
    monkeypatch.setattr(tt, "folds", lambda: [[25], [10], [], []])
    monkeypatch.setattr(tt, "N_PRE", 24)
    monkeypatch.setattr(tt, "BUDGETS", {12: [0, 1]})
    monkeypatch.setattr(tt, "FITS", {k: {**v, "epochs": 2} for k, v in tt.FITS.items()})
    state = {"head": HEAD, "src_dirty": False, "source_sha256": "s" * 64}
    monkeypatch.setattr(tt, "checkout_state", lambda: dict(state))
    monkeypatch.setenv("GENCODOE_COMMIT", HEAD)
    tt.freeze()
    return {"root": tmp_path / "out/labels", "state": state, "tmp": tmp_path}


def fit_all(root, dry=None):
    """Every fit that network 25 needs: PG-pre of its fold (three seeds), PG-net, PG-scratch and PG-ft."""
    for a in tt.all_fits():
        if a.get("fold", 0) != 0 or a.get("network", 25) != 25:
            continue
        tt.final_fit(root, a["kind"], "cpu", a.get("fold"), a.get("network"), a.get("budget"), a["seed"], dry)


def test_device_pipeline_follows_the_training_path(study):
    store = fd.LabelStore(study["root"])
    st = fm.Static(*fs._case(25))
    d = store.load(25, "val")
    s = ev.connection_power(st, d)
    dn = ev.DeviceNet(st, "cpu")
    v1 = fm.prior_u1(st, s.real, s.imag)
    assert np.abs(dn.u1(torch.as_tensor(s)).numpy() - v1).max() / V_PN_NOM < 1e-12     # U1 on the device
    summ, v, cur = ev.Pipeline(dn).predict_host(s, state=True)                       # the U1 prior alone
    assert np.abs(v - v1).max() / V_PN_NOM < 1e-6
    assert np.allclose(summ, fe1.summaries4(st, v1, s), rtol=1e-5, atol=1e-6)
    assert np.allclose(ev.root_proxy(v, cur), summ[:, 3], rtol=1e-5)

    res = tt.fit(store, [25], 1, "cpu", out_dir=study["tmp"] / "run")                 # a model, then both paths
    sd = torch.load(study["tmp"] / "run/best.pt", weights_only=True)
    pipe = ev.Pipeline(dn, res, sd, microbatch=5)
    v_pipe = pipe.predict_host(s, state=True)[1]
    val = tt.prepare(store, st, 25, "val", None)                                      # fit's own inputs
    x = tt.scaler_from(res["scalers"]["x"])(val["x"])
    b = fm.Batcher(st, tt.scaler_from(res["scalers"]["edge"])(st.edge_raw), torch.device("cpu"))
    with torch.no_grad():
        out = pipe.model(torch.as_tensor(x).reshape(len(x) * st.net.n_bus, -1), *b.graph(len(x)), len(x))
    out = out.view(len(x), st.net.n_bus, 8) * torch.as_tensor(np.array(res["scalers"]["residual"], np.float32))
    v_fit = val["v1"] + torch.complex(out[..., :4], out[..., 4:]).numpy() * V_PN_NOM
    assert np.abs(v_pipe - v_fit).max() / V_PN_NOM < 2e-5 and np.abs(v_fit - val["v1"]).max() > 0


def test_decision_metrics_follow_amendment_8():
    #        import-unsafe, export-unsafe, both safe, cable overload, indeterminate, both safe
    flags = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bool)
    conv = np.array([1, 1, 1, 1, 0, 1], bool)
    S, V, U = ev.SAFE, ev.VIOLATED, ev.UNCERTAIN
    codes = np.array([[S, S], [U, S], [V, S], [S, U], [S, V], [U, U]], np.uint8)
    m = ev.decision_metrics(codes, flags, conv)
    assert m["import"] == {"cleared": 3, "rejected": 1, "fallback": 2, "false_safe": 2, "cleared_indeterminate": 1,
                           "false_unsafe": 1, "converged_not_safe": 2}
    assert m["export"]["false_safe"] == 1 and m["export"]["cleared_indeterminate"] == 0
    assert m["export"]["false_unsafe"] == 0 and m["export"]["converged_not_safe"] == 2
    assert m["false_safe"] == 3 and m["cleared_indeterminate"] == 1 and m["false_unsafe"] == 1
    assert np.isclose(m["fallback_share"], (2 / 6 + 2 / 6) / 2) and np.isclose(m["false_safe_rate"], 3 / 4)
    from scipy.stats import binom
    up = ev.clopper_pearson_upper(0, 50)
    assert np.isclose(binom.cdf(0, 50, up), 0.05) and np.isclose(up, 1 - 0.05 ** (1 / 50))
    assert ev.clopper_pearson_upper(0, 0) is None and ev.clopper_pearson_upper(3, 3) == 1.0


def test_screen_is_the_enwl1_margins_and_defers_when_unusable():
    rng = np.random.default_rng(0)
    truth = np.column_stack([rng.uniform(0.9, 1.0, 800), rng.uniform(1.0, 1.06, 800), rng.uniform(0.2, 1.2, 800),
                             rng.uniform(0.2, 1.2, 800)])
    pred = (truth * (1 + rng.normal(0, 0.002, truth.shape))).astype(np.float32)
    conv = rng.uniform(size=800) > 0.1
    m = ev.fit_margins(pred, truth, conv)
    ref = fe1.GlobalMargins(pred, truth, conv, alpha=0.01)
    assert m.lo == ref.lo and m.hi == ref.hi and np.isfinite(m.hi).all()
    codes = ev.screen_codes(m, pred)
    for j, d in enumerate(ev.DIRECTIONS):
        want = ref.screen(pred, d)
        assert ((codes[:, j] == ev.SAFE) == (want == "safe")).all() and ((codes[:, j] == ev.VIOLATED) == (want == "violated")).all()
    assert set(np.unique(codes)) == {ev.SAFE, ev.VIOLATED, ev.UNCERTAIN}
    again = ev.margins_from(json.loads(json.dumps(ev.margins_state(m))))             # through a record
    assert (ev.screen_codes(again, pred) == codes).all()
    few = ev.fit_margins(pred[:398], truth[:398], np.ones(398, bool))                # fewer than 399 states
    assert np.isinf(few.hi).all() and (ev.screen_codes(few, pred) != ev.SAFE).all()
    assert np.isfinite(ev.fit_margins(pred[:399], truth[:399], np.ones(399, bool)).hi).all()
    bad = pred.copy()
    bad[np.flatnonzero(conv)[0], 2] = np.nan                                         # one non-finite prediction
    assert ev.fit_margins(bad, truth, conv) is None
    assert (ev.screen_codes(None, pred) == ev.UNCERTAIN).all()
    one = pred.copy()
    one[3] = np.nan                                                                  # at evaluation time: defers
    assert (ev.screen_codes(m, one)[3] == ev.UNCERTAIN).all()


def test_fit_rules_of_amendment_8(study):
    store = fd.LabelStore(study["root"])
    pre = tt.fit(store, [10], 2, "cpu", out_dir=study["tmp"] / "pre")
    assert pre["best_epoch"] >= 1 and pre["curve"][0]["epoch"] == 0                   # epoch 0 is not eligible
    assert pre["config"]["lr"] == 1e-3 and pre["config"]["patience"] == 8 and pre["scheduler_t_max"] == 2
    init = {"state_dict": torch.load(study["tmp"] / "pre/best.pt", weights_only=True), "scalers": pre["scalers"]}
    ft = tt.fit(store, [25], 1, "cpu", n_train=12, lr=3e-4, patience=5, init=init)
    assert ft["initial_weights_sha256"] == pre["selected_weights_sha256"]             # starts from the selected epoch
    assert ft["scalers"] == pre["scalers"] and ft["config"]["lr"] == 3e-4            # PG-pre's scalers, unchanged
    assert ft["curve"][0]["val_loss"] != pre["curve"][0]["val_loss"]                  # epoch 0 here is PG-pre on 25
    b12, b24 = (tt.fit(store, [25], 1, "cpu", n_train=n) for n in (12, 24))
    assert b12["states"]["train"] == 12 and b12["scalers"] != b24["scalers"]         # statistics of the B states only
    st = fm.Static(*fs._case(25))
    tr = tt.prepare(store, st, 25, "train", 12)
    assert np.allclose(b12["scalers"]["x"]["mu"], fm.Scaler(tr["x"]).mu)
    assert np.allclose(b12["scalers"]["residual"], tt.pooled_scales({25: tr}, {25: st})[0])

    nan = {k: torch.full_like(v, float("nan")) if v.is_floating_point() else v for k, v in init["state_dict"].items()}
    dead = tt.fit(store, [25], 2, "cpu", n_train=12, init={"state_dict": nan, "scalers": pre["scalers"]},
                  out_dir=study["tmp"] / "dead")
    assert dead["numerical_failure"] == {"what": "non-finite gradient", "epoch": 1, "step_in_epoch": 0}
    assert dead["best_epoch"] == -1 and dead["selected_weights_sha256"] is None
    assert not (study["tmp"] / "dead/best.pt").exists()                               # no model: the method defers


def test_final_fits_are_frozen_run_once_and_chained(study, monkeypatch):
    root = study["root"]
    assert len(tt.all_fits()) == 4 * 3 + 2 + 2 * 2 * 2                                # the fixture's folds and budgets
    monkeypatch.setenv("GENCODOE_COMMIT", "f" * 40)
    with pytest.raises(RuntimeError, match="GENCODOE_COMMIT"):
        tt.final_fit(root, "pgnet", "cpu", network=25)
    monkeypatch.delenv("GENCODOE_COMMIT")
    with pytest.raises(RuntimeError, match="not set to HEAD"):                        # a final fit must be pinned
        tt.final_fit(root, "pgnet", "cpu", network=25)
    monkeypatch.setenv("GENCODOE_COMMIT", HEAD)
    study["state"]["source_sha256"] = "t" * 64                                        # source changed after the freeze
    with pytest.raises(RuntimeError, match="differs from the freeze"):
        tt.final_fit(root, "pgnet", "cpu", network=25)
    study["state"].update(source_sha256="s" * 64, src_dirty=True)
    with pytest.raises(RuntimeError, match="modified"):
        tt.final_fit(root, "pgnet", "cpu", network=25)
    study["state"]["src_dirty"] = False
    assert not tt.runs().exists()                                                     # nothing was started

    with pytest.raises(FileNotFoundError):                                            # PG-ft needs its PG-pre
        tt.final_fit(root, "ft", "cpu", network=25, budget=12, seed=1)
    stale = tt.runs() / "pgnet_net25"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "best.pt").write_bytes(b"left by a killed job")
    fit_all(root)
    rec = json.loads((tt.runs() / "pgnet_net25/record.json").read_text())
    assert rec["incomplete_attempts"] == ["pgnet_net25.incomplete1"] and (tt.runs() / "pgnet_net25.incomplete1/best.pt").exists()
    assert rec["valid"] and rec["deterministic_algorithms"] and rec["dry_run"] is None and rec["fold"] == 0
    assert rec["source_sha256"] == "s" * 64 and rec["label_manifest_sha256"] == fd.MANIFEST_SHA256
    assert rec["checkpoint_sha256"] == hashlib.sha256((tt.runs() / "pgnet_net25/best.pt").read_bytes()).hexdigest()
    assert rec["states"]["train"] == 24 and rec["config"]["max_epochs"] == 2
    with pytest.raises(FileExistsError, match="never rerun"):
        tt.final_fit(root, "pgnet", "cpu", network=25)
    ft = json.loads((tt.runs() / "ft_net25_b12_seed1/record.json").read_text())
    pre = json.loads((tt.runs() / "pgpre_fold0_seed1/record.json").read_text())
    assert pre["pool"] == [10] and ft["pool"] == [25] and ft["budget"] == 12 and ft["states"]["train"] == 12
    assert ft["started_from"] == {"name": "pgpre_fold0_seed1", "checkpoint_sha256": pre["checkpoint_sha256"]}
    assert ft["initial_weights_sha256"] == pre["selected_weights_sha256"] and ft["scalers"] == pre["scalers"]
    assert ft["config"]["lr"] == 3e-4 and ft["config"]["patience"] == 5
    sc = json.loads((tt.runs() / "scratch_net25_b12_seed1/record.json").read_text())
    assert sc["config"]["lr"] == 1e-3 and sc["scalers"] != pre["scalers"] and not sc["started_from_given_weights"]
    index = json.loads(tt.collect().read_text())
    assert index["fits_expected"] == 22 and index["fits"] == index["valid"] == 8 and len(index["missing"]) == 14

    torch.use_deterministic_algorithms(False)


def test_evaluation_reads_the_held_out_split_once_and_only_when_final(study, monkeypatch):
    root = study["root"]
    opened = []
    load = fd.LabelStore.load
    monkeypatch.setattr(fd.LabelStore, "load", lambda self, n, split, k=None: (opened.append((type(self).__name__, n, split)),
                                                                              load(self, n, split, k))[1])
    dry = {"n_train": 12, "n_val": 12, "epochs": 1}
    fit_all(root, dry)
    out = ev.evaluate(root, 25, "cpu", dry=True)
    rep = json.loads(out.read_text())
    assert {s for _, _, s in opened} == {"train", "val"}                              # a dry run: train and val only
    assert rep["dry_run"] and rep["evaluated_split"] == "second half of val" and rep["states"] == 6
    assert json.loads((tt.runs(True) / "pgnet_net25/record.json").read_text())["dry_run"] == dry
    with pytest.raises(FileNotFoundError, match="after all its fits"):                # no final fit yet
        ev.evaluate(root, 25, "cpu")
    with pytest.raises(PermissionError):
        fd.LabelStore(root).load(25, "test")
    assert "TestStore" not in Path(tt.__file__).read_text() and "fourwire_scale_eval" not in Path(tt.__file__).read_text()

    del opened[:]
    fit_all(root)
    assert {s for _, _, s in opened} == {"train", "val"}                              # the fits never read calib or test
    del opened[:]
    out = ev.evaluate(root, 25, "cpu")
    rep = json.loads(out.read_text())
    assert [o for o in opened if o[2] == "test"] == [("TestStore", 25, "test")]       # once, by the one reader
    assert {o for o in opened if o[2] != "test"} == {("LabelStore", 25, "calib"), ("LabelStore", 10, "calib")}
    assert out == study["tmp"] / "results/fourwire_scale_eval/network_25.json" and rep["valid"] and not rep["dry_run"]
    assert rep["states"] == 12 and rep["evaluated_split"] == "test"
    keys = {"U1", "PG-net", "PG-pre-tc", "PG-zs", "PG-scratch_b12_seed0", "PG-scratch_b12_seed1", "PG-ft_b12_seed0",
            "PG-ft_b12_seed1"}
    assert set(rep["methods"]) == keys
    m = rep["methods"]
    assert m["U1"]["fit"] is None and m["PG-ft_b12_seed1"]["fit"] == "ft_net25_b12_seed1"
    assert m["PG-pre-tc"]["fit"] == m["PG-zs"]["fit"] == "pgpre_fold0_seed0" and m["PG-zs"]["state"] == m["PG-pre-tc"]["state"]
    assert m["PG-zs"]["calibration"]["set"] == "pool calib" and m["PG-zs"]["calibration"]["networks"] == [10]
    assert m["PG-net"]["calibration"]["set"] == "target calib" and m["PG-net"]["calibration"]["states"] == 12
    for r in m.values():                                                              # 12 calibration states: no finite margin
        assert r["screen"]["fallback_share"] == 1.0 and r["screen"]["false_safe"] == 0
        assert r["state"]["nonfinite_predictions"] == 0 and r["state"]["vpn"]["mae"] >= 0
        assert set(r) >= {"screen_" + v for v in ev.VARIANTS}
    assert m["PG-net"]["state"]["vpn"]["mae"] != m["U1"]["state"]["vpn"]["mae"]
    z = np.load(fs.OUT / "eval" / rep["per_state_file"])
    assert z["PG-net__codes"].shape == (12, 2) and z["U1__summary"].shape == (12, 4) and z["exact_safe"].shape == (12, 2)
    assert rep["exact_sweep_vs_opendss"]["states"] == 12
    with pytest.raises(FileExistsError, match="evaluated once"):
        ev.evaluate(root, 25, "cpu")
    torch.use_deterministic_algorithms(False)


def _record(tmp, n, fb):
    """An evaluation record with given fallback shares {method: share} and no errors."""
    methods = {k: {"screen": {"fallback_share": v, "false_safe": 0, "cleared_indeterminate": 0, "false_unsafe": 0},
                   "state": {"vpn": {"mae": 0.001}}, "calibration": {"usable": True}} for k, v in fb.items()}
    (tmp / "fourwire_scale_eval").mkdir(exist_ok=True)
    (tmp / f"fourwire_scale_eval/network_{n}.json").write_text(json.dumps({"valid": True, "dry_run": False,
                                                                            "methods": methods}))


def test_e3_report_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "RESULTS", tmp_path)
    monkeypatch.setattr(fs, "ELIGIBLE", [1, 2, 3, 4])
    monkeypatch.setattr(ev, "FRESH", [2, 3, 4])
    diffs = {1: 0.50, 2: 0.03, 3: 0.05, 4: 0.01}                                     # scratch minus ft, per network
    for n, d in diffs.items():
        fb = {"U1": 0.9, "PG-net": 0.2, "PG-pre-tc": 0.6, "PG-zs": 0.7}
        for b in tt.BUDGETS:
            for s in tt.BUDGETS[b]:
                fb[f"PG-scratch_b{b}_seed{s}"] = 0.4 + d + 0.01 * s                    # the seed shifts both alike
                fb[f"PG-ft_b{b}_seed{s}"] = 0.4 + 0.01 * s
        _record(tmp_path, n, fb)
    e3 = json.loads(ev.report().read_text())
    p = e3["primary_b1000"]
    assert p["networks"] == 3 and p["seeds"] == [0, 1, 2] and np.isclose(p["mean_difference"], 0.03)   # network 1 is left out
    rng = np.random.default_rng(0)
    d = np.array([0.03, 0.05, 0.01])
    means = d[rng.integers(0, 3, (2000, 3))].mean(1)
    assert np.allclose(p["ci95"], [np.percentile(means, 2.5), np.percentile(means, 97.5)])
    assert p["ge_2_points"] and p["ci_excludes_0"] and p["criterion_met"] and p["false_safe"] == {"PG-ft": 0, "PG-scratch": 0}
    assert np.isclose(e3["sensitivity_with_network_1"]["primary_b1000"]["mean_difference"], np.mean(list(diffs.values())))
    assert np.isclose(e3["methods_seed0_fresh"]["PG-zs"]["fallback_share"], 0.7)
    rec = json.loads((tmp_path / "fourwire_scale_eval/network_3.json").read_text())      # one more false-safe for PG-ft
    rec["methods"]["PG-ft_b1000_seed2"]["screen"]["false_safe"] = 1
    (tmp_path / "fourwire_scale_eval/network_3.json").write_text(json.dumps(rec))
    p = json.loads(ev.report().read_text())["primary_b1000"]
    assert p["ge_2_points"] and not p["ft_not_more_false_safe"] and not p["criterion_met"]
