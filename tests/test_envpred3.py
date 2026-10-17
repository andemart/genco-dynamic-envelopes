"""Envelope prediction v3: schema, leakage and edge-case tests (docs/envpred_protocol.md, amendment 3a)."""
import json

import pandas as pd

import numpy as np
import pytest

from gencodoe import envpred3 as e3
from gencodoe import feeder as fdr

DAY, HH, HOUR = "2024-01-02", 202, 13


def test_schema_exact_counts_order_and_uniqueness():
    counts = {"P": 224, "PC": 253, "PG": 1032, "PCG": 1061}
    blk = e3.block_names()
    assert {k: len(v) for k, v in blk.items()} == {"P": 224, "C": 29, "G": 220, "GA": 588}
    for arm, n in counts.items():
        names = e3.feature_names(arm)
        assert len(names) == n and len(set(names)) == n
        assert names == [x for b in e3.ARMS[arm] for x in blk[b]]
    ga = blk["GA"]
    assert ga[0] == "GA:L_G" and ga[85:87] == ["GA:share_LG", "GA:LG_is_zero"]
    assert ga[87:92] == [f"GA:share_c{v:03d}" for v in (0, 25, 50, 75, 100)]
    assert ga[-192].startswith("GA:dembLG_mean00") and blk["C"][0] == "C:L_ac"


class Stub:
    """Balanced exact sweep in place of GENCO; deterministic fake embeddings. Records every input."""

    def __init__(self, fail=False, nan_rows=False):
        self.calls, self.fail, self.nan_rows = [], fail, nan_rows

    def oracle(self, feeder, p, q):
        if self.fail:
            raise RuntimeError("inference failed")
        self.calls.append(p.copy())
        r = fdr.sweep_pf(feeder, p, q)
        if self.nan_rows:
            r.vm[::7] = np.nan
        return r

    def embed(self, feeder, p, q):
        r = self.oracle(feeder, p, q)
        v = np.nan_to_num(r.vm, nan=0.0)
        return r, np.hstack([np.repeat(v.mean(1, keepdims=True), 96, 1), np.repeat(v.max(1, keepdims=True), 96, 1)])


def _step():
    t0 = int((np.datetime64(DAY) - np.datetime64("2024-01-01")).astype(int)) * 96
    return t0 + 4 * HOUR


@pytest.fixture(scope="module")
def baseline():
    pytest.importorskip("opendssdirect")
    st = Stub()
    return e3.extract_block(DAY, HH, st.oracle, st.embed, steps={_step()}), st


def test_ga_independent_of_labels_and_opendss(baseline, tmp_path, monkeypatch):
    from gencodoe import envpred, unbalanced
    ex, _ = baseline
    # no label directory at all, a fake B file, and OpenDSS solves forbidden: GA unchanged
    monkeypatch.setattr(envpred, "LABELS", tmp_path / "none")
    monkeypatch.setattr(e3, "LABELS", tmp_path / "none")
    monkeypatch.setattr(unbalanced.UnbalancedENWL1, "solve", lambda *a, **k: pytest.fail("OpenDSS called"))
    st = Stub()
    ex2 = e3.extract_block(DAY, HH, st.oracle, st.embed, steps={_step()})
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / f"{DAY}_b_opendss_exact_sph1_seed101_hh{HH}_uk_esqcr.json").write_text(
        json.dumps({"meta": {}, "rows": [{"limit_kw": 99.0}]}))
    monkeypatch.setattr(envpred, "LABELS", tmp_path / "fake")
    ex3 = e3.extract_block(DAY, HH, st.oracle, st.embed, steps={_step()})
    for d in ("GA_export", "GA_import"):
        assert np.array_equal(np.array(ex[d]), np.array(ex2[d]), equal_nan=True)
        assert np.array_equal(np.array(ex[d]), np.array(ex3[d]), equal_nan=True)


def test_ga_uses_exactly_200_design_scenarios_and_bounds(baseline):
    ex, st = baseline
    assert all(len(c) % 200 == 0 for c in st.calls)
    for d, cap in (("GA_export", 5.0), ("GA_import", 7.4)):
        L_G = np.array(ex[d])[0, 0]
        assert 0.0 <= L_G <= cap
    assert not any(ex["failed"][0].values())
    with pytest.raises(ValueError):
        from gencodoe.enwl import load_enwl1
        net = load_enwl1()
        fd = net.to_feeder()
        p = np.zeros((50, fd.n_bus))
        e3.ga_block(st.oracle, st.embed, fd, None, p, p, "export", None, None, 1.0, {})


def test_inject_signs_and_caps():
    from gencodoe import envelopes
    cust = envelopes.Customers(bus=np.array([1, 2, 2]), export_cap=np.full(3, 0.005), import_cap=np.full(3, 0.0074))
    ex = e3.inject(cust, 4, 0.002, "export")
    im = e3.inject(cust, 4, 0.002, "import")
    assert np.allclose(ex, [[0, 0.002, 0.004, 0]]) and np.allclose(im, -ex)
    assert np.allclose(e3.inject(cust, 4, [0.001, 0.002], "export")[:, 2], [0.002, 0.004])


def test_zero_sentinel_failure_and_nonfinite_are_deterministic():
    pytest.importorskip("opendssdirect")
    from gencodoe import envelopes

    class Infeasible(Stub):
        def oracle(self, feeder, p, q):
            r = super().oracle(feeder, p, q)
            r.vm[:] = 0.5                                      # every scenario undervoltage: zero sentinel
            return r
    st = Infeasible()
    ex = e3.extract_block(DAY, HH, st.oracle, st.embed, steps={_step()})
    names = e3.block_names()["GA"]
    for d in ("GA_import",):
        ga = np.array(ex[d])[0]
        assert ga[0] == 0.0 and ga[names.index("GA:LG_is_zero")] == 1.0
        dl = [i for i, n in enumerate(names) if ":dLG_" in n]
        assert np.all(ga[dl] == 0.0)
    bad = Stub(fail=True)
    ex = e3.extract_block(DAY, HH, bad.oracle, Stub().embed, steps={_step()})
    assert np.isnan(np.array(ex["GA_export"])).all() and "inference failed" in ex["failed"][0]["export"]
    nanst = Stub(nan_rows=True)
    a = np.array(e3.extract_block(DAY, HH, nanst.oracle, nanst.embed, steps={_step()})["GA_export"])
    b = np.array(e3.extract_block(DAY, HH, nanst.oracle, nanst.embed, steps={_step()})["GA_export"])
    assert np.array_equal(a, b, equal_nan=True)
    del envelopes


def test_cached_exact_counts_counterfactual_calls():
    class Fake:
        def __init__(self):
            self.n = 0

        def share(self, p, q, L, d, canonical=False):
            self.n += 1
            return 1.0, len(p)

        def results(self, p, q, L, d, canonical=False):
            return [L] * len(p)
    f = Fake()
    c = e3.CachedExact(f)
    p = np.zeros((200, 3))
    assert c.share(p, p, 1.0, "export") == (1.0, 200) and c.share(p, p, 1.0, "export") == (1.0, 200)
    assert f.n == 1 and c.physical == 200
    c.reset()
    c.share(p, p, 1.0, "export")
    assert c.physical == 400


def test_genco_embed_keeps_final_layer_only():
    if not e3.RUN_LOCAL.exists():
        pytest.skip("frozen GENCO checkpoint not available")
    from gencodoe.enwl import load_enwl1
    oracle, embed = e3.load_genco("cpu", e3.RUN_LOCAL)
    fd = load_enwl1().to_feeder()
    p = np.full((300, fd.n_bus), -0.001)
    p[:, 0] = 0
    oracle.batch_size = 128                                   # three forward chunks, the last one partial
    _, pooled = embed(fd, p, np.zeros_like(p))
    assert pooled.shape == (300, 192)
    oracle.batch_size = 1024
    _, one = embed(fd, p, np.zeros_like(p))
    assert np.allclose(pooled, one, atol=1e-3)


def _fake_gate_rows(day="2024-01-10"):
    rows = []
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
    for k in range(24):
        for d, cap in (("export", 5.0), ("import", 7.4)):
            for arm in e3.ARMS:
                for q in e3.SHRINK_Q:
                    rows.append({"step": t0 + 4 * k, "direction": d, "arm": arm, "q": q, "L_B": 1.0, "L_s": 0.9,
                                 "gate_calls": 200, "fallback_calls": 0, "gate_pass": True, "fallback": False,
                                 "predicted_zero": False, "zero_sentinel": False, "issued": 0.9,
                                 "search_physical_calls": 0, "search_wall_s": 0.0, "fresh_p4": 0.99, "fresh_ac": 1.0})
    return rows


def test_gate_integrity_detects_problems(tmp_path, monkeypatch):
    from gencodoe.envelopes import acceptance_threshold
    from gencodoe.limits import record
    monkeypatch.setattr(e3, "OUT", tmp_path)
    monkeypatch.setattr(e3, "GATE", tmp_path / "gate")
    (tmp_path / "gate").mkdir()
    (tmp_path / e3.PRED_NAME).write_bytes(b"x")
    blocks = e3._test_blocks()
    assert len(blocks) == 180
    good = {"scenario_seed": 101, "threshold": acceptance_threshold(200, 0.05, 0.05), "caps_kw": e3.CAPS_KW,
            "voltage_limits": record(), "code_commit": "abc", "predictions_sha256": e3.sha256(tmp_path / e3.PRED_NAME),
            "physical_calls": 200, "physical_seconds": 1.0}
    for day, hh in blocks:
        (tmp_path / "gate" / f"{day}_hh{hh}.json").write_text(json.dumps({"day": day, "hh": hh, **good,
                                                                           "rows": _fake_gate_rows(day)}))
    rep = e3.gate_integrity(tmp_path / "gi.json", "abc")
    assert rep["complete"] and rep["rows"] == 138240 and rep["q95_rows"] == 34560
    day, hh = blocks[0]
    bad = _fake_gate_rows(day)[:-1]
    (tmp_path / "gate" / f"{day}_hh{hh}.json").write_text(json.dumps({"day": day, "hh": hh, **good, "rows": bad}))
    (tmp_path / "gate" / f"{blocks[1][0]}_hh{blocks[1][1]}.json").unlink()
    rep = e3.gate_integrity(tmp_path / "gi.json", "abc")
    assert not rep["complete"] and len(rep["missing"]) == 1 and rep["problems"]
    assert not e3.gate_integrity(tmp_path / "gi.json", "other-commit")["complete"]


def test_primary_statistics_use_capped_retention():
    g = pd.DataFrame(_fake_gate_rows("2024-01-10") + _fake_gate_rows("2024-04-09"))
    g["day"] = ["2024-01-10"] * 768 + ["2024-04-09"] * 768
    g["hh"] = 101
    g.loc[g.arm == "PG", "issued"] = 1.2                        # above B: retention capped at 100%, excess separate
    g["tol"] = g.direction.map(e3.CAPS_KW) * 1e-3
    g["kept"] = np.minimum(g.issued, g.L_B)
    g["excess"] = np.maximum(g.issued - g.L_B, 0)
    g["calls"] = g.gate_calls + g.fallback_calls
    g["b_calls"] = 1000
    q = g[g.q == 0.95]
    m = e3._arm_metrics(q[q.arm == "PG"])
    assert m["retention"] == 1.0 and abs(m["excess_over_B"] - 0.2) < 1e-12 and m["intervals_issued_above_B"] == 96
    lo, hi = e3._diff_boot(q, "PG", "P")
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9


def test_label_file_checks_flag_sentinel_commit_and_health(tmp_path):
    from gencodoe.envelopes import acceptance_threshold
    from gencodoe.limits import record
    day, hh = "2024-01-02", 101
    t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96

    def write(share, commit="abc", health=True):
        rows = [{"step": t0 + 4 * k, "direction": d, "method": m, "limit_kw": 0.5, "prob_ok_oracle": share,
                 "ac_calls": 400, "runtime_s": 1.0} for k in range(24) for d in ("export", "import") for m in ("ac", e3.B)]
        meta = {"day": day, "scenario_seed": 101, "household_seed": hh, "voltage_limits": record(),
                "acceptance_threshold": acceptance_threshold(200, 0.05, 0.05), "scenarios": 200, "code_commit": commit}
        if health:
            meta["engine_health"] = {"resets": 3, "nonconverged_returned": 0, "reset_failures": 0}
        f = tmp_path / "f.json"
        f.write_text(json.dumps({"meta": meta, "rows": rows}))
        return f
    assert e3.check_label_file(write(0.98), day, hh, "abc")[0] == []
    assert any("sentinel" in x for x in e3.check_label_file(write(0.0), day, hh, "abc")[0])
    assert any("commit" in x for x in e3.check_label_file(write(0.98), day, hh, "other")[0])
    assert any("engine health" in x for x in e3.check_label_file(write(0.98, health=False), day, hh, "abc")[0])
