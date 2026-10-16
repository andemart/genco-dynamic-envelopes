"""Envelope prediction v3: schema, leakage and edge-case tests (docs/envpred_protocol.md, amendment 3a)."""
import json

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
