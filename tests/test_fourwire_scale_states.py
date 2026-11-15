"""State generation for the 25-network study (docs/fourwire_scale_protocol.md §2.5, Amendment 1)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gencodoe import csiro_convert as cc
from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_scale as fs

pytestmark = pytest.mark.skipif(not (dk.ROOT / "data_out/fluvius/geen_ZP.npz").exists()
                                and not any((dk.ROOT / "data/load_profiles").glob("P6269_*")),
                                reason="Fluvius meter profiles not available")


def _same(a: dict, b: dict, n: int):
    for k in a:
        assert np.array_equal(a[k][:n], b[k][:n]), k


def test_customer_states_equal_enwl1_generator(tmp_path):
    """Same draw as FluviusScenarios (existing behaviour) when there is no shortage and no three-phase load."""
    n_cust, seed = 200, 11
    meta = tmp_path / "meta.parquet"
    g = dk.FluviusScenarios(np.arange(n_cust), meta, split="trainval", augment=0.5)
    out = g(SimpleNamespace(Pd=np.zeros(n_cust)), 40, None, None, seed)
    st = dk.customer_states(n_cust, "trainval", 40, seed, 0.5)
    m = pd.read_parquet(meta)
    assert np.allclose(out[:, :, 0].T * 1e3, st["p_kw"]) and np.allclose(out[:, :, 1].T * 1e3, st["q_kvar"])
    assert np.array_equal(np.array(m["meters"].tolist()), st["meters"])
    assert np.array_equal(m["step"].to_numpy(), st["step"]) and np.array_equal(m["flex_kw"].to_numpy(), st["flex_kw"])


def test_prefix_invariance_500_of_10000():
    """Amendment 1: a 500-state run equals states [:500] of a 10,000-state run (largest network, 883)."""
    a = dk.customer_states(883, "train", 500, fs.seed(17, "train"), fs.AUGMENT)
    b = dk.customer_states(883, "train", 10_000, fs.seed(17, "train"), fs.AUGMENT)
    _same(a, b, 500)


def test_meter_reuse_within_group_and_deterministic():
    n_cust = 883                                                   # > 300 meters per group: shortages occur
    a = dk.customer_states(n_cust, "train", 30, 5, 0.5)
    b = dk.customer_states(n_cust, "train", 30, 5, 0.5)
    _same(a, b, 30)
    eans = {(g.hp, g.ev, g.pv): set(g.ean.tolist()) for g in map(dk.load_group, dk.GROUPS)}
    shortage = False
    for s in range(30):
        for key, members in eans.items():
            who = (a["groups"][s] == np.array(key)).all(1)
            m = a["meters"][s][who]
            assert set(m.tolist()) <= members                     # reuse stays within the technology group
            if m.size > len(members):
                shortage = True
                assert np.unique(m).size == len(members)           # every meter used before any is reused
            else:
                assert np.unique(m).size == m.size                 # no shortage: no reuse
    assert shortage
    rep = fs.meter_reuse(a)
    assert any(v["repeated_share"] > 0 for v in rep.values())


def test_split_days_respected():
    st = dk.customer_states(10, "train", 200, 1, 0.0)
    assert set(dk.day_split(st["step"])) == {"train"}
    st = dk.customer_states(10, "val", 50, 2, 0.0)
    assert set(dk.day_split(st["step"])) == {"val"}


def test_three_phase_customer_expanded_with_one_profile(tmp_path):
    cc.convert(cc.network_dirs()[16], tmp_path)
    from gencodoe.fourwire_net import load_fourwire
    net = load_fourwire(tmp_path)
    names, idx, share = cc.customers(net.customer_names)
    st = dk.customer_states(len(names), "train", 20, fs.seed(16, "train"), fs.AUGMENT)
    p, q = fs.connection_states(st, idx, share)
    assert p.shape == (20, 240) and st["p_kw"].shape == (20, 238)
    split = [i for i, n in enumerate(net.customer_names) if cc.THREE_PHASE_SUFFIX in n]
    c = idx[split[0]]
    assert np.allclose(p[:, split], st["p_kw"][:, [c]] / 3) and np.allclose(q[:, split].sum(1), st["q_kvar"][:, c])
    assert np.allclose(p.sum(1), st["p_kw"].sum(1))


def test_fit_diagnostics_rank_and_scaling():
    rng = np.random.default_rng(0)
    i = rng.normal(size=(50, 3)) + 1j * rng.normal(size=(50, 3))
    d = fs.fit_diagnostics(np.zeros((50, 3)), i)
    assert d["rank"] == 4 and len(d["singular_values"]) == 4
    d2 = fs.fit_diagnostics(np.zeros((50, 3)), 1e-3 * i)
    assert d2["cond2"] != pytest.approx(d["cond2"])                # cond2 depends on units
    assert d2["cond2_column_normalized"] == pytest.approx(d["cond2_column_normalized"])
    i_col = np.repeat(i[:, :1], 3, axis=1)                         # collinear phase currents
    assert fs.fit_diagnostics(np.zeros((50, 3)), i_col)["rank"] == 2
