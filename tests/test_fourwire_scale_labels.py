"""Label shards, manifests and the integrity gate of the 25-network study
(docs/fourwire_scale_protocol.md §2.5, §3, §9 gate 4). Train states only."""
import json
import re

import numpy as np
import pytest

from gencodoe import csiro_convert as cc
from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_scale as fs
from gencodoe.fourwire_net import V_PN_NOM

pytest.importorskip("opendssdirect")
pytestmark = pytest.mark.skipif(not (dk.ROOT / "data_out/fluvius/geen_ZP.npz").exists()
                                and not any((dk.ROOT / "data/load_profiles").glob("P6269_*")),
                                reason="Fluvius meter profiles not available")
TASK = (25, "train")                                               # 42 customers, the smallest network


@pytest.fixture
def small(tmp_path, monkeypatch):
    """30 training states of network 25 in shards of 12, in a temporary output folder."""
    monkeypatch.setattr(fs, "OUT", tmp_path)
    monkeypatch.setattr(fs, "N_STATES", {**fs.N_STATES, "train": 30})
    monkeypatch.setattr(fs, "SHARD", 12)
    return fs.label_split(TASK)


def test_shards_and_manifest(small):
    folder = fs.labels_dir(*TASK)
    assert [(s["file"], s["first_state"], s["states"]) for s in small["shards"]] == [
        ("shard_000.npz", 0, 12), ("shard_001.npz", 12, 12), ("shard_002.npz", 24, 6)]
    assert sorted(f.name for f in folder.iterdir()) == ["manifest.json", "shard_000.npz", "shard_001.npz",
                                                        "shard_002.npz"]      # no partial file left
    assert json.loads((folder / "manifest.json").read_text()) == small
    o = small["opendss"]
    assert o["states"] == 30 and o["converged"] + o["solver_indeterminate"] == 30
    arrays, prov = fs._read_shard(folder / "shard_001.npz")
    assert fs.content_sha(arrays) == small["shards"][1]["content_sha256"]
    assert arrays["V"].shape == (12, small["retained_buses"], 4) and arrays["V"].dtype == complex
    assert prov["code_commit"] == small["provenance"][0]["code_commit"]
    # stored inputs are the frozen generator's states 12..23 (seed 1000 x 25 + 0, capped augmentation)
    st = dk.customer_states(42, "train", 30, 25_000, fs.AUGMENT, fs.CAP_KVA)
    assert np.array_equal(arrays["p_kw_customer"], st["p_kw"][12:24])
    assert np.array_equal(arrays["gamma"], st["gamma"][12:24])
    names, idx, share = cc.customers(fs._case(25)[0].customer_names)
    assert small["inputs_content_sha256"] == fs.content_sha(dict(zip("pq", fs.connection_states(st, idx, share))))


def test_resume_keeps_shards_and_reproduces_a_missing_one(small):
    folder = fs.labels_dir(*TASK)
    kept = (folder / "shard_000.npz").stat().st_mtime_ns
    (folder / "shard_001.npz").unlink()
    again = fs.label_split(TASK)                                   # shard 1 alone, from a fresh engine
    assert (folder / "shard_000.npz").stat().st_mtime_ns == kept   # existing shards are not rewritten
    assert again["content_sha256"] == small["content_sha256"]
    assert [s["content_sha256"] for s in again["shards"]] == [s["content_sha256"] for s in small["shards"]]


def test_integrity_passes_and_fails_closed(small):
    folder = fs.labels_dir(*TASK)
    rep = fs.integrity_split(TASK)
    assert rep["states"] == 30 and rep["content_hashes_match"] and rep["shards_contiguous"]
    assert rep["inputs_reproduced"] and rep["split_days_ok"] and rep["semantic"]["ok"]
    assert rep["semantic"]["states"] == 30 and rep["reproduces_earlier_labels"] is None
    assert rep["ok"] == rep["single_clean_commit"]                 # a dirty source tree is the only way to fail here

    arrays, prov = fs._read_shard(folder / "shard_002.npz")       # tampered input: hash and generator both disagree
    arrays["p_kw_customer"] = arrays["p_kw_customer"] + 1e-9
    np.savez(folder / "shard_002.npz", provenance=json.dumps(prov), **arrays)
    bad = fs.integrity_split(TASK)
    assert not bad["content_hashes_match"] and not bad["inputs_reproduced"] and not bad["ok"]

    man = {**small, "shards": small["shards"][:2]}                 # a missing shard
    (folder / "manifest.json").write_text(json.dumps(man))
    short = fs.integrity_split(TASK)
    assert short["states"] == 24 and not short["ok"]


def test_constant_power_range_uses_the_declared_load_rating(tmp_path):
    """Amendment 4, correction: OpenDSS switches the load model at 0.5 and 1.5 pu of the load's 0.23 kV
    (115 V and 345 V), not of the 240.18 V study base."""
    lo, hi = 0.5 * 0.23, 1.5 * 0.23                               # kV
    v = np.array([[lo, 0.24], [np.nextafter(lo, 0), 0.24], [hi, 0.24], [np.nextafter(hi, 1), 0.24],
                  [0.49 * V_PN_NOM, 0.24],                        # 117.7 V: the first rule counted it
                  [1.45 * V_PN_NOM, 0.24]])                       # 348.3 V: the first rule missed it
    assert fs.outside_constant_power_range(v).tolist() == [False, True, False, True, False, True]
    assert not fs.outside_constant_power_range(np.array([[0.4793 * V_PN_NOM]]))[0]   # network_7's lowest
    cc.convert_all(tmp_path)
    declared = {m for f in tmp_path.glob("network_*/Loads.dss") for m in re.findall(r"kV=([0-9.]+)", f.read_text())}
    assert declared == {str(fs.LOAD_KV)}                          # every load of the 25 networks
