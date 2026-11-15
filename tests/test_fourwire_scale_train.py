"""Training-side label access and the PG-pre smoke of the 25-network study
(docs/fourwire_scale_protocol.md §2.6, §4, §9 gate 5). Tiny label sets of networks 10 and 25, train/val/calib."""
import builtins
import inspect
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gencodoe import datakit_fluvius as dk
from gencodoe import fourwire_model as fm
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_data as fd
from gencodoe import fourwire_scale_train as tt

pytest.importorskip("opendssdirect")
pytest.importorskip("torch_geometric")
pytestmark = pytest.mark.skipif(not (dk.ROOT / "data_out/fluvius/geen_ZP.npz").exists()
                                and not any((dk.ROOT / "data/load_profiles").glob("P6269_*")),
                                reason="Fluvius meter profiles not available")
COMMIT = "0" * 40
POOL = [10, 25]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """24 train, 12 val and 12 calib states of networks 10 and 25 in shards of 12, with their manifest, and
    a decoy test tree that nothing may open or list."""
    monkeypatch.setattr(fs, "OUT", tmp_path / "out")
    monkeypatch.setattr(fs, "N_STATES", {"train": 24, "val": 12, "calib": 12, "test": 12})
    monkeypatch.setattr(fs, "SHARD", 12)
    monkeypatch.setattr(fs, "_provenance", lambda: {"code_commit": COMMIT, "src_dirty": False})
    mans = [fs.label_split((n, s)) for n in POOL for s in ("train", "val", "calib")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"states": fs.N_STATES, "shard_states": fs.SHARD, "label_code_commits": [COMMIT],
                                    "splits": mans}))
    decoy = tmp_path / "out/labels/test/network_25"
    decoy.mkdir(parents=True)
    (decoy / "shard_000.npz").write_bytes(b"not for training code")
    (decoy / "manifest.json").write_text("{}")
    monkeypatch.setattr(fd, "MANIFEST", manifest)                   # the store takes only the label root
    monkeypatch.setattr(fd, "LABEL_COMMIT", COMMIT)
    return fd.LabelStore(tmp_path / "out/labels")


@pytest.fixture
def test_tree_guard(store, monkeypatch):
    """Any open or directory listing inside <label root>/test fails and is recorded."""
    forbidden = os.path.realpath(store.root / "test")
    hits = []

    def guarded(fn):
        def wrapper(path=".", *args, **kwargs):
            try:
                p = os.path.realpath(os.fspath(path))
            except TypeError:                                      # a file descriptor
                p = ""
            if p == forbidden or p.startswith(forbidden + os.sep):
                hits.append(p)
                raise AssertionError(f"test tree touched: {p}")
            return fn(path, *args, **kwargs)
        return wrapper

    for mod, name in ((builtins, "open"), (io, "open"), (os, "scandir"), (os, "listdir")):
        monkeypatch.setattr(mod, name, guarded(getattr(mod, name)))
    return hits


def test_loader_serves_train_val_calib_and_refuses_test(store, test_tree_guard):
    d = store.load(25, "train")
    assert d["V"].shape[0] == 24 and d["V"].dtype == complex and np.array_equal(d["state"], np.arange(24))
    assert np.array_equal(store.load(25, "train", 12)["V"], d["V"][:12])
    assert len(store.load(10, "val")["converged"]) == 12 and len(store.load(10, "calib")["converged"]) == 12
    with pytest.raises(PermissionError):
        store.load(25, "test")
    assert test_tree_guard == []                                   # refused before any path is touched
    assert list(inspect.signature(fd.LabelStore.load).parameters) == ["self", "network", "split", "n_states"]
    assert list(inspect.signature(fd.LabelStore.__init__).parameters) == ["self", "root"]   # no manifest or commit override
    assert fd.SERVED == ("train", "val", "calib") and all(split != "test" for _, split in store.units)
    with pytest.raises(ValueError):
        store.load(25, "train", 25)


def test_loader_verifies_before_exposing_states(store, tmp_path, monkeypatch):
    folder = store.root / "train/network_25"
    arrays, prov = fs._read_shard(folder / "shard_001.npz")
    with monkeypatch.context() as m:
        m.setattr(fd, "LABEL_COMMIT", "1" * 40)
        with pytest.raises(AssertionError, match="label commits"):
            fd.LabelStore(store.root)
    np.savez(folder / "shard_001.npz", provenance=json.dumps({**prov, "src_dirty": True}), **arrays)
    with pytest.raises(AssertionError, match="differs from the committed manifest"):
        store.load(25, "train")                                    # shard written from modified source
    assert len(store.load(25, "train", 12)["state"]) == 12         # the untouched first shard still loads
    np.savez(folder / "shard_001.npz", provenance=json.dumps(prov), **{**arrays, "V": arrays["V"] + 1e-12})
    with pytest.raises(AssertionError, match="differs from the committed manifest"):
        store.load(25, "train")                                    # content hash
    np.savez(folder / "shard_001.npz", provenance=json.dumps(prov), **arrays)
    assert len(store.load(25, "train")["state"]) == 24
    man = json.loads((tmp_path / "manifest.json").read_text())
    unit = next(u for u in man["splits"] if (u["network"], u["split"]) == (25, "train"))
    unit["shards"] = unit["shards"][:1]                             # a split with a missing shard
    (tmp_path / "short.json").write_text(json.dumps(man))
    monkeypatch.setattr(fd, "MANIFEST", tmp_path / "short.json")
    with pytest.raises(AssertionError, match="not the frozen label set"):
        fd.LabelStore(store.root).load(25, "train", 12)


def test_loader_cannot_be_redirected_into_another_split(store, test_tree_guard, tmp_path, monkeypatch):
    """A manifest name with '..', a linked shard or a linked folder is refused before any byte is read."""
    decoy = store.root / "test/network_25/shard_000.npz"
    val = store.root / "val/network_25"
    (val / "shard_000.npz").unlink()
    (val / "shard_000.npz").symlink_to(decoy)                       # a shard that is a link into the test tree
    with pytest.raises(AssertionError, match="not a plain shard file"):
        store.load(25, "val")
    (store.root / "calib/network_25").rename(tmp_path / "moved")
    (store.root / "calib/network_25").symlink_to(store.root / "test/network_25")   # a linked folder
    with pytest.raises(AssertionError, match="not a plain shard file"):
        store.load(25, "calib")
    man = json.loads(fd.MANIFEST.read_text())
    unit = next(u for u in man["splits"] if (u["network"], u["split"]) == (25, "train"))
    unit["shards"][0]["file"] = "../../test/network_25/shard_000.npz"                # a path that climbs out
    (tmp_path / "evil.json").write_text(json.dumps(man))
    monkeypatch.setattr(fd, "MANIFEST", tmp_path / "evil.json")
    with pytest.raises(AssertionError, match="not a plain shard file"):
        fd.LabelStore(store.root).load(25, "train", 12)
    assert test_tree_guard == []


def test_training_and_smoke_never_open_or_list_the_test_tree(store, test_tree_guard, tmp_path, monkeypatch):
    decoy = store.root / "test/network_25/shard_000.npz"
    for touch in (lambda: open(decoy, "rb"), lambda: np.load(decoy), lambda: os.listdir(store.root / "test"),
                  lambda: list((store.root / "test").glob("*")), lambda: list(store.root.glob("*/network_25/*.npz"))):
        with pytest.raises(AssertionError, match="test tree touched"):
            touch()                                                # the guard is not vacuous
    del test_tree_guard[:]
    monkeypatch.setattr(tt, "N_PRE", 24)
    monkeypatch.setattr(tt, "pool_of", lambda fold: POOL)
    monkeypatch.setattr(fs, "ELIGIBLE", POOL)
    (tmp_path / "fourwire_scale_identification.json").write_bytes(
        (fs.RESULTS / "fourwire_scale_identification.json").read_bytes())      # the frozen Thevenin record
    monkeypatch.setattr(fs, "RESULTS", tmp_path)                                # the smoke report goes to tmp
    monkeypatch.setenv("GENCODOE_COMMIT", "f" * 40)
    with pytest.raises(RuntimeError, match="GENCODOE_COMMIT"):      # the pinned commit is asserted before training
        tt.smoke(store.root, 0, 2, "cpu")
    head = tt.checkout_state()["head"]
    monkeypatch.setenv("GENCODOE_COMMIT", head)
    report = tt.smoke(store.root, 0, 2, "cpu")
    rep = json.loads(report.read_text())
    assert test_tree_guard == [] and decoy.exists()
    assert rep["code_commit"] == rep["gencodoe_commit"] == head and rep["checkout_unchanged_during_run"]
    assert len(rep["initial_weights_sha256"]) == 64 and rep["initial_weights_sha256"] != rep["final_weights_sha256"]
    assert report.name.startswith("fourwire_scale_smoke_pgpre_local") and "upper bound" not in report.read_text()
    run = next((fs.OUT / "runs").iterdir())
    with pytest.raises(FileExistsError):                             # a run folder is never overwritten
        tt.pretrain(store, POOL, 1, "cpu", out_dir=run)
    assert rep["pool"] == POOL and rep["states"]["train"] == 48 and rep["states"]["val"] == 24
    assert [c["epoch"] for c in rep["curve"]] == [0, 1, 2] and all(np.isfinite(c["val_loss"]) for c in rep["curve"])
    assert all(v["train_ms_per_step"] > 0 and v["val_ms_per_state"] > 0 for v in rep["per_network"].values())
    assert rep["projection"]["gpu_hours_total"] > 0 and "label_root" not in rep
    for module in (fd, tt):                                        # and no code path names the split
        assert '"test"' not in Path(module.__file__).read_text() and "'test'" not in Path(module.__file__).read_text()


def test_folds_follow_the_protocol_rule():
    folds = tt.folds()
    buses = {u["network"]: u["retained_buses"] for u in json.loads(fd.MANIFEST.read_text())["splits"]}
    order = sorted(fs.ELIGIBLE, key=lambda n: (buses[n], n))
    assert [len(f) for f in folds] == [6] * 4 and sorted(sum(folds, [])) == fs.ELIGIBLE
    assert all(order.index(n) % 4 == k for k, f in enumerate(folds) for n in f)      # round-robin by size
    assert len(tt.pool_of(0)) == 18 and not set(tt.pool_of(0)) & set(folds[0])


def test_pooled_scaler_equals_the_enwl1_scaler_on_the_concatenation():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 7, 5)) * 50, rng.normal(size=(40, 3, 5))
    ref = fm.Scaler(np.concatenate([a.reshape(-1, 5), b.reshape(-1, 5)]))
    got = tt.PooledScaler([a, b])
    assert np.allclose(got.mu, ref.mu) and np.allclose(got.sd, ref.sd)
    assert np.allclose(got(a), ref(a), atol=1e-6)


def test_projection_counts_the_runs_of_section_8():
    buses = {u["network"]: u["retained_buses"] for u in json.loads(fd.MANIFEST.read_text())["splits"]}
    low = min(buses[n] for n in tt.pool_of(0))                      # never below the smallest measured network
    step, val = (lambda n: 5 + 0.01 * max(buses[n], low)), (lambda n: 0.1 + 0.001 * max(buses[n], low))   # ms
    per = {n: {"retained_buses": buses[n], "train_ms_per_step": step(n), "val_ms_per_state": val(n)}
           for n in tt.pool_of(0)}
    p = tt.projection(per, buses)
    assert low > min(buses.values())                               # fold 0 holds out the smallest network
    epoch = lambda n, k: (-(-k // 32) * step(n) + 3000 * val(n)) / 1e3                      # noqa: E731
    h = p["gpu_hours"]
    assert np.isclose(h["pg_net_24_fits"], 60 * sum(epoch(n, 10_000) for n in fs.ELIGIBLE) / 3600)
    assert np.isclose(h["pg_pre_12_runs"], 3 * 60 * sum(epoch(n, 5000) for f in range(4) for n in tt.pool_of(f)) / 3600)
    fits = sum(k * epoch(n, b) for n in fs.ELIGIBLE for b, k in ((200, 1), (1000, 3), (5000, 1)))
    assert np.isclose(h["pg_scratch_120_fits"], 60 * fits / 3600) and np.isclose(h["pg_ft_120_fits"], 30 * fits / 3600)
    assert np.isclose(p["gpu_hours_total"], sum(h.values())) and p["fits"]["train_ms_per_step"]["r2"] > 0.999


def test_cpu_training_is_bit_reproducible_and_scales_match_the_enwl1_formulas(store):
    """Two fits with the same seed on CPU: same initial weights, step losses and final weights. For a pool
    of one network the pooled target scales are the ENWL1 ones (fourwire_train.run_fit), up to its use of
    the sample standard deviation in float32 for the V_pn scale."""
    runs = [tt.pretrain(store, POOL, 1, "cpu", n_train=24, n_val=12, record_steps=True) for _ in range(2)]
    for k in ("initial_weights_sha256", "step_losses", "final_weights_sha256"):
        assert runs[0][k] == runs[1][k], k
    assert len(runs[0]["step_losses"]) == 2 and all(np.isfinite(runs[0]["step_losses"]))
    from gencodoe.fourwire_net import V_PN_NOM, phase_neutral_pu
    st = fm.Static(*fs._case(25))
    tr = tt.prepare(store, st, 25, "train", 24)
    rs, vpn, kvl = tt.pooled_scales({25: tr}, {25: st})
    c = tr["conv"]
    r = ((tr["V"] - tr["v1"]) / V_PN_NOM)[c]                                             # fourwire_train.residual_scale
    assert np.allclose(rs, np.concatenate([r.real.reshape(-1, 4).std(0), r.imag.reshape(-1, 4).std(0)]), rtol=1e-5)
    d = phase_neutral_pu(st.net, tr["V"][c][:2000]) - phase_neutral_pu(st.net, tr["v1"][c][:2000])
    assert np.isclose(vpn, d.std(ddof=1), rtol=5e-3)                                     # sample SD there, population SD here
    drop = (tr["V"][c][:2000][:, st.net.parent[1:]] - tr["V"][c][:2000][:, 1:]) / V_PN_NOM
    assert np.isclose(kvl, np.abs(drop).std(), rtol=1e-5)
