"""No unit test may generate or read a held-out state of the 25-network study (docs/fourwire_scale_protocol.md
§2.6). While tests run:
- the study's state generator (customer_states, in both modules that bind it) and the datakit generator
  (FluviusScenarios) refuse the calibration and test splits;
- numpy.load refuses any file under the real label folders of those splits.
Tests that need something to play those parts use `pretend_held_out`: states named calib or test in a
temporary label tree, drawn from val days with seeds of their own.
This guards the paths the study's code uses. It does not stop a test that opens a label file by other means."""
import os

import numpy as np
import pytest

HELD_OUT = ("calib", "test")


def _guard(real):
    def customer_states(n_cust, split, *args, **kwargs):
        if split in HELD_OUT:
            raise AssertionError(f"a unit test asked the generator for {split} states")
        return real(n_cust, split, *args, **kwargs)
    return customer_states


@pytest.fixture(autouse=True)
def no_held_out_states(monkeypatch):
    try:
        from gencodoe import datakit_fluvius as dk
        from gencodoe import fourwire_scale as fs
    except ImportError:
        yield
        return
    guarded = _guard(dk.customer_states)
    monkeypatch.setattr(dk, "customer_states", guarded)
    monkeypatch.setattr(fs, "customer_states", guarded)
    init = dk.FluviusScenarios.__init__

    def scenarios(self, customer_bus, meta_path, split="trainval", augment=0.0):
        if split in HELD_OUT:
            raise AssertionError(f"a unit test asked the generator for {split} states")
        init(self, customer_bus, meta_path, split, augment)

    monkeypatch.setattr(dk.FluviusScenarios, "__init__", scenarios)
    forbidden = [os.path.realpath(fs.ROOT / "data_out/fourwire_scale/labels" / s) for s in HELD_OUT]
    load = np.load

    def guarded_load(file, *args, **kwargs):
        try:
            path = os.path.realpath(os.fspath(file))
        except TypeError:                                            # a file object
            path = ""
        if any(path == f or path.startswith(f + os.sep) for f in forbidden):
            raise AssertionError(f"a unit test opened a held-out label file: {path}")
        return load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    yield


@pytest.fixture
def pretend_held_out(monkeypatch, no_held_out_states):
    """Label generation for the split names calib and test draws val-day states with other seeds."""
    from gencodoe import fourwire_scale as fs
    guarded = fs.customer_states

    def from_val(n_cust, split, n_states, seed, *args, **kwargs):
        if split in HELD_OUT:
            return guarded(n_cust, "val", n_states, seed + 500_000, *args, **kwargs)
        return guarded(n_cust, split, n_states, seed, *args, **kwargs)

    monkeypatch.setattr(fs, "customer_states", from_val)
