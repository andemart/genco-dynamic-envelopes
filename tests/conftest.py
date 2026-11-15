"""No unit test may generate a held-out state of the 25-network study (docs/fourwire_scale_protocol.md §2.6).

The state generator refuses the calibration and test splits while tests run. Tests that need something to
play those parts use `pretend_held_out`: states named calib or test in a temporary label tree, drawn from
val days with seeds of their own."""
import pytest


def _guard(real):
    def customer_states(n_cust, split, *args, **kwargs):
        if split in ("calib", "test"):
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
    yield


@pytest.fixture
def pretend_held_out(monkeypatch, no_held_out_states):
    """Label generation for the split names calib and test draws val-day states with other seeds."""
    from gencodoe import fourwire_scale as fs
    guarded = fs.customer_states

    def from_val(n_cust, split, n_states, seed, *args, **kwargs):
        if split in ("calib", "test"):
            return guarded(n_cust, "val", n_states, seed + 500_000, *args, **kwargs)
        return guarded(n_cust, split, n_states, seed, *args, **kwargs)

    monkeypatch.setattr(fs, "customer_states", from_val)
