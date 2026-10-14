import numpy as np

from gencodoe import enwl, feeder


def test_reduced_enwl1_matches_full_network():
    full, red = enwl.load_enwl1(reduce=False), enwl.load_enwl1()
    assert red.n_bus < full.n_bus / 5
    rng = np.random.default_rng(0)
    p_cust = rng.uniform(-6e-3, 6e-3, (5, len(full.customer_bus)))  # MW, both directions
    out = []
    for net in (full, red):
        p = np.zeros((5, net.n_bus))
        np.add.at(p, (slice(None), net.customer_bus), p_cust)
        r = feeder.sweep_pf(net.to_feeder(), p, 0.3 * p)
        out.append((r.vm[:, net.customer_bus], r.loading.max(axis=1)))
    np.testing.assert_allclose(out[0][0], out[1][0], atol=1e-9)
    # MVA loading of a merged chain is taken at its sending end: within the voltage ratio
    np.testing.assert_allclose(out[0][1], out[1][1], rtol=0.01)
