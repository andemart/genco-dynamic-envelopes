"""Val labels only: the optimised (Numba) sweep against the OpenDSS labels, on convergence, on the four limit
flags and on the complete decision per direction (docs/fourwire_scale_protocol.md, Amendment 9, disclosure).

    python scripts/fourwire_scale_val_sweep_vs_opendss.py results/fourwire_scale_val_sweep_vs_opendss.json
"""
import json
import sys

import numpy as np

from gencodoe import csiro_convert as cc
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_data as fd
from gencodoe import fourwire_scale_eval as ev
from gencodoe.fourwire_sweep_opt import NumbaSweep

store = fd.LabelStore(fs.OUT / "labels")
rows = []
for n in fs.ELIGIBLE:
    net, th = fs._case(n)
    d = store.load(n, "val")
    _, idx, share = cc.customers(net.customer_names)
    s = d["p_kw_customer"][:, idx] * share + 1j * d["q_kvar_customer"][:, idx] * share
    out = NumbaSweep(net, th).solve(s, return_state=False)
    c, sc = d["converged"], out["converged"]
    sm, fl = out["summary"], d["flags"]
    pred = ev.limit_flags(sm)
    both = c & sc
    diff = (pred != fl) & both[:, None]
    tr = d["summary"][both, 3] - sm[both, 3]
    high = d["summary"][both, 3] > 0.8
    truth, sweep = ev.safe_by_direction(fl, c), ev.safe_by_direction(pred, sc)      # (S, 2): import, export
    ruled = np.where(sc[:, None], sweep, truth)        # §3: a non-converged sweep state gets OpenDSS's decision
    pairs = lambda m: {k: int(m[:, j].sum()) for j, k in enumerate(ev.DIRECTIONS)}  # noqa: E731
    rows.append({"network": n, "buses": net.n_bus, "dss_indet": int((~c).sum()), "sweep_nonconv": int((~sc).sum()),
                 "sweep_nonconv_dss_conv": int((~sc & c).sum()), "sweep_conv_dss_indet": int((sc & ~c).sum()),
                 "flag_diff_uv_ov_cable_tr": diff.sum(0).tolist(),
                 "sweep_clears_dss_violates": [int(((~pred[:, j]) & fl[:, j] & both).sum()) for j in range(4)],
                 "tr_dss_minus_proxy_max": float(tr.max()), "tr_dss_minus_proxy_min": float(tr.min()),
                 "tr_rel_at_load_gt_0.8": float(np.median((tr / d["summary"][both, 3])[high])) if high.any() else None,
                 "decision_sweep_safe_dss_not_safe": pairs(sweep & ~truth),
                 "decision_sweep_not_safe_dss_safe_with_section3_rule": pairs(~ruled & truth),
                 "decision_sweep_not_safe_dss_safe_without_opendss": pairs(~sweep & truth),
                 "sweep_nonconv_dss_conv_and_safe": pairs(~sc[:, None] & truth)})
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
