"""Val labels only: the optimised sweep against OpenDSS, on convergence and on limit decisions."""
import json, sys, time
import numpy as np
from gencodoe import csiro_convert as cc, fourwire_scale as fs, fourwire_scale_data as fd
from gencodoe.fourwire_sweep_opt import NumbaSweep
from gencodoe.fourwire_train import VMIN, VMAX

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
    pred = np.column_stack([sm[:, 0] < VMIN, sm[:, 1] > VMAX, sm[:, 2] > 1.0, sm[:, 3] > 1.0])
    both = c & sc
    diff = (pred != fl) & both[:, None]
    tr = d["summary"][both, 3] - sm[both, 3]
    rows.append({"network": n, "buses": net.n_bus, "dss_indet": int((~c).sum()), "sweep_nonconv": int((~sc).sum()),
                 "sweep_nonconv_dss_conv": int((~sc & c).sum()), "sweep_conv_dss_indet": int((sc & ~c).sum()),
                 "flag_diff_uv_ov_cable_tr": diff.sum(0).tolist(),
                 "sweep_clears_dss_violates": [int(((~pred[:, j]) & fl[:, j] & both).sum()) for j in range(4)],
                 "tr_dss_minus_proxy_max": float(tr.max()), "tr_dss_minus_proxy_min": float(tr.min()),
                 "tr_rel_at_load_gt_0.8": float(np.median((tr / d["summary"][both, 3])[d["summary"][both, 3] > 0.8]))
                 if (d["summary"][both, 3] > 0.8).any() else None})
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
