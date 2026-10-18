"""Post-hoc per-boundary violation classification (unbalanced protocol amendment 3). Frozen checkpoints,
frozen A100 inference path; the primary union-mask result is not altered.

    python -m gencodoe.fourwire_posthoc --device cuda
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import fourwire_eval as fe
from . import fourwire_model as fm
from . import fourwire_train as ft
from .fourwire_net import load_fourwire

LABELS = ("undervoltage", "overvoltage", "cable_overload", "transformer_overload")


def per_boundary(pred_sum, truth_sum, flags, conv):
    p = np.column_stack([pred_sum[:, 0] < fe.VMIN, pred_sum[:, 1] > fe.VMAX, pred_sum[:, 2] > 1.0, pred_sum[:, 3] > 1.0])
    masks = [np.abs(truth_sum[:, 0] - fe.VMIN) < 0.02, np.abs(truth_sum[:, 1] - fe.VMAX) < 0.02,
             (truth_sum[:, 2] > 0.9) & (truth_sum[:, 2] < 1.1), (truth_sum[:, 3] > 0.9) & (truth_sum[:, 3] < 1.1)]
    out = {}
    for j, (name, m) in enumerate(zip(LABELS, masks)):
        m = m & conv
        y, yh = flags[m, j], p[m, j]
        tp, fp, fn, tn = (y & yh).sum(), (~y & yh).sum(), (y & ~yh).sum(), (~y & ~yh).sum()
        r = {"states": int(m.sum()), "positives": int(y.sum()),
             "precision": float(tp / (tp + fp)) if tp + fp else None, "recall": float(tp / (tp + fn)) if tp + fn else None,
             "balanced_accuracy": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))) if y.any() and (~y).any() else None}
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/fourwire_posthoc_per_boundary.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    st = fm.Static(load_fourwire(), ft.thevenin())
    learned = {"PhaseGraph": fe.Learned("phasegraph", "final", st, dev), "PhaseGENCO": fe.Learned("phasegenco", "final", st, dev)}
    d, s, V, summ, _ = fe.run_split("test", st, learned, dev, allow_test=True)
    sm = d["summary"]
    near = ((np.abs(sm[:, 0] - fe.VMIN) < 0.02) | (np.abs(sm[:, 1] - fe.VMAX) < 0.02)
            | ((sm[:, 2] > 0.9) & (sm[:, 2] < 1.1)) | ((sm[:, 3] > 0.9) & (sm[:, 3] < 1.1)))
    committed = json.loads(Path("results/fourwire_eval_test.json").read_text())["classification"]
    out = {"status": "post-hoc (unbalanced protocol amendment 3); primary union-mask result unchanged",
           "per_boundary": {}, "consistency_union_mask": {}}
    for k in summ:
        out["per_boundary"][k] = per_boundary(summ[k], sm, d["flags"], d["converged"])
        u = fe.classification(summ[k], sm, d["flags"], d["converged"], near)["near_limit"]["mean_balanced_accuracy"]
        c = committed[k]["near_limit"]["mean_balanced_accuracy"]
        out["consistency_union_mask"][k] = {"rerun": u, "committed": c, "abs_diff": abs(u - c),
                                            "within_1e-3": abs(u - c) <= 1e-3}
    tmp = a.out.with_name("." + a.out.name + ".partial")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, a.out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
