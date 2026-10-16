"""Option 2 (docs/option2_protocol.md; post-hoc, exploratory): a learned four-wire head on
phase-composition features, with no balanced model, the exact balanced summary, or GENCO's
predicted summary + pooled bus embeddings. Each head replaces the balanced state summary in
the frozen four-wire screen (FourWireMargins, calibrated on the calib days) and is evaluated
on the test days against four-wire truth.

    python -m gencodoe.option2 --run <GENCO run> --config <yaml> --out results/option2_screen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import feeder as fdr
from .doe_day import customer_groups
from .enwl import load_enwl1
from .evaluate import load_truth
from .four_wire import FourWireMargins, state_summary
from .limits import VMAX_PU, VMIN_PU, record as limits_record

TARGETS = ("v_pn_min", "v_pn_max", "line_loading_max", "transformer_loading")
ARMS = ("phase_only", "classical", "genco")


def load_split(data_dir: Path, split_file: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Truth arrays and four-wire truth of one dataset, restricted to the scenarios of split_file."""
    import torch
    truth = load_truth(data_dir)
    fw = pd.read_parquet(Path(data_dir) / "enwl1_fluvius" / "four_wire_truth.parquet").set_index("scenario")
    keep = np.ones(len(truth["scenario"]), bool)
    if split_file is not None:
        keep = np.isin(truth["scenario"], torch.load(split_file, weights_only=False).numpy())
    t = {k: truth[k][keep] for k in ("scenario", "pd", "qd")}
    return t, fw.loc[t["scenario"]].reset_index()


def phase_features(pd_mw: np.ndarray, qd_mvar: np.ndarray, cust_bus: np.ndarray, groups: np.ndarray,
                   phase: np.ndarray) -> np.ndarray:
    """Net consumption P, Q (kW, kvar) per (feeder x phase) group, per phase, and in total."""
    p, q = pd_mw[:, cust_bus] * 1e3, qd_mvar[:, cust_bus] * 1e3
    cols = []
    for lab in (groups, phase):
        for g in np.unique(lab):
            cols += [p[:, lab == g].sum(1), q[:, lab == g].sum(1)]
    cols += [p.sum(1), q.sum(1)]
    return np.column_stack(cols)


def genco_features(oracle, feeder, pd_mw, qd_mvar, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """GENCO state summary (4) and final-layer bus embeddings pooled by mean and max over buses."""
    captured = []
    hook = oracle.model.model.mlp_bus.register_forward_hook(lambda m, inp, out: captured.append(inp[0].detach()))
    summ, emb = [], []
    try:
        for i in range(0, len(pd_mw), batch):
            p, q = -pd_mw[i:i + batch] / feeder.base_mva, -qd_mvar[i:i + batch] / feeder.base_mva
            captured.clear()
            pr = oracle(feeder, p, q)
            h = captured[-1].reshape(len(p), feeder.n_bus, -1).cpu().numpy().astype(float)
            summ.append(np.column_stack(state_summary(pr)))
            emb.append(np.concatenate([h.mean(1), h.max(1)], axis=1))
    finally:
        hook.remove()
    return np.concatenate(summ), np.concatenate(emb)


def classical_features(feeder, pd_mw, qd_mvar) -> np.ndarray:
    pr = fdr.sweep_pf(feeder, -pd_mw / feeder.base_mva, -qd_mvar / feeder.base_mva)
    return np.column_stack(state_summary(pr))


def truth_ok(fw: pd.DataFrame, direction: str) -> np.ndarray:
    ok = fw.converged.to_numpy() & (fw.line_loading_max.to_numpy() <= 1.0) & (fw.transformer_loading.to_numpy() <= 1.0)
    if direction == "import":
        ok &= fw.v_pn_min.to_numpy() >= VMIN_PU
    else:
        ok &= fw.v_pn_max.to_numpy() <= VMAX_PU
    return ok


def screen_stats(margins: FourWireMargins, pred: np.ndarray, fw: pd.DataFrame, direction: str) -> dict:
    s = margins.screen(*pred.T, direction=direction)
    ok = truth_ok(fw, direction)
    safe = s == "safe"
    return {"direction": direction, "safe": float(safe.mean()), "certainly_violated": float((s == "violated").mean()),
            "sent_to_opendss": float((s == "uncertain").mean()), "false_safe": int((safe & ~ok).sum()),
            "false_violated": int(((s == "violated") & ok).sum()), "violating": int((~ok).sum()),
            "feasible_not_cleared": float(1.0 - (safe & ok).sum() / max(ok.sum(), 1))}


def fit_heads(X_tr, Y_tr, X_va, Y_va):
    from sklearn.ensemble import HistGradientBoostingRegressor
    heads = []
    for j in range(Y_tr.shape[1]):
        h = HistGradientBoostingRegressor(loss="squared_error", max_iter=500, learning_rate=0.05, early_stopping=True,
                                          random_state=0)
        h.fit(X_tr, Y_tr[:, j], X_val=X_va, y_val=Y_va[:, j])
        heads.append(h)
    return heads


def predict(heads, X) -> np.ndarray:
    return np.column_stack([h.predict(X) for h in heads])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--train", type=Path, default=Path("data_out/datakit_v2"))
    ap.add_argument("--calib", type=Path, default=Path("data_out/datakit_v2_calib"))
    ap.add_argument("--test", type=Path, default=Path("data_out/datakit_v2_test"))
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from .genco import GencoModel, GencoOracle, read_raw
    from .unbalanced import UnbalancedENWL1

    net = load_enwl1()
    feeder = net.to_feeder()
    ub = UnbalancedENWL1()
    groups = customer_groups(net, ub)
    phase = np.array([int(lb.split(".")[1]) for lb in ub.load_bus])
    model = GencoModel(a.config, a.run / "artifacts/model/best_model_state_dict.pt",
                       a.run / "artifacts/stats/normalizer_stats.pt")
    oracle = GencoOracle(model, read_raw(a.calib / "enwl1_fluvius" / "raw", scenarios=[0]))

    splits = {"train": load_split(a.train, a.train / "splits" / "train.pt"),
              "val": load_split(a.train, a.train / "splits" / "val.pt"),
              "calib": load_split(a.calib), "test": load_split(a.test)}
    feats, fws = {}, {}
    for name, (t, fw) in splits.items():
        ph = phase_features(t["pd"], t["qd"], net.customer_bus, groups, phase)
        cl = classical_features(feeder, t["pd"], t["qd"])
        gs, ge = genco_features(oracle, feeder, t["pd"], t["qd"])
        feats[name] = {"phase_only": ph, "classical": np.hstack([ph, cl]), "genco": np.hstack([ph, gs, ge]),
                       "_genco_summary": gs}
        fws[name] = fw
        print(f"{name}: {len(fw)} states, {int((~fw.converged).sum())} non-converged", flush=True)

    Y = {k: fws[k][list(TARGETS)].to_numpy() for k in fws}
    conv = {k: fws[k].converged.to_numpy() for k in fws}
    out = {"note": "post-hoc, exploratory (docs/option2_protocol.md); test days were also used in earlier development",
           "voltage_limits": limits_record(), "alpha": a.alpha, "run": str(a.run),
           "states": {k: int(len(v)) for k, v in fws.items()}, "rows": [], "head_mae_test": {},
           "n_features": {arm: int(feats["train"][arm].shape[1]) for arm in ARMS}}

    # reference: the frozen adapter (GENCO summary straight into the four-wire margins)
    ref = FourWireMargins.fit(*feats["calib"]["_genco_summary"].T, *Y["calib"].T, conv["calib"], a.alpha)
    for d in ("import", "export"):
        out["rows"].append({"arm": "frozen_adapter_genco (reference)", **screen_stats(ref, feats["test"]["_genco_summary"], fws["test"], d)})
    for arm in ARMS:
        heads = fit_heads(feats["train"][arm][conv["train"]], Y["train"][conv["train"]],
                          feats["val"][arm][conv["val"]], Y["val"][conv["val"]])
        pc, pt = predict(heads, feats["calib"][arm]), predict(heads, feats["test"][arm])
        m = FourWireMargins.fit(*pc.T, *Y["calib"].T, conv["calib"], a.alpha)
        ct = conv["test"]
        out["head_mae_test"][arm] = {k: float(np.abs(pt[ct, j] - Y["test"][ct, j]).mean()) for j, k in enumerate(TARGETS)}
        out["head_mae_test"][arm]["iterations"] = [int(h.n_iter_) for h in heads]
        for d in ("import", "export"):
            out["rows"].append({"arm": arm, **screen_stats(m, pt, fws["test"], d)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(out["rows"]).to_string(index=False))
    print(json.dumps(out["head_mae_test"], indent=1))


if __name__ == "__main__":
    main()
