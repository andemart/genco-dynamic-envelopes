"""Four-wire scale study: exploratory diagnostic of the PG-pre fits (docs/fourwire_scale_protocol.md, note of
2026-11-16 on the weak PG-pre seeds). Train and val states only. It reads run folders and never writes into
them; nothing here selects, changes or excludes a run.

checkpoints  For every PG-pre fit under --runs: the validation loss of the selected checkpoint per pool
             network, split into the four terms of the loss; the size of the predicted and of the true
             correction to the U1 prior; weight norms. The pooled loss is recomputed the way the fit computes
             it and compared with the recorded one: a fit whose loss is not reproduced is marked.
retrace      One PG-pre fit is run again with the frozen routine (same seed, deterministic algorithms),
             keeping every step loss, into --work. Its curve and weight hashes are compared with the record,
             and every training step is given its network by replaying the fit's shuffling.

    python scripts/fourwire_scale_pgpre_diagnostic.py checkpoints --runs <runs folder> --label-root <labels> \
        --device cuda --out results/fourwire_scale_pgpre_diagnostic.json
    python scripts/fourwire_scale_pgpre_diagnostic.py retrace --runs <runs folder> --name pgpre_fold2_seed1 \
        --label-root <labels> --device cuda --work <folder outside the runs> --out results/..._retrace_....json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from gencodoe import fourwire_scale_train as tt

fm, fs, fd = tt.fm, tt.fs, tt.fd
REL_TOL = 1e-5                      # recomputed against recorded validation loss


def read_record(runs: Path, name: str) -> tuple[Path, dict]:
    folder = runs / name
    rec = json.loads((folder / "record.json").read_text())
    if rec["name"] != name or rec["kind"] != "pgpre" or not rec["valid"]:
        raise RuntimeError(f"{name}: not a valid PG-pre record")
    return folder, rec


def n_val_of(rec: dict):                                              # a dry record used fewer val states
    return rec["dry_run"]["n_val"] if rec.get("dry_run") else None


def checkpoint_diagnostic(folder: Path, rec: dict, store, device: str) -> dict:
    """The selected checkpoint on the val states of its pool: loss per network and per term, as the fit's
    own validation computes it (batches of EVAL_BATCH, the near-limit weights, converged states only)."""
    base = {"name": rec["name"], "fold": rec["fold"], "seed": rec["seed"], "epochs_run": len(rec["curve"]) - 1,
            "best_epoch": rec["best_epoch"], "recorded_val_loss": rec["best_val_loss"]}
    if rec["checkpoint_sha256"] is None:
        return {**base, "has_model": False}
    ck = folder / "best.pt"
    if fs._sha(ck) != rec["checkpoint_sha256"]:
        raise RuntimeError(f"{rec['name']}: best.pt is not the recorded checkpoint")
    dev = torch.device(device)
    sc, cfg = rec["scalers"], tt.CONFIG
    xs, es = tt.scaler_from(sc["x"]), tt.scaler_from(sc["edge"])
    rs = torch.as_tensor(np.array(sc["residual"], np.float32), device=dev)
    vpn_scale, kvl_scale, W = float(sc["vpn"]), float(sc["kvl"]), cfg["loss_weights"]
    model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE, hidden=cfg["hidden"], layers=cfg["layers"],
                                       heads=cfg["heads"]).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev))
    model.eval()
    by_block = {}
    for key, p in model.named_parameters():
        by_block[key.split(".")[0]] = by_block.get(key.split(".")[0], 0.0) + float(p.detach().double().pow(2).sum())
    weights = {"l2": math.sqrt(sum(by_block.values())), "l2_by_block": {k: math.sqrt(v) for k, v in by_block.items()},
               "max_abs": max(float(p.detach().abs().max()) for p in model.parameters())}
    nets, tot, m_tot = {}, 0.0, 0.0
    for n in rec["pool"]:
        st = fm.Static(*fs._case(n))
        d = tt.prepare(store, st, n, "val", n_val_of(rec))
        for k in range(0, len(d["x"]), 256):
            d["x"][k:k + 256] = xs(d["x"][k:k + 256])
        batcher, phys, nb = fm.Batcher(st, es(st.edge_raw), dev), fm.Physics(st, dev), st.net.n_bus
        acc, m_n = {"loss": 0.0, "v": 0.0, "vpn": 0.0, "cur": 0.0, "kvl": 0.0}, 0.0
        sq, count = {"predicted": 0.0, "true": 0.0, "error": 0.0}, 0
        with torch.no_grad():
            for k in range(0, len(d["x"]), tt.EVAL_BATCH):
                idx = np.arange(k, min(k + tt.EVAL_BATCH, len(d["x"])))
                b = len(idx)
                x = torch.as_tensor(d["x"][idx], device=dev).reshape(b * nb, -1)
                ei, ea = batcher.graph(b)
                out = model(x, ei, ea, b).view(b, nb, 8) * rs
                v1 = torch.as_tensor(d["v1"][idx], device=dev)
                v = v1 + torch.complex(out[..., :4], out[..., 4:]) * tt.V_PN_NOM
                vt = torch.as_tensor(d["V"][idx], device=dev)
                s = torch.as_tensor(d["s"][idx], device=dev)
                it = torch.as_tensor(d["I"][idx], device=dev)
                conv = torch.as_tensor(d["conv"][idx], device=dev).float()
                w = conv * torch.as_tensor(np.where(d["near"][idx], cfg["near_limit_weight"], 1.0), device=dev).float()
                dv = (v - vt) / tt.V_PN_NOM
                lv = ((dv.real / rs[:4]) ** 2 + (dv.imag / rs[4:]) ** 2).mean((1, 2))
                lvpn = ((phys.vpn(v) - phys.vpn(vt)) / vpn_scale).pow(2).mean(1)
                cur = phys.currents(v, s)
                rat = phys.rating[1:].view(1, -1, 1)
                lcur = ((cur[:, 1:] - it[:, 1:]).abs() / rat).pow(2).mean((1, 2))
                lkvl = (phys.kvl(v, cur).abs() / kvl_scale).pow(2).mean((1, 2))
                terms = {"v": W["v"] * lv, "vpn": W["vpn"] * lvpn, "cur": W["cur"] * lcur, "kvl": W["kvl"] * lkvl}
                per = terms["v"] + terms["vpn"] + terms["cur"] + terms["kvl"]
                den, m = w.sum().clamp(min=1.0), float(d["conv"][idx].sum())
                acc["loss"] += float((per * w).sum() / den) * m
                for key, t in terms.items():
                    acc[key] += float((t * w).sum() / den) * m
                m_n += m
                c = conv.bool()                                        # correction to the U1 prior, pu of V_pn nominal
                pred, true = ((v - v1) / tt.V_PN_NOM)[c], ((vt - v1) / tt.V_PN_NOM)[c]
                sq["predicted"] += float(pred.abs().double().pow(2).sum())
                sq["true"] += float(true.abs().double().pow(2).sum())
                sq["error"] += float((pred - true).abs().double().pow(2).sum())
                count += pred.numel()
        nets[str(n)] = {"retained_buses": nb, "converged_val_states": int(m_n),
                        "val_loss": acc["loss"] / max(m_n, 1),
                        "terms": {key: acc[key] / max(m_n, 1) for key in ("v", "vpn", "cur", "kvl")},
                        "correction_rms_pu": {key: math.sqrt(v / max(count, 1)) for key, v in sq.items()}}
        tot += acc["loss"]
        m_tot += m_n
    pooled = tot / max(m_tot, 1)
    return {**base, "has_model": True, "recomputed_val_loss": pooled,
            "reproduces_recorded_loss": bool(abs(pooled - rec["best_val_loss"]) <= REL_TOL * abs(rec["best_val_loss"])),
            "weights": weights, "networks": nets}


def checkpoints(runs: Path, label_root: Path, device: str, out: Path) -> Path:
    prov = tt.start_provenance(device, tt.set_determinism(True))
    store = fd.LabelStore(label_root)
    names = sorted(p.parent.name for p in runs.glob("pgpre_*/record.json"))
    fits = []
    for name in names:
        fits.append(checkpoint_diagnostic(*read_record(runs, name), store, device))
        print(name, fits[-1].get("recomputed_val_loss"), fits[-1]["recorded_val_loss"], flush=True)
    fs._write_json(out, {"protocol": "docs/fourwire_scale_protocol.md, note of 2026-11-16 (exploratory diagnostic)",
                         "what": "selected PG-pre checkpoints on the val states of their pools; no test or calib state",
                         **prov, "relative_tolerance": REL_TOL, "fits": fits})
    return out


def retrace(runs: Path, name: str, label_root: Path, device: str, work: Path, out: Path) -> Path:
    folder, rec = read_record(runs, name)
    prov = tt.start_provenance(device, tt.set_determinism(True))
    store = fd.LabelStore(label_root)
    pool, n_train, cfg = rec["pool"], rec["states"]["train_per_network"], rec["config"]
    res = tt.fit(store, pool, rec["epochs_requested"], device, seed=rec["seed"], out_dir=work / name,
                 n_train=n_train, n_val=n_val_of(rec), record_steps=True, lr=cfg["lr"], patience=cfg["patience"])
    key = lambda c: (c["epoch"], c.get("train_loss"), c["val_loss"])   # noqa: E731
    differs = [a["epoch"] for a, b in zip(res["curve"], rec["curve"]) if key(a) != key(b)]
    same = {"curve": not differs and len(res["curve"]) == len(rec["curve"]),
            "first_epoch_that_differs": differs[0] if differs else None,
            **{h: res[h] == rec[h] for h in ("initial_weights_sha256", "selected_weights_sha256", "final_weights_sha256")}}
    # the network of every step: the fit's shuffling, replayed (fit: one permutation per network, then one of the batches)
    rng, batch = np.random.default_rng(rec["seed"]), cfg["batch"]
    conv = {n: store.load(n, "train", n_train)["converged"] for n in pool}
    ep_of, net_of = [], []
    for c in res["curve"][1:]:
        batches = []
        for n in pool:
            perm = rng.permutation(len(conv[n]))
            batches += [(n, np.sort(perm[k:k + batch])) for k in range(0, len(perm), batch)]
        nets = [batches[j][0] for j in rng.permutation(len(batches)) if conv[batches[j][0]][batches[j][1]].any()]
        if len(nets) != c["steps"]:
            raise RuntimeError(f"epoch {c['epoch']}: replay gives {len(nets)} steps, the fit made {c['steps']}")
        ep_of += [c["epoch"]] * len(nets)
        net_of += nets
    loss, ep_of, net_of = np.array(res["step_losses"]), np.array(ep_of), np.array(net_of)
    if len(loss) != len(net_of):
        raise RuntimeError("step losses and replayed steps differ in number")
    np.savez_compressed(work / f"{name}_steps.npz", epoch=ep_of, network=net_of, loss=loss)
    epochs = []
    for c in res["curve"][1:]:
        m = ep_of == c["epoch"]
        le, ne = loss[m], net_of[m]
        top = np.argsort(-le)[:10]
        epochs.append({"epoch": c["epoch"], "train_loss": c["train_loss"], "val_loss": c["val_loss"],
                       "median_step_loss": float(np.median(le)),
                       "largest_steps": [{"step_in_epoch": int(j), "network": int(ne[j]), "loss": float(le[j])} for j in top],
                       "by_network": {str(n): {"steps": int((ne == n).sum()), "mean": float(le[ne == n].mean()),
                                               "median": float(np.median(le[ne == n])), "max": float(le[ne == n].max()),
                                               "share_of_epoch_loss": float(le[ne == n].sum() / le.sum())}
                                      for n in pool}})
    fs._write_json(out, {"protocol": "docs/fourwire_scale_protocol.md, note of 2026-11-16 (exploratory diagnostic)",
                         "what": f"{name} run again with the frozen fit routine, every step loss kept; train and val states only",
                         **prov, "name": name, "record_job_id": rec["job_id"], "same_as_record": same,
                         "epochs_run": len(res["curve"]) - 1, "best_epoch": res["best_epoch"],
                         "steps_file": f"{name}_steps.npz", "steps_sha256": fs._sha(work / f"{name}_steps.npz"),
                         "epochs": epochs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checkpoints", "retrace"])
    ap.add_argument("--runs", type=Path, required=True, help="the folder that holds pgpre_*/record.json")
    ap.add_argument("--label-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", help="retrace: the fit")
    ap.add_argument("--work", type=Path, help="retrace: a folder outside the run folders")
    a = ap.parse_args()
    if a.cmd == "checkpoints":
        print(checkpoints(a.runs, a.label_root, a.device, a.out))
        return
    if a.name is None or a.work is None:
        ap.error("retrace needs --name and --work")
    if a.runs.resolve() == a.work.resolve() or a.runs.resolve() in a.work.resolve().parents:
        ap.error("--work must be outside the run folders")
    print(retrace(a.runs, a.name, a.label_root, a.device, a.work, a.out))


if __name__ == "__main__":
    main()
