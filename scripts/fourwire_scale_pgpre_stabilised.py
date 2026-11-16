"""Four-wire scale study: exploratory robustness analysis of PG-pre under a stabilised training recipe
(docs/fourwire_scale_protocol.md, Amendment 10). Train and val states only: the label reader is asked for
these two splits and for nothing else (it cannot serve test states). No fit of the study is changed or
replaced, and nothing is written under data_out/fourwire_scale/runs/final or runs/dry.

train() is fourwire_scale_train.fit for a PG-pre fit, with four switches (RECIPES):
    lr             peak learning rate of AdamW
    warmup_epochs  the learning rate rises linearly, step by step, from 0 to the peak over these epochs; the
                   cosine decay of the frozen recipe then runs over the remaining epochs
    clip_norm      the global gradient norm is clipped to this value before every optimiser step
    order          "shuffled": the batches of all networks in one random order (the frozen recipe);
                   "round_robin": rounds that hold one batch of every pool network, in a random order
With the switches of "frozen" it is the frozen routine; `equivalence` checks that bit for bit. It also keeps
what the frozen routine does not: the validation loss of every pool network and its four terms after every
epoch, and the loss, gradient norm and learning rate of every step.

    fit          one PG-pre fit into <runs>/<name>/: best.pt, curve.jsonl, steps.npz, record.json
    equivalence  fourwire_scale_train.fit and train() with the frozen switches on one reduced fit, compared
    collect      the twelve stabilised fits next to the twelve frozen PG-pre records

    python scripts/fourwire_scale_pgpre_stabilised.py fit --fold 2 --seed 1 --label-root <labels> --device cuda
    python scripts/fourwire_scale_pgpre_stabilised.py equivalence --fold 2 --seed 1 --n-train 650 --n-val 256 \
        --epochs 4 --label-root <labels> --device cuda --work <new folder> --out <new file>
    python scripts/fourwire_scale_pgpre_stabilised.py collect --frozen-runs <runs/final> --out <new file>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gencodoe import fourwire_scale_train as tt

fm, fs, fd, CONFIG = tt.fm, tt.fs, tt.fd, tt.CONFIG
PROTOCOL = "docs/fourwire_scale_protocol.md, Amendment 10 (exploratory robustness analysis)"
RECIPES = {"frozen": {"lr": CONFIG["lr"], "warmup_epochs": 0, "clip_norm": None, "order": "shuffled"},
           "stabilised": {"lr": 3e-4, "warmup_epochs": 5, "clip_norm": 1.0, "order": "round_robin"}}
TERMS = ("v", "vpn", "cur", "kvl")
WEAK_FACTOR = 5.0                   # weak: selected validation loss above this many times the frozen median
RUNS = fs.OUT / "runs" / "stabilised"


def inside(path: Path, folder: Path) -> bool:
    p, f = path.resolve(), folder.resolve()
    return p == f or f in p.parents


def guard(runs: Path | None = None, out: Path | None = None) -> None:
    """Nothing is written into the study's run folders, and no file is replaced. Checked before any work."""
    for own in (tt.runs(False), tt.runs(True)):
        for path in (runs, out):
            if path is not None and (inside(path, own) or inside(own, path)):
                raise SystemExit(f"{path}: the study's run folders are not written to")
    if out is not None and out.exists():
        raise SystemExit(f"{out} exists: it is not replaced")


def train(store, pool: list[int], recipe: dict, epochs: int, device: str, seed: int, out_dir: Path,
          n_train: int | None = None, n_val: int | None = None) -> tuple[dict, dict]:
    """fourwire_scale_train.fit (PG-pre: no given weights, the frozen patience and selection rule) with the
    switches of recipe. Statement for statement the frozen routine wherever a switch is off. Returns the
    result and the per-step arrays; out_dir must not exist."""
    n_train = tt.N_PRE if n_train is None else n_train
    out_dir.mkdir(parents=True, exist_ok=False)

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(out_dir / "curve.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    statics = {n: fm.Static(*fs._case(n)) for n in pool}
    tr = {n: tt.prepare(store, statics[n], n, "train", n_train) for n in pool}
    va = {n: tt.prepare(store, statics[n], n, "val", n_val) for n in pool}
    xs = tt.PooledScaler(tr[n]["x"] for n in pool)                    # training statistics only (§4)
    es = tt.PooledScaler(statics[n].edge_raw for n in pool)
    rs, vpn_scale, kvl_scale = tt.pooled_scales(tr, statics)
    for d in (tr, va):
        for n in pool:
            for k in range(0, len(d[n]["x"]), 256):
                d[n]["x"][k:k + 256] = xs(d[n]["x"][k:k + 256])

    model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE, hidden=CONFIG["hidden"], layers=CONFIG["layers"],
                                       heads=CONFIG["heads"]).to(dev)
    initial_sha = tt.weights_sha(model)
    batchers = {n: fm.Batcher(statics[n], es(statics[n].edge_raw), dev) for n in pool}
    physics = {n: fm.Physics(statics[n], dev) for n in pool}
    rs_t = torch.as_tensor(rs, device=dev)
    lr, warm, clip = recipe["lr"], recipe["warmup_epochs"], recipe["clip_norm"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warm, 1))
    W = CONFIG["loss_weights"]

    def terms_of(n, d, idx):
        """The frozen forward pass and loss, term by term: the four weighted terms per state, and the state weights."""
        b, nb = len(idx), statics[n].net.n_bus
        x = torch.as_tensor(d["x"][idx], device=dev).reshape(b * nb, -1)
        ei, ea = batchers[n].graph(b)
        out = model(x, ei, ea, b).view(b, nb, 8) * rs_t
        v1 = torch.as_tensor(d["v1"][idx], device=dev)
        v = v1 + torch.complex(out[..., :4], out[..., 4:]) * tt.V_PN_NOM
        phys = physics[n]
        vt = torch.as_tensor(d["V"][idx], device=dev)
        s = torch.as_tensor(d["s"][idx], device=dev)
        it = torch.as_tensor(d["I"][idx], device=dev)
        conv = torch.as_tensor(d["conv"][idx], device=dev).float()
        w = conv * torch.as_tensor(np.where(d["near"][idx], CONFIG["near_limit_weight"], 1.0), device=dev).float()
        dv = (v - vt) / tt.V_PN_NOM
        lv = ((dv.real / rs_t[:4]) ** 2 + (dv.imag / rs_t[4:]) ** 2).mean((1, 2))
        lvpn = ((phys.vpn(v) - phys.vpn(vt)) / vpn_scale).pow(2).mean(1)
        cur = phys.currents(v, s)
        rat = phys.rating[1:].view(1, -1, 1)
        lcur = ((cur[:, 1:] - it[:, 1:]).abs() / rat).pow(2).mean((1, 2))
        lkvl = (phys.kvl(v, cur).abs() / kvl_scale).pow(2).mean((1, 2))
        return {"v": W["v"] * lv, "vpn": W["vpn"] * lvpn, "cur": W["cur"] * lcur, "kvl": W["kvl"] * lkvl}, w

    def loss_of(t, w):
        per = t["v"] + t["vpn"] + t["cur"] + t["kvl"]
        return (per * w).sum() / w.sum().clamp(min=1.0)

    def evaluate():
        """The frozen validation loss (pooled over the converged val states of the pool), and the same
        quantity with its four terms for every pool network."""
        model.eval()
        tot, m_tot, nets = 0.0, 0.0, {}
        with torch.no_grad():
            for n in pool:
                d, acc, m_n = va[n], dict.fromkeys(("loss",) + TERMS, 0.0), 0.0
                for k in range(0, len(d["x"]), tt.EVAL_BATCH):
                    idx = np.arange(k, min(k + tt.EVAL_BATCH, len(d["x"])))
                    m = float(d["conv"][idx].sum())
                    t, w = terms_of(n, d, idx)
                    loss = float(loss_of(t, w))
                    tot += loss * m
                    m_tot += m
                    m_n += m
                    acc["loss"] += loss * m
                    den = w.sum().clamp(min=1.0)
                    for key in TERMS:
                        acc[key] += float((t[key] * w).sum() / den) * m
                nets[str(n)] = {key: v / max(m_n, 1) for key, v in acc.items()}
        model.train()
        return tot / max(m_tot, 1), nets

    rng = np.random.default_rng(seed)
    v0, nets0 = evaluate()
    curve, best, best_ep, bad = [{"epoch": 0, "val_loss": v0, "note": "before training", "val_by_network": nets0}], np.inf, -1, 0
    log(curve[-1])
    steps, failure, selected_sha = [], None, None                     # (epoch, network, loss, gradient norm, lr)
    scales = np.concatenate([rs, [vpn_scale, kvl_scale]])
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        failure = {"what": "zero or non-finite target scale", "epoch": 0, "step_in_epoch": 0}
        log({"numerical_failure": failure})
    for ep in range(1, (0 if failure else epochs) + 1):
        t_ep = time.perf_counter()
        per_net = []
        for n in pool:
            perm = rng.permutation(len(tr[n]["x"]))
            per_net.append([(n, np.sort(perm[k:k + CONFIG["batch"]])) for k in range(0, len(perm), CONFIG["batch"])])
        if recipe["order"] == "shuffled":                             # the frozen order
            batches = [b for bn in per_net for b in bn]
            order = [batches[j] for j in rng.permutation(len(batches))]
        else:                                                         # rounds of one batch per pool network
            order = []
            for r in range(max(len(bn) for bn in per_net)):
                order += [per_net[k][r] for k in rng.permutation(len(pool)) if r < len(per_net[k])]
        order = [(n, idx) for n, idx in order if tr[n]["conv"][idx].any()]   # as frozen: such a batch is no step
        tl, nb = 0.0, 0
        for j, (n, idx) in enumerate(order):
            if ep <= warm:
                for group in opt.param_groups:
                    group["lr"] = lr * ((ep - 1) + (j + 1) / len(order)) / warm
            t, w = terms_of(n, tr[n], idx)
            loss = loss_of(t, w)
            opt.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in grads):
                failure = {"what": "non-finite gradient", "epoch": ep, "step_in_epoch": nb}
                break
            if clip is None:                                          # measured only; the gradients are not touched
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            else:                                                     # returns the norm before clipping
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item()
            nb += 1
            steps.append((ep, n, loss.item(), float(norm), opt.param_groups[0]["lr"]))
        if failure is not None:                                       # the epoch is not completed
            log({"numerical_failure": failure})
            break
        if ep == warm:
            for group in opt.param_groups:
                group["lr"] = lr                                      # the decay starts from exactly the peak
        if ep > warm:
            sched.step()
        vl, nets = evaluate()
        norms = np.array([s[3] for s in steps[-nb:]]) if nb else np.array([np.nan])
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "steps": nb,
               "epoch_s": time.perf_counter() - t_ep,
               "lr_first_step": steps[-nb][4] if nb else None, "lr_last_step": steps[-1][4] if nb else None,
               "grad_norm": {"median": float(np.median(norms)), "p90": float(np.quantile(norms, 0.9)),
                             "max": float(norms.max()),
                             "share_clipped": float((norms > clip).mean()) if clip is not None else None},
               "val_by_network": nets}
        curve.append(rec)
        log(rec)
        if vl < best:
            best, best_ep, bad, selected_sha = vl, ep, 0, tt.weights_sha(model)
            torch.save(model.state_dict(), out_dir / ".best.partial.pt")
            os.replace(out_dir / ".best.partial.pt", out_dir / "best.pt")
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    res = {"pool": pool, "seed": seed, "recipe": recipe,
           "config": {**CONFIG, "lr": lr, "max_epochs": epochs, "seed": seed},
           "n_params": sum(p.numel() for p in model.parameters()), "epochs_requested": epochs,
           "scheduler_t_max": max(epochs - warm, 1), "numerical_failure": failure,
           "initial_weights_sha256": initial_sha, "selected_weights_sha256": selected_sha,
           "final_weights_sha256": tt.weights_sha(model),
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "states": {"train_per_network": n_train, "train": int(sum(len(tr[n]["x"]) for n in pool)),
                      "train_converged": int(sum(tr[n]["conv"].sum() for n in pool)),
                      "val": int(sum(len(va[n]["x"]) for n in pool)),
                      "val_converged": int(sum(va[n]["conv"].sum() for n in pool))},
           "scalers": {"x": xs.state(), "edge": es.state(), "residual": rs.tolist(), "vpn": vpn_scale, "kvl": kvl_scale},
           "best_epoch": best_ep, "best_val_loss": best if best_ep > 0 else None, "curve": curve}
    cols = list(zip(*steps)) if steps else [[], [], [], [], []]
    return res, {"epoch": np.array(cols[0], np.int32), "network": np.array(cols[1], np.int32),
                 "loss": np.array(cols[2], np.float64), "grad_norm": np.array(cols[3], np.float64),
                 "lr": np.array(cols[4], np.float64)}


def fit(recipe: str, fold: int, seed: int, label_root: Path, device: str, runs: Path, dry: dict | None) -> Path:
    """One PG-pre fit of a recipe. Full size (dry is None): the stabilised recipe only, and only as a final
    command of the study would run (tt.check_freeze: pinned to HEAD, the frozen source, inputs, libraries
    and GPU model, deterministic algorithms)."""
    guard(runs=runs)
    name = tt.run_name("pgpre", fold=fold, seed=seed)
    if dry is None and recipe != "stabilised":
        raise SystemExit("at full size only the stabilised recipe is run (Amendment 10)")
    prov = tt.start_provenance(device, tt.set_determinism(True))
    if dry is None:
        tt.check_freeze(prov)
    d = dry or {}
    res, steps = train(fd.LabelStore(label_root), tt.pool_of(fold), RECIPES[recipe],
                       d.get("epochs") or CONFIG["max_epochs"], device, seed, runs / name, d.get("n_train"), d.get("n_val"))
    folder = runs / name
    np.savez_compressed(folder / "steps.npz", **steps)
    ck = folder / "best.pt"
    fs._write_json(folder / "record.json",
                   {"protocol": PROTOCOL, "what": "PG-pre under a training recipe; train and val states only",
                    **prov, "unchanged_during_run": tt.checkout_unchanged(prov), "name": name, "kind": "pgpre",
                    "fold": fold, "recipe_name": recipe, "dry_run": dry,
                    "checkpoint_sha256": fs._sha(ck) if ck.exists() else None,
                    "steps_sha256": fs._sha(folder / "steps.npz"), **res})
    return folder / "record.json"


def equivalence(fold: int, seed: int, label_root: Path, device: str, work: Path, out: Path, n_train: int,
                n_val: int, epochs: int) -> bool:
    """fourwire_scale_train.fit and train() with the frozen switches, on the same reduced PG-pre fit: the
    curve, every step loss, the three weight hashes and the selected checkpoint file must be equal."""
    guard(runs=work, out=out)
    prov = tt.start_provenance(device, tt.set_determinism(True))
    store, pool = fd.LabelStore(label_root), tt.pool_of(fold)
    a = tt.fit(store, pool, epochs, device, seed=seed, out_dir=work / "frozen_routine", n_train=n_train,
               n_val=n_val, record_steps=True)
    b, steps = train(store, pool, RECIPES["frozen"], epochs, device, seed, work / "this_loop", n_train, n_val)
    key = lambda c: (c["epoch"], c.get("train_loss"), c["val_loss"], c.get("steps"))   # noqa: E731
    files = [work / k / "best.pt" for k in ("frozen_routine", "this_loop")]
    same = {"curve": [key(c) for c in a["curve"]] == [key(c) for c in b["curve"]],
            "step_losses": a["step_losses"] == steps["loss"].tolist(),
            **{h: a[h] == b[h] for h in ("initial_weights_sha256", "selected_weights_sha256",
                                         "final_weights_sha256", "best_epoch", "best_val_loss", "scalers")},
            "selected_checkpoint_file": all(f.exists() for f in files) and fs._sha(files[0]) == fs._sha(files[1])}
    fs._write_json(out, {"protocol": PROTOCOL,
                         "what": "the frozen fit routine and this script's loop with the frozen switches, same reduced fit",
                         **prov, "unchanged_during_run": tt.checkout_unchanged(prov), "fold": fold, "seed": seed,
                         "n_train": n_train, "n_val": n_val, "epochs": epochs, "epochs_run": len(a["curve"]) - 1,
                         "steps": len(a["step_losses"]), "best_epoch": a["best_epoch"],
                         "numerical_failure": [a["numerical_failure"], b["numerical_failure"]],
                         "same": same, "equal": all(same.values())})
    return all(same.values())


def collect(runs: Path, frozen_runs: Path, out: Path) -> Path:
    """The twelve stabilised fits next to the twelve frozen PG-pre records (read and checked as the study's
    commands check them). A fit is called weak when its selected validation loss is above WEAK_FACTOR times
    the median of the twelve frozen fits, or when it selected no checkpoint."""
    guard(out=out)
    names = sorted(k for k, v in tt.canonical_fits().items() if v["kind"] == "pgpre")
    frozen = {name: tt.fit_record(frozen_runs, name, False) for name in names}
    freeze = fs._sha(tt.freeze_file())
    threshold = WEAK_FACTOR * float(np.median([frozen[name]["best_val_loss"] for name in names]))
    weak = lambda loss: loss is None or loss > threshold               # noqa: E731
    rows = []
    for name in names:
        folder, fr = runs / name, frozen[name]
        rec = json.loads((folder / "record.json").read_text())
        ck = folder / "best.pt"
        problems = [what for what, bad in {
            "not that fit": rec["name"] != name or rec["fold"] != fr["fold"] or rec["seed"] != fr["seed"] or rec["pool"] != fr["pool"],
            "not the stabilised recipe at full size": rec["recipe"] != RECIPES["stabilised"] or rec["dry_run"] is not None,
            "not under the freeze of the frozen fits": rec["freeze_sha256"] != freeze or rec["freeze_sha256"] != fr["freeze_sha256"],
            "the checkout changed during the run": not rec["unchanged_during_run"],
            "other states or scalers than the frozen fit": rec["states"] != fr["states"] or rec["scalers"] != fr["scalers"],
            "the checkpoint is not the recorded one": (rec["checkpoint_sha256"] is None) == ck.exists()
            or (ck.exists() and fs._sha(ck) != rec["checkpoint_sha256"]),
            "the step file is not the recorded one": fs._sha(folder / "steps.npz") != rec["steps_sha256"],
        }.items() if bad]
        if problems:
            raise RuntimeError(f"{name}: " + "; ".join(problems))
        st = np.load(folder / "steps.npz")
        norm, clip = st["grad_norm"], rec["recipe"]["clip_norm"]
        sel = next((c for c in rec["curve"] if c["epoch"] == rec["best_epoch"]), None)
        both = fr["best_val_loss"] is not None and rec["best_val_loss"] is not None
        rows.append({
            "name": name, "fold": rec["fold"], "seed": rec["seed"],
            "frozen": {"job_id": fr["job_id"], "best_epoch": fr["best_epoch"], "best_val_loss": fr["best_val_loss"],
                       "epochs_run": len(fr["curve"]) - 1, "weak": weak(fr["best_val_loss"])},
            "stabilised": {"job_id": rec["job_id"], "best_epoch": rec["best_epoch"], "best_val_loss": rec["best_val_loss"],
                           "epochs_run": len(rec["curve"]) - 1, "weak": weak(rec["best_val_loss"]),
                           "numerical_failure": rec["numerical_failure"],
                           "val_by_network_at_selected_epoch": sel["val_by_network"] if sel else None,
                           "grad_norm_before_clipping": {
                               "median": float(np.median(norm)), "p99": float(np.quantile(norm, 0.99)),
                               "max": float(norm.max()), "share_of_steps_clipped": float((norm > clip).mean()),
                               "median_by_network": {str(n): float(np.median(norm[st["network"] == n])) for n in rec["pool"]}}},
            "stabilised_over_frozen": rec["best_val_loss"] / fr["best_val_loss"] if both else None,
            "earlier_attempts": sorted(p.name for p in runs.glob(name + ".attempt*"))})
    ratios = [r["stabilised_over_frozen"] for r in rows if r["stabilised_over_frozen"] is not None]
    fs._write_json(out, {"protocol": PROTOCOL,
                         "what": "PG-pre under the stabilised recipe next to the frozen PG-pre fits; validation "
                                 "loss on the val states of each fit's pool; no calib or test state",
                         "code_commit": tt.checkout_state()["head"], "freeze_sha256": freeze,
                         "recipes": RECIPES, "weak_factor": WEAK_FACTOR, "weak_above": threshold,
                         "weak_frozen": [r["name"] for r in rows if r["frozen"]["weak"]],
                         "weak_stabilised": [r["name"] for r in rows if r["stabilised"]["weak"]],
                         "median_stabilised_over_frozen": float(np.median(ratios)) if ratios else None,
                         "fits": rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "equivalence", "collect"])
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="stabilised")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--label-root", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=Path, default=RUNS, help="fit, collect: the folder of the stabilised fits")
    ap.add_argument("--frozen-runs", type=Path, help="collect: the folder of the final fits (read only)")
    ap.add_argument("--work", type=Path, help="equivalence: a new folder for its two fits")
    ap.add_argument("--out", type=Path, help="equivalence, collect: a new file")
    ap.add_argument("--n-train", type=int, help="fit: a dry run; equivalence: its size")
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    size = {"n_train": a.n_train, "n_val": a.n_val, "epochs": a.epochs}
    if a.cmd == "collect":
        if a.frozen_runs is None or a.out is None:
            ap.error("collect needs --frozen-runs and --out")
        print(collect(a.runs, a.frozen_runs, a.out))
        return
    if a.fold is None or a.seed is None or a.label_root is None:
        ap.error(f"{a.cmd} needs --fold, --seed and --label-root")
    if a.cmd == "fit":
        print(fit(a.recipe, a.fold, a.seed, a.label_root, a.device, a.runs,
                  size if any(v is not None for v in size.values()) else None))
        return
    if a.work is None or a.out is None or None in size.values():
        ap.error("equivalence needs --work, --out, --n-train, --n-val and --epochs")
    equal = equivalence(a.fold, a.seed, a.label_root, a.device, a.work, a.out, **size)
    print(a.out, "EQUAL" if equal else "NOT EQUAL")
    sys.exit(0 if equal else 1)


if __name__ == "__main__":
    main()
