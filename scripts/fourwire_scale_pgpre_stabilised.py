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

    fit          one PG-pre fit into <runs>/<name>/: best.pt, curve.jsonl, steps.npz, record.json. The run
                 folder is claimed as the final fits claim theirs (fourwire_scale_train.claim_run_folder): a
                 fit with a valid record is never run again (exit code 10), and a folder without a record is
                 set aside only when Slurm accounting shows an infrastructure end state.
    equivalence  fourwire_scale_train.fit and train() with the frozen switches on one reduced fit, compared
    collect      the twelve stabilised fits next to the twelve frozen PG-pre records

At full size (no --n-train/--n-val/--epochs; equivalence: no --dry) a command runs only as a final command of
the study would (pinned to HEAD, the frozen source, inputs, libraries and GPU model). No output may lie in a
run folder of the study (runs/final or runs/dry of any checkout, or --frozen-runs), and no file is replaced.

    python scripts/fourwire_scale_pgpre_stabilised.py fit --fold 2 --seed 1 --label-root <labels> --device cuda
    python scripts/fourwire_scale_pgpre_stabilised.py equivalence --fold 2 --seed 1 --n-train 648 --n-val 312 \
        --epochs 4 --label-root <labels> --device cuda --work <new folder> --out <new file>
    python scripts/fourwire_scale_pgpre_stabilised.py collect --frozen-runs <runs/final> \
        --equivalence <the equivalence record> --out <new file>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def guard(*outputs: Path, frozen_runs: Path | None = None) -> None:
    """Checked before any work: no output lies in a run folder of the study (runs/final or runs/dry of any
    checkout, or the folder given as --frozen-runs), and none contains one."""
    own = [tt.runs(False), tt.runs(True)] + ([frozen_runs] if frozen_runs is not None else [])
    for path in outputs:
        parts = path.resolve().parts
        named = any(a == "runs" and b in ("final", "dry") for a, b in zip(parts, parts[1:]))
        if named or any(inside(path, f) or inside(f, path) for f in own):
            raise SystemExit(f"{path}: the run folders of the study are not written to")


def write_new(path: Path, obj) -> None:
    """Write a JSON file that must not exist: an existing file is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=1))


def script_sha() -> str:
    return fs._sha(Path(__file__).resolve())


def script_committed() -> bool:
    """This file is tracked by git and identical to HEAD: the commit then names the analysis code."""
    f = str(Path(__file__).resolve())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], capture_output=True, cwd=fs.ROOT)
    status = subprocess.run(["git", "status", "--porcelain", "--", f], capture_output=True, text=True, cwd=fs.ROOT)
    return tracked.returncode == 0 and status.returncode == 0 and not status.stdout.strip()


def start(device: str, final: bool) -> dict:
    """Provenance read before a run. final: only as a final command of the study runs (tt.check_freeze:
    pinned to HEAD, the frozen source, inputs, libraries and GPU model, deterministic algorithms), and from
    this script as committed."""
    prov = tt.start_provenance(device, tt.set_determinism(True))
    prov["analysis_script_sha256"] = script_sha()
    if final:
        tt.check_freeze(prov)
        if not script_committed():
            raise RuntimeError("this script differs from HEAD")
    return prov


def unchanged(prov: dict) -> bool:
    return bool(tt.checkout_unchanged(prov) and script_sha() == prov["analysis_script_sha256"])


def train(store, pool: list[int], recipe: dict, epochs: int, device: str, seed: int, out_dir: Path,
          n_train: int | None = None, n_val: int | None = None) -> tuple[dict, dict]:
    """fourwire_scale_train.fit (PG-pre: no given weights, the frozen patience and selection rule) with the
    switches of recipe. Statement for statement the frozen routine wherever a switch is off. Returns the
    result and the per-step arrays. out_dir is new, or a claimed run folder that holds neither a checkpoint
    nor a curve: nothing is overwritten."""
    n_train = tt.N_PRE if n_train is None else n_train
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists() or (out_dir / "curve.jsonl").exists():
        raise FileExistsError(out_dir)

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


def check_record(rec: dict, name: str, dry: bool) -> dict:
    """A record of `fit` is the stabilised fit of that name and can be used: valid; and (full size) the
    stabilised recipe with the frozen training configuration, made as a final command of the study is made."""
    want = tt.canonical_fits()[name]
    bad = {"not that fit": rec["name"] != name or any(rec[k] != want[k] for k in ("kind", "fold", "seed", "pool")),
           "not valid": not rec["valid"],
           "a dry run where a full fit is asked for, or the reverse": (rec["dry_run"] is not None) != dry}
    if not dry:
        fr = json.loads(tt.freeze_file().read_text())
        cfg = {**CONFIG, "lr": RECIPES["stabilised"]["lr"], "max_epochs": CONFIG["max_epochs"], "seed": rec["seed"]}
        bad.update({
            "not the stabilised recipe": rec["recipe_name"] != "stabilised" or rec["recipe"] != RECIPES["stabilised"],
            "another training configuration": rec["config"] != cfg or rec["epochs_requested"] != CONFIG["max_epochs"]
            or rec["states"]["train_per_network"] != tt.N_PRE,
            "not under this freeze": rec["freeze_sha256"] != fs._sha(tt.freeze_file()),
            "not the frozen source, libraries or inputs": rec["source_sha256"] != fr["source_sha256"]
            or rec["software"] != fr["software"] or any(rec[k] != fr[k] for k in tt.input_hashes()),
            "not pinned to an unmodified checkout": rec["gencodoe_commit"] != rec["code_commit"] or rec["src_dirty"],
            "not the frozen GPU model or numerical mode": rec["cuda_device"] != tt.FINAL_DEVICE
            or not rec["deterministic_algorithms"] or rec["tf32"],
            "the checkout changed during the run": not rec["unchanged_during_run"]})
    problems = [what for what, is_bad in bad.items() if is_bad]
    if problems:
        raise AssertionError(f"{name}: " + "; ".join(problems))
    return rec


def check_files(folder: Path, rec: dict) -> None:
    ck, st = folder / "best.pt", folder / "steps.npz"
    if (rec["checkpoint_sha256"] is None) == ck.exists() or (ck.exists() and fs._sha(ck) != rec["checkpoint_sha256"]):
        raise AssertionError(f"{rec['name']}: the checkpoint is not the one its record names")
    if fs._sha(st) != rec["steps_sha256"]:
        raise AssertionError(f"{rec['name']}: the step file is not the one its record names")


def fit(recipe: str, fold: int, seed: int, label_root: Path, device: str, runs: Path, dry: dict | None) -> Path:
    """One PG-pre fit of a recipe. Full size (dry is None): the stabilised recipe, into this checkout's
    runs/stabilised, as a final command of the study runs. Attempts follow Amendment 7, item 3, through the
    study's own claim: a valid record is never run again (tt.AlreadyDone); a folder without a record is set
    aside only for an infrastructure end state in Slurm accounting; at most tt.MAX_RERUNS earlier attempts."""
    final, name = dry is None, tt.run_name("pgpre", fold=fold, seed=seed)
    if final and (recipe != "stabilised" or runs.resolve() != RUNS.resolve()):
        raise SystemExit("at full size: the stabilised recipe, into this checkout's runs/stabilised (Amendment 10)")
    guard(runs)
    prov = start(device, final)
    if final and len([q for q in runs.glob(f"{name}.*") if q.is_dir()]) >= tt.MAX_RERUNS:
        raise SystemExit(f"{name}: {tt.MAX_RERUNS} earlier attempts; a further one needs an amendment")
    out_dir, earlier = tt.claim_run_folder(runs, name, prov, final,
                                           validate=lambda rec: check_files(runs / name, check_record(rec, name, not final)))
    d = dry or {}
    res, steps = train(fd.LabelStore(label_root), tt.pool_of(fold), RECIPES[recipe],
                       d.get("epochs") or CONFIG["max_epochs"], device, seed, out_dir, d.get("n_train"), d.get("n_val"))
    np.savez_compressed(out_dir / "steps.npz", **steps)
    same = unchanged(prov)
    new = {c["epoch"]: (c.get("train_loss"), c["val_loss"]) for c in res["curve"]}
    for a in earlier:                                                 # a rerun must retrace what was recorded before
        old = tt._curve(runs / a["folder"])
        a.update({"epochs_recorded": len(old), "reproduced_by_this_run": all(new.get(e) == v for e, v in old.items())})
    ck = out_dir / "best.pt"
    rec = {"protocol": PROTOCOL, "what": "PG-pre under a training recipe; train and val states only",
           **prov, "unchanged_during_run": same, "name": name, "kind": "pgpre", "fold": fold,
           "recipe_name": recipe, "dry_run": dry, "earlier_attempts": earlier, **res,
           "checkpoint_sha256": fs._sha(ck) if ck.exists() else None,
           "steps_sha256": fs._sha(out_dir / "steps.npz")}
    rec["valid"] = bool(same and prov["deterministic_algorithms"])
    fs._write_json(out_dir / "record.json", rec)
    if not same:
        raise RuntimeError(f"checkout, script, numerical mode, libraries or inputs changed during the run; "
                           f"{name} is not a valid fit")
    return out_dir / "record.json"


def equivalence(fold: int, seed: int, label_root: Path, device: str, work: Path, out: Path, n_train: int,
                n_val: int, epochs: int, dry: bool) -> bool:
    """fourwire_scale_train.fit and train() with the frozen switches, on the same reduced PG-pre fit. Passed
    (True) only if the curve, every step loss, the three weight hashes and the selected checkpoint file are
    equal, both fits completed every epoch without a numerical failure, and nothing changed during the run.
    Without dry it runs only as a final command of the study runs."""
    guard(work, out)
    if work.exists() or out.exists() or inside(out, work):
        raise SystemExit("--work and --out must be new, and --out outside --work")
    prov = start(device, not dry)
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
    completed = len(a["curve"]) - 1 == epochs and len(b["curve"]) - 1 == epochs
    no_failure = a["numerical_failure"] is None and b["numerical_failure"] is None
    still = unchanged(prov)
    ok = bool(all(same.values()) and completed and no_failure and still)
    write_new(out, {"protocol": PROTOCOL,
                    "what": "the frozen fit routine and this script's loop with the frozen switches, same reduced fit",
                    **prov, "unchanged_during_run": still, "frozen_environment": not dry, "fold": fold, "seed": seed,
                    "n_train": n_train, "n_val": n_val, "epochs": epochs,
                    "epochs_run": [len(a["curve"]) - 1, len(b["curve"]) - 1],
                    "steps": [len(a["step_losses"]), len(steps["loss"])], "best_epoch": a["best_epoch"],
                    "numerical_failure": [a["numerical_failure"], b["numerical_failure"]],
                    "same": same, "equal": all(same.values()), "all_epochs_completed": completed,
                    "no_numerical_failure": no_failure, "passed": ok})
    return ok


NOTES = {"val loss, val_by_network": "as the fit's validation computes them: per batch of 128 states a mean with the "
                                     "near-limit weights, the batches then averaged by their number of converged "
                                     "states; not one globally weighted mean per network",
         "grad_norm_before_clipping": "over every optimiser step of the fit, the epochs after the selected one included",
         "median_stabilised_over_frozen": "over the fits in which both recipes selected a checkpoint (ratios_over)",
         "weak": "selected validation loss above weak_above, or no selected checkpoint"}
SET_ASIDE = ("Slurm accounting:", "DEVIATION", "its record is invalid")     # reasons the study's claim gives


def collect(runs: Path, frozen_runs: Path, equivalence_file: Path, out: Path) -> Path:
    """The twelve stabilised fits next to the twelve frozen PG-pre records. Every record is checked: the
    frozen ones as the study's commands check them; the stabilised ones by check_record and check_files, and
    also: same freeze, states and scalers as the frozen fit of that name; earlier attempts only those the
    claim set aside; one commit and one script for all twelve and for the equivalence check, which passed in
    the frozen environment."""
    guard(out, frozen_runs=frozen_runs)
    if out.exists():
        raise SystemExit(f"{out} exists: it is not replaced")
    names = sorted(k for k, v in tt.canonical_fits().items() if v["kind"] == "pgpre")
    frozen = {name: tt.fit_record(frozen_runs, name, False) for name in names}
    eq = json.loads(equivalence_file.read_text())
    recs = {}
    for name in names:
        rec, fr = check_record(json.loads((runs / name / "record.json").read_text()), name, False), frozen[name]
        check_files(runs / name, rec)
        on_disk = sorted(q.name for q in runs.glob(name + ".*") if q.is_dir())
        problems = [what for what, bad in {
            "not under the freeze of the frozen fit": rec["freeze_sha256"] != fr["freeze_sha256"],
            "other states or scalers than the frozen fit": rec["states"] != fr["states"] or rec["scalers"] != fr["scalers"],
            "attempt folders on disk are not those its record lists": on_disk != sorted(a["folder"] for a in rec["earlier_attempts"]),
            "an earlier attempt was not set aside by the claim": any(
                not str(a.get("set_aside_because") or "").startswith(SET_ASIDE) for a in rec["earlier_attempts"]),
        }.items() if bad]
        if problems:
            raise AssertionError(f"{name}: " + "; ".join(problems))
        recs[name] = rec
    code = {(r["code_commit"], r["analysis_script_sha256"]) for r in [*recs.values(), eq]}
    if len(code) != 1 or not eq["passed"] or not eq["frozen_environment"] or not eq["unchanged_during_run"] \
            or eq["freeze_sha256"] != fs._sha(tt.freeze_file()):
        raise AssertionError("the fits and the equivalence check are not one commit and one script, or the "
                             "equivalence check did not pass in the frozen environment")
    threshold = WEAK_FACTOR * float(np.median([frozen[name]["best_val_loss"] for name in names]))
    weak = lambda loss: loss is None or loss > threshold               # noqa: E731
    rows = []
    for name in names:
        rec, fr = recs[name], frozen[name]
        st = np.load(runs / name / "steps.npz")
        norm, clip = st["grad_norm"], rec["recipe"]["clip_norm"]
        sel = next((c for c in rec["curve"] if c["epoch"] == rec["best_epoch"]), None)
        both = fr["best_val_loss"] is not None and rec["best_val_loss"] is not None
        norms = None if not len(norm) else {
            "steps": int(len(norm)), "median": float(np.median(norm)), "p99": float(np.quantile(norm, 0.99)),
            "max": float(norm.max()), "share_of_steps_clipped": float((norm > clip).mean()),
            "median_by_network": {str(n): float(np.median(norm[st["network"] == n])) if (st["network"] == n).any() else None
                                  for n in rec["pool"]}}
        rows.append({
            "name": name, "fold": rec["fold"], "seed": rec["seed"],
            "frozen": {"job_id": fr["job_id"], "best_epoch": fr["best_epoch"], "best_val_loss": fr["best_val_loss"],
                       "epochs_run": len(fr["curve"]) - 1, "weak": weak(fr["best_val_loss"])},
            "stabilised": {"job_id": rec["job_id"], "best_epoch": rec["best_epoch"], "best_val_loss": rec["best_val_loss"],
                           "epochs_run": len(rec["curve"]) - 1, "weak": weak(rec["best_val_loss"]),
                           "numerical_failure": rec["numerical_failure"],
                           "val_by_network_at_selected_epoch": sel["val_by_network"] if sel else None,
                           "grad_norm_before_clipping": norms, "earlier_attempts": rec["earlier_attempts"]},
            "stabilised_over_frozen": rec["best_val_loss"] / fr["best_val_loss"] if both else None})
    ratios = [r["stabilised_over_frozen"] for r in rows if r["stabilised_over_frozen"] is not None]
    commit, script = next(iter(code))
    write_new(out, {"protocol": PROTOCOL,
                    "what": "PG-pre under the stabilised recipe next to the frozen PG-pre fits; validation loss on "
                            "the val states of each fit's pool; no calib or test state",
                    "fits_code_commit": commit, "fits_script_sha256": script,
                    "collected_at_commit": tt.checkout_state()["head"], "collected_with_script_sha256": script_sha(),
                    "freeze_sha256": fs._sha(tt.freeze_file()),
                    "equivalence": {k: eq[k] for k in ("job_id", "fold", "seed", "n_train", "n_val", "epochs", "steps", "passed")},
                    "recipes": RECIPES, "weak_factor": WEAK_FACTOR, "weak_above": threshold, "notes": NOTES,
                    "weak_frozen": [r["name"] for r in rows if r["frozen"]["weak"]],
                    "weak_stabilised": [r["name"] for r in rows if r["stabilised"]["weak"]],
                    "median_stabilised_over_frozen": float(np.median(ratios)) if ratios else None,
                    "ratios_over": len(ratios), "fits": rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "equivalence", "collect"])
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="stabilised")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--label-root", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=Path, default=RUNS, help="the folder of the stabilised fits (fit: dry runs only)")
    ap.add_argument("--frozen-runs", type=Path, help="collect: the folder of the final fits (read only)")
    ap.add_argument("--equivalence", type=Path, help="collect: the record of the equivalence check")
    ap.add_argument("--work", type=Path, help="equivalence: a new folder for its two fits")
    ap.add_argument("--out", type=Path, help="equivalence, collect: a new file")
    ap.add_argument("--n-train", type=int, help="fit: a dry run; equivalence: its size")
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--dry", action="store_true", help="equivalence: outside the frozen environment (a test)")
    a = ap.parse_args()
    size = {"n_train": a.n_train, "n_val": a.n_val, "epochs": a.epochs}
    if a.cmd == "collect":
        if a.frozen_runs is None or a.equivalence is None or a.out is None:
            ap.error("collect needs --frozen-runs, --equivalence and --out")
        print(collect(a.runs, a.frozen_runs, a.equivalence, a.out))
        return
    if a.fold is None or a.seed is None or a.label_root is None:
        ap.error(f"{a.cmd} needs --fold, --seed and --label-root")
    if a.cmd == "fit":
        try:
            print(fit(a.recipe, a.fold, a.seed, a.label_root, a.device, a.runs,
                      size if any(v is not None for v in size.values()) else None))
        except tt.AlreadyDone as e:
            print(e)
            sys.exit(10)
        return
    if a.work is None or a.out is None or None in size.values():
        ap.error("equivalence needs --work, --out, --n-train, --n-val and --epochs")
    passed = equivalence(a.fold, a.seed, a.label_root, a.device, a.work, a.out, dry=a.dry, **size)
    print(a.out, "PASSED" if passed else "NOT PASSED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
