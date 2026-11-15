"""Fits of the 25-network study (docs/fourwire_scale_protocol.md §4, §8, Amendments 5, 7 and 8).

    python -m gencodoe.fourwire_scale_train fit --kind pgpre --fold 0 --seed 0 --label-root <dir> --device cuda
    python -m gencodoe.fourwire_scale_train fit --kind pgnet --network 17 --label-root <dir> --device cuda
    python -m gencodoe.fourwire_scale_train fit --kind scratch|ft --network 17 --budget 1000 --seed 0 ...
    python -m gencodoe.fourwire_scale_train freeze | collect
    python -m gencodoe.fourwire_scale_train smoke --label-root <dir> --fold 0 --epochs 3 --device cuda
    python -m gencodoe.fourwire_scale_train determinism --label-root <dir> --device cuda [--deterministic] --out <file>
    python -m gencodoe.fourwire_scale_train determinism-report --out-dir <dir with the four runs> --out <file>

One training routine (fit) for PhaseGraph (fourwire_model.GNS_heterogeneous_4wire) with the ENWL1 amendment 1
configuration and loss of fourwire_train:
- PG-pre: the first 5,000 train states of each of the 18 networks outside a fold, validation on their val states;
- PG-net, PG-scratch(B): one network, all 10,000 or the first B train states, validation on its val states;
- PG-ft(B): as PG-scratch(B), starting from the selected PG-pre checkpoint with PG-pre's scalers, lr 3e-4,
  at most 30 epochs, patience 5.
Labels are read through fourwire_scale_data.LabelStore: train and val only, from the label root given on the
command line.

fit --kind ... is a final fit (Amendments 7 and 8): deterministic algorithms, GENCODOE_COMMIT equal to HEAD,
unmodified source equal to the freeze (results/fourwire_scale_freeze.json), one run folder per fit under
data_out/fourwire_scale/runs/final/ holding best.pt and record.json. A fit with a record is never rerun; a
folder without a record (an interrupted run) is set aside and listed in the record of the rerun. With
--dry-train/--dry-val/--dry-epochs it is a dry run under runs/dry/ (train and val only, never evaluated on
test). freeze writes the freeze file; collect gathers the final records into results/fourwire_scale_fits.json.

smoke writes results/fourwire_scale_smoke_pgpre_fold<f>_<job>.json: peak GPU memory, epoch time, time per
step and per validation state for each network, and the fit-only projection of the §8 fits. The commit and
a hash of src/gencodoe are read before the run (GENCODOE_COMMIT, when set, must equal HEAD) and compared
again at the end. That detects a checkout that is different at the end; only a checkout that is never
moved guarantees more.

determinism trains 32 steps on networks 10 and 17 and writes the initial-weight hash, every step loss and
the final-weight hash, so that two processes can be compared bit for bit, with or without PyTorch's
deterministic algorithms. determinism-report compares the pairs and refuses a verdict on invalid runs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import csiro_convert as cc
from . import fourwire_model as fm
from . import fourwire_scale as fs
from .fourwire_net import V_PN_NOM, phase_neutral_pu
from . import fourwire_scale_data as fd
from .fourwire_train import CONFIG, VMAX, VMIN

N_PRE = 5000                       # §4 PG-pre: the first 5,000 train states of each pool network
EVAL_BATCH = 128                   # as fourwire_train.evaluate
BUDGETS = {200: [0], 1000: [0, 1, 2], 5000: [0]}    # §4: target budget B -> seeds of PG-ft and PG-scratch
FT_EPOCHS = 30                     # §4 PG-ft
# Amendment 8, item 1: what differs between the four kinds of fit
FITS = {"pgpre": {"lr": CONFIG["lr"], "epochs": CONFIG["max_epochs"], "patience": CONFIG["patience"]},
        "pgnet": {"lr": CONFIG["lr"], "epochs": CONFIG["max_epochs"], "patience": CONFIG["patience"]},
        "scratch": {"lr": CONFIG["lr"], "epochs": CONFIG["max_epochs"], "patience": CONFIG["patience"]},
        "ft": {"lr": 3e-4, "epochs": FT_EPOCHS, "patience": 5}}


def runs(dry: bool = False) -> Path:
    """Run folders of the final fits, or of dry runs."""
    return fs.OUT / "runs" / ("dry" if dry else "final")


def freeze_file() -> Path:
    return fs.RESULTS / "fourwire_scale_freeze.json"


def folds() -> list[list[int]]:
    """§2.6: the eligible networks sorted by retained-bus count (ties by number), round-robin to folds 0-3."""
    buses = {u["network"]: u["retained_buses"] for u in json.loads(fd.MANIFEST.read_text())["splits"]}
    order = sorted(fs.ELIGIBLE, key=lambda n: (buses[n], n))
    return [sorted(order[k::4]) for k in range(4)]


def pool_of(fold: int) -> list[int]:
    """The 18 pretraining networks of a fold: every eligible network outside it."""
    return sorted(set(fs.ELIGIBLE) - set(folds()[fold]))


def source_sha() -> str:
    """SHA-256 over the names and contents of src/gencodoe/*.py in the checkout this code runs in."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def checkout_state() -> dict:
    """HEAD of the checkout this code runs in, whether src/ differs from it, and the source hash."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=fs.ROOT, check=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"), "src_dirty": bool(git("status", "--porcelain", "src")),
            "source_sha256": source_sha()}


def set_determinism(deterministic: bool) -> dict:
    """Switch PyTorch's deterministic algorithms on or off explicitly, and TF32 off. Call before the first
    CUDA call. Returns the settings in effect, not the request."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False                    # FP32 without TF32 (§6), for every run
    torch.backends.cudnn.allow_tf32 = False
    return numerical_settings()


def numerical_settings() -> dict:
    """The numerical mode in effect now (read at the start of a run and again at its end)."""
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)}


def start_provenance(device: str, settings: dict) -> dict:
    """Read before a run: the checkout, the machine, the library versions and the hashes of the inputs.
    GENCODOE_COMMIT, set by the job script, must equal HEAD."""
    import torch_geometric
    state, pinned = checkout_state(), os.environ.get("GENCODOE_COMMIT")
    if pinned is not None and pinned != state["head"]:
        raise RuntimeError(f"GENCODOE_COMMIT {pinned} != HEAD {state['head']}")
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return {"code_commit": state["head"], "gencodoe_commit": pinned, "src_dirty": state["src_dirty"],
            "source_sha256": state["source_sha256"], "job_id": f"{job}_{task}" if job and task else job,
            "node": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "device": device, **settings,
            "software": software(), **input_hashes(), "freeze_sha256": None,
            "cuda": torch.version.cuda if cuda else None,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "cuda_device": torch.cuda.get_device_name(dev) if cuda else None,
            "cuda_total_mb": torch.cuda.get_device_properties(dev).total_memory / 2 ** 20 if cuda else None}


def checkout_unchanged(start: dict, variants: bool = False, profiles: bool = False) -> bool:
    """HEAD, the modified flag, the source hash, the numerical mode, the library versions and the input
    hashes are the same as at the start. For a final run also: the freeze file, the job script and, where
    the stage uses them, the rating-variant networks and the meter profiles still match the freeze."""
    state = checkout_state()
    same = (state["head"] == start["code_commit"] and state["src_dirty"] == start["src_dirty"]
            and state["source_sha256"] == start["source_sha256"]
            and all(start[k] == v for k, v in numerical_settings().items())
            and all(start[k] == v for k, v in input_hashes().items()) and start["software"] == software())
    if same and start["freeze_sha256"] is not None:
        fr = json.loads(freeze_file().read_text())
        same = (start["freeze_sha256"] == fs._sha(freeze_file()) and fs._sha(JOB_SCRIPT) == fr["job_script_sha256"]
                and (not variants or variant_networks_sha() == fr["variant_networks_sha256"])
                and (not profiles or profiles_sha() == fr["profiles_sha256"]))
    return same


def weights_sha(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Moments:
    """Running mean and standard deviation per column (float64)."""

    def __init__(self):
        self.n, self.s1, self.s2 = 0, 0.0, 0.0

    def add(self, y: np.ndarray):
        y = y.reshape(-1, y.shape[-1]).astype(np.float64)
        self.n += len(y)
        self.s1 = self.s1 + y.sum(0)
        self.s2 = self.s2 + (y * y).sum(0)

    def mean(self):
        return self.s1 / self.n

    def std(self):
        return np.sqrt(np.maximum(self.s2 / self.n - self.mean() ** 2, 0.0))


class PooledScaler(fm.Scaler):
    """fm.Scaler (asinh, then standardise) with statistics pooled over several arrays (..., F): every row
    of every array counts once, as when ENWL1 flattens its states and buses."""

    def __init__(self, arrays):
        m = Moments()
        for x in arrays:
            for k in range(0, len(x), 256):
                m.add(np.arcsinh(x[k:k + 256].astype(np.float64)))
        self.mu, self.sd = m.mean(), m.std()
        self.sd[self.sd < 1e-8] = 1.0


def scaler_from(state: dict) -> fm.Scaler:
    """A fitted scaler from its recorded mean and standard deviation."""
    x = fm.Scaler.__new__(fm.Scaler)
    x.mu, x.sd = np.array(state["mu"]), np.array(state["sd"])
    return x


def prepare(store: fd.LabelStore, st: fm.Static, network: int, split: str, n_states: int | None) -> dict:
    """fourwire_train.prepare for one network of the study (raw node features, U1 prior, targets)."""
    d = store.load(network, split, n_states)
    _, idx, share = cc.customers(st.net.customer_names)
    p, q = d["p_kw_customer"][:, idx] * share, d["q_kvar_customer"][:, idx] * share
    v1 = fm.prior_u1(st, p, q)
    sm = d["summary"]
    near = (np.abs(sm[:, 0] - VMIN) < 0.02) | (np.abs(sm[:, 1] - VMAX) < 0.02) | ((sm[:, 2] > 0.9) & (sm[:, 2] < 1.1)) \
        | ((sm[:, 3] > 0.9) & (sm[:, 3] < 1.1))
    x = fm.node_features(st, p, q, v1)
    c = d["converged"]
    v, cur, v1 = d["V"].astype(np.complex64), d["I"].astype(np.complex64), v1.astype(np.complex64)
    v[~c], cur[~c] = v1[~c], 0.0             # solver-indeterminate states carry weight 0; keep them finite
    return {"x": x, "v1": v1, "V": v, "I": cur, "s": (p + 1j * q).astype(np.complex64), "conv": c, "near": near}


def pooled_scales(tr: dict, statics: dict) -> tuple[np.ndarray, float, float]:
    """The three target scales of fourwire_train (residual per conductor, V_pn change, voltage drop), pooled
    over the networks. As there: all converged train states for the residual, the first 2,000 converged
    train states of each network for the other two."""
    res, dvpn, drop = Moments(), Moments(), Moments()
    for n, d in tr.items():
        net, c = statics[n].net, d["conv"]
        for k in range(0, len(c), 256):
            r = ((d["V"][k:k + 256] - d["v1"][k:k + 256]) / V_PN_NOM)[c[k:k + 256]]
            res.add(np.concatenate([r.real, r.imag], axis=2))
        v, v1 = d["V"][c][:2000], d["v1"][c][:2000]
        dvpn.add((phase_neutral_pu(net, v) - phase_neutral_pu(net, v1)).reshape(-1, 1))
        drop.add(np.abs((v[:, net.parent[1:]] - v[:, 1:]) / V_PN_NOM).reshape(-1, 1))
    return res.std().astype(np.float32), float(dvpn.std()[0]), float(drop.std()[0])


def fit(store: fd.LabelStore, pool: list[int], epochs: int, device: str, seed: int = 0,
        out_dir: Path | None = None, n_train: int | None = None, n_val: int | None = None,
        record_steps: bool = False, lr: float = CONFIG["lr"], patience: int = CONFIG["patience"],
        init: dict | None = None, prepared: bool = False) -> dict:
    """One fit on the first n_train train states (default N_PRE) and the val states of every network of
    pool. Batches hold states of one network (the model's graph context needs equal graph sizes); an epoch
    visits every training state once, with the batches of all networks shuffled.
    - init (PG-ft): {"state_dict", "scalers"} of the model to start from; its scalers and target scales are
      used unchanged and nothing is fitted on the data. Otherwise they come from the training states.
    - The selected checkpoint is the epoch >= 1 with the lowest validation loss, the earliest on a tie
      (out_dir/best.pt); epoch 0 is recorded and not eligible.
    - A non-finite gradient ends the fit there (Amendment 7, item 4): the result names it and the
      checkpoint is the one selected among the completed epochs. A target scale that is zero or not finite
      ends it before the first step. An epoch whose validation loss is not finite is never selected.
    out_dir must not exist, or (prepared) hold neither a checkpoint nor a curve: a run folder is never
    overwritten. Every epoch is appended to out_dir/curve.jsonl as it completes. n_val and record_steps
    serve the determinism probe and dry runs. Deterministic algorithms are switched by set_determinism
    before this is called."""
    n_train = N_PRE if n_train is None else n_train
    if out_dir is not None:
        if not prepared:
            out_dir.mkdir(parents=True, exist_ok=False)
        elif (out_dir / "best.pt").exists() or (out_dir / "curve.jsonl").exists():
            raise FileExistsError(out_dir)

    def log(rec):                                                    # the curve, kept even if the run is killed
        print(json.dumps(rec), flush=True)
        if out_dir is not None:
            with open(out_dir / "curve.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    statics = {n: fm.Static(*fs._case(n)) for n in pool}
    tr = {n: prepare(store, statics[n], n, "train", n_train) for n in pool}
    va = {n: prepare(store, statics[n], n, "val", n_val) for n in pool}
    if init is None:
        xs = PooledScaler(tr[n]["x"] for n in pool)                   # training statistics only (§4)
        es = PooledScaler(statics[n].edge_raw for n in pool)
        rs, vpn_scale, kvl_scale = pooled_scales(tr, statics)
    else:                                                             # PG-ft: PG-pre's, unchanged
        sc = init["scalers"]
        xs, es = scaler_from(sc["x"]), scaler_from(sc["edge"])
        rs, vpn_scale, kvl_scale = np.array(sc["residual"], np.float32), float(sc["vpn"]), float(sc["kvl"])
    for d in (tr, va):
        for n in pool:
            for k in range(0, len(d[n]["x"]), 256):
                d[n]["x"][k:k + 256] = xs(d[n]["x"][k:k + 256])
    prep_s = time.perf_counter() - t0

    model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE, hidden=CONFIG["hidden"], layers=CONFIG["layers"],
                                       heads=CONFIG["heads"]).to(dev)
    if init is not None:
        model.load_state_dict(init["state_dict"])
    initial_sha = weights_sha(model)
    batchers = {n: fm.Batcher(statics[n], es(statics[n].edge_raw), dev) for n in pool}
    physics = {n: fm.Physics(statics[n], dev) for n in pool}
    rs_t = torch.as_tensor(rs, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    W = CONFIG["loss_weights"]

    def forward(n, d, idx):
        b, nb = len(idx), statics[n].net.n_bus
        x = torch.as_tensor(d["x"][idx], device=dev).reshape(b * nb, -1)
        ei, ea = batchers[n].graph(b)
        out = model(x, ei, ea, b).view(b, nb, 8) * rs_t
        v1 = torch.as_tensor(d["v1"][idx], device=dev)
        return v1 + torch.complex(out[..., :4], out[..., 4:]) * V_PN_NOM

    def loss_fn(n, d, idx, v):                                       # fourwire_train.loss_fn
        phys = physics[n]
        vt = torch.as_tensor(d["V"][idx], device=dev)
        s = torch.as_tensor(d["s"][idx], device=dev)
        it = torch.as_tensor(d["I"][idx], device=dev)
        conv = torch.as_tensor(d["conv"][idx], device=dev).float()
        w = conv * torch.as_tensor(np.where(d["near"][idx], CONFIG["near_limit_weight"], 1.0), device=dev).float()
        dv = (v - vt) / V_PN_NOM
        lv = ((dv.real / rs_t[:4]) ** 2 + (dv.imag / rs_t[4:]) ** 2).mean((1, 2))
        lvpn = ((phys.vpn(v) - phys.vpn(vt)) / vpn_scale).pow(2).mean(1)
        cur = phys.currents(v, s)
        rat = phys.rating[1:].view(1, -1, 1)
        lcur = ((cur[:, 1:] - it[:, 1:]).abs() / rat).pow(2).mean((1, 2))
        lkvl = (phys.kvl(v, cur).abs() / kvl_scale).pow(2).mean((1, 2))
        per = W["v"] * lv + W["vpn"] * lvpn + W["cur"] * lcur + W["kvl"] * lkvl
        return (per * w).sum() / w.sum().clamp(min=1.0)

    def evaluate():
        """Mean val loss over the converged val states of the pool, and the seconds spent per network."""
        model.eval()
        tot, m_tot, secs = 0.0, 0.0, {}
        with torch.no_grad():
            for n in pool:
                t = time.perf_counter()
                d = va[n]
                for k in range(0, len(d["x"]), EVAL_BATCH):
                    idx = np.arange(k, min(k + EVAL_BATCH, len(d["x"])))
                    m = float(d["conv"][idx].sum())
                    tot += float(loss_fn(n, d, idx, forward(n, d, idx))) * m
                    m_tot += m
                secs[n] = time.perf_counter() - t
        model.train()
        return tot / max(m_tot, 1), secs

    rng = np.random.default_rng(seed)
    v0, _ = evaluate()
    curve, best, best_ep, bad = [{"epoch": 0, "val_loss": v0, "note": "before training"}], np.inf, -1, 0
    log(curve[-1])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    step_s, step_n, eval_s = {n: 0.0 for n in pool}, {n: 0 for n in pool}, {n: 0.0 for n in pool}
    step_losses, failure, selected_sha = [], None, None
    scales = np.concatenate([rs, [vpn_scale, kvl_scale]])
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        failure = {"what": "zero or non-finite target scale", "epoch": 0, "step_in_epoch": 0}
        log({"numerical_failure": failure})
    for ep in range(1, (0 if failure else epochs) + 1):
        t_ep = time.perf_counter()
        batches = []
        for n in pool:
            perm = rng.permutation(len(tr[n]["x"]))
            batches += [(n, np.sort(perm[k:k + CONFIG["batch"]])) for k in range(0, len(perm), CONFIG["batch"])]
        tl, nb = 0.0, 0
        for j in rng.permutation(len(batches)):
            n, idx = batches[j]
            if not tr[n]["conv"][idx].any():
                continue
            t = time.perf_counter()
            loss = loss_fn(n, tr[n], idx, forward(n, tr[n], idx))
            opt.zero_grad()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                failure = {"what": "non-finite gradient", "epoch": ep, "step_in_epoch": nb}
                break
            opt.step()
            tl += loss.item()                                        # synchronises: the step time is complete
            nb += 1
            if record_steps:
                step_losses.append(loss.item())
            if ep > 1:                                               # epoch 1 carries warm-up and graph caches
                step_s[n] += time.perf_counter() - t
                step_n[n] += 1
        if failure is not None:                                       # the epoch is not completed
            log({"numerical_failure": failure})
            break
        train_s = time.perf_counter() - t_ep
        sched.step()
        vl, secs = evaluate()
        if ep > 1:
            for n in pool:
                eval_s[n] += secs[n]
        cuda = dev.type == "cuda"
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "epoch_s": time.perf_counter() - t_ep,
               "train_s": train_s, "eval_s": sum(secs.values()), "steps": nb,
               "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(dev) / 2 ** 20 if cuda else None,
               "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(dev) / 2 ** 20 if cuda else None}
        curve.append(rec)
        log(rec)
        if vl < best:
            best, best_ep, bad, selected_sha = vl, ep, 0, weights_sha(model)
            if out_dir is not None:
                torch.save(model.state_dict(), out_dir / ".best.partial.pt")
                os.replace(out_dir / ".best.partial.pt", out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    timed = max(len(curve) - 2, 0)                                    # epochs after the first
    return {"pool": pool, "seed": seed, "config": {**CONFIG, "lr": lr, "patience": patience, "max_epochs": epochs,
                                                   "seed": seed},
            "n_params": sum(p.numel() for p in model.parameters()),
            "epochs_requested": epochs, "scheduler_t_max": max(epochs, 1),
            "started_from_given_weights": init is not None, "numerical_failure": failure,
            "selected_weights_sha256": selected_sha,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "initial_weights_sha256": initial_sha,
            "final_weights_sha256": weights_sha(model), **({"step_losses": step_losses} if record_steps else {}),
            "states": {"train_per_network": n_train, "train": int(sum(len(tr[n]["x"]) for n in pool)),
                       "train_converged": int(sum(tr[n]["conv"].sum() for n in pool)),
                       "val": int(sum(len(va[n]["x"]) for n in pool)),
                       "val_converged": int(sum(va[n]["conv"].sum() for n in pool))},
            "scalers": {"x": xs.state(), "edge": es.state(), "residual": rs.tolist(), "vpn": vpn_scale, "kvl": kvl_scale},
            "prepare_s": prep_s, "host_peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2 ** 30 if platform.system() == "Darwin" else 2 ** 20),
            "best_epoch": best_ep, "best_val_loss": best if best_ep > 0 else None, "curve": curve,
            "per_network": {n: {"retained_buses": statics[n].net.n_bus,
                                "train_ms_per_step": 1e3 * step_s[n] / step_n[n] if step_n[n] else None,
                                "val_ms_per_state": 1e3 * eval_s[n] / (timed * len(va[n]["x"])) if timed else None}
                            for n in pool}}


def projection(per_network: dict, buses: dict) -> dict:
    """Fit-only projection of the GPU hours of the §8 fits: every fit at its maximum number of epochs
    (PG-scratch at PG-net's 60), no early stop; data preparation, the initial validation pass, checkpoint
    writes, calibration, test evaluation and timing runs are excluded, so this is not a total allocation. Time
    per training step and per validation state is a straight-line fit against retained buses on the
    networks measured in the smoke, applied to all 24 networks and never below the smallest measured value."""
    m = list(per_network.values())
    x = np.array([v["retained_buses"] for v in m], float)
    fits = {}
    for key in ("train_ms_per_step", "val_ms_per_state"):
        y = np.array([v[key] for v in m], float)
        a, b = np.polyfit(x, y, 1)
        fits[key] = {"slope_per_bus": a, "intercept": b, "smallest_measured": y.min(),
                     "r2": float(1 - ((y - (a * x + b)) ** 2).sum() / ((y - y.mean()) ** 2).sum())}

    def ms(key, n):
        return max(fits[key]["slope_per_bus"] * buses[n] + fits[key]["intercept"], fits[key]["smallest_measured"])

    def epoch_s(n, n_train):                                           # one network: train steps + its 3,000 val states
        return (-(-n_train // CONFIG["batch"]) * ms("train_ms_per_step", n)
                + fs.N_STATES["val"] * ms("val_ms_per_state", n)) / 1e3

    E = CONFIG["max_epochs"]
    pre_epoch = [sum(epoch_s(n, N_PRE) for n in pool_of(f)) for f in range(4)]
    hours = {"pg_pre_12_runs": 3 * E * sum(pre_epoch) / 3600,
             "pg_net_24_fits": E * sum(epoch_s(n, fs.N_STATES["train"]) for n in fs.ELIGIBLE) / 3600,
             "pg_scratch_120_fits": E * sum(len(s) * epoch_s(n, b) for n in fs.ELIGIBLE for b, s in BUDGETS.items()) / 3600,
             "pg_ft_120_fits": FT_EPOCHS * sum(len(s) * epoch_s(n, b) for n in fs.ELIGIBLE for b, s in BUDGETS.items()) / 3600}
    return {"assumption": "fit-only projection from an empirical straight-line timing fit: maximum epochs for "
                          "every fit (PG-scratch at 60), no early stopping; data preparation, calibration, test "
                          "evaluation and timing runs excluded",
            "fits": fits, "pg_pre_epoch_s_per_fold": pre_epoch, "gpu_hours": hours,
            "gpu_hours_total": sum(hours.values())}


def smoke(label_root: Path, fold: int, epochs: int, device: str, deterministic: bool = False) -> Path:
    if epochs < 2:
        raise ValueError("the smoke needs at least 2 epochs: the first one is not timed")
    prov = start_provenance(device, set_determinism(deterministic))   # before anything is trained
    tag = f"job{prov['job_id']}" if prov["job_id"] else time.strftime("local%Y%m%dT%H%M%S")
    out = fs.RESULTS / f"fourwire_scale_smoke_pgpre_fold{fold}_{tag}.json"
    if out.exists():
        raise FileExistsError(out)
    store = fd.LabelStore(label_root)
    res = fit(store, pool_of(fold), epochs, device, out_dir=fs.OUT / "runs" / f"pgpre_fold{fold}_seed0_smoke_{tag}")
    buses = {u["network"]: u["retained_buses"] for u in json.loads(fd.MANIFEST.read_text())["splits"]}
    unchanged = checkout_unchanged(prov)
    if out.exists():
        raise FileExistsError(out)
    fs._write_json(out, {
        "protocol": "docs/fourwire_scale_protocol.md §9 gate 5 (smoke fit, train and val only)",
        **prov, "checkout_unchanged_during_run": unchanged,
        "label_commit": fd.LABEL_COMMIT, "label_root_name": Path(label_root).name, "fold": fold, "folds": folds(),
        "held_out": folds()[fold], **res, "projection": projection(res["per_network"], buses)})
    if not unchanged:
        raise RuntimeError(f"the checkout differs from the start of the run; {out.name} is not a valid record")
    return out


def fold_of(network: int) -> int:
    return next(k for k, f in enumerate(folds()) if network in f)


def run_name(kind: str, fold: int | None = None, network: int | None = None, budget: int | None = None,
             seed: int = 0) -> str:
    if kind == "pgpre":
        return f"pgpre_fold{fold}_seed{seed}"
    if kind == "pgnet":
        return f"pgnet_net{network}"
    return f"{kind}_net{network}_b{budget}_seed{seed}"


def all_fits() -> list[dict]:
    """The 276 final fits of §8: 12 PG-pre, 24 PG-net, 120 PG-scratch and 120 PG-ft."""
    fits = [{"kind": "pgpre", "fold": f, "seed": s} for f in range(4) for s in (0, 1, 2)]
    fits += [{"kind": "pgnet", "network": n, "seed": 0} for n in fs.ELIGIBLE]
    fits += [{"kind": k, "network": n, "budget": b, "seed": s} for k in ("scratch", "ft") for n in fs.ELIGIBLE
             for b, seeds in BUDGETS.items() for s in seeds]
    return fits


def software() -> dict:
    """The library versions a run uses (Amendment 7, item 5: one software stack for all final runs), with
    the optional native backends of PyTorch Geometric. cuDNN is recorded per run (start_provenance) and not
    part of this: the model has no operation that uses it, and its version cannot be read without a GPU."""
    import numba
    import scipy
    import torch_geometric
    import importlib

    def optional(module):                                            # None where it is not installed
        try:
            return importlib.import_module(module).__version__
        except ImportError:
            return None

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "torch_geometric": torch_geometric.__version__,
            "numba": numba.__version__, "opendssdirect": optional("opendssdirect"),
            **{m: optional(m) for m in ("pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster")}}


def input_hashes() -> dict:
    """What a run depends on besides the source, hashed from the files as they are now: the label manifest
    and the identification record, which holds every network's Thevenin equivalent and network-file hashes."""
    return {"label_manifest_sha256": fs._sha(fd.MANIFEST),
            "identification_sha256": fs._sha(fs.RESULTS / "fourwire_scale_identification.json")}


def variant_networks_sha() -> str:
    """SHA-256 over the converted files of every eligible network under the two rating variants (the primary
    files are checked against the identification record whenever a network is loaded)."""
    h = hashlib.sha256()
    for n in fs.ELIGIBLE:
        for variant in cc.RATING_VARIANTS:
            if variant != "primary":
                for f in sorted(fs.ensure_network_folder(n, variant).glob("*.dss")):
                    h.update(f"{n}|{variant}|{f.name}|{fs._sha(f)}|".encode())
    return h.hexdigest()


def profiles_sha() -> str:
    """Content hash of the Fluvius meter profiles (the envelope scenarios are drawn from them)."""
    from .fluvius import GROUPS, load_group
    h = hashlib.sha256()
    for name in GROUPS:
        g = load_group(name)
        h.update(fs.content_sha({"ean": g.ean, "offtake_kw": g.offtake_kw, "injection_kw": g.injection_kw}).encode())
    return h.hexdigest()


JOB_SCRIPT = fs.ROOT / "scripts/hydra_fourwire_scale_final.slurm"
FINAL_DEVICE = "NVIDIA A100-PCIE-40GB"        # §6, Amendment 7: every final run


def freeze() -> Path:
    """Amendment 8, item 14: record what every final command must run with: the committed source, the label
    manifest, the identification record, the rating-variant networks, the meter profiles, the job script and
    the library versions of the machine the final runs use. Run it there, once, on unmodified source; the
    file is then committed. A later freeze (after an amendment) names the earlier freezes whose fits or
    evaluations it keeps."""
    if freeze_file().exists():
        raise FileExistsError(f"{freeze_file().name} exists: a new freeze needs an amendment")
    state = checkout_state()
    if state["src_dirty"]:
        raise RuntimeError("src/ differs from HEAD: commit it before freezing")
    if input_hashes()["label_manifest_sha256"] != fd.MANIFEST_SHA256:
        raise AssertionError("the label manifest is not the pinned one")
    files = {q.name: fs._sha(q) for q in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    fs._write_json(freeze_file(), {"protocol": "docs/fourwire_scale_protocol.md Amendment 8, item 14",
                                   "source_sha256": state["source_sha256"], **input_hashes(),
                                   "variant_networks_sha256": variant_networks_sha(),
                                   "profiles_sha256": profiles_sha(), "job_script_sha256": fs._sha(JOB_SCRIPT),
                                   "software": software(), "device": FINAL_DEVICE, "folds": folds(),
                                   "earlier_freezes_accepted_for": {"fits": [], "evaluation": []},
                                   "frozen_at_commit": state["head"], "files_sha256": files})
    return freeze_file()


def freeze_committed() -> bool:
    """The freeze file is tracked by git and identical to HEAD."""
    f = str(freeze_file())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], capture_output=True, cwd=fs.ROOT)
    status = subprocess.run(["git", "status", "--porcelain", "--", f], capture_output=True, text=True, cwd=fs.ROOT)
    return tracked.returncode == 0 and status.returncode == 0 and not status.stdout.strip()


def halt_file() -> Path:
    return runs() / "HALT.json"


@contextlib.contextmanager
def halting(what: str, prov: dict, final: bool):
    """Amendment 7, item 3: an error inside a final run is a defect of the code or the setup. It stops every
    later final command until it is resolved under an amendment (the halt file is then removed by hand)."""
    try:
        yield
    except Exception as e:
        if final:
            halt_file().parent.mkdir(parents=True, exist_ok=True)
            fs._write_json(halt_file(), {"what": what, "error": f"{type(e).__name__}: {e}", "job_id": prov["job_id"],
                                         "node": prov["node"], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise


def check_freeze(prov: dict, deterministic: bool = True, variants: bool = False, profiles: bool = False) -> dict:
    """A final command runs only: pinned to HEAD; from unmodified source equal to the committed freeze; with
    the frozen inputs, job script and library versions; on the frozen GPU model; in the numerical mode it
    must use (deterministic algorithms, except for timing); and while no halt is in force. variants and
    profiles: also check the rating-variant networks (evaluation) and the meter profiles (envelopes).
    Binds the run to the freeze (prov["freeze_sha256"]) and returns the freeze."""
    if halt_file().exists():
        raise RuntimeError(f"the final runs are halted: {halt_file().read_text()}")
    fr = json.loads(freeze_file().read_text())
    problems = []
    if not freeze_committed():
        problems.append("the freeze file is not committed")
    if prov["gencodoe_commit"] != prov["code_commit"]:
        problems.append("GENCODOE_COMMIT is not set to HEAD")
    if prov["src_dirty"]:
        problems.append("src/ is modified")
    if prov["source_sha256"] != fr["source_sha256"]:
        problems.append("the source differs from the freeze")
    if any(fr[k] != prov[k] for k in input_hashes()) or fr["folds"] != folds():
        problems.append("label manifest, identification record or folds differ from the freeze")
    if prov["label_manifest_sha256"] != fd.MANIFEST_SHA256:
        problems.append("the label manifest is not the pinned one")
    if fs._sha(JOB_SCRIPT) != fr["job_script_sha256"]:
        problems.append("the job script differs from the freeze")
    if prov["software"] != fr["software"]:
        problems.append("library versions differ from the freeze")
    if prov["cuda_device"] != FINAL_DEVICE or fr["device"] != FINAL_DEVICE:
        problems.append(f"not on {FINAL_DEVICE}")
    if prov["deterministic_algorithms"] != deterministic or prov["tf32"]:
        problems.append("wrong numerical mode")
    if variants and variant_networks_sha() != fr["variant_networks_sha256"]:
        problems.append("rating-variant networks differ from the freeze")
    if profiles and profiles_sha() != fr["profiles_sha256"]:
        problems.append("meter profiles differ from the freeze")
    if problems:
        raise RuntimeError("not a final run: " + "; ".join(problems))
    prov["freeze_sha256"] = fs._sha(freeze_file())
    return fr


def canonical_fits() -> dict:
    """Run name -> what the fit of that name is (§4, §8): kind, fold, network, budget, seed and pool."""
    out = {}
    for a in all_fits():
        pre = a["kind"] == "pgpre"
        fold = a["fold"] if pre else fold_of(a["network"])
        out[run_name(a["kind"], a.get("fold"), a.get("network"), a.get("budget"), a["seed"])] = {
            "kind": a["kind"], "fold": fold, "network": a.get("network"), "budget": a.get("budget"),
            "seed": a["seed"], "pool": pool_of(fold) if pre else [a["network"]]}
    return out


def check_final_record(rec: dict, what: str, stage: str) -> dict:
    """A final record (fit, evaluation, timing or envelopes) can be used: valid, not a dry run, made with
    today's inputs under this freeze (then also with its source and library versions) or under an earlier
    freeze that this one accepts for the stage."""
    fr = json.loads(freeze_file().read_text())
    if not rec["valid"] or rec["dry_run"]:
        raise AssertionError(f"{what}: not a valid final record")
    if rec["freeze_sha256"] == fs._sha(freeze_file()):
        if rec["source_sha256"] != fr["source_sha256"] or rec["software"] != fr["software"]:
            raise AssertionError(f"{what}: not made with the source and libraries of its freeze")
    elif rec["freeze_sha256"] not in fr["earlier_freezes_accepted_for"].get(stage, []):
        raise AssertionError(f"{what}: made under another freeze")
    if any(rec[k] != v for k, v in input_hashes().items()):
        raise AssertionError(f"{what}: made with other inputs")
    return rec


def check_record(rec: dict, name: str, dry: bool) -> dict:
    """A fit record is the one asked for: the fit that the name stands for in §8 (kind, fold, network, budget,
    seed, pool), valid, and (final) usable under this freeze (check_final_record)."""
    want = canonical_fits().get(name)
    if want is None or rec["name"] != name or any(rec[k] != v for k, v in want.items()):
        raise AssertionError(f"{name}: the record is not that fit (it says {rec['name']})")
    if dry:
        if not rec["valid"] or rec["dry_run"] is None:
            raise AssertionError(f"{name}: not a valid dry-run fit")
        return rec
    return check_final_record(rec, name, "fits")


def completed(folder: Path, validate) -> bool:
    """Does a run folder hold a completed run? If its record says valid, the record must pass validate
    (which raises otherwise): a run is skipped only when what it left can be used."""
    f = folder / "record.json"
    if not f.exists():
        return False
    rec = json.loads(f.read_text())
    if not rec["valid"]:
        return False
    validate(rec)
    return True


def publish(folder: Path, out: Path) -> None:
    """Copy a run folder's record to results/, atomically; also after a crash between the two writes."""
    data = (folder / "record.json").read_bytes()
    if not out.exists() or out.read_bytes() != data:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name("." + out.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, out)


def fit_record(base: Path, name: str, dry: bool) -> dict:
    """The record of a completed fit, checked (check_record), with its checkpoint where it has one."""
    rec = check_record(json.loads((base / name / "record.json").read_text()), name, dry)
    ck = base / name / "best.pt"
    if (rec["checkpoint_sha256"] is None) == ck.exists() or (ck.exists() and fs._sha(ck) != rec["checkpoint_sha256"]):
        raise AssertionError(f"{name}: the checkpoint is not the one its record names")
    return rec


def load_run(base: Path, name: str, device: str, dry: bool) -> dict:
    """Record and selected weights of a completed fit that has a model."""
    rec = fit_record(base, name, dry)
    if rec["checkpoint_sha256"] is None:
        raise AssertionError(f"{name}: the fit has no model")
    return {"record": rec, "state_dict": torch.load(base / name / "best.pt", map_location=device, weights_only=True)}


MAX_RERUNS = 2                     # Amendment 7, item 3
RERUN_STATES = ("FAILED", "NODE_FAIL", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL")


class AlreadyDone(FileExistsError):
    """The run has a valid record: it is never run again."""


def _slurm(command: list[str]) -> str:
    """Output of a Slurm query. Without an answer nothing is assumed: the caller refuses."""
    try:
        out = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{command[0]} is not available here: the attempt's job cannot be checked") from e
    if out.returncode != 0 and "Invalid job id" not in out.stderr:
        raise RuntimeError(f"{command[0]} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def _attempt_running(a: dict) -> bool:
    """Is the attempt still running? Its Slurm job is asked for (SLURM_JOB_ID is one id per array task);
    without a Slurm job, its process on this machine."""
    if a["job_id"]:
        return bool(_slurm(["squeue", "-h", "-j", a["job_id"].split("_")[0]]))
    if a["node"] != platform.node():
        raise RuntimeError(f"the attempt ran on {a['node']}: it cannot be checked from here")
    try:
        os.kill(a["pid"], 0)
        return a["pid"] != os.getpid()
    except OSError:
        return False


def rerun_reason(a: dict) -> str:
    """Why an attempt that left no record may be run again (Amendment 7, item 3). Raises if it may not.
    - A Slurm job: accounting must show an infrastructure end state, or a cancellation that names another
      account than ours.
    - Our own cancellation: only with GENCODOE_RERUN_DEVIATION set to its reason, which is recorded.
    - No Slurm job (a run on a workstation): its process has ended."""
    if not a["job_id"]:
        return "no Slurm job; its process has ended"
    state = _slurm(["sacct", "-n", "-X", "-j", a["job_id"].split("_")[0], "-o", "State%40"])
    words = state.split()
    word = words[0] if words else ""
    by = words[2] if len(words) == 3 and words[1] == "by" else None   # "CANCELLED by <uid>"
    if word in RERUN_STATES or (word == "CANCELLED" and by is not None and by != str(os.getuid())):
        return f"Slurm accounting: {state}"
    deviation = os.environ.get("GENCODOE_RERUN_DEVIATION")
    if word == "CANCELLED" and deviation:
        return f"DEVIATION, cancelled by us ({state}): {deviation}"
    raise RuntimeError(f"Slurm accounting says {state or 'nothing'} for job {a['job_id']}: no reason for a rerun")


def _curve(folder: Path) -> dict:
    """epoch -> (train loss, validation loss) of the epochs an attempt recorded."""
    f = folder / "curve.jsonl"
    rows = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
    return {r["epoch"]: (r.get("train_loss"), r["val_loss"]) for r in rows if "epoch" in r}


def claim_run_folder(base: Path, name: str, prov: dict, final: bool) -> tuple[Path, list[dict]]:
    """The run folder of a new attempt and what is known of the earlier attempts of the same run.
    - A run with a valid record is never run again (AlreadyDone).
    - A folder with an invalid record is set aside: the record states why it is invalid.
    - A folder without a record is set aside only if its attempt has ended and, for a final run, if
      rerun_reason allows it. A folder without an owner is refused.
    - At most MAX_RERUNS earlier attempts.
    - The new folder appears with its owner already in it, by one rename: two attempts cannot both start,
      and no folder is ever without an owner."""
    out_dir = base / name
    base.mkdir(parents=True, exist_ok=True)
    lock = base / f".{name}.lock"
    try:
        lock.mkdir()                                                 # one claimant at a time, from inspection to claim
    except FileExistsError as e:
        raise RuntimeError(f"{name}: another attempt is being claimed ({lock.name} exists; if no process holds "
                           "it, removing it is a recorded decision)") from e
    try:
        earlier = [q for q in sorted(base.glob(f"{name}.*")) if q.is_dir()]
        if out_dir.exists():
            rec, owner = out_dir / "record.json", out_dir / "attempt.json"
            if rec.exists() and json.loads(rec.read_text())["valid"]:
                raise AlreadyDone(f"{name} has a valid record: it is never run again")
            if not owner.exists():
                raise RuntimeError(f"{name}: a run folder without an owner; it needs a recorded decision")
            if len(earlier) >= MAX_RERUNS:
                if final:                                            # Amendment 7, item 3: a third failure stops the runs
                    halt_file().parent.mkdir(parents=True, exist_ok=True)
                    fs._write_json(halt_file(), {"what": name, "error": f"{len(earlier)} earlier attempts",
                                                 "job_id": prov["job_id"], "node": prov["node"],
                                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                raise RuntimeError(f"{name}: {len(earlier)} earlier attempts; a further rerun needs an amendment")
            a = json.loads(owner.read_text())
            if rec.exists():
                kind, reason = "invalid", "its record is invalid"
            else:
                if _attempt_running(a):
                    raise RuntimeError(f"{name}: another attempt of this run is running")
                kind, reason = "incomplete", rerun_reason(a) if final else "dry run"
            k = 1
            while (base / f"{name}.{kind}{k}").exists():
                k += 1
            fs._write_json(owner, {**a, "set_aside_because": reason})
            out_dir.rename(base / f"{name}.{kind}{k}")
            earlier.append(base / f"{name}.{kind}{k}")
        known = []
        for q in sorted(earlier):
            a = json.loads((q / "attempt.json").read_text())
            known.append({"folder": q.name, "job_id": a["job_id"], "started": a["started"],
                          "set_aside_because": a.get("set_aside_because")})
        tmp = base / f".{name}.claim{os.getpid()}"
        tmp.mkdir()
        fs._write_json(tmp / "attempt.json", {"job_id": prov["job_id"], "node": prov["node"], "pid": os.getpid(),
                                              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        try:
            os.rename(tmp, out_dir)                                  # the folder appears with its owner in it
        except OSError as e:
            (tmp / "attempt.json").unlink()
            tmp.rmdir()
            raise RuntimeError(f"{name}: another attempt of this run has just started") from e
        return out_dir, known
    finally:
        lock.rmdir()


def final_fit(label_root: Path, kind: str, device: str, fold: int | None = None, network: int | None = None,
              budget: int | None = None, seed: int = 0, dry: dict | None = None) -> Path:
    """One fit of §8 (Amendments 7 and 8). dry: None, or {"n_train", "n_val", "epochs"} for a dry run on
    train and val only, kept under runs/dry and never evaluated on test."""
    cfg, final = FITS[kind], dry is None
    name = run_name(kind, fold, network, budget, seed)
    base = runs(not final)
    prov = start_provenance(device, set_determinism(True))          # before anything is trained
    if final:
        check_freeze(prov)
    if completed(base / name, lambda rec: fit_record(base, name, not final)):
        raise AlreadyDone(f"{name} has a valid record: it is never run again")
    start = init = None
    if kind == "pgpre":
        pool, n_train = pool_of(fold), N_PRE
    else:
        pool, n_train = [network], fs.N_STATES["train"] if kind == "pgnet" else budget
        if kind == "ft":                                             # the selected PG-pre checkpoint, same seed;
            pre_name = run_name("pgpre", fold=fold_of(network), seed=seed)   # it must exist before PG-ft claims anything
            start = {"record": fit_record(base, pre_name, not final)}
            if start["record"]["checkpoint_sha256"] is not None:
                start = load_run(base, pre_name, device, not final)
                init = {"state_dict": start["state_dict"], "scalers": start["record"]["scalers"]}
    out_dir, earlier = claim_run_folder(base, name, prov, final)
    with halting(name, prov, final):
        store = fd.LabelStore(label_root)
        epochs, n_val = cfg["epochs"], None
        if not final:
            n_train, n_val, epochs = min(n_train, dry["n_train"]), dry["n_val"], dry["epochs"]
        t0 = time.perf_counter()
        if start is not None and init is None:                       # PG-pre ended without a model: neither has PG-ft
            res = {"pool": pool, "seed": seed, "best_epoch": -1, "curve": [], "scalers": None,
                   "numerical_failure": {"what": f"{start['record']['name']} has no model"},
                   "initial_weights_sha256": None, "selected_weights_sha256": None}
        else:
            res = fit(store, pool, epochs, device, seed=seed, out_dir=out_dir, n_train=n_train, n_val=n_val,
                      lr=cfg["lr"], patience=cfg["patience"], init=init, prepared=True)
        if start is not None and res["initial_weights_sha256"] != start["record"]["selected_weights_sha256"]:
            raise AssertionError("PG-ft did not start from the selected PG-pre weights")
        unchanged = checkout_unchanged(prov)
        ck = out_dir / "best.pt"
        new = {c["epoch"]: (c.get("train_loss"), c["val_loss"]) for c in res["curve"]}
        for a in earlier:                                            # a rerun must retrace what was recorded before
            old = _curve(base / a["folder"])
            a.update({"epochs_recorded": len(old), "reproduced_by_this_run": all(new.get(e) == v for e, v in old.items())})
        rec = {"protocol": "docs/fourwire_scale_protocol.md §4, §8, Amendments 5, 7 and 8", "name": name,
               "kind": kind, "fold": fold if kind == "pgpre" else fold_of(network), "network": network,
               "budget": budget, "dry_run": dry, **prov,
               "unchanged_during_run": unchanged, "label_commit": fd.LABEL_COMMIT,
               "started_from": None if start is None else {"name": start["record"]["name"],
                                                             "checkpoint_sha256": start["record"]["checkpoint_sha256"]},
               "earlier_attempts": earlier, "fit_s": time.perf_counter() - t0, **res,
               "checkpoint_sha256": fs._sha(ck) if ck.exists() else None}
        rec["valid"] = bool(unchanged and prov["deterministic_algorithms"])
        fs._write_json(out_dir / "record.json", rec)
        if not unchanged:
            raise RuntimeError(f"checkout, numerical mode, libraries or inputs changed during the run; "
                               f"{name} is not a valid fit")
    return out_dir / "record.json"


def collect() -> Path:
    """The records of the final fits, gathered for git (checkpoints stay outside it). Not during a halt."""
    if halt_file().exists():
        raise RuntimeError(f"the final runs are halted: {halt_file().read_text()}")
    recs, missing = [], []
    for a in all_fits():
        f = runs() / run_name(a["kind"], a.get("fold"), a.get("network"), a.get("budget"), a["seed"]) / "record.json"
        if f.exists():
            recs.append(json.loads(f.read_text()))
        else:
            missing.append(f.parent.name)
    index = fs.RESULTS / "fourwire_scale_fits.json"
    tmp = index.with_name("." + index.name + ".partial")
    tmp.write_text(json.dumps({"protocol": "docs/fourwire_scale_protocol.md §8, Amendments 7 and 8",
                               "fits_expected": len(all_fits()), "fits": len(recs),
                               "valid": sum(r["valid"] for r in recs), "missing": missing, "records": recs}))
    os.replace(tmp, index)
    return index


PROBE_RUNS = ("default_a", "default_b", "deterministic_a", "deterministic_b")
PROBE_STEPS = 32


def determinism(label_root: Path, device: str, deterministic: bool, out: Path) -> Path:
    """32 training steps on networks 10 and 17 (500 train and 500 val states each), every step loss kept.
    Two processes with the same settings are bit-identical only if training is deterministic. An error is
    recorded with its type; only a missing deterministic implementation says something about determinism."""
    if out.exists():
        raise FileExistsError(out)
    prov = start_provenance(device, set_determinism(deterministic))
    rec = {**prov, "deterministic_requested": deterministic}
    try:
        res = fit(fd.LabelStore(label_root), [10, 17], 1, device, n_train=500, n_val=500, record_steps=True)
        rec.update({k: res[k] for k in ("initial_weights_sha256", "final_weights_sha256", "step_losses",
                                        "epochs_requested", "scheduler_t_max")})
        rec["val_loss"] = [c["val_loss"] for c in res["curve"]]
    except RuntimeError as e:
        rec.update({"error_type": type(e).__name__, "error": str(e),
                    "missing_deterministic_implementation": "deterministic" in str(e).lower()})
    rec["checkout_unchanged_during_run"] = checkout_unchanged(prov)
    fs._write_json(out, rec)
    return out


def determinism_report(folder: Path, out: Path) -> dict:
    """Compare the two default runs with each other and the two deterministic runs with each other. A pair
    gets a verdict only if both runs are valid: pinned commit equal to HEAD, unmodified and unchanged
    source, the requested setting in effect, all 32 step losses and both validation losses finite, no
    error, and the four runs share commit, source, node, GPU and library versions."""
    if out.exists():
        raise FileExistsError(out)
    runs = {k: json.loads((folder / f"{k}.json").read_text()) for k in PROBE_RUNS}

    def problems(k):
        r, p = runs[k], []
        if "error" in r:
            p.append(f"error: {r['error_type']}")
        elif (len(r["step_losses"]) != PROBE_STEPS or not np.isfinite(r["step_losses"]).all()
              or len(r["val_loss"]) != 2 or not np.isfinite(r["val_loss"]).all()):
            p.append("step or validation losses incomplete or not finite")
        if r["gencodoe_commit"] != r["code_commit"]:
            p.append("commit not pinned")
        if r["src_dirty"] or not r["checkout_unchanged_during_run"]:
            p.append("source modified or checkout changed")
        if r["deterministic_algorithms"] != k.startswith("deterministic"):
            p.append("requested setting not in effect")
        return p

    probs = {k: problems(k) for k in PROBE_RUNS}
    build = ("code_commit", "source_sha256", "node", "cuda_device", "python", "numpy", "torch", "torch_geometric",
             "cuda", "cudnn")
    same_build = len({tuple(runs[k][b] for b in build) for k in PROBE_RUNS}) == 1
    mode = ("deterministic_algorithms", "cublas_workspace_config", "cudnn_benchmark", "cudnn_deterministic", "tf32")

    def pair(a, b):
        x, y = runs[a], runs[b]
        if probs[a] or probs[b] or not same_build or any(x.get(k) != y.get(k) for k in mode):
            return {"valid": False}
        d = np.abs(np.array(x["step_losses"]) - np.array(y["step_losses"]))
        return {"valid": True, "same_initial_weights": x["initial_weights_sha256"] == y["initial_weights_sha256"],
                "same_final_weights": x["final_weights_sha256"] == y["final_weights_sha256"],
                "identical_step_losses": x["step_losses"] == y["step_losses"],
                "identical_val_losses": x["val_loss"] == y["val_loss"], "steps": len(d),
                "first_step_that_differs": int(np.flatnonzero(d > 0)[0]) if (d > 0).any() else None,
                "max_abs_step_loss_difference": float(d.max())}

    rep = {"what": "two processes per setting, 32 PG-pre training steps on networks 10 and 17 (train and val "
                   "only); a falsification probe, not a certificate for a full fit",
           "problems": probs, "same_commit_source_node_gpu_and_libraries": same_build,
           "default": pair("default_a", "default_b"),
           "deterministic_algorithms": pair("deterministic_a", "deterministic_b"), "runs": runs}
    fs._write_json(out, rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "freeze", "collect", "smoke", "determinism", "determinism-report"])
    ap.add_argument("--label-root", type=Path, help="folder holding <split>/network_N/ label shards")
    ap.add_argument("--kind", choices=list(FITS), help="fit: which fit of §4")
    ap.add_argument("--fold", type=int, default=0, choices=range(4), help="fit --kind pgpre, smoke")
    ap.add_argument("--network", type=int, choices=fs.ELIGIBLE, help="fit: the target network")
    ap.add_argument("--budget", type=int, choices=list(BUDGETS), help="fit --kind scratch|ft: B")
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--dry-train", type=int, help="fit: dry run with at most this many train states per network")
    ap.add_argument("--dry-val", type=int, help="fit: dry run, val states per network")
    ap.add_argument("--dry-epochs", type=int, help="fit: dry run, epochs")
    ap.add_argument("--epochs", type=int, default=3, help="smoke")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deterministic", action="store_true", help="smoke, determinism: deterministic algorithms")
    ap.add_argument("--out", type=Path, help="determinism, determinism-report: output file")
    ap.add_argument("--out-dir", type=Path, help="determinism-report: folder with the four probe runs")
    a = ap.parse_args()
    if a.cmd == "determinism-report":
        rep = determinism_report(a.out_dir, a.out)
        print(json.dumps({k: rep[k] for k in rep if k != "runs"}, indent=1))
        sys.exit(0 if rep["default"]["valid"] and rep["deterministic_algorithms"]["valid"] else 4)
    if a.cmd in ("freeze", "collect"):
        print((freeze if a.cmd == "freeze" else collect)())
        return
    if a.label_root is None:
        ap.error("--label-root is required")
    if a.cmd == "fit":
        dry = (a.dry_train, a.dry_val, a.dry_epochs)
        if any(v is not None for v in dry) and None in dry:
            ap.error("a dry run needs --dry-train, --dry-val and --dry-epochs")
        if a.kind is None or (a.kind == "pgpre") == (a.network is not None) \
                or (a.kind in ("scratch", "ft")) != (a.budget is not None):
            ap.error("--kind pgpre takes --fold; pgnet takes --network; scratch and ft take --network and --budget")
        if a.kind != "pgpre" and a.seed not in ([0] if a.kind == "pgnet" else BUDGETS[a.budget]):
            ap.error("that seed is not among the fits of §4")
        try:
            print(final_fit(a.label_root, a.kind, a.device, a.fold if a.kind == "pgpre" else None, a.network,
                            a.budget, a.seed,
                            None if dry[0] is None else {"n_train": dry[0], "n_val": dry[1], "epochs": dry[2]}))
        except AlreadyDone as e:                                     # the job script reads 10 as "skip"
            print(e)
            sys.exit(10)
    elif a.cmd == "determinism":
        print(determinism(a.label_root, a.device, a.deterministic, a.out))
    else:
        print(smoke(a.label_root, a.fold, a.epochs, a.device, a.deterministic))


if __name__ == "__main__":
    main()
