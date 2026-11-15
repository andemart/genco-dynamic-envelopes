"""PG-pre for the 25-network study (docs/fourwire_scale_protocol.md §4, §9 gate 5).

    python -m gencodoe.fourwire_scale_train smoke --label-root <dir> --fold 0 --epochs 3 --device cuda
    python -m gencodoe.fourwire_scale_train determinism --label-root <dir> --device cuda [--deterministic] --out <file>
    python -m gencodoe.fourwire_scale_train determinism-report --out-dir <dir with the four runs> --out <file>

PhaseGraph (fourwire_model.GNS_heterogeneous_4wire) pretrained on the 18 networks outside a fold: the first
5,000 train states of each, validation on their val states, with the ENWL1 amendment 1 configuration and
loss of fourwire_train, unchanged. Labels are read through fourwire_scale_data.LabelStore: train and val
only, from the label root given on the command line.

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
    """Switch PyTorch's deterministic algorithms on or off explicitly. Call before the first CUDA call.
    Returns the settings in effect, not the request."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def start_provenance(device: str, settings: dict) -> dict:
    """Read before a run. GENCODOE_COMMIT, set by the job script, must equal HEAD."""
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
            "cuda": torch.version.cuda if cuda else None,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "cuda_device": torch.cuda.get_device_name(dev) if cuda else None,
            "cuda_total_mb": torch.cuda.get_device_properties(dev).total_memory / 2 ** 20 if cuda else None}


def checkout_unchanged(start: dict) -> bool:
    """HEAD, the modified flag and the source hash are the same as at the start."""
    state = checkout_state()
    return (state["head"] == start["code_commit"] and state["src_dirty"] == start["src_dirty"]
            and state["source_sha256"] == start["source_sha256"])


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


def pretrain(store: fd.LabelStore, pool: list[int], epochs: int, device: str, seed: int = 0,
             out_dir: Path | None = None, n_train: int | None = None, n_val: int | None = None,
             record_steps: bool = False) -> dict:
    """One PG-pre fit on the first N_PRE train states and the val states of every pool network. Batches
    hold states of one network (the model's graph context needs equal graph sizes); an epoch visits every
    training state once, with the batches of all networks shuffled. out_dir must not exist: a run folder is
    never overwritten. n_train, n_val and record_steps serve the determinism probe. Deterministic
    algorithms are switched by set_determinism before this is called."""
    n_train = N_PRE if n_train is None else n_train
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=False)
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
    xs = PooledScaler(tr[n]["x"] for n in pool)                       # pool training statistics only (§4)
    es = PooledScaler(statics[n].edge_raw for n in pool)
    rs, vpn_scale, kvl_scale = pooled_scales(tr, statics)
    for d in (tr, va):
        for n in pool:
            for k in range(0, len(d[n]["x"]), 256):
                d[n]["x"][k:k + 256] = xs(d[n]["x"][k:k + 256])
    prep_s = time.perf_counter() - t0

    model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE, hidden=CONFIG["hidden"], layers=CONFIG["layers"],
                                       heads=CONFIG["heads"]).to(dev)
    initial_sha = weights_sha(model)
    batchers = {n: fm.Batcher(statics[n], es(statics[n].edge_raw), dev) for n in pool}
    physics = {n: fm.Physics(statics[n], dev) for n in pool}
    rs_t = torch.as_tensor(rs, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
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
    curve, best, best_ep, bad = [{"epoch": 0, "val_loss": v0, "note": "untrained = U1 prior"}], np.inf, -1, 0
    print(json.dumps(curve[-1]), flush=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_s, step_n, eval_s = {n: 0.0 for n in pool}, {n: 0 for n in pool}, {n: 0.0 for n in pool}
    step_losses = []
    for ep in range(1, epochs + 1):
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
                raise FloatingPointError("non-finite gradient")
            opt.step()
            tl += loss.item()                                        # synchronises: the step time is complete
            nb += 1
            if record_steps:
                step_losses.append(loss.item())
            if ep > 1:                                               # epoch 1 carries warm-up and graph caches
                step_s[n] += time.perf_counter() - t
                step_n[n] += 1
        train_s = time.perf_counter() - t_ep
        sched.step()
        vl, secs = evaluate()
        if ep > 1:
            for n in pool:
                eval_s[n] += secs[n]
        cuda = dev.type == "cuda"
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "epoch_s": time.perf_counter() - t_ep,
               "train_s": train_s, "eval_s": sum(secs.values()), "steps": nb,
               "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None,
               "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20 if cuda else None}
        curve.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            if out_dir is not None:
                torch.save(model.state_dict(), out_dir / ".best.partial.pt")
                os.replace(out_dir / ".best.partial.pt", out_dir / "best.pt")
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    timed = max(len(curve) - 2, 0)                                    # epochs after the first
    return {"pool": pool, "seed": seed, "config": CONFIG, "n_params": sum(p.numel() for p in model.parameters()),
            "epochs_requested": epochs, "scheduler_t_max": max(epochs, 1),
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
            "best_epoch": best_ep, "best_val_loss": best, "curve": curve,
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
    res = pretrain(store, pool_of(fold), epochs, device, out_dir=fs.OUT / "runs" / f"pgpre_fold{fold}_seed0_smoke_{tag}")
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
        res = pretrain(fd.LabelStore(label_root), [10, 17], 1, device, n_train=500, n_val=500, record_steps=True)
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
    source, the requested setting in effect, all 32 step losses finite, no error, and the four runs share
    commit, source, node, GPU and library versions."""
    if out.exists():
        raise FileExistsError(out)
    runs = {k: json.loads((folder / f"{k}.json").read_text()) for k in PROBE_RUNS}

    def problems(k):
        r, p = runs[k], []
        if "error" in r:
            p.append(f"error: {r['error_type']}")
        elif len(r["step_losses"]) != PROBE_STEPS or not np.isfinite(r["step_losses"]).all():
            p.append("step losses incomplete or not finite")
        if r["gencodoe_commit"] != r["code_commit"]:
            p.append("commit not pinned")
        if r["src_dirty"] or not r["checkout_unchanged_during_run"]:
            p.append("source modified or checkout changed")
        if r["deterministic_algorithms"] != k.startswith("deterministic"):
            p.append("requested setting not in effect")
        return p

    probs = {k: problems(k) for k in PROBE_RUNS}
    build = ("code_commit", "source_sha256", "node", "cuda_device", "torch", "torch_geometric", "cuda", "cudnn")
    same_build = len({tuple(runs[k][b] for b in build) for k in PROBE_RUNS}) == 1

    def pair(a, b):
        if probs[a] or probs[b] or not same_build:
            return {"valid": False}
        x, y = runs[a], runs[b]
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
    ap.add_argument("cmd", choices=["smoke", "determinism", "determinism-report"])
    ap.add_argument("--label-root", type=Path, help="folder holding <split>/network_N/ label shards")
    ap.add_argument("--fold", type=int, default=0, choices=range(4))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deterministic", action="store_true", help="use PyTorch's deterministic algorithms")
    ap.add_argument("--out", type=Path, help="determinism, determinism-report: output file")
    ap.add_argument("--out-dir", type=Path, help="determinism-report: folder with the four probe runs")
    a = ap.parse_args()
    if a.cmd == "determinism-report":
        rep = determinism_report(a.out_dir, a.out)
        print(json.dumps({k: rep[k] for k in rep if k != "runs"}, indent=1))
        sys.exit(0 if rep["default"]["valid"] and rep["deterministic_algorithms"]["valid"] else 4)
    if a.label_root is None:
        ap.error("--label-root is required")
    if a.cmd == "determinism":
        print(determinism(a.label_root, a.device, a.deterministic, a.out))
    else:
        print(smoke(a.label_root, a.fold, a.epochs, a.device, a.deterministic))


if __name__ == "__main__":
    main()
