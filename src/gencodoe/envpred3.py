"""Envelope prediction v3 (docs/envpred_protocol.md, amendments 1-3a; frozen prospective analysis within an
already explored benchmark). GENCO is evaluated as a learned proposal engine for expensive four-wire envelope
optimisation; exact OpenDSS remains the final decision authority.

    python -m gencodoe.envpred3 smoke     --device cuda               # fixed 12-interval smoke test
    python -m gencodoe.envpred3 features  --device cuda --chunk i --n-chunks k
    python -m gencodoe.envpred3 integrity                              # label integrity, before fitting
    python -m gencodoe.envpred3 fit
    python -m gencodoe.envpred3 gate      --file-index i               # one test file, all arms x shrink points
    python -m gencodoe.envpred3 report    --out results/envpred3_report.json

Feature extraction is keyed by (date, household seed): it replays the design scenarios from the seeds and
never opens a B file, OpenDSS or any label. L_ac (block C) is read from the label file's `ac` row only when
the training table is assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .envpred import ANNUAL, B, CAPS_KW, LABELS, N_S, N_U, SCENARIO_SEED, SPLIT, W1, replay, stats

OUT = Path("data_out/envpred")
FEAT = OUT / "features_v3"
GATE = OUT / "gate_v3"
CONFIG = Path("configs/genco_pf_enwl1_lv_prior_v2.yaml")
CALIB = Path("data_out/datakit_v2_calib")
RUN_LOCAL = Path("mlruns_hydra/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee")
RUN_HYDRA = Path("mlruns/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee")
RECOVERY_MANIFEST = Path("results/envpred_label_recovery_manifest.json")     # amendment 3c
LABEL_RUN_MANIFEST = Path("results/run_manifest_envpred_labels_recovery.json")
LABEL_INTEGRITY = Path("results/envpred_label_integrity_recovery.json")
PRED_NAME, FIT_NAME, RUN_NAME = "predictions_v3_valid.parquet", "fit_v3_valid.json", "fit_v3_valid_run.json"
SENTINEL_MEAN_SHARE = 0.2   # dataset-specific catastrophic engine-failure sentinel (amendment 3c), not a physical rule

STATS = ("mean", "min", "p02", "p10", "p90", "p98", "max")          # order of envpred.stats
Q4 = ("vmin", "vmax", "load", "trafo")
CURVE = (0.25, 0.50, 0.75, 1.00)                                      # fractions of the direction cap
N_GROUPS, PHASES, EMB = 12, (1, 2, 3), 96
SHRINK_Q = (0.80, 0.90, 0.95, 0.99)
PRIMARY_Q = 0.95
DIRS = ("export", "import")
ARMS = {"P": ("P",), "PC": ("P", "C"), "PG": ("P", "G", "GA"), "PCG": ("P", "C", "G", "GA")}
FAMILY = {"non_genco": ("P", "PC"), "genco": ("PG", "PCG")}
STRATA = {"familiar": (101, 202), "unseen": (303, 404, 505), "combined": (101, 202, 303, 404, 505)}
SMOKE = [("2024-01-01", 101, 8, "import"), ("2024-01-02", 202, 13, "export"), ("2024-02-05", 101, 18, "import"),
         ("2024-03-01", 202, 12, "export"), ("2024-04-01", 101, 11, "export"), ("2024-04-21", 202, 19, "import"),
         ("2024-05-11", 101, 14, "export"), ("2024-06-30", 202, 13, "export"), ("2024-07-21", 101, 20, "import"),
         ("2024-08-10", 202, 10, "import"), ("2024-10-02", 101, 12, "export"), ("2024-11-13", 202, 18, "import")]


# ----------------------------------------------------------------------------------------------- schema
def _stat_names(prefix: str, series) -> list[str]:
    return [f"{prefix}:{s}:{x}" for s in STATS for x in series]


def block_names() -> dict[str, list[str]]:
    p_series = ([f"grp{g:02d}_{k}" for g in range(N_GROUPS) for k in ("P", "Q")]
                + [f"ph{g}_{k}" for g in PHASES for k in ("P", "Q")] + ["tot_P", "tot_Q"])
    emb = [f"mean{i:02d}" for i in range(EMB)] + [f"max{i:02d}" for i in range(EMB)]
    ga = (["GA:L_G"] + _stat_names("GA", [f"LG_{q}" for q in Q4]) + _stat_names("GA", [f"dLG_{q}" for q in Q4])
          + _stat_names("GA", [f"hLG_{q}" for q in Q4]) + ["GA:share_LG", "GA:LG_is_zero"]
          + [f"GA:share_c{int(f * 100):03d}" for f in (0.0,) + CURVE]
          + [n for f in CURVE for n in _stat_names("GA", [f"c{int(f * 100):03d}_{q}" for q in Q4])]
          + [f"GA:embLG_{e}" for e in emb] + [f"GA:dembLG_{e}" for e in emb])
    return {"P": _stat_names("P", p_series),
            "C": ["C:L_ac"] + _stat_names("C", [f"bal0_{q}" for q in Q4]),
            "G": _stat_names("G", [f"genco0_{q}" for q in Q4]) + [f"G:emb0_{e}" for e in emb],
            "GA": ga}


def feature_names(arm: str) -> list[str]:
    b = block_names()
    return [n for blk in ARMS[arm] for n in b[blk]]


def schema_hash() -> str:
    return hashlib.sha256(json.dumps({a: feature_names(a) for a in ARMS}).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- GENCO
class GencoEmbed:
    """(feeder, p, q) -> (PFResult, pooled embedding (n, 192): mean and max over buses of the bus-head input)."""

    def __init__(self, oracle):
        self.oracle = oracle

    def __call__(self, feeder, p, q):
        captured = []
        hook = self.oracle.model.model.mlp_bus.register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))
        try:
            pr = self.oracle(feeder, p, q)
        finally:
            hook.remove()
        import torch
        # the bus head runs after every layer (physics feedback); keep the final layer of each forward chunk
        n_chunks = -(-len(p) // self.oracle.batch_size)
        per = len(captured) // n_chunks
        h = torch.cat(captured[per - 1::per]).reshape(len(p), feeder.n_bus, -1)
        if h.shape[-1] != EMB:
            raise AssertionError(f"embedding width {h.shape[-1]} != {EMB}")
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=1).cpu().numpy().astype(float)
        return pr, pooled


def load_genco(device: str, run: Path):
    import torch

    from .genco import GencoModel, GencoOracle, read_raw
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False          # the frozen A100 FP32 inference path (amendment 3b)
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    model = GencoModel(CONFIG, run / "artifacts/model/best_model_state_dict.pt",
                       run / "artifacts/stats/normalizer_stats.pt", device=device)
    oracle = GencoOracle(model, read_raw(CALIB / "enwl1_fluvius" / "raw", scenarios=[0]), batch_size=1024)
    return oracle, GencoEmbed(oracle)


# ----------------------------------------------------------------------------------------------- features
def inject(customers, n_bus: int, level_pu, direction: str) -> np.ndarray:
    """Bus injection (pu) of a uniform level per customer: + for export, - for import. level_pu scalar or (k,)."""
    from .envelopes import DIRECTIONS
    lv = np.atleast_1d(np.asarray(level_pu, float))
    return DIRECTIONS[direction] * lv[:, None] * customers.incidence(n_bus).sum(0)[None, :]


def _summary(pr) -> np.ndarray:
    from .four_wire import state_summary
    return np.column_stack(state_summary(pr))


def ga_block(oracle, embed, fd, customers, p, q, direction, zero_sum, zero_emb, share0, timer: dict) -> np.ndarray:
    """The 588 GA features of one interval and direction (amendments 3 / 3a). Balanced GENCO only."""
    from . import envelopes
    if p.shape[0] != N_S:
        raise ValueError(f"GA needs exactly {N_S} design scenarios, got {p.shape[0]}")
    cap_pu = float(customers.cap(direction)[0])
    t = time.perf_counter()
    env = envelopes.scalar_envelope(envelopes.exact(oracle, fd), fd.n_bus, customers, p, q, direction, "uniform",
                                    0.05, 4, beta=0.05)
    timer["LG_search"] = timer.get("LG_search", 0.0) + time.perf_counter() - t
    L = float(np.clip(env.level, 0.0, cap_pu))
    t = time.perf_counter()
    if L > 0:
        pr, emb = embed(fd, p + inject(customers, fd.n_bus, L, direction), q)
        s_L, share_L = _summary(pr), float(envelopes.is_feasible(fd, pr, direction).mean())
    else:                                                    # zero sentinel: the response is the zero state
        s_L, emb, share_L = zero_sum, zero_emb, share0
    timer["LG_response_emb"] = timer.get("LG_response_emb", 0.0) + time.perf_counter() - t
    vmin, vmax = fd.vmin, fd.vmax
    head = np.column_stack([s_L[:, 0] - vmin, vmax - s_L[:, 1], 1.0 - s_L[:, 2], 1.0 - s_L[:, 3]])
    t = time.perf_counter()
    lv = np.array(CURVE) * cap_pu
    pc = (p[None] + inject(customers, fd.n_bus, lv, direction)[:, None, :]).reshape(-1, fd.n_bus)
    prc = oracle(fd, pc, np.tile(q, (len(CURVE), 1)))
    ok = envelopes.is_feasible(fd, prc, direction).reshape(len(CURVE), N_S)
    sc = _summary(prc).reshape(len(CURVE), N_S, 4)
    timer["curve"] = timer.get("curve", 0.0) + time.perf_counter() - t
    ga = np.concatenate([[L * 1e3 * fd.base_mva], stats(s_L), stats(s_L - zero_sum), stats(head), [share_L, float(L == 0)],
                         [share0], ok.mean(1), *[stats(sc[i]) for i in range(len(CURVE))],
                         emb.mean(0), emb.mean(0) - zero_emb.mean(0)])
    if ga.shape != (588,):
        raise AssertionError(ga.shape)
    return ga


def extract_block(day: str, hh: int, oracle, embed, steps=None) -> dict:
    """All feature blocks of one (date, placement): P, C (L_ac left NaN), G, GA per direction. Opens no label file."""
    from . import envelopes
    from . import feeder as fdr
    from .doe_day import customer_groups
    from .enwl import load_enwl1
    from .option2 import phase_features
    from .unbalanced import UnbalancedENWL1
    net = load_enwl1()
    fd = net.to_feeder()
    ub = UnbalancedENWL1()                                    # only for customer phases (Loads.dss); never solved
    groups = customer_groups(net, ub)
    phase = np.array([int(lb.split(".")[1]) for lb in ub.load_bus])
    if len(np.unique(groups)) != N_GROUPS or tuple(np.unique(phase)) != PHASES:
        raise AssertionError("feeder x phase schema changed")
    from .doe_day import FLEX_EXPORT_KW, FLEX_IMPORT_KW
    n_c = len(net.customer_bus)
    customers = envelopes.Customers(bus=net.customer_bus, export_cap=np.full(n_c, FLEX_EXPORT_KW / 1e3),
                                    import_cap=np.full(n_c, FLEX_IMPORT_KW / 1e3))
    out = {"P": [], "C": [], "G": [], "GA_export": [], "GA_import": [], "steps": [], "timing": [], "failed": []}
    t_prep = time.perf_counter()
    for step, p, q, _, _ in replay(day, hh):
        prep = time.perf_counter() - t_prep
        if steps is not None and step not in steps:
            t_prep = time.perf_counter()
            continue
        timer = {"prep": prep}
        pd_mw, qd = -p * fd.base_mva, -q * fd.base_mva
        t = time.perf_counter()
        out["P"].append(stats(phase_features(pd_mw, qd, net.customer_bus, groups, phase)))
        timer["P"] = time.perf_counter() - t
        t = time.perf_counter()
        out["C"].append(np.concatenate([[np.nan], stats(_summary(fdr.sweep_pf(fd, p, q)))]))
        timer["C_sweep"] = time.perf_counter() - t
        t = time.perf_counter()
        pr0, emb0 = embed(fd, p, q)
        s0 = _summary(pr0)
        out["G"].append(np.concatenate([stats(s0), emb0.mean(0)]))
        timer["G_zero"] = time.perf_counter() - t
        failed = {}
        for d in DIRS:
            share0 = float(envelopes.is_feasible(fd, pr0, d).mean())
            tm = {}
            try:
                ga = ga_block(oracle, embed, fd, customers, p, q, d, s0, emb0, share0, tm)
            except Exception as e:                            # failed inference: whole GA vector NaN, flagged
                ga, failed[d] = np.full(588, np.nan), repr(e)
            out[f"GA_{d}"].append(ga)
            timer.update({f"{k}_{d}": v for k, v in tm.items()})
        out["steps"].append(int(step))
        out["timing"].append(timer)
        out["failed"].append(failed)
        t_prep = time.perf_counter()
    return out


def blocks_list() -> list[tuple[str, int]]:
    sp = json.loads(SPLIT.read_text())
    return [(d, hh) for d in sp["test"] for hh in sp["test_placements"]] + \
           [(d, hh) for d in sorted(sp["train"] + sp["val"] + sp["calib"]) for hh in sp["label_placements"]]


def save_block(day, hh, ex, out_dir: Path = FEAT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{day}_hh{hh}.npz"
    arr = {k: np.array(ex[k]) for k in ("P", "C", "G", "GA_export", "GA_import", "steps")}
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in arr.items() if k not in ("steps", "C")}
    tmp = out_dir / f".{day}_hh{hh}.partial.npz"                # atomic: a killed task leaves no valid-looking file
    np.savez(tmp, **arr, day=day, hh=hh, timing=json.dumps(ex["timing"]), failed=json.dumps(ex["failed"]),
             nonfinite=json.dumps(nonfinite), schema=schema_hash())
    import os
    os.replace(tmp, f)
    return f


# ----------------------------------------------------------------------------------------------- labels
def label_path(day: str, hh: int) -> Path:
    sp = json.loads(SPLIT.read_text())
    if day in sp["test"]:
        return (ANNUAL / f"{day}_annual_4w_{B}_sph1_seed101_uk_esqcr.json" if hh == 101
                else W1 / f"{day}_{B}_sph1_seed101_hh{hh}_uk_esqcr.json")
    return LABELS / f"{day}_{B}_sph1_seed101_hh{hh}_uk_esqcr.json"


def check_label_file(f: Path, day: str, hh: int, run_commit: str) -> tuple[list[str], float]:
    """Per-file label checks (amendment 3c): structure, configuration, finiteness, engine health, sentinel."""
    from .envelopes import acceptance_threshold
    from .limits import record as limits_record
    doc = json.loads(f.read_text())
    m, rows = doc["meta"], doc["rows"]
    prob = []
    lim = limits_record()
    if m.get("day") != day:
        prob.append("day")
    if m.get("scenario_seed") != SCENARIO_SEED or m.get("household_seed") != hh:
        prob.append("seeds")
    vl = m.get("voltage_limits", {})
    if vl.get("jurisdiction") != lim["jurisdiction"] or abs(vl.get("vmin_pu", 0) - lim["vmin_pu"]) > 1e-12 \
            or abs(vl.get("vmax_pu", 0) - lim["vmax_pu"]) > 1e-12:
        prob.append("voltage limits")
    if abs(m.get("acceptance_threshold", 0) - acceptance_threshold(200, 0.05, 0.05)) > 1e-12 or m.get("scenarios") != N_S:
        prob.append("threshold / scenarios")
    if m.get("code_commit") != run_commit:
        prob.append(f"commit {m.get('code_commit')}")
    eh = m.get("engine_health")
    if not eh or eh.get("reset_failures", 1) != 0:
        prob.append(f"engine health {eh}")
    keys = [(r["step"], r["direction"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        prob.append("duplicate keys")
    for meth in ("ac", B):
        if sum(r["method"] == meth for r in rows) != 48:
            prob.append(f"{meth}: not 48 rows")
    for r in rows:
        vals = [r["limit_kw"], r["prob_ok_oracle"], r["ac_calls"], r["runtime_s"]]
        if not np.all(np.isfinite(vals)):
            prob.append("non-finite value")
            break
        if not 0.0 <= r["prob_ok_oracle"] <= 1.0 or not -1e-9 <= r["limit_kw"] <= CAPS_KW[r["direction"]] + 1e-9:
            prob.append("share / limit out of range")
            break
    shares = [r["prob_ok_oracle"] for r in rows if r["method"] == B]
    mean_share = float(np.mean(shares)) if shares else float("nan")
    if not mean_share >= SENTINEL_MEAN_SHARE:
        prob.append(f"catastrophic engine-failure sentinel: file-mean exact design share {mean_share:.3f} < 0.2")
    return prob, mean_share


def integrity(out_json: Path, label_dir: Path | None = None, blocks: list | None = None) -> dict:
    """Corrected label integrity (amendment 3c). Expected commit from the committed recovery run manifest."""
    import os
    run = json.loads(LABEL_RUN_MANIFEST.read_text())
    rec = json.loads(RECOVERY_MANIFEST.read_text())
    label_dir = label_dir or LABELS
    sp = json.loads(SPLIT.read_text())
    if blocks is None:
        blocks = [(d, hh) for d in sorted(sp["train"] + sp["val"] + sp["calib"]) for hh in sp["label_placements"]]
    rep = {"report": "corrected label integrity (amendment 3c); supersedes the quarantined obsolete report",
           "expected_blocks": len(blocks), "run_commit": run["run_commit"], "label_code_commit": rec["label_code_commit"],
           "source_sha256": rec["source_sha256"], "missing": [], "problems": [],
           "partial_files": [str(x) for x in label_dir.glob(".*partial*")], "files": {}, "file_mean_share": {}}
    for d, hh in blocks:
        f = label_dir / f"{d}_{B}_sph1_seed101_hh{hh}_uk_esqcr.json"
        if not f.exists():
            rep["missing"].append(f.name)
            continue
        try:
            prob, ms = check_label_file(f, d, hh, run["run_commit"])
        except Exception as e:
            prob, ms = [f"unreadable: {e!r}"], float("nan")
        if prob:
            rep["problems"].append({"block": f"{d}_hh{hh}", "problems": prob})
        rep["files"][str(f)] = sha256(f)
        rep["file_mean_share"][f"{d}_hh{hh}"] = ms
    v = np.array(list(rep["file_mean_share"].values()), float)
    rep["integrity_diagnostics_file_mean_exact_share"] = (
        {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v)),
         "files_below_0.2": int((v < SENTINEL_MEAN_SHARE).sum())} if len(v) else None)
    rep["complete"] = (not rep["missing"] and not rep["problems"] and not rep["partial_files"]
                       and len(rep["files"]) == rep["expected_blocks"])
    tmp = out_json.with_name("." + out_json.name + ".partial")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out_json)
    return {k: v for k, v in rep.items() if k not in ("files", "file_mean_share")}


def feature_integrity(out_json: Path, manifest: Path = Path("results/run_manifest_envpred3_features.json")) -> dict:
    """Amendment 3a / step 4: structural integrity of features_v3 before fitting (features only)."""
    sp = json.loads(SPLIT.read_text())
    expected = blocks_list()
    files = sorted(FEAT.glob("*.npz"))
    rep = {"expected_blocks": len(expected), "unique_expected": len(set(expected)), "files_found": len(files),
           "schema_sha256": schema_hash(), "manifest_sha256": json.loads(manifest.read_text())["sha256"],
           "partial_files": [str(x) for x in FEAT.glob(".*partial*")], "missing": [], "unexpected": [],
           "problems": [], "failed_items": 0, "nonfinite_by_block": {}, "L_G_out_of_bounds": 0, "files": {}}
    exp_names = {f"{d}_hh{h}.npz" for d, h in expected}
    rep["unexpected"] = sorted({x.name for x in files} - exp_names)
    caps = {"export": CAPS_KW["export"], "import": CAPS_KW["import"]}
    for day, hh in expected:
        f = FEAT / f"{day}_hh{hh}.npz"
        if not f.exists():
            rep["missing"].append(f.name)
            continue
        prob = []
        try:
            z = np.load(f)
            if str(z["day"]) != day or int(z["hh"]) != hh:
                prob.append("identity")
            if str(z["schema"]) != schema_hash():
                prob.append("schema")
            t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
            if list(z["steps"]) != [t0 + 4 * k for k in range(24)]:
                prob.append("steps")
            for k, n in (("P", 224), ("C", 29), ("G", 220), ("GA_export", 588), ("GA_import", 588)):
                if z[k].shape != (24, n):
                    prob.append(f"{k} shape {z[k].shape}")
                nf = int((~np.isfinite(z[k])).sum()) - (24 if k == "C" else 0)      # C:L_ac is filled at fit time
                rep["nonfinite_by_block"][k] = rep["nonfinite_by_block"].get(k, 0) + nf
            for dname in ("export", "import"):
                L = z[f"GA_{dname}"][:, 0]
                rep["L_G_out_of_bounds"] += int(((L < 0) | (L > caps[dname] + 1e-9)).sum())
            rep["failed_items"] += sum(len(x) for x in json.loads(str(z["failed"])))
            hhs = sp["test_placements"] if day in sp["test"] else sp["label_placements"]
            if hh not in hhs:
                prob.append("placement not in split")
        except Exception as e:
            prob.append(f"unreadable: {e!r}")
        if prob:
            rep["problems"].append({"block": f.name, "problems": prob})
        rep["files"][f.name] = sha256(f)
    rep["complete"] = (not rep["missing"] and not rep["unexpected"] and not rep["problems"] and not rep["partial_files"]
                       and rep["files_found"] == rep["expected_blocks"] == rep["unique_expected"] == 840
                       and rep["L_G_out_of_bounds"] == 0 and rep["failed_items"] == 0)
    out_json.write_text(json.dumps(rep, indent=1))
    return {k: v for k, v in rep.items() if k != "files"}


def load_table():
    """One row per (date, placement, step, direction): labels from the B file, features from features_v3."""
    sp = json.loads(SPLIT.read_text())
    of = {d: s for s in ("train", "val", "calib", "test") for d in sp[s]}
    rows, X = [], {"P": [], "C": [], "G": [], "GA": []}
    for day, hh in blocks_list():
        z = np.load(FEAT / f"{day}_hh{hh}.npz")
        if str(z["schema"]) != schema_hash():
            raise SystemExit(f"{day}_hh{hh}: feature schema mismatch")
        doc = json.loads(label_path(day, hh).read_text())
        r_ = {(r["step"], r["direction"], r["method"]): r for r in doc["rows"]}
        for i, step in enumerate(z["steps"]):
            for d in DIRS:
                b, ac = r_[(int(step), d, B)], r_[(int(step), d, "ac")]
                c = z["C"][i].copy()
                c[0] = ac["limit_kw"]
                X["P"].append(z["P"][i]), X["C"].append(c), X["G"].append(z["G"][i]), X["GA"].append(z[f"GA_{d}"][i])
                rows.append({"day": day, "hh": hh, "split": of[day], "step": int(step), "direction": d,
                             "L_B": b["limit_kw"], "b_calls": b["ac_calls"], "b_runtime_s": b["runtime_s"],
                             "b_p4": b.get("prob_ok_unbalanced", np.nan), "ac_runtime_s": ac["runtime_s"],
                             "file": str(label_path(day, hh))})
    return pd.DataFrame(rows), {k: np.array(v) for k, v in X.items()}


def arm_X(X, arm, mask):
    return np.hstack([X[b][mask] for b in ARMS[arm]])


def shrink_delta(L_hat, L_B, q):
    r = np.sort(L_hat - L_B)
    k = int(np.ceil((len(r) + 1) * q)) - 1
    return max(float(r[min(k, len(r) - 1)]), 0.0)


def fit(out_json: Path):
    from .envpred import proxy_retention
    from .option2 import fit_heads, predict
    integ = json.loads(LABEL_INTEGRITY.read_text())
    if not integ["complete"]:
        raise SystemExit("label integrity incomplete: fix before fitting")
    df, X = load_table()
    info = {"schema_sha256": schema_hash(), "n_features": {a: len(feature_names(a)) for a in ARMS},
            "rows": df.split.value_counts().to_dict(), "dates": df.groupby("split").day.nunique().to_dict(),
            "delta_kw": {}, "iterations": {}, "inference_s_per_interval": {}, "val_proxy_retention": {}}
    for a in ARMS:
        if arm_X(X, a, np.zeros(len(df), bool)).shape[1] != len(feature_names(a)):
            raise AssertionError(a)
    out = df[["day", "hh", "split", "step", "direction", "L_B", "file"]].copy()
    for arm in ARMS:
        for d in DIRS:
            m = {s: ((df.split == s) & (df.direction == d)).to_numpy() for s in ("train", "val", "calib")}
            heads = fit_heads(arm_X(X, arm, m["train"]), df.L_B[m["train"]].to_numpy()[:, None],
                              arm_X(X, arm, m["val"]), df.L_B[m["val"]].to_numpy()[:, None])
            dm = (df.direction == d).to_numpy()
            t = time.perf_counter()
            L_hat = np.clip(predict(heads, arm_X(X, arm, dm))[:, 0], 0, CAPS_KW[d])
            info["inference_s_per_interval"][f"{arm}_{d}"] = (time.perf_counter() - t) / dm.sum()
            info["iterations"][f"{arm}_{d}"] = int(heads[0].n_iter_)
            out.loc[dm, f"Lhat_{arm}"] = L_hat
            cm = m["calib"][dm]
            for q in SHRINK_Q:
                delta = shrink_delta(L_hat[cm], df.L_B[dm].to_numpy()[cm], q)
                info["delta_kw"][f"{arm}_{d}_q{int(q * 100)}"] = delta
                out.loc[dm, f"Ls_{arm}_q{int(q * 100)}"] = np.maximum(L_hat - delta, 0.0)
        v = (df.split == "val").to_numpy()
        info["val_proxy_retention"][arm] = proxy_retention(out[f"Ls_{arm}_q95"][v].to_numpy(), df.L_B[v].to_numpy(),
                                                           df.direction[v].to_numpy())
    vr = info["val_proxy_retention"]
    info["selected"] = {fam: max(arms, key=lambda a: (vr[a], -len(feature_names(a)))) for fam, arms in FAMILY.items()}
    import os
    tmp = OUT / ("." + PRED_NAME + ".partial")
    out.to_parquet(tmp)
    os.replace(tmp, OUT / PRED_NAME)
    info["predictions_sha256"] = sha256(OUT / PRED_NAME)
    tmp = out_json.with_name("." + out_json.name + ".partial")
    tmp.write_text(json.dumps(info, indent=1))
    os.replace(tmp, out_json)
    return info


CODE_FILES = ("src/gencodoe/envpred3.py", "src/gencodoe/envpred.py", "src/gencodoe/option2.py")


def fit_job(expect_code: dict, commit: str) -> dict:
    """The single v3 fit (Hydra Zen4 CPU). Stops on any missing input, hash or schema mismatch."""
    import os
    import platform
    import sklearn
    t0 = time.perf_counter()
    for f, h in expect_code.items():
        if sha256(Path(f)) != h:
            raise SystemExit(f"code hash mismatch: {f}")
    fi = json.loads(Path("results/envpred3_feature_integrity.json").read_text())
    li = json.loads(LABEL_INTEGRITY.read_text())
    if not (fi["complete"] and li["complete"]):
        raise SystemExit("an integrity report is not complete")
    if fi["schema_sha256"] != schema_hash():
        raise SystemExit("feature schema mismatch")
    for name, h in fi["files"].items():
        if sha256(FEAT / name) != h:
            raise SystemExit(f"feature file changed since integrity: {name}")
    for f, h in li["files"].items():
        if sha256(Path(f)) != h:
            raise SystemExit(f"label file changed since integrity: {f}")
    sp = json.loads(SPLIT.read_text())
    test_hashes = {}
    for d in sp["test"]:
        for hh in sp["test_placements"]:
            f = label_path(d, hh)
            if not f.exists():
                raise SystemExit(f"missing test label file {f}")
            test_hashes[str(f)] = sha256(f)
    info = fit(OUT / FIT_NAME)
    env = {"code_commit": commit, "code_sha256": expect_code, "python": platform.python_version(),
           "sklearn": sklearn.__version__, "numpy": np.__version__, "platform": platform.platform(),
           "cpu": platform.processor() or platform.machine(),
           "threads_available": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "elapsed_s": time.perf_counter() - t0,
           "inputs": {"feature_integrity_sha256": sha256(Path("results/envpred3_feature_integrity.json")),
                      "label_integrity_sha256": sha256(LABEL_INTEGRITY),
                      "split_sha256": sha256(SPLIT), "test_label_files_sha256": test_hashes},
           "outputs": {PRED_NAME: info["predictions_sha256"],
                       FIT_NAME: sha256(OUT / FIT_NAME)}}
    tmp = OUT / ("." + RUN_NAME + ".partial")
    tmp.write_text(json.dumps(env, indent=1))
    os.replace(tmp, OUT / RUN_NAME)
    return env


# ----------------------------------------------------------------------------------------------- gate
class CachedExact:
    """Shares one interval's exact evaluations across arms and shrink points. share() and results() return the
    counterfactual call count (as if the caller were alone); `physical` counts the solves actually made."""

    def __init__(self, oracle):
        self.o, self.physical, self.seconds = oracle, 0, 0.0
        self.reset()

    def reset(self):
        self._share, self._res = {}, {}

    def share(self, p, q, level_kw, direction, canonical=False):
        k = (direction, round(float(level_kw), 9))
        if k not in self._share:
            t = time.perf_counter()
            self._share[k], n = self.o.share(p, q, level_kw, direction)
            self.physical += n
            self.seconds += time.perf_counter() - t
        return self._share[k], len(p)

    def results(self, p, q, level_kw, direction, canonical=False):
        k = (direction, round(float(level_kw), 9))
        if k not in self._res:
            t = time.perf_counter()
            self._res[k] = self.o.results(p, q, level_kw, direction)
            self.physical += len(p)
            self.seconds += time.perf_counter() - t
        return self._res[k]


def gate_file(day: str, hh: int) -> Path:
    from .cplus import Counter, FourWireOracle, repair, verify_repaired
    from .enwl import load_enwl1
    from .envelopes import acceptance_threshold
    from .unbalanced import UnbalancedENWL1
    out = GATE / f"{day}_hh{hh}.json"
    if out.exists():
        return out
    pr = pd.read_parquet(OUT / PRED_NAME)
    pr = pr[(pr.day == day) & (pr.hh == hh)]
    net = load_enwl1()
    fd = net.to_feeder()
    ex = CachedExact(FourWireOracle(net, UnbalancedENWL1()))
    thr = acceptance_threshold(N_S, 0.05, 0.05)
    res = []
    for step, p, q, pv, qv in replay(day, hh):
        for d in DIRS:
            ex.reset()
            row = pr[(pr.step == step) & (pr.direction == d)].iloc[0]
            tol = 1e-3 * CAPS_KW[d]
            for arm in ARMS:
                for qq in SHRINK_Q:
                    L_s = float(row[f"Ls_{arm}_q{int(qq * 100)}"])
                    t0, ph0 = time.perf_counter(), ex.physical
                    r = {"step": int(step), "direction": d, "arm": arm, "q": qq, "L_B": float(row.L_B), "L_s": L_s,
                         "gate_calls": 0, "fallback_calls": 0, "gate_pass": None, "fallback": False,
                         "predicted_zero": L_s <= tol, "zero_sentinel": False}
                    issued = 0.0
                    if not r["predicted_zero"]:
                        share, n = ex.share(p, q, L_s, d)
                        r.update(gate_calls=n, gate_share=share, gate_pass=bool(share >= thr))
                        if share >= thr:
                            issued = L_s
                        else:
                            issued, _, calls, zinf = repair(ex, p, q, L_s, d, thr)
                            r.update(fallback=True, fallback_calls=calls, zero_sentinel=bool(zinf))
                    r["issued"] = float(issued)
                    r["search_physical_calls"] = ex.physical - ph0
                    r["search_wall_s"] = time.perf_counter() - t0
                    if issued > tol:
                        v = verify_repaired(ex, fd, net.customer_bus, pv, qv, N_U, issued, d, Counter())
                        r.update(fresh_p4=v["prob_ok_unbalanced"], fresh_ac=v["prob_ok_ac_fresh"])
                    res.append(r)
    import os

    from .limits import record as limits_record
    GATE.mkdir(parents=True, exist_ok=True)
    tmp = GATE / f".{day}_hh{hh}.partial.json"                    # atomic
    tmp.write_text(json.dumps({"day": day, "hh": hh, "scenario_seed": SCENARIO_SEED, "threshold": thr,
                               "caps_kw": CAPS_KW, "voltage_limits": limits_record(),
                               "code_commit": os.environ.get("GENCODOE_COMMIT"),
                               "predictions_sha256": sha256(OUT / PRED_NAME),
                               "physical_calls": ex.physical, "physical_seconds": ex.seconds, "rows": res}, indent=1))
    os.replace(tmp, out)
    return out


def _test_blocks():
    sp = json.loads(SPLIT.read_text())
    return [(d, h) for d in sp["test"] for h in sp["test_placements"]]


def gate_integrity(out_json: Path, commit: str) -> dict:
    """Structural integrity of the 180 gate files (no performance statistic is computed)."""
    import os
    from .envelopes import acceptance_threshold
    from .limits import record as limits_record
    pred_sha = sha256(OUT / PRED_NAME)
    rep = {"expected_files": 180, "expected_rows_per_file": 768, "expected_rows": 138240, "expected_q95_rows": 34560,
           "missing": [], "problems": [], "partial_files": [str(x) for x in GATE.glob(".*partial*")],
           "unexpected": sorted({x.name for x in GATE.glob("*.json")} - {f"{d}_hh{h}.json" for d, h in _test_blocks()}),
           "rows": 0, "q95_rows": 0, "files": {}}
    thr = acceptance_threshold(N_S, 0.05, 0.05)
    lim = limits_record()
    for day, hh in _test_blocks():
        f = GATE / f"{day}_hh{hh}.json"
        if not f.exists():
            rep["missing"].append(f.name)
            continue
        prob = []
        try:
            doc = json.loads(f.read_text())
            r = pd.DataFrame(doc["rows"])
            if (doc["day"], doc["hh"]) != (day, hh) or doc["scenario_seed"] != SCENARIO_SEED:
                prob.append("identity/seed")
            if doc["threshold"] != thr or doc["caps_kw"] != CAPS_KW or doc["voltage_limits"] != lim:
                prob.append("threshold/caps/limits")
            if doc["code_commit"] != commit or doc["predictions_sha256"] != pred_sha:
                prob.append("commit/predictions hash")
            if len(r) != 768 or r.duplicated(["step", "direction", "arm", "q"]).any():
                prob.append(f"rows {len(r)}")
            if sorted(r.groupby(["arm", "q"]).size().unique()) != [48] or r.step.nunique() != 24:
                prob.append("arm x q x interval grid")
            cap = r.direction.map(CAPS_KW)
            if not np.isfinite(r.issued).all() or (r.issued < 0).any() or (r.issued > cap + 1e-9).any():
                prob.append("issued not finite / out of cap")
            nz = r.issued > 1e-3 * cap
            if r.loc[nz, ["fresh_p4", "fresh_ac"]].isna().any().any() or not np.isfinite(r.loc[nz, ["fresh_p4", "fresh_ac"]]).all().all():
                prob.append("fresh feasibility missing / non-finite")
            if (r.gate_calls % N_S).any() or (r.fallback_calls % N_S).any() or (r.search_physical_calls > r.gate_calls + r.fallback_calls).any():
                prob.append("call accounting")
            if doc["physical_calls"] > (r.gate_calls + r.fallback_calls).sum() + N_U * nz.sum():
                prob.append("physical > counterfactual + fresh")
            rep["rows"] += len(r)
            rep["q95_rows"] += int((r.q == PRIMARY_Q).sum())
        except Exception as e:
            prob.append(f"unreadable: {e!r}")
        if prob:
            rep["problems"].append({"file": f.name, "problems": prob})
        rep["files"][f.name] = sha256(f)
    rep["predictions_sha256"] = pred_sha
    rep["code_commit"] = commit
    rep["complete"] = (not rep["missing"] and not rep["problems"] and not rep["partial_files"] and not rep["unexpected"]
                       and len(rep["files"]) == 180 and rep["rows"] == 138240 and rep["q95_rows"] == 34560)
    tmp = out_json.with_name("." + out_json.name + ".partial")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out_json)
    return {k: v for k, v in rep.items() if k != "files"}


# ----------------------------------------------------------------------------------------------- report
def _boot(per_day: pd.DataFrame, stat, n_boot=2000, seed=0) -> np.ndarray:
    """Season-stratified date-block bootstrap; per_day indexed by date, stat(frame) -> float."""
    from .report_annual import season
    rng = np.random.default_rng(seed)
    days = np.array(sorted(per_day.index.unique()))
    by = {}
    for d in days:
        by.setdefault(season(d), []).append(d)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = [x for ds in by.values() for x in rng.choice(ds, len(ds))]
        out[b] = stat(per_day.loc[pick])
    return out


def _arm_metrics(gs: pd.DataFrame) -> dict:
    nz = gs[gs.issued > gs.tol]
    return {"intervals": int(len(gs)), "retention": float(gs.kept.sum() / gs.L_B.sum()),
            "excess_over_B": float(gs.excess.sum() / gs.L_B.sum()),
            "intervals_issued_above_B": int((gs.issued > gs.L_B + gs.tol).sum()),
            "exact_calls_alone": int(gs.calls.sum()), "gate_calls_alone": int(gs.gate_calls.sum()),
            "fallback_calls_alone": int(gs.fallback_calls.sum()), "exact_calls_B": int(gs.b_calls.sum()),
            "exact_call_ratio_vs_B": float(gs.calls.sum() / gs.b_calls.sum()),
            "gate_pass_rate": float(gs.gate_pass.dropna().astype(bool).mean()) if gs.gate_pass.notna().any() else None,
            "fallbacks": int(gs.fallback.sum()), "predicted_zero": int(gs.predicted_zero.sum()),
            "predicted_zero_where_B_positive": int((gs.predicted_zero & (gs.L_B > gs.tol)).sum()),
            "zero_sentinels_from_fallback": int(gs.zero_sentinel.sum()), "B_zero": int((gs.L_B <= gs.tol).sum()),
            "fresh_p4_mean": float(nz.fresh_p4.mean()) if len(nz) else None,
            "fresh_p4_share_ge_0.95": float((nz.fresh_p4 >= 0.95).mean()) if len(nz) else None,
            "fresh_ac_mean": float(nz.fresh_ac.mean()) if len(nz) else None}


def _feas_ok(a: dict, b: dict) -> bool:
    return bool(a["fresh_p4_mean"] >= b["fresh_p4_mean"] - 0.005
                and a["fresh_p4_share_ge_0.95"] >= b["fresh_p4_share_ge_0.95"] - 0.01)


def _diff_boot(g: pd.DataFrame, A: str, Bm: str) -> list[float]:
    def stat(x):
        return (x[x.arm == A].kept.sum() - x[x.arm == Bm].kept.sum()) / x[x.arm == A].L_B.sum()
    bt = _boot(g.set_index("day", drop=False), stat)
    return [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]


def _paired_feasibility(g: pd.DataFrame, A: str, Bm: str) -> dict:
    a = g[g.arm == A].set_index(["day", "hh", "step", "direction"])
    b = g[g.arm == Bm].set_index(["day", "hh", "step", "direction"])
    a, b = a.align(b, join="inner", axis=0)
    ia, ib = a.issued > a.tol, b.issued > b.tol
    out = {}
    for name, m in (("both_issue", ia & ib), (f"only_{A}", ia & ~ib), (f"only_{Bm}", ib & ~ia)):
        out[name] = {"intervals": int(m.sum()),
                     f"{A}_fresh_p4_mean": float(a.fresh_p4[m].mean()) if m.any() and ia[m].all() else None,
                     f"{Bm}_fresh_p4_mean": float(b.fresh_p4[m].mean()) if m.any() and ib[m].all() else None}
    return out


def _load_gate(only_q: float | None):
    rows = []
    for day, hh in _test_blocks():
        doc = json.loads((GATE / f"{day}_hh{hh}.json").read_text())
        rows += [{**r, "day": day, "hh": hh} for r in doc["rows"] if only_q is None or r["q"] == only_q]
    g = pd.DataFrame(rows)
    df, _ = load_table()
    t = df[df.split == "test"][["day", "hh", "step", "direction", "b_calls", "b_p4", "b_runtime_s", "ac_runtime_s"]]
    g = g.merge(t, on=["day", "hh", "step", "direction"], how="left", validate="many_to_one")
    g["tol"] = g.direction.map(CAPS_KW) * 1e-3
    g["kept"] = np.minimum(g.issued, g.L_B)
    g["excess"] = np.maximum(g.issued - g.L_B, 0)
    g["calls"] = g.gate_calls + g.fallback_calls
    return g, df, t


def report_primary(out_json: Path, commit: str) -> dict:
    """Frozen primary result: q = 0.95 rows only; no frontier statistic. Fails unless gate integrity passed."""
    import os
    gi = json.loads(Path("results/envpred3_gate_integrity.json").read_text())
    if not gi["complete"]:
        raise SystemExit("gate integrity incomplete")
    for name, h in gi["files"].items():
        if sha256(GATE / name) != h:
            raise SystemExit(f"gate file changed since integrity: {name}")
    info = json.loads((OUT / FIT_NAME).read_text())
    g, df, t = _load_gate(PRIMARY_Q)
    if len(g) != 34560 or g.b_calls.isna().any():
        raise SystemExit(f"{len(g)} q95 rows (expected 34,560) or unmatched B rows")
    G, N = info["selected"]["genco"], info["selected"]["non_genco"]
    out = {"artifact": "frozen primary result (q = 0.95 only)", "protocol": "docs/envpred_protocol.md (amendments 1-3b)",
           "status": "frozen prospective analysis within an already explored benchmark",
           "framing": "GENCO is evaluated as a learned proposal engine for expensive four-wire envelope optimisation; "
                      "exact OpenDSS remains the final decision authority",
           "computation_note": "All shrink points were computed in one cached evaluation, but the frozen q95 primary "
                               "result was reported and committed before the secondary frontier was examined.",
           "code_commit": commit, "expected_q95_rows": 34560, "q95_rows": int(len(g)),
           "inputs_sha256": {"gate_integrity": sha256(Path("results/envpred3_gate_integrity.json")),
                             FIT_NAME: sha256(OUT / FIT_NAME),
                             PRED_NAME: sha256(OUT / PRED_NAME),
                             "gate_files": gi["files"]},
           "selected_on_validation": {"genco": G, "non_genco": N, "val_proxy_retention": info["val_proxy_retention"]},
           "strata": {}}
    for sname, hhs in STRATA.items():
        gs = g[g.hh.isin(hhs)]
        m = {a: {"combined": _arm_metrics(x), **{d: _arm_metrics(y) for d, y in x.groupby("direction")}}
             for a, x in gs.groupby("arm")}
        mc = {a: m[a]["combined"] for a in m}
        dec = {"retention_diff": mc[G]["retention"] - mc[N]["retention"],
               "c1_retention_ge_3pp": mc[G]["retention"] - mc[N]["retention"] >= 0.03,
               "c2_exact_calls_not_more": mc[G]["exact_calls_alone"] <= mc[N]["exact_calls_alone"],
               "bootstrap_ci95": _diff_boot(gs, G, N), "c4_feasibility_not_worse": _feas_ok(mc[G], mc[N])}
        dec["c3_bootstrap_lower_gt_0"] = dec["bootstrap_ci95"][0] > 0
        dec["genco_positive"] = bool(dec["c1_retention_ge_3pp"] and dec["c2_exact_calls_not_more"]
                                     and dec["c3_bootstrap_lower_gt_0"] and dec["c4_feasibility_not_worse"])
        pairs = {}
        for A, Bm, calls in (("PG", "P", False), ("PCG", "PC", True)):
            ci = _diff_boot(gs, A, Bm)
            diff = mc[A]["retention"] - mc[Bm]["retention"]
            material = bool(diff >= 0.03 and ci[0] > 0 and _feas_ok(mc[A], mc[Bm])
                            and (not calls or mc[A]["exact_calls_alone"] <= mc[Bm]["exact_calls_alone"]))
            pairs[f"{A}_vs_{Bm}"] = {"retention_diff": diff, "bootstrap_ci95": ci, "material": material,
                                     "detectable_not_material": bool(ci[0] > 0 and diff < 0.03),
                                     "paired_feasibility": _paired_feasibility(gs, A, Bm)}
        out["strata"][sname] = {"dates": int(gs.day.nunique()), "placements": list(hhs), "arms": m, "decision": dec,
                                "pairs": pairs, "selected_paired_feasibility": _paired_feasibility(gs, G, N),
                                "physical_calls_while_processing_q95_rows": int(gs.search_physical_calls.sum())}
    dec = out["strata"]["combined"]["decision"]
    pr = out["strata"]["combined"]["pairs"]
    out["primary_decision"] = dec
    out["tier"] = ("strong GENCO-positive" if dec["genco_positive"] else
                   "additive value beyond classical balanced physics" if pr["PCG_vs_PC"]["material"] else
                   "additive representation value beyond phase-only information (does not beat the classical baseline)"
                   if pr["PG_vs_P"]["material"] else
                   "detectable but not practically material" if (pr["PG_vs_P"]["detectable_not_material"]
                                                                 or pr["PCG_vs_PC"]["detectable_not_material"])
                   else "no measurable GENCO value")
    # cost by hardware (features and stored L_ac are shared by all shrink points)
    feat = {"prep_cpu": 0.0, "P_cpu": 0.0, "C_sweep_cpu": 0.0, "G_zero_a100": 0.0, "GA_a100": 0.0}
    for day, hh in _test_blocks():
        z = np.load(FEAT / f"{day}_hh{hh}.npz")
        for tm in json.loads(str(z["timing"])):
            for k, v in tm.items():
                key = ("GA_a100" if k.split("_")[0] in ("LG", "curve") else "C_sweep_cpu" if k == "C_sweep" else
                       "G_zero_a100" if k == "G_zero" else "P_cpu" if k == "P" else "prep_cpu")
                feat[key] += v
    phys = sum(json.loads((GATE / f"{d}_hh{h}.json").read_text())["physical_calls"] for d, h in _test_blocks())
    phys_s = sum(json.loads((GATE / f"{d}_hh{h}.json").read_text())["physical_seconds"] for d, h in _test_blocks())
    sps = phys_s / max(phys, 1)
    out["cost"] = {"feature_seconds_test": feat, "L_ac_seconds_test_stored_cpu": float(t.ac_runtime_s.sum()),
                   "head_inference_s_per_interval_cpu": info["inference_s_per_interval"],
                   "measured_s_per_exact_solve_cpu": sps,
                   "physical_exact_calls_all_arms_all_shrink_points": int(phys),
                   "physical_exact_seconds_all_arms_all_shrink_points": phys_s,
                   "per_arm": {a: {"exact_calls_alone": int(x.calls.sum()), "gate_calls_alone": int(x.gate_calls.sum()),
                                   "fallback_calls_alone": int(x.fallback_calls.sum()),
                                   "estimated_exact_seconds_alone_cpu": float(x.calls.sum() * sps),
                                   "observed_search_wall_s_with_cache": float(x.search_wall_s.sum())}
                               for a, x in g.groupby("arm")},
                   "B_exact_calls": int(t.b_calls.sum()), "B_search_seconds_cpu": float(t.b_runtime_s.sum()),
                   "notes": ["features and L_ac are shared by every shrink point and arm that uses them",
                             "physical calls were shared across arms and shrink points; no superiority claim rests on caching",
                             "exact-call savings are not automatically wall-clock savings; GENCO proposal construction "
                             "(~3 s, A100) and the classical proposal (~3 s, one CPU core) cost about the same"]}
    out["dimensionality"] = {"train_rows_per_direction": int(((df.split == "train") & (df.direction == "export")).sum()),
                             "train_dates": int(df[df.split == "train"].day.nunique())}
    tmp = out_json.with_name("." + out_json.name + ".partial")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, out_json)
    return out


def report(out_json: Path):
    info = json.loads((OUT / FIT_NAME).read_text())
    rows = []
    for day, hh in [(d, h) for d, h in blocks_list() if d in json.loads(SPLIT.read_text())["test"]]:
        doc = json.loads((GATE / f"{day}_hh{hh}.json").read_text())
        rows += [{**r, "day": day, "hh": hh} for r in doc["rows"]]
    g = pd.DataFrame(rows)
    df, _ = load_table()
    t = df[df.split == "test"][["day", "hh", "step", "direction", "b_calls", "b_p4", "b_runtime_s", "ac_runtime_s"]]
    g = g.merge(t, on=["day", "hh", "step", "direction"], how="left")
    if len(g) != 8640 * len(ARMS) * len(SHRINK_Q):
        raise SystemExit(f"{len(g)} gate rows: incomplete")
    g["tol"] = g.direction.map(CAPS_KW) * 1e-3
    g["kept"] = np.minimum(g.issued, g.L_B)
    g["excess"] = np.maximum(g.issued - g.L_B, 0)
    g["calls"] = g.gate_calls + g.fallback_calls
    G, N = info["selected"]["genco"], info["selected"]["non_genco"]
    out = {"protocol": "docs/envpred_protocol.md (amendments 1-3a)",
           "status": "frozen prospective analysis within an already explored benchmark",
           "framing": "GENCO is evaluated as a learned proposal engine for expensive four-wire envelope optimisation; "
                      "exact OpenDSS remains the final decision authority",
           "fit": info, "selected": {"genco": G, "non_genco": N}, "strata": {}}
    for sname, hhs in STRATA.items():
        gs = g[g.hh.isin(hhs)]
        st = {"dates": int(gs.day.nunique()), "placements": list(hhs), "points": {}}
        for (arm, qq), ga in gs.groupby(["arm", "q"]):
            st["points"][f"{arm}_q{int(qq * 100)}"] = {"combined": _arm_metrics(ga),
                                                        **{d: _arm_metrics(x) for d, x in ga.groupby("direction")}}
        p95 = gs[gs.q == PRIMARY_Q]
        m = {a: st["points"][f"{a}_q95"]["combined"] for a in ARMS}
        dec = {"retention_diff": m[G]["retention"] - m[N]["retention"],
               "c1_retention_ge_3pp": m[G]["retention"] - m[N]["retention"] >= 0.03,
               "c2_exact_calls_not_more": m[G]["exact_calls_alone"] <= m[N]["exact_calls_alone"],
               "bootstrap_ci95": _diff_boot(p95, G, N), "c4_feasibility_not_worse": _feas_ok(m[G], m[N])}
        dec["c3_bootstrap_lower_gt_0"] = dec["bootstrap_ci95"][0] > 0
        dec["genco_positive"] = bool(dec["c1_retention_ge_3pp"] and dec["c2_exact_calls_not_more"]
                                     and dec["c3_bootstrap_lower_gt_0"] and dec["c4_feasibility_not_worse"])
        st["decision_95"] = dec
        pairs = {}
        for A, Bm, calls in (("PG", "P", False), ("PCG", "PC", True)):
            ci = _diff_boot(p95, A, Bm)
            diff = m[A]["retention"] - m[Bm]["retention"]
            ok = diff >= 0.03 and ci[0] > 0 and _feas_ok(m[A], m[Bm]) and (
                not calls or m[A]["exact_calls_alone"] <= m[Bm]["exact_calls_alone"])
            pairs[f"{A}_vs_{Bm}"] = {"retention_diff": diff, "bootstrap_ci95": ci, "material": bool(ok),
                                     "detectable_not_material": bool(ci[0] > 0 and diff < 0.03),
                                     "paired_feasibility": _paired_feasibility(p95, A, Bm)}
        st["pairs_95"] = pairs
        st["selected_paired_feasibility_95"] = _paired_feasibility(p95, G, N)
        if sname == "combined":
            tier = ("strong GENCO-positive" if dec["genco_positive"] else
                    "additive value beyond classical physics" if pairs["PCG_vs_PC"]["material"] else
                    "additive value beyond phase-only (does not beat the classical baseline)"
                    if pairs["PG_vs_P"]["material"] else "no measurable GENCO value")
            out["primary_decision"] = dec
            out["tier"] = tier
        out["strata"][sname] = st
    # frontier (secondary, exploratory) and matched budgets (interpolated, descriptive)
    comb = out["strata"]["combined"]["points"]
    front = {a: sorted((comb[f"{a}_q{int(qq * 100)}"]["combined"]["exact_calls_alone"],
                        comb[f"{a}_q{int(qq * 100)}"]["combined"]["retention"]) for qq in SHRINK_Q) for a in ARMS}
    budgets = {f"best_non_genco_{N}_q95": comb[f"{N}_q95"]["combined"]["exact_calls_alone"],
               f"best_genco_{G}_q95": comb[f"{G}_q95"]["combined"]["exact_calls_alone"]}
    mb = {}
    for bname, bud in budgets.items():
        mb[bname] = {}
        for a, pts in front.items():
            c, r = np.array(pts).T
            mb[bname][a] = (None if not (c.min() <= bud <= c.max()) else
                            {"retention_interpolated": float(np.interp(bud, c, r)), "interpolated": True})
    out["frontier"] = {"status": "secondary, exploratory; never overrides the primary 95% result",
                       "points_calls_retention": front, "matched_budget": mb}
    # cost
    feat = {"P": 0.0, "C_sweep": 0.0, "G_zero": 0.0, "GA": 0.0, "prep": 0.0}
    for day, hh in {(d, h) for d, h in zip(g.day, g.hh)}:
        z = np.load(FEAT / f"{day}_hh{hh}.npz")
        for tm in json.loads(str(z["timing"])):
            for k, v in tm.items():
                key = "GA" if k.split("_")[0] in ("LG", "curve") else ("C_sweep" if k == "C_sweep" else k)
                feat[key] = feat.get(key, 0.0) + v
    phys = sum(json.loads((GATE / f"{d}_hh{h}.json").read_text())["physical_calls"]
               for d, h in {(d, h) for d, h in zip(g.day, g.hh)})
    phys_s = sum(json.loads((GATE / f"{d}_hh{h}.json").read_text())["physical_seconds"]
                 for d, h in {(d, h) for d, h in zip(g.day, g.hh)})
    s_per_solve = phys_s / max(phys, 1)
    t95 = g[g.q == PRIMARY_Q]
    out["cost_test"] = {
        "feature_seconds_by_block": feat, "feature_device": "GENCO on A100 (FP32, TF32 off); P, C on CPU",
        "L_ac_seconds_stored": float(df[df.split == "test"].ac_runtime_s.sum()),
        "head_inference_s_per_interval": info["inference_s_per_interval"],
        "physical_exact_calls_all_arms_points": int(phys), "physical_exact_seconds": phys_s,
        "measured_s_per_exact_solve_cpu": s_per_solve,
        "per_arm_95": {a: {"exact_calls_alone": int(x.calls.sum()),
                           "estimated_exact_seconds_alone_cpu": float(x.calls.sum() * s_per_solve)}
                       for a, x in t95.groupby("arm")},
        "B_exact_calls": int(t[["b_calls"]].sum().iloc[0]), "B_search_seconds": float(t.b_runtime_s.sum()),
        "note": "exact-call savings are not automatically wall-clock savings; cross-arm caching is not credited"}
    out["dimensionality"] = {"train_rows_per_direction": int(((df.split == "train") & (df.direction == "export")).sum()),
                             "train_dates": int(df[df.split == "train"].day.nunique()),
                             "features": {a: len(feature_names(a)) for a in ARMS}}
    out_json.write_text(json.dumps(out, indent=1, default=float))
    return out


# ----------------------------------------------------------------------------------------------- smoke
def smoke(device: str, run: Path, out_json: Path):
    import torch

    from . import envelopes
    from . import feeder as fdr
    from .doe_day import FLEX_EXPORT_KW, FLEX_IMPORT_KW
    from .enwl import load_enwl1
    oracle, embed = load_genco(device, run)
    res = {"device": device, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
           "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32), "schema_sha256": schema_hash(), "items": []}
    net = load_enwl1()
    fd = net.to_feeder()
    n_c = len(net.customer_bus)
    cust = envelopes.Customers(net.customer_bus, np.full(n_c, FLEX_EXPORT_KW / 1e3), np.full(n_c, FLEX_IMPORT_KW / 1e3))
    cpu = load_genco("cpu", run)[1] if device != "cpu" else None
    for day, hh, hour, d in SMOKE:
        t0 = int((np.datetime64(day) - np.datetime64("2024-01-01")).astype(int)) * 96
        step = t0 + 4 * hour
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        runs = [extract_block(day, hh, oracle, embed, steps={step}) for _ in range(2)]
        ga = [np.array(r[f"GA_{d}"])[0] for r in runs]
        item = {"day": day, "hh": hh, "hour": hour, "direction": d,
                "n_features": {"P": len(runs[0]["P"][0]), "C": len(runs[0]["C"][0]), "G": len(runs[0]["G"][0]),
                               "GA": len(ga[0])},
                "nonfinite_GA": int((~np.isfinite(ga[0])).sum()), "failed": runs[0]["failed"][0],
                "determinism_max_abs_diff_GA": float(np.nanmax(np.abs(ga[0] - ga[1]))),
                "determinism_L_G_equal": bool(ga[0][0] == ga[1][0]),
                "timing_s": runs[1]["timing"][0],
                "gpu_peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else None}
        # L_ac cost on the same interval: balanced sweep search, CPU
        for st_, p, q, _, _ in replay(day, hh):
            if st_ == step:
                t = time.perf_counter()
                envelopes.scalar_envelope(envelopes.exact(fdr.sweep_pf, fd), fd.n_bus, cust, p, q, d, "uniform",
                                          0.05, 4, beta=0.05)
                item["L_ac_search_s_cpu"] = time.perf_counter() - t
                if cpu is not None:
                    a, _ = embed(fd, p, q)
                    b, _ = cpu(fd, p, q)
                    item["zero_state_vm_max_abs_diff_gpu_vs_cpu"] = float(np.abs(a.vm - b.vm).max())
                break
        res["items"].append(item)
        print(json.dumps(item), flush=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "features", "integrity", "feature-integrity", "fit", "fit-job", "gate",
                                    "gate-integrity", "report-primary", "report", "label-smoke-check"])
    ap.add_argument("--expect-code", type=Path, default=None, help="fit-job: JSON {file: sha256} pinning the code")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--file-index", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run = a.run or (RUN_LOCAL if RUN_LOCAL.exists() else RUN_HYDRA)
    if a.cmd == "smoke":
        smoke(a.device, run, a.out or Path("results/envpred3_smoke.json"))
    elif a.cmd == "features":
        oracle, embed = load_genco(a.device, run)
        todo = blocks_list()[a.chunk::a.n_chunks]
        for i, (day, hh) in enumerate(todo):
            if (FEAT / f"{day}_hh{hh}.npz").exists():
                continue
            t = time.perf_counter()
            f = save_block(day, hh, extract_block(day, hh, oracle, embed))
            print(f"[{i + 1}/{len(todo)}] {f} {time.perf_counter() - t:.0f}s", flush=True)
    elif a.cmd == "feature-integrity":
        print(json.dumps(feature_integrity(Path("results/envpred3_feature_integrity.json")), indent=1))
    elif a.cmd == "integrity":
        print(json.dumps(integrity(LABEL_INTEGRITY), indent=1))
    elif a.cmd == "label-smoke-check":
        sm = json.loads(RECOVERY_MANIFEST.read_text())["smoke"]
        print(json.dumps(integrity(Path("results/envpred_label_recovery_smoke.json"), Path(sm["output_dir"]),
                                   [tuple(x) for x in sm["blocks"]]), indent=1))
    elif a.cmd == "fit-job":
        env = fit_job(json.loads(a.expect_code.read_text()), a.commit)
        print(json.dumps({k: v for k, v in env.items() if k != "inputs"}, indent=1))
    elif a.cmd == "fit":
        print(json.dumps(fit(OUT / FIT_NAME), indent=1))
    elif a.cmd == "gate":
        sp = json.loads(SPLIT.read_text())
        test = [(d, h) for d in sp["test"] for h in sp["test_placements"]]
        print(gate_file(*test[a.file_index]))
    elif a.cmd == "gate-integrity":
        print(json.dumps(gate_integrity(Path("results/envpred3_gate_integrity.json"), a.commit), indent=1))
    elif a.cmd == "report-primary":
        out = report_primary(Path("results/envpred3_primary_q95.json"), a.commit)
        print(json.dumps({k: out[k] for k in ("selected_on_validation", "primary_decision", "tier")}, indent=1,
                         default=float))
    else:                                             # secondary, exploratory frontier (after the primary is committed)
        out = report(a.out or Path("results/envpred3_report.json"))
        print(json.dumps({k: out[k] for k in ("selected", "primary_decision", "tier")}, indent=1, default=float))


if __name__ == "__main__":
    main()
