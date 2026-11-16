"""Screen calibration and evaluation of the 25-network study (docs/fourwire_scale_protocol.md §5-§7,
Amendments 8 and 9).

    python -m gencodoe.fourwire_scale_eval networks
    python -m gencodoe.fourwire_scale_eval evaluate --network 17 --label-root <dir> --device cuda [--dry]
    python -m gencodoe.fourwire_scale_eval report

networks converts every eligible network and its two rating variants once, so that parallel jobs only read.

evaluate, for one target network: every model of that network (PG-net, PG-scratch, PG-ft, PG-pre used
directly) and the U1 prior go through one device path (Pipeline): U1 on the device, features, the model in
FP32, KCL currents and the four screening quantities. The screen is the ENWL1 GlobalMargins, fitted on the
calibration states; decisions, state accuracy, the rating-sensitivity variants and the agreement of the
exact sweep with OpenDSS follow Amendment 8, items 8 to 11.
- Final: runs from the frozen source only and is the only code that reads the held-out labels. It first
  claims the network's evaluation folder (one valid evaluation per network; an attempt that died is set
  aside under the rules of the fits), and TestStore serves only the process that owns that claim. Writes
  results/fourwire_scale_eval/network_<n>.json and per-state outcomes outside git, and prints no result.
- --dry: models of dry-run fits, train and val labels only: the first half of the val states stands in for
  the calibration set and the second half for the held-out set.

report computes the E3 statistics of §7 from the 24 evaluation records.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import csiro_convert as cc
from . import fourwire_model as fm
from . import fourwire_scale as fs
from . import fourwire_scale_data as fd
from . import fourwire_scale_train as tt
from .fourwire_eval import GlobalMargins
from .fourwire_net import V_PN_NOM, load_fourwire, phase_neutral_pu
from .fourwire_sweep_opt import TR_KVA, NumbaSweep, TorchSweep
from .fourwire_train import CONFIG, VMAX, VMIN

ALPHA = 0.01                       # §5
MICROBATCH = 256                   # Amendment 8, item 13
DIRECTIONS = ("import", "export")
SAFE, VIOLATED, UNCERTAIN = 0, 1, 2
VARIANTS = [v for v in cc.RATING_VARIANTS if v != "primary"]       # §2.4 rating sensitivity
FRESH = [n for n in fs.ELIGIBLE if n != 1]                         # network_1 = ENWL1 is not a fresh network


def eval_runs() -> Path:
    """One folder per evaluated network: its claim, its record and its per-state outcomes."""
    return fs.OUT / "eval"


class TestStore(fd.LabelStore):
    """The one reader of the held-out labels. It serves one network, to the process that owns the claim on
    that network's evaluation (tt.claim_run_folder), after that evaluation has checked its fits, under the
    freeze, and only until the evaluation has a record. All of this is checked when the reader is made and
    again at every read. evaluate creates it; anything else that tries is refused here."""
    __test__ = False               # not a pytest class
    served = ("test",)

    def __init__(self, root, network: int, prov: dict):
        self.network, self.prov = network, prov
        self._authorised()
        super().__init__(root)

    def _authorised(self) -> None:
        tt.check_freeze(self.prov, variants=True)
        folder = eval_runs() / f"network_{self.network}"
        owner, fits = folder / "attempt.json", folder / "fits_checked.json"
        if not owner.exists() or (folder / "record.json").exists():
            raise PermissionError(f"no open claim on the evaluation of network {self.network}")
        a = json.loads(owner.read_text())
        if (a["pid"], a["node"], a["job_id"]) != (os.getpid(), self.prov["node"], self.prov["job_id"]):
            raise PermissionError(f"the evaluation of network {self.network} is claimed by another process")
        want = sorted({f for f in method_fits(self.network).values() if f is not None})
        if not fits.exists() or sorted(json.loads(fits.read_text())) != want:
            raise PermissionError(f"the fits of network {self.network} have not been checked by this evaluation")

    def load(self, network: int, split: str, n_states: int | None = None) -> dict:
        if network != self.network:
            raise PermissionError(f"this reader serves network {self.network} only")
        self._authorised()
        return super().load(network, split, n_states)


# ------------------------------------------------------------------------------------------- device path
class DeviceNet:
    """One network's constants on a device, shared by every model evaluated on it."""

    def __init__(self, st: fm.Static, device):
        self.st, self.dev, self.n = st, torch.device(device), st.net.n_bus
        self.ts = TorchSweep(st.net, st.th, self.dev)                 # U1 is one iteration of this sweep
        self.phys = fm.Physics(st, self.dev)
        cb = torch.as_tensor(st.net.customer_bus, device=self.dev)
        cp = torch.as_tensor(st.net.customer_phase, device=self.dev)
        self.idx_p = cb * 6 + 2 * cp
        self.static = torch.as_tensor(st.node_static, dtype=torch.float32, device=self.dev)

    def u1(self, s):
        """One sweep iteration from the flat start, FP64: (B, n_bus, 4) kV in the network's bus order."""
        v0 = torch.zeros(len(s), self.n, 4, dtype=torch.complex128, device=self.dev)
        v0[:, :, :3] = self.ts.e
        return self.ts._forward(self.ts._currents(v0, s))[:, self.ts.inv]


class Pipeline:
    """Inputs of a batch -> predicted state and the four screening quantities, on the device.
    record and state_dict: a fit's record (its scalers) and selected weights; without them, the U1 prior."""

    def __init__(self, dn: DeviceNet, record: dict | None = None, state_dict: dict | None = None,
                 microbatch: int = MICROBATCH):
        self.dn, self.mb, self.model = dn, microbatch, None
        if record is not None:
            sc, dev = record["scalers"], dn.dev
            self.mu = torch.as_tensor(np.array(sc["x"]["mu"]), dtype=torch.float64, device=dev)
            self.sd = torch.as_tensor(np.array(sc["x"]["sd"]), dtype=torch.float64, device=dev)
            self.batcher = fm.Batcher(dn.st, tt.scaler_from(sc["edge"])(dn.st.edge_raw), dev)
            self.rs = torch.as_tensor(np.array(sc["residual"], np.float32), device=dev)
            self.model = fm.GNS_heterogeneous_4wire(fm.N_NODE_BASE, hidden=CONFIG["hidden"], layers=CONFIG["layers"],
                                                    heads=CONFIG["heads"]).to(dev)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _residual(self, s, v1):
        """The model's voltage correction (B, n_bus, 4) complex64 kV, as fourwire_scale_train.fit's forward."""
        dn, b = self.dn, len(s)
        inj = torch.zeros(b, dn.n * 6, dtype=torch.float64, device=dn.dev)
        inj.index_add_(1, dn.idx_p, s.real)
        inj.index_add_(1, dn.idx_p + 1, s.imag)
        pv = v1 / V_PN_NOM
        x = torch.cat([inj.view(b, dn.n, 6).float(), dn.static.expand(b, -1, -1), pv.real.float(), pv.imag.float()], 2)
        x = ((torch.asinh(x.double()) - self.mu) / self.sd).float()
        ei, ea = self.batcher.graph(b)
        out = self.model(x.reshape(b * dn.n, -1), ei, ea, b).view(b, dn.n, 8) * self.rs
        return torch.complex(out[..., :4], out[..., 4:]) * V_PN_NOM

    @torch.no_grad()
    def predict(self, s, state: bool = False):
        """s (B, n_c) complex kVA per connection, on the host or the device. Returns tensors on the device:
        the four quantities (B, 4) float32 and, with state, voltages and KCL currents (B, n_bus, 4) complex64."""
        dn = self.dn
        s = torch.as_tensor(s, dtype=torch.complex128, device=dn.dev)
        v1 = dn.u1(s)
        v = v1.to(torch.complex64)
        if self.model is not None:
            for k in range(0, len(s), self.mb):
                v[k:k + self.mb] += self._residual(s[k:k + self.mb], v1[k:k + self.mb])
        cur = dn.phys.currents(v, s.to(torch.complex64))
        summ = fm.summaries(dn.phys, v, cur)
        return (summ, v, cur) if state else (summ,)

    def predict_host(self, s, state: bool = False, chunk: int = 1024):
        """predict in chunks, as NumPy arrays."""
        parts = [[x.cpu().numpy() for x in self.predict(s[k:k + chunk], state)] for k in range(0, len(s), chunk)]
        return tuple(np.concatenate(x) for x in zip(*parts))


# ------------------------------------------------------------------------------------------- screen and metrics
def fit_margins(pred: np.ndarray, truth: np.ndarray, conv: np.ndarray):
    """The ENWL1 margins on the converged calibration states; None if a predicted quantity is not finite
    there (Amendment 8, item 8: the method then has no screen and defers on every state)."""
    if not np.isfinite(pred[conv]).all():
        return None
    return GlobalMargins(pred, truth, conv, alpha=ALPHA)


def margins_state(m) -> dict | None:
    return None if m is None else {"lo": [float(x) for x in m.lo], "hi": [float(x) for x in m.hi]}


def margins_from(state: dict | None):
    if state is None:
        return None
    m = GlobalMargins.__new__(GlobalMargins)
    m.lo, m.hi = list(state["lo"]), list(state["hi"])
    return m


def screen_codes(margins, pred: np.ndarray) -> np.ndarray:
    """(S, 2) screen outcome per direction (import, export): SAFE, VIOLATED or UNCERTAIN."""
    out = np.full((len(pred), 2), UNCERTAIN, np.uint8)
    if margins is not None:
        for j, d in enumerate(DIRECTIONS):
            s = margins.screen(pred, d)
            out[s == "safe", j], out[s == "violated", j] = SAFE, VIOLATED
    return out


def limit_flags(summ: np.ndarray) -> np.ndarray:
    """(S, 4) undervoltage, overvoltage, cable overload, transformer overload from the four quantities."""
    return np.column_stack([summ[:, 0] < VMIN, summ[:, 1] > VMAX, summ[:, 2] > 1.0, summ[:, 3] > 1.0])


def safe_by_direction(flags: np.ndarray, conv: np.ndarray) -> np.ndarray:
    """(S, 2): converged, no cable or transformer overload, and no undervoltage (import) / overvoltage (export)."""
    base = conv & ~flags[:, 2] & ~flags[:, 3]
    return np.column_stack([base & ~flags[:, 0], base & ~flags[:, 1]])


def clopper_pearson_upper(k: int, n: int, level: float = 0.95) -> float | None:
    """One-sided upper bound of a binomial rate."""
    from scipy.stats import beta
    if n == 0:
        return None
    return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))


def decision_metrics(codes: np.ndarray, flags: np.ndarray, conv: np.ndarray) -> dict:
    """Amendment 8, item 9. Truth: OpenDSS flags and convergence. Rates and their bounds are per direction:
    the two directions share the state, so their pairs are not independent trials. The totals over the two
    directions are counts only."""
    safe, out = safe_by_direction(flags, conv), {}
    for j, d in enumerate(DIRECTIONS):
        c = codes[:, j]
        fs_, n = int(((c == SAFE) & conv & ~safe[:, j]).sum()), int((conv & ~safe[:, j]).sum())
        out[d] = {"cleared": int((c == SAFE).sum()), "rejected": int((c == VIOLATED).sum()),
                  "fallback": int((c == UNCERTAIN).sum()), "false_safe": fs_,
                  "cleared_indeterminate": int(((c == SAFE) & ~conv).sum()),
                  "false_unsafe": int(((c == VIOLATED) & safe[:, j]).sum()), "converged_not_safe": n,
                  "false_safe_rate": fs_ / n if n else None, "false_safe_rate_upper95": clopper_pearson_upper(fs_, n)}
    tot = {k: sum(out[d][k] for d in DIRECTIONS) for k in ("false_safe", "cleared_indeterminate", "false_unsafe")}
    return {**out, **tot, "states": int(len(codes)),
            "fallback_share": float(np.mean([out[d]["fallback"] / len(codes) for d in DIRECTIONS]))}


def _stats(e: np.ndarray, quantiles: bool = False) -> dict:
    out = {"mae": float(e.mean()), "max": float(e.max())}
    if quantiles:
        out.update({"p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99))})
    return out


def root_proxy(v: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """The LV-root loading proxy of fourwire_model.summaries from voltages and root currents (kV, kA)."""
    return np.abs(v[:, 0, :3] * np.conj(cur[:, 0, :3])).sum(1) * 1e3 / TR_KVA


def state_metrics(net, v: np.ndarray, cur: np.ndarray, summ: np.ndarray, truth: dict) -> dict:
    """Amendment 8, item 10: errors against OpenDSS on the converged states with a finite prediction."""
    finite = np.isfinite(v).all((1, 2)) & np.isfinite(cur).all((1, 2)) & np.isfinite(summ).all(1)
    m = truth["converged"] & finite
    out = {"converged_states": int(truth["converged"].sum()),
           "nonfinite_predictions": int((truth["converged"] & ~finite).sum())}
    if not m.any():
        return out
    vt, it = truth["V"][m], truth["I"][m]
    v, cur = v[m].astype(complex), cur[m].astype(complex)
    rat = (net.rating_a[1:] / 1e3)[None, :, None]
    ei = np.abs(np.abs(cur[:, 1:]) - np.abs(it[:, 1:])) / rat
    out.update({
        "vpn": _stats(np.abs(phase_neutral_pu(net, v) - phase_neutral_pu(net, vt)), quantiles=True),
        "complex_voltage": _stats(np.abs(v - vt) / V_PN_NOM),
        "neutral_voltage": _stats(np.abs(np.abs(v[:, :, 3]) - np.abs(vt[:, :, 3])) / V_PN_NOM),
        "phase_current_over_rating": _stats(ei[..., :3]),
        "neutral_current_over_rating": _stats(ei[..., 3]),
        "transformer_loading": _stats(np.abs(summ[m, 3] - truth["summary"][m, 3])),
        "transformer_loading_proxy_of_opendss_state": _stats(np.abs(root_proxy(vt, it) - truth["summary"][m, 3]))})
    return out


def sweep_agreement(st: fm.Static, s: np.ndarray, flags: np.ndarray, conv: np.ndarray) -> tuple[dict, np.ndarray]:
    """The exact sweep against OpenDSS on the evaluated states (Amendment 9, item 6). Returns the report and
    the exact engine's decision (S, 2) under the §3 rule: the sweep's where it converges, OpenDSS's elsewhere."""
    out = NumbaSweep(st.net, st.th).solve(s, return_state=False)
    sc, sf = out["converged"], limit_flags(out["summary"])
    both = sc & conv
    truth_safe, sweep_safe = safe_by_direction(flags, conv), safe_by_direction(sf, sc)
    exact_safe = np.where(sc[:, None], sweep_safe, truth_safe)
    names = ("undervoltage", "overvoltage", "cable_overload", "transformer_overload")
    rep = {"states": int(len(conv)), "opendss_indeterminate": int((~conv).sum()),
           "sweep_nonconverged": int((~sc).sum()),
           "sweep_nonconverged_opendss_converged": int((~sc & conv).sum()),
           "sweep_converged_opendss_indeterminate": int((sc & ~conv).sum()),
           "sweep_clears_opendss_indeterminate_pairs": int((sweep_safe & ~conv[:, None]).sum()),
           "flag_differs_where_both_converge": {k: int(((sf[:, j] != flags[:, j]) & both).sum())
                                                for j, k in enumerate(names)},
           "sweep_clears_opendss_flags": {k: int((~sf[:, j] & flags[:, j] & both).sum()) for j, k in enumerate(names)},
           "decision_differs_pairs": int((exact_safe != truth_safe).sum()),
           "sweep_safe_opendss_not_safe_pairs": int((exact_safe & ~truth_safe).sum())}
    return rep, exact_safe


def fallback_agreement(codes: np.ndarray, exact_safe: np.ndarray, flags: np.ndarray, conv: np.ndarray) -> dict:
    """What the exact engine decides on the (state, direction) pairs a screen sends to it, against OpenDSS."""
    fb, truth_safe = codes == UNCERTAIN, safe_by_direction(flags, conv)
    return {"fallback_pairs": int(fb.sum()), "decision_differs_pairs": int((fb & (exact_safe != truth_safe)).sum()),
            "exact_safe_opendss_not_safe_pairs": int((fb & exact_safe & ~truth_safe).sum())}


# ------------------------------------------------------------------------------------------- evaluation
def connection_power(st: fm.Static, d: dict) -> np.ndarray:
    """(S, n_c) complex kVA per phase connection from a label set's customer powers."""
    _, idx, share = cc.customers(st.net.customer_names)
    return d["p_kw_customer"][:, idx] * share + 1j * d["q_kvar_customer"][:, idx] * share


def variant_static(network: int, variant: str) -> fm.Static:
    return fm.Static(load_fourwire(fs.ensure_network_folder(network, variant)), fs.frozen_thevenin(network))


def materialise_networks() -> int:
    """Convert every eligible network and its rating variants once (parallel jobs then only read them)."""
    for n in fs.ELIGIBLE:
        for variant in cc.RATING_VARIANTS:
            fs.ensure_network_folder(n, variant)
    return len(fs.ELIGIBLE)


def method_fits(network: int) -> dict:
    """Method name -> run name of the fit whose model it uses (None for the U1 prior)."""
    pre = tt.run_name("pgpre", fold=tt.fold_of(network), seed=0)
    m = {"U1": None, "PG-net": tt.run_name("pgnet", network=network), "PG-pre-tc": pre, "PG-zs": pre}
    for kind, label in (("scratch", "PG-scratch"), ("ft", "PG-ft")):
        for b, seeds in tt.BUDGETS.items():
            for s in seeds:
                m[f"{label}_b{b}_seed{s}"] = tt.run_name(kind, network=network, budget=b, seed=s)
    return m


def eval_dir() -> Path:
    return fs.RESULTS / "fourwire_scale_eval"


def _halves(d: dict) -> tuple[dict, dict]:
    h = len(d["converged"]) // 2
    return {k: v[:h] for k, v in d.items()}, {k: v[h:] for k, v in d.items()}


def missing_evaluations() -> list[int]:
    """Networks without a valid final evaluation record under this freeze."""
    out = []
    for n in fs.ELIGIBLE:
        f = eval_dir() / f"network_{n}.json"
        if not f.exists() or not evaluation_record(n)["valid"]:
            out.append(n)
    return out


def evaluation_record(network: int) -> dict:
    """A final evaluation record, checked: the network's, usable under this freeze (tt.check_final_record),
    identical to the copy in the evaluation folder, with the per-state file its record names."""
    f = eval_dir() / f"network_{network}.json"
    rec = json.loads(f.read_text())
    if rec["network"] != network:
        raise AssertionError(f"{f.name} is not the evaluation of network {network}")
    tt.check_final_record(rec, f.name, "evaluation")
    if fs._sha(eval_runs() / f"network_{network}" / "record.json") != fs._sha(f):
        raise AssertionError(f"{f.name} differs from the record in the evaluation folder")
    per_state(network, rec)
    return rec


def per_state(network: int, rec: dict):
    """The per-state outcomes of an evaluation, with the hash its record names."""
    f = eval_runs() / f"network_{network}" / rec["per_state_file"]
    if fs._sha(f) != rec["per_state_sha256"]:
        raise AssertionError(f"{f} does not have the hash of its record")
    return np.load(f)


def evaluate(label_root: Path, network: int, device: str, dry: bool = False) -> Path:
    final = not dry
    prov = tt.start_provenance(device, tt.set_determinism(True))
    earlier = []
    if final:
        tt.check_freeze(prov, variants=True)
        out, done = eval_dir() / f"network_{network}.json", eval_runs() / f"network_{network}"
        if out.exists() and not (done / "record.json").exists():
            raise RuntimeError(f"{out} exists without a completed evaluation behind it")
    else:
        out = fs.OUT / "eval_dry" / f"network_{network}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    base, fits = tt.runs(dry), method_fits(network)
    runs = {}
    for name in sorted({f for f in fits.values() if f is not None}):   # every fit must be complete first
        f = base / name / "record.json"
        if not f.exists():
            if dry:
                continue
            raise FileNotFoundError(f"{name}: no fit record; the network is evaluated after all its fits")
        rec = tt.fit_record(base, name, dry)
        if rec["kind"] == "ft":                                       # the chain PG-pre -> PG-ft, same fold and seed
            pre_name = tt.run_name("pgpre", fold=rec["fold"], seed=rec["seed"])
            pre = tt.fit_record(base, pre_name, dry)
            if (rec["started_from"] != {"name": pre["name"], "checkpoint_sha256": pre["checkpoint_sha256"]}
                    or rec["initial_weights_sha256"] != pre["selected_weights_sha256"]):
                raise AssertionError(f"{name}: did not start from the selected checkpoint of {pre['name']}")
        runs[name] = (tt.load_run(base, name, device, dry) if rec["checkpoint_sha256"]
                      else {"record": rec, "state_dict": None})
    if final:                                                         # the claim comes before any held-out label

        def usable(rec):                                              # a completed evaluation: its record and its file
            tt.check_final_record(rec, done.name, "evaluation")
            if rec["network"] != network or fs._sha(done / rec["per_state_file"]) != rec["per_state_sha256"]:
                raise AssertionError(f"{done / rec['per_state_file']} does not have the hash of its record")

        folder, earlier = tt.claim_run_folder(eval_runs(), f"network_{network}", prov, final=True, validate=usable,
                                              publish_to=out)
        fs._write_json(folder / "fits_checked.json", sorted(runs))    # what the reader of the held-out labels asks for
    else:
        folder = out.parent
        folder.mkdir(parents=True, exist_ok=True)
    with tt.halting(f"evaluation of network {network}", prov, final):
        store = fd.LabelStore(label_root)
        st = fm.Static(*fs._case(network))
        if dry:                                                       # train and val only
            cal, te = _halves(store.load(network, "val"))
        else:
            cal, te = store.load(network, "calib"), TestStore(label_root, network, prov).load(network, "test")
        s_cal, s_te = connection_power(st, cal), connection_power(st, te)
        nets = {"primary": DeviceNet(st, device), **{v: DeviceNet(variant_static(network, v), device) for v in VARIANTS}}
        flags = {"primary": te["flags"]}
        for v in VARIANTS:                                            # truth cable flag under the variant's ratings
            rat = (nets[v].st.net.rating_a[1:] / 1e3)[None, :, None]
            f = te["flags"].copy()
            f[:, 2] = (np.abs(te["I"][:, 1:, :3]) / rat).max((1, 2)) > 1.0
            flags[v] = f
        agree, exact_safe = sweep_agreement(st, s_te, te["flags"], te["converged"])
        arrays = {"exact_safe": exact_safe, "converged": te["converged"],
                  "truth_safe": safe_by_direction(te["flags"], te["converged"])}
        methods, cache = {}, {}

        def pool_margins(rec, sd):                                    # PG-zs: the pool's calibration states
            pred, truth, conv = [], [], []
            for m in rec["pool"]:
                stm = fm.Static(*fs._case(m))
                d = _halves(store.load(m, "val"))[0] if dry else store.load(m, "calib")
                pred.append(Pipeline(DeviceNet(stm, device), rec, sd).predict_host(connection_power(stm, d))[0])
                truth.append(d["summary"])
                conv.append(d["converged"])
            pred, truth, conv = np.concatenate(pred), np.concatenate(truth), np.concatenate(conv)
            return fit_margins(pred, truth, conv), {"set": "pool calib", "networks": rec["pool"],
                                                   "states": int(len(conv)), "converged": int(conv.sum())}

        for key, name in fits.items():
            if name is not None and name not in runs:
                continue                                              # dry run without that fit
            rec, sd = (None, None) if name is None else (runs[name]["record"], runs[name]["state_dict"])
            res = {"fit": name, "checkpoint_sha256": rec["checkpoint_sha256"] if rec else None,
                   "selected_epoch": rec["best_epoch"] if rec else None}
            if name is not None and sd is None:                       # Amendment 7, item 4: no model
                res.update({"no_model": True, "calibration": None, "state": None})
                codes = {v: np.full((len(s_te), 2), UNCERTAIN, np.uint8) for v in nets}
            else:
                if name not in cache:                                 # PG-pre-tc and PG-zs share one model
                    pipes = {v: Pipeline(nets[v], rec, sd) for v in nets}
                    summ, v_pred, cur = pipes["primary"].predict_host(s_te, state=True)
                    cache = {name: {"cal": pipes["primary"].predict_host(s_cal)[0], "summ": summ,
                                    "state": state_metrics(st.net, v_pred, cur, summ, te),
                                    "variants": {v: pipes[v].predict_host(s_te)[0] for v in VARIANTS}}}
                    del v_pred, cur, pipes
                c = cache[name]
                if key == "PG-zs":
                    margins, cal_info = pool_margins(rec, sd)
                else:
                    margins = fit_margins(c["cal"], cal["summary"], cal["converged"])
                    cal_info = {"set": "target calib", "states": int(len(cal["converged"])),
                                "converged": int(cal["converged"].sum())}
                res.update({"calibration": {**cal_info, "usable": margins is not None, "margins": margins_state(margins)},
                            "state": c["state"]})
                codes = {"primary": screen_codes(margins, c["summ"]),
                         **{v: screen_codes(margins, c["variants"][v]) for v in VARIANTS}}
                arrays[f"{key}__summary"] = c["summ"]
            res["screen"] = decision_metrics(codes["primary"], flags["primary"], te["converged"])
            res["fallback_exact_engine"] = fallback_agreement(codes["primary"], exact_safe, te["flags"], te["converged"])
            for v in VARIANTS:
                res[f"screen_{v}"] = decision_metrics(codes[v], flags[v], te["converged"])
            arrays[f"{key}__codes"] = codes["primary"]
            methods[key] = res
            if dry:                                                   # a final run prints no result (Amendment 8, item 14)
                print(key, json.dumps({k: res["screen"][k] for k in ("fallback_share", "false_safe",
                                                                     "cleared_indeterminate")}), flush=True)
            else:
                print(key, "done", flush=True)
        unchanged = tt.checkout_unchanged(prov, variants=final)
        npz = folder / ("per_state.npz" if final else out.stem + ".npz")
        np.savez_compressed(npz, **arrays)
        record = {
            "protocol": "docs/fourwire_scale_protocol.md §5-§7, Amendments 8 and 9", "network": network,
            "fold": tt.fold_of(network), "dry_run": dry, "evaluated_split": "second half of val" if dry else "test",
            "calibration_split": "first half of val" if dry else "calib", **prov,
            "unchanged_during_run": unchanged, "valid": bool(unchanged), "label_commit": fd.LABEL_COMMIT,
            "earlier_attempts": earlier,
            "screen_implementation_sha256": fs._sha(Path(__file__).with_name("fourwire_eval.py")),
            "alpha": ALPHA, "microbatch": MICROBATCH, "retained_buses": st.net.n_bus,
            "states": int(len(te["converged"])), "converged": int(te["converged"].sum()),
            "truth": {"converged_not_safe": {d: int((te["converged"] & ~safe_by_direction(te["flags"], te["converged"])[:, j]).sum())
                                             for j, d in enumerate(DIRECTIONS)}},
            "exact_sweep_vs_opendss": agree, "per_state_file": npz.name, "per_state_sha256": fs._sha(npz),
            "methods": methods}
        if final:                                                     # the folder's record first: it closes the claim
            fs._write_json(folder / "record.json", record)
            tt.publish(folder, out)
        else:
            fs._write_json(out, record)
        if not unchanged:
            raise RuntimeError(f"checkout, numerical mode, libraries or inputs changed during the run; "
                               f"{out.name} is not a valid record")
    return out


# ------------------------------------------------------------------------------------------- E3 report
def _boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    """Percentile interval of the mean, resampling networks with replacement (§7)."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def report() -> Path:
    if tt.halt_file().exists():
        raise RuntimeError(f"the final runs are halted: {tt.halt_file().read_text()}")
    recs = {n: evaluation_record(n) for n in fs.ELIGIBLE}
    if not all(r["valid"] for r in recs.values()):
        raise AssertionError("an evaluation record is not valid")

    def get(n, key, *path):
        """A value of a method's record; None where the method has no model, no screen or no finite prediction."""
        x = recs[n]["methods"][key]
        for k in path:
            x = None if x is None else x.get(k)
        return x

    def compare(budget, seeds, nets):
        """PG-scratch minus PG-ft fallback share per network (mean over seeds), with its mean and interval."""
        per = {n: {s: get(n, f"PG-scratch_b{budget}_seed{s}", "screen", "fallback_share")
                   - get(n, f"PG-ft_b{budget}_seed{s}", "screen", "fallback_share") for s in seeds} for n in nets}
        d = np.array([np.mean(list(per[n].values())) for n in nets])
        tot = {m: {k: sum(get(n, f"{m}_b{budget}_seed{s}", "screen", k) for n in nets for s in seeds)
                   for k in ("false_safe", "cleared_indeterminate")} for m in ("PG-ft", "PG-scratch")}
        ci = _boot(d)
        return {"networks": len(nets), "seeds": list(seeds), "mean_difference": float(d.mean()), "ci95": ci,
                "ge_2_points": bool(d.mean() >= 0.02), "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0),
                "false_safe": {m: tot[m]["false_safe"] for m in tot},
                "cleared_indeterminate": {m: tot[m]["cleared_indeterminate"] for m in tot},
                "ft_not_more_false_safe": tot["PG-ft"]["false_safe"] <= tot["PG-scratch"]["false_safe"],
                "ft_not_more_cleared_indeterminate":
                    tot["PG-ft"]["cleared_indeterminate"] <= tot["PG-scratch"]["cleared_indeterminate"],
                "per_network_per_seed": {str(n): {str(s): v for s, v in per[n].items()} for n in nets}}

    def table(keys, nets):
        """Unweighted means over networks, and totals of the error counts."""
        out = {}
        for key in keys:
            mae = [get(n, key, "state", "vpn", "mae") for n in nets]
            out[key] = {"fallback_share": float(np.mean([get(n, key, "screen", "fallback_share") for n in nets])),
                        "false_safe": sum(get(n, key, "screen", "false_safe") for n in nets),
                        "cleared_indeterminate": sum(get(n, key, "screen", "cleared_indeterminate") for n in nets),
                        "false_unsafe": sum(get(n, key, "screen", "false_unsafe") for n in nets),
                        "vpn_mae": None if any(m is None for m in mae) else float(np.mean(mae)),
                        "networks_without_model_or_screen": sum(
                            1 for n in nets if get(n, key, "calibration") is None
                            or not get(n, key, "calibration", "usable"))}
        return out

    primary = compare(1000, (0, 1, 2), FRESH)
    primary["criterion_met"] = bool(primary["ge_2_points"] and primary["ci95"][0] > 0
                                    and primary["ft_not_more_false_safe"]
                                    and primary["ft_not_more_cleared_indeterminate"])
    seed0 = ["U1", "PG-net", "PG-pre-tc", "PG-zs"] + [f"{m}_b{b}_seed0" for b in tt.BUDGETS for m in ("PG-scratch", "PG-ft")]
    out = fs.RESULTS / "fourwire_scale_e3.json"
    fs._write_json(out, {
        "protocol": "docs/fourwire_scale_protocol.md §7 E3, Amendment 8",
        "fresh_networks": FRESH, "fallback_share_unit": "fraction of test states, mean of the two directions",
        "primary_b1000": primary,
        "secondary_seed0": {f"b{b}": compare(b, (0,), FRESH) for b in (200, 5000)},
        "sensitivity_with_network_1": {"primary_b1000": compare(1000, (0, 1, 2), fs.ELIGIBLE)},
        "methods_seed0_fresh": table(seed0, FRESH), "methods_seed0_all_24": table(seed0, fs.ELIGIBLE),
        "per_network": {str(n): {key: {"fallback_share": get(n, key, "screen", "fallback_share"),
                                       "false_safe": get(n, key, "screen", "false_safe"),
                                       "cleared_indeterminate": get(n, key, "screen", "cleared_indeterminate"),
                                       "vpn_mae": get(n, key, "state", "vpn", "mae")}
                                 for key in recs[n]["methods"]} for n in fs.ELIGIBLE},
        "evaluation_records_sha256": {str(n): fs._sha(eval_dir() / f"network_{n}.json") for n in fs.ELIGIBLE}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["networks", "evaluate", "report"])
    ap.add_argument("--label-root", type=Path, help="folder holding <split>/network_N/ label shards")
    ap.add_argument("--network", type=int, choices=fs.ELIGIBLE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry", action="store_true", help="evaluate dry-run fits on val states only")
    a = ap.parse_args()
    if a.cmd == "networks":
        print(materialise_networks(), "networks converted, with their rating variants")
    elif a.cmd == "report":
        print(report())
    else:
        if a.label_root is None or a.network is None:
            ap.error("evaluate needs --label-root and --network")
        try:
            print(evaluate(a.label_root, a.network, a.device, a.dry))
        except tt.AlreadyDone as e:                                  # the job script reads 10 as "skip"
            print(e)
            raise SystemExit(10)


if __name__ == "__main__":
    main()
