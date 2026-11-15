"""E2 of the 25-network study: end-to-end timing and the envelope check
(docs/fourwire_scale_protocol.md §6, §7 E2, Amendments 3 and 9).

    python -m gencodoe.fourwire_scale_e2 timing --network 17 --device cuda [--dry --warmup 1 --repeats 3]
    python -m gencodoe.fourwire_scale_e2 envelopes --network 17 --device cuda [--dry --days 1 --hours 2]
    python -m gencodoe.fourwire_scale_e2 report

timing: per batch size (1, 256, 4,096) every series delivers, in host memory, the four screening quantities,
the decision per direction and the convergence flag of every state of the batch:
- opendss, numpy (the reference sweep, batch 1 only), sweep_cpu (Numba), sweep_gpu (PyTorch);
- u1_* and pgnet_*: the screen pipeline of fourwire_scale_eval with the CPU or the GPU sweep as its
  exact fallback;
- *_noopendss: the same, with non-converged sweep states declared unsafe without an OpenDSS solve
  (sensitivity, outside the criterion).
The series run round-robin, 3 warm-up and 100 timed rounds. Inputs are regenerated from the seeds: no label
is read. Margins come from the evaluation record, weights from the fit.

envelopes: uniform hourly envelopes on 12 held-out days, both directions, with the ENWL1 rule
(envelopes.scalar_envelope), evaluated by the Numba sweep (A) and by the PG-net screen with the sweep as
its fallback (B). A state on which the sweep does not converge is infeasible for both.

report: speed-ups with their bootstrap intervals, the criterion of §7, the sensitivity series, scaling
exponents and the envelope agreement, from the timing, evaluation and envelope records.

--dry uses val states (timing) or a val day (envelopes) and the models and margins of dry runs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from . import csiro_convert as cc
from . import datakit_fluvius as dk
from . import fourwire_scale as fs
from .fourwire_net import V_PN_NOM, phase_neutral_pu
from .fourwire_sweep_opt import TR_KVA

E2_NETWORKS = [25, 10, 1, 7, 2, 17]                 # §7 E2, by size
BATCHES = (1, 256, 4096)                            # §6
WARMUP, REPEATS = 3, 100                            # §6
OPENDSS_LARGE = (1, 10)                             # Amendment 9, item 3: OpenDSS at batch 4,096
E2_CORES = 16                                       # Amendment 9, item 1
SCREENS = ("u1", "pgnet")
ENGINES = ("cpu", "gpu")
N_SCENARIOS, EPS, BETA, LEVELS, REL_TOL = 200, 0.05, 0.05, 4, 1e-3     # §6 envelope rule
CAP_KW = {"export": dk.FLEX_EXPORT_KW, "import": dk.FLEX_IMPORT_KW}
MIX = (0.3, 0.3, 0.4)                               # heat pump, EV, PV: the ENWL1 DOE shares (doe_day.MIX)


def allocated_cores() -> int:
    """Cores of the job: what Slurm allocated to the task, else what the process may run on."""
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


# ------------------------------------------------------------------------------------------- OpenDSS workers
_ENGINE = None


def _init_engine(network: int, ready):
    """One OpenDSS engine per worker, built and primed (no-load solve) once; the worker then reports ready."""
    global _ENGINE
    from .unbalanced import UnbalancedENWL1
    _ENGINE = UnbalancedENWL1(fs.ensure_network_folder(network), **fs.OPENDSS)
    ready.put(os.getpid())


def _solve_chunk(args) -> np.ndarray:
    """(k, 5): the four quantities and the convergence flag of a chunk. The engine first solves the no-load
    case, so a chunk does not depend on what the worker solved before; its states are then solved in order
    under the §3 failure rule, each from the previous solution (as in label generation)."""
    p, q = args
    out = np.empty((len(p), 5))
    if len(p):
        _ENGINE._reset()
    for i in range(len(p)):
        r = _ENGINE.solve(p[i], q[i])
        out[i] = [r["v_pn_min"], r["v_pn_max"], r["line_loading_max"], r["transformer_loading"], r["converged"]]
    return out


class OpenDSSPool:
    def __init__(self, network: int, processes: int):
        ctx = mp.get_context("spawn")
        ready = ctx.Queue()
        self.processes = processes
        self.pool = ctx.Pool(processes, _init_engine, (network, ready))
        pids = {ready.get(timeout=900) for _ in range(processes)}     # every engine is built before anything is timed
        if len(pids) != processes:
            raise RuntimeError("not every OpenDSS worker started")

    def solve(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """s (B, n_c) complex kVA -> quantities (B, 4), converged (B,). At most one chunk per worker."""
        if len(s) == 0:
            return np.zeros((0, 4)), np.zeros(0, bool)
        parts = np.array_split(np.arange(len(s)), min(self.processes, len(s)))
        res = np.concatenate(self.pool.map(_solve_chunk, [(s.real[i], s.imag[i]) for i in parts], chunksize=1))
        return res[:, :4], res[:, 4] > 0.5

    def close(self):
        self.pool.terminate()


# ------------------------------------------------------------------------------------------- timed series
def decide(summ: np.ndarray, conv: np.ndarray) -> np.ndarray:
    """Exact-engine decision (B, 2), import and export: converged and within the limits of the direction."""
    from .fourwire_scale_eval import limit_flags, safe_by_direction
    return safe_by_direction(limit_flags(summ), conv)


class Series:
    """The timed methods of one network. Every method maps a batch s (B, n_c) complex kVA (P + jQ per
    connection) in host memory to (quantities (B, 4), safe (B, 2), converged (B,)) in host memory."""

    def __init__(self, network: int, device: str, processes: int, models: dict):
        """models: screen name -> {"record", "state_dict", "margins", "model"}. record None is the U1 prior;
        model False is a fit that ended without a model (Amendment 7, item 4): its screen defers every state."""
        import torch

        from . import fourwire_model as fm
        from . import fourwire_scale_eval as ev
        from .fourwire_sweep_opt import NumbaSweep
        self.torch, self.ev = torch, ev
        self.net, self.th = fs._case(network)
        self.pool = OpenDSSPool(network, processes)                    # before CUDA and Numba start threads
        self.dn = ev.DeviceNet(fm.Static(self.net, self.th), device)
        self.numba = NumbaSweep(self.net, self.th)
        self.pipes = {k: (ev.Pipeline(self.dn, m["record"], m["state_dict"]) if m["model"] else None, m["margins"])
                      for k, m in models.items()}
        self.rating = self.net.rating_a[1:] / 1e3
        self.opendss_calls = 0

    def _to_opendss(self, s, summ, conv):
        """§3: the states an exact sweep did not converge on get OpenDSS's quantities and convergence flag."""
        bad = ~conv
        if bad.any():
            self.opendss_calls += int(bad.sum())
            summ[bad], conv[bad] = self.pool.solve(s[bad])
        return summ, conv

    def opendss(self, s):
        summ, conv = self.pool.solve(s)
        return summ, decide(summ, conv), conv

    def numpy(self, s):
        """The reference sweep, one state at a time (Amendment 3), with the quantities of fourwire_model.summaries."""
        summ, conv = np.empty((len(s), 4)), np.empty(len(s), bool)
        for k in range(len(s)):
            v, cur, _, conv[k] = fs.reference_sweep_state(self.net, self.th, s[k])
            vpn = phase_neutral_pu(self.net, v[None])[0]
            summ[k] = [vpn.min(), vpn.max(), (np.abs(cur[1:, :3]) / self.rating[:, None]).max(),
                       np.abs(v[0, :3] * np.conj(cur[0, :3])).sum() * 1e3 / TR_KVA]
        summ, conv = self._to_opendss(s, summ, conv)
        return summ, decide(summ, conv), conv

    def sweep_cpu(self, s, opendss=True, s_dev=None):
        out = self.numba.solve(s, return_state=False)
        summ, conv = out["summary"], out["converged"]
        if opendss:
            summ, conv = self._to_opendss(s, summ, conv)
        return summ, decide(summ, conv), conv

    def sweep_gpu(self, s, opendss=True, s_dev=None):
        """The batch goes to the device (unless it is there already); the four quantities and the
        convergence flags come back, nothing else."""
        out = self.dn.ts.solve(s if s_dev is None else s_dev, return_state=False, to_host=False)
        summ, conv = out["summary"].cpu().numpy(), out["converged"].cpu().numpy()
        if opendss:
            summ, conv = self._to_opendss(s, summ, conv)
        return summ, decide(summ, conv), conv

    def screened(self, screen: str, engine: str, s, opendss=True):
        """Amendment 9, item 4: device path; the four quantities to the host; the ENWL1 screen there, in both
        directions; then one exact solve for every state that is uncertain in at least one direction."""
        torch, ev = self.torch, self.ev
        pipe, margins = self.pipes[screen]
        s_dev = torch.as_tensor(s, dtype=torch.complex128, device=self.dn.dev)
        if pipe is None:                                              # no model: every state is deferred
            summ, codes = np.full((len(s), 4), np.nan), np.full((len(s), 2), ev.UNCERTAIN, np.uint8)
        else:
            summ = pipe.predict(s_dev)[0].cpu().numpy().astype(np.float64)
            codes = ev.screen_codes(margins, summ)
        safe, conv = codes == ev.SAFE, np.ones(len(s), bool)
        fb = (codes == ev.UNCERTAIN).any(1)
        if fb.any():
            solve = self.sweep_cpu if engine == "cpu" else self.sweep_gpu
            sub = s_dev[torch.as_tensor(fb, device=self.dn.dev)] if engine == "gpu" else None
            summ[fb], safe[fb], conv[fb] = solve(s[fb], opendss, sub)
        return summ, safe, conv, fb

    def table(self) -> dict:
        """Series name -> function of a batch, in the order of a round (Amendment 9, item 5)."""
        t = {"opendss": self.opendss, "numpy": self.numpy}
        for tag, od in (("", True), ("_noopendss", False)):
            t[f"sweep_cpu{tag}"] = lambda s, od=od: self.sweep_cpu(s, od)
            t[f"sweep_gpu{tag}"] = lambda s, od=od: self.sweep_gpu(s, od)
            for screen in self.pipes:
                for engine in ENGINES:
                    t[f"{screen}_{engine}{tag}"] = lambda s, sc=screen, e=engine, od=od: self.screened(sc, e, s, od)
        return t


def split_inputs(network: int, split: str) -> tuple[np.ndarray, str]:
    """(S, n_c) complex kVA per connection of a split, regenerated from the seed, and their content hash,
    which must be the one in the label manifest. No label is read."""
    from . import fourwire_scale_data as fd
    net, _ = fs._case(network)
    names, idx, share = cc.customers(net.customer_names)
    st = dk.customer_states(len(names), split, fs.N_STATES[split], fs.seed(network, split), fs.AUGMENT, fs.CAP_KVA)
    p, q = fs.connection_states(st, idx, share)
    sha = fs.content_sha({"p": p, "q": q})
    unit = next(u for u in json.loads(fd.MANIFEST.read_text())["splits"] if (u["network"], u["split"]) == (network, split))
    if unit["inputs_content_sha256"] != sha:
        raise AssertionError(f"regenerated {split} inputs of network {network} differ from the labelled ones")
    return p + 1j * q, sha


def _summary(times: list[float], batch: int) -> dict:
    t = np.array(times)
    return {"repeats": len(t), "p50_s": float(np.median(t)), "p95_s": float(np.percentile(t, 95)),
            "states_per_s": float(batch / np.median(t)), "times_s": [float(x) for x in t]}


def _models(network: int, device: str, dry: bool) -> tuple[dict, dict]:
    """The U1 and PG-net screens of a network: fit record, weights and the margins of its evaluation record.
    Final: only once every network has a valid evaluation record (no selective release, Amendment 8, item 14)."""
    from . import fourwire_scale_eval as ev
    from . import fourwire_scale_train as tt
    if dry:
        files = sorted((fs.OUT / "eval_dry").glob(f"network_{network}_*.json"))
        if not files:
            raise FileNotFoundError("no dry-run evaluation record of this network")
        rec_file, rec = files[-1], json.loads(files[-1].read_text())
        if not rec["valid"] or not rec["dry_run"]:
            raise AssertionError(f"{rec_file.name} is not a valid dry-run evaluation record")
    else:
        missing = ev.missing_evaluations()
        if missing:
            raise RuntimeError(f"networks {missing} have no valid evaluation record yet")
        rec_file, rec = ev.eval_dir() / f"network_{network}.json", ev.evaluation_record(network)
    name = tt.run_name("pgnet", network=network)
    fit = tt.fit_record(tt.runs(dry), name, dry)
    if fit["checkpoint_sha256"] != rec["methods"]["PG-net"]["checkpoint_sha256"]:
        raise AssertionError("the PG-net checkpoint is not the one that was evaluated")
    has_model = fit["checkpoint_sha256"] is not None
    sd = tt.load_run(tt.runs(dry), name, device, dry)["state_dict"] if has_model else None

    def margins(method):
        cal = rec["methods"][method]["calibration"]
        return None if cal is None else ev.margins_from(cal["margins"])

    return ({"u1": {"record": None, "state_dict": None, "margins": margins("U1"), "model": True},
             "pgnet": {"record": fit, "state_dict": sd, "margins": margins("PG-net"), "model": has_model}},
            {"evaluation_record": rec_file.name, "evaluation_record_sha256": fs._sha(rec_file),
             "pgnet_checkpoint_sha256": fit["checkpoint_sha256"]})


def _run_files(kind: str, network: int, dry: bool, prov: dict):
    """Where a timing or envelope run writes: (folder, record path, earlier attempts). Final: the folder is
    claimed as a fit's is (one valid run per network; an attempt that died is set aside and named)."""
    from . import fourwire_scale_train as tt
    if dry:
        folder = fs.OUT / "e2_dry"
        folder.mkdir(parents=True, exist_ok=True)
        return folder, folder / f"{kind}_network_{network}_{time.strftime('%Y%m%dT%H%M%S')}.json", []
    out, done = fs.RESULTS / f"fourwire_scale_e2_{kind}" / f"network_{network}.json", fs.OUT / "e2" / f"{kind}_network_{network}"
    if tt.completed(done, lambda rec: tt.check_final_record(rec, done.name, kind)):
        tt.publish(done, out)                                         # also after a crash between the two writes
        e2_record(kind, network)
        raise tt.AlreadyDone(f"{out.name}: a network has one {kind} run")
    if out.exists():
        raise RuntimeError(f"{out} exists without a completed run behind it")
    folder, earlier = tt.claim_run_folder(fs.OUT / "e2", f"{kind}_network_{network}", prov, final=True)
    return folder, out, earlier


def _write_record(folder: Path, out: Path, record: dict, dry: bool) -> None:
    if dry:
        fs._write_json(out, record)
    else:                                                             # the folder's record closes the claim
        from . import fourwire_scale_train as tt
        fs._write_json(folder / "record.json", record)
        tt.publish(folder, out)


def e2_record(kind: str, network: int) -> dict:
    """A final timing or envelope record, checked: the network's, valid, under this freeze, identical to the
    copy in its run folder."""
    from . import fourwire_scale_train as tt
    f = fs.RESULTS / f"fourwire_scale_e2_{kind}" / f"network_{network}.json"
    folder = fs.OUT / "e2" / f"{kind}_network_{network}"
    rec = json.loads(f.read_text())
    if rec["network"] != network:
        raise AssertionError(f"{f}: not the {kind} record of network {network}")
    tt.check_final_record(rec, str(f), kind)
    if fs._sha(folder / "record.json") != fs._sha(f):
        raise AssertionError(f"{f} differs from the record in its run folder")
    if kind == "timing" and fs._sha(folder / rec["decisions_file"]) != rec["decisions_sha256"]:
        raise AssertionError(f"{folder / rec['decisions_file']} does not have the hash of its record")
    if kind == "envelopes":
        check_envelopes(rec, str(f))
    return rec


def check_envelopes(rec: dict, what: str) -> None:
    """A final envelope record holds exactly the check of Amendment 9, item 7: the 12 days x 24 hours x 2
    directions, each once, with the frozen settings, and the summary that follows from its intervals."""
    want = {(day, day * 96 + 4 * hour, d) for day in envelope_days() for hour in range(24) for d in ("import", "export")}
    got = [(r["day"], r["step"], r["direction"]) for r in rec["intervals"]]
    settings = {"scenarios": N_SCENARIOS, "eps": EPS, "beta": BETA, "levels_per_round": LEVELS,
                "relative_tolerance": REL_TOL, "cap_kw": CAP_KW, "mix_hp_ev_pv": list(MIX), "hours": 24,
                "days": envelope_days()}
    if len(got) != len(want) or set(got) != want:
        raise AssertionError(f"{what}: not the {len(want)} intervals of the envelope check, each once")
    if any(rec[k] != v for k, v in settings.items()):
        raise AssertionError(f"{what}: not the frozen envelope settings")
    if rec["summary"] != json.loads(json.dumps(envelope_summary(rec["intervals"]))):
        raise AssertionError(f"{what}: the summary is not that of its intervals")


def timing(network: int, device: str, dry: bool = False, warmup: int = WARMUP, repeats: int = REPEATS,
           batches: tuple = BATCHES, opendss_large: tuple = OPENDSS_LARGE) -> Path:
    import numba
    import torch

    from . import fourwire_scale_train as tt
    final = not dry
    prov = tt.start_provenance(device, tt.set_determinism(False))    # timing: PyTorch's default algorithms
    cores = allocated_cores()
    if final:
        tt.check_freeze(prov, deterministic=False)
        if cores != E2_CORES or numba.config.NUMBA_NUM_THREADS < E2_CORES:
            raise RuntimeError(f"timing needs {E2_CORES} cores (Amendment 9, item 1); this job has {cores}")
    models, sources = _models(network, device, dry)
    folder, out, earlier = _run_files("timing", network, dry, prov)
    with tt.halting(f"timing of network {network}", prov, final):
        ser = Series(network, device, cores, models)                 # starts the OpenDSS workers first
        numba.set_num_threads(min(cores, numba.config.NUMBA_NUM_THREADS))
        torch.set_num_threads(cores)
        dev = torch.device(device)
        split = "val" if dry else "test"
        s_all, inputs_sha = split_inputs(network, split)
        n_all = len(s_all)
        table = ser.table()

        def timed(fn, s):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = fn(s)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - t0, res

        def keep(decisions, k, res):
            """Per state, over every timed repeat of the whole split: the first repeat's delivery, whether a
            state was ever delivered safe and whether it always was (Amendment 9, item 6)."""
            if k not in decisions:
                decisions[k] = {"safe": res[1].copy(), "converged": res[2].copy(), "quantities": res[0].copy(),
                                "ever_safe": res[1].copy(), "always_safe": res[1].copy(), "other": 0}
                if len(res) > 3:
                    decisions[k]["fallback"] = res[3].copy()
            else:
                d = decisions[k]
                d["ever_safe"] |= res[1]
                d["always_safe"] &= res[1]
                d["other"] += int(not np.array_equal(res[1], d["safe"]))

        t0 = time.perf_counter()
        ser.numba.solve(s_all[:2], return_state=False)               # Numba compilation, excluded (§6)
        compile_s = time.perf_counter() - t0
        result, decisions = {}, {}
        for batch in batches:
            b = min(batch, n_all)
            names = [k for k in table if k not in ("opendss", "numpy")]
            if b == 1:
                names = ["opendss", "numpy"] + names                 # the NumPy sweep is timed at batch 1 only
            elif batch != batches[-1]:
                names = ["opendss"] + names
            times = {k: [] for k in names}
            info = {k: {} for k in names}
            for r in range(warmup + repeats):
                idx = (r * b + np.arange(b)) % n_all                 # the same states for every series
                s = s_all[idx]
                for k in names[r % len(names):] + names[:r % len(names)]:   # the starting series rotates
                    ser.opendss_calls = 0
                    dt, res = timed(table[k], s)
                    if r < warmup:
                        continue
                    times[k].append(dt)
                    if r == warmup:                                  # the first timed repeat, described
                        info[k] = {"opendss_solves_first_timed_repeat": ser.opendss_calls,
                                   "fallback_states_first_timed_repeat": int(res[3].sum()) if len(res) > 3 else None}
                    if b == n_all:
                        keep(decisions, k, res)
            if "opendss" not in names:                               # the largest batch: fewer repeats, outside the rounds
                times["opendss"], info["opendss"] = [], {}
                for r in range(opendss_large[0] + opendss_large[1]):
                    dt, res = timed(table["opendss"], s_all[(r * b + np.arange(b)) % n_all])
                    if r >= opendss_large[0]:
                        times["opendss"].append(dt)
                        if b == n_all:
                            keep(decisions, "opendss", res)
            result[str(b)] = {k: {**_summary(times[k], b), **info[k]} for k in times}
            if dry:                                                  # a final run prints no result
                print(b, json.dumps({k: round(v["p50_s"], 5) for k, v in result[str(b)].items()}), flush=True)
            else:
                print(b, "done", flush=True)
        ser.pool.close()
        unchanged = tt.checkout_unchanged(prov)
        npz = folder / ("decisions.npz" if final else out.stem + "_decisions.npz")
        np.savez_compressed(npz, **{f"{k}__{a}": v for k, d in decisions.items() for a, v in d.items()
                                    if isinstance(v, np.ndarray)})
        _write_record(folder, out, {
            "protocol": "docs/fourwire_scale_protocol.md §6, §7 E2, Amendments 3 and 9", "network": network,
            "retained_buses": ser.net.n_bus, "phase_connections": len(ser.net.customer_names), "dry_run": dry,
            "states_from": split, "inputs_content_sha256": inputs_sha, **prov,
            "unchanged_during_run": unchanged, "valid": bool(unchanged), **sources, "earlier_attempts": earlier,
            "allocated_cores": cores, "numba_threads": numba.get_num_threads(), "torch_threads": torch.get_num_threads(),
            "opendss_processes": cores, "numba_compile_s": compile_s, "series_order": list(table),
            "warmup_rounds": warmup, "timed_rounds": repeats, "opendss_at_largest_batch": list(opendss_large),
            "repeats_with_other_decisions": {k: d["other"] for k, d in decisions.items()},
            "decisions_file": npz.name, "decisions_sha256": fs._sha(npz), "batches": result}, dry)
        if not unchanged:
            raise RuntimeError(f"checkout, numerical mode, libraries or inputs changed during the run; "
                               f"{out.name} is not a valid record")
    return out


# ------------------------------------------------------------------------------------------- envelope check
def envelope_days(dry: bool = False) -> list[int]:
    """Day-of-year indices (0 = 1 January 2024): every third held-out day, from the first (Amendment 9,
    item 7). A dry run uses val days."""
    d = np.arange(366)
    days = d[dk.day_split(d * 96) == ("val" if dry else "test")]
    return [int(x) for x in days[::3]]


def day_scenarios(network: int, n_cust: int, day: int, hours: int = 24):
    """Yield (step, p_kw, q_kvar), each (200, n_cust) customer net consumption, for the hours of a day.
    Technology flags once per network and day; per scenario, a meter of its group for every customer."""
    from .fluvius import GROUPS, load_group
    groups = [load_group(g) for g in GROUPS]
    by_flags = {(g.hp, g.ev, g.pv): g for g in groups}
    flags = np.random.default_rng([network, day, 0]).uniform(size=(n_cust, 3)) < np.array(MIX)
    keys = sorted({tuple(int(v) for v in f) for f in flags})
    who = {key: np.flatnonzero((flags == key).all(1)) for key in keys}
    rng = np.random.default_rng([network, day, 1])
    for hour in range(hours):
        step = day * 96 + 4 * hour
        p, q = np.empty((N_SCENARIOS, n_cust)), np.empty((N_SCENARIOS, n_cust))
        for sc in range(N_SCENARIOS):
            for key in keys:
                g = by_flags[key]
                pick = dk._pick_meters(rng, len(g.ean), len(who[key]))
                off, inj = g.offtake_kw[pick, step], g.injection_kw[pick, step]
                p[sc, who[key]], q[sc, who[key]] = off - inj, off * dk.TAN_PHI
        yield step, p, q


def envelopes(network: int, device: str, dry: bool = False, days: int | None = None, hours: int = 24) -> Path:
    from . import envelopes as env
    from . import fourwire_model as fm
    from . import fourwire_scale_eval as ev
    from . import fourwire_scale_train as tt
    from .fourwire_sweep_opt import NumbaSweep
    final = not dry
    prov = tt.start_provenance(device, tt.set_determinism(True))
    if final:
        tt.check_freeze(prov, profiles=True)
    models, sources = _models(network, device, dry)
    folder, out, earlier = _run_files("envelopes", network, dry, prov)
    with tt.halting(f"envelopes of network {network}", prov, final):
        net, th = fs._case(network)
        names, idx, share = cc.customers(net.customer_names)
        n_cust = len(names)
        sweep = NumbaSweep(net, th)
        m = models["pgnet"]
        pipe = ev.Pipeline(ev.DeviceNet(fm.Static(net, th), device), m["record"], m["state_dict"]) if m["model"] else None
        customers = env.Customers(bus=np.arange(n_cust), export_cap=np.full(n_cust, CAP_KW["export"]),
                                  import_cap=np.full(n_cust, CAP_KW["import"]))
        count = {d: {"states": 0, "to_fallback": 0} for d in ev.DIRECTIONS}      # evaluator B, per direction
        nonconv = [0]

        def connections(p, q):                                        # envelope rows (+ = export) -> kVA per connection
            return -(p[:, idx] * share) - 1j * (q[:, idx] * share)

        def exact(s, direction):                                      # a non-converged sweep state is infeasible
            o = sweep.solve(s, return_state=False)
            nonconv[0] += int((~o["converged"]).sum())
            return decide(o["summary"], o["converged"])[:, ev.DIRECTIONS.index(direction)]

        def feasible_a(p, q, direction):
            return exact(connections(p, q), direction)

        def feasible_b(p, q, direction):
            s = connections(p, q)
            ok = np.zeros(len(s), bool)
            for k in range(0, len(s), 2048):
                part = s[k:k + 2048]
                if pipe is None:                                      # no model: every state goes to the sweep
                    c = np.full(len(part), ev.UNCERTAIN, np.uint8)
                else:
                    summ = pipe.predict(part)[0].cpu().numpy().astype(np.float64)
                    c = ev.screen_codes(m["margins"], summ)[:, ev.DIRECTIONS.index(direction)]
                fb = c == ev.UNCERTAIN
                ok[k:k + 2048] = c == ev.SAFE
                if fb.any():
                    ok[k:k + 2048][fb] = exact(part[fb], direction)
                count[direction]["states"] += len(c)
                count[direction]["to_fallback"] += int(fb.sum())
            return ok

        rows = []
        day_list = envelope_days(dry)[:days]
        for day in day_list:
            for step, p_kw, q_kvar in day_scenarios(network, n_cust, day, hours):
                for direction in ev.DIRECTIONS:
                    levels = {}
                    for name, feas in (("A", feasible_a), ("B", feasible_b)):
                        before = nonconv[0]
                        e = env.scalar_envelope(feas, n_cust, customers, -p_kw, -q_kvar, direction, "uniform", EPS,
                                                LEVELS, REL_TOL, beta=BETA)
                        levels[name] = {"level_kw": float(e.level), "share_feasible": float(e.prob_ok),
                                        "states": int(e.n_evals), "sweep_nonconverged": nonconv[0] - before}
                    rows.append({"day": day, "date": str(np.datetime64("2024-01-01") + day), "step": int(step),
                                 "direction": direction,
                                 **{f"{k}_{a}": v for k, d in levels.items() for a, v in d.items()}})
            print(day, len(rows), flush=True)
        unchanged = tt.checkout_unchanged(prov, profiles=final)
        for d in count.values():
            d["fallback_share"] = d["to_fallback"] / max(d["states"], 1)
        _write_record(folder, out, {
            "protocol": "docs/fourwire_scale_protocol.md §6, Amendment 9 item 7", "network": network, "dry_run": dry,
            **prov, "unchanged_during_run": unchanged, "valid": bool(unchanged), **sources,
            "earlier_attempts": earlier, "days": day_list, "hours": hours, "scenarios": N_SCENARIOS, "eps": EPS,
            "beta": BETA, "acceptance_share": env.acceptance_threshold(N_SCENARIOS, EPS, BETA),
            "levels_per_round": LEVELS, "relative_tolerance": REL_TOL, "cap_kw": CAP_KW, "mix_hp_ev_pv": list(MIX),
            "pgnet_has_model": m["model"], "evaluator_b": count,
            "summary": envelope_summary(rows), "intervals": rows}, dry)
        if not unchanged:
            raise RuntimeError(f"checkout, numerical mode, libraries or inputs changed during the run; "
                               f"{out.name} is not a valid record")
    return out


def envelope_summary(rows: list[dict]) -> dict:
    """§6: agreement within 1e-3 x cap, and unsafe envelopes (B above A by more than that)."""
    out = {}
    for scope in ("import", "export", "both"):
        r = [x for x in rows if scope == "both" or x["direction"] == scope]
        if not r:
            continue
        cap = np.array([CAP_KW[x["direction"]] for x in r])
        d = np.array([x["B_level_kw"] - x["A_level_kw"] for x in r])
        out[scope] = {"intervals": len(r), "share_within_tolerance": float((np.abs(d) <= 1e-3 * cap).mean()),
                      "unsafe_envelopes": int((d > 1e-3 * cap).sum()),
                      "largest_b_above_a_kw": float(d.max()), "largest_b_below_a_kw": float(-d.min()),
                      "zero_envelopes_a": int(sum(x["A_level_kw"] == 0 for x in r)),
                      "zero_envelopes_b": int(sum(x["B_level_kw"] == 0 for x in r))}
    if "both" in out:
        out["agreement_criterion_met"] = bool(out["both"]["share_within_tolerance"] >= 0.95)
    return out


# ------------------------------------------------------------------------------------------- report
def speedup(times: dict, baseline: tuple, pipeline: tuple, n: int = 2000, seed: int = 0) -> dict:
    """Ratio of the lower median of the baseline series to the lower median of the pipeline series.
    - ci95 (§7, the criterion): percentile interval of a bootstrap that resamples the repeats of every series
      independently and recomputes both minima.
    - ci95_paired_rounds (sensitivity, Amendment 9, item 6): the same with whole rounds resampled, which
      keeps what the series of one round share (the round-robin runs them one after another)."""
    t = {k: np.array(times[k]) for k in baseline + pipeline}
    med = {k: float(np.median(v)) for k, v in t.items()}

    def interval(index):
        boot = {k: np.median(v[index(len(v))], axis=1) for k, v in t.items()}
        ratio = np.min([boot[k] for k in baseline], 0) / np.min([boot[k] for k in pipeline], 0)
        return [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]

    rng = np.random.default_rng(seed)
    independent = interval(lambda m: rng.integers(0, m, (n, m)))
    rounds = np.random.default_rng(seed).integers(0, len(t[baseline[0]]), (n, len(t[baseline[0]])))
    return {"baseline": min(baseline, key=med.get), "pipeline": min(pipeline, key=med.get),
            "baseline_p50_s": min(med[k] for k in baseline), "pipeline_p50_s": min(med[k] for k in pipeline),
            "speedup": min(med[k] for k in baseline) / min(med[k] for k in pipeline),
            "ci95": independent, "ci95_paired_rounds": interval(lambda m: rounds)}


def report() -> Path:
    """E2 from the final records of the six networks. Every record is checked against the freeze and against
    the copy in its run folder, and every per-state file against the hash its record names."""
    from . import fourwire_scale_eval as ev
    from . import fourwire_scale_train as tt
    if tt.halt_file().exists():
        raise RuntimeError(f"the final runs are halted: {tt.halt_file().read_text()}")
    nets = []
    for n in E2_NETWORKS:
        tim, rec, envl = e2_record("timing", n), ev.evaluation_record(n), e2_record("envelopes", n)
        f = ev.eval_dir() / f"network_{n}.json"
        if not tim["evaluation_record_sha256"] == envl["evaluation_record_sha256"] == fs._sha(f):
            raise AssertionError(f"network {n}: timed or enveloped with the margins of another evaluation record")
        z, e = np.load(fs.OUT / "e2" / f"timing_network_{n}" / tim["decisions_file"]), ev.per_state(n, rec)
        screens = {k: {a: rec["methods"][m]["screen"][a] for a in ("fallback_share", "false_safe",
                                                                  "cleared_indeterminate", "false_unsafe")}
                   for k, m in (("pgnet", "PG-net"), ("u1", "U1"))}
        row = {"network": n, "retained_buses": tim["retained_buses"], "allocated_cores": tim["allocated_cores"],
               "pgnet_screen": screens["pgnet"], "u1_screen": screens["u1"],
               "pgnet_has_model": "no_model" not in rec["methods"]["PG-net"],
               "exact_sweep_vs_opendss": rec["exact_sweep_vs_opendss"],
               "pgnet_fallback_exact_engine": rec["methods"]["PG-net"]["fallback_exact_engine"], "batches": {}}
        for b, series in tim["batches"].items():
            t = {k: v["times_s"] for k, v in series.items()}
            sw, sw0 = ("sweep_cpu", "sweep_gpu"), ("sweep_cpu_noopendss", "sweep_gpu_noopendss")
            sp = {"pgnet": speedup(t, sw, ("pgnet_cpu", "pgnet_gpu")), "u1": speedup(t, sw, ("u1_cpu", "u1_gpu")),
                  "pgnet_without_opendss_solves": speedup(t, sw0, ("pgnet_cpu_noopendss", "pgnet_gpu_noopendss")),
                  "u1_without_opendss_solves": speedup(t, sw0, ("u1_cpu_noopendss", "u1_gpu_noopendss"))}
            row["batches"][b] = {"p50_s": {k: v["p50_s"] for k, v in series.items()},
                                 "p95_s": {k: v["p95_s"] for k, v in series.items()},
                                 "states_per_s": {k: v["states_per_s"] for k, v in series.items()},
                                 "opendss_solves_first_timed_repeat": {
                                     k: v.get("opendss_solves_first_timed_repeat") for k, v in series.items()},
                                 "speedup": sp}
        truth, conv = e["truth_safe"], e["converged"][:, None]

        def delivered(k):
            """What a timed series delivered at the primary batch, against OpenDSS, over every timed repeat:
            a state counts as cleared if any repeat cleared it, and as rejected if any repeat rejected it
            (Amendment 9, item 6). Every decision counts, from the screen or from an exact solve."""
            ever, always = z[f"{k}__ever_safe"], z[f"{k}__always_safe"]
            fs_, ci, fu = ever & ~truth & conv, ever & ~conv, ~always & truth
            return {"false_safe": int(fs_.sum()), "cleared_indeterminate": int(ci.sum()), "false_unsafe": int(fu.sum()),
                    "by_direction": {d: {"false_safe": int(fs_[:, j].sum()), "cleared_indeterminate": int(ci[:, j].sum()),
                                         "false_unsafe": int(fu[:, j].sum())} for j, d in enumerate(ev.DIRECTIONS)}}

        row["delivered_decisions_vs_opendss"] = {k[:-11]: delivered(k[:-11]) for k in z.files if k.endswith("__ever_safe")}

        def verdict(sp, screen):
            """§7 with Amendment 9, item 6: the speed-up, its interval, the screen's errors (evaluation record)
            and the errors in what the faster timed variant delivered."""
            d = row["delivered_decisions_vs_opendss"][sp["pipeline"]]
            screen_ok = screen["false_safe"] == 0 and screen["cleared_indeterminate"] == 0
            delivered_ok = d["false_safe"] == 0 and d["cleared_indeterminate"] == 0
            return {"speedup_ge_1.10": bool(sp["speedup"] >= 1.10),
                    "ci_excludes_1": bool(sp["ci95"][0] > 1.0 or sp["ci95"][1] < 1.0),
                    "screen_no_false_safe_no_cleared_indeterminate": bool(screen_ok),
                    "delivered_no_false_safe_no_cleared_indeterminate": bool(delivered_ok),
                    "faster": bool(sp["speedup"] >= 1.10 and sp["ci95"][0] > 1.0 and screen_ok and delivered_ok)}

        last = row["batches"][str(BATCHES[-1])]["speedup"]
        row["criterion"] = verdict(last["pgnet"], screens["pgnet"])                       # the criterion of §7
        row["sensitivity_without_opendss_solves"] = verdict(last["pgnet_without_opendss_solves"], screens["pgnet"])
        row["u1_screen_same_rule"] = {"with_opendss_solves": verdict(last["u1"], screens["u1"]),
                                      "without_opendss_solves": verdict(last["u1_without_opendss_solves"], screens["u1"])}
        if row["pgnet_has_model"]:
            fb = (e["PG-net__codes"] == ev.UNCERTAIN).any(1)                 # one solve serves both directions
            want = np.where(fb[:, None], e["exact_safe"], e["PG-net__codes"] == ev.SAFE)
            row["first_timed_repeat_vs_evaluation_record"] = {
                k: int((z[f"{k}__safe"] != want).sum()) for k in ("pgnet_cpu", "pgnet_gpu")}
        row["timed_repeats_with_other_decisions"] = tim["repeats_with_other_decisions"]
        row["envelopes"] = {**envl["summary"], "evaluator_b": envl["evaluator_b"]}
        nets.append(row)
    big = str(BATCHES[-1])
    x = np.log([r["retained_buses"] for r in nets])
    exponents = {k: float(np.polyfit(x, np.log([r["batches"][big]["p50_s"][k] for r in nets]), 1)[0])
                 for k in nets[0]["batches"][big]["p50_s"]}
    faster = [r["criterion"]["faster"] for r in nets]                      # networks are in order of size
    first = next((k for k in range(len(nets)) if all(faster[k:])), None)
    out = fs.RESULTS / "fourwire_scale_e2.json"
    fs._write_json(out, {
        "protocol": "docs/fourwire_scale_protocol.md §7 E2, Amendment 9", "primary_batch": BATCHES[-1],
        "break_even_retained_buses": None if first is None else nets[first]["retained_buses"],
        "networks_where_genco_is_faster": [r["network"] for r in nets if r["criterion"]["faster"]],
        "networks_where_it_is_faster_without_the_opendss_solves":
            [r["network"] for r in nets if r["sensitivity_without_opendss_solves"]["faster"]],
        "networks_where_the_u1_screen_is_faster_by_the_same_rule":
            [r["network"] for r in nets if r["u1_screen_same_rule"]["with_opendss_solves"]["faster"]],
        "loglog_exponent_of_p50_time_against_retained_buses": exponents, "networks": nets})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timing", "envelopes", "report"])
    ap.add_argument("--network", type=int, choices=E2_NETWORKS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry", action="store_true", help="val states or a val day, dry-run models and margins")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="dry timing only")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="dry timing only")
    ap.add_argument("--days", type=int, help="dry envelopes only: number of days")
    ap.add_argument("--hours", type=int, default=24, help="dry envelopes only: hours per day")
    a = ap.parse_args()
    if a.cmd == "report":
        print(report())
        return
    if a.network is None:
        ap.error("--network is required")
    if not a.dry and ((a.warmup, a.repeats) != (WARMUP, REPEATS) or a.days is not None or a.hours != 24):
        ap.error("the final runs take no reductions")
    from . import fourwire_scale_train as tt
    try:
        if a.cmd == "timing":
            print(timing(a.network, a.device, a.dry, a.warmup, a.repeats,
                         opendss_large=(0, 1) if a.dry else OPENDSS_LARGE))
        else:
            print(envelopes(a.network, a.device, a.dry, a.days, a.hours))
    except tt.AlreadyDone as e:                                      # the job script reads 10 as "skip"
        print(e)
        raise SystemExit(10)


if __name__ == "__main__":
    main()
