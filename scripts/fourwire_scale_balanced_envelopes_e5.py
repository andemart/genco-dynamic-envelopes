"""Family E5 of the four-wire scale study (docs/fourwire_scale_protocol.md, Amendment 12): the experiment of
E4 (Amendment 11) with a third design model, the positive-sequence reduction, and under two further
technology mixes. No learned model; no label is read: every state is regenerated from the seeds and the meter
profiles.

    python scripts/fourwire_scale_balanced_envelopes_e5.py run --network 17 --mix half
    python scripts/fourwire_scale_balanced_envelopes_e5.py run --network 17 --mix pv --dry --days 2 --hours 6
    python scripts/fourwire_scale_balanced_envelopes_e5.py report

run: for one network and one technology mix, every hour of the 36 held-out days and both directions, as in E4:
- four-wire-verified envelope: the ENWL1 rule (envelopes.scalar_envelope) judged by the Numba four-wire sweep
  with every customer on its own phase connection;
- balanced-load envelope (the balanced model of E4): the same rule and sweep on the same network and source,
  with every customer's power spread equally over the three phases at its bus;
- positive-sequence envelope: the same rule and sweep on the positive-sequence reduction of the network (every
  branch reduced to its positive-sequence impedance, the positive-sequence equivalent of the frozen source,
  every customer's power spread equally over the three phases). With these inputs the three phases are
  decoupled and equal in magnitude, so the sweep solves the positive-sequence power flow; a run checks that
  against an independent scalar power flow in NumPy on every day and stops, without a record, if it fails;
- four-wire check on fresh draws (stream 2), for every positive envelope of the three models, and on three of
  the days also by OpenDSS: as in E4.
One record per network and mix, results/fourwire_scale_e5/<mix>/network_N.json, published once and never
replaced; every start of a final run leaves an attempt file in results/fourwire_scale_e5/<mix>/attempts.

report: per mix and design model (balanced, posseq), the summary, classification and criterion of
Amendment 12, computed by the functions of the E4 script, and the comparison of the base mix with the
committed E4 records (levels, results and classification), which must pass.

--dry uses validation days only and writes under data_out/fourwire_scale/e5_dry; it is not a final run.
No command takes an output path.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import time
from pathlib import Path

import numba
import numpy as np

from gencodoe import csiro_convert as cc
from gencodoe import datakit_fluvius as dk
from gencodoe import envelopes as env
from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_e2 as e2
from gencodoe import fourwire_scale_eval as ev
from gencodoe import fourwire_scale_train as tt
from gencodoe.fourwire_net import Thevenin
from gencodoe.fourwire_sweep_opt import NumbaSweep

SCRIPT = Path(__file__).resolve()
E4_SCRIPT = SCRIPT.with_name("fourwire_scale_balanced_envelopes.py")
_spec = importlib.util.spec_from_file_location("fourwire_scale_balanced_envelopes", E4_SCRIPT)
e4 = importlib.util.module_from_spec(_spec)     # the audited E4 script, used as a library and left unchanged
_spec.loader.exec_module(e4)

E4_SCRIPT_SHA256 = "e817d926c4e6938500af391148a99fd71eb881892cb8ed24ad5cd734914fed25"   # as in the E4 records
RECORDS = fs.RESULTS / "fourwire_scale_e5"
REPORT = fs.RESULTS / "fourwire_scale_e5.json"
DRY = fs.OUT / "e5_dry"
MIXES = {"base": (0.3, 0.3, 0.4),       # heat pump, EV, PV: the mix of E2 and E4
         "half": (0.15, 0.15, 0.2),     # every share halved
         "pv": (0.3, 0.3, 0.8)}         # the PV share doubled
MODELS = ("fourwire", "balanced", "posseq")
DESIGN_MODELS = ("balanced", "posseq")  # the models whose envelopes are classified against the four-wire check
HOURS, CORES, N_FRESH = e4.HOURS, e4.CORES, e4.N_FRESH
STREAM_DESIGN, STREAM_FRESH = e4.STREAM_DESIGN, e4.STREAM_FRESH
MAJORITY = e4.MAJORITY
OPENDSS_DAYS = e4.OPENDSS_DAYS
E4_TOLERANCE_SHARE = 0.005              # report: share of base-mix levels, and of envelopes with other results,
#                                         that may differ from the E4 records
POSSEQ_TOLERANCE_PU = 1e-6              # the sweep on the reduced network against the scalar power flow
POSSEQ_CHECK_STATES = 8                 # design scenarios of the first hour of a day, each also at POSSEQ_LIGHT
POSSEQ_LIGHT = 0.1                      # of its power, so that a day has states on which both solvers converge
SAME_AS_REPORT = (*e4.SAME_AS_REPORT, "e4_script_sha256")
A = np.exp(2j * np.pi / 3)
POSITIVE = np.array([1, A ** 2, A])     # a positive-sequence set of phasors, phases a, b, c
if tuple(e2.MIX) != MIXES["base"]:
    raise AssertionError("the base mix is not the mix of E2 and E4")


def settings(mix: str) -> dict:
    return {**e4.SETTINGS, "mix_hp_ev_pv": list(MIXES[mix]), "mix": mix, "models": list(MODELS)}


# ------------------------------------------------------------------------------------------- provenance
def script_committed() -> bool:
    rel = str(SCRIPT.relative_to(fs.ROOT))
    return (e4._git("ls-files", "--error-unmatch", rel).returncode == 0
            and not e4._git("status", "--porcelain", "--", rel).stdout.strip())


def start(final: bool) -> dict:
    """Provenance of a run: that of E4 (for a final run: commit, frozen source, inputs, libraries, limits and
    the E4 script as committed), and in addition this script as committed and the E4 script being the one
    that wrote the E4 records."""
    prov = e4.start(final)
    prov["e4_script_sha256"] = prov["analysis_script_sha256"]
    prov["analysis_script_sha256"] = fs._sha(SCRIPT)
    if final:
        problems = []
        if prov["e4_script_sha256"] != E4_SCRIPT_SHA256:
            problems.append("the E4 script is not the one of the E4 records")
        if not script_committed():
            problems.append("this script differs from HEAD")
        if problems:
            raise RuntimeError("not a final run: " + "; ".join(problems))
    return prov


def unchanged(prov: dict, final: bool) -> bool:
    return bool(tt.checkout_unchanged(prov, profiles=final) and fs._sha(SCRIPT) == prov["analysis_script_sha256"]
                and fs._sha(E4_SCRIPT) == prov["e4_script_sha256"])


# ------------------------------------------------------------------------------------------- scenarios
def scenarios(network: int, n_cust: int, day: int, hours: int, stream: int, mix: tuple):
    """The generator of the E4 script (and of fourwire_scale_e2.day_scenarios) with the technology mix as an
    argument: (step, p_kw, q_kvar), each (200, n_cust) customer net consumption. The uniform numbers behind
    the technology flags are those of the network and day (stream 0) for every mix, so a household with a
    technology under a lower share has it under a higher one."""
    from gencodoe.fluvius import GROUPS, load_group
    if not 1 <= hours <= HOURS:
        raise ValueError("hours must be 1 to 24: a later hour would be a step of the next day")
    groups = [load_group(g) for g in GROUPS]
    by_flags = {(g.hp, g.ev, g.pv): g for g in groups}
    flags = np.random.default_rng([network, day, 0]).uniform(size=(n_cust, 3)) < np.array(mix)
    keys = sorted({tuple(int(v) for v in f) for f in flags})
    who = {key: np.flatnonzero((flags == key).all(1)) for key in keys}
    rng = np.random.default_rng([network, day, stream])
    for hour in range(hours):
        step = day * 96 + 4 * hour
        p, q = np.empty((N_FRESH, n_cust)), np.empty((N_FRESH, n_cust))
        for sc in range(N_FRESH):
            for key in keys:
                g = by_flags[key]
                pick = dk._pick_meters(rng, len(g.ean), len(who[key]))
                off, inj = g.offtake_kw[pick, step], g.injection_kw[pick, step]
                p[sc, who[key]], q[sc, who[key]] = off - inj, off * dk.TAN_PHI
        yield step, p, q


# ------------------------------------------------------------------------------------------- positive sequence
def positive_sequence(net, th: Thevenin) -> tuple[np.ndarray, complex, complex]:
    """Positive-sequence data of a network: z1 (n_bus,) ohm of the branch into each bus, by the reduction of
    the ENWL1 study (gencodoe.enwl.read_linecodes: Kron-reduce the neutral out of the 4x4 matrix, then mean
    self minus mean mutual of the 3x3 phase matrix), here on the 4x4 matrix of each retained branch; e1 (kV)
    and z1s (ohm), the positive-sequence voltage and impedance of the frozen source."""
    z1 = np.zeros(net.n_bus, complex)
    for i in range(1, net.n_bus):
        z = net.z[i]
        zp = z[:3, :3] - np.outer(z[:3, 3], z[3, :3]) / z[3, 3]
        z1[i] = np.mean(np.diag(zp)) - np.mean(zp[~np.eye(3, dtype=bool)])
    e1 = complex(POSITIVE.conj() @ np.asarray(th.e) / 3)
    z1s = complex(POSITIVE.conj() @ np.asarray(th.zth) @ POSITIVE / 3)
    return z1, e1, z1s


def reduced_case(net, th: Thevenin, names: list[str], bus: np.ndarray):
    """The positive-sequence reduction as a network the four-wire sweep can solve: every branch with z1 on each
    phase and no coupling (the neutral keeps its own impedance and carries nothing), a balanced source with
    z1s on each phase, three connections per customer."""
    z1, e1, z1s = positive_sequence(net, th)
    z = np.zeros_like(net.z)
    for i in range(1, net.n_bus):
        z[i] = np.diag([z1[i], z1[i], z1[i], net.z[i][3, 3]])
    reduced = dataclasses.replace(
        net, z=z, customer_names=[f"{n}_{k}" for n in names for k in "abc"], customer_bus=np.repeat(bus, 3),
        customer_phase=np.tile(np.arange(3), len(names)))
    return reduced, Thevenin(e=e1 * POSITIVE, zth=z1s * np.eye(3)), (z1, e1, z1s)


def scalar_power_flow(net, z1: np.ndarray, e1: complex, z1s: complex, bus: np.ndarray, s_kva: np.ndarray,
                      iterations: int = 500, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Single-phase power flow of the positive-sequence network, written independently of the sweep: customer
    consumption s_kva (B, n_cust) in kVA (three-phase total). Returns the phase-to-neutral voltage (B, n_bus)
    in kV and, per state, whether its last update was below tol (kV)."""
    b = len(s_kva)
    s_bus = np.zeros((b, net.n_bus), complex)
    np.add.at(s_bus, (slice(None), bus), s_kva / 3)
    v = np.full((b, net.n_bus), e1, complex)
    step = np.full(b, np.inf)
    for _ in range(iterations):
        cur = np.conj(s_bus / 1e3 / v)                       # kA per phase drawn at each bus
        for i in net.order[::-1][:-1]:
            cur[:, net.parent[i]] += cur[:, i]
        new = np.empty_like(v)
        new[:, 0] = e1 - z1s * cur[:, 0]
        for i in net.order[1:]:
            new[:, i] = new[:, net.parent[i]] - z1[i] * cur[:, i]
        step = np.abs(new - v).max(1)
        v = new
        if (step < tol).all():
            break
    with np.errstate(invalid="ignore"):
        return v, step < tol


def check_reduction(net, sweep: NumbaSweep, parts: tuple, bus: np.ndarray, s_cust: np.ndarray) -> dict:
    """The sweep on the reduced network against the scalar power flow. States: s_cust (B, n_cust) kVA consumed
    per customer, and the same states at POSSEQ_LIGHT of their power. On every state on which both solvers
    converge, the three phase-to-neutral voltage magnitudes at every bus must be equal to each other and to
    the scalar solution within POSSEQ_TOLERANCE_PU. A day without such a state, or out of tolerance, stops the
    run before any record is published."""
    from gencodoe.fourwire_net import V_PN_NOM
    states = np.concatenate([s_cust, POSSEQ_LIGHT * s_cust])
    o = sweep.solve(np.repeat(states, 3, axis=1) / 3, return_state=True)
    v_scalar, scalar_ok = scalar_power_flow(net, *parts, bus, states)
    ok = o["converged"] & scalar_ok
    if not ok.any():
        raise RuntimeError("positive-sequence check: no state on which the sweep and the scalar power flow converge")
    vpn = np.abs(o["V"][ok][:, :, :3] - o["V"][ok][:, :, 3:4])
    out = {"states": int(ok.sum()),
           "phases_differ_pu": float((vpn.max(2) - vpn.min(2)).max() / V_PN_NOM),
           "against_scalar_pu": float(np.abs(vpn[:, :, 0] - np.abs(v_scalar[ok])).max() / V_PN_NOM)}
    if not max(out["phases_differ_pu"], out["against_scalar_pu"]) <= POSSEQ_TOLERANCE_PU:
        raise RuntimeError(f"the sweep on the reduced network is not the positive-sequence power flow: {out}")
    return out


# ------------------------------------------------------------------------------------------- one network
def run(network: int, mix: str, dry: bool, days: int | None = None, hours: int = HOURS, opendss: bool = True,
        processes: int | None = None) -> Path:
    final = not dry
    if final and (days is not None or hours != HOURS or not opendss or processes is not None):
        raise SystemExit("a final run takes no --days, --hours, --skip-opendss or --processes")
    if not 1 <= hours <= HOURS or (days is not None and days < 1):
        raise SystemExit("--hours must be 1 to 24 and --days at least 1")
    if final and e2.allocated_cores() != CORES:
        raise SystemExit(f"a final run needs {CORES} cores; this job has {e2.allocated_cores()}")
    folder = RECORDS / mix
    out = (folder / f"network_{network}.json" if final
           else DRY / f"{mix}_network_{network}_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json")
    if out.exists():
        raise FileExistsError(f"{out} exists")
    prov = start(final)
    attempt, earlier = e4.claim_attempt(network, prov, folder) if final else (None, [])
    t0 = time.time()
    split = "val" if dry else "test"
    shares = MIXES[mix]
    net, th = fs._case(network)
    names, idx, share = cc.customers(net.customer_names)
    n_cust = len(names)
    first = np.array([np.flatnonzero(idx == j)[0] for j in range(n_cust)])
    bus = np.asarray(net.customer_bus)[first]
    if not (np.asarray(net.customer_bus) == bus[idx]).all():
        raise AssertionError("the phase connections of a customer are not at one bus")
    spread = dataclasses.replace(
        net, customer_names=[f"{n}_{k}" for n in names for k in "abc"], customer_bus=np.repeat(bus, 3),
        customer_phase=np.tile(np.arange(3), n_cust))
    reduced, reduced_source, parts = reduced_case(net, th, names, bus)
    sweeps = {"fourwire": NumbaSweep(net, th), "balanced": NumbaSweep(spread, th),
              "posseq": NumbaSweep(reduced, reduced_source)}
    customers = env.Customers(bus=np.arange(n_cust), export_cap=np.full(n_cust, e2.CAP_KW["export"]),
                              import_cap=np.full(n_cust, e2.CAP_KW["import"]))
    nonconv = {m: 0 for m in MODELS}

    def as_connected(p, q):       # envelope rows (+ = export), kW per customer -> kVA consumed per phase connection
        return -(p[:, idx] * share) - 1j * (q[:, idx] * share)

    def as_balanced(p, q):        # the same customer power, one third on each phase at the customer's bus
        return np.repeat(-p - 1j * q, 3, axis=1) / 3

    to_states = {"fourwire": as_connected, "balanced": as_balanced, "posseq": as_balanced}

    def judged(model: str, s, direction: str) -> np.ndarray:
        o = sweeps[model].solve(s, return_state=False)
        nonconv[model] += int((~o["converged"]).sum())
        return e2.decide(o["summary"], o["converged"])[:, ev.DIRECTIONS.index(direction)]

    feasible = {m: (lambda p, q, d, m=m: judged(m, to_states[m](p, q), d)) for m in MODELS}

    all_days = e4.day_list(dry)[:days]
    dss_days = [all_days[k] for k in OPENDSS_DAYS if k < len(all_days)] if opendss else []
    processes = CORES if final else (processes or e2.allocated_cores())
    pool = e2.OpenDSSPool(network, processes) if dss_days else None
    rows, reduction_checks = [], []
    try:
        for day in all_days:
            design = scenarios(network, n_cust, day, hours, STREAM_DESIGN, shares)
            fresh = scenarios(network, n_cust, day, hours, STREAM_FRESH, shares)
            reference = e2.day_scenarios(network, n_cust, day, hours) if mix == "base" else None
            for hour, ((step, p_kw, q_kvar), (step_f, pf, qf)) in enumerate(zip(design, fresh)):
                if not (step == step_f and step // 96 == day and dk.day_split(step) == split):
                    raise AssertionError(f"step {step} is not a step of {split} day {day} in every stream")
                if reference is not None:      # with the base mix this script's generator is the one of E2 and E4
                    step_r, pr, qr = next(reference)
                    if not (step_r == step and np.array_equal(pr, p_kw) and np.array_equal(qr, q_kvar)):
                        raise AssertionError("scenarios() with the base mix differs from e2.day_scenarios")
                if hour == 0:                  # the reduced network solved by the sweep is the positive sequence
                    reduction_checks.append(check_reduction(
                        net, sweeps["posseq"], parts, bus, (p_kw + 1j * q_kvar)[:POSSEQ_CHECK_STATES]))
                for direction in ev.DIRECTIONS:
                    k = ev.DIRECTIONS.index(direction)
                    row = {"day": day, "step": int(step), "direction": direction}
                    checked = {}               # level -> results: equal levels are the same fresh states
                    for model in MODELS:
                        before = nonconv[model]
                        e = env.scalar_envelope(feasible[model], n_cust, customers, -p_kw, -q_kvar, direction,
                                                "uniform", e2.EPS, e2.LEVELS, e2.REL_TOL, beta=e2.BETA)
                        level = float(e.level)
                        row.update({f"{model}_level_kw": level, f"{model}_design_share": float(e.prob_ok),
                                    f"{model}_states": int(e.n_evals),
                                    f"{model}_sweep_nonconverged": nonconv[model] - before})
                        if level > 0 and level not in checked:   # fresh draws with the envelope taken up in
                            s = as_connected(-pf + env.DIRECTIONS[direction] * level, -qf)   # full, four-wire
                            o = sweeps["fourwire"].solve(s, return_state=False)
                            ok = e2.decide(o["summary"], o["converged"])[:, k]
                            res = [int(ok.sum()), None, None]
                            if day in dss_days:
                                quant, conv = pool.solve(s)
                                ok_dss = e2.decide(quant, conv)[:, k]
                                res[1:] = [int(ok_dss.sum()), int((ok != ok_dss).sum())]
                            checked[level] = res
                        res = checked[level] if level > 0 else [None, None, None]
                        row.update({f"{model}_fresh_feasible": res[0], f"{model}_fresh_feasible_opendss": res[1],
                                    f"{model}_fresh_decisions_differ": res[2]})
                    rows.append(row)
            print(day, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
    same = unchanged(prov, final)
    e4.publish(out, {
        "protocol": "docs/fourwire_scale_protocol.md Amendment 12 (E5)", "network": network, "dry_run": dry,
        **prov, "unchanged_during_run": same, "valid": same, "attempt": attempt, "earlier_attempts": earlier,
        "customers": n_cust, "phase_connections": int(len(idx)), "retained_buses": int(net.n_bus),
        "days": all_days, "opendss_days": dss_days, "numba_threads": int(numba.get_num_threads()),
        **settings(mix), "hours": hours, "opendss_processes": processes if dss_days else None,
        "positive_sequence_check": {
            "states_per_day": [c["states"] for c in reduction_checks],
            "phases_differ_pu": max(c["phases_differ_pu"] for c in reduction_checks),
            "against_scalar_pu": max(c["against_scalar_pu"] for c in reduction_checks)},
        "seconds": time.time() - t0, "intervals": {key: [r[key] for r in rows] for key in rows[0]}})
    if not same:
        raise RuntimeError(f"checkout, libraries, inputs or a script changed during the run; "
                           f"{out.name} is not a valid record")
    return out


# ------------------------------------------------------------------------------------------- reading records
def rows_of(rec: dict) -> list[dict]:
    """The intervals of a record (stored by column) as one dict per interval and direction."""
    cols = rec["intervals"]
    keys = list(cols)
    if len({len(cols[k]) for k in keys}) != 1:
        raise ValueError("columns of different length")
    return [dict(zip(keys, vals)) for vals in zip(*(cols[k] for k in keys))]


def e4_rows(rows: list[dict], model: str) -> list[dict]:
    """Rows in the layout of the E4 script with `model` in the place of its balanced model, so that the
    summary, the bootstrap over days, the comparison of the engines and the classification are those of the
    E4 code (e4.summarise, e4.classify)."""
    out = []
    for r in rows:
        x = {"day": r["day"], "step": r["step"], "direction": r["direction"]}
        for src, dst in (("fourwire", "fourwire"), (model, "balanced")):
            x[f"{dst}_level_kw"] = r[f"{src}_level_kw"]
            for count, share in (("fresh_feasible", "fresh_share"), ("fresh_feasible_opendss", "fresh_share_opendss")):
                x[f"{dst}_{share}"] = None if r[f"{src}_{count}"] is None else r[f"{src}_{count}"] / N_FRESH
            x[f"{dst}_fresh_decisions_differ"] = r[f"{src}_fresh_decisions_differ"]
        out.append(x)
    return out


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def record_problems(rec: dict, network: int, mix: str, prov: dict, days: list[int], folder: Path) -> list[str]:
    """What keeps a record out of the report: not a valid final record of this network, mix, commit, scripts,
    freeze and settings; an attempt that is not in the ledger; a positive-sequence check without a whole number
    of 1 to 16 states on every day or out of tolerance; or not the complete set of intervals with every result a positive
    envelope must have (counts over the 200 fresh draws, a number of differing decisions that the two counts
    allow, equal results for equal levels of one interval)."""
    dss = [days[k] for k in OPENDSS_DAYS]
    chk = rec.get("positive_sequence_check")
    chk = chk if isinstance(chk, dict) else {}
    per_day = chk.get("states_per_day")
    bad = [what for what, wrong in [
        ("not valid", rec.get("valid") is not True or rec.get("unchanged_during_run") is not True),
        ("a dry run", rec.get("dry_run") is not False), ("another network", rec.get("network") != network),
        ("other days", rec.get("days") != days), ("other OpenDSS days", rec.get("opendss_days") != dss),
        ("no positive-sequence check on every day", not (
            isinstance(per_day, list) and len(per_day) == len(days)
            and all(_whole(x) and 1 <= x <= 2 * POSSEQ_CHECK_STATES for x in per_day))),
        ("positive-sequence check out of tolerance", not all(
            e4._number(chk.get(k)) and 0 <= chk[k] <= POSSEQ_TOLERANCE_PU
            for k in ("phases_differ_pu", "against_scalar_pu"))),
    ] if wrong]
    bad += e4.ledger_problems(rec, network, prov, folder)
    bad += [f"another {k}" for k in SAME_AS_REPORT if rec.get(k) != prov[k]]
    bad += [f"another {k}" for k, v in settings(mix).items() if rec.get(k) != v]
    try:
        rows = rows_of(rec)
        if [(r["day"], r["step"], r["direction"]) for r in rows] != e4.expected_intervals(days):
            bad.append("not the expected intervals")
        for r in rows:
            seen = {}
            for m in MODELS:
                level = r[f"{m}_level_kw"]
                res = [r[f"{m}_fresh_feasible"], r[f"{m}_fresh_feasible_opendss"], r[f"{m}_fresh_decisions_differ"]]
                need = [level > 0, level > 0 and r["day"] in dss, level > 0 and r["day"] in dss]
                if (not e4._number(level) or not 0 <= level <= e2.CAP_KW[r["direction"]]
                        or [x is not None for x in res] != need
                        or any(n and not (_whole(x) and 0 <= x <= N_FRESH) for n, x in zip(need, res))):
                    raise ValueError
                if need[2]:       # d states decided differently: x feasible for the sweep only, y for OpenDSS only
                    a, b, d = res
                    if (d + a - b) % 2:
                        raise ValueError
                    x, y = (d + a - b) // 2, (d - a + b) // 2
                    if not (0 <= x <= min(a, N_FRESH - b) and 0 <= y <= min(b, N_FRESH - a)):
                        raise ValueError
                if seen.setdefault(level, res) != res:
                    raise ValueError
    except (KeyError, TypeError, ValueError):
        bad.append("an interval without the results it must have")
    return bad


# ------------------------------------------------------------------------------------------- report
def e4_comparison(records: dict, recomputed: dict) -> dict:
    """The base mix against the committed E4 records, which must be the files the E4 report names. Compared: the
    levels of the four-wire and the balanced-load model; for every envelope with an equal positive level, its
    three results (feasible fresh draws by the sweep, by OpenDSS, decisions that differ); and the classification
    of every network in the recomputed E4 cell against the E4 report. Every interval and model with any
    difference is listed. Passed: at most E4_TOLERANCE_SHARE of the levels beyond tolerance, at most that share
    of the compared envelopes with another result, and the same classification of all 24 networks."""
    e4_report = json.loads(e4.REPORT.read_text())
    out = {"levels_compared": 0, "levels_equal": 0, "levels_beyond_tolerance": 0, "largest_difference_kw": 0.0,
           "envelopes_compared": 0, "envelopes_with_equal_results": 0, "e4_report_sha256": fs._sha(e4.REPORT),
           "differences": []}
    for n in fs.ELIGIBLE:
        path = e4.RECORDS / f"network_{n}.json"
        if fs._sha(path) != e4_report["per_network"][str(n)]["record_sha256"]:
            raise RuntimeError(f"{path.name} is not the E4 record of the E4 report")
        old = {(r["day"], r["step"], r["direction"]): r for r in json.loads(path.read_text())["intervals"]}
        for r in rows_of(records[n]):
            o = old[(r["day"], r["step"], r["direction"])]
            for m in ("fourwire", "balanced"):
                d = abs(r[f"{m}_level_kw"] - o[f"{m}_level_kw"])
                out["levels_compared"] += 1
                out["levels_equal"] += int(d == 0)
                out["levels_beyond_tolerance"] += int(d > e2.REL_TOL * e2.CAP_KW[r["direction"]])
                out["largest_difference_kw"] = max(out["largest_difference_kw"], float(d))
                new = [r[f"{m}_fresh_feasible"], r[f"{m}_fresh_feasible_opendss"], r[f"{m}_fresh_decisions_differ"]]
                was = [None if o[f"{m}_{k}"] is None else round(o[f"{m}_{k}"] * N_FRESH)
                       for k in ("fresh_share", "fresh_share_opendss")] + [o[f"{m}_fresh_decisions_differ"]]
                same = d == 0 and new == was
                if d == 0 and r[f"{m}_level_kw"] > 0:
                    out["envelopes_compared"] += 1
                    out["envelopes_with_equal_results"] += int(same)
                if not same:
                    out["differences"].append({
                        "network": n, "day": r["day"], "step": r["step"], "direction": r["direction"], "model": m,
                        "e4_level_kw": o[f"{m}_level_kw"], "e5_level_kw": r[f"{m}_level_kw"],
                        "e4_results": was, "e5_results": new})
    keys = ("balanced_unsafe", "engines_classify_alike", "counted_unsafe")
    out["networks_classified_differently"] = [
        n for n in fs.ELIGIBLE
        if any(recomputed["per_network"][str(n)][k] != e4_report["per_network"][str(n)][k] for k in keys)]
    out["passed"] = bool(
        out["levels_beyond_tolerance"] <= E4_TOLERANCE_SHARE * out["levels_compared"]
        and out["envelopes_compared"] - out["envelopes_with_equal_results"]
        <= E4_TOLERANCE_SHARE * out["envelopes_compared"]
        and not out["networks_classified_differently"]
        and recomputed["unsafe_counted"] == e4_report["balanced_unsafe_counted"])
    return out


def cell(records: dict, model: str) -> dict:
    """One mix and one design model over the 24 networks: per network the E4 summary and classification with
    `model` as the balanced model, and the count over the 23 fresh networks."""
    per = {}
    for n, rec in records.items():
        rows = e4_rows(rows_of(rec), model)
        per[str(n)] = {"customers": rec["customers"], "retained_buses": rec["retained_buses"],
                       "fresh": n in ev.FRESH, "summary": e4.summarise(rows), **e4.classify(rows)}
    unsafe = [n for n in ev.FRESH if per[str(n)]["counted_unsafe"]]
    return {
        "unsafe_by_sweep": [n for n in ev.FRESH if per[str(n)]["balanced_unsafe"]],
        "engines_classify_differently": [n for n in fs.ELIGIBLE if per[str(n)]["engines_classify_alike"] is False],
        "engines_not_compared": [n for n in fs.ELIGIBLE if per[str(n)]["engines_classify_alike"] is None],
        "without_positive_envelope": [
            n for n in fs.ELIGIBLE if per[str(n)]["summary"]["both"]["balanced"]["positive"] == 0],
        "unsafe_counted": unsafe, "count": len(unsafe), "of": len(ev.FRESH),
        "criterion_met": bool(len(unsafe) >= MAJORITY), "network_1": per["1"]["counted_unsafe"],
        "per_network": per}


def report(folder: Path = RECORDS, out: Path = REPORT) -> Path:
    if out.exists():
        raise FileExistsError(f"{out} exists")
    prov = start(final=True)
    days, records = e4.day_list(False), {}
    for mix in MIXES:
        records[mix] = {}
        for n in fs.ELIGIBLE:
            rec = json.loads((folder / mix / f"network_{n}.json").read_text())
            problems = record_problems(rec, n, mix, prov, days, folder / mix)
            if problems:
                raise RuntimeError(f"record of network {n}, mix {mix}: " + ", ".join(problems))
            records[mix][n] = rec
    cells = {f"{mix}/{model}": cell(records[mix], model) for mix in MIXES for model in DESIGN_MODELS}
    cmp = e4_comparison(records["base"], cells["base/balanced"])
    if not cmp["passed"]:
        raise RuntimeError(f"the base mix differs from the E4 records: "
                           f"{ {k: v for k, v in cmp.items() if k != 'differences'} }")
    same = unchanged(prov, True)
    e4.publish(out, {
        "protocol": "docs/fourwire_scale_protocol.md Amendment 12 (E5)", **prov, "unchanged_during_run": same,
        "valid": same, "target": e4.TARGET, "unsafe_below": e4.UNSAFE_BELOW, "majority": MAJORITY,
        "fresh_networks": ev.FRESH, "mixes": {k: list(v) for k, v in MIXES.items()},
        "criteria": {k: {"count": c["count"], "of": c["of"], "criterion_met": c["criterion_met"]}
                     for k, c in cells.items()},
        "recomputed_e4_cell": "base/balanced",
        "attempts": {mix: {str(n): e4.attempts(n, folder / mix) for n in fs.ELIGIBLE} for mix in MIXES},
        "runs_with_more_than_one_attempt": {
            f"{mix}/{n}": e4.attempts(n, folder / mix) for mix in MIXES for n in fs.ELIGIBLE
            if len(e4.attempts(n, folder / mix)) != 1},
        "record_sha256": {mix: {str(n): fs._sha(folder / mix / f"network_{n}.json") for n in fs.ELIGIBLE}
                          for mix in MIXES},
        "positive_sequence_check": {mix: {str(n): records[mix][n]["positive_sequence_check"] for n in fs.ELIGIBLE}
                                    for mix in MIXES},
        "e4_comparison": cmp, "cells": cells})
    if not same:
        raise RuntimeError("checkout, libraries, inputs or a script changed during the report")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--network", type=int, required=True, choices=fs.ELIGIBLE)
    r.add_argument("--mix", required=True, choices=list(MIXES))
    r.add_argument("--dry", action="store_true")
    r.add_argument("--days", type=int)
    r.add_argument("--hours", type=int, default=HOURS, choices=range(1, HOURS + 1), metavar="1..24")
    r.add_argument("--skip-opendss", action="store_true")
    r.add_argument("--processes", type=int)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        print(run(a.network, a.mix, a.dry, a.days, a.hours, not a.skip_opendss, a.processes))
    else:
        print(report())


if __name__ == "__main__":
    main()
