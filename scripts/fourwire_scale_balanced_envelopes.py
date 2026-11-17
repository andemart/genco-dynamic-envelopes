"""Family E4 of the four-wire scale study (docs/fourwire_scale_protocol.md, Amendment 11 and its corrections):
envelopes designed with a balanced-load model against four-wire-verified envelopes, on every eligible
network. No learned model; no label is read: every state is regenerated from the seeds and the meter profiles.

    python scripts/fourwire_scale_balanced_envelopes.py run --network 17
    python scripts/fourwire_scale_balanced_envelopes.py run --network 17 --dry --days 2 --hours 6
    python scripts/fourwire_scale_balanced_envelopes.py report

run: for every hour of the 36 held-out days and both directions, on the design scenarios of the E2 envelope
check (fourwire_scale_e2.day_scenarios: same technology flags, same meter draws):
- four-wire-verified envelope: the ENWL1 rule (envelopes.scalar_envelope), judged by the Numba four-wire
  sweep with every customer on its own phase connection (evaluator A of the E2 envelope check, unchanged);
- balanced-model envelope: the same rule, judged by the same sweep on the same network, source and limits,
  with every customer's power spread equally over the three phases at its bus;
- four-wire check on fresh draws: 200 further meter draws of the same hour (stream 2); the share of them
  that is feasible in the four-wire sweep when every customer takes up the envelope in full, for every
  positive envelope. A state on which the sweep does not converge is infeasible;
- on three of the days, the same fresh states of every positive envelope are also solved by OpenDSS under
  the failure rule of §3.
One record per network, results/fourwire_scale_e4/network_N.json, published once and never replaced; every
start of a final run leaves an attempt file in results/fourwire_scale_e4/attempts.

report: the per-network summary, the classification and the criterion of Amendment 11, the comparison of the
two engines on the three days, and the comparison with the committed E2 envelope records.

--dry uses validation days only and writes under data_out/fourwire_scale/e4_dry; it is not a final run.
No command takes an output path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
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
from gencodoe import limits
from gencodoe.fourwire_sweep_opt import NumbaSweep

SCRIPT = Path(__file__).resolve()
RECORDS = fs.RESULTS / "fourwire_scale_e4"
REPORT = fs.RESULTS / "fourwire_scale_e4.json"
DRY = fs.OUT / "e4_dry"
E2_MANIFEST = fs.RESULTS / "fourwire_scale_final_outputs_sha256.txt"
HOURS = 24
CORES = 16                          # of a final run: Numba threads and OpenDSS workers (one chunk per worker)
N_FRESH = e2.N_SCENARIOS            # 200 fresh draws per interval, as on ENWL1
STREAM_DESIGN, STREAM_FRESH = 1, 2  # third key of the generator seed [network, day, stream]; 0 is the flags
TARGET = 0.95                       # four-wire feasibility on fresh draws an envelope has to reach
UNSAFE_BELOW = 0.5                  # a network is "balanced unsafe" if less than this share of its positive
#                                     balanced envelopes reaches TARGET
MAJORITY = 12                       # of the 23 fresh networks
OPENDSS_DAYS = (0, 12, 24)          # positions in the list of days: the days also solved by OpenDSS
E2_TOLERANCE_SHARE = 0.005          # report: share of envelopes that may differ from the E2 records
MODELS = ("fourwire", "balanced")
SCOPES = ("import", "export", "both")
N_BOOT = 2000
SETTINGS = {"hours": HOURS, "opendss_processes": CORES, "scenarios": e2.N_SCENARIOS, "fresh_draws": N_FRESH,
            "eps": e2.EPS, "beta": e2.BETA,
            "acceptance_share": env.acceptance_threshold(e2.N_SCENARIOS, e2.EPS, e2.BETA),
            "levels_per_round": e2.LEVELS, "relative_tolerance": e2.REL_TOL, "cap_kw": e2.CAP_KW,
            "mix_hp_ev_pv": list(e2.MIX), "target": TARGET}
SAME_AS_REPORT = ("code_commit", "analysis_script_sha256", "freeze_sha256", "source_sha256", "software",
                  "label_manifest_sha256", "identification_sha256", "voltage_limits")


# ------------------------------------------------------------------------------------------- provenance
def _git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=fs.ROOT)


def script_committed() -> bool:
    """This file is tracked and identical to HEAD."""
    rel = str(SCRIPT.relative_to(fs.ROOT))
    return (_git("ls-files", "--error-unmatch", rel).returncode == 0
            and not _git("status", "--porcelain", "--", rel).stdout.strip())


def start(final: bool) -> dict:
    """Provenance of a run. A final run needs: GENCODOE_COMMIT equal to HEAD, unmodified source equal to the
    freeze (the sweep, the scenario generator and the envelope rule are the audited ones), the frozen
    identification record (Thevenin equivalents, network files), meter profiles, library versions and job
    script of the final runs, this script as committed, and the UK statutory limits. Not required, because
    nothing here uses them: the GPU of the freeze, PyTorch's deterministic mode, the folds, and the halt file
    of the final runs."""
    prov = tt.start_provenance("cpu", tt.numerical_settings())
    prov["analysis_script_sha256"] = fs._sha(SCRIPT)
    prov["voltage_limits"] = limits.record()
    if final:
        fr = json.loads(tt.freeze_file().read_text())
        problems = []
        if not tt.freeze_committed():
            problems.append("the freeze file is not committed")
        if prov["gencodoe_commit"] != prov["code_commit"]:
            problems.append("GENCODOE_COMMIT is not set to HEAD")
        if prov["src_dirty"]:
            problems.append("src/ is modified")
        if prov["source_sha256"] != fr["source_sha256"]:
            problems.append("the source differs from the freeze")
        if any(fr[k] != prov[k] for k in tt.input_hashes()):
            problems.append("label manifest or identification record differ from the freeze")
        if fs._sha(tt.JOB_SCRIPT) != fr["job_script_sha256"]:
            problems.append("the job script of the final runs differs from the freeze")
        if prov["software"] != fr["software"]:
            problems.append("library versions differ from the freeze")
        if tt.profiles_sha() != fr["profiles_sha256"]:
            problems.append("meter profiles differ from the freeze")
        if not script_committed():
            problems.append("this script differs from HEAD")
        if limits.JURISDICTION != "uk_esqcr":
            problems.append("voltage limits are not uk_esqcr")
        if problems:
            raise RuntimeError("not a final run: " + "; ".join(problems))
        prov["freeze_sha256"] = fs._sha(tt.freeze_file())
    return prov


def unchanged(prov: dict, final: bool) -> bool:
    return bool(tt.checkout_unchanged(prov, profiles=final) and fs._sha(SCRIPT) == prov["analysis_script_sha256"])


def publish(path: Path, obj) -> None:
    """Publish a complete file under a name that must not exist: written next to it first, then linked. A
    process that dies before the link leaves only the partial file, which is kept and never read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    with open(tmp, "x") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)                                  # fails if path exists: nothing is ever replaced
    os.unlink(tmp)


def attempts(network: int, folder: Path) -> list[str]:
    return sorted(p.name for p in (folder / "attempts").glob(f"network_{network}_*.json"))


def claim_attempt(network: int, prov: dict, folder: Path) -> tuple[str, list[str]]:
    """Every start of a final run of a network leaves a file that is never removed (Amendment 11, item 7):
    the ledger of attempts. Returns its name and the names of the earlier attempts."""
    earlier = attempts(network, folder)
    name = f"network_{network}_{time.strftime('%Y%m%dT%H%M%S')}_{prov['job_id'] or os.getpid()}.json"
    publish(folder / "attempts" / name, {
        "network": network, "job_id": prov["job_id"], "node": prov["node"], "code_commit": prov["code_commit"],
        "analysis_script_sha256": prov["analysis_script_sha256"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "earlier_attempts": earlier})
    return name, earlier


# ------------------------------------------------------------------------------------------- scenarios
def day_list(dry: bool) -> list[int]:
    """Day-of-year indices (0 = 1 January 2024): every held-out (test) day; validation days in a dry run."""
    d = np.arange(366)
    return [int(x) for x in d[dk.day_split(d * 96) == ("val" if dry else "test")]]


def scenarios(network: int, n_cust: int, day: int, hours: int, stream: int):
    """fourwire_scale_e2.day_scenarios with the stream of the meter draws as an argument: (step, p_kw, q_kvar),
    each (200, n_cust) customer net consumption. The technology flags are those of the network and day
    (stream 0) for every stream; stream 1 gives the design scenarios of the E2 envelope check."""
    from gencodoe.fluvius import GROUPS, load_group
    if not 1 <= hours <= HOURS:
        raise ValueError("hours must be 1 to 24: a later hour would be a step of the next day")
    groups = [load_group(g) for g in GROUPS]
    by_flags = {(g.hp, g.ev, g.pv): g for g in groups}
    flags = np.random.default_rng([network, day, 0]).uniform(size=(n_cust, 3)) < np.array(e2.MIX)
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


# ------------------------------------------------------------------------------------------- one network
def run(network: int, dry: bool, days: int | None = None, hours: int = HOURS, opendss: bool = True,
        processes: int | None = None) -> Path:
    final = not dry
    if final and (days is not None or hours != HOURS or not opendss or processes is not None):
        raise SystemExit("a final run takes no --days, --hours, --skip-opendss or --processes")
    if not 1 <= hours <= HOURS or (days is not None and days < 1):
        raise SystemExit("--hours must be 1 to 24 and --days at least 1")
    if final and e2.allocated_cores() != CORES:
        raise SystemExit(f"a final run needs {CORES} cores; this job has {e2.allocated_cores()}")
    out = (RECORDS / f"network_{network}.json" if final
           else DRY / f"network_{network}_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json")
    if out.exists():
        raise FileExistsError(f"{out} exists")
    prov = start(final)
    attempt, earlier = claim_attempt(network, prov, RECORDS) if final else (None, [])
    t0 = time.time()
    split = "val" if dry else "test"
    net, th = fs._case(network)
    names, idx, share = cc.customers(net.customer_names)
    n_cust = len(names)
    first = np.array([np.flatnonzero(idx == j)[0] for j in range(n_cust)])
    bus = np.asarray(net.customer_bus)[first]
    if not (np.asarray(net.customer_bus) == bus[idx]).all():
        raise AssertionError("the phase connections of a customer are not at one bus")
    sweep = NumbaSweep(net, th)
    balanced = NumbaSweep(dataclasses.replace(
        net, customer_names=[f"{n}_{k}" for n in names for k in "abc"], customer_bus=np.repeat(bus, 3),
        customer_phase=np.tile(np.arange(3), n_cust)), th)
    customers = env.Customers(bus=np.arange(n_cust), export_cap=np.full(n_cust, e2.CAP_KW["export"]),
                              import_cap=np.full(n_cust, e2.CAP_KW["import"]))
    nonconv = {m: 0 for m in MODELS}

    def as_connected(p, q):       # envelope rows (+ = export), kW per customer -> kVA consumed per phase connection
        return -(p[:, idx] * share) - 1j * (q[:, idx] * share)

    def as_balanced(p, q):        # the same customer power, one third on each phase at the customer's bus
        return np.repeat(-p - 1j * q, 3, axis=1) / 3

    def judged(model: str, s, direction: str) -> np.ndarray:
        o = (sweep if model == "fourwire" else balanced).solve(s, return_state=False)
        nonconv[model] += int((~o["converged"]).sum())
        return e2.decide(o["summary"], o["converged"])[:, ev.DIRECTIONS.index(direction)]

    feasible = {"fourwire": lambda p, q, d: judged("fourwire", as_connected(p, q), d),
                "balanced": lambda p, q, d: judged("balanced", as_balanced(p, q), d)}

    all_days = day_list(dry)[:days]
    dss_days = [all_days[k] for k in OPENDSS_DAYS if k < len(all_days)] if opendss else []
    processes = CORES if final else (processes or e2.allocated_cores())
    pool = e2.OpenDSSPool(network, processes) if dss_days else None
    rows = []
    try:
        for day in all_days:
            design = e2.day_scenarios(network, n_cust, day, hours)
            again = scenarios(network, n_cust, day, hours, STREAM_DESIGN)
            fresh = scenarios(network, n_cust, day, hours, STREAM_FRESH)
            for (step, p_kw, q_kvar), (step_s, ps, qs), (step_f, pf, qf) in zip(design, again, fresh):
                if not (step == step_s == step_f and step // 96 == day and dk.day_split(step) == split):
                    raise AssertionError(f"step {step} is not a step of {split} day {day} in every stream")
                if not (np.array_equal(ps, p_kw) and np.array_equal(qs, q_kvar)):   # this script's generator
                    raise AssertionError("scenarios() with stream 1 differs from e2.day_scenarios")  # is the E2 one
                for direction in ev.DIRECTIONS:
                    k = ev.DIRECTIONS.index(direction)
                    row = {"day": day, "date": str(np.datetime64("2024-01-01") + day), "step": int(step),
                           "direction": direction}
                    for model in MODELS:
                        before = nonconv[model]
                        e = env.scalar_envelope(feasible[model], n_cust, customers, -p_kw, -q_kvar, direction,
                                                "uniform", e2.EPS, e2.LEVELS, e2.REL_TOL, beta=e2.BETA)
                        level = float(e.level)
                        row.update({f"{model}_level_kw": level, f"{model}_design_share": float(e.prob_ok),
                                    f"{model}_states": int(e.n_evals),
                                    f"{model}_sweep_nonconverged": nonconv[model] - before,
                                    f"{model}_fresh_share": None, f"{model}_fresh_share_opendss": None,
                                    f"{model}_fresh_decisions_differ": None})
                        if level > 0:     # fresh draws with the envelope taken up in full, in the four-wire sweep
                            s = as_connected(-pf + env.DIRECTIONS[direction] * level, -qf)
                            o = sweep.solve(s, return_state=False)
                            ok = e2.decide(o["summary"], o["converged"])[:, k]
                            row[f"{model}_fresh_share"] = float(ok.mean())
                            if day in dss_days:
                                quant, conv = pool.solve(s)
                                ok_dss = e2.decide(quant, conv)[:, k]
                                row[f"{model}_fresh_share_opendss"] = float(ok_dss.mean())
                                row[f"{model}_fresh_decisions_differ"] = int((ok != ok_dss).sum())
                    rows.append(row)
            print(day, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
    same = unchanged(prov, final)
    publish(out, {
        "protocol": "docs/fourwire_scale_protocol.md Amendment 11 (E4)", "network": network, "dry_run": dry,
        **prov, "unchanged_during_run": same, "valid": same, "attempt": attempt, "earlier_attempts": earlier,
        "customers": n_cust, "phase_connections": int(len(idx)), "retained_buses": int(net.n_bus),
        "days": all_days, "opendss_days": dss_days, "numba_threads": int(numba.get_num_threads()),
        **SETTINGS, "hours": hours, "opendss_processes": processes if dss_days else None,
        "seconds": time.time() - t0, "summary": summarise(rows), "intervals": rows})
    if not same:
        raise RuntimeError(f"checkout, libraries, inputs or this script changed during the run; "
                           f"{out.name} is not a valid record")
    return out


# ------------------------------------------------------------------------------------------- summary
def _share(a: int, b: int) -> float | None:
    return a / b if b else None


def reaches(share: float) -> bool:
    """At least 190 of the 200 fresh draws feasible (the share is a count over 200; no rounding at 0.95)."""
    return bool(round(share * N_FRESH) >= round(TARGET * N_FRESH))


def scoped(rows: list[dict], scope: str) -> list[dict]:
    return [r for r in rows if scope == "both" or r["direction"] == scope]


def counts(rows: list[dict], key: str = "fresh_share") -> dict:
    """Positive envelopes of each model and how many reach TARGET on the fresh draws (as judged under key)."""
    out = {}
    for m in MODELS:
        pos = [r for r in rows if r[f"{m}_level_kw"] > 0 and r[f"{m}_{key}"] is not None]
        meet = sum(reaches(r[f"{m}_{key}"]) for r in pos)
        out[m] = {"positive": len(pos), "reaching_target": int(meet), "share_reaching_target": _share(meet, len(pos)),
                  "mean_fresh_share": float(np.mean([r[f"{m}_{key}"] for r in pos])) if pos else None}
    return out


def summarise(rows: list[dict]) -> dict:
    """Per direction and for both: counts, the two models against each other, and capacity."""
    out = {}
    n_days = len({r["day"] for r in rows})
    for scope in SCOPES:
        r = scoped(rows, scope)
        if not r:
            continue
        cap = np.array([e2.CAP_KW[x["direction"]] for x in r])
        fw = np.array([x["fourwire_level_kw"] for x in r])
        bal = np.array([x["balanced_level_kw"] for x in r])
        both = (fw > 0) & (bal > 0)
        tol = e2.REL_TOL * cap
        s = {"intervals": len(r), **counts(r),
             "fourwire_zero_balanced_positive": int(((fw == 0) & (bal > 0)).sum()),
             "balanced_above_fourwire": int((bal > fw + tol).sum()),
             "balanced_below_fourwire": int((bal < fw - tol).sum()),
             "both_at_cap": int(((fw == cap) & (bal == cap)).sum()),
             "median_ratio_balanced_to_fourwire": float(np.median(bal[both] / fw[both])) if both.any() else None,
             "fourwire_smallest_level_kw": float(fw.min())}
        if scope != "both":       # envelope-hours per household per year (1 h intervals, 366 days of 2024)
            s["fourwire_mwh_per_household_year"] = float(fw.sum() / n_days * 366 / 1e3)
            s["balanced_mwh_per_household_year"] = float(bal.sum() / n_days * 366 / 1e3)
        out[scope] = s
    return out


def day_bootstrap(rows: list[dict]) -> dict:
    """95% interval of the share of positive balanced envelopes reaching TARGET, resampling whole days. A
    replicate without a positive balanced envelope has no share: it is left out and counted, so the interval
    is conditional on at least one positive envelope."""
    days = sorted({r["day"] for r in rows})
    pos = {d: 0 for d in days}
    meet = {d: 0 for d in days}
    for r in rows:
        if r["balanced_level_kw"] > 0:
            pos[r["day"]] += 1
            meet[r["day"]] += reaches(r["balanced_fresh_share"])
    p, m = np.array([pos[d] for d in days]), np.array([meet[d] for d in days])
    pick = np.random.default_rng(0).integers(0, len(days), size=(N_BOOT, len(days)))
    den, num = p[pick].sum(1), m[pick].sum(1)
    ok = den > 0
    if not ok.any():
        return {"replicates": N_BOOT, "without_positive_envelope": N_BOOT, "ci95": None}
    lo, hi = np.percentile(num[ok] / den[ok], [2.5, 97.5])
    return {"replicates": N_BOOT, "without_positive_envelope": int((~ok).sum()), "ci95": [float(lo), float(hi)]}


def engines(rows: list[dict]) -> dict:
    """The OpenDSS days: the sweep's and OpenDSS's decisions on the same fresh states. engines_classify_alike
    is None where no positive balanced envelope was solved by OpenDSS: the engines are then not compared."""
    dss = [r for r in rows if any(r[f"{m}_fresh_share_opendss"] is not None for m in MODELS)]
    checked = [(r, m) for r in dss for m in MODELS if r[f"{m}_fresh_share_opendss"] is not None]
    by_sweep, by_opendss = counts(dss), counts(dss, "fresh_share_opendss")
    sw, od = by_sweep["balanced"], by_opendss["balanced"]
    return {"envelopes": len(checked), "states": N_FRESH * len(checked),
            "decisions_differ": int(sum(r[f"{m}_fresh_decisions_differ"] for r, m in checked)),
            "envelopes_with_other_status": int(sum(
                reaches(r[f"{m}_fresh_share"]) != reaches(r[f"{m}_fresh_share_opendss"]) for r, m in checked)),
            "sweep": by_sweep, "opendss": by_opendss,
            "engines_classify_alike": None if od["positive"] == 0 else bool(
                (sw["share_reaching_target"] < UNSAFE_BELOW) == (od["share_reaching_target"] < UNSAFE_BELOW))}


def classify(rows: list[dict]) -> dict:
    """Amendment 11, item 4, the two directions together. Balanced unsafe: the network has a positive balanced
    envelope and less than half of them reach TARGET. Counted: balanced unsafe, unless the two engines
    classify the OpenDSS days differently. Where the OpenDSS days have no positive balanced envelope the
    engines are not compared and the classification by the sweep stands."""
    c = counts(rows)["balanced"]
    unsafe = bool(c["positive"] > 0 and c["share_reaching_target"] < UNSAFE_BELOW)
    alike = engines(rows)["engines_classify_alike"]
    return {"balanced_unsafe": unsafe, "engines_classify_alike": alike,
            "counted_unsafe": bool(unsafe and alike is not False),
            "by_scope": {s: {"balanced_share_ci95_days": day_bootstrap(scoped(rows, s)),
                             "opendss_days": engines(scoped(rows, s))} for s in SCOPES}}


# ------------------------------------------------------------------------------------------- report
def expected_intervals(days: list[int]) -> list[tuple]:
    return [(d, d * 96 + 4 * h, direction) for d in days for h in range(HOURS) for direction in ev.DIRECTIONS]


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _count(share) -> int | None:
    """The number of feasible states a share stands for, if it is a count over the 200 fresh draws."""
    if not _number(share) or not 0 <= share <= 1 or abs(share * N_FRESH - round(share * N_FRESH)) > 1e-9:
        return None
    return round(share * N_FRESH)


def ledger_problems(rec: dict, network: int, prov: dict, folder: Path) -> list[str]:
    """The record's attempt must be a file of the ledger, written by the run that wrote the record."""
    name, earlier = rec.get("attempt"), rec.get("earlier_attempts")
    listed = attempts(network, folder)
    if not isinstance(name, str) or not isinstance(earlier, list) or name not in listed:
        return ["its attempt is not in the ledger"]
    try:
        att = json.loads((folder / "attempts" / name).read_text())
    except (OSError, ValueError):
        return ["its attempt file cannot be read"]
    same = (isinstance(att, dict) and att.get("network") == network and att.get("earlier_attempts") == earlier
            and all(att.get(k) == rec.get(k) for k in ("job_id", "node"))
            and all(att.get(k) == prov[k] for k in ("code_commit", "analysis_script_sha256")))
    return ([] if same else ["its attempt file is of another run"]) + (
        [] if all(e in listed for e in earlier) and name not in earlier else ["its earlier attempts are not in the ledger"])


def record_problems(rec: dict, network: int, prov: dict, days: list[int], folder: Path) -> list[str]:
    """What keeps a record out of the report: not a valid final record of this network, commit, script, freeze
    and settings; an attempt that is not in the ledger; or not the complete set of intervals with every result
    a positive envelope must have, each a count over the 200 fresh draws, with a number of differing decisions
    that the two counts allow."""
    dss = [days[k] for k in OPENDSS_DAYS]
    bad = [what for what, wrong in [
        ("not valid", rec.get("valid") is not True or rec.get("unchanged_during_run") is not True),
        ("a dry run", rec.get("dry_run") is not False), ("another network", rec.get("network") != network),
        ("other days", rec.get("days") != days), ("other OpenDSS days", rec.get("opendss_days") != dss),
    ] if wrong]
    bad += ledger_problems(rec, network, prov, folder)
    bad += [f"another {k}" for k in SAME_AS_REPORT if rec.get(k) != prov[k]]
    bad += [f"another {k}" for k, v in SETTINGS.items() if rec.get(k) != v]
    rows = rec.get("intervals", [])
    try:
        if [(r["day"], r["step"], r["direction"]) for r in rows] != expected_intervals(days):
            bad.append("not the expected intervals")
        for r in rows:
            for m in MODELS:
                level = r[f"{m}_level_kw"]
                res = [r[f"{m}_fresh_share"], r[f"{m}_fresh_share_opendss"], r[f"{m}_fresh_decisions_differ"]]
                need = [level > 0, level > 0 and r["day"] in dss, level > 0 and r["day"] in dss]
                if (not _number(level) or not 0 <= level <= e2.CAP_KW[r["direction"]]
                        or [x is not None for x in res] != need):
                    raise ValueError
                a, b, d = (_count(res[0]) if need[0] else None), (_count(res[1]) if need[1] else None), res[2]
                if (need[0] and a is None) or (need[1] and b is None):
                    raise ValueError
                if need[2]:       # d states decided differently: x feasible for the sweep only, y for OpenDSS only
                    if not isinstance(d, int) or isinstance(d, bool) or (d + a - b) % 2:
                        raise ValueError
                    x, y = (d + a - b) // 2, (d - a + b) // 2
                    if not (0 <= x <= min(a, N_FRESH - b) and 0 <= y <= min(b, N_FRESH - a)):
                        raise ValueError
    except (KeyError, TypeError, ValueError):
        bad.append("an interval without the results it must have")
    return bad


def e2_manifest() -> dict:
    """Path -> SHA-256 of the final outputs committed with the outcome of E2 and E3."""
    lines = [ln.split() for ln in E2_MANIFEST.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {path: digest for digest, path in lines}


def e2_comparison(records: dict) -> dict:
    """The four-wire levels of the E2 networks on the E2 days against evaluator A of the committed E2 envelope
    records, which must be the files of the manifest. Every difference is listed."""
    manifest = e2_manifest()
    out = {"compared": 0, "equal": 0, "beyond_tolerance": 0, "largest_difference_kw": 0.0,
           "manifest_sha256": fs._sha(E2_MANIFEST), "e2_records_sha256": {}, "differences": []}
    for n in e2.E2_NETWORKS:
        path = f"results/fourwire_scale_e2_envelopes/network_{n}.json"
        digest = fs._sha(fs.ROOT / path)
        if manifest.get(path) != digest:
            raise RuntimeError(f"{path} is not the file of {E2_MANIFEST.name}")
        out["e2_records_sha256"][str(n)] = digest
        mine = {(r["day"], r["step"], r["direction"]): r["fourwire_level_kw"] for r in records[n]["intervals"]}
        for r in json.loads((fs.ROOT / path).read_text())["intervals"]:
            new = mine[(r["day"], r["step"], r["direction"])]
            d = abs(new - r["A_level_kw"])
            out["compared"] += 1
            out["equal"] += int(d == 0)
            out["beyond_tolerance"] += int(d > e2.REL_TOL * e2.CAP_KW[r["direction"]])
            out["largest_difference_kw"] = max(out["largest_difference_kw"], float(d))
            if d > 0:
                out["differences"].append({"network": n, "day": r["day"], "step": r["step"],
                                           "direction": r["direction"], "e2_level_kw": r["A_level_kw"],
                                           "e4_level_kw": new})
    out["passed"] = bool(out["beyond_tolerance"] <= E2_TOLERANCE_SHARE * out["compared"])
    return out


def report(folder: Path = RECORDS, out: Path = REPORT) -> Path:
    if out.exists():
        raise FileExistsError(f"{out} exists")
    prov = start(final=True)
    days, records = day_list(False), {}
    for n in fs.ELIGIBLE:
        rec = json.loads((folder / f"network_{n}.json").read_text())
        problems = record_problems(rec, n, prov, days, folder)
        if problems:
            raise RuntimeError(f"record of network {n}: " + ", ".join(problems))
        records[n] = rec
    cmp = e2_comparison(records)
    if not cmp["passed"]:
        raise RuntimeError(f"four-wire levels differ from the E2 envelope records: "
                           f"{ {k: v for k, v in cmp.items() if k != 'differences'} }")
    per = {}
    for n, rec in records.items():
        per[str(n)] = {"customers": rec["customers"], "retained_buses": rec["retained_buses"],
                       "fresh": n in ev.FRESH, "attempts": attempts(n, folder),
                       "summary": summarise(rec["intervals"]), **classify(rec["intervals"]),
                       "record_sha256": fs._sha(folder / f"network_{n}.json")}
    fresh = [per[str(n)] for n in ev.FRESH]
    unsafe = [n for n in ev.FRESH if per[str(n)]["counted_unsafe"]]
    same = unchanged(prov, True)
    publish(out, {
        "protocol": "docs/fourwire_scale_protocol.md Amendment 11 (E4)", **prov, "unchanged_during_run": same,
        "valid": same, "target": TARGET, "unsafe_below": UNSAFE_BELOW, "majority": MAJORITY,
        "fresh_networks": ev.FRESH,
        "balanced_unsafe_by_sweep": [n for n in ev.FRESH if per[str(n)]["balanced_unsafe"]],
        "engines_classify_differently": [n for n in fs.ELIGIBLE if per[str(n)]["engines_classify_alike"] is False],
        "engines_not_compared": [n for n in fs.ELIGIBLE if per[str(n)]["engines_classify_alike"] is None],
        "without_positive_balanced_envelope": [
            n for n in fs.ELIGIBLE if per[str(n)]["summary"]["both"]["balanced"]["positive"] == 0],
        "balanced_unsafe_counted": unsafe, "count": len(unsafe), "of": len(fresh),
        "criterion_met": bool(len(unsafe) >= MAJORITY), "network_1": per["1"]["counted_unsafe"],
        "networks_run_more_than_once": {str(n): per[str(n)]["attempts"] for n in fs.ELIGIBLE
                                        if len(per[str(n)]["attempts"]) != 1},
        "e2_comparison": cmp, "per_network": per})
    if not same:
        raise RuntimeError("checkout, libraries, inputs or this script changed during the report")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--network", type=int, required=True, choices=fs.ELIGIBLE)
    r.add_argument("--dry", action="store_true")
    r.add_argument("--days", type=int)
    r.add_argument("--hours", type=int, default=HOURS, choices=range(1, HOURS + 1), metavar="1..24")
    r.add_argument("--skip-opendss", action="store_true")
    r.add_argument("--processes", type=int)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        print(run(a.network, a.dry, a.days, a.hours, not a.skip_opendss, a.processes))
    else:
        print(report())


if __name__ == "__main__":
    main()
