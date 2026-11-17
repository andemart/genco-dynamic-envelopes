"""Exploratory specification table for E4 (docs/fourwire_scale_protocol.md, Amendment 12, item 6): how many of
the 23 fresh networks are balanced unsafe under other choices of the rules that Amendment 11 fixed. Computed
from the committed E4 records only; no run, no new state. It changes no result or criterion of E4.

    python scripts/fourwire_scale_e4_robustness.py

Writes results/fourwire_scale_e4_robustness.json once (it refuses if the file exists).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import platform
from pathlib import Path

import numpy as np

from gencodoe import fourwire_scale as fs
from gencodoe import fourwire_scale_eval as ev

SCRIPT = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("fourwire_scale_balanced_envelopes",
                                               SCRIPT.with_name("fourwire_scale_balanced_envelopes.py"))
e4 = importlib.util.module_from_spec(_spec)     # the E4 script: constants, paths and publish()
_spec.loader.exec_module(e4)

OUT = fs.RESULTS / "fourwire_scale_e4_robustness.json"
N = e4.N_FRESH
CHOICES = {                                     # the first value of each is the rule of Amendment 11
    "target": (0.95, 0.90, 0.98),               # four-wire feasibility an envelope has to reach (190, 180, 196 of 200)
    "unsafe_below": (0.5, 0.25, 0.75),          # share of positive balanced envelopes reaching the target
    "scope": ("both", "import", "export"),      # directions classified
    "intervals": ("all", "fourwire_positive"),  # all positive balanced envelopes, or only those in intervals
    #                                             with a positive four-wire-verified envelope
    "engine": ("sweep_36_days", "sweep_opendss_days", "opendss_opendss_days"),   # who judges, on which days
}
REGISTERED = {k: v[0] for k, v in CHOICES.items()}


def classify(rows: list[dict], spec: dict) -> bool | None:
    """Balanced unsafe under spec: the network has a positive balanced envelope among the rows of the spec and
    less than spec['unsafe_below'] of them reach spec['target']. None without such an envelope."""
    key = "balanced_fresh_share_opendss" if spec["engine"] == "opendss_opendss_days" else "balanced_fresh_share"
    need = round(spec["target"] * N)
    pos = meet = 0
    for r in rows:
        if r["balanced_level_kw"] <= 0 or (spec["scope"] != "both" and r["direction"] != spec["scope"]):
            continue
        if spec["intervals"] == "fourwire_positive" and r["fourwire_level_kw"] <= 0:
            continue
        if spec["engine"] != "sweep_36_days" and r["balanced_fresh_share_opendss"] is None:
            continue                            # not a day on which OpenDSS solved the fresh states
        pos += 1
        meet += round(r[key] * N) >= need
    return None if pos == 0 else bool(meet / pos < spec["unsafe_below"])


def count(records: dict, spec: dict) -> dict:
    by = {n: classify(records[n], spec) for n in ev.FRESH}
    unsafe = [n for n in ev.FRESH if by[n]]
    return {**spec, "count": len(unsafe), "count_without_network_18": len([n for n in unsafe if n != 18]),
            "not_unsafe": [n for n in ev.FRESH if by[n] is False],
            "without_positive_envelope": [n for n in ev.FRESH if by[n] is None]}


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists")
    report = json.loads(e4.REPORT.read_text())
    records = {}
    for n in ev.FRESH:
        path = e4.RECORDS / f"network_{n}.json"
        if fs._sha(path) != report["per_network"][str(n)]["record_sha256"]:
            raise RuntimeError(f"{path.name} is not the E4 record of the E4 report")
        records[n] = json.loads(path.read_text())["intervals"]
    registered = count(records, REGISTERED)
    if registered["count"] != report["count"] or sorted(registered["not_unsafe"]) != sorted(
            set(ev.FRESH) - set(report["balanced_unsafe_counted"])):
        raise RuntimeError("the registered rule does not reproduce the E4 report")
    one_at_a_time = [count(records, {**REGISTERED, k: v}) for k, values in CHOICES.items() for v in values[1:]]
    grid = [count(records, dict(zip(CHOICES, values))) for values in itertools.product(*CHOICES.values())]
    counts = [g["count"] for g in grid]
    e4.publish(OUT, {
        "protocol": "docs/fourwire_scale_protocol.md Amendment 12, item 6 (exploratory; from the E4 records)",
        "code_commit": e4._git("rev-parse", "HEAD").stdout.strip(),
        "tracked_files_modified": bool(e4._git("status", "--porcelain", "--untracked-files=no").stdout.strip()),
        "script_sha256": fs._sha(SCRIPT), "e4_script_sha256": fs._sha(e4.SCRIPT),
        "e4_report_sha256": fs._sha(e4.REPORT), "python": platform.python_version(), "numpy": np.__version__,
        "fresh_networks": len(ev.FRESH), "majority": e4.MAJORITY, "choices": {k: list(v) for k, v in CHOICES.items()},
        "registered": registered, "one_change_at_a_time": one_at_a_time,
        "grid": {"combinations": len(grid), "smallest_count": min(counts), "largest_count": max(counts),
                 "combinations_with_at_least_12": int(sum(c >= e4.MAJORITY for c in counts)),
                 "smallest_count_without_network_18": min(g["count_without_network_18"] for g in grid),
                 "all": grid}})
    print(OUT)


if __name__ == "__main__":
    main()
