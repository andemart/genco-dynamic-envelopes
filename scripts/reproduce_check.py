"""Regenerate the aggregate report files covered below from their recorded inputs and compare them byte for byte
with the committed versions. Needs the locally retained run artifacts under data_out/ (not redistributed), so it
does not run on a fresh clone. The case118 studies (branch ander-case118) are not covered here.

    GENCODOE_VOLTAGE_LIMITS=uk_esqcr .venv/bin/python scripts/reproduce_check.py [--include-slow]

Exit status 0 only if every checked file is identical. Results that need the A100 (the four-wire
evaluation) are checked against the SHA-256 recorded in their run manifest instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable


def inputs_of(report: str) -> list[str]:
    return [i["file"] for i in json.loads(Path(report).read_text())["inputs"]]


def jobs(tmp: Path, include_slow: bool):
    yield ("annual DOE + C+", "results/annual_report_with_cplus.json",
           [PY, "-m", "gencodoe.report_annual", "--runs", *inputs_of("results/annual_report_with_cplus.json"),
            "--out", str(tmp / "annual.json")], tmp / "annual.json")
    for day in ("2024-06-28", "2024-12-15"):
        ref = f"results/report_{day}_with_cplus.json"
        yield (f"15-min day {day} + C+", ref, [PY, "-m", "gencodoe.report_final", "--runs", *inputs_of(ref),
                                                "--out", str(tmp / f"{day}.json")], tmp / f"{day}.json")
    yield ("W1-W3 weekend report", "results/weekend_report.json",
           [PY, "-m", "gencodoe.report_weekend", "--out", str(tmp / "weekend.json")], tmp / "weekend.json")
    prim = json.loads(Path("results/envpred3_primary_q95.json").read_text())
    yield ("envpred3 frozen primary (q95)", "results/envpred3_primary_q95.json",
           [PY, "-m", "gencodoe.envpred3", "report-primary", "--commit", prim["code_commit"]],
           Path("results/envpred3_primary_q95.json"))
    yield ("envpred3 frontier", "results/envpred3_frontier_report.json",
           [PY, "-m", "gencodoe.envpred3", "report", "--out", str(tmp / "frontier.json")], tmp / "frontier.json")
    # GENCO-OPF layer (docs/genco_opf_protocol.md): reports regenerate from the raw day files under data_out/opf_case33
    # (hashes in results/opf_artifact_hashes.json); each ends in a bootstrap with a fixed seed.
    yield ("GENCO-OPF stage A report", "results/genco_opf_stageA_report.json",
           [PY, "-m", "gencodoe.opf_report", "--test-dir", "data_out/opf_case33/test", "--out", str(tmp / "opfA.json")], tmp / "opfA.json")
    yield ("GENCO-OPF amendment 3 report", "results/genco_opf_amendment3_report.json",
           [PY, "-m", "gencodoe.opf_report", "--test-dir", "data_out/opf_case33/test", "--extra-dir", "data_out/opf_case33/test_genco",
            "--out", str(tmp / "opf3.json")], tmp / "opf3.json")
    yield ("GENCO-OPF amendment 4 report (run 2)", "results/genco_opf_amendment4_report.json",
           [PY, "-m", "gencodoe.opf_se", "report", "--dir", "data_out/opf_case33/se", "--out", str(tmp / "opf4.json")], tmp / "opf4.json")
    yield ("GENCO-OPF amendment 4 report (run 1)", "results/genco_opf_amendment4_report_run1.json",
           [PY, "-m", "gencodoe.opf_se", "report", "--dir", "data_out/opf_case33/se_run1", "--out", str(tmp / "opf4r1.json")], tmp / "opf4r1.json")
    if include_slow:
        yield ("Option 2 screen (GENCO on CPU, slow)", "results/option2_screen.json",
               [PY, "-m", "gencodoe.option2", "--run", "mlruns_hydra/877677247744645366/d42d5c34e3ed4988b020d8ab292201ee",
                "--config", "configs/genco_pf_enwl1_lv_prior_v2.yaml", "--out", str(tmp / "option2.json")], tmp / "option2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-slow", action="store_true")
    a = ap.parse_args()
    env = {**os.environ, "GENCODOE_VOLTAGE_LIMITS": os.environ.get("GENCODOE_VOLTAGE_LIMITS", "uk_esqcr")}
    ok = True
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, ref, cmd, out in jobs(tmp, a.include_slow):
            before = Path(ref).read_bytes()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            same = r.returncode == 0 and out.exists() and out.read_bytes() == before
            if out == Path(ref) and not same:              # report-primary rewrites in place: restore on mismatch
                Path(ref).write_bytes(before)
            print(f"{'OK  ' if same else 'DIFF'} {name}: {ref}" + ("" if r.returncode == 0 else f" (exit {r.returncode})"))
            ok &= same
    m = json.loads(Path("results/run_manifest_fourwire.json").read_text())
    ev = [s for s in m["stages"] if s["stage"] == "frozen evaluation"][0]
    same = hashlib.sha256(Path(ev["output"]).read_bytes()).hexdigest() == ev["sha256"]
    print(f"{'OK  ' if same else 'DIFF'} four-wire evaluation (A100; hash vs run manifest): {ev['output']}")
    ok &= same
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
