"""Move (never delete) the artifacts invalidated by the OpenDSS engine-lifecycle defect into a quarantine
directory and write a hash manifest (docs/envpred_protocol.md, amendment 3c).

    python scripts/quarantine_envpred_poisoned.py --site hydra|local
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

Q = Path("data_out/envpred/quarantine_opendss_poisoned_2026-10-17")
REASON = ("produced while the OpenDSS engine was poisoned by a divergent first solve (every exact B label 0), or "
          "derived from those labels; see docs/envpred_protocol.md amendment 3c")
GROUPS = [
    ("data_out/doe/envpred_labels/*.json", "13337097 (hh101) / 13337098 (hh202)", "28ed289b429e1c4555fe469afaa092363b213de3"),
    ("data_out/doe/envpred_labels/*.csv", "13337097 (hh101) / 13337098 (hh202)", "28ed289b429e1c4555fe469afaa092363b213de3"),
    ("data_out/envpred/predictions_v3.parquet", "13338884", "da36fc997835d998a7d69f090188d38deaffe2a2"),
    ("data_out/envpred/fit_v3.json", "13338884", "da36fc997835d998a7d69f090188d38deaffe2a2"),
    ("data_out/envpred/fit_v3_run.json", "13338884", "da36fc997835d998a7d69f090188d38deaffe2a2"),
    ("results/envpred_label_integrity.json", "integrity run (local, structural only; incorrectly passed)",
     "68d69d9 (committed report)"),
    ("logs/envpred-hh101-13337097_*.out", "13337097", "28ed289b429e1c4555fe469afaa092363b213de3"),
    ("logs/envpred-hh202-13337098_*.out", "13337098", "28ed289b429e1c4555fe469afaa092363b213de3"),
    ("logs/envpred3-fit-13338884.out", "13338884", "da36fc997835d998a7d69f090188d38deaffe2a2"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    a = ap.parse_args()
    entries = []
    for pat, job, commit in GROUPS:
        for src in sorted(glob.glob(pat)):
            dst = Q / src
            dst.parent.mkdir(parents=True, exist_ok=True)
            h = sha(src)
            if dst.exists():
                raise SystemExit(f"refusing to overwrite {dst}")
            shutil.move(src, dst)
            if sha(dst) != h:
                raise SystemExit(f"hash changed while moving {src}")
            entries.append({"original": src, "quarantined": str(dst), "sha256": h, "job": job, "commit": commit})
    gate = [p for p in ("data_out/envpred/gate_v3", "data_out/envpred/gate") if Path(p).exists() and any(Path(p).iterdir())]
    m = {"site": a.site, "reason": REASON, "n_files": len(entries),
         "counts": {pat: sum(e["original"] in glob.glob(pat) or Path(e["original"]).match(pat) for e in entries)
                    for pat, _, _ in GROUPS},
         "gate_outputs_exist": bool(gate), "gate_dirs_checked": ["data_out/envpred/gate_v3", "data_out/envpred/gate"],
         "valid_features_untouched": "data_out/envpred/features_v3 (840 blocks) not moved", "files": entries}
    out = Q / f"quarantine_manifest_{a.site}.json"
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
