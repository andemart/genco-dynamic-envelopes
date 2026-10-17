"""Commit and hash pinning for Hydra runs (docs/envpred_protocol.md, amendment 3c).

    python -m gencodoe.pin verify results/envpred_label_recovery_manifest.json   # exit 1 on any mismatch
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(manifest: Path) -> list[str]:
    m = json.loads(Path(manifest).read_text())
    bad = []
    for key in ("source_sha256", "input_data_sha256"):
        for f, h in m.get(key, {}).items():
            if not Path(f).exists():
                bad.append(f"missing {f}")
            elif sha256(f) != h:
                bad.append(f"hash mismatch {f}")
    return bad


if __name__ == "__main__":
    if sys.argv[1] != "verify":
        raise SystemExit("usage: python -m gencodoe.pin verify <manifest>")
    problems = verify(Path(sys.argv[2]))
    if problems:
        print("PIN VERIFICATION FAILED:", *problems, sep="\n  ")
        raise SystemExit(1)
    print("pin verification ok:", sys.argv[2])
