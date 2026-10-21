"""Amendment 5 option 2: fixed, nested label subsets for graphkit's split_from_existing_files (amendment 5a).

    python scripts/amendment5_splits.py --raw data_out/case118_flex/raw --out data_out/case118_flex/splits

A fixed random permutation (seed 0) of the solved scenario ids: the first 3,000 are the test split, the next 3,000
validation, the rest the training pool; train subsets are the first N of the pool for N in {100, 1000, 10000}.
Writes splits/N{N}/{train,val,test}.pt and splits/manifest.json with SHA-256 of every file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_TEST, N_VAL, SUBSETS, SEED = 3000, 3000, (100, 1000, 10000), 0


def make_splits(ids: np.ndarray, out: Path, seed: int = SEED) -> dict:
    import torch
    ids = np.sort(np.unique(ids))
    if len(ids) < N_TEST + N_VAL + max(SUBSETS):
        raise ValueError(f"{len(ids)} scenarios are too few for the preregistered splits")
    perm = ids[np.random.default_rng(seed).permutation(len(ids))]
    test, val, pool = perm[:N_TEST], perm[N_TEST:N_TEST + N_VAL], perm[N_TEST + N_VAL:]
    man = {"seed": seed, "n_scenarios": int(len(ids)), "n_test": N_TEST, "n_val": N_VAL, "subsets": {}}
    for n in SUBSETS:
        d = out / f"N{n}"
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, arr in (("train", pool[:n]), ("val", val), ("test", test)):
            p = d / f"{name}.pt"
            torch.save(torch.tensor(np.sort(arr), dtype=torch.long), p)
            files[name] = {"n": int(len(arr)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        man["subsets"][str(n)] = files
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pandas as pd
    ids = pd.read_parquet(a.raw / "bus_data.parquet", columns=["scenario"]).scenario.unique()
    man = make_splits(np.asarray(ids), a.out)
    print(json.dumps({k: v for k, v in man.items() if k != "subsets"}), {n: f["train"]["n"] for n, f in man["subsets"].items()})


if __name__ == "__main__":
    main()
