"""Label access for the training code of the 25-network study (docs/fourwire_scale_protocol.md §2.6,
Amendment 4).

The only label reader for training, smoke, debugging and calibration code.
- It serves train, val and calib. The test split is refused before any path is built, and no argument
  allows it.
- It never lists a directory: file names come from the committed manifest.
- The label root has no default; the caller passes it from its configuration.
- Before a split's states are returned they are checked against the committed manifest
  (results/fourwire_scale_labels_manifest.json): the label commit, the split's state count, contiguous
  shards, and the content hash and provenance of every shard that is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fourwire_scale import RESULTS, content_sha

SERVED = ("train", "val", "calib")
LABEL_COMMIT = "6ad2af67233ebb162ece7640cd047e1a4f1dd631"
MANIFEST = RESULTS / "fourwire_scale_labels_manifest.json"


class LabelStore:
    def __init__(self, root, manifest: Path = MANIFEST, label_commit: str = LABEL_COMMIT):
        """root: the folder holding <split>/network_N/shard_KKK.npz."""
        self.root, self.label_commit = Path(root), label_commit
        man = json.loads(Path(manifest).read_text())
        if man["label_code_commits"] != [label_commit]:
            raise AssertionError(f"manifest label commits {man['label_code_commits']} != [{label_commit}]")
        self.states, self.shard = man["states"], man["shard_states"]
        self.units = {(u["network"], u["split"]): u for u in man["splits"] if u["split"] in SERVED}

    def load(self, network: int, split: str, n_states: int | None = None) -> dict:
        """The first n_states states (default: all) of one network and split, as stored (FP64)."""
        if split not in SERVED:
            raise PermissionError(f"split {split!r} is not served to training code (only {', '.join(SERVED)})")
        u = self.units[(network, split)]
        total = self.states[split]
        n = total if n_states is None else n_states
        if not 0 < n <= total:
            raise ValueError(f"n_states {n} outside 1..{total}")
        clean = all(p["code_commit"] == self.label_commit and not p["src_dirty"] for p in u["provenance"])
        if (not clean or u["states"] != total or u["opendss"]["states"] != total
                or [s["first_state"] for s in u["shards"]] != list(range(0, total, self.shard))
                or sum(s["states"] for s in u["shards"]) != total):
            raise AssertionError(f"manifest entry of network {network} {split} is not the frozen label set")
        parts = []
        for s in u["shards"]:
            if s["first_state"] >= n:
                break
            name = f"{split}/network_{network}/{s['file']}"
            with np.load(self.root / name) as z:
                arrays = {k: z[k] for k in z.files if k != "provenance"}
                prov = json.loads(str(z["provenance"]))
            states = np.arange(s["first_state"], s["first_state"] + s["states"])
            if (content_sha(arrays) != s["content_sha256"] or prov["code_commit"] != self.label_commit
                    or prov["src_dirty"] or not np.array_equal(arrays["state"], states)):
                raise AssertionError(f"{name} differs from the committed manifest")
            parts.append(arrays)
        return {k: np.concatenate([p[k] for p in parts])[:n] for k in parts[0]}
