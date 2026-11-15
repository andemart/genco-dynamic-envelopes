"""Label access for the training code of the 25-network study (docs/fourwire_scale_protocol.md §2.6,
Amendments 4 and 5).

The only label reader for training, smoke, debugging and calibration code.
- It serves train, val and calib. Any other split is refused before a path is built.
- The manifest and the label commit are fixed here; the caller supplies only the label root.
- It never lists a directory. A file is read only if its manifest name is shard_NNN.npz and its resolved
  path is exactly <root>/<split>/network_<n>/<name>, so a symbolic link or a path with ".." cannot redirect
  a read into another split's folder.
- Before a split's states are returned they are checked against the committed manifest
  (results/fourwire_scale_labels_manifest.json): the label commit, the split's state count, contiguous
  shards, and the content hash and provenance of every shard that is read.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .fourwire_scale import RESULTS, content_sha

SERVED = ("train", "val", "calib")
LABEL_COMMIT = "6ad2af67233ebb162ece7640cd047e1a4f1dd631"
MANIFEST = RESULTS / "fourwire_scale_labels_manifest.json"
SHARD_NAME = re.compile(r"shard_\d{3}\.npz")


class LabelStore:
    def __init__(self, root):
        """root: the folder holding <split>/network_N/shard_NNN.npz."""
        self.root = Path(root).resolve()
        man = json.loads(MANIFEST.read_text())
        if man["label_code_commits"] != [LABEL_COMMIT]:
            raise AssertionError(f"manifest label commits {man['label_code_commits']} != [{LABEL_COMMIT}]")
        self.states, self.shard = man["states"], man["shard_states"]
        self.units = {(u["network"], u["split"]): u for u in man["splits"] if u["split"] in SERVED}

    def _path(self, network: int, split: str, name: str) -> Path:
        path = self.root / split / f"network_{int(network)}" / name
        if not SHARD_NAME.fullmatch(name) or path.resolve() != path:
            raise AssertionError(f"{split}/network_{network}/{name} is not a plain shard file under the label root")
        return path

    def load(self, network: int, split: str, n_states: int | None = None) -> dict:
        """The first n_states states (default: all) of one network and split, as stored (FP64)."""
        if split not in SERVED:
            raise PermissionError(f"split {split!r} is not served to training code (only {', '.join(SERVED)})")
        u = self.units[(network, split)]
        total = self.states[split]
        n = total if n_states is None else n_states
        if not 0 < n <= total:
            raise ValueError(f"n_states {n} outside 1..{total}")
        clean = all(p["code_commit"] == LABEL_COMMIT and not p["src_dirty"] for p in u["provenance"])
        if (not clean or u["states"] != total or u["opendss"]["states"] != total
                or [s["first_state"] for s in u["shards"]] != list(range(0, total, self.shard))
                or sum(s["states"] for s in u["shards"]) != total):
            raise AssertionError(f"manifest entry of network {network} {split} is not the frozen label set")
        parts = []
        for s in u["shards"]:
            if s["first_state"] >= n:
                break
            path = self._path(network, split, s["file"])
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files if k != "provenance"}
                prov = json.loads(str(z["provenance"]))
            states = np.arange(s["first_state"], s["first_state"] + s["states"])
            if (content_sha(arrays) != s["content_sha256"] or prov["code_commit"] != LABEL_COMMIT
                    or prov["src_dirty"] or not np.array_equal(arrays["state"], states)):
                raise AssertionError(f"{split}/network_{network}/{s['file']} differs from the committed manifest")
            parts.append(arrays)
        return {k: np.concatenate([p[k] for p in parts])[:n] for k in parts[0]}
