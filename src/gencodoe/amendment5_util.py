"""Amendment 5 helpers shared by the pre-check and the warm-start study."""
from __future__ import annotations

import shutil
from pathlib import Path

import torch


def run_dir_from_hub(hub: Path, out: Path) -> Path:
    """graphkit's run layout from the Hub files; last.pt (a Lightning checkpoint) -> plain state dict."""
    (out / "artifacts/model").mkdir(parents=True, exist_ok=True)
    (out / "artifacts/stats").mkdir(parents=True, exist_ok=True)
    state = torch.load(hub / "last.pt", map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = {k: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    state = {k.removeprefix("_orig_mod.").removeprefix("model.").removeprefix("_orig_mod."): v for k, v in state.items()}
    torch.save(state, out / "artifacts/model/best_model_state_dict.pt")
    shutil.copy(hub / "normalizer_stats.pt", out / "artifacts/stats/normalizer_stats.pt")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert"])
    ap.add_argument("--hub", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    # graphkit `finetune` loads the state dict into the Lightning task, so keys keep the "model." prefix
    # (the Hub last.pt has the task's 416 keys, some under "model._orig_mod." from torch.compile).
    state = torch.load(a.hub / "last.pt", map_location="cpu", weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state
    state = {k.replace("model._orig_mod.", "model."): v for k, v in state.items()}
    state = {(k if k.startswith("model.") else "model." + k): v for k, v in state.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.out)
    print(f"wrote {a.out} ({len(state)} tensors, task-level keys)")


if __name__ == "__main__":
    main()
