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
