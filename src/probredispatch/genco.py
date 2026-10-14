"""Wrapper around gridfm-graphkit for training and batched GENCO OPF inference.

Not yet implemented: this module documents the intended interface and the CLI
calls it will wrap. See docs/workplan.md, days 3-6.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def train(config: Path, data_path: Path, exp_name: str = "genco-opf") -> None:
    """Train GENCO (Tiny/Small) from scratch on gridfm-datakit OPF data."""
    subprocess.run(
        ["gridfm_graphkit", "train", "--config", str(config), "--data_path", str(data_path), "--exp_name", exp_name],
        check=True,
    )


def predict(config: Path, model_path: Path, data_path: Path, output_path: Path, normalizer_stats: Path | None = None) -> Path:
    """Run inference; graphkit writes <grid_name>_predictions.parquet to output_path."""
    cmd = ["gridfm_graphkit", "predict", "--config", str(config), "--model_path", str(model_path),
           "--data_path", str(data_path), "--output_path", str(output_path)]
    if normalizer_stats:
        cmd += ["--normalizer_stats", str(normalizer_stats)]
    subprocess.run(cmd, check=True)
    return output_path


def scenarios_to_datakit_format(*args, **kwargs):
    """TODO: write sampled scenarios as gridfm-datakit parquet (bus/gen/branch/y_bus)
    so graphkit can run inference on them. Requires checking BUS_COLUMNS,
    GEN_COLUMNS and BRANCH_COLUMNS in gridfm_datakit."""
    raise NotImplementedError


def line_loading_from_predictions(*args, **kwargs):
    """TODO: compute branch flows from predicted V, theta with the branch admittance
    model and divide by thermal limits (rate_a)."""
    raise NotImplementedError
