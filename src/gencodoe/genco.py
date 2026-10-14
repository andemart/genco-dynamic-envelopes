"""Wrapper around gridfm-graphkit for training and batched GENCO power flow inference.

GENCO is used in power flow (PF) mode: given the bus injections of one
(scenario, candidate envelope) row it predicts voltage magnitude and angle at
every bus. Branch loadings follow from the voltages (feeder.branch_flows_from_voltages),
so GencoOracle has the same signature as sweep_pf and plugs into envelopes.py.

The parquet I/O between our injections and graphkit is not implemented yet;
see docs/workplan.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .feeder import Feeder, PFResult, branch_flows_from_voltages


def train(config: Path, data_path: Path, exp_name: str = "genco-pf-feeder") -> None:
    """Train GENCO (Tiny/Small) from scratch on gridfm-datakit PF data for the feeder."""
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


def injections_to_datakit_format(feeder: Feeder, p: np.ndarray, q: np.ndarray, out_dir: Path) -> Path:
    """TODO: write rows of bus injections as gridfm-datakit PF parquet (bus/gen/branch/y_bus)
    so graphkit can run inference on them. Check BUS_COLUMNS, GEN_COLUMNS and
    BRANCH_COLUMNS in gridfm_datakit; flexible exports enter as negative load (or as
    PQ generators) at the customer buses."""
    raise NotImplementedError


def read_predictions(path: Path, n_bus: int) -> tuple[np.ndarray, np.ndarray]:
    """TODO: read graphkit's predictions parquet into (vm, va) arrays of shape (n, n_bus), va in radians."""
    raise NotImplementedError


class GencoOracle:
    """Batched GENCO PF inference behind the oracle interface (feeder, p, q) -> PFResult."""

    def __init__(self, config: Path, model_path: Path, work_dir: Path, normalizer_stats: Path | None = None):
        self.config, self.model_path, self.work_dir = config, model_path, work_dir
        self.normalizer_stats = normalizer_stats

    def __call__(self, feeder: Feeder, p: np.ndarray, q: np.ndarray) -> PFResult:
        data = injections_to_datakit_format(feeder, p, q, self.work_dir / "in")
        out = predict(self.config, self.model_path, data, self.work_dir / "out", self.normalizer_stats)
        vm, va = read_predictions(out, feeder.n_bus)
        return branch_flows_from_voltages(feeder, vm, va)
