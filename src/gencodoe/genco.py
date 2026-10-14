"""GENCO power flow behind the oracle interface, run in-process with gridfm-graphkit.

GENCO is used in power flow (PF) mode: given the bus injections of one
(scenario, candidate envelope) row it predicts voltage magnitude and angle at
every bus. Branch flows then follow from the voltages
(feeder.branch_flows_from_voltages), so GencoOracle has the same signature as
sweep_pf and plugs straight into envelopes.py.

Rather than round-tripping parquet files through `gridfm_graphkit predict`, the
oracle builds the feeder's graph once from a datakit template scenario
(normalised and PF-masked exactly as in training), batches copies of it, and
overwrites only the per-row inputs: Pd and Qd at every bus (and the slack Vm,
if the PF mask leaves it visible; gridfm-graphkit 0.9 masks it, so the model
assumes the slack voltage it was trained at).

Training: scripts/make_training_data.py, then
    gridfm_graphkit train --config configs/genco_pf_case33bw.yaml --data_path ~/gencodoe_work/data_doe \
        --log_dir ~/gencodoe_work/mlruns --exp_name genco-pf-case33bw
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from gridfm_graphkit.datasets.globals import PD_H, QD_H, REF_H, VA_OUT, VM_H, VM_OUT
from gridfm_graphkit.datasets.graph_builder import build_hetero_data
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer

from .feeder import Feeder, PFResult, branch_flows_from_injections, branch_flows_from_voltages


def _read_scenario(raw_dir: Path, name: str, scenario: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir / name)
    return df[df.scenario == scenario].drop(columns=["scenario_partition"], errors="ignore").reset_index(drop=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class GencoOracle:
    """Batched GENCO PF inference: (feeder, p, q) -> PFResult.

    run_dir: an MLflow run directory from `gridfm_graphkit train` (holds
    artifacts/model/best_model_state_dict.pt and artifacts/stats/normalizer_stats.pt).
    raw_dir: datakit raw directory of the same grid; scenario 0 is the template.
    flows: "voltages" computes branch flows from predicted voltage differences
    (pure GENCO); "injections" uses one backward sweep of the injections with the
    predicted voltages, which is far less sensitive to voltage error on short lines.
    """

    def __init__(self, config: str | Path, run_dir: str | Path, raw_dir: str | Path, network: str = "case33bw",
                 device: str | None = None, chunk: int = 4096, flows: str = "voltages"):
        if flows not in ("voltages", "injections"):
            raise ValueError(flows)
        self.flows = flows
        with open(config) as f:
            self.args = NestedNamespace(**yaml.safe_load(f))
        run_dir, raw_dir = Path(run_dir), Path(raw_dir)
        self.device = device or default_device()
        self.chunk = chunk

        self.model = load_model(self.args)
        state = torch.load(run_dir / "artifacts/model/best_model_state_dict.pt", map_location="cpu")
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        self.model.eval().to(self.device)

        self.normalizer = load_normalizer(self.args)
        stats = torch.load(run_dir / "artifacts/stats/normalizer_stats.pt", map_location="cpu")
        self.normalizer.fit_from_dict(stats[network])

        bus = _read_scenario(raw_dir, "bus_data.parquet").sort_values("bus").reset_index(drop=True)
        gen = _read_scenario(raw_dir, "gen_data.parquet")
        branch = _read_scenario(raw_dir, "branch_data.parquet")
        bus = bus.merge(gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index(), on="bus", how="left").fillna(0)
        graph = build_hetero_data(bus, gen, branch)
        self.normalizer.transform(graph)
        self.template = get_task_transforms(self.args)(graph)
        self.n_bus = len(bus)
        self.ref = self.template["bus"].x[:, REF_H].numpy() == 1
        self.ref_vm_visible = not bool(self.template.mask_dict["bus"][self.ref, VM_H].any())
        self._batches: dict[int, Batch] = {}

    def _batch(self, k: int) -> Batch:
        if k not in self._batches:
            self._batches[k] = Batch.from_data_list([self.template] * k).to(self.device)
        return self._batches[k]

    @torch.inference_mode()
    def predict(self, pd_mw: np.ndarray, qd_mvar: np.ndarray, v_slack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw interface: loads in MW/MVAr (n, n_bus), slack Vm (n,). Returns (vm, va[rad])."""
        n, out_vm, out_va = len(pd_mw), [], []
        if n == 1:   # the model squeezes per-graph tensors; a batch of one graph breaks it
            vm, va = self.predict(np.repeat(pd_mw, 2, 0), np.repeat(qd_mvar, 2, 0), np.repeat(v_slack, 2))
            return vm[:1], va[:1]
        for s in range(0, n, self.chunk):
            k = min(self.chunk, n - s)
            batch = self._batch(k)
            x = batch["bus"].x.clone()
            x[:, PD_H] = torch.as_tensor(pd_mw[s:s + k].ravel() / self.normalizer.baseMVA, dtype=x.dtype, device=x.device)
            x[:, QD_H] = torch.as_tensor(qd_mvar[s:s + k].ravel() / self.normalizer.baseMVA, dtype=x.dtype, device=x.device)
            if self.ref_vm_visible:
                ref = torch.as_tensor(np.tile(self.ref, k), device=x.device)
                x[ref, VM_H] = torch.as_tensor(v_slack[s:s + k], dtype=x.dtype, device=x.device)
            saved = batch["bus"].x
            batch["bus"].x = x
            out = self.model(batch)["bus"]
            batch["bus"].x = saved
            out_vm.append(out[:, VM_OUT].float().cpu().numpy().reshape(k, self.n_bus))
            out_va.append(out[:, VA_OUT].float().cpu().numpy().reshape(k, self.n_bus))
        vm, va = np.vstack(out_vm), np.vstack(out_va)
        vm[:, self.ref] = v_slack[:, None]   # slack is an input; the model's echo of it is not used
        va[:, self.ref] = 0.0
        return vm, va

    def __call__(self, feeder: Feeder, p: np.ndarray, q: np.ndarray, v_slack: np.ndarray | None = None) -> PFResult:
        p, q = np.atleast_2d(p), np.atleast_2d(q)
        vs = np.full(len(p), feeder.v_slack) if v_slack is None else np.asarray(v_slack, float)
        vm, va = self.predict(-p * feeder.base_mva, -q * feeder.base_mva, vs)
        if self.flows == "injections":
            return branch_flows_from_injections(feeder, p, q, vm, va)
        return branch_flows_from_voltages(feeder, vm, va)


def latest_run(log_dir: str | Path = "mlruns", exp_name: str = "genco-pf-case33bw") -> Path:
    """Most recent MLflow run of an experiment that has a saved best model."""
    log_dir = Path(log_dir)
    for exp in log_dir.iterdir():
        meta = exp / "meta.yaml"
        if meta.exists() and yaml.safe_load(meta.read_text()).get("name") == exp_name:
            runs = [r for r in exp.iterdir() if (r / "artifacts/model/best_model_state_dict.pt").exists()]
            if runs:
                return max(runs, key=os.path.getmtime)
    raise FileNotFoundError(f"no trained run for experiment {exp_name!r} in {log_dir}")
