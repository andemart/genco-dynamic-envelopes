"""GENCO (gridfm-graphkit) training and batched power-flow inference.

GENCO runs in power flow (PF) mode: given the bus injections of one state it
predicts voltage magnitude (pu) and angle at every bus. Branch loadings follow
from the predicted voltages and the injections by Kirchhoff's current law
(feeder.branch_flows_kcl), so GencoOracle
has the same signature as sweep_pf and plugs into envelopes.py.

Inference runs in-process and reuses graphkit's own pipeline, in the same order
as its training dataset: build_hetero_data -> normalizer.transform -> PF masking
transforms -> model. Training goes through graphkit's CLI
(python -m gencodoe.graphkit_cli train ...), which also registers the LV model
variant in genco_model.py.

Units: datakit bus tables hold Pd/Qd in MW/MVAr (positive = consumption) and Va
in degrees; graphkit predicts Va in radians. Our Feeder arrays are pu on the
feeder's base_mva with positive = export.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .feeder import Feeder, PFResult, branch_flows_kcl

NETWORK = "enwl1_fluvius"


def train(config: Path, data_path: Path, exp_name: str = "genco-pf-feeder", log_dir: Path = Path("mlruns")) -> None:
    """Train GENCO with graphkit's CLI (with the LV variant registered)."""
    subprocess.run([sys.executable, "-m", "gencodoe.graphkit_cli", "train", "--config", str(config),
                    "--data_path", str(data_path), "--exp_name", exp_name, "--log_dir", str(log_dir)], check=True)


# --------------------------------------------------------------------------- data frames

def read_raw(raw_dir: Path, scenarios=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """datakit bus/gen/branch tables, with per-bus generator Q limits merged in
    exactly as graphkit's dataset does (sum over generators, 0 where none)."""
    filt = None if scenarios is None else [("scenario", "in", list(map(int, scenarios)))]
    bus = pd.read_parquet(Path(raw_dir) / "bus_data.parquet", filters=filt)
    gen = pd.read_parquet(Path(raw_dir) / "gen_data.parquet", filters=filt)
    branch = pd.read_parquet(Path(raw_dir) / "branch_data.parquet", filters=filt)
    for df in (bus, gen, branch):
        if "scenario_partition" in df:
            df["scenario_partition"] = df["scenario_partition"].astype(int)
    agg = gen.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus = bus.merge(agg, on=["scenario", "bus"], how="left").fillna({"min_q_mvar": 0.0, "max_q_mvar": 0.0})
    return (bus.sort_values(["scenario", "bus"], kind="stable").reset_index(drop=True),
            gen.sort_values(["scenario", "idx"], kind="stable").reset_index(drop=True),
            branch.sort_values(["scenario", "idx"], kind="stable").reset_index(drop=True))


def injections_to_datakit_format(template: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], p: np.ndarray,
                                 q: np.ndarray, base_mva: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bus/gen/branch tables for rows of net injections, ready for GencoModel.

    template: one solved datakit scenario (its static columns are reused).
    p, q: (n, n_bus) net injections in pu on base_mva, positive = export (the
    Feeder convention); they become Pd = -p * base_mva MW at every bus.
    In PF mode the model only reads Pd/Qd, the slack Vm/Va and static data;
    the remaining target columns are masked, so they are left as in the template.
    """
    bus0, gen0, br0 = (df[df.scenario == df.scenario.iloc[0]] for df in template)
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    n, n_bus = p.shape
    if n_bus != len(bus0):
        raise ValueError(f"injections have {n_bus} buses, template has {len(bus0)}")
    sc = np.repeat(np.arange(n), n_bus)
    bus = pd.concat([bus0] * n, ignore_index=True)
    bus["scenario"] = sc
    bus["Pd"] = (-p * base_mva).ravel()
    bus["Qd"] = (-q * base_mva).ravel()

    def tile(df):
        out = pd.concat([df] * n, ignore_index=True)
        out["scenario"] = np.repeat(np.arange(n), len(df))
        return out

    return bus, tile(gen0), tile(br0)


def frames_to_graphs(bus: pd.DataFrame, gen: pd.DataFrame, branch: pd.DataFrame) -> list:
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data

    bg, gg, rg = (df.groupby("scenario", sort=True) for df in (bus, gen, branch))
    return [build_hetero_data(b.sort_values("bus"), gg.get_group(s), rg.get_group(s), int(s)) for s, b in bg]


def graphs_from_injections(template_graph, p: np.ndarray, q: np.ndarray, base_mva: float = 1.0) -> list:
    """Fast path: copies of one built graph with Pd/Qd replaced (same result as
    frames_to_graphs(injections_to_datakit_format(...)), without pandas)."""
    import torch
    from gridfm_graphkit.datasets.globals import PD_H, QD_H

    p, q = np.atleast_2d(p), np.atleast_2d(q)
    pd_ = torch.tensor(-p * base_mva, dtype=torch.float)
    qd_ = torch.tensor(-q * base_mva, dtype=torch.float)
    out = []
    for i in range(p.shape[0]):
        g = template_graph.clone()
        for store in (g["bus"].x, g["bus"].y):
            store[:, PD_H] = pd_[i]
            store[:, QD_H] = qd_[i]
        g["scenario_id"] = torch.tensor([i], dtype=torch.long)
        out.append(g)
    return out


def read_predictions(path: Path, n_bus: int) -> tuple[np.ndarray, np.ndarray]:
    """graphkit `predict` output parquet -> (vm, va) of shape (n, n_bus), va in radians."""
    df = pd.read_parquet(path, columns=["scenario", "bus", "Vm_pred", "Va_pred"]).sort_values(["scenario", "bus"])
    n = df.scenario.nunique()
    if len(df) != n * n_bus:
        raise ValueError(f"expected {n} x {n_bus} rows, got {len(df)}")
    return df.Vm_pred.to_numpy().reshape(n, n_bus), df.Va_pred.to_numpy().reshape(n, n_bus)


# --------------------------------------------------------------------------- model

class GencoModel:
    """A trained graphkit PF checkpoint for batched in-process inference."""

    def __init__(self, config: Path, model_path: Path, normalizer_stats: Path, device: str = "cpu",
                 network: str = NETWORK):
        import torch
        import yaml
        from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
        from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model

        from . import genco_model  # noqa: F401  (registers the LV variant)

        self.args = NestedNamespace(**yaml.safe_load(Path(config).read_text()))
        self.device = torch.device(device)
        self.model = load_model(self.args)
        state = torch.load(model_path, map_location="cpu")
        state = {k.removeprefix("_orig_mod.").removeprefix("model."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        stats = torch.load(normalizer_stats, map_location="cpu")
        self.normalizer = HeteroDataMVANormalizer(self.args)
        self.normalizer.fit_from_dict(stats[network] if network in stats else stats)
        self.transform = get_task_transforms(self.args)

    def predict_graphs(self, graphs: list, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """(vm pu, va rad), each (n_graphs, n_bus), in the order of `graphs`."""
        import torch
        from gridfm_graphkit.datasets.globals import VA_OUT, VM_OUT
        from torch_geometric.data import Batch

        vm, va = [], []
        with torch.no_grad():
            for i in range(0, len(graphs), batch_size):
                chunk = []
                for g in graphs[i:i + batch_size]:
                    g = g.clone()
                    self.normalizer.transform(g)
                    chunk.append(self.transform(g))
                k = len(chunk)
                if k == 1:   # graphkit squeezes the gen output; a lone generator would become 0-d
                    chunk = chunk * 2
                batch = Batch.from_data_list(chunk).to(self.device)
                out = self.model(batch)["bus"]
                n_bus = chunk[0]["bus"].num_nodes
                vm.append(out[:, VM_OUT].reshape(len(chunk), n_bus)[:k].cpu().numpy())
                va.append(out[:, VA_OUT].reshape(len(chunk), n_bus)[:k].cpu().numpy())
        return np.concatenate(vm).astype(float), np.concatenate(va).astype(float)

    def predict_frames(self, bus, gen, branch, batch_size: int = 256):
        return self.predict_graphs(frames_to_graphs(bus, gen, branch), batch_size)


class GencoOracle:
    """Batched GENCO PF inference behind the oracle interface (feeder, p, q) -> PFResult."""

    def __init__(self, model: GencoModel, template: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                 batch_size: int = 512):
        self.model, self.batch_size = model, batch_size
        self.template_graph = frames_to_graphs(*(df[df.scenario == df.scenario.iloc[0]] for df in template))[0]
        self.calls = 0   # number of power flows evaluated

    def __call__(self, feeder: Feeder, p: np.ndarray, q: np.ndarray) -> PFResult:
        graphs = graphs_from_injections(self.template_graph, p, q, feeder.base_mva)
        vm, va = self.model.predict_graphs(graphs, batch_size=self.batch_size)
        self.calls += len(graphs)
        return branch_flows_kcl(feeder, vm, va, np.atleast_2d(p), np.atleast_2d(q))
