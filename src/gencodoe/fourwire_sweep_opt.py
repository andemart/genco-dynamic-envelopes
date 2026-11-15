"""Optimised FP64 four-wire backward/forward sweeps (docs/fourwire_scale_protocol.md §3, Amendment 3).

The algorithm is `fourwire_net.sweep` (U3): constant-power phase-to-neutral customers, KCL branch currents,
KVL forward pass from a Thevenin root, flat start, convergence when the largest voltage update of an
iteration is below 1e-8 pu, at most 100 iterations. Two implementations, both complex128 and both with
per-state convergence (each state returns the iterate of its own convergence iteration, exactly what the
one-state NumPy reference returns):

- `NumbaSweep`: parallel over states; within a state, buses are processed level by level (tree depth)
  in compiled loops.
- `TorchSweep`: tensors batched over states; one tensor operation per tree depth in the backward and the
  forward pass; converged states leave the working batch. Runs on CPU or CUDA.

Both return node voltages (kV), branch currents from the final voltages (kA), iteration counts, convergence
flags and the four screening quantities of `fourwire_model.summaries` (min V_pn, max V_pn, max phase
loading of the non-root branches, LV-root transformer-loading proxy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fourwire_net import V_PN_NOM, FourWireNet, Thevenin

TOL, MAX_IT = 1e-8, 100
TR_KVA = 800.0


@dataclass
class LevelNet:
    """A FourWireNet with buses renumbered level by level (root first), so parents precede children and
    each tree depth is a contiguous slice."""
    perm: np.ndarray          # (n,) level position -> original bus index
    inv: np.ndarray           # (n,) original bus index -> level position
    parent: np.ndarray        # (n,) level position of the parent, -1 at the root
    level_ptr: np.ndarray     # (depth + 2,) start of each level; level 0 is the root
    z: np.ndarray             # (n, 4, 4) complex128 ohm, branch into each bus
    rating_ka: np.ndarray     # (n,) inf at the root
    cust_bus: np.ndarray      # (n_c,) level position
    cust_phase: np.ndarray    # (n_c,)
    e: np.ndarray             # (3,) complex128 kV
    zth: np.ndarray           # (3, 3) complex128 ohm

    @property
    def n_bus(self) -> int:
        return len(self.perm)


def level_net(net: FourWireNet, th: Thevenin) -> LevelNet:
    depth = np.asarray(net.depth)
    perm = np.argsort(depth, kind="stable")
    if perm[0] != 0 or depth[0] != 0:
        raise AssertionError("the root must be bus 0 at depth 0")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    parent = np.where(net.parent[perm] >= 0, inv[np.maximum(net.parent[perm], 0)], -1)
    if not (depth[net.parent[1:]] == depth[1:] - 1).all() or not (parent[1:] < np.arange(1, len(perm))).all():
        raise AssertionError("depth is not the tree depth")
    level_ptr = np.concatenate([[0], np.cumsum(np.bincount(depth))])
    return LevelNet(perm=perm, inv=inv, parent=parent.astype(np.int64), level_ptr=level_ptr.astype(np.int64),
                    z=np.ascontiguousarray(net.z[perm], dtype=np.complex128),
                    rating_ka=np.asarray(net.rating_a[perm], float) / 1e3,
                    cust_bus=inv[net.customer_bus].astype(np.int64), cust_phase=net.customer_phase.astype(np.int64),
                    e=np.asarray(th.e, np.complex128), zth=np.ascontiguousarray(th.zth, dtype=np.complex128))


# ------------------------------------------------------------------------------------------- Numba
from numba import njit, prange  # noqa: E402


@njit(cache=True)
def _currents(v, s_row, parent, cust_bus, cust_phase, cur):
    """Branch current into each bus (kA): customer injections summed over each subtree."""
    n = parent.shape[0]
    for i in range(n):
        for k in range(4):
            cur[i, k] = 0.0
    for c in range(cust_bus.shape[0]):
        bus, ph = cust_bus[c], cust_phase[c]
        vpn = v[bus, ph] - v[bus, 3]
        if abs(vpn) <= 1e-9:
            vpn = 1e-9 + 0j
        ic = np.conj(s_row[c] / 1e3 / vpn)
        cur[bus, ph] += ic
        cur[bus, 3] -= ic
    for i in range(n - 1, 0, -1):                           # deepest level first: children before parents
        p = parent[i]
        for k in range(4):
            cur[p, k] += cur[i, k]


@njit(parallel=True, cache=True)
def _solve_numba(s, parent, z, rating_ka, cust_bus, cust_phase, e, zth, tol, max_it, v_pn_nom, tr_kva,
                 v_out, cur_out, it_out, conv_out, summ_out, keep_state):
    n = parent.shape[0]
    for b in prange(s.shape[0]):
        v = np.empty((n, 4), np.complex128)
        vn = np.empty((n, 4), np.complex128)
        cur = np.empty((n, 4), np.complex128)
        for i in range(n):
            v[i, 0] = e[0]
            v[i, 1] = e[1]
            v[i, 2] = e[2]
            v[i, 3] = 0.0
        it = 0
        conv = False
        while it < max_it:
            it += 1
            _currents(v, s[b], parent, cust_bus, cust_phase, cur)
            delta = 0.0
            for a in range(3):
                vn[0, a] = e[a] - (zth[a, 0] * cur[0, 0] + zth[a, 1] * cur[0, 1] + zth[a, 2] * cur[0, 2])
            vn[0, 3] = 0.0
            for k in range(4):
                delta = max(delta, abs(vn[0, k] - v[0, k]))
            for i in range(1, n):                           # level by level: parents before children
                p = parent[i]
                for a in range(4):
                    vn[i, a] = vn[p, a] - (z[i, a, 0] * cur[i, 0] + z[i, a, 1] * cur[i, 1]
                                           + z[i, a, 2] * cur[i, 2] + z[i, a, 3] * cur[i, 3])
                    delta = max(delta, abs(vn[i, a] - v[i, a]))
            tmp = v
            v = vn
            vn = tmp
            if delta / v_pn_nom < tol:
                conv = True
                break
        _currents(v, s[b], parent, cust_bus, cust_phase, cur)   # currents from the final voltages
        it_out[b] = it
        conv_out[b] = conv
        vmin = np.inf
        vmax = 0.0
        for c in range(cust_bus.shape[0]):
            m = abs(v[cust_bus[c], cust_phase[c]] - v[cust_bus[c], 3]) / v_pn_nom
            vmin = min(vmin, m)
            vmax = max(vmax, m)
        load = 0.0
        for i in range(1, n):
            for k in range(3):
                load = max(load, abs(cur[i, k]) / rating_ka[i])
        s_root = 0.0
        for k in range(3):
            s_root += abs(v[0, k] * np.conj(cur[0, k]))
        summ_out[b, 0] = vmin
        summ_out[b, 1] = vmax
        summ_out[b, 2] = load
        summ_out[b, 3] = s_root * 1e3 / tr_kva
        if keep_state:
            v_out[b, :, :] = v
            cur_out[b, :, :] = cur


class NumbaSweep:
    def __init__(self, net: FourWireNet, th: Thevenin):
        self.ln = level_net(net, th)

    def solve(self, s_kva: np.ndarray, return_state: bool = True) -> dict:
        """s_kva (B, n_c) complex customer consumption in kVA. Arrays are returned in the original bus order."""
        ln = self.ln
        s = np.ascontiguousarray(s_kva, dtype=np.complex128)
        b = len(s)
        shape = (b, ln.n_bus, 4) if return_state else (1, 1, 4)
        v, cur = np.empty(shape, np.complex128), np.empty(shape, np.complex128)
        it, conv, summ = np.zeros(b, np.int64), np.zeros(b, np.bool_), np.zeros((b, 4))
        _solve_numba(s, ln.parent, ln.z, ln.rating_ka, ln.cust_bus, ln.cust_phase, ln.e, ln.zth, TOL, MAX_IT,
                     V_PN_NOM, TR_KVA, v, cur, it, conv, summ, return_state)
        out = {"iterations": it, "converged": conv, "summary": summ}
        if return_state:
            out["V"], out["I"] = v[:, ln.inv], cur[:, ln.inv]
        return out


# ------------------------------------------------------------------------------------------- PyTorch
class TorchSweep:
    def __init__(self, net: FourWireNet, th: Thevenin, device="cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        ln = self.ln = level_net(net, th)
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)   # noqa: E731
        c128 = torch.complex128
        self.levels = [(int(a), int(b)) for a, b in zip(ln.level_ptr[1:-1], ln.level_ptr[2:])]
        self.prev = [(int(a), int(b)) for a, b in zip(ln.level_ptr[:-2], ln.level_ptr[1:-1])]   # parents' level
        self.par = [t(ln.parent[a:b], torch.long) for a, b in self.levels]
        self.par_rel = [p - pa for p, (pa, _) in zip(self.par, self.prev)]     # index within the parents' level
        self.z = t(ln.z, c128)
        self.rating = t(ln.rating_ka, torch.float64)
        self.cb, self.cp = t(ln.cust_bus, torch.long), t(ln.cust_phase, torch.long)
        self.idx_ph, self.idx_n = self.cb * 4 + self.cp, self.cb * 4 + 3
        self.e, self.zth_t = t(ln.e, c128), t(ln.zth.T.copy(), c128)
        self.inv = t(ln.inv, torch.long)
        self.n = ln.n_bus

    def _vpn(self, v):
        vb = v[:, self.cb]
        return vb.gather(2, self.cp.view(1, -1, 1).expand(v.shape[0], -1, 1)).squeeze(2) - vb[..., 3]

    def _currents(self, v, s):
        torch = self.torch
        vpn = self._vpn(v)
        ic = torch.conj(s / 1e3 / torch.where(vpn.abs() > 1e-9, vpn, torch.full_like(vpn, 1e-9)))
        cur = torch.zeros(v.shape[0], self.n * 4, dtype=v.dtype, device=v.device)
        cur.index_add_(1, self.idx_ph, ic)
        cur.index_add_(1, self.idx_n, -ic)
        cur = cur.view(v.shape[0], self.n, 4)
        for (a, b), (pa, pb), par in zip(reversed(self.levels), reversed(self.prev), reversed(self.par_rel)):
            cur[:, pa:pb].index_add_(1, par, cur[:, a:b])                          # deepest level first
        return cur

    def _forward(self, cur):
        torch = self.torch
        drop = torch.matmul(self.z, cur.permute(1, 2, 0)).permute(2, 0, 1)       # (B, n, 4): z[i] @ cur[i]
        vn = torch.empty_like(cur)
        vn[:, 0, :3] = self.e - cur[:, 0, :3] @ self.zth_t
        vn[:, 0, 3] = 0.0
        for (a, b), par in zip(self.levels, self.par):
            vn[:, a:b] = vn[:, par] - drop[:, a:b]
        return vn

    def solve(self, s_kva, return_state: bool = True, to_host: bool = True) -> dict:
        """s_kva (B, n_c) complex kVA: a NumPy array (moved to the device here) or a tensor on the device.
        to_host: return NumPy arrays (original bus order); otherwise tensors on the device."""
        torch = self.torch
        with torch.no_grad():
            s = s_all = torch.as_tensor(s_kva, dtype=torch.complex128, device=self.device)
            b = s.shape[0]
            v_out = torch.empty(b, self.n, 4, dtype=torch.complex128, device=self.device)
            it_out = torch.full((b,), MAX_IT, dtype=torch.long, device=self.device)
            conv = torch.zeros(b, dtype=torch.bool, device=self.device)
            idx = torch.arange(b, device=self.device)
            v = torch.zeros(b, self.n, 4, dtype=torch.complex128, device=self.device)
            v[:, :, :3] = self.e
            for it in range(1, MAX_IT + 1):
                vn = self._forward(self._currents(v, s))
                done = (vn - v).abs().amax(dim=(1, 2)) / V_PN_NOM < TOL
                if it == MAX_IT:
                    v_out[idx] = vn
                    conv[idx] = done
                    break
                if done.any():
                    fin = idx[done]
                    v_out[fin], it_out[fin], conv[fin] = vn[done], it, True
                    keep = ~done
                    v, s, idx = vn[keep], s[keep], idx[keep]
                    if idx.numel() == 0:
                        break
                else:
                    v = vn
            cur = self._currents(v_out, s_all)
            vpn = self._vpn(v_out).abs() / V_PN_NOM
            load = (cur[:, 1:, :3].abs() / self.rating[1:].view(1, -1, 1)).amax(dim=(1, 2))
            s_root = (v_out[:, 0, :3] * torch.conj(cur[:, 0, :3])).abs().sum(1) * 1e3 / TR_KVA
            summ = torch.stack([vpn.amin(1), vpn.amax(1), load, s_root], 1)
            out = {"iterations": it_out, "converged": conv, "summary": summ}
            if return_state:
                out["V"], out["I"] = v_out[:, self.inv], cur[:, self.inv]
            return {k: x.cpu().numpy() for k, x in out.items()} if to_host else out
