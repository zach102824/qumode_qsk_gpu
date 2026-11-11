"""Quantum SK model on the device: couplings, diagonals, state evolution.

Coupling convention (frozen): J_ij ~ N(0, 1/N) for i<j, then symmetrized.
H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i
Initial state |+>^{otimes N}. Exact statevector energy, no shots.

The J matrix, ZZ diagonals and shell operators are built on the host with
the same formulas as the CPU Model, then uploaded once. Gate evolution
stays on the device. Do not transfer statevectors each layer.
"""
from __future__ import annotations

import numpy as np

from . import runtime, kernels
from .dtypes import REAL, CPLX, check_cplx, check_real
from .protocol import NS, param_count


def make_J(n, seed):
    n = int(n)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n), dtype=REAL)
    iu = np.triu_indices(n, 1)
    J[iu] = rng.normal(0.0, 1.0 / np.sqrt(n), size=iu[0].size)
    J = J + J.T
    return J


def nov3_groups(n):
    n = int(n)
    if n % 3 != 0:
        raise ValueError(f"N={n} is not a multiple of 3")
    return np.asarray([[i, i + n // 3, i + 2 * n // 3] for i in range(n // 3)], dtype=np.int64)


def build_diags(J, n):
    """Return (J, K, M) after the frozen jrms / mean-shell normalization, and their ZZ diags."""
    n = int(n)
    J = np.array(J, dtype=REAL, copy=True)
    iu = np.triu_indices(n, 1)
    groups = nov3_groups(n)
    K = J @ J
    np.fill_diagonal(K, 0.0)
    M = np.zeros_like(K)
    for a, C in enumerate(groups):
        for T in groups[a + 1:]:
            mmean = J[np.ix_(C, T)].mean()
            M[np.ix_(C, T)] = mmean
            M[np.ix_(T, C)] = mmean
    jrms = float(np.sqrt(np.mean(J[iu] ** 2)))
    mats = []
    diags = []
    for mat in (J, K, M):
        mat = mat.copy()
        norm = float(np.sqrt(np.mean(mat[iu] ** 2)))
        if norm:
            mat = mat * (jrms / norm)
        mats.append(mat)
        diags.append(kernels.zz_diag_host(mat, n))
    shell = kernels.shell_ops_host(groups, n)
    return dict(J=J, K=mats[1], M=mats[2], diags=diags, shell=shell, groups=groups, jrms=jrms)


class Model:
    """Device-resident model. Construct after runtime.acquire()."""

    def __init__(self, n, seed, b, J=None, compute_gs=False, upload=True, built=None):
        n = int(n)
        seed = int(seed)
        if n not in NS and n % 3 != 0:
            raise ValueError(f"N={n} not a supported multiple of 3")
        if compute_gs:
            raise NotImplementedError(
                "Full-space GPU eigensolve is not the production path. "
                "Load a stored ground state with load_gs(), or use the CPU "
                "oracle / cpu fallback entropy scan to obtain (gs, E0)."
            )
        self.n = n
        self.seed = seed
        self.b = float(b)
        self.J = np.array(J, dtype=REAL) if J is not None else make_J(n, seed)
        if self.J.shape != (n, n):
            raise ValueError("J shape")
        if built is None:
            built = build_diags(self.J, n)
        self.groups = built["groups"]
        self.diags_host = built["diags"]  # list of 3 float64 (dim,)
        self.shell_host = built["shell"]
        self.jrms = built.get("jrms")
        self.plus_host = np.ones(1 << n, dtype=CPLX) / np.sqrt(1 << n)
        self.gs_host = None
        self.e0 = None
        self.e1 = None
        self.gap = None
        self.residual = None
        self._uploaded = False
        self.diags = None
        self.shell = None
        self.plus = None
        self.gs = None
        if upload:
            self.upload()

    def upload(self):
        if not runtime.is_acquired():
            raise RuntimeError("runtime.acquire() before Model.upload()")
        self.diags = [runtime.to_device(d, dtype=REAL) for d in self.diags_host]
        self.shell = runtime.to_device(self.shell_host, dtype=REAL)
        self.plus = runtime.to_device(self.plus_host, dtype=CPLX)
        if self.gs_host is not None:
            self.gs = runtime.to_device(self.gs_host, dtype=CPLX)
        self._uploaded = True
        return self

    @property
    def J_upper(self):
        iu = np.triu_indices(self.n, 1)
        return self.J[iu].copy()

    def load_gs(self, gs, E0, gap=None, residual=None, upload=True):
        self.gs_host = np.ascontiguousarray(gs, dtype=CPLX)
        self.e0 = float(E0)
        self.gap = None if gap is None else float(gap)
        if residual is None:
            # residual is computed on the device after upload, or on the host if not yet
            self.residual = None
        else:
            self.residual = float(residual)
        if upload and runtime.is_acquired():
            self.gs = runtime.to_device(self.gs_host, dtype=CPLX)
            if residual is None:
                self.residual = self.residual_on_device()
        return self

    def residual_on_device(self):
        if self.gs is None:
            raise RuntimeError("GS not loaded")
        Hv = kernels.apply_H(self.gs.reshape(1, -1), self.diags[0], self.b, self.n)
        diff = Hv - self.e0 * self.gs.reshape(1, -1)
        nrm = kernels.norms(diff)
        return float(runtime.as_scalar(nrm) ** 0.5)

    def _prep_x(self, x, L, method):
        from .protocol import param_count
        x = runtime.as_host_array(x, dtype=REAL)
        p = param_count(method, L)
        if x.ndim == 1:
            if x.shape != (p,):
                raise AssertionError(f"x shape {x.shape} != ({p},)")
            x = x.reshape(1, p)
        elif x.ndim != 2 or x.shape[1] != p:
            raise AssertionError(f"x shape {x.shape} != (batch, {p})")
        return x

    def _broadcast_plus(self, batch, out=None):
        xp = runtime.xp()
        if out is None:
            out = xp.empty((batch, 1 << self.n), dtype=CPLX)
        out[...] = self.plus.reshape(1, -1)
        return out

    def state(self, x, L, method="nov3", out=None, work=None):
        """Evolve |+>^N under Nov3 or QAOA. x is (p,) or (batch, p).

        Host parameter vectors are always uploaded (never tested with
        hasattr(x, "device"), which is True for NumPy >= 2 ndarrays).
        Returns a device array of shape (batch, 2**N).
        """
        if not self._uploaded:
            raise RuntimeError("Model.upload() first")
        L = int(L)
        x = self._prep_x(x, L, method)
        batch = int(x.shape[0])
        xd = runtime.to_device(x, dtype=REAL)
        v = self._broadcast_plus(batch, out=out)
        if method == "nov3":
            return self._state_nov3(v, xd, L)
        if method == "qaoa":
            return self._state_qaoa(v, xd, L)
        raise ValueError(method)

    def _state_nov3(self, buf, xd, L):
        xp = runtime.xp()
        n = self.n
        batch = buf.shape[0]
        g0 = xd[:, 0].reshape(batch, 1)
        a1 = xd[:, 1].reshape(batch, 1)
        a2 = xd[:, 2].reshape(batch, 1)
        D0, D1, D2 = self.diags
        gen = D0.reshape(1, -1) + a1 * D1.reshape(1, -1) + a2 * D2.reshape(1, -1)
        phase = xp.exp((-1j) * g0 * gen)
        rows = xd[:, 3:].reshape(batch, int(L), 3)
        cur = buf
        for ell in range(int(L)):
            cur *= phase
            cur = kernels.walsh_inplace(cur, n)
            row = rows[:, ell, :]
            ang = xp.matmul(row, self.shell.T)
            cur *= xp.exp((-1j) * ang)
            cur = kernels.walsh_inplace(cur, n)
        return cur

    def _state_qaoa(self, buf, xd, L):
        xp = runtime.xp()
        n = self.n
        batch = buf.shape[0]
        D = self.diags[0]
        cur = buf
        for ell in range(int(L)):
            gamma = xd[:, 2 * ell].reshape(batch, 1)
            beta = xd[:, 2 * ell + 1]
            cur *= xp.exp((-1j) * gamma * D.reshape(1, -1))
            cur = kernels.rx_inplace(cur, beta, n)
        return cur

    def _host_energy(self, E):
        """Always return host float64: a Python float if batch==1, else a 1-d array."""
        Eh = runtime.to_host(E).reshape(-1).astype(REAL, copy=False)
        if Eh.size == 1:
            return float(Eh[0])
        return np.ascontiguousarray(Eh, dtype=REAL)

    def energy(self, v):
        if not runtime.is_device_array(v):
            v = runtime.to_device(v, dtype=CPLX)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        check_cplx(v, "v")
        E, _Hv = kernels.energy(v, self.diags[0], self.b, self.n)
        return self._host_energy(E)

    def energy_of_params(self, x, L, method="nov3"):
        """Return (E_host, psi_device). E is always host float64 (H2)."""
        v = self.state(x, L, method)
        E, _ = kernels.energy(v, self.diags[0], self.b, self.n)
        return self._host_energy(E), v

    def fidelity(self, v):
        if self.gs is None:
            raise RuntimeError("GS not loaded")
        if not runtime.is_device_array(v):
            v = runtime.to_device(v, dtype=CPLX)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        F = kernels.fidelity(v, self.gs.reshape(1, -1))
        Fh = runtime.to_host(F).reshape(-1).astype(REAL, copy=False)
        if Fh.size == 1:
            return float(Fh[0])
        return np.ascontiguousarray(Fh, dtype=REAL)

    def parity_px(self, v):
        """Re(<v| X^{otimes N} |v>) = Re(vdot(v, v[::-1]))."""
        if not runtime.is_device_array(v):
            v = runtime.to_device(v, dtype=CPLX)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        xp = runtime.xp()
        ov = xp.sum(xp.conj(v) * v[:, ::-1], axis=1)
        val = runtime.to_host(xp.real(ov)).reshape(-1)
        if val.size == 1:
            return float(val[0])
        return np.ascontiguousarray(val, dtype=REAL)
