"""Reverse-mode analytic dE/dθ for Nov3 and QAOA, batched on device.

Matches cpu_oracle/src/hamiltonian.py::{nov3,qaoa}_energy_grad.

Convention
----------
E = <ψ|H|ψ> with un-normalized algebra (states are unitary images of |+>^N;
we check ||ψ||=1 but we do not renormalize or invert a norm factor).
The Euclidean (real) gradient is

    ∂E/∂θ = 2 Re( λ† (∂U/∂θ) ξ )

Shared Nov3 parameters (γ0, α1, α2) appear in every layer and are **summed**.

Buffer rules (C1, C2)
---------------------
- Diagonal multiplies on λ are in-place so λ never leaves {lam_a, lam_b}.
- Walsh and RX clobber their source. The backward `vA = vin * phase` is
  kept in its own buffer; Walsh/RX run on a copy (fwd). vA is then used
  for the shared-parameter / γ gradients.
- sph is cached in the forward pass and reused backward (no recompute).

Reductions (E4)
---------------
For real weights w, 2 Re(λ† (-i w) v) = 2 w·Im(conj(λ) v). Shell-angle
gradients use one Im(conj(λ)·vC) and a matmul with shell. Shared-parameter
gradients use Im(conj(λ)·vA) against [gen, γ0 D1, γ0 D2]. Reduction
rounding may differ from the per-component CPU vdots at ULP level.

Host download (E5)
------------------
norm, E and g are packed into one device-to-host transfer.
"""
from __future__ import annotations

import numpy as np

from . import runtime, kernels
from .dtypes import REAL, CPLX
from .protocol import param_count


NORM_ATOL = 1e-12


def _as_batch_x(x, method, L):
    x = runtime.as_host_array(x, dtype=REAL)
    p = param_count(method, L)
    if x.ndim == 1:
        if x.shape != (p,):
            raise AssertionError(f"x shape {x.shape} != ({p},)")
        x = x.reshape(1, p)
    elif x.ndim != 2 or x.shape[1] != p:
        raise AssertionError(f"x shape {x.shape} != (batch, {p})")
    return x, int(x.shape[0]), p


def _multiply_inplace(dst, factor):
    """dst *= factor, alias-safe for a diagonal (elementwise) multiply.

    Keeps `dst` the same array object so it stays in a ping-pong pair (C1).
    """
    xp = runtime.xp()
    xp.multiply(dst, factor, out=dst)
    return dst


def nov3_energy_grad(m, x, L, norm_atol=NORM_ATOL):
    """Return E, g, psi. E,g are host float64; psi is a device (batch, 2**N) array."""
    xp = runtime.xp()
    L = int(L)
    x, batch, p = _as_batch_x(x, "nov3", L)
    n = m.n
    dim = 1 << n
    xd = runtime.to_device(x, dtype=REAL)
    g0 = xd[:, 0].reshape(batch, 1)
    a1 = xd[:, 1].reshape(batch, 1)
    a2 = xd[:, 2].reshape(batch, 1)
    D0, D1, D2 = m.diags
    gen = D0.reshape(1, -1) + a1 * D1.reshape(1, -1) + a2 * D2.reshape(1, -1)
    phase = xp.exp((-1j) * g0 * gen)
    rows = xd[:, 3:].reshape(batch, L, 3)

    layer_in = xp.empty((L, batch, dim), dtype=CPLX)
    sph_cache = xp.empty((L, batch, dim), dtype=CPLX)
    buf_a = xp.empty((batch, dim), dtype=CPLX)
    buf_b = xp.empty((batch, dim), dtype=CPLX)
    buf_a[...] = m.plus.reshape(1, -1)
    cur = buf_a
    for ell in range(L):
        layer_in[ell] = cur
        cur *= phase
        cur = kernels.walsh_inplace(cur, n)
        row = rows[:, ell, :]
        sph = xp.exp((-1j) * xp.matmul(row, m.shell.T))
        sph_cache[ell] = sph
        cur *= sph
        cur = kernels.walsh_inplace(cur, n)

    psi = cur.copy()
    E_dev, Hv = kernels.energy(psi, D0, m.b, n)
    nrm_dev = kernels.norms(psi)

    lam = Hv.copy()  # lam lives in this allocation; in-place updates only
    fwd = xp.empty((batch, dim), dtype=CPLX)
    g_dev = xp.zeros((batch, p), dtype=REAL)
    two = REAL(2.0)

    for ell in range(L - 1, -1, -1):
        vin = layer_in[ell]
        sph = sph_cache[ell]
        vA = vin * phase  # dedicated buffer; never passed as a transform src
        # vB = Walsh(vA) in fwd, vA remains intact (C2)
        fwd[...] = vA
        vB = kernels.walsh_inplace(fwd, n)
        vC = vB * sph
        lam = kernels.walsh_inplace(lam, n)
        # 2 Re(λ† (-i w) vC) = 2 w · Im(conj(λ) vC) for real w = shell[:,j]
        tC = xp.imag(xp.conj(lam) * vC)
        g_dev[:, 3 + 3 * ell:6 + 3 * ell] += two * (tC @ m.shell)
        _multiply_inplace(lam, xp.conj(sph))
        lam = kernels.walsh_inplace(lam, n)
        tA = xp.imag(xp.conj(lam) * vA)
        g_dev[:, 0] += two * xp.sum(tA * gen, axis=1)
        g_dev[:, 1] += two * xp.sum(tA * (g0 * D1.reshape(1, -1)), axis=1)
        g_dev[:, 2] += two * xp.sum(tA * (g0 * D2.reshape(1, -1)), axis=1)
        _multiply_inplace(lam, xp.conj(phase))

    nrm_h, E_h, g_h = runtime.pack_to_host(nrm_dev, E_dev, g_dev)
    if np.any(np.abs(nrm_h - 1.0) >= norm_atol):
        raise AssertionError(f"nov3 state norm {nrm_h} (atol={norm_atol})")
    return _pack_result(E_h, g_h, psi, batch)


def qaoa_energy_grad(m, x, L, norm_atol=NORM_ATOL):
    xp = runtime.xp()
    L = int(L)
    x, batch, p = _as_batch_x(x, "qaoa", L)
    n = m.n
    dim = 1 << n
    xd = runtime.to_device(x, dtype=REAL)
    D = m.diags[0]

    layer_in = xp.empty((L, batch, dim), dtype=CPLX)
    buf_a = xp.empty((batch, dim), dtype=CPLX)
    buf_a[...] = m.plus.reshape(1, -1)
    cur = buf_a
    for ell in range(L):
        layer_in[ell] = cur
        gamma = xd[:, 2 * ell].reshape(batch, 1)
        beta = xd[:, 2 * ell + 1]
        cur *= xp.exp((-1j) * gamma * D.reshape(1, -1))
        cur = kernels.rx_inplace(cur, beta, n)

    psi = cur.copy()
    E_dev, Hv = kernels.energy(psi, D, m.b, n)
    nrm_dev = kernels.norms(psi)

    lam = Hv.copy()
    fwd = xp.empty((batch, dim), dtype=CPLX)
    xacc = xp.empty((batch, dim), dtype=CPLX)
    g_dev = xp.zeros((batch, p), dtype=REAL)
    two = REAL(2.0)
    for ell in range(L - 1, -1, -1):
        vin = layer_in[ell]
        ig, ib = 2 * ell, 2 * ell + 1
        gamma = xd[:, ig].reshape(batch, 1)
        beta = xd[:, ib]
        phase = xp.exp((-1j) * gamma * D.reshape(1, -1))
        vA = vin * phase  # dedicated; not used as RX src
        fwd[...] = vA
        vB = kernels.rx_inplace(fwd, beta, n)
        Xv = kernels.apply_X(vB, n, out=xacc)
        g_dev[:, ib] += two * xp.real(xp.sum(xp.conj(lam) * ((-1j) * Xv), axis=1))
        # adjoint RX(-beta) in place on lam (stays the same array, C1)
        lam = kernels.rx_inplace(lam, -beta, n)
        # γ: 2 Re(λ† (-i D) vA) = 2 D · Im(conj(λ) vA)
        tA = xp.imag(xp.conj(lam) * vA)
        g_dev[:, ig] += two * xp.sum(tA * D.reshape(1, -1), axis=1)
        _multiply_inplace(lam, xp.conj(phase))

    nrm_h, E_h, g_h = runtime.pack_to_host(nrm_dev, E_dev, g_dev)
    if np.any(np.abs(nrm_h - 1.0) >= norm_atol):
        raise AssertionError(f"qaoa state norm {nrm_h} (atol={norm_atol})")
    return _pack_result(E_h, g_h, psi, batch)


def _pack_result(E, g, psi, batch):
    E = np.ascontiguousarray(E, dtype=REAL)
    g = np.ascontiguousarray(g, dtype=REAL)
    if batch == 1:
        return float(np.asarray(E).reshape(-1)[0]), g.reshape(-1), psi
    return E.reshape(batch), g.reshape(batch, -1), psi


def energy_grad(m, x, method, L, norm_atol=NORM_ATOL):
    if method == "nov3":
        return nov3_energy_grad(m, x, L, norm_atol=norm_atol)
    if method == "qaoa":
        return qaoa_energy_grad(m, x, L, norm_atol=norm_atol)
    raise ValueError(method)


def central_diff_grad(m, x, method, L, eps=1e-7):
    """Central difference of energy at a single parameter vector."""
    x = runtime.as_host_array(x, dtype=REAL)
    if x.ndim != 1:
        raise ValueError("central_diff_grad expects a single parameter vector")
    g = np.zeros_like(x, dtype=REAL)
    eps = float(eps)
    for k in range(x.size):
        xp = x.copy(); xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        Ep, _ = m.energy_of_params(xp, L, method)
        Em, _ = m.energy_of_params(xm, L, method)
        g[k] = (float(Ep) - float(Em)) / (2 * eps)
    return g
