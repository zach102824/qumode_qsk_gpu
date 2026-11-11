"""Batched FWHT / RX / apply_X / apply_H / energy kernels.

Layout
------
Every state array is contiguous complex128 with shape (batch, 2**N).
Qubit 0 is the least-significant bit (stride 1), matching the CPU Numba
kernels. Do not pass (B, N, 2**N) or a dense 2**N x 2**N operator.

Aliasing and source clobber
---------------------------
The *Python* ping-pong path (fallback) uses two buffers and **clobbers
src**: after qubit 1, src is used as scratch. Callers that still need the
pre-transform vector (the backward `vA` in gradients) must copy it first,
or call with preserve_src=True.

The default *RawKernel* path is in-place on a single buffer, one launch per
qubit, with each thread owning a pair (read a,b then write both). That
matches the Numba in-place convention and does not need ping-pong. It still
overwrites the buffer it is applied to, so callers must copy if they need
the input later.

Conventions (must match cpu_oracle/src/kernels_numba.py)
-------------------------------------------------------
- Walsh on qubit q: even' = (a+b)/sqrt(2), odd' = (a-b)/sqrt(2),
  using division by sqrt(2) (same as the Numba loops). This is not a
  bitwise-identity claim against the CPU: other ULP sources remain
  (libm, CuPy ufuncs/reductions, FMA outside these RawKernels).
- RX(beta) on each qubit: even' = cos(beta)*a + (-i sin(beta))*d, and
  odd'  = (-i sin(beta))*a + cos(beta)*d. Textbook RX(2*beta). Frozen.
- apply_X is sum_q X_q, accumulated in ascending q (same order as Numba).
- H|v> = D0 * v - B * apply_X(v). Energy is Re(<v|H|v>).

Raw kernels are compiled lazily on first use after runtime.acquire(), with
``--fmad=false`` so FMA contraction does not widen the gap to Numba
(fastmath off). libm exp/cos/sin and reductions can still differ at ULP
level. No compile or allocation happens at import time.
"""
from __future__ import annotations

import numpy as np

from . import runtime
from .dtypes import REAL, CPLX, SQRT2, check_cplx, check_real
from .kernels_cuda import WALSH, RX, APPLYX

_THREADS = 256
_RAW_OPTIONS = ("--fmad=false",)


def _xp():
    return runtime.xp()


def _is_c_contiguous(arr):
    flags = getattr(arr, "flags", None)
    if flags is None:
        return True
    return bool(getattr(flags, "c_contiguous", False))


def _require_contiguous(arr, name):
    """RawKernel arguments are raw pointers; strides are ignored."""
    if not _is_c_contiguous(arr):
        raise ValueError(
            f"{name} must be C-contiguous for RawKernel pointer passing "
            f"(got shape={getattr(arr, 'shape', None)} strides={getattr(arr, 'strides', None)})"
        )
    return arr


def _require_states(states, n):
    check_cplx(states, "states")
    n = int(n)
    dim = 1 << n
    if states.ndim != 2 or int(states.shape[-1]) != dim:
        raise ValueError(
            f"states must have shape (batch, {dim}) = (batch, 2**{n}), got {getattr(states, 'shape', None)}"
        )
    _require_contiguous(states, "states")
    return states, n, dim, int(states.shape[0])


def _reshape_qubit(arr, n, q):
    """View (B, 2**N) as (B, 2**(n-q-1), 2, 2**q) so axis -2 is qubit q (LSB=0)."""
    batch, dim = arr.shape
    step = 1 << int(q)
    return arr.reshape(batch, dim // (2 * step), 2, step)


def _grid(n_threads):
    blocks = (int(n_threads) + _THREADS - 1) // _THREADS
    return (max(blocks, 1),), (_THREADS,)


def _ensure_raw():
    """Build RawKernels once per acquire() and force NVRTC compile in the guard.

    CuPy RawKernel construction is lazy: without an explicit compile (or a
    launch) the try/except would not see compile errors. The stamp records
    which path actually compiled (raw vs python fallback).
    Never runs at import.
    """
    bundle = runtime._STATE.get("kernels")
    if bundle is not None:
        return bundle
    cp = runtime.xp()
    try:
        with runtime.device_context():
            walsh = cp.RawKernel(WALSH, "walsh_qubit", options=_RAW_OPTIONS)
            rx = cp.RawKernel(RX, "rx_qubit", options=_RAW_OPTIONS)
            ax = cp.RawKernel(APPLYX, "apply_X_fused", options=_RAW_OPTIONS)
            for k in (walsh, rx, ax):
                if not hasattr(k, "compile"):
                    raise RuntimeError(
                        "CuPy RawKernel has no compile(); refusing a lazy path "
                        "that would hide compile errors until first launch"
                    )
                k.compile()
        bundle = dict(walsh=walsh, rx=rx, apply_X=ax, ok=True, error=None,
                      path="raw", options=" ".join(_RAW_OPTIONS))
    except Exception as exc:
        bundle = dict(walsh=None, rx=None, apply_X=None, ok=False, error=repr(exc),
                      path="python", options=" ".join(_RAW_OPTIONS))
    runtime._STATE["kernels"] = bundle
    return bundle


def ensure_compiled():
    """Force the compile/fallback decision after acquire(). Returns the bundle."""
    if not runtime.is_acquired():
        raise RuntimeError("runtime.acquire() before compiling kernels")
    return _ensure_raw()


def raw_kernels_available():
    if not runtime.is_acquired():
        return False
    return bool(_ensure_raw().get("ok"))


def raw_kernels_error():
    if not runtime.is_acquired():
        return "runtime not acquired"
    return _ensure_raw().get("error")


def kernel_path_info():
    """Binding fields: which kernel path is live in this process."""
    if not runtime.is_acquired():
        return dict(kernel_path=None, raw_kernel_options=" ".join(_RAW_OPTIONS),
                    raw_kernels_error="runtime not acquired")
    b = _ensure_raw()
    return dict(
        kernel_path=b.get("path") or ("raw" if b.get("ok") else "python"),
        raw_kernel_options=" ".join(_RAW_OPTIONS),
        raw_kernels_error=b.get("error"),
    )


def prepare_beta(beta, batch, xp=None):
    """C-contiguous 1-d float64 beta and a scalar flag.

    `beta[:, column]` (and any other strided view) is copied. RawKernels
    read beta[b] as a flat pointer and ignore strides, so a view of
    shape (batch,) with stride 2 would otherwise feed the QAOA minus
    probe the wrong angle (R2-H1).
    """
    if xp is None:
        xp = np
    arr = xp.asarray(beta, dtype=REAL)
    flat = xp.ascontiguousarray(arr.reshape(-1))
    if int(flat.size) == 1:
        return flat, 1
    if int(flat.shape[0]) != int(batch):
        raise ValueError(f"beta batch {tuple(flat.shape)} != states batch {batch}")
    flags = getattr(flat, "flags", None)
    if flags is not None and not bool(getattr(flags, "c_contiguous", False)):
        raise ValueError("beta is not C-contiguous after ascontiguousarray")
    return flat, 0


def _beta_device(beta, batch):
    """Return (beta_dev 1-d C-contiguous float64, beta_scalar int)."""
    xp = _xp()
    beta_dev, beta_scalar = prepare_beta(beta, batch, xp=xp)
    _require_contiguous(beta_dev, "beta")
    return beta_dev, beta_scalar


def walsh_pingpong(src, dst, n, preserve_src=False):
    """Walsh-Hadamard with a two-buffer scheme.

    **src is clobbered** unless ``preserve_src=True``. After qubit 1 the
    original src buffer is used as scratch in the Python path; the RawKernel
    path overwrites src in place. ``preserve_src=True`` copies src into dst
    (src is dst is rejected) and transforms dst, leaving src intact.

    Prefer ``walsh_inplace`` when the input buffer may be overwritten.
    """
    src, n, dim, batch = _require_states(src, n)
    if preserve_src:
        if src is dst:
            raise ValueError("preserve_src=True requires a distinct dst; src would be destroyed")
        dst[...] = src
        return walsh_inplace(dst, n)
    if src is dst:
        raise ValueError(
            "walsh_pingpong requires two distinct buffers when preserve_src is False "
            "(src is scratch and is clobbered). Use walsh_inplace or preserve_src=True."
        )
    if dst.shape != src.shape or str(dst.dtype) != "complex128":
        raise ValueError("dst must match src shape and complex128")
    bundle = _ensure_raw()
    if bundle.get("ok"):
        return walsh_inplace(src, n)
    return _walsh_pingpong_python(src, dst, n)


def _walsh_pingpong_python(src, dst, n):
    s = SQRT2
    cur, other = src, dst
    for q in range(n):
        x = _reshape_qubit(cur, n, q)
        y = _reshape_qubit(other, n, q)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        y[:, :, 0, :] = (a + b) / s
        y[:, :, 1, :] = (a - b) / s
        cur, other = other, cur
    return cur


def walsh_inplace(v, n):
    """In-place Walsh on v. The input values are overwritten (same as Numba)."""
    v, n, dim, batch = _require_states(v, n)
    bundle = _ensure_raw()
    if bundle.get("ok"):
        k = bundle["walsh"]
        with runtime.device_context():
            for q in range(n):
                step = 1 << q
                grid, block = _grid(batch * (dim >> 1))
                k(grid, block, (v, v, np.int64(batch), np.int64(dim), np.int64(step)))
        return v
    xp = _xp()
    work = xp.empty_like(v)
    out = _walsh_pingpong_python(v, work, n)
    if out is not v:
        v[...] = out
    return v


def walsh(states, n, work=None, preserve_src=False):
    """Return H^{otimes n} |states|.

    If preserve_src is False (default), ``states`` may be clobbered and the
    returned buffer may be ``states`` or ``work``. If preserve_src is True,
    ``states`` is left intact and the result is written to ``work``.
    """
    xp = _xp()
    states, n, dim, batch = _require_states(states, n)
    if preserve_src:
        if work is None:
            work = xp.empty_like(states)
        if work is states:
            raise ValueError("preserve_src=True requires a distinct work buffer")
        work[...] = states
        return walsh_inplace(work, n)
    return walsh_inplace(states, n)


def rx_pingpong(src, dst, beta, n, preserve_src=False):
    """Product of per-qubit RX(beta). **src is clobbered** unless preserve_src=True.

    beta may be a Python float, a 0-d array, or shape (batch,) / (batch, 1)
    float64. Per-batch coefficients are shaped (batch, 1, 1) against the
    3-D slices (batch, n_blocks, step) — never (batch, 1, 1, 1).
    Adjoint is rx(..., -beta, ...).
    """
    src, n, dim, batch = _require_states(src, n)
    if preserve_src:
        if src is dst:
            raise ValueError("preserve_src=True requires a distinct dst; src would be destroyed")
        dst[...] = src
        return rx_inplace(dst, beta, n)
    if src is dst:
        raise ValueError(
            "rx_pingpong requires two distinct buffers when preserve_src is False "
            "(src is scratch and is clobbered). Use rx_inplace or preserve_src=True."
        )
    bundle = _ensure_raw()
    if bundle.get("ok"):
        return rx_inplace(src, beta, n)
    return _rx_pingpong_python(src, dst, beta, n)


def _rx_pingpong_python(src, dst, beta, n):
    src, n, dim, batch = _require_states(src, n)
    xp = _xp()
    beta = xp.asarray(beta, dtype=REAL)
    if beta.ndim == 0 or beta.size == 1:
        c = xp.asarray(xp.cos(beta), dtype=CPLX)
        s = xp.asarray((-1j) * xp.sin(beta), dtype=CPLX)
    else:
        beta = beta.reshape(-1)
        if int(beta.shape[0]) != int(batch):
            raise ValueError(f"beta batch {beta.shape} != states batch {batch}")
        # (batch, 1, 1) broadcasts against 3-D slices (batch, n_blocks, step).
        # A (batch, 1, 1, 1) reshape is wrong: c*a would go to (B, B, ...).
        c = xp.cos(beta).astype(CPLX, copy=False).reshape(batch, 1, 1)
        s = ((-1j) * xp.sin(beta)).astype(CPLX, copy=False).reshape(batch, 1, 1)
    cur, other = src, dst
    for q in range(n):
        x = _reshape_qubit(cur, n, q)
        y = _reshape_qubit(other, n, q)
        a = x[:, :, 0, :]
        d = x[:, :, 1, :]
        y[:, :, 0, :] = c * a + s * d
        y[:, :, 1, :] = s * a + c * d
        cur, other = other, cur
    return cur


def rx_inplace(v, beta, n):
    """In-place product of per-qubit RX(beta). Overwrites v."""
    v, n, dim, batch = _require_states(v, n)
    bundle = _ensure_raw()
    if bundle.get("ok"):
        beta_dev, beta_scalar = _beta_device(beta, batch)
        k = bundle["rx"]
        with runtime.device_context():
            for q in range(n):
                step = 1 << q
                grid, block = _grid(batch * (dim >> 1))
                k(grid, block, (v, v, beta_dev, np.int64(batch), np.int64(dim),
                                np.int64(step), np.int32(beta_scalar)))
        return v
    xp = _xp()
    work = xp.empty_like(v)
    out = _rx_pingpong_python(v, work, beta, n)
    if out is not v:
        v[...] = out
    return v


def rx(states, beta, n, work=None, preserve_src=False):
    xp = _xp()
    states, n, dim, batch = _require_states(states, n)
    if preserve_src:
        if work is None:
            work = xp.empty_like(states)
        if work is states:
            raise ValueError("preserve_src=True requires a distinct work buffer")
        work[...] = states
        return rx_inplace(work, beta, n)
    return rx_inplace(states, beta, n)


def apply_X(states, n, out=None):
    """(sum_q X_q) |states|, accumulated into `out` in ascending q (Numba order)."""
    xp = _xp()
    states, n, dim, batch = _require_states(states, n)
    if out is None:
        out = xp.empty_like(states)
    else:
        if out is states:
            raise ValueError("apply_X refuses to accumulate into the input buffer")
        _require_contiguous(out, "apply_X out")
    bundle = _ensure_raw()
    if bundle.get("ok"):
        k = bundle["apply_X"]
        with runtime.device_context():
            grid, block = _grid(batch * dim)
            k(grid, block, (states, out, np.int64(batch), np.int64(dim), np.int32(n)))
        return out
    out.fill(0)
    for q in range(n):
        x = _reshape_qubit(states, n, q)
        y = _reshape_qubit(out, n, q)
        y[:, :, 0, :] += x[:, :, 1, :]
        y[:, :, 1, :] += x[:, :, 0, :]
    return out


def apply_H(states, D0, b, n, out=None, xacc=None):
    xp = _xp()
    states, n, dim, batch = _require_states(states, n)
    check_real(D0, "D0")
    if xacc is None:
        xacc = xp.empty_like(states)
    apply_X(states, n, out=xacc)
    b_arr = xp.asarray(b, dtype=REAL)
    if b_arr.ndim == 0:
        scale = b_arr
    else:
        scale = b_arr.reshape(batch, 1)
    if out is None:
        out = D0 * states - scale * xacc
    else:
        out[...] = D0 * states - scale * xacc
    return out


def energy(states, D0, b, n, Hv=None, xacc=None):
    """Return (E, Hv) with E shape (batch,) float64 on device: Re(<v|H|v>)."""
    xp = _xp()
    states, n, dim, batch = _require_states(states, n)
    Hv = apply_H(states, D0, b, n, out=Hv, xacc=xacc)
    acc = xp.sum(xp.conj(states) * Hv, axis=1)
    E = xp.real(acc).astype(REAL, copy=False)
    return E, Hv


def norms(states):
    xp = _xp()
    check_cplx(states, "states")
    return xp.real(xp.sum(xp.conj(states) * states, axis=-1)).astype(REAL, copy=False)


def fidelity(states, gs):
    """|<gs|psi>|^2 for gs shape (2**N,) or (batch, 2**N)."""
    xp = _xp()
    check_cplx(states, "states")
    check_cplx(gs, "gs")
    ov = xp.sum(xp.conj(gs) * states, axis=-1)
    return xp.real(ov * xp.conj(ov)).astype(REAL, copy=False)


def max_abs_diff(a, b):
    """max|a-b| reduced on device, returned as a 0-d device scalar."""
    xp = _xp()
    return xp.max(xp.abs(a - b))


def zz_diag_host(J, n):
    """out[s] = sum_{i<j} J_ij Z_i(s) Z_j(s), Z = +1 for bit 0, -1 for bit 1.

    Accumulation order matches the Numba loop (pairs i<j, then all s).
    """
    n = int(n)
    J = np.ascontiguousarray(J, dtype=REAL)
    if J.shape != (n, n):
        raise ValueError("J shape")
    dim = 1 << n
    s = np.arange(dim, dtype=np.int64)
    out = np.zeros(dim, dtype=REAL)
    for i in range(n):
        zi = (1 - 2 * ((s >> i) & 1)).astype(REAL, copy=False)
        for j in range(i + 1, n):
            coeff = J[i, j]
            if coeff == 0.0:
                continue
            zj = (1 - 2 * ((s >> j) & 1)).astype(REAL, copy=False)
            out += coeff * zi * zj
    return np.ascontiguousarray(out, dtype=REAL)


def shell_ops_host(groups, n):
    """shell[s, 0/1/2] = sum_groups C(k, 1/2/3) with k = popcount of the triplet.

    Loops groups one at a time so the temporary is (dim,) not (dim, n/3, 3).
    At N=21 that avoids a ~350 MiB int64 spike.
    """
    n = int(n)
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    if groups.ndim != 2 or groups.shape[1] != 3:
        raise ValueError("groups must be (n/3, 3)")
    dim = 1 << n
    s = np.arange(dim, dtype=np.int64)
    shell = np.zeros((dim, 3), dtype=REAL)
    for g in groups:
        k = (
            ((s >> int(g[0])) & 1)
            + ((s >> int(g[1])) & 1)
            + ((s >> int(g[2])) & 1)
        ).astype(REAL, copy=False)
        shell[:, 0] += k
        shell[:, 1] += k * (k - 1) * REAL(0.5)
        shell[:, 2] += k * (k - 1) * (k - 2) / REAL(6.0)
    return np.ascontiguousarray(shell, dtype=REAL)


def mul_diag(states, diag):
    """Elementwise states * diag, with diag (dim,) or (batch, dim)."""
    return states * diag
