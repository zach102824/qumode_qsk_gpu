"""Device-memory bounds for batched statevectors, including adjoints.

States are stored as shape (batch, 2**N) complex128, never (B, N, 2**N)
and never a dense 2**N x 2**N matrix.

Peak estimates include:
- live state and ping-pong Walsh/RX buffers
- reverse-mode layer inputs (L copies)
- adjoint lambda, H|psi>, apply_X temporaries
- uploaded constants (3 diags, shell, plus, optional gs)
- a conservative eigensolver workspace term (entropy scan; usually CPU)

These are planning bounds, not a CUDA memory-pool query.
"""
from __future__ import annotations

from .dtypes import CPLX, REAL
from .protocol import param_count

BYTES_CPLX = 16  # complex128
BYTES_REAL = 8


def dim(n):
    return 1 << int(n)


def bytes_per_state(n):
    return dim(n) * BYTES_CPLX


def constant_bytes(n, with_gs=True):
    """J is n x n and stays host-side; diags/shell/plus/gs may sit on device."""
    d = dim(n)
    diags = 3 * d * BYTES_REAL
    shell = d * 3 * BYTES_REAL
    plus = d * BYTES_CPLX
    gs = d * BYTES_CPLX if with_gs else 0
    # phase / gen / sph workspaces often allocated per eval
    return diags + shell + plus + gs


def energy_only_vector_count():
    # v, Hv, apply_X acc, phase scratch, plus a couple of elementwise temporaries
    return 6


def grad_vector_count(L):
    """Worst-case live complex vectors during reverse-mode, per batch row.

    Count at the backward peak (C1/C2 workspace): layer_in (L), sph cache (L),
    psi, Hv, lam, fwd, vA, vC, leftover buf, phase, plus product/conj
    temporaries (~4). Real diags/gen/shell are in constant_bytes.
    """
    L = int(L)
    return 2 * L + 14


def eigensolver_workspace_bytes(n, parity_sector=True):
    """Documented upper bound if a GPU eigensolver is used.

    The default entropy-peak scan uses the CPU SciPy path, so this term is
    not allocated on the device in production unless the experimental GPU
    solver is explicitly enabled. Included so planners do not forget it.
    """
    half = 1 << (int(n) - 1) if parity_sector else dim(n)
    # A few dense work vectors + a sparse X pair (~N nnz per row, 2 sectors)
    dense = 8 * half * BYTES_REAL
    sparse = 2 * (int(n) * half) * (BYTES_REAL + 2 * 8)
    return dense + sparse


def peak_bytes(n, L, batch, method="nov3", with_gs=True, include_eigensolver=False):
    L = int(L)
    batch = int(batch)
    vecs = grad_vector_count(L) if method in ("nov3", "qaoa") else energy_only_vector_count()
    total = batch * vecs * bytes_per_state(n) + constant_bytes(n, with_gs=with_gs)
    if include_eigensolver:
        total += eigensolver_workspace_bytes(n)
    # 20% fragmentation / allocator slack
    return int(total * 1.20)


def max_batch(n, L, device_total_bytes, method="nov3", with_gs=True,
              safety=0.4, include_eigensolver=False):
    """Largest batch that fits in safety * device_total_bytes.

    Default safety=0.4 leaves room for the CUDA context, memory pool, and
    the OS. Returns at least 0 if even batch=1 does not fit.
    """
    if device_total_bytes is None:
        return 0
    budget = int(float(safety) * int(device_total_bytes))
    # binary search batch
    lo, hi = 0, 64
    while True:
        need = peak_bytes(n, L, hi, method, with_gs, include_eigensolver)
        if need > budget or hi >= 4096:
            break
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if peak_bytes(n, L, mid, method, with_gs, include_eigensolver) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)


def plan(n, L, method, device_total_bytes, requested_batch=None, with_gs=True):
    cap = max_batch(n, L, device_total_bytes, method, with_gs=with_gs)
    req = cap if requested_batch is None else int(requested_batch)
    batch = max(0, min(req, cap))
    return dict(
        N=int(n), L=int(L), method=method,
        param_count=param_count(method, L),
        dim=dim(n),
        bytes_per_state=bytes_per_state(n),
        peak_bytes_batch1=peak_bytes(n, L, 1, method, with_gs),
        peak_bytes_chosen=peak_bytes(n, L, max(batch, 1), method, with_gs),
        eigensolver_workspace_bytes=eigensolver_workspace_bytes(n),
        eigensolver_default="cpu_scipy_eigsh",
        device_total_bytes=device_total_bytes,
        max_batch=cap,
        chosen_batch=batch,
        dtype="complex128",
        layout="[batch, 2**N]",
        note="Bounds include reverse-mode layer inputs, sph cache, adjoints, and temporaries. Production consults max_batch and refuses a cell that does not fit at its required batch (1 for BFGS, 2 for SPSA).",
    )
