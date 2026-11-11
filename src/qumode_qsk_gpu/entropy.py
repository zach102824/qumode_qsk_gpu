"""Ground-state entropy and the official B* scan.

Partition
---------
A = lowest floor(N/2) *logical spin labels* (least-significant bits).
Natural-log von Neumann entropy of the pure-state reduced density on A:

    M = reshape(ψ) with rows = A,   S = -sum_k σ_k log σ_k,
    σ = (SVD singular values of M)^2,   σ <= 1e-15 dropped (0 log 0).

Default production path for the entropy-peak *eigensolve* is the CPU
SciPy / LinearOperator path in cpu_oracle (same as the campaign). A GPU
eigensolver is experimental, opt-in, and is **not** claimed. Do not claim
that the whole entropy pipeline runs on the GPU.

Given a statevector, the Schmidt SVD for S(ψ) may run on the device.
"""
from __future__ import annotations

import numpy as np

from .dtypes import REAL, CPLX
from . import runtime


SVD_CUT = 1e-15


def schmidt_host(gs, n):
    """CPU Schmidt matrix, shape (2**na, 2**(n-na)), A = low bits.

    Equivalent to cpu_oracle: gs[aa + (bb << na)] with aa in A, bb in B.
    """
    n = int(n)
    na = n // 2
    gs = np.ascontiguousarray(gs, dtype=CPLX)
    if gs.shape != (1 << n,):
        gs = np.asarray(gs).reshape(-1)
        if gs.shape != (1 << n,):
            raise ValueError("gs shape")
    return gs.reshape(1 << (n - na), 1 << na).T


def entropy_from_schmidt(M, cut=SVD_CUT):
    """Natural-log von Neumann entropy from a Schmidt matrix (host)."""
    M = np.ascontiguousarray(M)
    sv = np.linalg.svd(M, compute_uv=False)
    p = (sv.astype(REAL, copy=False) ** 2)
    p = p[p > cut]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def entropy_host(gs, n, cut=SVD_CUT):
    return entropy_from_schmidt(schmidt_host(gs, n), cut=cut)


def schmidt_device(gs, n):
    """Device Schmidt matrix. gs is (2**N,) or (1, 2**N) complex128 on device."""
    n = int(n)
    na = n // 2
    if gs.ndim == 2:
        if gs.shape[0] != 1:
            raise ValueError("batched Schmidt is not used; pass one state")
        gs = gs.reshape(-1)
    dim = 1 << n
    if int(gs.shape[0]) != dim:
        raise ValueError("gs dim")
    return gs.reshape(1 << (n - na), 1 << na).T


def entropy_device(gs, n, cut=SVD_CUT):
    """SVD on the smaller Schmidt matrix. Host entropy is the official twin.

    0 log 0 is avoided by the same σ > 1e-15 cut. This path accelerates
    S(ψ) *given* a state; it does not run the B* eigensolve on the GPU.
    """
    xp = runtime.xp()
    M = schmidt_device(gs, n)
    sv = xp.linalg.svd(M, full_matrices=False, compute_uv=False)
    p = (sv.astype(REAL) ** 2)
    # Avoid a boolean-mask sync where possible: use a clipped log.
    # Official cut is still applied for the reported value.
    p_host = runtime.to_host(p)
    p_host = p_host[p_host > cut]
    if p_host.size == 0:
        S = 0.0
    else:
        S = float(-np.sum(p_host * np.log(p_host)))
    return S


def scan_bstar_cpu(n, seed, J, progress_path=None):
    """Run the frozen CPU scan via package-qualified oracle imports (H4).

    Returns (report, peak_gs) exactly as cpu_oracle.src.entropy.scan_bstar.
    """
    from . import oracle
    ham = oracle.hamiltonian()
    ent = oracle.entropy()
    base = ham.Model(n, seed, 1.0, J=J, compute_gs=False)
    return ent.scan_bstar(base, progress_path=progress_path)


def experimental_gpu_eigensolve_note():
    return dict(
        available=False,
        default="cpu_oracle.src.entropy.scan_bstar (SciPy eigsh / dense eigh for N<=9)",
        experimental_gpu="not wired as a production solver",
        reason=(
            "Parity-sector sparse Hermitian eigensolves, residual tolerances, "
            "even-branch gauge (positivity / sign), and near-degeneracy handling "
            "are frozen to the CPU campaign. A CuPy eigsh/LOBPCG path may be "
            "added later as an opt-in experiment; it must not silently replace "
            "the official B* or stored ground-state vectors."
        ),
        gpu_used_for="S(ψ) given a statevector (Schmidt SVD) is optional on device",
        end_to_end_note=(
            "Per-instance wall time includes a CPU scan. task_entropy may "
            "build a GPU Model only as an apply_H energy cross-check."
        ),
    )
