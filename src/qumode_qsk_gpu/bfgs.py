"""Unlimited-query BFGS baseline: 16 cold IID starts, SciPy BFGS on the host.

Each start is an independent `scipy.optimize.minimize(..., method="BFGS", jac=True)`
call. The objective/jacobian is evaluated on the GPU; only a few scalars
(E and the p-vector g, plus the p-vector x) cross the bus per eval.

We do **not** replace SciPy BFGS with a custom batched optimizer. Independent
starts may be run sequentially, or a caller may launch several host BFGS
loops in series. Micro-batching is used only when a caller explicitly
evaluates many parameter vectors at once (see gradients.energy_grad with
x.shape == (batch, p)), which this driver does not do inside SciPy.

Full-trajectory bitwise agreement with the CPU BFGS path is **not**
promised: line-search and Wolfe decisions can diverge after tiny energy
differences. GPU_test compares E and g at fixed stored parameter vectors,
plus one evaluation (and optionally the first SciPy iteration) at a frozen
start. See docs/LIMITATIONS.md.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import minimize

from .protocol import param_count, BFGS_MAXITER, BFGS_GTOL, BFGS_TIGHT
from .gradients import energy_grad, central_diff_grad
from .starts import make_start, all_starts, N_STARTS, start_rng_seed
from .io_util import x_sha256
from .dtypes import REAL

MAXITER = BFGS_MAXITER
GTOL = BFGS_GTOL
TIGHT_INF = BFGS_TIGHT
TIGHT_L2 = BFGS_TIGHT


def run_start(m, method, L, start, maxiter=MAXITER, gtol=GTOL):
    x0 = np.asarray(start["x"] if isinstance(start, dict) else start, dtype=REAL)
    evals = {"n": 0}

    def fun(x):
        E, g, v = energy_grad(m, x, method, L)
        evals["n"] += 1
        evals["last_v"] = v
        evals["last_E"] = E
        evals["last_g"] = g
        return E, g

    E0, g0, v0 = energy_grad(m, x0, method, L)
    F0 = m.fidelity(v0) if m.gs is not None else None
    res = minimize(lambda x: fun(x), x0, jac=True, method="BFGS",
                   options=dict(maxiter=maxiter, gtol=gtol, norm=np.inf, disp=False))
    x = np.asarray(res.x, dtype=REAL)
    E, g, v = energy_grad(m, x, method, L)
    from . import kernels, runtime
    vv = v if v.ndim == 2 else v.reshape(1, -1)
    nrm = float(runtime.to_host(kernels.norms(vv)).reshape(-1)[0])
    E_ref, v_ref = m.energy_of_params(x, L, method)
    # max|Δ| reduced on device; do not download two full N=21 statevectors
    vr = v_ref if v_ref.ndim == 2 else v_ref.reshape(1, -1)
    state_err = float(runtime.as_scalar(kernels.max_abs_diff(vv, vr)))
    g_inf = float(np.max(np.abs(g)))
    g_l2 = float(np.linalg.norm(g))
    if os.environ.get("QSK_FD_CHECK", "0") not in ("0", "false", "False"):
        g_fd = central_diff_grad(m, x, method, L, eps=1e-7)
        fd_err = float(np.max(np.abs(g - g_fd)))
    else:
        fd_err = None
    success = bool(res.success)
    hit_cap = bool((not success) and (int(res.nit) >= maxiter or "iteration" in str(res.message).lower()))
    tight = bool(success and g_inf < TIGHT_INF and g_l2 < TIGHT_L2 and abs(nrm - 1) < 1e-10)
    F = m.fidelity(v) if m.gs is not None else None
    out = dict(
        start_index=int(start.get("start_index", -1)) if isinstance(start, dict) else -1,
        start_kind=start.get("kind") if isinstance(start, dict) else None,
        start_width=start.get("width") if isinstance(start, dict) else None,
        start_width_class=start.get("width_class") if isinstance(start, dict) else None,
        start_rng_seed=start.get("rng_seed") if isinstance(start, dict) else None,
        start_x_sha256=start.get("x_sha256") if isinstance(start, dict) else x_sha256(x0),
        E_initial=float(E0), F_initial=F0, grad_l2_initial=float(np.linalg.norm(g0)),
        x=x.tolist(), x_sha256=x_sha256(x),
        best_E=float(E), F=F, norm=nrm,
        E_vs_driver=abs(float(E) - float(E_ref)),
        state_max_abs_err_vs_driver=state_err,
        final_grad=np.asarray(g, dtype=REAL).tolist(),
        final_grad_inf=g_inf, final_grad_l2=g_l2,
        final_grad_max_abs_err_vs_central=fd_err,
        success=success, status=int(res.status), message=str(res.message),
        nit=int(res.nit), nfev=int(res.nfev), njev=int(getattr(res, "njev", -1)),
        unique_energy_grad_evals=int(evals["n"]),
        hit_iteration_cap=hit_cap, tightly_converged=tight,
        objective="energy_only", fidelity_not_optimized=True,
        maxiter=int(maxiter), gtol=float(gtol),
        init_scheme="ALL_16_IID_COLD_8NARROW_8BROAD_NO_LAYER_TRANSFER_NO_SPSA_WARM",
        energy_backend="gpu_statevector",
        optimizer="scipy.optimize.minimize_BFGS_host",
    )
    if m.e0 is not None:
        out.update(E0=m.e0, energy_error=float(E - m.e0), energy_error_over_N=float((E - m.e0) / m.n))
    return out


def select_by_energy(start_results):
    best = min(start_results, key=lambda z: (z["best_E"], z.get("start_index", 0)))
    Fvals = [z for z in start_results if z.get("F") is not None]
    bestF = max(Fvals, key=lambda z: (z["F"], -z.get("start_index", 0))) if Fvals else None
    return dict(
        selected_start=best.get("start_index"),
        selected_E=best["best_E"],
        selected_F=best.get("F"),
        selected_success=best.get("success"),
        selected_tightly_converged=best.get("tightly_converged"),
        selected_hit_cap=best.get("hit_iteration_cap"),
        selection="lowest_energy_among_16_cold_starts",
        max_F_start=None if bestF is None else bestF.get("start_index"),
        max_F=None if bestF is None else bestF.get("F"),
        max_F_from_different_start=bool(bestF is not None and bestF.get("start_index") != best.get("start_index")),
        n_success=sum(1 for z in start_results if z.get("success")),
        n_tight=sum(1 for z in start_results if z.get("tightly_converged")),
        n_hit_cap=sum(1 for z in start_results if z.get("hit_iteration_cap")),
    )
