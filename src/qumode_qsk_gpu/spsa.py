"""Frozen R9 IID SPSA: Q=1024*L inclusive, 2 cold starts (narrow 0.16, broad 0.6).

The query ledger, RNG (numpy default_rng on the host), role counts, padding
and late-iterate readouts are identical to cpu_oracle/src/spsa.py. Only the
energy evaluation itself runs on the GPU. Directions stay on the host.

The two ±c energy queries of one update can be batched (two states, one
upload of the two parameter vectors) without changing the RNG order.
Starts are still processed in the original order.

This is not a custom SPSA algorithm. Do not change T_of_Q, the gain rule,
or the inclusive 1024 L budget.
"""
from __future__ import annotations

import collections, hashlib

import numpy as np

from .protocol import param_count
from .starts import spsa_starts
from .dtypes import REAL
from .io_util import x_sha256


def T_of_Q(Q):
    Q = int(Q)
    T = 0
    while 18 + 4 * (T + 1) + 2 * ((T + 1) // 5) <= Q - 6:
        T += 1
    prefix = 18 + 4 * T + 2 * (T // 5)
    padding = Q - 6 - prefix
    return dict(Q=Q, T=T, prefix_queries=prefix, padding=padding, readouts=6)


def query_schedule(L):
    L = int(L)
    Q = 1024 * L
    d = T_of_Q(Q)
    T = d["T"]
    roles = dict(
        initial=2,
        calibration_plus=8,
        calibration_minus=8,
        spsa_plus=2 * T,
        spsa_minus=2 * T,
        candidate_monitor=2 * (T // 5),
        charged_prefix_padding_remeasurement=d["padding"],
        readout_late_iterate=6,
    )
    total = sum(roles.values())
    if total != Q or not (0 <= d["padding"] <= 5):
        raise AssertionError((L, Q, roles, total, d))
    d.update(L=L, roles=roles, total=total)
    return d


def optimize(m, method, L, Q=None):
    """Run SPSA on a device Model `m` (already uploaded, GS loaded if F is wanted)."""
    L = int(L)
    Q = 1024 * L if Q is None else int(Q)
    assert method in ("nov3", "qaoa") and 1 <= L <= 6 and Q == 1024 * L
    sched = query_schedule(L)
    T = sched["T"]
    p = param_count(method, L)
    seed = m.seed
    starts, rng = spsa_starts(seed, L, method)
    calls = 0
    bestE = np.inf
    best = None
    log = []
    snaps = []
    histories = []
    direction_hashes = []
    marks = {128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, Q - 6}
    marks.discard(Q)

    def ev(x, kind, meta=None):
        nonlocal calls, bestE, best
        E, _v = m.energy_of_params(x, L, method)
        E = float(E)
        assert np.isfinite(E)
        calls += 1
        z = dict(query=calls, kind=kind, E=E, x_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
        if meta:
            z.update(meta)
        log.append(z)
        if E < bestE:
            bestE = E
            best = np.array(x, dtype=REAL, copy=True)
        if calls in marks:
            snaps.append((calls, bestE, best.copy()))
        return E

    def ev_pair(xplus, xminus, kind_plus, kind_minus):
        """Batch the two SPSA energy queries; still records them in + then - order."""
        nonlocal calls, bestE, best
        xs = np.stack([np.ascontiguousarray(xplus, dtype=REAL),
                       np.ascontiguousarray(xminus, dtype=REAL)], axis=0)
        E, _v = m.energy_of_params(xs, L, method)
        # energy_of_params always returns host float64 (H2); still go through
        # a host helper, never np.asarray on a CuPy array.
        from . import runtime
        E = np.asarray(runtime.as_host_array(E, dtype=REAL), dtype=REAL).reshape(2)
        out = []
        for x, Ev, kind in ((xplus, float(E[0]), kind_plus), (xminus, float(E[1]), kind_minus)):
            assert np.isfinite(Ev)
            calls += 1
            z = dict(query=calls, kind=kind, E=Ev,
                     x_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
            log.append(z)
            if Ev < bestE:
                bestE = Ev
                best = np.array(x, dtype=REAL, copy=True)
            if calls in marks:
                snaps.append((calls, bestE, best.copy()))
            out.append(Ev)
        return out[0], out[1]

    norms = []
    for x in starts:
        ev(x, "initial")
        for _ in range(4):
            delta = rng.choice([-1.0, 1.0], p)
            Ep, Em = ev_pair(x + 0.1 * delta, x - 0.1 * delta, "calibration_plus", "calibration_minus")
            g = (Ep - Em) / 0.2 * delta
            norms.append(min(float(np.linalg.norm(g)), 5.0))
    gain = float(np.clip(0.15 / ((0.12 / 10 ** 0.602) * max(float(np.median(norms)), 1e-12)), 1.0, 4.0))
    for x0 in starts:
        x = x0.copy()
        hist = []
        dirs = []
        for k in range(T):
            delta = rng.choice([-1.0, 1.0], p)
            dirs.append(delta.astype(np.int8))
            c = 0.1 / (k + 1) ** 0.101
            a = gain * 0.12 / (k + 10) ** 0.602
            Ep, Em = ev_pair(x + c * delta, x - c * delta, "spsa_plus", "spsa_minus")
            g = (Ep - Em) / (2 * c) * delta
            g *= min(1.0, 5.0 / max(float(np.linalg.norm(g)), 1e-15))
            x = x - a * g
            hist.append(x.copy())
            if (k + 1) % 5 == 0:
                ev(x, "candidate_monitor")
        histories.append(np.array(hist))
        direction_hashes.append(hashlib.sha256(np.ascontiguousarray(dirs, dtype=np.int8).tobytes()).hexdigest())
    padding = Q - 6 - calls
    assert 0 <= padding <= 5
    for _ in range(padding):
        ev(best, "charged_prefix_padding_remeasurement")
    assert calls == Q - 6
    prefix_E = bestE
    readouts = []
    for restart, h in enumerate(histories):
        for k in [k for k in range(T, 0, -1) if k % 5 != 0][:3]:
            x = h[k - 1]
            E = ev(x, "readout_late_iterate", dict(restart=restart, iterate_index=k))
            readouts.append(dict(E=E, x=x.tolist(), restart=restart, iterate_index=k))
    assert calls == Q
    v = m.state(best, L, method)
    nrm = float(np.asarray(runtime_norm(v)).reshape(-1)[0])
    F = m.fidelity(v) if m.gs is not None else None
    trace = []
    for q, E, x in snaps:
        rec = dict(query=q, E=E)
        if m.gs is not None:
            rec["F"] = m.fidelity(m.state(x, L, method))
        trace.append(rec)
    recQ = dict(query=Q, E=bestE)
    if F is not None:
        recQ["F"] = F
    trace.append(recQ)
    return dict(
        calls=Q, L=L, method=method, x=best.tolist(), best_E=bestE, F=F,
        norm=nrm,
        query_log=log, trace=trace,
        updates_per_trajectory=T, prefix_padding_queries=padding,
        prefix_best_E=prefix_E,
        readout_candidates=[{k: r[k] for k in r if k != "x"} | {"x_sha256": hashlib.sha256(np.ascontiguousarray(r["x"]).tobytes()).hexdigest()} for r in readouts],
        gain_multiplier=gain,
        query_roles=dict(collections.Counter(z["kind"] for z in log)),
        initial_points=[x.tolist() for x in starts],
        direction_hashes=direction_hashes,
        parameter_count=p,
        schedule=sched,
        fidelity_only_after_all_energy_selection=True,
        init_scheme="TWO_COLD_NARROW_0.16_BROAD_0.6_NO_WARM_NO_LAYER_TRANSFER",
        energy_backend="gpu_statevector",
    )


def runtime_norm(v):
    from . import kernels, runtime
    if v.ndim == 1:
        v = v.reshape(1, -1)
    return runtime.to_host(kernels.norms(v))
