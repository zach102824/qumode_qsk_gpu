"""CPU-entropy / GPU-optimizer pipeline.

Entropy work is host-only (no CUDA, no CuPy allocations). Optional GPU
cross-checks and all optimizer tasks run on the caller's thread. The
producer is a process pool when entropy_workers > 1 (spawn, BLAS threads
= 1, at most one N>=21 scan at a time). Exit is liveness-based: if the
producer stops early (halt, or an exception) the main loop does not
busy-spin.

This module is deliberately free of CUDA imports so host tests can drive
it with fake runners.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED


def fail_record(spec, exc, reused=False):
    return dict(
        spec={k: spec[k] for k in spec} if isinstance(spec, dict) else spec,
        verdict="FAIL",
        error=repr(exc),
        traceback=traceback.format_exc(),
        failed=True,
        reused=bool(reused),
        result_file=None,
    )


def _as_record(spec, rec):
    if rec is None:
        return fail_record(spec, RuntimeError("runner returned None"))
    out = dict(rec)
    out.setdefault("spec", {k: spec[k] for k in spec} if isinstance(spec, dict) else spec)
    failed = bool(out.get("failed") or out.get("verdict") == "FAIL" or out.get("error"))
    out["failed"] = failed
    if failed:
        out.setdefault("verdict", "FAIL")
    return out


def run_pipeline(entropy_order, opt_by_key, *, halt, entropy_workers,
                 entropy_fn, opt_fn, finalize_fn=None,
                 use_processes=None, poll_s=0.05, join_timeout=None):
    """Run entropy then per-instance optimizer tasks.

    Parameters
    ----------
    entropy_fn : callable(spec) -> rec
        Must not use CUDA. Pickleable if use_processes is True.
    opt_fn : callable(spec) -> rec
        Runs on this thread (GPU-safe).
    finalize_fn : callable(spec, rec) -> rec or None
        Optional, on this thread after a successful host entropy scan
        (GPU apply_H cross-check, provenance attach, result write).
    use_processes : bool or None
        Default True when entropy_workers > 1, else a thread (still
        host-only, so it can overlap a GPU opt_fn if SciPy releases the GIL).
    join_timeout : float or None
        If set, fut.result(timeout=...) is used only as a watchdog for
        tests; production should pass None and wait for in-flight N=21.
    """
    entropy_workers = max(1, int(entropy_workers))
    if use_processes is None:
        use_processes = entropy_workers > 1
    pending = list(entropy_order)
    gpu_q = deque()
    results = []
    n_fail = n_ok = n_reused = 0
    stop = False
    next_i = 0
    in_flight = {}
    heavy = 0

    def _tally(rec):
        nonlocal n_fail, n_ok, n_reused
        results.append(rec)
        if rec.get("reused") and not rec.get("failed"):
            n_reused += 1
        elif rec.get("failed"):
            n_fail += 1
            print("FAIL", rec.get("error"), file=sys.stderr)
        else:
            n_ok += 1

    def _enqueue_opts(spec):
        for ot in opt_by_key.get((spec["N"], spec["seed"]), []) if isinstance(spec, dict) else []:
            gpu_q.append(ot)
        # also allow tuple keys already
        if isinstance(spec, dict):
            return
        for ot in opt_by_key.get(spec, []):
            gpu_q.append(ot)

    def can_submit(spec):
        if stop:
            return False
        if len(in_flight) >= entropy_workers:
            return False
        n = int(spec["N"])
        if n >= 21 and heavy:
            return False
        return True

    def _handle_entropy_result(spec, rec):
        nonlocal stop
        rec = _as_record(spec, rec)
        if not rec.get("failed") and finalize_fn is not None:
            try:
                rec = _as_record(spec, finalize_fn(spec, rec))
            except Exception as ex:
                rec = fail_record(spec, ex)
        _tally(rec)
        if rec.get("failed"):
            if halt:
                stop = True
            return
        _enqueue_opts(spec)

    pool_cm = None
    if use_processes:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        pool_cm = ProcessPoolExecutor(
            max_workers=entropy_workers,
            mp_context=ctx,
            initializer=_process_worker_init,
        )
        submit = pool_cm.submit
    else:
        pool_cm = ThreadPoolExecutor(max_workers=entropy_workers, thread_name_prefix="entropy-host")
        submit = pool_cm.submit

    try:
        while True:
            while next_i < len(pending) and can_submit(pending[next_i]):
                spec = pending[next_i]
                next_i += 1
                fut = submit(entropy_fn, spec)
                in_flight[fut] = spec
                if int(spec["N"]) >= 21:
                    heavy += 1

            finished = [f for f in list(in_flight) if f.done()]
            for fut in finished:
                spec = in_flight.pop(fut)
                if int(spec["N"]) >= 21:
                    heavy -= 1
                try:
                    rec = fut.result(timeout=0)
                except Exception as ex:
                    rec = fail_record(spec, ex)
                _handle_entropy_result(spec, rec)

            ran_gpu = False
            if gpu_q and not (halt and stop):
                spec = gpu_q.popleft()
                try:
                    rec = _as_record(spec, opt_fn(spec))
                except Exception as ex:
                    rec = fail_record(spec, ex)
                _tally(rec)
                if rec.get("failed") and halt:
                    stop = True
                ran_gpu = True

            # On halt/stop, do not require the rest of the entropy queue to be
            # submitted (that was the R2-H2 livelock: next_i < len(pending) forever).
            no_more_entropy = (not in_flight) and (next_i >= len(pending) or stop)
            no_more_gpu = (not gpu_q) or (halt and stop)
            if no_more_entropy and no_more_gpu:
                break
            if not finished and not ran_gpu:
                if in_flight:
                    wait(list(in_flight), timeout=poll_s, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(min(poll_s, 0.2))

        # Join any leftover in-flight entropy (do not orphan an N=21 scan).
        for fut, spec in list(in_flight.items()):
            try:
                rec = fut.result(timeout=join_timeout)
            except Exception as ex:
                rec = fail_record(spec, ex)
            if int(spec["N"]) >= 21:
                heavy = max(0, heavy - 1)
            in_flight.pop(fut, None)
            _handle_entropy_result(spec, rec)
    finally:
        if pool_cm is not None:
            pool_cm.shutdown(wait=True, cancel_futures=False)

    return n_fail, n_ok, n_reused, results


def _process_worker_init():
    """Spawn-child initializer. Env is also set in entropy_worker at import."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
