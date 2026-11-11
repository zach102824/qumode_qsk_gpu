#!/usr/bin/env python3
"""Kernel / energy / gradient microbenchmarks using CUDA events.

Times are end-to-end per eval (they include the host syncs in energy_grad).
No speedup versus the CPU campaign is claimed or printed.

Fail-closed: refuses to run unless GPU_TEST_PASS matches, unless you pass
--allow-unstamped (results are then labelled unvalidated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def bench(n=15, L=3, method="nov3", batch=1, repeats=20, device_id=None, warmup=5):
    from qumode_qsk_gpu import runtime, memory
    from qumode_qsk_gpu.model import Model, make_J
    from qumode_qsk_gpu.gradients import energy_grad
    from qumode_qsk_gpu.starts import make_start
    from qumode_qsk_gpu.protocol import main_seed, NS

    info = runtime.acquire(device_id=device_id, activate=True)
    seed = main_seed(n, 0) if n in NS else 0
    J = make_J(n, seed)
    m = Model(n, seed, 0.5, J=J, compute_gs=False)
    plan = memory.plan(n, L, method, info.total_bytes, requested_batch=batch)
    if plan["chosen_batch"] < batch or plan["max_batch"] < batch:
        raise SystemExit(f"requested batch={batch} does not fit; plan={plan}")
    xs = np.stack([make_start(seed, L, method, k)["x"] for k in range(batch)], axis=0)
    arg = xs if batch > 1 else xs[0]
    for _ in range(warmup):
        energy_grad(m, arg, method, L)
        runtime.synchronize()
    start, end = runtime.event_pair()
    runtime.synchronize()
    start.record()
    for _ in range(repeats):
        energy_grad(m, arg, method, L)
    end.record()
    end.synchronize()
    ms_grad = float(runtime.elapsed_time_ms(start, end)) / repeats
    start.record()
    for _ in range(repeats):
        m.energy_of_params(arg, L, method)
    end.record()
    end.synchronize()
    ms_e = float(runtime.elapsed_time_ms(start, end)) / repeats
    return dict(
        N=n, L=L, method=method, batch=batch, repeats=repeats, warmup=warmup,
        ms_per_energy_end_to_end=ms_e,
        ms_per_energy_grad_end_to_end=ms_grad,
        timing="CUDA events, end-to-end per eval (includes host syncs in the driver)",
        device=info.device_name, cupy=info.cupy_version,
        dtype="complex128", plan=plan,
        speedup_vs_cpu="not_claimed",
        raw_kernels=None,
        note="Times are CUDA-event averages on this device only. Do not advertise a speedup.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--method", choices=("nov3", "qaoa"), default="nov3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device-id", type=int, default=None)
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="Run without a GPU_TEST_PASS stamp. Results are unvalidated.")
    ap.add_argument("--allow-version-drift", action="store_true",
                    help="Allow an unofficial (version-drift) stamp.")
    args = ap.parse_args()
    from qumode_qsk_gpu import runtime, gate, kernels
    if getattr(args, 'allow_version_drift', False):
        gate.set_allow_version_drift(True)
    runtime.acquire(device_id=args.device_id, activate=True)
    kernels.ensure_compiled()
    stamped = True
    try:
        gate.assert_production_allowed()
    except Exception as ex:
        stamped = False
        if not args.allow_unstamped:
            print("benchmark is fail-closed without GPU_TEST_PASS:", ex, file=sys.stderr)
            print("Re-run after scripts/run_gpu_test.py PASSes, or pass --allow-unstamped.", file=sys.stderr)
            return 2
    rec = bench(n=args.N, L=args.L, method=args.method, batch=args.batch,
                repeats=args.repeats, device_id=args.device_id)
    rec["gpu_test_stamp_valid"] = stamped
    rec["validated"] = stamped
    kinfo = kernels.kernel_path_info()
    rec["kernel_path"] = kinfo.get("kernel_path")
    rec["raw_kernel_options"] = kinfo.get("raw_kernel_options")
    rec["raw_kernels_available"] = kernels.raw_kernels_available()
    rec["raw_kernels_error"] = kernels.raw_kernels_error()
    print(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
