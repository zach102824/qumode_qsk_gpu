#!/usr/bin/env python3
"""GPU production runner. Default is a dry run. Never auto-starts.

Writes only under this repository. Refuses to start if GPU_TEST_PASS does
not match this source, fixtures, tolerances, kernel path, device, dtypes
and library versions. Does not write into a CPU campaign tree (a sibling
qumode_qsk directory, or $QUMODE_QSK_CPU_CAMPAIGN, if present).

Entropy scans run on the host only (process pool when --entropy-workers > 1,
spawn, BLAS threads = 1, at most one N>=21 scan at a time). Small-N scans
are started first so GPU optimizer work can begin before any N=21 eigensolve.
The optional apply_H cross-check and all optimizer tasks run on this thread.
Independent task FAILs are recorded and the run continues; the process exits
nonzero at the end. Seeds are never replaced.

Usage
-----
    python3 scripts/run_production.py --dry-run --round 1
    python3 scripts/run_production.py --run --round 1 --device-id 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ROOT only — never cpu_oracle/src (breaks relative imports / shadows names).
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=None,
                    help="Plan only (default unless --run is given). No CUDA.")
    ap.add_argument("--run", action="store_true",
                    help="Actually execute tasks. Required to leave dry-run mode.")
    ap.add_argument("--round", type=int, default=1, help="Official 10-per-N round id (1..20).")
    ap.add_argument("--device-id", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on the total number of planned tasks (entropy + SPSA + BFGS packs), in pipeline order.")
    ap.add_argument("--entropy-workers", type=int, default=1,
                    help="CPU processes for host-only entropy scans (default 1). "
                         "N>=21 is always serialized. Workers never use CUDA.")
    ap.add_argument("--force-unordered", action="store_true",
                    help="Allow starting round r before round r-1 is complete (not recommended).")
    ap.add_argument("--halt-on-fail", action="store_true",
                    help="Stop starting new work at the first FAIL (in-flight entropy, including N=21, is joined). Default is to record FAIL and continue.")
    ap.add_argument("--allow-version-drift", action="store_true",
                    help="Allow Python/NumPy/SciPy/Numba to differ from the campaign pins. "
                         "Stamp/results are unofficial.")
    args = ap.parse_args()
    dry = True if not args.run else False
    if args.run and args.dry_run:
        raise SystemExit("choose one of --run or --dry-run")
    halt = bool(args.halt_on_fail)
    workers = max(1, int(args.entropy_workers))

    from qumode_qsk_gpu import runtime, gate, jobs, memory
    from qumode_qsk_gpu.protocol import NS, DEPTHS
    from qumode_qsk_gpu.summaries import summarize_round, write_round_manifest
    from qumode_qsk_gpu.pipeline import run_pipeline

    if args.allow_version_drift:
        gate.set_allow_version_drift(True)

    plan = jobs.plan_round(args.round)
    counts = dict(entropy=len(plan["entropy"]), spsa=len(plan["spsa"]),
                  bfgs_pack=len(plan["bfgs_pack"]),
                  indices=plan["indices"], Ns=list(NS), depths=list(DEPTHS))
    print("plan:", json.dumps(counts, indent=2))
    if dry:
        print("dry-run: no jobs executed, no CUDA acquire required, production remains untouched.")
        print("To run after GPU_test PASS: python3 scripts/run_production.py --run --round", args.round, "--device-id 0")
        return 0

    # Acquire and bind BEFORE official_complete / resume checks (R2-H3).
    info = runtime.acquire(device_id=args.device_id, activate=True)
    stamp = gate.assert_production_allowed()
    gate.verify_cached_binding()
    print("gate open:", stamp["binding"]["device_name"], stamp["binding"]["dtype_cplx"],
          "kernel_path=", stamp["binding"].get("kernel_path"),
          "unofficial=", bool(stamp.get("unofficial") or stamp["binding"].get("unofficial")))
    print("memory example N=21 L=6:", json.dumps(memory.plan(21, 6, "nov3", info.total_bytes), indent=2))

    if args.round > 1 and not args.force_unordered:
        prev = jobs.official_complete(args.round - 1)
        if not prev["complete"]:
            print(f"round {args.round-1} is incomplete ({prev['n_missing']} missing). "
                  f"Finish it first, or pass --force-unordered.", file=sys.stderr)
            return 2

    man_path, man = write_round_manifest(args.round, stamp.get("binding"))
    print("round manifest:", man_path)

    entropy_by_key = {(t["N"], t["seed"]): t for t in plan["entropy"]}
    opt_by_key = {}
    for t in plan["spsa"] + plan["bfgs_pack"]:
        opt_by_key.setdefault((t["N"], t["seed"]), []).append(t)
    # Small-N first so GPU work is not blocked behind the first N=21 scan.
    entropy_order = [entropy_by_key[k] for k in sorted(entropy_by_key, key=lambda k: (k[0], k[1]))]
    if args.limit is not None:
        flat = list(entropy_order)
        for spec in entropy_order:
            flat.extend(opt_by_key.get((spec["N"], spec["seed"]), []))
        allowed = set(id(s) for s in flat[: int(args.limit)])
        entropy_order = [s for s in entropy_order if id(s) in allowed]
        for k in list(opt_by_key):
            opt_by_key[k] = [s for s in opt_by_key[k] if id(s) in allowed]

    from qumode_qsk_gpu.entropy_worker import cpu_entropy_scan_only

    last_n = {"N": None}

    def opt_fn(spec):
        rec = _dispatch_record(spec)
        n = int(spec.get("N") or 0)
        if last_n["N"] is not None and n != last_n["N"]:
            try:
                runtime.free_memory_pool()
            except Exception:
                pass
        last_n["N"] = n
        return rec

    n_fail = n_ok = n_reused = 0
    results = []
    extra = None
    try:
        n_fail, n_ok, n_reused, results = run_pipeline(
            entropy_order, opt_by_key, halt=halt, entropy_workers=workers,
            entropy_fn=cpu_entropy_scan_only,
            opt_fn=opt_fn,
            finalize_fn=jobs.finalize_entropy_gpu,
            use_processes=(workers > 1),
        )
        try:
            gate.verify_cached_binding()
        except Exception as ex:
            print("binding changed before finish:", ex, file=sys.stderr)
            extra = repr(ex)
            n_fail += 1
    except Exception as ex:
        print("pipeline aborted:", ex, file=sys.stderr)
        extra = repr(ex)
        n_fail += 1
        results.append(dict(failed=True, error=repr(ex), verdict="FAIL"))
    _finish(args.round, n_ok, n_fail, n_reused, results, man_path, extra_error=extra)
    return 1 if n_fail else 0


def _dispatch_record(spec):
    from qumode_qsk_gpu import jobs
    existing = jobs.load_existing(spec)
    if existing is not None:
        return dict(spec=spec, reused=True, result_file=existing.get("result_file"),
                    failed=False, verdict="ok")
    rec = jobs.dispatch(spec, require_gate=True)
    failed = rec.get("verdict") == "FAIL" or rec.get("error")
    return dict(spec={k: spec[k] for k in spec},
                verdict="FAIL" if failed else "ok",
                result_file=rec.get("result_file"),
                error=rec.get("error"), failed=failed, reused=False)


def _finish(round_id, n_ok, n_fail, n_reused, results, man_path, extra_error=None):
    from qumode_qsk_gpu.summaries import summarize_round
    from qumode_qsk_gpu.io_util import save_json, read_json
    try:
        man = read_json(man_path)
        man.update(status="finished", n_ok=n_ok, n_fail=n_fail, n_reused=n_reused)
        if extra_error:
            man["error"] = extra_error
        save_json(man_path, man)
    except Exception as ex:
        print("manifest update failed:", ex, file=sys.stderr)
    try:
        summary = summarize_round(round_id)
        print("summary:", summary.get("summary_file"), "complete", summary.get("official_complete"))
    except Exception as ex:
        print("summary failed:", ex, file=sys.stderr)
    print(json.dumps(dict(n=len(results), n_ok=n_ok, n_fail=n_fail, n_reused=n_reused,
                          results=results[:20], truncated=len(results) > 20), indent=2))


if __name__ == "__main__":
    from qumode_qsk_gpu.dual import lock
    with lock(ROOT / "runs" / "dual_gpu" / "campaign.lock"):
        raise SystemExit(main())
