#!/usr/bin/env python3
"""First command to run on the GPU host, after installing the pinned stack + CuPy.

Does not write into a CPU campaign tree. Does not start production.
On PASS, writes stamps/GPU_TEST_PASS.json bound to source, fixture files,
tolerances, kernel path, device name, dtypes, and library versions.

A PASS is necessary for production. It is not a proof of bitwise CPU
identity or of any speedup. Static/AST checks cannot prove CUDA correctness.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPU_test" / "tests"))


def _write(path, doc, code):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")
    return code


def main():
    ap = argparse.ArgumentParser(description="Run GPU_test against copied CPU fixtures.")
    ap.add_argument("--device-id", type=int, default=None)
    ap.add_argument("--max-bfgs-starts-per-cell", type=int, default=2)
    ap.add_argument("--out", type=str, default=str(ROOT / "GPU_test" / "last_report.json"))
    ap.add_argument("--import-only", action="store_true",
                    help="Only check that the package imports without acquiring CUDA.")
    ap.add_argument("--host-only", action="store_true",
                    help="Run host contracts only (no CUDA).")
    ap.add_argument("--allow-version-drift", action="store_true",
                    help="Allow Python/NumPy/SciPy/Numba to differ from the campaign pins. "
                         "The stamp is then unofficial.")
    args = ap.parse_args()

    from test_import_no_cuda import test_import_does_not_touch_cuda
    imp = test_import_does_not_touch_cuda()
    print("import_no_cuda:", json.dumps(imp))
    if args.import_only:
        return 0

    from qumode_qsk_gpu import gate
    if args.allow_version_drift:
        gate.set_allow_version_drift(True)
    try:
        pins = gate.check_library_pins()
        print("library_pins:", json.dumps({k: pins[k] for k in pins if k != "pins"}, indent=2))
    except Exception as ex:
        doc = dict(verdict="FAIL", stage="library_pins", error=repr(ex),
                   traceback=traceback.format_exc())
        print("library_pins FAILED:", ex, file=sys.stderr)
        return _write(args.out, doc, 1)

    from test_host_contracts import run_all_host
    try:
        hostc = run_all_host()
        print("host_contracts:", json.dumps({k: hostc[k] for k in hostc if k != "checks"}, indent=2))
        if hostc.get("verdict") != "PASS":
            return _write(args.out, dict(verdict="FAIL", stage="host_contracts", report=hostc), 1)
    except Exception as ex:
        doc = dict(verdict="FAIL", stage="host_contracts", error=repr(ex),
                   traceback=traceback.format_exc())
        print("host_contracts FAILED:", ex, file=sys.stderr)
        return _write(args.out, doc, 1)

    from test_host_oracle_light import test_starts_and_diags
    try:
        host = test_starts_and_diags()
        print("host_oracle_light:", json.dumps({k: host[k] for k in host if k != "instances"}))
        if not host.get("ok"):
            raise SystemExit("host oracle / start-recipe check failed")
    except Exception as ex:
        print("host_oracle_light FAILED:", ex, file=sys.stderr)
        traceback.print_exc()
        return _write(args.out, dict(verdict="FAIL", stage="host_oracle_light",
                                     error=repr(ex), traceback=traceback.format_exc()), 1)

    if args.host_only:
        return _write(args.out, dict(verdict="PASS", stage="host_only",
                                     host_contracts= {k: hostc[k] for k in hostc if k != "checks"},
                                     host_oracle_light={k: host[k] for k in host if k != "instances"},
                                     note="Host-only; CUDA was not executed. Not a production gate."), 0)

    from test_gpu_validation import run_all
    try:
        report = run_all(device_id=args.device_id,
                         max_bfgs_starts_per_cell=args.max_bfgs_starts_per_cell)
    except Exception as ex:
        report = dict(verdict="FAIL", stage="run_all", error=repr(ex),
                      traceback=traceback.format_exc(),
                      note="Exception escaped the per-check wrappers.")
        print(json.dumps({k: report[k] for k in report if k != "traceback"}, indent=2))
        return _write(args.out, report, 1)
    if report.get("binding_unofficial") or gate.allow_version_drift():
        report["unofficial"] = True
    brief = {k: report[k] for k in report if k not in ("checks",)}
    print(json.dumps(brief, indent=2))
    print("wrote", args.out)
    if report.get("verdict") != "PASS":
        print("GPU_TEST FAILED. Production remains closed.", file=sys.stderr)
        return _write(args.out, report, 1)
    print("GPU_TEST PASS. Production gate is open for this source/fixtures/tolerances/kernel path/device/dtype/libraries.")
    if report.get("unofficial"):
        print("Stamp is unofficial (version drift was allowed).", file=sys.stderr)
    return _write(args.out, report, 0)


if __name__ == "__main__":
    raise SystemExit(main())
