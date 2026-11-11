#!/usr/bin/env python3
"""AST / syntax / import-safety checks. No CUDA, no campaign writes, no heavy compute."""
from __future__ import annotations

import ast
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_IMPORT_TIME = (
    "cuda.Device(",
    "cupy.cuda.Device(",
    "cp.cuda.Device(",
    ".use()",
)

# Modules that may mention Device() — only inside functions, checked via AST.


def python_files():
    files = []
    for sub in ("src", "scripts", "GPU_test/tests", "cpu_oracle"):
        d = ROOT / sub
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return files


def check_compile(paths):
    errors = []
    for p in paths:
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as ex:
            errors.append(dict(path=str(p.relative_to(ROOT)), error=repr(ex)))
    return errors


def _calls_name(node, name):
    # Device(...) or cuda.Device(...)
    func = node.func
    if isinstance(func, ast.Name) and func.id == name:
        return True
    if isinstance(func, ast.Attribute) and func.attr == name:
        return True
    return False


def check_no_import_time_cuda(paths):
    """Flag Device() / .use() at module top level (not inside a function/class)."""
    findings = []
    for p in paths:
        if "cpu_oracle" in p.parts:
            continue
        src = p.read_text()
        tree = ast.parse(src, filename=str(p))
        for node in tree.body:
            # skip function and class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and (
                    _calls_name(child, "Device") or _calls_name(child, "use")
                ):
                    findings.append(dict(
                        path=str(p.relative_to(ROOT)),
                        lineno=getattr(child, "lineno", None),
                        msg="CUDA Device/use at import-time (module top level)",
                    ))
    return findings


def check_imports_without_acquire():
    sys.path.insert(0, str(ROOT / "src"))
    import qumode_qsk_gpu
    import qumode_qsk_gpu.runtime as rt
    import qumode_qsk_gpu.kernels  # noqa: F401
    import qumode_qsk_gpu.model  # noqa: F401
    import qumode_qsk_gpu.gradients  # noqa: F401
    import qumode_qsk_gpu.jobs  # noqa: F401
    import qumode_qsk_gpu.gate  # noqa: F401
    if rt.is_acquired():
        return [dict(msg="runtime was acquired as a side effect of import")]
    return []


def check_starts_against_catalog():
    """Pure-numpy start recipe vs copied catalog (no Numba, no states)."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from qumode_qsk_gpu.starts import spsa_starts, make_start
    cat = json.loads((ROOT / "GPU_test/fixtures/validation_catalog.json").read_text())
    n = 0
    for e in cat["entries"]:
        if e["kind"] == "spsa_fixed_params":
            xs, _ = spsa_starts(e["seed"], e["L"], e["method"])
            for a, b in zip(xs, e["initial_points"]):
                if not np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)):
                    return [dict(msg="spsa start mismatch", seed=e["seed"], L=e["L"], method=e["method"])], n
            n += 2
        elif e["kind"] == "bfgs_fixed_params":
            st = make_start(e["seed"], e["L"], e["method"], e["start_index"])
            if st["x_sha256"] != e["start_x_sha256"]:
                return [dict(msg="bfgs start sha mismatch", entry=e["result_file"])], n
            n += 1
    return [], n


def main():
    paths = python_files()
    report = dict(n_files=len(paths), compile_errors=[], import_time_cuda=[],
                  import_side_effects=[], starts_errors=[], starts_checked=0)
    report["compile_errors"] = check_compile(paths)
    report["import_time_cuda"] = check_no_import_time_cuda(paths)
    try:
        report["import_side_effects"] = check_imports_without_acquire()
    except Exception as ex:
        report["import_side_effects"] = [dict(error=repr(ex))]
    try:
        err, n = check_starts_against_catalog()
        report["starts_errors"] = err
        report["starts_checked"] = n
    except Exception as ex:
        report["starts_errors"] = [dict(error=repr(ex))]
    ok = not (report["compile_errors"] or report["import_time_cuda"]
              or report["import_side_effects"] or report["starts_errors"])
    report["verdict"] = "PASS" if ok else "FAIL"
    report["cuda_validated"] = False
    report["note"] = "Syntax/AST/import/start-hash checks only. GPU numerics are not validated here."
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
