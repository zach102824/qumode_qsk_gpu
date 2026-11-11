"""Fail-closed production gate.

Production jobs refuse to run unless a GPU_test PASS stamp exists and matches:
- SHA-256 of the GPU source tree and the cpu_oracle snapshot
- SHA-256 of every fixture file and of GPU_test/tolerances.json
- pinned Python / NumPy / SciPy / Numba versions (campaign box), unless an
  explicit --allow-version-drift override is recorded (results unofficial)
- kernel path (raw vs python fallback) and RawKernel compile options
- the acquired CUDA device name / compute capability
- dtypes float64 / complex128
- CuPy and CUDA runtime versions recorded on the stamp

The binding is computed once per process after acquire() and reused for
resume checks. It is re-hashed at stamp write, at round start/finish, and
on an explicit verify, so a mid-run mutation of source/fixtures/tolerances
is detected. reviews/ is not part of the source pin set.

Stamps are written only by scripts/run_gpu_test.py after all required checks
pass on the user's GPU. This package never auto-starts production work.
"""
from __future__ import annotations

import os

import json
import platform

import numpy as np
from pathlib import Path

from .io_util import sha256_file, sha256_json, save_json, now_shanghai, read_json
from .paths import ROOT, SRC, CPU_ORACLE, FIXTURES, STAMPS, PROTOCOL
from . import runtime
from .dtypes import REAL, CPLX

STAMP_NAME = "GPU_TEST_PASS.json"

_CACHE = dict(binding=None, pins_check=None, allow_version_drift=False)


def set_allow_version_drift(flag):
    """Record an explicit override. Results and stamps are marked unofficial."""
    _CACHE["allow_version_drift"] = bool(flag)
    # Do not drop a computed binding; unofficial is a flag on top of it.
    return _CACHE["allow_version_drift"]


def allow_version_drift():
    return bool(_CACHE["allow_version_drift"])


def invalidate_binding_cache():
    _CACHE["binding"] = None
    _CACHE["pins_check"] = None


def _iter_source_files():
    # reviews/ is excluded so appending ROUND*.md / FIXLOG after a PASS stamp
    # does not silently invalidate production. Review documents are not executed.
    roots = [SRC / "qumode_qsk_gpu", CPU_ORACLE, PROTOCOL, Path(ROOT) / "scripts",
             Path(ROOT) / "GPU_test" / "tests", Path(ROOT) / "docs"]
    extra = [
        Path(ROOT) / "GPU_test" / "tolerances.json",
        Path(ROOT) / "GPU_test" / "README.md",
        Path(ROOT) / "README.md",
        Path(ROOT) / "requirements.txt",
        PROTOCOL / "PROVENANCE_ADDENDUM.json",
    ]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".md", ".json", ".txt"} and "__pycache__" not in p.parts:
                if p not in seen:
                    seen.add(p)
                    yield p
    for p in extra:
        if p.exists() and p not in seen:
            seen.add(p)
            yield p


def source_pins():
    rows = []
    for p in _iter_source_files():
        rows.append(dict(
            path=str(p.relative_to(ROOT)),
            sha256=sha256_file(p),
            bytes=p.stat().st_size,
        ))
    return rows


def fixture_manifest_sha():
    path = FIXTURES / "MANIFEST.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return sha256_file(path), str(path.relative_to(ROOT))


def fixture_file_pins():
    """SHA-256 of every file under GPU_test/fixtures, plus tolerances.json."""
    rows = []
    for p in sorted(FIXTURES.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rows.append(dict(path=str(p.relative_to(ROOT)), sha256=sha256_file(p),
                             bytes=p.stat().st_size))
    tol = ROOT / "GPU_test" / "tolerances.json"
    if tol.exists():
        rec = dict(path=str(tol.relative_to(ROOT)), sha256=sha256_file(tol),
                   bytes=tol.stat().st_size)
        if rec not in rows:
            rows.append(rec)
    return rows


def verify_fixture_manifest(rows=None):
    """Re-hash every MANIFEST-listed fixture and compare. Raises on mismatch."""
    man_path = FIXTURES / "MANIFEST.json"
    man = read_json(man_path)
    failures = []
    listed = list(man.get("copied_files") or []) + list(man.get("starts_slices") or [])
    for rec in listed:
        rel = rec.get("bundle_path")
        if not rel:
            continue
        path = ROOT / rel
        if not path.exists():
            failures.append(dict(path=rel, error="missing"))
            continue
        got = sha256_file(path)
        want = rec.get("sha256")
        if got != want:
            failures.append(dict(path=rel, manifest=want, actual=got))
    cat = man.get("validation_catalog")
    if cat:
        path = ROOT / cat
        got = sha256_file(path)
        if got != man.get("validation_catalog_sha256"):
            failures.append(dict(path=cat, manifest=man.get("validation_catalog_sha256"), actual=got))
    return dict(ok=not failures, failures=failures,
                manifest_sha256=sha256_file(man_path),
                n_listed=len(listed))


def campaign_library_pins():
    """Pinned Python/NumPy/SciPy/Numba from the provenance addendum."""
    path = PROTOCOL / "PROVENANCE_ADDENDUM.json"
    if not path.exists():
        raise RuntimeError("missing protocol/PROVENANCE_ADDENDUM.json (library pins)")
    doc = read_json(path)
    pins = doc.get("campaign_library_versions_on_authoring_box") or {}
    out = dict(
        python=pins.get("python"),
        numpy=pins.get("numpy"),
        scipy=pins.get("scipy"),
        numba=pins.get("numba"),
    )
    if not all(out.values()):
        raise RuntimeError(f"incomplete campaign library pins in addendum: {out}")
    return out


def host_library_versions():
    """Python / NumPy / SciPy / Numba on this interpreter (no CUDA required)."""
    import numpy
    import scipy
    try:
        import numba
        numba_v = getattr(numba, "__version__", None)
    except Exception:
        numba_v = None
    return dict(
        python=platform.python_version(),
        numpy=getattr(numpy, "__version__", None),
        scipy=getattr(scipy, "__version__", None),
        numba=numba_v,
    )


def check_library_pins(allow_drift=None):
    """Refuse unless host versions equal the campaign pins, or drift is allowed."""
    pins = campaign_library_pins()
    have = host_library_versions()
    mismatches = []
    for key in ("python", "numpy", "scipy", "numba"):
        if have.get(key) != pins.get(key):
            mismatches.append({key: dict(pinned=pins.get(key), have=have.get(key))})
    allow = _CACHE["allow_version_drift"] if allow_drift is None else bool(allow_drift)
    unofficial = bool(mismatches)
    rec = dict(pins=pins, have=have, mismatches=mismatches,
               unofficial=unofficial, allow_version_drift=allow)
    _CACHE["pins_check"] = rec
    if mismatches and not allow:
        raise RuntimeError(
            "library versions do not match the campaign pins in "
            "protocol/PROVENANCE_ADDENDUM.json: " + json.dumps(mismatches) +
            ". Install the pinned requirements (and a Python " + str(pins["python"]) +
            ") or pass --allow-version-drift (stamp and results are then unofficial)."
        )
    return rec


def _kernel_fields():
    from . import kernels
    return kernels.kernel_path_info()


def _compute_binding():
    if not runtime.is_acquired():
        raise RuntimeError("runtime.acquire() before building a stamp binding")
    # Force the RawKernel compile/fallback decision so the stamp records the
    # path that will actually run, and mixed raw/python use is rejected later.
    from . import kernels
    kernels.ensure_compiled()
    pins = check_library_pins()
    info = runtime.info()
    man_sha, man_path = fixture_manifest_sha()
    src_pins = source_pins()
    fix_pins = fixture_file_pins()
    versions = runtime.library_versions()
    kinfo = _kernel_fields()
    tol = ROOT / "GPU_test" / "tolerances.json"
    host = pins["have"]
    binding = dict(
        source_pins=src_pins,
        source_pins_sha256=sha256_json(src_pins),
        fixture_manifest=man_path,
        fixture_manifest_sha256=man_sha,
        fixture_file_pins=fix_pins,
        fixture_files_sha256=sha256_json(fix_pins),
        tolerances_path="GPU_test/tolerances.json" if tol.exists() else None,
        tolerances_sha256=sha256_file(tol) if tol.exists() else None,
        device_id=info.device_id,
        device_name=info.device_name,
        compute_capability=info.compute_capability,
        cupy_version=info.cupy_version,
        cuda_version=info.cuda_version,
        numpy_version=host.get("numpy") or versions.get("numpy"),
        scipy_version=host.get("scipy") or versions.get("scipy"),
        numba_version=host.get("numba"),
        python_version=host.get("python"),
        pinned_numpy=pins["pins"]["numpy"],
        pinned_scipy=pins["pins"]["scipy"],
        pinned_numba=pins["pins"]["numba"],
        pinned_python=pins["pins"]["python"],
        unofficial=bool(pins["unofficial"]),
        allow_version_drift=bool(pins["allow_version_drift"]),
        version_mismatches=pins["mismatches"],
        kernel_path=kinfo.get("kernel_path"),
        raw_kernel_options=kinfo.get("raw_kernel_options"),
        raw_kernels_error=kinfo.get("raw_kernels_error"),
        dtype_real=np.dtype(REAL).name,
        dtype_cplx=np.dtype(CPLX).name,
        backend=info.backend,
    )
    binding["binding_sha256"] = sha256_json({k: v for k, v in binding.items() if k != "binding_sha256"})
    return binding


def current_binding(force_refresh=False):
    if _CACHE["binding"] is not None and not force_refresh:
        return _CACHE["binding"]
    binding = _compute_binding()
    _CACHE["binding"] = binding
    return binding


def verify_cached_binding():
    """Re-hash source/fixtures/tolerances/versions and compare to the cache.

    Call at stamp write, round start, and round finish. Mid-run resume checks
    use the cached SHA so a 12k-file official_complete does not re-hash the
    tree on every record.
    """
    if _CACHE["binding"] is None:
        return current_binding(force_refresh=True)
    cached = _CACHE["binding"]
    fresh = _compute_binding()
    if fresh.get("binding_sha256") != cached.get("binding_sha256"):
        _CACHE["binding"] = fresh
        raise RuntimeError(
            "source, fixtures, tolerances, kernel path, or library versions "
            "changed after the binding was cached: "
            + json.dumps(dict(cached=cached.get("binding_sha256"),
                              current=fresh.get("binding_sha256")))
        )
    return cached


def stamp_path():
    return Path(os.environ.get("QSK_VALIDATION_STAMPS", str(STAMPS))) / STAMP_NAME


def write_pass_stamp(test_report):
    STAMPS.mkdir(parents=True, exist_ok=True)
    binding = current_binding(force_refresh=True)
    unofficial = bool(binding.get("unofficial") or binding.get("allow_version_drift"))
    doc = dict(
        verdict="PASS",
        unofficial=unofficial,
        shanghai=now_shanghai(),
        binding=binding,
        test_report=test_report,
        note=(
            "Production may run only while this stamp still matches the current "
            "source, fixture files, tolerances, kernel path, device name, dtypes, "
            "and the Python/NumPy/SciPy/Numba/CuPy/CUDA versions recorded here. "
            "An unofficial (version-drift) stamp does not authorize official results."
        ),
    )
    if stamp_path().exists():
        import uuid
        history = stamp_path().parent / "history" / (uuid.uuid4().hex + ".json")
        save_json(history, read_json(stamp_path()))
    save_json(stamp_path(), doc)
    return doc


def load_stamp():
    path = stamp_path()
    if not path.exists():
        return None
    return read_json(path)


_COMPARE_KEYS = (
    "source_pins_sha256", "fixture_manifest_sha256", "fixture_files_sha256",
    "tolerances_sha256", "device_id", "device_name", "dtype_real", "dtype_cplx", "backend",
    "cupy_version", "cuda_version", "compute_capability",
    "numpy_version", "scipy_version", "numba_version", "python_version",
    "kernel_path", "raw_kernel_options",
)


def assert_production_allowed(allow_unofficial=None):
    """Raise if production must stay closed."""
    if not runtime.is_acquired():
        raise RuntimeError("runtime.acquire() first")
    info = runtime.info()
    if np.dtype(REAL) != np.dtype("float64") or np.dtype(CPLX) != np.dtype("complex128"):
        raise RuntimeError("production requires float64/complex128")
    if info.dtype_real != "float64" or info.dtype_cplx != "complex128":
        raise RuntimeError("runtime dtypes are not float64/complex128")
    check_library_pins()
    stamp = load_stamp()
    if stamp is None:
        raise RuntimeError(
            "No GPU_TEST_PASS stamp. Run scripts/run_gpu_test.py on this GPU "
            "and obtain verdict=PASS before production. Fail-closed."
        )
    if stamp.get("verdict") != "PASS":
        raise RuntimeError(f"stamp verdict is {stamp.get('verdict')!r}, not PASS")
    current = current_binding(force_refresh=False if _CACHE["binding"] is not None else True)
    # cheap mutation check of versions/tolerances is included when verify is called
    # by the runner; here compare the (possibly cached) binding to the stamp.
    want = stamp.get("binding") or {}
    mismatches = []
    for key in _COMPARE_KEYS:
        if want.get(key) != current.get(key):
            mismatches.append({key: dict(stamp=want.get(key), current=current.get(key))})
    if mismatches:
        raise RuntimeError(
            "GPU_TEST_PASS stamp does not match this source/fixtures/tolerances/"
            "kernel path/device/dtype/library versions: " + json.dumps(mismatches)
        )
    unofficial = bool(stamp.get("unofficial") or want.get("unofficial") or want.get("allow_version_drift"))
    allow = _CACHE["allow_version_drift"] if allow_unofficial is None else bool(allow_unofficial)
    if unofficial and not allow:
        raise RuntimeError(
            "GPU_TEST_PASS is unofficial (library version drift was allowed at "
            "test time). Re-run GPU_test on the pinned stack, or pass "
            "--allow-version-drift again; results will remain unofficial."
        )
    return stamp


def provenance_fields():
    """Small dict stored on every production result for resume checks.

    Raises if the runtime is not acquired: a partial dict without
    binding_sha256 would make every previous result look non-resumable (R2-H3).
    """
    if not runtime.is_acquired():
        raise RuntimeError(
            "provenance_fields() requires runtime.acquire(); a partial dict "
            "would break resume and official_complete"
        )
    versions = runtime.library_versions()
    host = host_library_versions()
    b = current_binding()
    rec = dict(
        numpy_version=b.get("numpy_version") or versions.get("numpy") or host.get("numpy"),
        scipy_version=b.get("scipy_version") or versions.get("scipy") or host.get("scipy"),
        numba_version=b.get("numba_version") or host.get("numba"),
        python_version=b.get("python_version") or host.get("python"),
        cupy_version=versions.get("cupy") or b.get("cupy_version"),
        cuda_version=versions.get("cuda_runtime") or b.get("cuda_version"),
        device_name=versions.get("device_name") or b.get("device_name"),
        dtype_real=np.dtype(REAL).name,
        dtype_cplx=np.dtype(CPLX).name,
        kernel_path=b.get("kernel_path"),
        raw_kernel_options=b.get("raw_kernel_options"),
        unofficial=bool(b.get("unofficial") or b.get("allow_version_drift")),
        binding_sha256=b.get("binding_sha256"),
        source_pins_sha256=b.get("source_pins_sha256"),
        fixture_manifest_sha256=b.get("fixture_manifest_sha256"),
    )
    return rec
