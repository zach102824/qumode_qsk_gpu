"""Lazy CUDA runtime.

Importing this module must not:
- import cupy until acquire() is called
- allocate device memory
- call cupy.cuda.Device().use() or change the current device
- set a default device from this library

Scripts (run_gpu_test, run_production, benchmark) must call
acquire(..., activate=True) or activate_acquired_device() so later
allocations go to the chosen device. Library import never does this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dtypes import REAL, CPLX


_STATE = {
    "acquired": False,
    "module": None,
    "device_id": None,
    "device_name": None,
    "backend": None,
    "info": None,
    "activated": False,
    "kernels": None,
}


class RuntimeError_(RuntimeError):
    pass


def cupy_is_importable():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError_(
            "CuPy could not be imported. Install a wheel that matches your CUDA "
            "toolkit (see README.md). This box may have no GPU; that is expected "
            "during the code-only authoring step."
        ) from exc
    return cp


@dataclass
class RuntimeInfo:
    backend: str
    device_id: int
    device_name: str
    cuda_version: Optional[str]
    cupy_version: str
    runtime_version: Optional[str]
    compute_capability: Optional[str]
    total_bytes: Optional[int]
    fp64_support: str
    dtype_real: str
    dtype_cplx: str
    numpy_version: str
    scipy_version: str
    activated: bool
    note: str


def _lib_versions():
    import numpy
    import scipy
    return str(numpy.__version__), str(scipy.__version__)


def acquire(device_id=None, require_device=True, activate=False):
    """Initialize CuPy.

    Importing this package never calls this function.

    If ``activate=True`` (scripts only), pin the process current device with
    ``Device(k).use()`` so subsequent ``xp.empty`` allocations land on k.
    Default is False so accidental library use does not change the process
    device. Tests and production scripts should pass activate=True, or call
    ``activate_acquired_device()``, and also wrap work in ``device_context()``.
    """
    if _STATE["acquired"] and (device_id is None or device_id == _STATE["device_id"]):
        if activate and not _STATE["activated"]:
            activate_acquired_device()
        return _STATE["info"]
    cp = _import_cupy()
    if device_id is not None:
        dev = cp.cuda.Device(int(device_id))
        with dev:
            info = _query(cp, int(device_id), activated=False)
        _STATE["device_id"] = int(device_id)
    else:
        cur = int(cp.cuda.Device().id)
        info = _query(cp, cur, activated=False)
        _STATE["device_id"] = cur
    _STATE["acquired"] = True
    _STATE["module"] = cp
    _STATE["backend"] = "cupy"
    _STATE["device_name"] = info.device_name
    _STATE["info"] = info
    _STATE["activated"] = False
    _STATE["kernels"] = None
    if require_device and info.total_bytes is None:
        raise RuntimeError_("CUDA device memory query failed")
    if activate:
        activate_acquired_device()
        # refresh activated flag in info
        info = _STATE["info"]
    return info


def activate_acquired_device():
    """Pin the process current device to the acquired id. Scripts only.

    This is the H6 fix: without it, Device(k) is used only for queries and
    uploads, and every other allocation uses the process current device
    (often 0).
    """
    if not _STATE["acquired"]:
        raise RuntimeError_("runtime.acquire() has not been called")
    cp = _STATE["module"]
    cp.cuda.Device(int(_STATE["device_id"])).use()
    _STATE["activated"] = True
    if _STATE["info"] is not None:
        _STATE["info"].activated = True
    return _STATE["device_id"]


def _query(cp, device_id: int, activated: bool) -> RuntimeInfo:
    dev = cp.cuda.Device(device_id)
    with dev:
        try:
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            cc = "%d.%d" % (props.get("major", 0), props.get("minor", 0))
            total = int(props.get("totalGlobalMem", 0)) or None
        except Exception:
            name = f"cuda:{device_id}"
            cc = None
            total = None
        try:
            cuda_ver = str(cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            cuda_ver = None
        try:
            rt_ver = str(cp.cuda.runtime.driverGetVersion())
        except Exception:
            rt_ver = None
        npv, spv = _lib_versions()
        fp64 = (
            "used exclusively (complex128/float64); throughput depends on the "
            "GPU. Consumer parts with weak FP64 may be correct but slow. "
            "No speedup is claimed from this package."
        )
        return RuntimeInfo(
            backend="cupy",
            device_id=int(device_id),
            device_name=str(name),
            cuda_version=cuda_ver,
            cupy_version=str(cp.__version__),
            runtime_version=rt_ver,
            compute_capability=cc,
            total_bytes=total,
            fp64_support=fp64,
            dtype_real=str(REAL),
            dtype_cplx=str(CPLX),
            numpy_version=npv,
            scipy_version=spv,
            activated=bool(activated),
            note="FP64/complex128 only; no autocast; no import-time allocation.",
        )


def is_acquired():
    return bool(_STATE["acquired"])


def xp():
    if not _STATE["acquired"] or _STATE["module"] is None:
        raise RuntimeError_("runtime.acquire() has not been called")
    return _STATE["module"]


def device_id():
    if not _STATE["acquired"]:
        raise RuntimeError_("runtime.acquire() has not been called")
    return _STATE["device_id"]


def device_name():
    if not _STATE["acquired"]:
        raise RuntimeError_("runtime.acquire() has not been called")
    return _STATE["device_name"]


def info():
    if not _STATE["acquired"]:
        raise RuntimeError_("runtime.acquire() has not been called")
    return _STATE["info"]


def device_context():
    """Context manager pinning work to the acquired device, then restoring."""
    cp = xp()
    return cp.cuda.Device(device_id())


def is_device_array(x):
    """True only for a CuPy ndarray. Never use hasattr(x, 'device') (NumPy 2)."""
    if not _STATE["acquired"] or _STATE["module"] is None:
        return False
    return isinstance(x, _STATE["module"].ndarray)


def as_host_array(x, dtype=None):
    """Return a contiguous NumPy array. Downloads if x is a CuPy ndarray."""
    if is_device_array(x):
        x = to_host(x)
    if dtype is None:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(x, dtype=dtype)


def synchronize():
    """Explicit device sync. Do not call inside hot untimed kernels."""
    with device_context():
        xp().cuda.runtime.deviceSynchronize()


def event_pair():
    """CUDA events for benchmark timing. Not used to invent speedup claims."""
    cp = xp()
    return cp.cuda.Event(), cp.cuda.Event()


def elapsed_time_ms(start, end):
    """Elapsed milliseconds between two recorded events.

    CuPy Events have no .elapsed_time method; the documented API is
    cupy.cuda.get_elapsed_time(start, end).
    """
    cp = xp()
    return float(cp.cuda.get_elapsed_time(start, end))


def to_device(host, dtype=None):
    """Upload an array once. Callers must keep constants on device afterwards."""
    cp = xp()
    if is_device_array(host):
        if dtype is not None and str(host.dtype) != str(np.dtype(dtype)):
            with device_context():
                return host.astype(dtype, copy=False)
        return host
    arr = np.ascontiguousarray(host)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    with device_context():
        return cp.asarray(arr)


def to_host(dev):
    """Download. Use only for small scalars/vectors or test comparisons."""
    if is_device_array(dev):
        return np.ascontiguousarray(dev.get())
    return np.ascontiguousarray(dev)


def as_scalar(dev):
    """Copy one value back. Accepts 0-d or size-1 arrays."""
    v = to_host(dev)
    return np.asarray(v).reshape(-1)[0]


def pack_to_host(*arrays):
    """One device-to-host transfer of several small 1-d arrays, concatenated."""
    cp = xp()
    parts = []
    shapes = []
    for a in arrays:
        a = cp.asarray(a)
        flat = a.reshape(-1)
        parts.append(flat)
        shapes.append(a.shape)
    with device_context():
        packed = cp.concatenate(parts)
        host = packed.get()
    out = []
    off = 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(np.ascontiguousarray(host[off:off + n].reshape(sh)))
        off += n
    return out


def library_versions():
    npv, spv = _lib_versions()
    import platform
    try:
        import numba
        numba_v = getattr(numba, "__version__", None)
    except Exception:
        numba_v = None
    out = dict(numpy=npv, scipy=spv, python=platform.python_version(),
               numba=numba_v, dtype_real=str(REAL), dtype_cplx=str(CPLX))
    if _STATE["acquired"] and _STATE["info"] is not None:
        info = _STATE["info"]
        out.update(
            cupy=info.cupy_version,
            cuda_runtime=info.cuda_version,
            cuda_driver=info.runtime_version,
            compute_capability=info.compute_capability,
            device_name=info.device_name,
            device_id=info.device_id,
        )
    else:
        out.update(cupy=None, cuda_runtime=None, cuda_driver=None,
                   compute_capability=None, device_name=None, device_id=None)
    return out


def free_memory_pool():
    """Release the current-device CuPy memory pool. Safe no-op if not acquired."""
    if not is_acquired():
        return
    cp = xp()
    try:
        with device_context():
            cp.get_default_memory_pool().free_all_blocks()
            pinned = getattr(cp, "get_default_pinned_memory_pool", None)
            if callable(pinned):
                pinned().free_all_blocks()
    except Exception:
        pass


def reset_for_tests():
    """Drop the acquired handle. Does not close the CUDA context or free pools."""
    _STATE.update(acquired=False, module=None, device_id=None,
                  device_name=None, backend=None, info=None,
                  activated=False, kernels=None)
