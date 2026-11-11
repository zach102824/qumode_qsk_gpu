"""Importing the package must not initialize CUDA or allocate a device."""
from __future__ import annotations

import sys
from conftest_paths import setup_paths
setup_paths()


def test_import_does_not_touch_cuda():
    # If cupy is present, record device count before and after import.
    pre = None
    try:
        import cupy
        pre = "cupy_already_imported"
    except Exception:
        pre = "cupy_not_imported"
    import qumode_qsk_gpu
    import qumode_qsk_gpu.protocol
    import qumode_qsk_gpu.dtypes
    import qumode_qsk_gpu.paths
    import qumode_qsk_gpu.io_util
    import qumode_qsk_gpu.memory
    import qumode_qsk_gpu.runtime as rt
    assert rt.is_acquired() is False
    assert rt._STATE["module"] is None
    # Importing kernels/model modules is allowed, but they must not call acquire.
    import qumode_qsk_gpu.kernels
    import qumode_qsk_gpu.model
    import qumode_qsk_gpu.gradients
    import qumode_qsk_gpu.entropy
    import qumode_qsk_gpu.spsa
    import qumode_qsk_gpu.bfgs
    import qumode_qsk_gpu.jobs
    import qumode_qsk_gpu.gate
    assert rt.is_acquired() is False
    return dict(ok=True, cupy_before=pre, acquired=False, version=qumode_qsk_gpu.__version__)


if __name__ == "__main__":
    print(test_import_does_not_touch_cuda())
