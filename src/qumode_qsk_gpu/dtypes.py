"""Frozen dtypes. FP64 / complex128 only. No autocast, no implicit downcast."""
import numpy as np

REAL = np.float64
CPLX = np.complex128
INDEX = np.int64
INV_SQRT2 = np.float64(1.0 / np.sqrt(np.float64(2.0)))
SQRT2 = np.float64(np.sqrt(np.float64(2.0)))

ALLOWED_REAL = ("float64",)
ALLOWED_CPLX = ("complex128",)


def check_real(x, name="array"):
    dt = getattr(x, "dtype", None)
    if dt is None or str(dt) != "float64":
        raise TypeError(f"{name} must be float64, got {dt}")
    return x


def check_cplx(x, name="array"):
    dt = getattr(x, "dtype", None)
    if dt is None or str(dt) != "complex128":
        raise TypeError(f"{name} must be complex128, got {dt}")
    return x
