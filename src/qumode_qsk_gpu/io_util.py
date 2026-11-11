import json, hashlib, time
from pathlib import Path


def save_json(path, data, indent=2, fsync=False):
    """Atomic replace: write a sibling .tmp then os.replace.

    fsync is optional (off by default). Even with fsync=True a crash between
    replace and a directory fsync can theoretically lose the directory entry;
    this is the same class of risk as the CPU campaign save_json.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent) + "\n")
    if fsync:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_json(data) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":")))


def x_sha256(x) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def now_shanghai():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def save_npz_atomic(path, **arrays):
    """Write a compressed npz via a sibling .tmp, fsync, then os.replace."""
    import os
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    with open(tmp, "rb") as f:
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass
    return path
