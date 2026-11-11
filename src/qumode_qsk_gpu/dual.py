"""Instance ownership and durable receipts; no CUDA at import time."""
import fcntl
import json
import os
from pathlib import Path
from contextlib import contextmanager
from .io_util import save_json, sha256_file
from . import protocol as P


@contextmanager
def lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Another process owns {path}') from None
        yield


def instances(round_id, device):
    if device not in (0, 1):
        raise ValueError(device)
    return [dict(N=n, index=i, seed=P.main_seed(n, i), round=round_id)
            for n in P.NS for i in P.round_indices(round_id) if i % 2 == device]


def seal(directory, name, binding, files):
    directory = Path(directory).resolve()
    hashes = {str(Path(p).resolve().relative_to(directory)): sha256_file(p) for p in files}
    save_json(directory / name, dict(binding=binding, files=hashes), fsync=True)


def intact(directory, name, binding):
    directory = Path(directory).resolve()
    try:
        rec = json.loads((directory / name).read_text())
        if rec['binding'] != binding or not rec['files']:
            return False
        for rel, digest in rec['files'].items():
            p = (directory / rel).resolve()
            p.relative_to(directory)
            if sha256_file(p) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def configure(directory):
    from . import jobs
    directory = Path(directory)
    jobs.configure_paths(**{k: directory / k for k in ('runs', 'inst', 'ent', 'ref', 'logs')})


def execute_instance(spec, base, binding, entropy_lock):
    """Retry in new directories; immutable earlier attempts remain inspectable."""
    import shutil
    import uuid
    from . import jobs
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(base.glob('attempt-*'), key=lambda p: p.stat().st_mtime_ns)
    for old in attempts:
        if intact(old, 'complete.json', binding):
            return 'reused'
    dest = base / ('attempt-' + uuid.uuid4().hex)
    dest.mkdir()
    configure(dest)
    # Copy only files accepted by the existing scientific resume validator.
    # A receipt additionally detects changes to any output or reference.
    copied = False
    for old in reversed(attempts):
        if intact(old, 'checkpoint.json', binding):
            for sub in ('runs', 'inst', 'ent', 'ref'):
                if (old / sub).exists():
                    shutil.copytree(old / sub, dest / sub)
            copied = True
            break
    if not copied:
        import_legacy(spec, dest)
        for old in reversed(attempts):
            if intact(old, 'host-receipt.json', binding):
                from .io_util import read_json
                host = read_json(old / 'host.json')
                for sub in ('inst', 'ent', 'ref'):
                    shutil.copytree(old / sub, dest / sub, dirs_exist_ok=True)
                for key in ('instance', 'scan', 'reference'):
                    host['paths'][key] = str(dest / Path(host['paths'][key]).relative_to(old))
                save_json(dest / 'host.json', host)
                seal(dest, 'host-receipt.json', binding, [dest / 'host.json'] + [Path(host['paths'][k]) for k in ('instance', 'scan', 'reference')])
                break
    def checkpoint():
        files = [p for sub in ('runs', 'inst', 'ent', 'ref')
                 for p in (dest / sub).glob('*') if p.is_file()]
        seal(dest, 'checkpoint.json', binding, files)
    entropy = dict(spec, task='entropy')
    if jobs.load_existing(entropy) is None:
        # One host scan across both GPUs, including N=21. No process pools.
        with lock(entropy_lock, blocking=True):
            host_file = dest / 'host.json'
            from .io_util import read_json
            if intact(dest, 'host-receipt.json', binding):
                host = read_json(host_file)
            else:
                host = jobs.cpu_entropy_scan_only(entropy)
                save_json(host_file, host)
                seal(dest, 'host-receipt.json', binding,
                     [host_file] + [Path(host['paths'][k]) for k in ('instance', 'scan', 'reference')])
            jobs.finalize_entropy_gpu(entropy, host)
        checkpoint()
    for depth in P.DEPTHS:
        for method in P.METHODS:
            for task in ('spsa', 'bfgs_pack'):
                cell = dict(spec, task=task, L=depth, method=method)
                if jobs.load_existing(cell) is None:
                    rec = jobs.dispatch(cell)
                    # Save partial BFGS starts even when a later start fails.
                    checkpoint()
                    if rec.get('error') or rec.get('verdict') == 'FAIL':
                        raise RuntimeError(f'{cell}: {rec.get("error")}')
                checkpoint()
    from .io_util import read_json
    # Seal all scientific files, including every BFGS start and reference.
    receipt = read_json(dest / 'checkpoint.json')
    seal(dest, 'complete.json', binding, [dest / p for p in receipt['files']])
    return 'completed'


def import_legacy(spec, dest):
    """Copy accepted prior results; never modify the legacy campaign."""
    import shutil
    import numpy as np
    from . import jobs
    from .paths import ROOT
    from .io_util import read_json
    snap = jobs._path_snapshot()
    legacy = dict(runs=ROOT / 'runs/main', inst=ROOT / 'data/instances/main',
                  ent=ROOT / 'data/entropy/main', ref=ROOT / 'data/references/main', logs=ROOT / 'logs')
    try:
        jobs.configure_paths(**legacy)
        ent = dict(spec, task='entropy')
        rec = jobs.load_existing(ent)
        if rec is None:
            return
        n, seed = spec['N'], spec['seed']
        refs = [legacy['inst'] / f'N{n}_seed{seed}.json',
                legacy['ent'] / f'N{n}_seed{seed}_scan.json',
                legacy['ref'] / f'N{n}_seed{seed}_reference.npz']
        meta = read_json(refs[0])
        import hashlib
        with np.load(refs[2], allow_pickle=False) as z:
            if int(z['N']) != n or int(z['seed']) != seed:
                return
            if hashlib.sha256(np.ascontiguousarray(z['gs']).tobytes()).hexdigest() != meta['gs_sha256']:
                return
        if sha256_file(refs[2]) != rec['paths']['reference_sha256']:
            return
        for sub, source in zip(('inst', 'ent', 'ref'), refs):
            (dest / sub).mkdir(exist_ok=True)
            shutil.copy2(source, dest / sub / source.name)
        candidates = [ent]
        for depth in P.DEPTHS:
            for method in P.METHODS:
                candidates.append(dict(spec, task='spsa', L=depth, method=method))
                for k in range(P.BFGS_STARTS):
                    candidates.append(dict(spec, task='bfgs', L=depth, method=method, start_index=k))
        (dest / 'runs').mkdir(exist_ok=True)
        for cell in candidates:
            if jobs.load_existing(cell) is not None:
                source = jobs.result_path(cell)
                shutil.copy2(source, dest / 'runs' / source.name)
    except (OSError, ValueError, KeyError):
        pass
    finally:
        jobs._restore_paths(snap)


def compatible_binding(old, current):
    """Permit scheduler-only source changes, keeping original result provenance."""
    from . import gate
    from .io_util import sha256_json
    if old.get('unofficial') or old.get('allow_version_drift'):
        return False
    if old.get('binding_sha256') != sha256_json({k: v for k, v in old.items() if k != 'binding_sha256'}):
        return False
    # Device ID can differ: both devices independently pass the current gate.
    for key in gate._COMPARE_KEYS:
        if key not in ('source_pins_sha256', 'device_id'):
            if key not in old or old[key] != current.get(key):
                return False
    def physics(binding):
        return {p['path']: p['sha256'] for p in binding.get('source_pins', [])
                if p['path'].startswith(('src/', 'cpu_oracle/', 'protocol/'))
                and p['path'] not in ('src/qumode_qsk_gpu/dual.py', 'src/qumode_qsk_gpu/gate.py')}
    pinned = physics(old)
    return bool(pinned) and pinned == physics(current)


def enable_legacy_resume():
    """Accept only records matching a preserved official validation binding."""
    from . import gate, jobs
    from .paths import ROOT
    from .io_util import read_json
    current = gate.current_binding()
    accepted = {}
    candidates = [ROOT / 'stamps/GPU_TEST_PASS.json']
    candidates += list((ROOT / 'runs/dual_gpu/validation').glob('gpu*/history/*.json'))
    for path in candidates:
        try:
            stamp = read_json(path)
            old = stamp['binding']
            if stamp.get('verdict') == 'PASS' and not stamp.get('unofficial') and compatible_binding(old, current):
                accepted[old['binding_sha256']] = old
        except (OSError, ValueError, KeyError, TypeError):
            continue
    original = jobs._provenance_ok
    def accepted_provenance(rec):
        if original(rec):
            return True
        old = accepted.get(rec.get('binding_sha256'))
        if old is None or rec.get('unofficial'):
            return False
        fields = dict(source_pins_sha256=old['source_pins_sha256'],
                      numpy_version=old['numpy_version'], scipy_version=old['scipy_version'],
                      numba_version=old['numba_version'], python_version=old['python_version'],
                      cupy_version=old['cupy_version'], dtype_real=old['dtype_real'],
                      dtype_cplx=old['dtype_cplx'], kernel_path=old['kernel_path'])
        return all(k in rec and rec[k] == v for k, v in fields.items())
    jobs._provenance_ok = accepted_provenance
    return accepted
