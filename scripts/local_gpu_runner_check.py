#!/usr/bin/env python3
"""Verify the existing production gate on GPU 1 without dispatching jobs."""
import argparse
import runpy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['validation', 'gate'])
    args = ap.parse_args()
    import cupy
    props = cupy.cuda.runtime.getDeviceProperties(1)
    name = props['name']
    if isinstance(name, bytes):
        name = name.decode()
    if 'RTX 5090' not in name:
        raise RuntimeError(f'GPU 1 must be RTX 5090, got {name}')
    if args.mode == 'validation':
        # Do not acquire qumode runtime before the import-safety test.
        sys.argv = ['run_gpu_test.py', '--device-id', '1', '--out', 'validation-report.json']
        runpy.run_path(str(ROOT / 'scripts/run_gpu_test.py'), run_name='__main__')
        return
    from qumode_qsk_gpu import runtime, gate
    info = runtime.acquire(device_id=1, activate=True)
    stamp = gate.assert_production_allowed()
    gate.verify_cached_binding()
    print(json.dumps({'verdict': 'PASS', 'binding': stamp['binding']['binding_sha256'],
                      'device': info.device_name, 'note': 'No production jobs dispatched.'}, indent=2))


if __name__ == '__main__':
    main()
