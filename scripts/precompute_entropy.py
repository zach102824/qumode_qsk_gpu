#!/usr/bin/env python3
"""CPU-only entropy reference cohort, one fresh process per instance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, default=24)
    ap.add_argument('--count', type=int, default=30)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--output', type=Path, default=ROOT / 'runs/cpu_entropy_N24')
    ap.add_argument('--index', type=int, help=argparse.SUPPRESS)
    ap.add_argument('--run', action='store_true', help='Execute; otherwise print plan only.')
    args = ap.parse_args()
    from qumode_qsk_gpu import gate, jobs, protocol
    from qumode_qsk_gpu.dual import lock, seal, intact
    from qumode_qsk_gpu.io_util import save_json, sha256_json
    if args.N < 3 or args.N % 3 or args.count < 1:
        ap.error('N must be a positive multiple of 3 and count must be positive')
    if args.start_index < 0 or args.start_index + args.count > protocol.N_INSTANCES:
        ap.error('indices must be within the main seed cohort')
    indices = list(range(args.start_index, args.start_index + args.count))
    specs = [dict(N=args.N, index=i, seed=protocol.main_seed(args.N, i)) for i in indices]
    if not args.run:
        print(json.dumps(dict(cpu_only=True, scan_policy='initial_61_only', instances=specs), indent=2))
        return 0
    output = args.output.resolve()
    binding = dict(host_scan_only=True, source_pins_sha256=sha256_json(gate.source_pins()),
                   libraries=gate.host_library_versions(), scan_policy='initial_61_only')
    if args.index is not None:
        spec = dict(N=args.N, index=args.index, seed=protocol.main_seed(args.N, args.index))
        base = output / f'N{args.N}_seed{spec["seed"]}'
        base.mkdir(parents=True, exist_ok=True)
        for old in base.glob('attempt-*'):
            if intact(old, 'complete.json', binding):
                print('reused', spec, flush=True)
                return 0
        dest = base / ('attempt-' + uuid.uuid4().hex)
        dest.mkdir()
        jobs.configure_paths(**{k: dest / k for k in ('runs', 'inst', 'ent', 'ref', 'logs')})
        save_json(dest / 'started.json', dict(spec=spec, pid=os.getpid(), binding=binding))
        print('scanning', spec, str(dest), flush=True)
        rec = jobs.cpu_entropy_scan_only(spec)
        save_json(dest / 'host.json', rec)
        scan = json.loads(Path(rec['paths']['scan']).read_text())
        expected = [round(0.01 + 0.05 * k, 9) for k in range(60)] + [3.0]
        if [row['b'] for row in scan['scan']] != expected:
            raise RuntimeError('Entropy reference does not contain the expected 61 fields')
        seal(dest, 'complete.json', binding, [dest / 'host.json'] +
             [Path(rec['paths'][k]) for k in ('instance', 'scan', 'reference')])
        print('completed', spec, 'scan_wall_s', rec['wall_s_scan'], flush=True)
        return 0
    with lock(output / 'cohort.lock'):
        done = []
        for spec in specs:
            save_json(output / 'status.json', dict(state='running', pid=os.getpid(),
                      current=spec, completed=done, target=len(specs), binding=binding))
            cmd = [sys.executable, __file__, '--run', '--N', str(args.N),
                   '--output', str(output), '--index', str(spec['index'])]
            code = subprocess.call(cmd)
            if code:
                save_json(output / 'status.json', dict(state='failed', current=spec,
                          completed=done, target=len(specs), returncode=code))
                return code
            done.append(spec)
        save_json(output / 'status.json', dict(state='finished', completed=done,
                  target=len(specs), binding=binding))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
