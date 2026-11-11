#!/usr/bin/env python3
"""Safe dual RTX 5090 campaign launcher. Default: plan only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
STATE = ROOT / 'runs' / 'dual_gpu'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('plan', 'launch', 'monitor', 'validate', '_worker'))
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--device', type=int, choices=(0, 1))
    ap.add_argument('--session')
    ap.add_argument('--min-n', type=int, choices=(6, 9, 12, 15, 18, 21), default=6,
                    help='Start at this N, preserving smaller-N historical results.')
    args = ap.parse_args()
    from qumode_qsk_gpu.dual import instances, lock, execute_instance
    from qumode_qsk_gpu.io_util import save_json
    if args.action == 'plan':
        print(json.dumps({d: [s for s in instances(args.round, d) if s['N'] >= args.min_n]
                          for d in (0, 1)}, indent=2))
        return 0
    if args.action == 'monitor':
        for p in sorted((STATE / 'sessions').glob('*/gpu*.json')):
            rec = json.loads(p.read_text())
            pid = rec.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                    rec['process_alive'] = True
                except ProcessLookupError:
                    rec['process_alive'] = False
            print(p, json.dumps(rec))
        for p in sorted((STATE / 'sessions').glob('*/exit*.json')):
            print(p, p.read_text())
        return 0
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = '1'
    if args.action in ('validate', 'launch'):
        with lock(STATE / 'campaign.lock'):
            session = STATE / 'sessions' / (args.action + '-' + uuid.uuid4().hex)
            session.mkdir(parents=True)
            children = []
            try:
                for device in (0, 1):
                    child_env = dict(env, QSK_VALIDATION_STAMPS=str(STATE / 'validation' / f'gpu{device}'))
                    if args.action == 'validate':
                        child_env['QSK_VALIDATION_SCRATCH'] = str(session / f'scratch{device}')
                        cmd = ['flock', '--nonblock', str(STATE / f'gpu{device}.lock'), sys.executable, str(ROOT / 'scripts/run_gpu_test.py'), '--device-id', str(device), '--out', str(session / f'report{device}.json')]
                    else:
                        cmd = [sys.executable, __file__, '_worker', '--device', str(device), '--round', str(args.round), '--session', str(session), '--min-n', str(args.min_n)]
                    with (session / f'gpu{device}.log').open('x') as log:
                        children.append(subprocess.Popen(cmd, env=child_env, stdout=log, stderr=subprocess.STDOUT))
                    # Validation is sequential: its CPU fixture checks use one slot.
                    if args.action == 'validate' and children[-1].wait():
                        return 1
                codes = [p.wait() for p in children]
                for device, code in enumerate(codes):
                    save_json(session / f'exit{device}.json', dict(device=device, returncode=code))
                print('Session:', session, 'exit codes:', codes)
                return int(any(codes))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    child.wait()
    if args.device is None or not args.session:
        ap.error('_worker requires --device and --session')
    from qumode_qsk_gpu import runtime, gate, jobs
    session = Path(args.session)
    status = session / f'gpu{args.device}.json'
    # Worker lock survives coordinator death; a second launch cannot duplicate work.
    with lock(STATE / f'gpu{args.device}.lock'):
        runtime.acquire(device_id=args.device, activate=True)
        stamp = gate.assert_production_allowed()
        if stamp['binding']['device_id'] != args.device:
            raise RuntimeError('Validation belongs to another device')
        if 'RTX 5090' not in runtime.device_name():
            raise RuntimeError('Expected RTX 5090')
        from qumode_qsk_gpu.dual import enable_legacy_resume
        accepted = enable_legacy_resume()
        save_json(session / f'resume-bindings{args.device}.json', accepted)
        binding = gate.provenance_fields()
        done = failed = 0
        for spec in instances(args.round, args.device):
            if spec['N'] < args.min_n:
                continue
            save_json(status, dict(state='running', pid=os.getpid(), current=spec, done=done, failed=failed))
            try:
                gate.verify_cached_binding()
                outcome = execute_instance(spec, STATE / 'instances' / f'N{spec["N"]}_seed{spec["seed"]}', binding, STATE / 'entropy.lock')
                done += 1
                print(outcome, spec, flush=True)
            except Exception as exc:
                failed += 1
                print('FAIL', spec, repr(exc), flush=True)
            save_json(status, dict(state='running', pid=os.getpid(), current=spec, done=done, failed=failed))
        save_json(status, dict(state='finished', done=done, failed=failed))
        return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
