#!/usr/bin/env python3
"""Run requested dual-GPU rounds sequentially; default is a read-only plan."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    from qumode_qsk_gpu.dual import instances, lock
    from qumode_qsk_gpu.io_util import save_json
    for round_id in args.rounds:
        if round_id < 1 or round_id > 20:
            ap.error('rounds must be in 1..20')
        if not args.run:
            print('round', round_id, 'instances', sum(len(instances(round_id, d)) for d in (0, 1)))
    if not args.run:
        return 0
    state = ROOT / 'runs/dual_gpu'
    with lock(state / 'sequence.lock'):
        directory = state / 'sequences' / ('rounds-' + uuid.uuid4().hex)
        directory.mkdir(parents=True)
        completed = []
        for round_id in args.rounds:
            save_json(directory / 'status.json', dict(state='running', pid=os.getpid(),
                      current_round=round_id, rounds=args.rounds, completed_rounds=completed))
            with (directory / f'round{round_id}.log').open('x') as log:
                code = subprocess.call([sys.executable, str(ROOT / 'scripts/dual_gpu.py'),
                                        'launch', '--round', str(round_id)],
                                       stdout=log, stderr=subprocess.STDOUT)
            if code:
                save_json(directory / 'status.json', dict(state='failed', current_round=round_id,
                          rounds=args.rounds, completed_rounds=completed, returncode=code))
                return code
            completed.append(round_id)
        save_json(directory / 'status.json', dict(state='finished', rounds=args.rounds,
                  completed_rounds=completed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
