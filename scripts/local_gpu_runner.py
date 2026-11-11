#!/usr/bin/env python3
"""Owner-approved, at-most-once local queue. No shell or remote code execution."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.local-gpu-runner'
REPO = 'zach102824/qumode_qsk_gpu'
TASKS = ('gpu-validation', 'production-dry-run', 'production-gate-verification')
TREES = ('src', 'scripts', 'cpu_oracle', 'protocol', 'docs', 'GPU_test/fixtures', 'GPU_test/tests')
FILES = ('README.md', 'requirements.txt', 'GPU_test/README.md', 'GPU_test/tolerances.json')
STOP = threading.Event()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload(root=ROOT):
    """Only reviewed code/fixtures, no research outputs, symlinks or bytecode."""
    paths = []
    for name in TREES:
        base = root / name
        for p in sorted(base.rglob('*')):
            if p.is_symlink():
                raise RuntimeError(f'symlink forbidden: {p}')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.nbc', '.nbi'}:
                paths.append(p)
    paths.extend(root / name for name in FILES)
    result = {}
    for p in paths:
        if p.is_symlink() or any(q.is_symlink() for q in p.parents):
            raise RuntimeError(f'symlink forbidden: {p}')
        result[str(p.relative_to(root))] = p.read_bytes()
    return result


def fingerprint(blobs):
    return digest(json.dumps([(p, digest(b)) for p, b in sorted(blobs.items())], separators=(',', ':')).encode())


def task_of(issue):
    body = issue.get('body') or ''
    if body.strip() not in TASKS:
        return None
    if issue.get('state') != 'open' or 'pull_request' in issue:
        return None
    if 'gpu-run' not in {x['name'] for x in issue.get('labels', [])}:
        return None
    return body.strip()


def approved(issue, comments, owner_id, code_sha):
    if not task_of(issue):
        return False
    want = f"/gpu-approve {digest(issue['body'].encode())} {code_sha}"
    # The latest owner control comment wins. Edited comments never authorize.
    controls = [c for c in comments if c.get('user', {}).get('id') == owner_id
                and c['user'].get('type') == 'User'
                and (c.get('body') or '').startswith(('/gpu-approve', '/gpu-revoke'))]
    if not controls:
        return False
    c = max(controls, key=lambda c: c['id'])
    return c['body'] == want and c['created_at'] == c['updated_at']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('GitHub redirect refused')


class GitHub:
    def __init__(self):
        token = os.environ.get('GPU_RUNNER_TOKEN', '')
        if not token:
            raise RuntimeError('set GPU_RUNNER_TOKEN to an Issues read/write token')
        self.token = token
        self.actor_id = None
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, path, body=None, *, identity=False):
        url = 'https://api.github.com/user' if identity else 'https://api.github.com/repos/' + REPO + path
        req = urllib.request.Request(url,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.token,
                     'Accept': 'application/vnd.github+json',
                     'X-GitHub-Api-Version': '2022-11-28',
                     'Content-Type': 'application/json', 'User-Agent': 'local-gpu-runner'})
        with self.opener.open(req, timeout=30) as response:
            return json.load(response)

    def pages(self, path):
        result = []
        for page in range(1, 1001):
            data = self.request(path + ('&' if '?' in path else '?') + f'per_page=100&page={page}')
            result.extend(data)
            if len(data) < 100:
                return result
        raise RuntimeError('pagination limit reached; refusing incomplete authorization')

    def actor(self):
        if self.actor_id is None:
            self.actor_id = self.request('', identity=True)['id']
        return self.actor_id

    def owner(self):
        repo = self.request('')
        if repo['full_name'].lower() != REPO.lower() or repo['owner']['type'] != 'User':
            raise RuntimeError('unexpected repository or organization owner')
        return repo['owner']['id']

    def issues(self):
        return self.pages('/issues?state=open&labels=gpu-run&sort=created&direction=asc')

    def issue(self, number):
        return self.request(f'/issues/{number}')

    def comments(self, number):
        return self.pages(f'/issues/{number}/comments')

    def publish(self, number, body):
        self.request(f'/issues/{number}/comments', {'body': body})


def database(state):
    db = sqlite3.connect(state / 'jobs.sqlite3')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, number INTEGER, status TEXT, report TEXT, published INTEGER DEFAULT 0)')
    # Claim-before-launch deliberately sacrifices retries for at-most-once safety.
    db.execute("UPDATE jobs SET status='interrupted', report='Runner interrupted; outcome uncertain. Inspect local artifacts; never automatically retry.' WHERE status='claimed'")
    db.commit()
    return db


def command(task, python):
    commands = {
        'gpu-validation': [str(python), 'scripts/local_gpu_runner_check.py', 'validation'],
        'production-dry-run': [str(python), 'scripts/run_production.py', '--dry-run', '--round', '1'],
        'production-gate-verification': [str(python), 'scripts/local_gpu_runner_check.py', 'gate'],
    }
    return commands[task]


def execute(argv, cwd, log, timeout, stop=STOP):
    # Secrets and Python/CUDA overrides from the polling environment are excluded.
    env = {k: v for k, v in os.environ.items() if k in {'HOME', 'PATH', 'LD_LIBRARY_PATH', 'SYSTEMROOT', 'WSL_DISTRO_NAME'}}
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log.open('wb') as out:
        if stop.is_set():
            return -signal.SIGTERM, 'stopped'
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        reason = None
        while p.poll() is None:
            if stop.is_set() or time.monotonic() >= deadline:
                reason = 'stopped' if stop.is_set() else 'timeout'
                break
            stop.wait(.2)
        # Also clean up any descendant process group after normal exit.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
        if reason:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    return p.returncode, reason


def run_task(task, blobs, directory, python, timeout):
    workspace = directory / 'workspace'
    workspace.mkdir(parents=True)
    for rel, data in blobs.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    stamp = ROOT / 'stamps/GPU_TEST_PASS.json'
    if stamp.exists():
        target = workspace / 'stamps/GPU_TEST_PASS.json'
        target.parent.mkdir()
        target.write_bytes(stamp.read_bytes())
    argv = command(task, python)
    (directory / 'command.json').write_text(json.dumps(argv, indent=2))
    code, reason = execute(argv, workspace, directory / 'complete.log', timeout)
    with (directory / 'complete.log').open('rb') as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 12000))
        text = log.read().decode(errors='replace')
    status = 'PASS' if code == 0 and reason is None else 'FAIL'
    diagnosis = ('Review device 1 availability, CUDA/CuPy compatibility and pinned libraries; '
                 'for binding mismatches run a separately approved validation. '
                 'For numerical failures inspect validation-report.json against frozen fixtures. '
                 'No source or tolerance changes were made.') if status == 'FAIL' else ''
    # Keep the complete traceback/report locally; bounded excerpt for GitHub.
    report = {'status': status, 'exit_code': code, 'reason': reason, 'diagnosis': diagnosis,
              'task': task, 'code_sha256': fingerprint(blobs), 'log_tail': text[-7000:]}
    (directory / 'result.json').write_text(json.dumps(report, indent=2))
    # Quoted text avoids issue-supplied mentions and Markdown directives.
    excerpt = '\n'.join('> ' + line.replace('@', '@\u200b') for line in text[-3500:].splitlines())
    return f'{task}: **{status}** (exit {code}, reason {reason}).\n\n{diagnosis}\n\n{excerpt}\n\nComplete logs, numerical reports and scratch outputs retained locally for issue task.'


def flush(db, api):
    for job_id, number, status, report in db.execute('SELECT id, number, status, report FROM jobs WHERE published=0 AND status != "claimed"').fetchall():
        marker = f'<!-- local-gpu-runner:{job_id}:final -->'
        if not any(marker in (c.get('body') or '') and c.get('user', {}).get('id') == api.actor()
                   for c in api.comments(number)):
            api.publish(number, marker + '\n' + report)
        db.execute('UPDATE jobs SET published=1 WHERE id=?', (job_id,))
        db.commit()


def poll(db, api, state, python, timeout, once=False):
    flush(db, api)
    owner = api.owner()
    blobs = payload()
    sha = fingerprint(blobs)
    for candidate in api.issues():
        if STOP.is_set():
            break
        if not task_of(candidate) or db.execute('SELECT 1 FROM jobs WHERE id=?', (candidate['id'],)).fetchone():
            continue
        number = int(candidate['number'])
        issue = api.issue(number)
        if issue['id'] != candidate['id'] or not approved(issue, api.comments(number), owner, sha):
            continue
        # Authorization is rechecked immediately before the durable claim.
        if STOP.is_set():
            break
        if fingerprint(payload()) != sha:
            raise RuntimeError('local code changed during poll')
        with db:
            db.execute('INSERT INTO jobs(id,number,status) VALUES (?,?,"claimed")', (issue['id'], number))
        directory = state / str(issue['id'])
        try:
            directory.mkdir()
            (directory / 'issue.json').write_text(json.dumps(issue, indent=2))
            api.publish(number, f'<!-- local-gpu-runner:{issue["id"]}:started -->\nClaimed {task_of(issue)} on local GPU runner. Production execution is disabled.')
            report = run_task(task_of(issue), blobs, directory, python, timeout)
        except Exception:
            report = 'FAIL: local runner error. Inspect local traceback. Check permissions, disk space and environment; no automatic retry.\n\n' + traceback.format_exc()[-5000:]
            (state / f'{issue["id"]}-error.txt').write_text(traceback.format_exc())
        with db:
            db.execute('UPDATE jobs SET status="finished",report=? WHERE id=?', (report, issue['id']))
        flush(db, api)
        if once:
            return True
    return False


def preflight():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'dev/codex-debug':
        raise RuntimeError('runner requires dev/codex-debug')
    if platform.python_version() != '3.13.5' or Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise RuntimeError('start with this repository .venv Python 3.13.5')
    if 'wsl2' not in platform.release().lower() or 'ubuntu' not in Path('/etc/os-release').read_text().lower():
        raise RuntimeError('WSL2 Ubuntu required')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['start', 'stop', 'once', 'fingerprint'])
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--interval', type=int, default=60)
    args = ap.parse_args()
    if args.action == 'fingerprint':
        print(fingerprint(payload()))
        return
    STATE.mkdir(mode=0o700, exist_ok=True)
    if STATE.is_symlink():
        raise RuntimeError('state directory may not be a symlink')
    if args.action == 'stop':
        (STATE / 'stop').touch()
        print('Stop requested; active task will be terminated within one second.')
        return
    if args.timeout < 1 or args.interval < 10:
        ap.error('timeout >= 1, interval >= 10 required')
    preflight()
    with (STATE / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (STATE / 'stop').unlink(missing_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        def watch():
            while not STOP.wait(.5):
                if (STATE / 'stop').exists():
                    STOP.set()
        threading.Thread(target=watch, daemon=True).start()
        db = database(STATE)
        api = GitHub()
        startup_sha = fingerprint(payload())
        while not STOP.is_set():
            try:
                preflight()
                if fingerprint(payload()) != startup_sha:
                    raise RuntimeError('local code changed; stop and restart runner after review')
                poll(db, api, STATE, ROOT / '.venv/bin/python', args.timeout, args.action == 'once')
            except Exception:
                traceback.print_exc()
                if args.action == 'once':
                    raise
            if args.action == 'once':
                break
            STOP.wait(args.interval)
        db.close()


if __name__ == '__main__':
    main()
