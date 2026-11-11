"""Offline safety tests; only the real production planner executes end to end."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/local_gpu_runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Queue:
    def __init__(self, sha):
        self.item = {'id': 123, 'number': 7, 'state': 'open', 'body': 'production-dry-run',
                     'labels': [{'name': 'gpu-run'}]}
        self.notes = [{'id': 1, 'user': {'id': 42, 'type': 'User'}, 'created_at': 'same',
                       'updated_at': 'same', 'body': f'/gpu-approve {r.digest(self.item["body"].encode())} {sha}'}]
        self.fail_publish = False

    def actor(self): return 900
    def owner(self): return 42
    def issues(self): return [self.item]
    def issue(self, number): return dict(self.item)
    def comments(self, number): return list(self.notes)
    def publish(self, number, body):
        if self.fail_publish:
            raise RuntimeError('simulated GitHub outage')
        self.notes.append({'body': body, 'user': {'id': 900}})


class SafetyTests(unittest.TestCase):
    def setUp(self):
        r.STOP.clear()
        self.api = Queue('abc')

    def test_authorization(self):
        a = self.api
        self.assertTrue(r.approved(a.item, a.notes, 42, 'abc'))
        a.notes[0]['user']['id'] = 99
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))
        a.notes[0]['user']['id'] = 42
        a.notes[0]['updated_at'] = 'edited'
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))
        a.notes[0]['updated_at'] = 'same'
        self.assertFalse(r.approved(a.item, a.notes, 42, 'different-code'))
        a.item['body'] += '\n'
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))

    def test_revocation_and_prs(self):
        a = self.api
        a.notes.append({'id': 2, 'user': {'id': 42, 'type': 'User'}, 'body': '/gpu-revoke',
                        'created_at': 'same', 'updated_at': 'same'})
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))
        a.item['pull_request'] = {}
        self.assertIsNone(r.task_of(a.item))

    def test_command_injection_and_no_production_switch(self):
        for body in ['gpu-validation; touch /tmp/injected', '$(id)', '--run',
                     'production-dry-run\n--run', 'gpu-validation --allow-version-drift']:
            self.api.item['body'] = body
            self.assertIsNone(r.task_of(self.api.item))
            with self.assertRaises(KeyError):
                r.command(body, Path(sys.executable))
        for task in r.TASKS:
            self.assertNotIn('--run', r.command(task, Path(sys.executable)))

    def test_crash_claim_never_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            db = r.database(state)
            db.execute('INSERT INTO jobs(id,number,status) VALUES (123,7,"claimed")')
            db.commit()
            db.close()
            db = r.database(state)
            self.assertEqual(db.execute('SELECT status FROM jobs').fetchone()[0], 'interrupted')
            a = Queue(r.fingerprint(r.payload()))
            with patch.object(r, 'run_task') as launch:
                r.poll(db, a, state, Path(sys.executable), 10, once=True)
                launch.assert_not_called()
            db.close()

    def test_outbox_failure_does_not_repeat_job(self):
        with tempfile.TemporaryDirectory() as d:
            db = r.database(Path(d))
            db.execute('INSERT INTO jobs(id,number,status,report) VALUES (123,7,"finished","PASS")')
            db.commit()
            a = self.api
            a.fail_publish = True
            with self.assertRaises(RuntimeError): r.flush(db, a)
            self.assertEqual(db.execute('SELECT published FROM jobs').fetchone()[0], 0)
            a.fail_publish = False
            r.flush(db, a)
            r.flush(db, a)
            self.assertEqual(len(a.notes), 2)
            db.close()

    def test_timeout_stop_and_secret_stripping(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            code = 'import os,time; print(os.getenv("GPU_RUNNER_TOKEN"),flush=True); time.sleep(20)'
            with patch.dict(os.environ, {'GPU_RUNNER_TOKEN': 'must-not-leak'}):
                rc, reason = r.execute([sys.executable, '-c', code], path, path/'log', .3, threading.Event())
            self.assertEqual(reason, 'timeout')
            self.assertNotEqual(rc, 0)
            self.assertNotIn('must-not-leak', (path/'log').read_text())
            stop = threading.Event()
            stop.set()
            _, reason = r.execute([sys.executable, '-c', 'import time; time.sleep(20)'], path, path/'stop-log', 30, stop)
            self.assertEqual(reason, 'stopped')

    def test_snapshot_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'src').mkdir()
            (root/'src/escape').symlink_to('/etc/passwd')
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                r.payload(root)

    def test_forged_summary_marker_cannot_suppress_publication(self):
        with tempfile.TemporaryDirectory() as d:
            db = r.database(Path(d))
            db.execute('INSERT INTO jobs(id,number,status,report) VALUES (123,7,"finished","PASS")')
            db.commit()
            a = self.api
            a.notes.append({'body': '<!-- local-gpu-runner:123:final -->', 'user': {'id': 99}})
            r.flush(db, a)
            self.assertEqual(len(a.notes), 3)
            db.close()

    def test_exclusive_local_lock(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'lock'
            with path.open('a') as first, path.open('a') as second:
                r.fcntl.flock(first, r.fcntl.LOCK_EX | r.fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    r.fcntl.flock(second, r.fcntl.LOCK_EX | r.fcntl.LOCK_NB)

    def test_closed_unlabeled_and_bot_rejected(self):
        a = self.api
        a.item['state'] = 'closed'
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))
        a.item['state'] = 'open'
        a.item['labels'] = []
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))
        a.item['labels'] = [{'name': 'gpu-run'}]
        a.notes[0]['user']['type'] = 'Bot'
        self.assertFalse(r.approved(a.item, a.notes, 42, 'abc'))

    def test_initial_publication_failure_consumes_claim(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            db = r.database(state)
            a = Queue(r.fingerprint(r.payload()))
            a.fail_publish = True
            with patch.object(r, 'run_task') as launch:
                with self.assertRaises(RuntimeError):
                    r.poll(db, a, state, Path(sys.executable), 10, once=True)
                launch.assert_not_called()
            self.assertEqual(db.execute('SELECT status FROM jobs').fetchone()[0], 'finished')
            a.fail_publish = False
            with patch.object(r, 'run_task') as launch:
                r.poll(db, a, state, Path(sys.executable), 10, once=True)
                launch.assert_not_called()
            db.close()

    def test_pagination_owner_and_redirect(self):
        with patch.dict(os.environ, {'GPU_RUNNER_TOKEN': 'test-token'}):
            api = r.GitHub()
        with patch.object(api, 'request', side_effect=[[{}] * 100, [{'id': 101}]]) as req:
            self.assertEqual(len(api.pages('/issues?state=open')), 101)
            self.assertIn('page=2', req.call_args.args[0])
        for repo in [{'full_name': r.REPO, 'owner': {'type': 'Organization', 'id': 42}},
                     {'full_name': 'other/repo', 'owner': {'type': 'User', 'id': 42}}]:
            with patch.object(api, 'request', return_value=repo):
                with self.assertRaises(RuntimeError): api.owner()
        with self.assertRaises(RuntimeError):
            r.NoRedirect().redirect_request(None, None, None, None, None, None)

    def test_gpu_wrapper_preserves_import_safety(self):
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location('check', ROOT/'scripts/local_gpu_runner_check.py')
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
            getDeviceProperties=lambda device: {'name': b'NVIDIA GeForce RTX 5090'})))
        with patch.dict(sys.modules, {'cupy': cp}), patch.object(sys, 'argv', ['check', 'validation']):
            with patch.object(check.runpy, 'run_path') as launch:
                check.main()
                launch.assert_called_once()
                self.assertEqual(sys.argv[1:3], ['--device-id', '1'])
        cp.cuda.runtime.getDeviceProperties = lambda device: {'name': b'Wrong GPU'}
        with patch.dict(sys.modules, {'cupy': cp}), patch.object(sys, 'argv', ['check', 'validation']):
            with self.assertRaisesRegex(RuntimeError, 'RTX 5090'):
                check.main()

    def test_safe_one_task_end_to_end(self):
        """Real snapshot, planner subprocess, logs, ledger, publication and replay."""
        state = ROOT / '.local-gpu-runner/self-test'
        state.mkdir(parents=True, exist_ok=True)
        # Every test gets a new retained directory; never delete earlier evidence.
        directory = Path(tempfile.mkdtemp(prefix='test-', dir=state))
        before = protected_hashes()
        db = r.database(directory)
        api = Queue(r.fingerprint(r.payload()))
        self.assertTrue(r.poll(db, api, directory, ROOT/'.venv/bin/python', 60, once=True))
        result = json.loads((directory/'123/result.json').read_text())
        self.assertEqual(result['status'], 'PASS')
        self.assertIn('dry-run: no jobs executed', (directory/'123/complete.log').read_text())
        self.assertEqual(db.execute('SELECT published FROM jobs').fetchone()[0], 1)
        with patch.object(r, 'run_task') as launch:
            self.assertFalse(r.poll(db, api, directory, ROOT/'.venv/bin/python', 60, once=True))
            launch.assert_not_called()
        self.assertEqual(before, protected_hashes())
        db.close()
        print(f'\nRetained offline end-to-end evidence: {directory}')


def protected_hashes():
    files = []
    for name in ['stamps', 'runs', 'data', 'logs', 'summaries']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    files.extend(p for p in (ROOT/'GPU_test').rglob('*') if p.is_file() and
                 ('scratch' in str(p.relative_to(ROOT)) or p.name == 'last_report.json'))
    return {str(p.relative_to(ROOT)): r.digest(p.read_bytes()) for p in files}


if __name__ == '__main__':
    unittest.main()
