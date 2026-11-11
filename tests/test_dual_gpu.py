import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qumode_qsk_gpu import dual, protocol as P


def hold_lock(path, conn):
    with dual.lock(path):
        conn.send(True)
        conn.recv()


class DualTests(unittest.TestCase):
    def test_partition_all_official_instances(self):
        seen = set()
        for r in range(1, P.N_ROUNDS + 1):
            a, b = [dual.instances(r, d) for d in (0, 1)]
            keys = lambda xs: {(s['N'], s['seed']) for s in xs}
            self.assertFalse(keys(a) & keys(b))
            self.assertEqual(len(a), 30)
            self.assertEqual(len(b), 30)
            for s in a + b:
                self.assertEqual(s['seed'], P.main_seed(s['N'], s['index']))
            self.assertFalse(seen & (keys(a) | keys(b)))
            seen |= keys(a) | keys(b)
        self.assertEqual(len(seen), 1200)
        self.assertEqual(P.DEPTHS, (1, 2, 3, 4, 5, 6))

    def test_legacy_resume_rejects_physics_drift(self):
        from qumode_qsk_gpu.io_util import sha256_json
        from qumode_qsk_gpu import gate
        old = {k: 'value' for k in gate._COMPARE_KEYS}
        old['source_pins'] = [dict(path='src/qumode_qsk_gpu/bfgs.py', sha256='physics'), dict(path='src/qumode_qsk_gpu/gate.py', sha256='old-gate')]
        old['binding_sha256'] = sha256_json(old)
        current = dict(old, device_id=0, source_pins_sha256='new')
        current['source_pins'] = [dict(path='src/qumode_qsk_gpu/bfgs.py', sha256='physics'), dict(path='src/qumode_qsk_gpu/gate.py', sha256='new-gate'), dict(path='src/qumode_qsk_gpu/dual.py', sha256='scheduler')]
        self.assertTrue(dual.compatible_binding(old, current))
        current['source_pins'][0]['sha256'] = 'changed-optimizer'
        self.assertFalse(dual.compatible_binding(old, current))
        current['source_pins'][0]['sha256'] = 'physics'
        current['dtype_real'] = 'float32'
        self.assertFalse(dual.compatible_binding(old, current))
        current['dtype_real'] = old['dtype_real']
        old['binding_sha256'] = 'corrupt'
        self.assertFalse(dual.compatible_binding(old, current))

    def test_integrity_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'result.json'
            p.write_text('{"energy": 1}')
            dual.seal(root, 'complete.json', {'device': 0}, [p])
            self.assertTrue(dual.intact(root, 'complete.json', {'device': 0}))
            self.assertFalse(dual.intact(root, 'complete.json', {'device': 1}))
            p.write_text('{"energy": 2}')
            self.assertFalse(dual.intact(root, 'complete.json', {'device': 0}))
            p.unlink()
            self.assertFalse(dual.intact(root, 'complete.json', {'device': 0}))

    def test_receipt_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'attempt'
            root.mkdir()
            (root / 'complete.json').write_text(json.dumps(dict(binding={}, files={'../outside': 'abc'})))
            self.assertFalse(dual.intact(root, 'complete.json', {}))

    def test_device_specific_stamp_path(self):
        from qumode_qsk_gpu import gate
        with patch.dict(os.environ, {'QSK_VALIDATION_STAMPS': '/tmp/qsk-device0'}):
            self.assertEqual(gate.stamp_path(), Path('/tmp/qsk-device0/GPU_TEST_PASS.json'))
        self.assertIn('device_id', gate._COMPARE_KEYS)

    def test_process_death_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'lock')
            ctx = mp.get_context('spawn')
            parent, child = ctx.Pipe()
            process = ctx.Process(target=hold_lock, args=(path, child))
            process.start()
            self.assertTrue(parent.poll(15))
            parent.recv()
            with self.assertRaises(RuntimeError):
                with dual.lock(path):
                    pass
            process.terminate()
            process.join(15)
            self.assertFalse(process.is_alive())
            with dual.lock(path):
                pass

    def test_failure_resume_no_duplicate_scan(self):
        from qumode_qsk_gpu import jobs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(N=6, index=0, seed=P.main_seed(6, 0), round=1)
            calls = []
            fail = [True]
            def existing(s):
                p = jobs.result_path(s)
                return json.loads(p.read_text()) if p.exists() else None
            def scan(s):
                calls.append('scan')
                paths = {}
                for key, directory in [('instance', jobs.INST_DIR), ('scan', jobs.ENT_DIR), ('reference', jobs.REF_DIR)]:
                    directory.mkdir(parents=True, exist_ok=True)
                    p = directory / 'data.json'
                    p.write_text('{}')
                    paths[key] = str(p)
                return dict(paths=paths)
            def write(s, rec=None):
                p = jobs.result_path(s)
                p.parent.mkdir(exist_ok=True)
                p.write_text('{"ok": true}')
                return {}
            def dispatch(s):
                calls.append(s['task'])
                if fail[0]:
                    fail[0] = False
                    return dict(error='simulated process failure')
                return write(s)
            snap = jobs._path_snapshot()
            try:
                with patch.object(jobs, 'load_existing', existing), patch.object(jobs, 'cpu_entropy_scan_only', scan), patch.object(jobs, 'finalize_entropy_gpu', write), patch.object(jobs, 'dispatch', dispatch), patch.object(dual, 'import_legacy'):
                    with self.assertRaises(RuntimeError):
                        dual.execute_instance(spec, root / 'instance', {}, root / 'cpu.lock')
                    self.assertEqual(dual.execute_instance(spec, root / 'instance', {}, root / 'cpu.lock'), 'completed')
                    count = len(calls)
                    self.assertEqual(dual.execute_instance(spec, root / 'instance', {}, root / 'cpu.lock'), 'reused')
                    self.assertEqual(len(calls), count)
                    self.assertEqual(calls.count('scan'), 1)
                    attempts = list((root / 'instance').glob('attempt-*'))
                    self.assertEqual(len(attempts), 2)
            finally:
                jobs._restore_paths(snap)


if __name__ == '__main__':
    unittest.main()
