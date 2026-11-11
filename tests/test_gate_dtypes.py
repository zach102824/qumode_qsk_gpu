"""Dtype regression tests exercise the gate without starting production or CUDA."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from qumode_qsk_gpu import gate, runtime


class GateDtypeTests(unittest.TestCase):
    def setUp(self):
        self.info = SimpleNamespace(dtype_real='float64', dtype_cplx='complex128')
        self.stamp = dict(verdict='PASS', binding={})
        self.patches = [
            patch.object(runtime, 'is_acquired', return_value=True),
            patch.object(runtime, 'info', return_value=self.info),
            patch.object(gate, 'check_library_pins', return_value={}),
            patch.object(gate, 'load_stamp', return_value=self.stamp),
            patch.object(gate, 'current_binding', return_value={}),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_numpy_classes_pass_matching_stamp(self):
        self.assertIs(gate.REAL, np.float64)
        self.assertIs(gate.CPLX, np.complex128)
        self.assertIs(gate.assert_production_allowed(), self.stamp)

    def test_lower_precision_constants_rejected(self):
        for key, value in [('REAL', np.float32), ('CPLX', np.complex64)]:
            with self.subTest(key=key), patch.object(gate, key, value):
                with self.assertRaisesRegex(RuntimeError, 'production requires'):
                    gate.assert_production_allowed()

    def test_lower_precision_runtime_rejected(self):
        for key, value in [('dtype_real', 'float32'), ('dtype_cplx', 'complex64')]:
            with self.subTest(key=key), patch.object(self.info, key, value):
                with self.assertRaisesRegex(RuntimeError, 'runtime dtypes'):
                    gate.assert_production_allowed()

    def test_missing_stamp_still_rejected(self):
        with patch.object(gate, 'load_stamp', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'No GPU_TEST_PASS'):
                gate.assert_production_allowed()

    def test_stale_dtype_stamp_rejected(self):
        self.stamp['binding'] = dict(dtype_real=str(np.float64))
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            gate.assert_production_allowed()

    def test_binding_uses_canonical_names(self):
        from qumode_qsk_gpu import kernels
        info = SimpleNamespace(device_id=1, device_name='test', compute_capability='8.0',
                               cupy_version='test', cuda_version='12000', backend='cupy')
        pins = dict(have={}, pins=dict.fromkeys(('numpy', 'scipy', 'numba', 'python'), 'test'),
                    unofficial=False, allow_version_drift=False, mismatches=[])
        with patch.object(runtime, 'info', return_value=info), \
             patch.object(kernels, 'ensure_compiled'), \
             patch.object(gate, 'check_library_pins', return_value=pins), \
             patch.object(gate, 'fixture_manifest_sha', return_value=('sha', 'manifest')), \
             patch.object(gate, 'source_pins', return_value=[]), \
             patch.object(gate, 'fixture_file_pins', return_value=[]), \
             patch.object(gate, '_kernel_fields', return_value={}):
            rec = gate._compute_binding()
        self.assertEqual((rec['dtype_real'], rec['dtype_cplx']), ('float64', 'complex128'))

    def test_provenance_uses_canonical_names(self):
        rec = gate.provenance_fields()
        self.assertEqual((rec['dtype_real'], rec['dtype_cplx']), ('float64', 'complex128'))


class RuntimeDtypeTests(unittest.TestCase):
    def test_runtime_metadata_uses_canonical_names(self):
        dev = nullcontext()
        api = SimpleNamespace(
            getDeviceProperties=lambda _: dict(name=b'test', major=8, minor=0, totalGlobalMem=1024),
            runtimeGetVersion=lambda: 12000, driverGetVersion=lambda: 12000)
        cp = SimpleNamespace(__version__='test', cuda=SimpleNamespace(Device=lambda _: dev, runtime=api))
        info = runtime._query(cp, 1, False)
        versions = runtime.library_versions()
        for rec in (vars(info), versions):
            self.assertEqual((rec['dtype_real'], rec['dtype_cplx']), ('float64', 'complex128'))

    def test_device_upload_keeps_matching_dtype(self):
        array = np.ones(2, dtype=np.float64)
        with patch.object(runtime, 'xp'), patch.object(runtime, 'is_device_array', return_value=True):
            for dtype in (np.float64, np.dtype('float64'), 'float64'):
                self.assertIs(runtime.to_device(array, dtype=dtype), array)


if __name__ == '__main__':
    unittest.main()
