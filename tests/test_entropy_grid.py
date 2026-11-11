"""Field-selection regression checks without large eigensolves."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpu_oracle.src import entropy as ent


class EntropyGridTests(unittest.TestCase):
    def test_exact_grid_and_boundary_peaks_for_every_size(self):
        expected = [round(0.01 + 0.05 * k, 9) for k in range(60)] + [3.0]
        for n in (6, 9, 12, 15, 18, 21):
            for center in (0.01, 1.01, 3.0):
                with self.subTest(N=n, peak=center):
                    calls = []

                    def evaluate(base, b, cache_ops=None):
                        calls.append(b)
                        row = dict(b=b, S=-(b-center)**2, E0=-1.0,
                                   gap=1.0, even_gap=1.0, residual=0.0,
                                   parity=1.0, near_degenerate=False,
                                   positivity_ok=True)
                        return row, np.ones(2), {}

                    # Tiny fake state keeps this independent of Hilbert-space size.
                    with patch.object(ent, 'evaluate_field', evaluate), patch.object(ent, 'entropy', return_value=0.0):
                        report, _ = ent.scan_bstar(SimpleNamespace(n=n, seed=1))
                    self.assertEqual(calls, expected)
                    self.assertEqual([r['b'] for r in report['scan']], expected)
                    self.assertEqual(report['peak']['b'], center)
                    self.assertEqual(report['scan_policy']['name'], 'initial_61_only')
                    self.assertEqual(report['valid_for_fits'], center == 1.01)
                    if center != 1.01:
                        self.assertEqual(report['label'], 'boundary_or_plateau_UNRESOLVED')


if __name__ == '__main__':
    unittest.main()
