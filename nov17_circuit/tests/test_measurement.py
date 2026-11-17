"""Independent tensor-order and retained measurement-regression checks."""
import csv,json,unittest
import numpy as np
from ..models.observables import P,load_design,probabilities,allocate
from ..models.estimation import estimate
from cpu_oracle.src.hamiltonian import make_J

class MeasurementTests(unittest.TestCase):
    def test_complex_entangled_state_tensor_order(self):
        rng=np.random.default_rng(95)
        v=rng.normal(size=64)+1j*rng.normal(size=64);v/=np.linalg.norm(v)
        d=load_design();got=probabilities(d['E'],v.reshape(8,8).T,d['settings'])
        expected=[[np.vdot(v,np.kron(a,b)@v).real for a in d['E'][sa] for b in d['E'][sb]] for sa,sb in d['settings']]
        np.testing.assert_allclose(got,expected,atol=1e-13)

    def test_saved_example_and_fixed_budget(self):
        d=load_design();J=make_J(6,831106000)
        with (P/'examples/samples.csv').open() as h:rows=list(csv.DictReader(h))
        got=estimate(rows,d,J,.183);expected=json.loads((P/'examples/expected_estimate.json').read_text())
        for k in ('energy','chi_sg','energy_estimated_sem_per_spin'):
            self.assertAlmostEqual(got[k],expected[k],places=13)
        np.testing.assert_array_equal(allocate(d,dict(J=J,B=.183),8192),[179,1243,1291,5479])

    def test_all_saved_povms_are_positive_and_complete(self):
        for path in (P/'calibration').glob('*.npz'):
            with self.subTest(bank=path.name),np.load(path,allow_pickle=False) as d:
                E=d['effects'];self.assertTrue(np.isfinite(E).all())
                np.testing.assert_allclose(E.sum(axis=1),np.broadcast_to(np.eye(8),E.sum(axis=1).shape),atol=1e-9)
                self.assertGreater(np.linalg.eigvalsh(E).min(),-1e-9)
