"""Small regression tests for the unchanged ansatz and hardware phase tables."""
import unittest
from ..sequence import *
from ..scripts.verify import check

class SequenceTests(unittest.TestCase):
    def test_saved_states_both_frames(self):
        for model,x,L,_ in load_saved()[:2]:check(model,x,L,'test')

    def test_all_fifteen_terms(self):
        model,x,L,_=load_saved()[0]
        f,a,b=cost_phases(model,x)
        np.testing.assert_allclose(f,a+b,atol=2e-14)
        self.assertEqual(sum(1 for i in range(6) for j in range(i+1,6)),15)
        self.assertEqual(sum(1 for i in GROUPS[0] for j in GROUPS[1]),9)

    def test_shell_higher_orders_are_required(self):
        m=Model(6,831106000,.4);x=np.array([.4,-.2,.1,.7,.9,-.8])
        full=state(m,x,1);reduced=state(m,x,1,singles_only=True)
        self.assertLess(abs(np.vdot(full,reduced))**2,.99)

    def test_random_deeper_state(self):
        rng=np.random.default_rng(3);L=6;m=Model(6,831106004,.8)
        check(m,rng.uniform(-1,1,3+3*L),L,'test')

    def test_zero_templates_match_oracle(self):
        # A single edge has J^2 diagonal, hence a zero K template.
        for edge in (0.,1.):
            J=np.zeros((6,6));J[0,1]=J[1,0]=edge
            m=Model(6,3,.4,J=J);x=np.array([.4,-.2,.1,.7,.9,-.8])
            with self.subTest(edge=edge):
                self.assertTrue(np.isfinite(coefficient_matrix(m,x)).all())
                np.testing.assert_allclose(state(m,x,1),m.state(x,1),atol=1e-13)

if __name__=='__main__':unittest.main()
