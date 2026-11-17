"""Units, trace preservation, relaxation and dephasing conventions."""
import unittest
import numpy as np
from ..models.system import *

class SystemTests(unittest.TestCase):
    def test_hamiltonian_hermitian(self):
        h0,hc=hamiltonians(2)
        self.assertEqual(h0.shape,(32,32));self.assertEqual(hc.shape,(10,32,32))
        np.testing.assert_allclose(h0,h0.conj().T)
        np.testing.assert_allclose(hc,hc.conj().transpose(0,2,1))

    def test_rhs_preserves_trace_and_hermiticity(self):
        rng=np.random.default_rng(1);v=rng.normal(size=(32,32))+1j*rng.normal(size=(32,32))
        rho=v@v.conj().T;rho/=np.trace(rho)
        h=system_hamiltonian(2,np.arange(10)*.01)
        drho=lindblad_rhs(rho,h,named_collapse_operators(2))
        self.assertLess(abs(np.trace(drho)),1e-12)
        np.testing.assert_allclose(drho,drho.conj().T,atol=1e-12)

    def test_local_transmon_T1_T2(self):
        N=2;g=0;e=np.ravel_multi_index((0,1,0,0,0),(2,2,N,2,N))
        v=np.zeros(32,complex);v[g]=v[e]=1/np.sqrt(2);rho=np.outer(v,v.conj())
        r=lindblad_rhs(rho,np.zeros((32,32)),named_collapse_operators(N))
        self.assertAlmostEqual(r[e,e].real,-.5/50,places=12)
        self.assertAlmostEqual(r[g,e].real,-.5/30,places=12)

    def test_cavity_energy_lifetime(self):
        N=2;k=np.ravel_multi_index((0,0,1,0,0),(2,2,N,2,N))
        rho=np.zeros((32,32),complex);rho[k,k]=1
        r=lindblad_rhs(rho,np.zeros((32,32)),named_collapse_operators(N))
        self.assertAlmostEqual(r[k,k].real,-1/600,places=12)

    def test_two_dephasing_conventions_equivalent(self):
        rho=np.array([[.5,.2+.1j],[.2-.1j,.5]],complex);p=np.diag([0,1]);z=np.diag([1,-1])
        def dissip(L):
            a=L.conj().T@L
            return L@rho@L.conj().T-.5*(a@rho+rho@a)
        gamma=1/30-1/100
        np.testing.assert_allclose(dissip(np.sqrt(2*gamma)*p),dissip(np.sqrt(gamma/2)*z),atol=1e-15)

    def test_unphysical_T2_rejected(self):
        with self.assertRaises(ValueError):parameters({'T2_bus_us':101.})

if __name__=='__main__':unittest.main()
