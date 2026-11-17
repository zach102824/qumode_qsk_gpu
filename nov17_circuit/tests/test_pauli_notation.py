"""Check the published Pauli convention independently of excitation projectors."""
import unittest
import numpy as np
from ..models.system import parameters,system_hamiltonian,kron,named_collapse_operators,lindblad_rhs

class PauliNotationTests(unittest.TestCase):
    def test_full_pauli_hamiltonian_and_lindblad_equivalence(self):
        p=parameters({'detuning_cavity_A_MHz':.12,'detuning_cavity_B_MHz':-.08,'detuning_local_A_MHz':.16,'detuning_local_B_MHz':-.11})
        rng=np.random.default_rng(1717);N=2;d=32;u=rng.normal(0,.1,10)
        q=np.eye(2);I=np.eye(N);a=np.diag([1.],1);n=a.T@a;z=np.diag([-1,1]);sm=np.array([[0,1],[0,0]])
        def embed(op,slot):
            f=[q,q,I,q,I];f[slot]=op;return kron(*f)
        Z={r:embed(z,slot) for r,slot in [('bus',0),('local_A',1),('local_B',3)]}
        chi={'A':{'local_A':p['chi_local_A_MHz'],'bus':p['chi_bus_A_MHz']},'B':{'local_B':p['chi_local_B_MHz'],'bus':p['chi_bus_B_MHz']}}
        h=np.zeros((d,d),complex)
        for j,slot,c in [('A',2,0),('B',4,2)]:
            nj=embed(n,slot);h+=(p[f'detuning_cavity_{j}_MHz']-sum(chi[j].values())/2)*nj
            for r,shift in chi[j].items():h-=shift/2*nj@Z[r]
            eps=u[c]+1j*u[c+1];aa=embed(a,slot);h+=eps*aa.conj().T+eps.conjugate()*aa
        for r,slot,c in [('local_A',1,4),('local_B',3,6),('bus',0,8)]:
            h+=p[f'detuning_{r}_MHz']/2*Z[r]
            omega=u[c]+1j*u[c+1];h+=omega*embed(sm.T,slot)+omega.conjugate()*embed(sm,slot)
        h*=2*np.pi;old=system_hamiltonian(N,u,p)
        offset=np.pi*sum(p[f'detuning_{r}_MHz'] for r in ('local_A','local_B','bus'))
        np.testing.assert_allclose(h,old-offset*np.eye(d),atol=2e-13)
        v=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));rho=v@v.conj().T;rho/=np.trace(rho)
        old_ops=named_collapse_operators(N,p);new_ops=[]
        for label,L in old_ops:
            if label.endswith('_dephasing'):
                r=label[:-len('_dephasing')];gamma=1/p[f'T2_{r}_us']-1/(2*p[f'T1_{r}_us']);L=np.sqrt(gamma/2)*Z[r]
            new_ops.append((label,L))
        np.testing.assert_allclose(lindblad_rhs(rho,h,new_ops),lindblad_rhs(rho,old,old_ops),atol=2e-13)

    def test_nonphysical_parameters_and_complex_drives_rejected(self):
        for override in ({'T1_bus_us':np.nan},{'chi_bus_A_MHz':np.inf},{'T2_buss_us':30},{'cavity_A_number_dephasing_per_us':np.nan}):
            with self.subTest(override=override),self.assertRaises(ValueError):parameters(override)
        with self.assertRaises(ValueError):system_hamiltonian(2,np.ones(10)*1j)
