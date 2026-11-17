"""Schedule validation and optional independent GPU-1 noise benchmarks.

Run GPU checks with NOV17_RUN_GPU_TESTS=1; they use synthetic stronger noise
for statistical sensitivity and do not overwrite any saved hardware results.
"""
import os,unittest
import numpy as np
from scipy.integrate import solve_ivp
from ..models.trajectories import simulate,validate_run

class TrajectoryValidationTests(unittest.TestCase):
    def test_invalid_inputs_fail_before_device_allocation(self):
        rows=np.zeros((2,10));durations=np.ones(2)*.01;tar={'U':np.eye(64)}
        bad=[dict(durations=[.01]),dict(durations=[0,.01]),dict(dt_max=0),dict(samples=0),dict(N=7),dict(rows=rows+1j),dict(initial=np.ones(64)),dict(tar={'U':np.zeros((64,64))})]
        for change in bad:
            args=dict(rows=rows,durations=durations,N=8,samples=2,dt_max=.005,tar=tar);args.update(change)
            with self.subTest(change=change.keys()),self.assertRaises(ValueError):validate_run(**args)

@unittest.skipUnless(os.environ.get('NOV17_RUN_GPU_TESTS')=='1','set NOV17_RUN_GPU_TESTS=1 for GPU 1')
class TrajectoryPhysicsTests(unittest.TestCase):
    def check_result(self,r,expected):
        self.assertLess(abs(r['average_gate_fidelity']-expected),5*r['standard_error']+.003)
        self.assertFalse(r['postselection'])

    def run_case(self,row,T,v,parameters,seed):
        return simulate(np.array([row]),[T],None,N=8,samples=4096,dt_max=.005,device=1,initial=v,desired=v,parameters=parameters,seed=seed)

    def test_cavity_loss_matches_exponential(self):
        v=np.zeros(64,complex);v[8]=1 # cavity A=1, B=0
        r=self.run_case(np.zeros(10),.5,v,{'T1_cavity_A_us':.5},17171)
        self.check_result(r,np.exp(-1))

    def test_cavity_dephasing_matches_coherence_decay(self):
        v=np.zeros(64,complex);v[0]=v[8]=1/np.sqrt(2)
        r=self.run_case(np.zeros(10),.5,v,{'T1_cavity_A_us':np.inf,'cavity_A_number_dephasing_per_us':4.},17172)
        self.check_result(r,(1+np.exp(-1))/2)

    def test_driven_bus_matches_density_matrix_ode(self):
        row=np.zeros(10);row[-2:]=[.8,.2];T=.15
        v=np.zeros(64,complex);v[0]=1
        r=self.run_case(row,T,v,{'detuning_bus_MHz':0.,'T1_bus_us':.4,'T2_bus_us':.3},17173)
        omega=2*np.pi*(.8+.2j);h=np.array([[0,omega.conjugate()],[omega,0]])
        ops=[np.array([[0,1],[0,0]])/np.sqrt(.4),np.diag([0,1])*np.sqrt(2*(1/.3-1/.8))]
        def rhs(t,x):
            rho=x.reshape(2,2);dr=-1j*(h@rho-rho@h)
            for L in ops:
                a=L.conj().T@L;dr+=L@rho@L.conj().T-.5*(a@rho+rho@a)
            return dr.ravel()
        rho=solve_ivp(rhs,(0,T),np.array([1,0,0,0],complex),rtol=1e-10,atol=1e-12).y[:,-1].reshape(2,2)
        self.check_result(r,rho[0,0].real)
        self.assertAlmostEqual(r['cavity_only_average_fidelity'],1.,places=4)
