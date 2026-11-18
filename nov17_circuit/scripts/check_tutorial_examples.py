"""Recompute tutorial examples against the stored model and basic operators."""
import json
import numpy as np
from ..sequence import P,load_saved,coefficient_matrix,W8

def main():
    m,x,L,_=next(r for r in load_saved() if r[0].seed==831106000 and r[2]==1)
    z=np.diag([1.,-1.]);xx=np.array([[0.,1.],[1.,0.]])
    H=.4*np.kron(z,z)-.2*(np.kron(xx,np.eye(2))+np.kron(np.eye(2),xx))
    candidates=[np.eye(4)[:,0],np.eye(4)[:,1],np.ones(4)/2]
    energies=[float(v@H@v) for v in candidates]
    np.testing.assert_allclose(energies,[.4,-.4,-.4])
    np.testing.assert_allclose(W8[:,1],(-1.)**np.arange(8)/np.sqrt(8))
    k=np.arange(4);f=x[3]*k+x[4]*k*(k-1)/2+x[5]*k*(k-1)*(k-2)/6
    np.testing.assert_allclose(f,[0,.8301622,1.6142441,2.3183338],atol=1e-7)
    assert abs(coefficient_matrix(m,x)[0,1]-.0264089)<1e-7
    # Both computational inputs give uniform joint X probabilities, but opposite ZZ.
    W2=np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2),np.array([[1,1],[1,-1]])/np.sqrt(2))
    np.testing.assert_allclose(abs(W2[:,0])**2,abs(W2[:,1])**2)
    entangled=np.array([[1,1],[1,-1]])/2
    assert np.linalg.matrix_rank(entangled)==2
    saved=json.loads((P/'results/ground_state_independent_checks.json').read_text())
    np.testing.assert_allclose([r['ground_state_fidelity_cavities'] for r in saved['noise_checks']],[.3513,.3466],atol=.00005)
    np.testing.assert_allclose([r['ground_state_fidelity_all_transmons_g'] for r in saved['noise_checks']],[.3270,.3226],atol=.00005)
    assert abs(saved['sparse_complex128_replay']['cavity_fidelity']-.71844)<.000005
    results=dict(two_spin_candidate_energies=energies,pair_phase_A01=float(coefficient_matrix(m,x)[0,1]),
                 shell_phases=f.tolist(),gamma_phi_per_us=1/30-1/100,T_phi_us=1/(1/30-1/100),
                 pi_pulse_us=np.pi/(2*2*np.pi*.1),Fock7_first_loss_mean_us=600/7,
                 stored_fidelities='Rounded table values checked against ground_state_independent_checks.json')
    (P/'results/tutorial_example_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
