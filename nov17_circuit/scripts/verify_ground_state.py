"""Independent dense spin eigensolve and sparse complex128 hardware propagation.

Optional --noise checks timestep sensitivity with 1024 GPU-1 trajectories
at 5 ns and 2.5 ns; preserves the original 512-trajectory result.
"""
import argparse,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from ..sequence import P,load_saved,W,ORDER,state
from .hardware_test import schedule
from ..models.system import DEFAULT_CONFIG as p

def reference():
    m,x,L,_=next(r for r in load_saved() if r[0].seed==831106000 and r[2]==1)
    # Construct each spin operator explicitly; bypass the oracle Hamiltonian kernels.
    I=np.eye(2);X=np.array([[0,1],[1,0]]);Z=np.diag([1,-1])
    def spin(op,i):
        a=np.ones((1,1))
        for j in reversed(range(6)):a=np.kron(a,op if i==j else I)
        return a
    zs=[spin(Z,i) for i in range(6)];xs=[spin(X,i) for i in range(6)]
    H=sum(m.J[i,j]*zs[i]@zs[j] for i in range(6) for j in range(i+1,6))-m.b*sum(xs)
    eig,V=np.linalg.eigh(H);gs=V[:,0];target=(W@gs)[ORDER]
    m.compute_gs_fullspace()
    report={'E0':float(eig[0]),'gap':float(eig[1]-eig[0]),'eigenvector_residual':float(np.linalg.norm(H@gs-eig[0]*gs)),
            'overlap_with_oracle_ground_state':float(abs(np.vdot(gs,m.gs))**2),
            'ideal_ansatz_ground_state_fidelity':float(abs(np.vdot(gs,state(m,x,1)))**2)}
    report['frame_overlap_error']=float(abs(abs(np.vdot(target,state(m,x,1,'X')[ORDER]))**2-report['ideal_ansatz_ground_state_fidelity']))
    return target,report

def sparse_replay(target,N=24):
    dims=(2,2,N,2,N);D=np.prod(dims)
    def embed(op,slot):
        result=sparse.csr_matrix([[1.]])
        for j,d in enumerate(dims):result=sparse.kron(result,op if slot==j else sparse.eye(d),format='csr')
        return result
    a=sparse.diags(np.sqrt(np.arange(1,N)),1,shape=(N,N));n=a.T@a
    aa,ab=embed(a,2),embed(a,4);na,nb=embed(n,2),embed(n,4)
    e=sparse.diags([0.,1.]);es=[embed(e,k) for k in (1,3,0)]
    X=sparse.csr_matrix([[0,1],[1,0]]);Y=sparse.csr_matrix([[0,-1j],[1j,0]])
    # Native excitation form, including the whole auxiliary state throughout.
    h0=p['detuning_bus_MHz']*es[2]-p['chi_local_A_MHz']*na@es[0]-p['chi_local_B_MHz']*nb@es[1]-(p['chi_bus_A_MHz']*na+p['chi_bus_B_MHz']*nb)@es[2]
    hc=[aa+aa.T,1j*(aa.T-aa),ab+ab.T,1j*(ab.T-ab)]+[embed(op,k) for k in (1,3,0) for op in (X,Y)]
    rows,durations,_=schedule(1,local_time=48.)
    v=np.zeros(D,complex);v[0]=1
    for k,(row,T) in enumerate(zip(rows,durations)):
        h=h0.copy()
        for c,op in zip(row,hc):
            if c:h=h+c*op
        A=-2j*np.pi*T*h
        v=expm_multiply(A,v,traceA=A.diagonal().sum())
        if k%150==0:print('SPARSE',k,flush=True)
    # Explicit independent partial trace over all physical auxiliary indices.
    psi=v.reshape(dims);rho=np.zeros((64,64),complex)
    for bus in range(2):
        for qa in range(2):
            for qb in range(2):
                b=psi[bus,qa,:8,qb,:8].ravel();rho+=np.outer(b,b.conj())
    ground=psi[0,0,:8,0,:8].ravel()
    return {'norm':float(np.vdot(v,v).real),'cavity_fidelity':float(np.vdot(target,rho@target).real),
            'all_transmons_g_fidelity':float(abs(np.vdot(target,ground))**2),'logical_survival':float(np.trace(rho).real)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--noise',action='store_true');args=ap.parse_args()
    target,out=reference();out['sparse_complex128_replay']=sparse_replay(target)
    old=json.loads((P/'results/ground_state_fidelity_N6_L1_H48.json').read_text())
    out['difference_from_previous_coherent']=out['sparse_complex128_replay']['cavity_fidelity']-old['results']['without_noise']['ground_state_fidelity_cavities']
    path=P/'results/ground_state_independent_checks.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if args.noise:
        from ..models.trajectories import simulate
        rows,durations,_=schedule(1,local_time=48.);vac=np.zeros(64,complex);vac[0]=1
        out['noise_checks']=[]
        for dt,seed in ((.005,181701),(.0025,181702)):
            r=simulate(rows,durations,None,N=24,samples=1024,dt_max=dt,device=1,initial=vac,desired=target,seed=seed)
            r['fidelity_kind']='ground-state preparation fidelity'
            r['ground_state_fidelity_all_transmons_g']=r.pop('average_gate_fidelity')
            r['ground_state_fidelity_cavities']=r.pop('cavity_only_average_fidelity')
            out['noise_checks'].append(r);path.write_text(json.dumps(out,indent=2)+'\n')
            print('NOISY_CHECK',dt,r['ground_state_fidelity_cavities'],r['cavity_only_standard_error'],flush=True)

if __name__=='__main__':main()
