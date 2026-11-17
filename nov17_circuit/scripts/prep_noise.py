"""Local state-preparation density-matrix replay with the inherited noise model."""
from .stateprep import *
from scipy.linalg import expm
from scipy.special import comb
from ..models.system import DEFAULT_CONFIG as NOISE

def evaluate(rec,N=24,subdivide=2):
    H0,Hc=controls(N);rows=np.array(rec['parameters']).reshape(-1,4)
    dt=rec['duration_us']/len(rows)/subdivide;rho=np.zeros((2*N,2*N),complex);rho[0,0]=1
    eta=np.exp(-dt/(2*NOISE['T1_cavity_A_us']));ea=np.exp(-dt/(2*NOISE['T1_local_A_us']));deph=np.exp(-(1/NOISE['T2_local_A_us']-1/(2*NOISE['T1_local_A_us']))*dt/2)
    coeff=[]
    for l in range(N):
        k=np.arange(l,N)
        coeff.append((l,np.sqrt(comb(k,l)[:,None]*comb(k,l)[None,:])*(1-eta)**l*
            eta**((np.arange(N-l)[:,None]+np.arange(N-l)[None,:])/2)))
    def noise(r):
        b=r.reshape(2,N,2,N).transpose(0,2,1,3);out=np.zeros_like(b)
        for l,c in coeff:out[...,:N-l,:N-l]+=b[...,l:,l:]*c
        out[0,0]+=(1-ea)*out[1,1];out[1,1]*=ea
        out[0,1]*=np.sqrt(ea)*deph;out[1,0]*=np.sqrt(ea)*deph
        return out.transpose(0,2,1,3).reshape(2*N,2*N)
    for row in rows:
        U=expm(-1j*dt*2*np.pi*(H0+np.einsum('a,aij->ij',row,Hc)))
        for _ in range(subdivide):rho=noise(U@noise(rho)@U.conj().T)
    target=np.ones(8)/np.sqrt(8)
    assert abs(np.trace(rho)-1)<1e-10
    ground=float(np.vdot(target,rho[:8,:8]@target).real)
    traced=float(np.vdot(target,(rho[:8,:8]+rho[N:N+8,N:N+8])@target).real)
    return dict(ground_ancilla_state_fidelity=ground,cavity_only_state_fidelity=traced,
        ancilla_ground_probability=float(np.trace(rho[:N,:N]).real),
        two_cavity_ground_ancilla_product_fidelity=ground**2)

def main():
    out=[]
    for T in (12,24,36):
        rows=json.loads((P/f'results/optimization/stateprep_T{T}.json').read_text())['records']
        best=max(rows,key=lambda r:r['cutoff32']['preparation_fidelity'])
        base=evaluate(best);fine=evaluate(best,subdivide=4)
        rec=dict(chi_time=T,duration_us=best['duration_us'],noiseless=best['cutoff32'],noisy=base,
            timestep_change=abs(base['ground_ancilla_state_fidelity']-fine['ground_ancilla_state_fidelity']),
            parameters=best['parameters'],steps=96,assumptions='Local A noise parameters from config/noise_model.json; identical-cavity product estimate; chi/2pi=1MHz')
        assert rec['timestep_change']<1e-4
        out.append(rec);print(T,rec['duration_us'],base,flush=True)
    (P/'results/stateprep_noise.json').write_text(json.dumps(dict(records=out),indent=2)+'\n')
    best=max(out,key=lambda r:r['noisy']['ground_ancilla_state_fidelity'])
    (P/'pulses/selected_initial_prep.json').write_text(json.dumps(best,indent=2)+'\n')

if __name__=='__main__':main()
