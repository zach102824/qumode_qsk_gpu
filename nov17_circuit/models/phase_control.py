"""Shared/local-transmon number-selective phase control and channel scores."""
import numpy as np
from scipy.linalg import expm
from .system import DEFAULT_CONFIG
def gate_metrics(K,U):
    d=U.shape[0];trace=np.vdot(U,K);survival=float(np.vdot(K,K).real/d)
    return dict(average_gate_fidelity=float((abs(trace)**2+np.vdot(K,K).real)/(d*(d+1))),
                process_overlap=float(abs(trace)**2/d**2),mean_logical_survival=survival,
                mean_leakage=1-survival)

def bus_detunings(N=8,chiA=None,chiB=None):
    chiA=DEFAULT_CONFIG['chi_bus_A_MHz'] if chiA is None else chiA
    chiB=DEFAULT_CONFIG['chi_bus_B_MHz'] if chiB is None else chiB
    n,m=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    # Fixed carrier at center of the logical 0..7 manifold, even in larger replays.
    return 2*np.pi*(chiA*n+chiB*m-DEFAULT_CONFIG['detuning_bus_MHz'])

def propagate(x,T,delta,gradient=False,phases=None):
    rows=x.reshape(-1,2);dt=T/len(rows);v=np.zeros((len(delta),2),complex);v[:,0]=1
    X=np.array([[0,1],[1,0]],complex)*2*np.pi;Y=np.array([[0,-1j],[1j,0]])*2*np.pi
    hc=np.array([X,Y]);history=[]
    for row in rows:
        H=np.zeros((len(delta),2,2),complex);H[:,1,1]=-delta
        H+=np.einsum('a,aij->ij',row,hc)[None]
        lam,V=np.linalg.eigh(H);p=np.exp(-1j*dt*lam)
        U=(V*p[:,None,:])@V.conj().transpose(0,2,1)
        if gradient:
            F=-1j*dt*np.exp(-1j*dt*(lam[:,:,None]+lam[:,None,:])/2)*np.sinc(dt*(lam[:,:,None]-lam[:,None,:])/(2*np.pi))
            ds=np.array([V@(F*(V.conj().transpose(0,2,1)@c@V))@V.conj().transpose(0,2,1) for c in hc])
            history.append((v,U,ds))
        v=np.einsum('bij,bj->bi',U,v)
    if not gradient:return v
    d=len(delta);target=np.exp(-1j*phases);K=v[:,0];trace=np.vdot(target,K)
    score=(abs(trace)**2+np.vdot(K,K).real)/(d*(d+1))
    adj=np.zeros_like(v);adj[:,0]=(trace*target+K)/(d*(d+1));g=np.zeros_like(rows)
    for j in reversed(range(len(rows))):
        before,U,ds=history[j]
        g[j]=-2*np.real(np.einsum('bi,abij,bj->a',adj.conj(),ds,before))
        adj=np.einsum('bji,bj->bi',U.conj(),adj)
    return float(1-score),g.ravel()

def noisy_bus(rec,tar,subdivide=2):
    """Exact process fidelity for the diagonal bus gate with traced bus.

    Photon-loss jumps have zero target trace because photon number only drops.
    Their contribution to entanglement fidelity is therefore zero. Ancilla
    noise is evolved on every off-diagonal Fock-pair block, without postselection.
    The cavity-output channel remains trace-preserving within levels 0..7.
    """
    rows=np.array(rec['parameters']).reshape(-1,2);dt=rec['duration_us']/len(rows)/subdivide
    delta=bus_detunings().ravel();rho=np.zeros((64,64,2,2),complex);rho[:,:,0,0]=1
    T1=DEFAULT_CONFIG['T1_bus_us'];T2=DEFAULT_CONFIG['T2_bus_us']
    ea=np.exp(-dt/(2*T1));deph=np.exp(-(1/T2-1/(2*T1))*dt/2)
    def noise(r):
        r=r.copy();r[:,:,0,0]+=(1-ea)*r[:,:,1,1];r[:,:,1,1]*=ea
        r[:,:,0,1]*=np.sqrt(ea)*deph;r[:,:,1,0]*=np.sqrt(ea)*deph
        return r
    for row in rows:
        H=np.zeros((64,2,2),complex);H[:,1,1]=-delta
        H[:,0,1]=2*np.pi*(row[0]-1j*row[1]);H[:,1,0]=H[:,0,1].conj()
        U=np.array([expm(-1j*dt*h) for h in H])
        for _ in range(subdivide):
            rho=noise(rho);rho=np.einsum('iab,ijbc,jdc->ijad',U,rho,U.conj(),optimize=True);rho=noise(rho)
    n,m=np.meshgrid(np.arange(8),np.arange(8),indexing='ij')
    damping=np.exp(-(n/DEFAULT_CONFIG['T1_cavity_A_us']+m/DEFAULT_CONFIG['T1_cavity_B_us']).ravel()*rec['duration_us']/2)
    d=np.exp(-1j*tar['phase'].ravel());coherence=np.trace(rho,axis1=2,axis2=3)
    Fe=float(np.einsum('i,j,ij,i,j->',d.conj(),d,coherence,damping,damping).real/4096)
    return dict(average_gate_fidelity=float((64*Fe+1)/65),entanglement_fidelity=Fe,
                bus_excitation_without_photon_jumps=float(np.mean(rho[np.arange(64),np.arange(64),1,1]).real),
                note='Bus phase ONLY; bus traced/reset assumed, no local-H errors or reset cost. Cavity and bus noise included.')
