"""Readout POVMs with cavity loss and optional ancilla damping/dephasing.

The ancilla is traced out. Fock >=8 is retained as a leakage flag.
These are illustrative model parameters, not fitted device specifications.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import comb
from .grape import controls
from .system import DEFAULT_CONFIG as NOISE
def segments(rec,N):
    """Return physical Hamiltonians and durations in microseconds at chi/2pi=1 MHz."""
    x=np.array(rec['parameters']);chi=2*np.pi
    H0,Hc=controls(N);T=rec.get('total_chi_time') or rec['value']
    return [(chi*(H0+np.einsum('a,aij->ij',row,Hc)),T/(len(x)//4)/chi,'grape')
            for row in x.reshape(-1,4)]

def native():
    E=np.zeros((9,8,8),complex)
    for k in range(8):E[k,k,k]=1
    return E

def loss_adjoint(E,coeff,anc,N,eta_a,dephase):
    q=2 if anc else 1
    batch=len(E)
    b=E.reshape(batch,q,N,q,N).transpose(0,1,3,2,4).copy()
    if anc:
        b[:,1,1]=eta_a*b[:,1,1]+(1-eta_a)*b[:,0,0]
        b[:,0,1]*=np.sqrt(eta_a)*dephase
        b[:,1,0]*=np.sqrt(eta_a)*dephase
    out=np.zeros_like(b)
    for l,c in coeff:
        out[...,l:,l:]+=b[...,:N-l,:N-l]*c
    return out.transpose(0,1,3,2,4).reshape(batch,q*N,q*N)

def effects(rec,N=24,T1=NOISE['T1_cavity_A_us'],ancilla_noise=False,subdivide=2):
    anc=True;q=2
    E=np.zeros((9,q*N,q*N),complex)
    for a in range(q):
        for n in range(N):E[min(n,8),a*N+n,a*N+n]=1
    seg=segments(rec,N);cache={}
    for H,T,kind in reversed(seg):
        count=max(subdivide,int(np.ceil(T/.15))*subdivide)
        dt=T/count
        U=expm(-1j*H*dt)
        if dt not in cache:
            eta=np.exp(-dt/(2*T1)) if T1 else 1.
            coeff=[]
            for l in range(N if T1 else 1):
                k=np.arange(l,N)
                c=np.sqrt(comb(k,l)[:,None]*comb(k,l)[None,:])*(1-eta)**l*eta**((np.arange(N-l)[:,None]+np.arange(N-l)[None,:])/2)
                coeff.append((l,c))
            ea=np.exp(-dt/(2*NOISE['T1_local_A_us'])) if ancilla_noise and anc else 1.
            dephase=np.exp(-(1/NOISE['T2_local_A_us']-1/(2*NOISE['T1_local_A_us']))*dt/2) if ancilla_noise and anc else 1.
            cache[dt]=(coeff,ea,dephase)
        coeff,ea,dephase=cache[dt]
        for _ in range(count):
            E=loss_adjoint(E,coeff,anc,N,ea,dephase)
            E=U.conj().T@E@U
            E=loss_adjoint(E,coeff,anc,N,ea,dephase)
    E=E[:,:8,:8]
    assert np.max(abs(E.sum(axis=0)-np.eye(8)))<1e-9
    assert min(np.linalg.eigvalsh(E).ravel())>-1e-9
    return E,sum(T for _,T,_ in seg)

def assignment(E,p=.01):
    out=E.copy();out[:8]=(1-p)*E[:8]+p*np.roll(E[:8],1,axis=0)
    return out
