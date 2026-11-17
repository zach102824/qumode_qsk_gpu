"""Calibrated local observable weights, independent of the incoming state."""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
D=8

def targets():
    masks=(1,2,4,3,5,6)
    matrices=[np.eye(8)[np.arange(8)^m] for m in masks]
    for m in (1,2,4):
        matrices.append(np.diag([(-1)**((k&m).bit_count()) for k in range(8)]))
    return ['X'+str(m) for m in masks]+['Z1','Z2','Z4'],np.array(matrices)

def vector(A):
 A=(A+A.conj().T)/2
 vals=[np.diag(A).real]
 iu=np.triu_indices(D,1);vals += [np.sqrt(2)*A[iu].real,np.sqrt(2)*A[iu].imag]
 return np.concatenate(vals)

def calibration(E,ridge=0.,optimize=True):
    S=len(E);V=np.array([[vector(e) for e in row] for row in E])
    prior=np.maximum(np.trace(E,axis1=2,axis2=3).real/8,1e-10)
    Gs=np.einsum('sna,snb,sn->sab',V,V,1/prior)
    O=np.array([vector(o) for o in targets()[1]]).T
    importance=np.array([3,3,3,1,1,1,1,1,1.])
    def solve(f):
        G=np.einsum('s,sab->ab',f,Gs)
        e,U=eigh(G);threshold=max(e[-1]*1e-12,1e-13)
        inv=(U*np.where(e>threshold,1/(np.maximum(e,threshold)+ridge),0))@U.T
        return inv@O,e
    def fg(f):
        Q,e=solve(f)
        return float(np.sum(O*Q*importance)), -np.einsum('ak,sab,bk,k->s',Q,Gs,Q,importance)
    f=np.ones(S)/S
    if optimize:
        fit=minimize(fg,f,jac=True,method='SLSQP',bounds=[(.003,1)]*S,
                     constraints={'type':'eq','fun':lambda f:f.sum()-1,'jac':lambda f:np.ones(S)},
                     options={'maxiter':200,'ftol':1e-8})
        f=np.maximum(fit.x,.003);f/=f.sum()
    Q,e=solve(f)
    weights=np.einsum('sna,ak->ksn',V,Q)/prior[None]
    R=np.einsum('s,ksn,snij->kij',f,weights,E)
    residual=max(np.linalg.norm(a-b,2) for a,b in zip(R,targets()[1]))
    return dict(weights=weights,f=f,rank=int(np.sum(e>max(e[-1]*1e-12,1e-13))),
                residual=float(residual),max_weight=float(abs(weights).max()),ridge=ridge)
