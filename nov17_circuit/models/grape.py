"""One dispersively coupled cavity and ancilla: measurement-GRAPE synthesis.

Units: chi=1, time in chi^-1. Logical input: ancilla g and Fock levels 0..7.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from scipy.linalg import expm, eigh, hadamard
D=8
W=hadamard(D)/np.sqrt(D)
def metrics(v,ancilla=False):
 K=v[:D]; amp=np.trace(W.conj().T@K)
 retained=float(np.vdot(K,K).real/D)
 return dict(measurement_fidelity=float(np.sum(abs(np.sum(W.conj()*K,axis=1))**2)/D),process_overlap=float(abs(amp)**2/D**2),average_fidelity=float((np.vdot(K,K).real+abs(amp)**2)/(D*(D+1))),mean_leakage=1-retained,worst_input_leakage=float(np.linalg.eigvalsh(np.eye(D)-K.conj().T@K)[-1]))

def oscillator(N):
 a=np.diag(np.sqrt(np.arange(1,N)),1);return a,a.T@a

def controls(N):
 a,n=oscillator(N);I=np.eye(N);sx=np.array([[0,1],[1,0]]);sy=np.array([[0,-1j],[1j,0]]);e=np.diag([0,1])
 H0=-np.kron(e,n)
 Hc=np.array([np.kron(np.eye(2),a+a.T),np.kron(np.eye(2),1j*(a.T-a)),np.kron(sx,I),np.kron(sy,I)])
 return H0,Hc

def grape_eval(x,T,steps,N,gradient=True,measurement=False):
 H0,Hc=controls(N);dt=T/steps;v=np.eye(2*N,dtype=complex)[:,:D];target=np.zeros_like(v);target[:D]=W
 vs=[];ops=[];ds=[]
 for row in x.reshape(steps,4):
  lam,V=eigh(H0+np.einsum('a,aij->ij',row,Hc));p=np.exp(-1j*dt*lam);u=(V*p)@V.conj().T
  if gradient:
   F=-1j*dt*np.exp(-1j*dt*(lam[:,None]+lam[None,:])/2)*np.sinc(dt*(lam[:,None]-lam[None,:])/(2*np.pi))
   deriv=np.array([V@(F*(V.conj().T@c@V))@V.conj().T for c in Hc]);ds.append(deriv);vs.append(v)
  ops.append(u);v=u@v
 z=np.vdot(target,v)
 if not gradient:return v
 score=abs(z)**2/D**2;denom=D**2
 if measurement:
  zr=np.sum(W.conj()*v[:D],axis=1);score=float(np.sum(abs(zr)**2)/D);target[:D]=W*zr[:,None];z=1.;denom=D
 adj=target.copy();g=np.zeros((steps,4))
 for j in reversed(range(steps)):
  for c in range(4):g[j,c]=-2*np.real(z.conjugate()*np.vdot(adj,ds[j][c]@vs[j]))/denom
  adj=ops[j].conj().T@adj
 return 1-score,g.ravel()

def check(fun,x):
 e,g=fun(x)
 for j in [0,len(x)//2,len(x)-1]:
  delta=np.zeros_like(x);delta[j]=1e-6
  fd=(fun(x+delta)[0]-fun(x-delta)[0])/2e-6
  assert abs(fd-g[j])<2e-7,(j,fd,g[j])
