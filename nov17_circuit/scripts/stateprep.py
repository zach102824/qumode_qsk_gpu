"""Benchmark preparation of the uniform 0..7 superposition from vacuum."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from ..models.grape import controls,grape_eval
from ..sequence import P

def objective(x,T,N=16):
    steps=len(x)//4;dt=T/steps;H0,Hc=controls(N)
    v=np.zeros((2*N,1),complex);v[0,0]=1
    target=np.zeros_like(v);target[:8,0]=1/np.sqrt(8);history=[]
    for row in x.reshape(-1,4):
        lam,V=eigh(H0+np.einsum('a,aij->ij',row,Hc));p=np.exp(-1j*dt*lam);U=(V*p)@V.conj().T
        F=-1j*dt*np.exp(-1j*dt*(lam[:,None]+lam[None,:])/2)*np.sinc(dt*(lam[:,None]-lam[None,:])/(2*np.pi))
        ds=np.array([V@(F*(V.conj().T@c@V))@V.conj().T for c in Hc])
        history.append((v,U,ds));v=U@v
    z=np.vdot(target,v);adj=z*target;g=np.zeros((steps,4))
    for j in reversed(range(steps)):
        before,U,ds=history[j];g[j]=-2*np.real(np.einsum('ij,aij->a',adj.conj(),ds@before));adj=U.conj().T@adj
    return float(1-abs(z)**2),g.ravel()

def replay(x,T,N):
    v=grape_eval(x,T,len(x)//4,N,gradient=False)[:,0]
    return dict(preparation_fidelity=float(abs(np.sum(v[:8])/np.sqrt(8))**2),
                logical_ground_ancilla_survival=float(np.sum(abs(v[:8])**2)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--T',type=float,required=True);p.add_argument('--iterations',type=int,default=1000)
    args=p.parse_args();rng=np.random.default_rng(171118+int(args.T));records=[]
    for restart in range(2):
        x=rng.normal(0,.08,96*4);fun=lambda x:objective(x,args.T)
        if restart==0:
            _,g=fun(x)
            for k in (0,99,383):
                d=np.zeros_like(x);d[k]=1e-6
                assert abs((fun(x+d)[0]-fun(x-d)[0])/2e-6-g[k])<2e-7
        start=time.monotonic()
        fit=minimize(fun,x,jac=True,method='L-BFGS-B',bounds=[(-.25,.25),(-.25,.25),(-1,1),(-1,1)]*96,
                     options={'maxiter':args.iterations,'ftol':1e-12,'gtol':1e-7,'maxls':30})
        r=dict(method='grape',objective='state preparation only: ancilla g, uniform Fock 0..7',
            value=args.T,cutoff=16,steps=96,parameters=fit.x.tolist(),duration_us=args.T/(2*np.pi),
            seconds=time.monotonic()-start,iterations=int(fit.nit),success=bool(fit.success),
            restart=restart,cutoff24=replay(fit.x,args.T,24),cutoff32=replay(fit.x,args.T,32))
        records.append(r);(P/f'results/optimization/stateprep_T{args.T:g}.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
        print('PREP',args.T,restart,r['duration_us'],r['seconds'],r['cutoff32'],flush=True)

if __name__=='__main__':main()
