"""Energy and original-spin ZZ susceptibility from paired cavity measurements."""
from pathlib import Path
from itertools import combinations
import json
import os
import numpy as np
from .reconstruction import calibration
P=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('NOV17_PROJECT_ROOT',P.parent))
SHOTS=8192
PAIRS=list(combinations(range(6),2))
GROUPS=[[0,2,4],[1,3,5]]
LOCAL={i:(g,q) for g,C in enumerate(GROUPS) for q,i in enumerate(C)}
MASKS=[1,2,4,3,5,6]
def states():
    from cpu_oracle.src.hamiltonian import Model,walsh
    reorder=[sum(((a>>q)&1)<<GROUPS[0][q] for q in range(3))+
             sum(((b>>q)&1)<<GROUPS[1][q] for q in range(3)) for b in range(8) for a in range(8)]
    old=json.loads((P/'results/state_manifest.json').read_text())
    out=[]
    for r in old['records']:
        saved=json.loads((ROOT/r['source']).read_text())
        m=Model(6,r['seed'],saved['b']);v=m.state(saved['x'],r['L'])
        zz=1-2*((np.arange(64)[:,None]>>np.arange(6))&1)
        C=zz.T@((abs(v)**2)[:,None]*zz)
        out.append(dict(seed=r['seed'],L=r['L'],J=m.J,B=m.b,
                        psi=walsh(v,6)[reorder].reshape(8,8),energy=saved['best_E'],
                        C=np.array([C[i,j] for i,j in PAIRS]),chi=float(np.sum(C*C)/6)))
    return out

def design_frame(E,cal):
    S=len(E);w=cal['weights'];f=cal['f'];observations=[]
    for a in range(S):
        for b in range(S):
            y=np.zeros((9,9,16));local=[w[:,a],w[:,b]]
            for k,(i,j) in enumerate(PAIRS):
                g,q=LOCAL[i];h,r=LOCAL[j]
                if g==h:
                    weight=local[g][MASKS.index((1<<q)|(1<<r))]
                    y[:,:,k]=weight[:,None] if g==0 else weight[None,:]
                else:
                    left=local[0][q if g==0 else r]
                    right=local[1][r if h==1 else q]
                    y[:,:,k]=left[:,None]*right[None,:]
            y[:,:,-1]=w[6:,a].sum(axis=0)[:,None]+w[6:,b].sum(axis=0)[None,:]
            observations.append(y.reshape(81,16))
    return dict(E=E,y=np.array(observations),f=np.outer(f,f).ravel(),
                settings=[(a,b) for a in range(S) for b in range(S)],cal=cal)

def design_raw(E,ideal=False):
    # Two global configurations: native/native and full/full.
    z=1-2*((np.arange(8)[:,None]>>np.arange(3))&1)
    z=np.vstack([z,np.zeros((1,3))]);y=np.zeros((2,9,9,16))
    y[0,:,:,-1]=z.sum(axis=1)[:,None]+z.sum(axis=1)[None,:]
    for k,(i,j) in enumerate(PAIRS):
        g,q=LOCAL[i];h,r=LOCAL[j]
        if g==h:
            val=z[:,q]*z[:,r]
            y[1,:,:,k]=val[:,None] if g==0 else val[None,:]
        else:
            y[1,:,:,k]=z[:,q if g==0 else r,None]*z[:,r if h==1 else q][None,:]
    # Inverse setting probabilities are unnecessary: each observable uses its batch.
    return dict(E=E[:2],y=y.reshape(2,81,16),f=np.ones(2),settings=[(0,0),(1,1)],
                cal=dict(rank=None,residual=None,max_weight=1,ridge=None))

def probabilities(E,psi,settings):
    flat=E.reshape(-1,8,8)
    p=np.einsum('ba,tac,ubd,dc->tu',psi.conj(),flat,flat,psi,optimize=True).real
    S=len(E);p=p.reshape(S,9,S,9).transpose(0,2,1,3)
    assert p.min()>-1e-9
    p=np.maximum(p,0)
    result=np.array([p[a,b].ravel() for a,b in settings])
    assert np.max(abs(result.sum(axis=1)-1))<1e-8
    return result/result.sum(axis=1)[:,None]

def moments(p,y):
    mu=np.einsum('go,gok->gk',p,y)
    second=np.einsum('go,gok,gol->gkl',p,y,y,optimize=True)
    return mu,second-np.einsum('gk,gl->gkl',mu,mu)

def allocate(design,state,shots=SHOTS):
    E=design['E'];settings=design['settings'];y=design['y'];f=design['f']
    local=np.trace(E,axis1=2,axis2=3).real/8
    prior=np.array([np.outer(local[a],local[b]).ravel() for a,b in settings])
    mu,cov=moments(prior,y)
    h=np.r_[[state['J'][i,j] for i,j in PAIRS],-state['B']]
    # Equal-sized energy/spin and average-correlation precision penalties.
    cost=np.einsum('k,gkl,l->g',h,cov,h)/36+np.trace(cov[:,:15,:15],axis1=1,axis2=2)/15
    fraction=f*np.sqrt(np.maximum(cost,1e-15));fraction/=fraction.sum()
    if shots<2*len(f):
        raise ValueError('Budget requires at least two shots per setting pair')
    n=2+np.floor((shots-2*len(f))*fraction).astype(int)
    n[np.argmax(fraction)]+=shots-n.sum()
    assert n.sum()==shots and n.min()>=2
    return n

def risk(design,state,actual_E=None,return_detail=False):
    y=design['y'];f=design['f'];n=allocate(design,state)
    p=probabilities(design['E'] if actual_E is None else actual_E,state['psi'],design['settings'])
    mu,cov=moments(p,y);mean=f@mu
    V=np.einsum('g,gkl->kl',f*f/n,cov)
    h=np.r_[[state['J'][i,j] for i,j in PAIRS],-state['B']]
    biasE=float(h@mean-state['energy']);varE=float(h@V@h)
    c=mean[:15];Vc=V[:15,:15];covc=cov[:,:15,:15]
    expected_chi=1+np.dot(c,c)/3
    biasChi=expected_chi-state['chi']
    # Exact Hoeffding decomposition, including fixed-batch finite-n correction.
    varChi=(4*c@Vc@c+2*np.sum(Vc*Vc)+
            2*np.sum(f**4/(n*n*(n-1))*np.sum(covc*covc,axis=(1,2))))/9
    result=dict(seed=state['seed'],L=state['L'],energy_bias_per_spin=biasE/6,
                energy_sd_per_spin=float(np.sqrt(max(0,varE))/6),
                energy_rmse_per_spin=float(np.sqrt(max(0,varE)+biasE*biasE)/6),
                chi_bias=float(biasChi),chi_sd=float(np.sqrt(max(0,varChi))),
                chi_rmse=float(np.sqrt(max(0,varChi)+biasChi*biasChi)),
                mean_abs_C_error=float(np.mean(abs(c-state['C']))),
                naive_square_chi_extra_bias=float(np.trace(Vc)/3),
                native_native_shots=int(n[0]),min_shots=int(n.min()),max_shots=int(n.max()))
    if return_detail:return result,dict(p=p,mu=mu,cov=cov,n=n,y=y,f=f,V=V,mean=mean)
    return result

def summarize(rows):
    out=[]
    for L in (1,2):
        r=[v for v in rows if v['L']==L]
        out.append(dict(L=L,energy_rmse_per_spin=float(np.sqrt(np.mean([v['energy_rmse_per_spin']**2 for v in r]))),
                        chi_rmse=float(np.sqrt(np.mean([v['chi_rmse']**2 for v in r]))),
                        mean_absolute_energy_bias_per_spin=float(np.mean([abs(v['energy_bias_per_spin']) for v in r])),
                        mean_absolute_chi_bias=float(np.mean([abs(v['chi_bias']) for v in r]))))
    return out


def load_design(scenario='ancilla',full=False):
    if scenario not in ('coherent','loss','ancilla'):
        raise ValueError('Unknown calibration scenario')
    path=P/'calibration'/f"{'full' if full else 'compact'}_{scenario}.npz"
    with np.load(path) as data:
        return dict(E=data['effects'],y=data['y'],f=data['f'],settings=data['settings'].tolist(),cal={})
