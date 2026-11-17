"""Original-Z-coordinate circuit and independently constructed diagonal phases.

Chronological: prepare H|0>, then [cost_ZZ, H, local_shell_Z, H] per layer.
For original-X readout omit only the last H. Production Nov3 is unchanged.
"""
from pathlib import Path
import json
import os
import numpy as np
from scipy.linalg import hadamard
from cpu_oracle.src.hamiltonian import Model

P=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('NOV17_PROJECT_ROOT',P.parent))
GROUPS=((0,2,4),(1,3,5))
W=hadamard(64)/8
W8=hadamard(8)/np.sqrt(8)
BITS=((np.arange(64)[:,None]>>np.arange(6))&1)
Z=1-2*BITS
ORDER=np.array([sum(((a>>q)&1)<<GROUPS[0][q] for q in range(3))+
                sum(((b>>q)&1)<<GROUPS[1][q] for q in range(3))
                for a in range(8) for b in range(8)])

def coefficient_matrix(model,x):
    J=model.J;K=J@J;np.fill_diagonal(K,0.)
    M=np.zeros_like(J);A,B=GROUPS;mean=J[np.ix_(A,B)].mean()
    M[np.ix_(A,B)]=mean;M[np.ix_(B,A)]=mean
    iu=np.triu_indices(6,1);rms=lambda a:np.sqrt(np.mean(a[iu]**2))
    if rms(K):K*=rms(J)/rms(K)
    if rms(M):M*=rms(J)/rms(M)
    return x[0]*(J+x[1]*K+x[2]*M)

def cost_phases(model,x):
    A=coefficient_matrix(model,x)
    full=np.zeros(64);local=np.zeros(64);cross=np.zeros(64)
    for i in range(6):
        for j in range(i+1,6):
            value=A[i,j]*Z[:,i]*Z[:,j]
            full+=value
            if any(i in group and j in group for group in GROUPS):local+=value
            else:cross+=value
    return full,local,cross

def shell_phases(row,singles_only=False):
    b1,b2,b3=row;value=np.zeros(64)
    if singles_only:
        # Drop ZZ/ZZZ Pauli terms, preserving the full generator's scalar and Z terms.
        c0=1.5*b1+.75*b2+b3/8
        c1=-b1/2-b2/2-b3/8
        return 2*c0+c1*Z.sum(axis=1)
    for group in GROUPS:
        k=BITS[:,group].sum(axis=1)
        value+=b1*k+b2*k*(k-1)/2+b3*k*(k-1)*(k-2)/6
    return value

def state(model,x,L,output='Z',singles_only=False):
    if model.n!=6:raise ValueError('Nov17 hardware layout currently implements N=6 only')
    if output not in ('Z','X'):raise ValueError('Output must be Z or X')
    x=np.array(x);v=np.zeros(64,complex);v[0]=1
    v=W@v # First stage only needs state preparation, not a general W unitary.
    phase=cost_phases(model,x)[0]
    for layer,row in enumerate(x[3:].reshape(L,3)):
        v*=np.exp(-1j*phase)
        v=W@v
        v*=np.exp(-1j*shell_phases(row,singles_only))
        if output=='Z' or layer<L-1:v=W@v
    return v

def native_correlations(v):return Z.T@((abs(v)**2)[:,None]*Z)

def chi(v):return float(np.sum(native_correlations(v)**2)/6)

def x_means(v):
    indices=np.arange(64)
    return np.array([np.vdot(v,v[indices^(1<<i)]).real for i in range(6)])

def load_saved():
    manifest=json.loads((P/'results/state_manifest.json').read_text())
    out=[]
    for row in manifest['records']:
        saved=json.loads((ROOT/row['source']).read_text())
        out.append((Model(6,row['seed'],saved['b']),np.array(saved['x']),row['L'],row['source']))
    return out
