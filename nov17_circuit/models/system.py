"""Authoritative two-cavity/three-transmon Hamiltonian and Lindblad operators.

Time is in microseconds; h=H/hbar has units rad/us. Drive values are MHz
(cycles/us), so h(t)=h0+sum_m u_m(t)*hc[m] includes the factor 2*pi.
Tensor order: bus, local_A, cavity_A, local_B, cavity_B.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=json.loads((ROOT/'config/noise_model.json').read_text())
CONTROL_NAMES=('cavity_A_I','cavity_A_Q','cavity_B_I','cavity_B_Q',
               'local_A_I','local_A_Q','local_B_I','local_B_Q','bus_I','bus_Q')

def parameters(override=None):
    if override and set(override)-set(DEFAULT_CONFIG):
        raise ValueError(f'Unknown model parameters: {sorted(set(override)-set(DEFAULT_CONFIG))}')
    p={**DEFAULT_CONFIG,**(override or {})}
    if p['transmon_levels']!=2:raise NotImplementedError('Only g/e transmon truncation is implemented')
    if p['thermal_occupation']!=0:raise NotImplementedError('Thermal excitation is not included in this baseline')
    if p['logical_levels_per_cavity']!=8:raise NotImplementedError('The encoding uses eight levels per cavity')
    for key,value in p.items():
        if key.endswith('_MHz') and not np.isfinite(value):raise ValueError(f'{key} must be finite')
    for name in ('cavity_A','cavity_B','local_A','local_B','bus'):
        if not p[f'T1_{name}_us']>0:raise ValueError('T1 must be positive (infinity disables relaxation)')
    for name in ('local_A','local_B','bus'):
        if not 0<p[f'T2_{name}_us']<=2*p[f'T1_{name}_us']:
            raise ValueError('Require 0 < T2 <= 2*T1 for nonnegative pure dephasing')
    for name in ('A','B'):
        rate=p[f'cavity_{name}_number_dephasing_per_us']
        if not np.isfinite(rate) or rate<0:raise ValueError('Cavity dephasing rate must be finite and nonnegative')
    return p

def validate_cutoff(N):
    if isinstance(N,(bool,np.bool_)) or not isinstance(N,(int,np.integer)) or N<2:
        raise ValueError('Cavity cutoff must be an integer of at least two')

def kron(*items):
    result=items[0]
    for item in items[1:]:result=np.kron(result,item)
    return result

def hamiltonians(N,override=None):
    """Return h0 and ten control matrices; do not add the logical SK Hamiltonian."""
    validate_cutoff(N)
    p=parameters(override);a=np.diag(np.sqrt(np.arange(1,N)),1);n=a.T@a
    q=np.eye(2);I=np.eye(N);e=np.diag([0,1]);X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    h=p['detuning_cavity_A_MHz']*kron(q,q,n,q,I)+p['detuning_cavity_B_MHz']*kron(q,q,I,q,n)
    h+=p['detuning_local_A_MHz']*kron(q,e,I,q,I)+p['detuning_local_B_MHz']*kron(q,q,I,e,I)
    h+=p['detuning_bus_MHz']*kron(e,q,I,q,I)
    h-=p['chi_local_A_MHz']*kron(q,e,n,q,I)+p['chi_local_B_MHz']*kron(q,q,I,e,n)
    h-=p['chi_bus_A_MHz']*kron(e,q,n,q,I)+p['chi_bus_B_MHz']*kron(e,q,I,q,n)
    if p['Kerr_A_MHz']:h+=p['Kerr_A_MHz']/2*kron(q,q,n@(n-I),q,I)
    if p['Kerr_B_MHz']:h+=p['Kerr_B_MHz']/2*kron(q,q,I,q,n@(n-I))
    if p['cross_Kerr_MHz']:h+=p['cross_Kerr_MHz']*kron(q,q,n,q,n)
    if p['exchange_MHz']:h+=p['exchange_MHz']*(kron(q,q,a.T,q,a)+kron(q,q,a,q,a.T))
    aa=kron(q,q,a,q,I);ab=kron(q,q,I,q,a)
    hc=np.array([aa+aa.T,1j*(aa.T-aa),ab+ab.T,1j*(ab.T-ab),
        kron(q,X,I,q,I),kron(q,Y,I,q,I),kron(q,q,I,X,I),kron(q,q,I,Y,I),
        kron(X,q,I,q,I),kron(Y,q,I,q,I)])*2*np.pi
    return h*2*np.pi,hc

def named_collapse_operators(N,override=None):
    """Return (label,L) pairs with L in us^(-1/2); gamma_phi uses Pe convention."""
    validate_cutoff(N)
    p=parameters(override);a=np.diag(np.sqrt(np.arange(1,N)),1);q=np.eye(2);I=np.eye(N)
    e=np.diag([0,1]);sm=np.array([[0,1],[0,0]])
    ops=[('cavity_A_loss',kron(q,q,a,q,I)/np.sqrt(p['T1_cavity_A_us'])),
         ('cavity_B_loss',kron(q,q,I,q,a)/np.sqrt(p['T1_cavity_B_us']))]
    for name,slot in (('bus',0),('local_A',1),('local_B',3)):
        factors=[q,q,I,q,I];factors[slot]=sm
        ops.append((name+'_relaxation',kron(*factors)/np.sqrt(p[f'T1_{name}_us'])))
        gamma_phi=1/p[f'T2_{name}_us']-1/(2*p[f'T1_{name}_us'])
        factors[slot]=e
        ops.append((name+'_dephasing',kron(*factors)*np.sqrt(2*gamma_phi)))
    for name in ('A','B'):
        rate=p[f'cavity_{name}_number_dephasing_per_us']
        if rate:
            factors=[q,q,I,q,I];factors[2 if name=='A' else 4]=a.T@a
            ops.append((f'cavity_{name}_number_dephasing',np.sqrt(rate)*kron(*factors)))
    return ops

def system_hamiltonian(N,quadratures,override=None):
    """h(t)=H(t)/hbar in rad/us, with quadratures ordered as CONTROL_NAMES."""
    u=np.asarray(quadratures)
    if u.shape!=(10,) or not np.isrealobj(u) or not np.isfinite(u).all():
        raise ValueError('Expected ten finite real drive quadratures')
    h0,hc=hamiltonians(N,override)
    return h0+np.einsum('m,mij->ij',u,hc)

def lindblad_rhs(rho,h,collapse_operators):
    """Matrix-free density-matrix RHS; h=H/hbar. No d^2 by d^2 Liouvillian."""
    result=-1j*(h@rho-rho@h)
    for op in collapse_operators:
        L=op[1] if isinstance(op,tuple) else op
        rate=L.conj().T@L
        result+=L@rho@L.conj().T-.5*(rate@rho+rho@rate)
    return result
