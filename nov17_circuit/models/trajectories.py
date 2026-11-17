"""Noisy whole-circuit average fidelity using quantum-jump Haar trajectories.

Includes both cavities, both local transmons and the bus, with no resets or
postselection. A finite jump timestep is replayed for convergence checks.
"""
import sys,time,os
from pathlib import Path
import numpy as np
from .system import hamiltonians as full_hamiltonians, named_collapse_operators, DEFAULT_CONFIG, parameters as model_parameters, validate_cutoff
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!='torch':raise
    # Optional sibling environment used by this workstation; no fixed username.
    site=Path(os.environ.get('NOV17_TORCH_SITE',str(Path(__file__).resolve().parents[3]/'pytorch_gpu'/'.venv'/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages')))
    if not (site/'torch').is_dir():
        raise ImportError('Install PyTorch in the active environment or set NOV17_TORCH_SITE') from exc
    sys.path.append(str(site))
    import torch

def validate_run(rows,durations,N,samples,dt_max,initial=None,desired=None,tar=None,resets=None):
    """Reject invalid schedules before allocating large Hamiltonians or CUDA memory."""
    validate_cutoff(N)
    if N<8:raise ValueError('Encoded trajectory replay requires cutoff >= 8')
    rows=np.asarray(rows);durations=np.asarray(durations)
    if rows.ndim!=2 or rows.shape[1]!=10 or len(rows)==0 or not np.isrealobj(rows) or not np.isfinite(rows).all():
        raise ValueError('Expected a nonempty (segments,10) finite real waveform')
    if durations.shape!=(len(rows),) or not np.isrealobj(durations) or not np.isfinite(durations).all() or np.any(durations<=0):
        raise ValueError('One finite positive duration is required for every segment')
    if isinstance(samples,(bool,np.bool_)) or not isinstance(samples,(int,np.integer)) or samples<1:
        raise ValueError('samples must be a positive integer')
    if not np.isfinite(dt_max) or dt_max<=0:raise ValueError('dt_max must be finite and positive')
    for label,v in (('initial',initial),('desired',desired)):
        if v is not None:
            v=np.asarray(v)
            if v.shape!=(64,) or not np.isfinite(v).all() or not np.isclose(np.vdot(v,v).real,1,rtol=0,atol=1e-7):
                raise ValueError(f'{label} must be a normalized finite 64-component state')
    if desired is None:
        U=np.asarray(tar['U']) if tar is not None and 'U' in tar else np.array([])
        if U.shape!=(64,64) or not np.isfinite(U).all() or not np.allclose(U.conj().T@U,np.eye(64),atol=1e-7,rtol=0):
            raise ValueError('tar[U] must be a finite 64-by-64 unitary')
    for k,axes in (resets or {}).items():
        if not isinstance(k,(int,np.integer)) or not 0<=k<len(rows) or any(axis not in (0,1,3) for axis in axes):
            raise ValueError('Reset boundaries must index segments and transmon axes 0,1,3')
    return rows,durations

def jumps(N,device,parameters=None):
    named=named_collapse_operators(N,parameters)
    names=[name for name,_ in named]
    ops=np.array([op for _,op in named])
    weights=np.sum(abs(ops)**2,axis=1)
    return (torch.tensor(ops,device=device,dtype=torch.complex64),
            torch.tensor(weights,device=device,dtype=torch.float32),names)

@torch.no_grad()
def simulate(rows,durations,tar,N=DEFAULT_CONFIG['default_cavity_cutoff'],samples=1024,dt_max=DEFAULT_CONFIG['default_jump_step_us'],noise=True,seed=161119,device=1,resets=None,initial=None,desired=None,parameters=None):
    rows,durations=validate_run(rows,durations,N,samples,dt_max,initial,desired,tar,resets)
    config=model_parameters(parameters)
    dev=torch.device('cpu' if device=='cpu' else f'cuda:{device}')
    if dev.type=='cuda':
        torch.cuda.set_device(dev)
        torch.cuda.set_per_process_memory_fraction(.45,dev)
    torch.manual_seed(seed)
    h0,hc=full_hamiltonians(N,config)
    h0=torch.tensor(h0,device=dev,dtype=torch.complex64);hc=torch.tensor(hc,device=dev,dtype=torch.complex64)
    L,rate,jump_names=jumps(N,dev,parameters)
    if noise:h0=h0-.5j*torch.diag(rate.sum(dim=0)).to(torch.complex64)
    index=torch.tensor([a*(2*N)+b for a in range(8) for b in range(8)],device=dev)
    if initial is None:
        alpha=torch.randn((64,samples),device=dev,dtype=torch.complex64)
        alpha/=torch.linalg.vector_norm(alpha,dim=0)
    else:alpha=torch.tensor(initial,device=dev,dtype=torch.complex64)[:,None].repeat(1,samples)
    targetvec=torch.tensor(tar['U'],device=dev,dtype=torch.complex64)@alpha if desired is None else torch.tensor(desired,device=dev,dtype=torch.complex64)[:,None].repeat(1,samples)
    psi=torch.zeros((len(h0),samples),device=dev,dtype=torch.complex64);psi[index]=alpha
    reset_indices={}
    if resets:
        grid=np.indices((2,2,N,2,N))
        for axis in (0,1,3):
            reset_indices[axis]=[torch.tensor(np.flatnonzero(grid[axis].ravel()==value),device=dev) for value in (0,1)]
    alljumps=0;start=time.monotonic();jump_hist=torch.zeros(len(L),device=dev,dtype=torch.long)
    for k,(row,T) in enumerate(zip(rows,durations)):
        count=int(np.ceil(T/dt_max));dt=T/count
        H=h0+torch.einsum('c,cij->ij',torch.tensor(row,device=dev,dtype=torch.complex64),hc)
        U=torch.linalg.matrix_exp(-1j*dt*H)
        for _ in range(count):
            psi=U@psi
            norm2=torch.sum(abs(psi)**2,dim=0).real
            selected=torch.where(torch.rand(samples,device=dev)>norm2.clamp(0,1))[0] if noise else torch.empty(0,device=dev,dtype=torch.long)
            psi/=norm2.clamp_min(1e-20).sqrt()[None]
            if len(selected):
                weights=rate@abs(psi[:,selected])**2
                channel=torch.multinomial(weights.T,1).ravel()
                jump_hist+=torch.bincount(channel,minlength=len(L))
                for j in range(len(L)):
                    chosen=selected[channel==j]
                    if len(chosen):
                        v=L[j]@psi[:,chosen]
                        psi[:,chosen]=v/torch.linalg.vector_norm(v,dim=0)[None]
                alljumps+=len(selected)
        for axis in (resets or {}).get(k,()):
            ground,excited=reset_indices[axis]
            probability=torch.sum(abs(psi[excited])**2,dim=0)
            choose_e=torch.rand(samples,device=dev)<probability
            out=torch.zeros_like(psi)
            for value,indices in [(False,ground),(True,excited)]:
                cols=torch.where(choose_e==value)[0]
                if len(cols):
                    v=psi[indices[:,None],cols[None,:]]
                    out[ground[:,None],cols[None,:]]=v/torch.linalg.vector_norm(v,dim=0)[None]
            psi=out
        if k%150==0:print('TRAJECTORY',k,'/',len(rows),'jumps',alljumps,'elapsed',round(time.monotonic()-start,1),flush=True)
    fidelities=abs(torch.sum(targetvec.conj()*psi[index],dim=0))**2
    branches=psi.reshape(2,2,N,2,N,samples)[:,:,:8,:,:8].permute(0,1,3,2,4,5).reshape(8,64,samples)
    cavity_fidelity=torch.sum(abs(torch.sum(targetvec.conj()[None]*branches,dim=1))**2,dim=0)
    cavity_f=cavity_fidelity.cpu().numpy()
    f=fidelities.cpu().numpy()
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    return dict(average_gate_fidelity=float(f.mean()),standard_error=float(f.std(ddof=1)/np.sqrt(samples)) if samples>1 else 0.,
                cavity_only_average_fidelity=float(cavity_f.mean()),
                cavity_only_standard_error=float(cavity_f.std(ddof=1)/np.sqrt(samples)) if samples>1 else 0.,
                cavity_only_note='Trace all three transmons; ideal end reset would still be required for reuse.',
                samples=samples,dt_max_us=dt_max,cutoff=N,seconds=time.monotonic()-start,
                jump_events=alljumps,final_required='all three transmons g; both cavities in logical space',
                jump_counts=dict(zip(jump_names,jump_hist.cpu().tolist())),
                postselection=False,noise=noise,seed=seed,
                fidelity_kind='state fidelity' if initial is not None else 'Haar-average gate fidelity estimate',
                model_parameters=config,device=str(dev))
