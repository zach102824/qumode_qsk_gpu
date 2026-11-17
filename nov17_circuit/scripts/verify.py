"""Verify Nov17 ordering, output frames, all 15 phases, and retained shell terms."""
from ..sequence import *

def check(model,x,L,label):
    original=model.state(x,L);zstate=state(model,x,L,'Z');xstate=state(model,x,L,'X')
    full,local,cross=cost_phases(model,x)
    phase_error=float(np.max(abs(full-x[0]*(model.diags[0]+x[1]*model.diags[1]+x[2]*model.diags[2]))))
    split_error=float(np.max(abs(np.exp(-1j*full)-np.exp(-1j*local)*np.exp(-1j*cross))))
    shell_error=max(float(np.max(abs(shell_phases(row)-model.shell@row))) for row in x[3:].reshape(L,3))
    zerror=float(np.max(abs(original-zstate)))
    xerror=float(np.max(abs(xstate-W@original)))
    # Independently execute the old physical XX-cost / Z-shell circuit from vacuum.
    old=np.zeros(64,complex);old[0]=1
    for row in x[3:].reshape(L,3):
        old=W@old;old*=np.exp(-1j*full);old=W@old;old*=np.exp(-1j*shell_phases(row))
    old_error=float(np.max(abs(old-xstate)))
    rotated_energy=float(np.vdot(xstate,W@model.apply_H(W@xstate)).real)
    energy_error=abs(rotated_energy-model.energy(original))
    magnetization_error=float(np.max(abs((abs(xstate)**2)@Z-x_means(original))))
    chi_error=abs(chi(zstate)-chi(original))
    simplified=state(model,x,L,'Z',singles_only=True)
    simplified_fidelity=float(abs(np.vdot(zstate,simplified))**2)
    # Diagonal shell phases do not change an immediate native histogram. This
    # identity alone does not authorize eliding a noisy preparation channel.
    without_last_shell=xstate*np.exp(1j*shell_phases(x[3:].reshape(L,3)[-1]))
    x_histogram_shell_elision_error=float(np.max(abs(abs(without_last_shell)**2-abs(xstate)**2)))
    assert max(phase_error,split_error,shell_error,zerror,xerror,old_error,energy_error,magnetization_error,chi_error)<1e-9
    return dict(label=label,seed=model.seed,L=L,phase_error=phase_error,split_error=split_error,
                shell_error=shell_error,z_state_error=zerror,x_frame_error=xerror,
                same_as_old_physical_state_error=old_error,energy_error=energy_error,
                magnetization_error=magnetization_error,chi_error=chi_error,
                singles_only_fidelity=simplified_fidelity,original_chi=chi(original),
                x_histogram_shell_elision_error=x_histogram_shell_elision_error,
                Z_output_H_stages=2*L+1,X_output_H_stages=2*L)

def main():
    rows=[check(m,x,L,'saved') for m,x,L,_ in load_saved()]
    rng=np.random.default_rng(171117)
    for L in range(1,7):
        for k in range(20):
            model=Model(6,831106000+k%10,.4)
            x=rng.uniform(-np.pi,np.pi,3+3*L);x[1:3]=rng.uniform(-1,1,2)
            rows.append(check(model,x,L,'random'))
    errors={key:max(r[key] for r in rows) for key in rows[0] if key.endswith('error')}
    out=dict(states=len(rows),saved_states=20,random_states=120,max_errors=errors,
        min_saved_singles_only_fidelity=min(r['singles_only_fidelity'] for r in rows if r['label']=='saved'),
        min_random_singles_only_fidelity=min(r['singles_only_fidelity'] for r in rows if r['label']=='random'),records=rows)
    (P/'results/verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))

if __name__=='__main__':main()
