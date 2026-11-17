"""Replay retained artifacts and audit calibration, schedules, paths and tests.

CPU-only by default. GPU physics checks are available separately through
NOV17_RUN_GPU_TESTS=1 python -m unittest nov17_circuit.tests.test_trajectories.
Historical hardware replays are validated structurally, not rerun here.
"""
import csv,hashlib,json,re,unittest
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ..sequence import P,ROOT,W8,ORDER,load_saved,cost_phases
from ..models.grape import grape_eval,metrics
from ..models.phase_control import bus_detunings,propagate,gate_metrics
from ..models.readout_effects import effects
from ..models.reconstruction import targets
from .prep_noise import evaluate
from .hardware_test import schedule


def audit_artifacts():
    report={}
    jsons=list(P.rglob('*.json'))
    def reject_constant(x):raise ValueError(f'Nonstandard JSON number: {x}')
    for path in jsons:json.loads(path.read_text(),parse_constant=reject_constant)
    report['strict_json_files']=len(jsons)
    # All saved reference-state paths must resolve inside the project.
    for record in json.loads((P/'results/state_manifest.json').read_text())['records']:
        assert (ROOT/record['source']).is_file(),record['source']
    for dest in re.findall(r'\]\(([^)]+)\)',(P/'README.md').read_text()):
        if '://' not in dest:assert (P/dest.split('#')[0]).exists(),dest
    saved={(m.seed,L):(m,x) for m,x,L,_ in load_saved()}
    report['saved_reference_states']=len(saved)
    for name in ('all15_phase_search','shell_phase_search'):
        errors=[]
        for r in json.loads((P/f'pulses/{name}.json').read_text())['records']:
            m,x=saved[(r['seed'],r['L'])]
            if name=='all15_phase_search':
                target=cost_phases(m,x)[0][ORDER].reshape(8,8);delta=bus_detunings().ravel()
            else:
                k=np.array([n.bit_count() for n in range(8)]);b=x[3:].reshape(r['L'],3)[r['layer']]
                target=b[0]*k+b[1]*k*(k-1)/2+b[2]*k*(k-1)*(k-2)/6;delta=2*np.pi*np.arange(8)
            assert np.max(abs(target-np.array(r['phase_table'])))<1e-12
            assert np.max(abs(np.array(r['parameters'])))<=1+1e-12
            assert len(r['parameters'])==2*r['steps']
            v=propagate(np.array(r['parameters']),r['duration_us'],delta)
            score=gate_metrics(np.diag(v[:,0]),np.diag(np.exp(-1j*target.ravel())))
            errors.append(max(abs(score[k]-r['metrics'][k]) for k in score))
        assert max(errors)<1e-12
        report[name]={'pulses':len(errors),'max_metric_error':max(errors)}
    errors=[];cutoff_errors=[]
    for r in json.loads((P/'pulses/coherent_h_search.json').read_text())['records']:
        x=np.array(r['parameters']).reshape(-1,4)
        assert np.max(abs(x[:,:2]))<=.25+1e-12 and np.max(abs(x[:,2:]))<=1+1e-12
        assert abs(r['duration_us']-r['value']/(2*np.pi))<1e-12
        v=grape_eval(x.ravel(),r['value'],r['steps'],r['validation_cutoff'],gradient=False)
        score=gate_metrics(v[:8],W8);err=max(abs(score[k]-r['metrics'][k]) for k in score)
        assert err<1e-12;errors.append(err)
        large=gate_metrics(grape_eval(x.ravel(),r['value'],r['steps'],32,gradient=False)[:8],W8)
        cutoff_errors.append(abs(large['average_gate_fidelity']-score['average_gate_fidelity']))
    report['coherent_H']={'pulses':len(errors),'max_metric_error':max(errors),'cutoff24_to32_differences':cutoff_errors}
    rec=json.loads((P/'pulses/selected_initial_prep.json').read_text())
    v=grape_eval(np.array(rec['parameters']),rec['chi_time'],96,32,gradient=False)[:,0]
    err=abs(abs(v[:8].sum()/np.sqrt(8))**2-rec['noiseless']['preparation_fidelity'])
    noise=evaluate(rec);noise_err=max(abs(noise[k]-rec['noisy'][k]) for k in noise)
    assert err<1e-12 and noise_err<1e-12
    report['selected_preparation']={'coherent_error':float(err),'density_matrix_error':noise_err}
    rec=json.loads((P/'pulses/readout.json').read_text());errs=[]
    for N,key in ((16,'metrics'),(24,'larger_cutoff_metrics')):
        v=grape_eval(np.array(rec['parameters']),rec['value'],len(rec['parameters'])//4,N,gradient=False)
        score=metrics(v);err=max(abs(score[k]-rec[key][k]) for k in score)
        assert err<1e-12;errs.append(err)
    with (P/'pulses/readout_waveform.csv').open() as h:rows=list(csv.DictReader(h))
    x=np.array([[float(row[k]) for k in ('cavity_I_over_chi','cavity_Q_over_chi','ancilla_I_over_chi','ancilla_Q_over_chi')] for row in rows])
    assert np.array_equal(x.ravel(),rec['parameters'])
    assert abs(float(rows[-1]['end_us'])-rec['value']/(2*np.pi))<1e-12
    report['readout_coherent_metric_max_error']=max(errs)
    for scenario in ('coherent','loss','ancilla'):
        with np.load(P/f'calibration/bank_{scenario}.npz',allow_pickle=False) as d:bank=d['effects']
        records=json.loads((P/'pulses/correction_bank.json').read_text())['records']
        errs=[]
        for k,record in enumerate(records,1):
            E,_=effects(record,T1=None if scenario=='coherent' else 600.,ancilla_noise=scenario=='ancilla')
            errs.append(float(np.max(abs(E-bank[k]))))
        err=max(errs);assert err<1e-12
        report[f'readout_effect_error_{scenario}']=err
        report[f'readout_pulses_replayed_{scenario}']=len(records)
        for variant in ('compact','full'):
            stem=P/f'calibration/{variant}_{scenario}'
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as d:
                E=d['effects'];w=d['weights'];f=d['local_f']
                assert np.max(abs(E-(bank if variant=='full' else bank[:2])))<1e-14
                assert np.allclose(d['f'],np.outer(f,f).ravel()) and abs(f.sum()-1)<1e-12
                R=np.einsum('s,ksn,snij->kij',f,w,E)
                residual=max(np.linalg.norm(a-b,2) for a,b in zip(R,targets()[1]))
                meta=json.loads(stem.with_suffix('.json').read_text())
                assert abs(residual-meta['residual'])<1e-12 and abs(abs(w).max()-meta['max_weight'])<1e-12
    replay_count=0
    for path in (P/'results/replays').glob('*.json'):
        for r in json.loads(path.read_text())['records']:
            rows,durations,stages=schedule(r['L'],r['initial_general_H'],10.,r['local_chi_time'])
            assert rows.shape==(len(durations),10) and np.isfinite(rows).all()
            assert abs(durations.sum()-r['duration_us'])<1e-10
            assert len(stages)==len(r['stages'])
            for new,old in zip(stages,r['stages']):
                assert new['kind']==old['kind'] and abs(new['duration_us']-old['duration_us'])<1e-10
            assert not r['postselection']
            assert r['cavity_only_state_fidelity']+1e-6>=r['state_fidelity']
            replay_count+=1
    report['historical_replay_schedules_verified']=replay_count
    tex=(P/'docs/nov17_methods.tex').read_text()
    assert '\\Pe' not in tex and 'P_e' not in tex
    assert (P/'docs/nov17_methods.pdf').read_bytes().startswith(b'%PDF-')
    report['noise_config_sha256']=hashlib.sha256((P/'config/noise_model.json').read_bytes()).hexdigest()
    report['methods_source_sha256']=hashlib.sha256((P/'docs/nov17_methods.tex').read_bytes()).hexdigest()
    report['methods_pdf_sha256']=hashlib.sha256((P/'docs/nov17_methods.pdf').read_bytes()).hexdigest()
    return report


def main():
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(P/'tests'),top_level_dir=str(P.parent)))
    if not result.wasSuccessful():raise SystemExit(1)
    report=audit_artifacts()
    report['unit_tests']={'run':result.testsRun,'passed':result.testsRun-len(result.skipped),'skipped':len(result.skipped)}
    report['timestamp_utc']=datetime.now(timezone.utc).isoformat()
    report['scope']='Retained artifacts replayed; historical full hardware runs not repeated. Optional GPU physics tests require NOV17_RUN_GPU_TESTS=1.'
    (P/'results/folder_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
