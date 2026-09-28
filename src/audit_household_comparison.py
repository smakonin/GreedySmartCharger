"""Verify stored matrix completeness, fingerprints, aliases and decoded flows."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from household_artifacts import sha

def audit(base):
    started=time.perf_counter();root=Path(__file__).resolve().parents[1]
    complete=json.loads((base/'execution_completion.json').read_text());assert complete['all_cases_complete']
    rounding=json.loads((base/'event_rounding_audit.json').read_text())
    assert rounding['passed'] and rounding['unique_traces_checked']==54069
    freeze=json.loads((base/'execution_freeze.json').read_text());checked=0
    for name,digest in freeze['source_sha256'].items():
        assert sha(base/'release_sources'/name)==digest,name
        assert sha(root/name)==digest,name
        checked+=1
    for name,digest in freeze['design_sha256'].items():assert sha(base/name)==digest,name;checked+=1
    configs={r['config_id']:r for r in json.loads((base/'configurations.json').read_text())}
    cases=json.loads((base/'cases.json').read_text());diagnostics=json.loads((base/'diagnostic_matrix.json').read_text());selected={r['case_id'] for r in diagnostics['cases']}
    matrix=list(csv.DictReader((base/'execution_matrix.csv').open()));rows=[];digests={};residual=0.;segments=decisions=0;selected_files=0
    def close(a,b,tolerance=1e-7):
        nonlocal residual
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        residual=max(residual,err)
        assert err<=tolerance,(err,tolerance)
    for case in cases:
        cid=case['case_id'];folder=base/'cases'/cid;cr=json.loads((folder/'results.json').read_text())['rows'];rows.extend(cr);assert len(cr)==255
        by_id={r['row_id']:r for r in cr}
        with np.load(folder/'inputs.npz') as f:inputload=f['load'][1344:];inputpv=f['pv'][1344:]
        for r in cr:
            if r['status']!='ok':assert r['config_id']=='capacity_5__OMG';continue
            c=configs[r['config_id']];m=r['metrics']
            if r['reused_physical_trajectory']:
                parent=by_id[r['execution_id']]
                assert r['trace_sha256']==parent['trace_sha256'] and r['trace_file']==parent['trace_file']
                for key in ('cost','final_soc','charge_ac_kwh','discharge_ac_kwh','efc','peak_kw'):assert m[key]==parent['metrics'][key]
                close(m['objective'],m['cost']-c['terminal_value']*(m['final_soc']-m['initial_soc']))
                continue
            path=folder/r['trace_file'];digest=sha(path);assert digest==r['trace_sha256'],str(path);digests[str(path.relative_to(base))]=digest
            if c['certificate']:
                assert m['max_prefix_exposure']<=c['budget']+1e-8 and m['terminal_difference']<=c['budget']+1e-8
            if cid not in selected:continue
            selected_files+=1
            with np.load(path) as f:
                layout=json.loads(str(f['layout']));tables={name:{key:f[name][:,i] for i,key in enumerate(layout[name]['numeric'])} for name in ('trace','intervals','decisions')}
            tr=tables['trace'];iv=tables['intervals'];de=tables['decisions'];n=len(tr['start']);segments+=n;decisions+=len(de['time'])
            assert len(iv['end_soc'])==1440 and n>=1440
            dt=(tr['end']-tr['start'])/3600;assert np.all(dt>=0)
            # Equal binary64 timestamps are independently bounded in every trace
            # by event_rounding_audit.json; keep their stored energy and costs.
            close(tr['start'][0],0);close(tr['end'][-1],30*86400);close(tr['start'][1:],tr['end'][:-1])
            idx=np.floor(tr['start']/1800).astype(int);load=inputload[idx]*2;pv=inputpv[idx]*2
            close(iv['load_kwh'],inputload);close(iv['pv_kwh'],inputpv)
            for prefix,state,before,grid,cost in [('power.','soc','soc_before','grid_kwh','cost'),('shadow_power.','shadow_soc','shadow_soc_before','shadow_grid_kwh','shadow_cost')]:
                direct=tr[prefix+'direct'];pvc=tr[prefix+'pv_charge'];dis=tr[prefix+'discharge'];gc=tr[prefix+'grid_charge'];charge=pvc+gc
                assert np.min(np.concatenate([direct,pvc,dis,gc]))>=-1e-8
                assert np.all(charge<=c['charge_kw']+1e-8) and np.all(dis<=c['discharge_kw']+1e-8)
                assert np.all((charge<=1e-8)|(dis<=1e-8));assert np.all(direct+dis<=load+1e-8) and np.all(direct+pvc<=pv+1e-8)
                close(tr[state]-tr[before],(.95*charge-dis/.95)*dt);close(tr[before][1:],tr[state][:-1]);close(tr[before][0],c['initial_fraction']*c['capacity'])
                assert np.all(tr[state]>=-1e-8) and np.all(tr[state]<=c['capacity']+1e-8)
                close(tr[grid],(load-direct-dis+gc)*dt);close(tr[cost],tr[grid]*iv['price'][idx])
            close(np.cumsum(tr['cost']-tr['shadow_cost']),tr['regret'])
            close(tr['cost'].sum(),m['cost']);close(tr['soc'][-1],m['final_soc']);close(tr['shadow_cost'].sum(),m['shadow_cost'])
            charge=(tr['power.pv_charge']+tr['power.grid_charge'])*dt;dis=tr['power.discharge']*dt
            close(charge.sum(),m['charge_ac_kwh']);close(dis.sum(),m['discharge_ac_kwh']);close((.95*charge.sum()+dis.sum()/.95)/(2*c['capacity']),m['efc'])
            imports=np.bincount(idx,weights=tr['grid_kwh'],minlength=1440);close(imports,iv['import_kwh']);close(imports.max()*2,m['peak_kw'])
            assert np.all(de['latest_meter_end']<=de['time']) and np.all(de['latest_meter_ready_at']<=de['time'])
            di=(de['time']/1800).astype(int);close(de['soc'],iv['start_soc'][di]);close(de['shadow_soc'],iv['start_shadow_soc'][di])
            if c['certificate']:assert np.max(tr['segment_peak'])<=c['budget']+1e-8 and np.max(tr['regret']+tr['liability'])<=c['budget']+1e-8
        print(json.dumps(dict(audited_cases=cases.index(case)+1,total_cases=len(cases),hashed_traces=len(digests),decoded_segments=segments)),flush=True)
    by_id={r['row_id']:r for r in rows};assert len(by_id)==len(rows)==len(matrix)==68595
    for e in matrix:
        r=by_id[e['row_id']]
        for key in ('case_id','config_id','execution_id','tariff'):assert r[key]==e[key]
        assert r['reused_physical_trajectory']==(e['reused_physical_trajectory']=='True')
    expected=sum(r['status']=='expected_parameter_domain_failure' for r in rows);assert expected==807
    assert len(digests)==54876-807
    result=dict(passed=True,logical_rows=len(rows),unique_successful_traces_hashed=len(digests),expected_domain_failures=expected,frozen_files_checked=checked,
        decoded_hash_selected_households=len(diagnostics['households']),decoded_windows=len(selected),decoded_unique_trajectories=selected_files,
        decoded_physical_segments=segments,decoded_decisions=decisions,max_decoded_arithmetic_residual=residual,elapsed_seconds=time.perf_counter()-started,
        timestamp_collapsed_events_all_traces=rounding['timestamp_collapsed_events'],event_rounding_audit_passed=True,
        scope='Every physical trace hashed and matrix row reconciled; every unique trace from the predeclared 10-household/20-window diagnostic subset decoded and independently recomputed. Runtime execution separately audited every segment before saving.')
    (base/'stored_trace_hashes.json').write_text(json.dumps(digests,indent=2)+'\n');(base/'stored_file_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.package),indent=2))
