"""Bound timestamp-collapsed numerical events without altering saved trajectories."""
import argparse,json,math,time
from pathlib import Path
import numpy as np


def audit(base):
    assert json.loads((base/'execution_completion.json').read_text())['all_cases_complete']
    cases=json.loads((base/'cases.json').read_text());records=[];traces=0;events=0;started=time.perf_counter()
    eps=np.finfo(float).eps;tiny=np.nextafter(0.,1.)
    maxima=dict(grid_kwh=0.,cost_cu=0.,state_change_kwh=0.,sum_abs_cost_per_trace_cu=0.,duration_upper_seconds=0.)
    for case in cases:
        folder=base/'cases'/case['case_id'];rows=json.loads((folder/'results.json').read_text())['rows']
        for row in rows:
            if row['status']!='ok' or row['reused_physical_trajectory']:continue
            path=folder/row['trace_file'];traces+=1
            with np.load(path) as f:
                layout=json.loads(str(f['layout']));columns=layout['trace']['numeric'];values=f['trace'];tr={k:values[:,i] for i,k in enumerate(columns)}
                elapsed=tr['end']-tr['start'];assert np.all(elapsed>=0),(str(path),'negative duration')
                zero=np.flatnonzero(elapsed==0)
                if not len(zero):continue
                interval_values=f['intervals'];iv={k:interval_values[:,i] for i,k in enumerate(layout['intervals']['numeric'])}
                branches=layout['trace']['categorical']['branch']
                labels=f['trace__branch']
                trace_cost=0.
                for j in zero:
                    t=tr['start'][j];assert t>0
                    # For positive dt, fl(t+dt)==t implies dt <= half an upward ULP.
                    duration=.5*np.spacing(t)*(1+8*eps);hours=duration/3600
                    k=int(t//1800);load=2*iv['load_kwh'][k];price=iv['price'][k]
                    item=dict(case_id=case['case_id'],trace_file=row['trace_file'],event_index=int(j),time=float(t),
                        duration_upper_seconds=float(duration),branch=None if labels[j]<0 else branches[int(labels[j])])
                    for prefix,state,before,gkey,ckey in [('power.','soc','soc_before','grid_kwh','cost'),
                                                          ('shadow_power.','shadow_soc','shadow_soc_before','shadow_grid_kwh','shadow_cost')]:
                        direct=tr[prefix+'direct'][j];dis=tr[prefix+'discharge'][j];gc=tr[prefix+'grid_charge'][j];pvc=tr[prefix+'pv_charge'][j]
                        grid_rate=load-direct-dis+gc
                        grid_bound=(abs(grid_rate)+64*eps*(abs(load)+abs(direct)+abs(dis)+abs(gc)))*hours+tiny
                        assert abs(tr[gkey][j])<=grid_bound,(str(path),int(j),gkey,float(tr[gkey][j]),grid_bound)
                        cost_bound=abs(price)*grid_bound*(1+64*eps)+tiny
                        assert abs(tr[ckey][j])<=cost_bound,(str(path),int(j),ckey,float(tr[ckey][j]),cost_bound)
                        # Preserve the executor's existing 1e-12 boundary snap and
                        # 1e-11 repair-synchronization snap; no new execution slack.
                        state_bound=(.95*(pvc+gc)+dis/.95)*hours+1.1e-11
                        change=abs(tr[state][j]-tr[before][j]);assert change<=state_bound,(str(path),int(j),state,change,state_bound)
                        maxima['grid_kwh']=max(maxima['grid_kwh'],float(abs(tr[gkey][j])))
                        maxima['cost_cu']=max(maxima['cost_cu'],float(abs(tr[ckey][j])))
                        maxima['state_change_kwh']=max(maxima['state_change_kwh'],float(change))
                        item[gkey]=float(tr[gkey][j]);item[ckey]=float(tr[ckey][j]);item[state+'_change_kwh']=float(change)
                        trace_cost+=abs(tr[ckey][j])
                    records.append(item);events+=1
                    maxima['duration_upper_seconds']=max(maxima['duration_upper_seconds'],float(duration))
                maxima['sum_abs_cost_per_trace_cu']=max(maxima['sum_abs_cost_per_trace_cu'],float(trace_cost))
        print(json.dumps(dict(cases_checked=cases.index(case)+1,traces_checked=traces,collapsed_events=events)),flush=True)
    assert traces==54069
    result=dict(passed=True,unique_traces_checked=traces,timestamp_collapsed_events=events,affected_traces=len({(r['case_id'],r['trace_file']) for r in records}),
        maximum=maxima,events=records,elapsed_seconds=time.perf_counter()-started,
        scope='All unique successful traces checked for nonnegative stored elapsed time. Equal endpoints are retained only when stored energy/cost fits a positive duration below half a timestamp ULP and state changes fit the existing execution snap bounds. No controller, stored trace, metric or numerical execution tolerance was changed. Raw duration is not stored separately; bounds on collapsed events are inferred from timestamp rounding and independently checked against flows.')
    (base/'event_rounding_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='events'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.package),indent=2))
