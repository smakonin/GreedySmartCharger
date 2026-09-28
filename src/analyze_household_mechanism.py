"""Same-state certificate comparisons on the preselected diagnostic sample.

This reads executed trajectories; it does not rerun or tune a controller.
Admission probes on an observed trajectory are not counterfactual trajectories.
"""
import argparse,csv,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from causal_replay import plan_pair,exposure
from household_meter import BatteryRequest,local_command,tariff_bands
from household_liability_v2 import ExecutedLiability
from physical_simulator import BatterySpec


def read_trace(path):
    with np.load(path) as f:
        layout=json.loads(str(f['layout']))
        out={table:{name:f[table][:,i] for i,name in enumerate(layout[table]['numeric'])}
             for table in ('trace','decisions','intervals')}
        labels=layout['decisions']['categorical']['request.source']
        out['sources']=[labels[int(i)] for i in f['decisions__request.source']]
    return out


def run(base):
    assert json.loads((base/'execution_completion.json').read_text())['all_cases_complete']
    design=json.loads((base/'diagnostic_matrix.json').read_text())
    b=BatterySpec();rows=[];max_repair_residual=0.
    for case in design['cases']:
        for tariff in ('flat','tou','signed'):
            model=ExecutedLiability(tariff_bands(30,tariff),b)
            for trajectory in ('MPC','SCR-repair'):
                tables=read_trace(base/'cases'/case['case_id']/(tariff+'__primary__'+trajectory+'.npz'))
                d=tables['decisions'];iv=tables['intervals'];tr=tables['trace']
                for k in design['runtime']['decision_indices']:
                    t=float(d['time'][k]);s=float(d['soc'][k]);z=float(d['shadow_soc'][k]);regret=float(d['regret_before'][k])
                    q=BatteryRequest(float(d['request.kw'][k]),tables['sources'][k],bool(d['request.local_greedy'][k]))
                    load=float(iv['load_kwh'][k])*2;pv=float(iv['pv_kwh'][k])*2;price=float(iv['price'][k])
                    cmd=local_command(q,load,pv,price,s,b);shadow=local_command(BatteryRequest(local_greedy=True),load,pv,price,z,b)
                    original=plan_pair(cmd,shadow,load,pv,s,z,b,1800.)
                    commitment=original[0].duration
                    plan=plan_pair(cmd,shadow,load,pv,s,z,b,commitment)
                    peaks={cert:exposure(plan,model,cert,t,regret,load,price) for cert in ('global','copy','repair')}
                    assert peaks['repair']<=peaks['copy']+1e-8 and peaks['copy']<=peaks['global']+1e-8
                    index=int(np.searchsorted(tr['start'],t));assert tr['start'][index]==t
                    action_change=max(abs(getattr(plan[0].power,key)-tr['power.'+key][index]) for key in asdict(cmd))
                    if trajectory=='SCR-repair':
                        residual=abs(peaks['repair']-tr['proposed_peak'][index]);max_repair_residual=max(max_repair_residual,residual)
                        assert residual<=1e-8
                        assert bool(tr['fallback'][index])==(peaks['repair']>1.)
                    for budget in (0.,1.):
                        admit={cert:peaks[cert]<=budget for cert in peaks}
                        rows.append(dict(case_id=case['case_id'],customer=case['customer'],tariff=tariff,trajectory=trajectory,
                            decision_index=k,time=t,soc=s,shadow_soc=z,regret_before=regret,commitment_seconds=commitment,budget=budget,
                            **{cert+'_peak':v for cert,v in peaks.items()},**{cert+'_admitted':admit[cert] for cert in peaks},
                            admission_near_roundoff=any(abs(v-budget)<=1e-8 for v in peaks.values()),
                            copy_new_admission=admit['copy'] and not admit['global'],repair_new_admission=admit['repair'] and not admit['copy'],
                            actual_first_segment_differs_from_nominal=bool(action_change>1e-9),
                            max_actual_nominal_power_difference_kw=float(action_change)))
        print(case['case_id'],len(rows),flush=True)
    out=base/'mechanism';out.mkdir(exist_ok=False)
    with (out/'same_state_points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for tariff in ('flat','tou','signed'):
        for trajectory in ('MPC','SCR-repair'):
            for budget in (0.,1.):
                rr=[r for r in rows if (r['tariff'],r['trajectory'],r['budget'])==(tariff,trajectory,budget)]
                assert len(rr)==400
                summary.append(dict(tariff=tariff,trajectory=trajectory,budget=budget,states=len(rr),households=10,windows=20,
                    global_admitted=sum(r['global_admitted'] for r in rr),copy_admitted=sum(r['copy_admitted'] for r in rr),repair_admitted=sum(r['repair_admitted'] for r in rr),
                    copy_new_admissions=sum(r['copy_new_admission'] for r in rr),repair_new_admissions=sum(r['repair_new_admission'] for r in rr),
                    strict_copy_tightenings=sum(r['global_peak']-r['copy_peak']>1e-8 for r in rr),
                    strict_repair_tightenings=sum(r['copy_peak']-r['repair_peak']>1e-8 for r in rr),
                    admission_near_roundoff=sum(r['admission_near_roundoff'] for r in rr),
                    observed_action_changes=sum(r['actual_first_segment_differs_from_nominal'] for r in rr)))
    result=dict(rows=len(rows),unique_sampled_states=len(rows)//2,max_repair_reproduction_residual=max_repair_residual,summary=summary,
        scope='Exploratory same-state probes at the 20 predeclared hash-selected decision indices in each of 20 predeclared windows, on primary MPC and SCR-repair trajectories. Both budgets are evaluated on these observed states, not on trajectories generated under each queried certificate/budget. Admission counts do not imply achievable full-episode cost. Action changes describe the observed first physical segment only, and are not caused anew by a queried budget.',
        analysis_specification='HOUSEHOLD_MECHANISM_ANALYSIS_V1.md')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();result=run(a.package)
    print(json.dumps({k:result[k] for k in ('rows','unique_sampled_states','max_repair_reproduction_residual')}))
