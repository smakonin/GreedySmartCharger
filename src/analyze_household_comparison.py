"""Implement the frozen household estimands; never alter controller parameters."""
import argparse,csv,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from household_artifacts import sha

METRICS=('cost','objective','import_kwh','curtailment_kwh','charge_ac_kwh','discharge_ac_kwh','efc','peak_kw','final_soc','last_day_cost','fallback_segments','terminal_difference')
TARIFFS=('flat','tou','signed')
POLICIES=('PFG','THB','OMG','MPC','SCR-repair')

def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def household_mean(pairs):
    h=defaultdict(list)
    for customer,value in pairs:h[customer].append(value)
    return {k:float(np.mean(v)) for k,v in sorted(h.items())}

def stats(h,bootstrap=True):
    ids=sorted(h);a=np.array([h[k] for k in ids]);n=len(a)
    if not n:return dict(n_households=0)
    mean=float(a.mean());out=dict(n_households=n,mean=mean,median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
        q05=float(np.quantile(a,.05)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),q95=float(np.quantile(a,.95)),
        wins=int((a < -1e-7).sum()),losses=int((a>1e-7).sum()),ties=int((np.abs(a)<=1e-7).sum()),
        leave_one_out_mean_range=None)
    if n>1:out['leave_one_out_mean_range']=[float(((a.sum()-a)/(n-1)).min()),float(((a.sum()-a)/(n-1)).max())]
    if bootstrap and n>=20:
        rng=np.random.default_rng(20260928);means=a[rng.integers(0,n,(10000,n))].mean(axis=1)
        out['bootstrap_95_percentile']=[float(x) for x in np.quantile(means,[.025,.975])]
        out['bootstrap_upper_six_screen']=float(np.quantile(means,1-.05/6))
    return out

def paired(rows,lookup,ref_config,metric='cost'):
    return [(r['customer'],r['metrics'][metric]-lookup[(r['case_id'],r['tariff'],ref_config)]['metrics'][metric]) for r in rows
            if r['status']=='ok' and lookup[(r['case_id'],r['tariff'],ref_config)]['status']=='ok']

def analyze(base):
    complete=json.loads((base/'execution_completion.json').read_text())
    if not complete['all_cases_complete']:raise ValueError('Incomplete household execution cannot be analysed as the study')
    out=base/'analysis';out.mkdir(exist_ok=True)
    configs={c['config_id']:c for c in json.loads((base/'configurations.json').read_text())}
    cases=json.loads((base/'cases.json').read_text());rows=[];features={}
    for case in cases:
        folder=base/'cases'/case['case_id'];result=json.loads((folder/'results.json').read_text());rows.extend(result['rows'])
        features[case['case_id']]=json.loads((folder/'history_features.json').read_text())
    lookup={(r['case_id'],r['tariff'],r['config_id']):r for r in rows}
    if len(lookup)!=68595 or len(rows)!=68595:raise AssertionError('matrix cardinality')
    expected=list(csv.DictReader((base/'execution_matrix.csv').open()))
    by_id={r['row_id']:r for r in rows}
    for e in expected:
        r=by_id[e['row_id']]
        for key in ('case_id','tariff','config_id','execution_id'):assert e[key]==r[key],(e['row_id'],key)
        assert (e['reused_physical_trajectory']=='True')==r['reused_physical_trajectory']
    failures=[r for r in rows if r['status']!='ok']
    assert len(failures)==807 and all(r['config_id']=='capacity_5__OMG' and r['status']=='expected_parameter_domain_failure' for r in failures)
    csvsave(out/'domain_failures.csv',[{k:r[k] for k in ('row_id','customer','test_start','tariff','config_id','status','error')} for r in failures])
    flat=[]
    for r in rows:
        d={k:r[k] for k in ('row_id','execution_id','case_id','customer','test_start','tariff','config_id','group','status','reused_physical_trajectory')}
        d.update({k:r.get('metrics',{}).get(k) for k in METRICS});flat.append(d)
    csvsave(out/'all_comparison_metrics.csv',flat)
    balanced={k for k,c in Counter(x['customer'] for x in cases).items() if c==2}
    root=Path(__file__).resolve().parents[1]
    cohort=json.loads((root/'data/metadata/external_cohort.json').read_text())
    screen_path=root/'data/metadata/quality_screen.json'
    assert sha(screen_path)==cohort['source_screen_sha256']
    screen=json.loads(screen_path.read_text())
    attrition=[{k:q[k] for k in ('test_start','screened_customers','whole_site_quality','whole_site_clock_and_quality_candidates','general_load_only_quality','general_load_clock_and_quality_candidates')} for q in screen['summaries']]
    csvsave(out/'cohort_screen_summary.csv',attrition)
    summary=dict(households=len({c['customer'] for c in cases}),windows=len(cases),logical_rows=len(rows),physical_attempts=sum(not r['reused_physical_trajectory'] for r in rows),
                 successful_logical_rows=len(rows)-len(failures),expected_parameter_failures=len(failures),balanced_households=len(balanced),
                 independent_max_residual=max(r['audit']['independent_max_residual'] for r in rows if r['status']=='ok'),
                 unique_segments_checked=sum(r['audit']['segments_checked'] for r in rows if r['status']=='ok' and not r['reused_physical_trajectory']),
                 unique_decisions_checked=sum(r['audit']['decisions_checked'] for r in rows if r['status']=='ok' and not r['reused_physical_trajectory']),
                 cohort_screen=attrition,primary={},screens=[],sensitivity=[],numerical=[],certificate=[],regimes=[],wear=[],forecast=[],references=[],runtime=[])
    # Every saved row, including undefined ratios, survives in companion data.
    primary_points=[];group_summary=[]
    for tariff in TARIFFS:
        tr=[r for r in rows if r['tariff']==tariff]
        for config,c in configs.items():
            rr=[r for r in tr if r['config_id']==config and r['status']=='ok'];d=dict(tariff=tariff,config_id=config,group=c['group'],policy=c['policy'],n_windows=len(rr),n_households=len({r['customer'] for r in rr}))
            for metric in METRICS:
                hh=household_mean([(r['customer'],r['metrics'][metric]) for r in rr]);d[metric]=float(np.mean(list(hh.values()))) if hh else None
            group_summary.append(d)
        p=[r for r in tr if r['group']=='primary']
        res={}
        for policy in POLICIES:
            rr=[r for r in p if configs[r['config_id']]['policy']==policy]
            res[policy]={m:stats(household_mean([(r['customer'],r['metrics'][m]) for r in rr]),bootstrap=False) for m in METRICS}
            res[policy]['versus_pfg']=stats(household_mean(paired(rr,lookup,'primary__PFG')))
            res[policy]['balanced_versus_pfg']=stats(household_mean([(k,v) for k,v in paired(rr,lookup,'primary__PFG') if k in balanced]))
            for case in cases:
                r=lookup[(case['case_id'],tariff,'primary__'+policy)];ref=lookup[(case['case_id'],tariff,'primary__PFG')];f=features[case['case_id']]
                d=dict(customer=case['customer'],case_id=case['case_id'],test_start=case['test_start'],tariff=tariff,policy=policy,
                    cost_difference_from_pfg=r['metrics']['cost']-ref['metrics']['cost'],cost_per_100_load_kwh=100*r['metrics']['cost']/r['metrics']['load_kwh'] if r['metrics']['load_kwh'] else None,
                    saving_percent_from_pfg=100*(ref['metrics']['cost']-r['metrics']['cost'])/ref['metrics']['cost'] if ref['metrics']['cost']>0 else None,
                    pv_load_ratio=f['pv_load_ratio'],storage_daily_load_ratio=15/f['mean_daily_load_kwh'] if f['mean_daily_load_kwh'] else None,pv_coincidence=f['pv_coincidence'],
                    epsilon=.01*30/28*f['history_load_kwh']*(.08 if tariff=='flat' else .15))
                d.update({k:r['metrics'][k] for k in METRICS});primary_points.append(d)
        scr=[r for r in p if r['config_id']=='primary__SCR-repair'];res['SCR-repair']['versus_mpc']=stats(household_mean(paired(scr,lookup,'primary__MPC')))
        res['SCR-repair']['balanced_versus_mpc']=stats(household_mean([(k,v) for k,v in paired(scr,lookup,'primary__MPC') if k in balanced]))
        summary['primary'][tariff]=res
        for name,a,b in [('greedy','PFG','MPC'),('protected','SCR-repair','MPC')]:
            for tol in (0.,.005,.01,.02,.05):
                hh=[]
                for case in cases:
                    cid=case['case_id'];epsilon=tol*30/28*features[cid]['history_load_kwh']*(.08 if tariff=='flat' else .15)
                    hh.append((case['customer'],lookup[(cid,tariff,'primary__'+a)]['metrics']['cost']-lookup[(cid,tariff,'primary__'+b)]['metrics']['cost']-epsilon))
                ss=stats(household_mean(hh));summary['screens'].append(dict(tariff=tariff,comparison=name,tolerance_fraction=tol,primary=tol==.01,
                    supported_cohort_mean=ss['bootstrap_upper_six_screen']<0,**ss))
        for group in sorted({c['group'] for c in configs.values()}-{'primary','certificate'}):
            for a,b in [('MPC','PFG'),('SCR-repair','MPC')]:
                rr=[r for r in tr if r['config_id']==group+'__'+a]
                for metric in ('cost','objective'):
                    hh=household_mean(paired(rr,lookup,group+'__'+b,metric));summary['sensitivity'].append(dict(tariff=tariff,group=group,contrast=a+' minus '+b,metric=metric,**stats(hh)))
        for v in (0.,.076,.1425):
            for policy in ('MPC','SCR-repair'):
                coarse=f'endpoint_grid_251_{v:g}__{policy}';fine=('primary' if v==0 else f'endpoint_grid_501_{v:g}')+'__'+policy
                rr=[r for r in tr if r['config_id']==coarse];dif=paired(rr,lookup,fine,'objective')
                summary['numerical'].append(dict(tariff=tariff,terminal_value=v,policy=policy,contrast='251 minus 501 states',**stats(household_mean(dif)),max_absolute_case_difference=max(abs(x[1]) for x in dif)))
        for cert in ('global','copy','repair'):
            for budget in (0,1):
                cid='primary__SCR-repair' if cert=='repair' and budget==1 else f'certificate_{cert}_{budget}'
                rr=[r for r in tr if r['config_id']==cid];summary['certificate'].append(dict(tariff=tariff,certificate=cert,budget=budget,
                    versus_mpc=stats(household_mean(paired(rr,lookup,'primary__MPC'))),versus_pfg=stats(household_mean(paired(rr,lookup,'primary__PFG'))),
                    mean_fallback_segments=float(np.mean(list(household_mean([(r['customer'],r['metrics']['fallback_segments']) for r in rr]).values()))),
                    max_terminal_difference=max(r['metrics']['terminal_difference'] for r in rr),max_prefix_exposure=max(r['metrics']['max_prefix_exposure'] for r in rr)))
        for feature,edges in [('pv_load_ratio',[.5,1]),('storage_daily_load_ratio',[.5,1]),('pv_coincidence',[.5])]:
            points=[r for r in primary_points if r['tariff']==tariff and r['policy']=='MPC']
            for index in range(len(edges)+2):
                selected=[r for r in points if (len(edges)+1 if r[feature] is None else sum(r[feature]>x for x in edges))==index]
                label='undefined' if index==len(edges)+1 else ('<= '+str(edges[0]) if index==0 else '> '+str(edges[-1]) if index==len(edges) else f'({edges[index-1]}, {edges[index]}]')
                hh=household_mean([(r['customer'],r['cost_difference_from_pfg']) for r in selected])
                summary['regimes'].append(dict(tariff=tariff,feature=feature,bin=label,n_windows=len(selected),**stats(hh)))
        for coefficient in (0.,.005,.02,.05,.1):
            for policy in ('THB','OMG','MPC','SCR-repair'):
                rr=[r for r in p if r['config_id']=='primary__'+policy];dif=[]
                for r in rr:
                    ref=lookup[(r['case_id'],tariff,'primary__PFG')];dif.append((r['customer'],r['metrics']['cost']-ref['metrics']['cost']+coefficient*(r['metrics']['discharge_ac_kwh']-ref['metrics']['discharge_ac_kwh'])))
                summary['wear'].append(dict(tariff=tariff,policy=policy,coefficient_cu_per_ac_kwh=coefficient,**stats(household_mean(dif))))
    # Forecast error: target is the first completed-next interval and all valid 24 h horizon targets.
    forecast_rows=[]
    for case in cases:
        folder=base/'cases'/case['case_id']
        with np.load(folder/'inputs.npz') as f:actual=np.column_stack([f['load'][1344:],f['pv'][1344:]])
        for method in ('weekday','previous_day','persistence'):
            with np.load(folder/f'forecast_flat_{method}_1.npz') as f:pred=f['prediction']
            for scope in ('first_interval','whole_horizon'):
                idx=np.arange(1440)[:,None]+np.arange(1 if scope=='first_interval' else 48)[None,:];valid=idx<1440;target=actual[np.minimum(idx,1439)];prediction=pred[:,:idx.shape[1]]
                for j,name in enumerate(('load','pv')):
                    err=np.abs(prediction[:,:,j]-target[:,:,j])[valid];den=target[:,:,j][valid].sum()
                    forecast_rows.append(dict(customer=case['customer'],case_id=case['case_id'],forecast=method,scope=scope,channel=name,mae_kwh=float(err.mean()),wape=float(err.sum()/den) if den else None,n_predictions=int(valid.sum())))
    csvsave(out/'forecast_errors.csv',forecast_rows)
    for method in ('weekday','previous_day','persistence'):
        for scope in ('first_interval','whole_horizon'):
            for channel in ('load','pv'):
                rr=[r for r in forecast_rows if r['forecast']==method and r['scope']==scope and r['channel']==channel]
                summary['forecast'].append(dict(forecast=method,scope=scope,channel=channel,mae=stats(household_mean([(r['customer'],r['mae_kwh']) for r in rr]),False),
                    wape=stats(household_mean([(r['customer'],r['wape']) for r in rr if r['wape'] is not None]),False),undefined_wape_windows=sum(r['wape'] is None for r in rr)))
    csvsave(out/'primary_case_points.csv',primary_points);csvsave(out/'configuration_household_means.csv',group_summary)
    household_points=[]
    for tariff in TARIFFS:
        for policy in POLICIES:
            pp=[r for r in primary_points if r['tariff']==tariff and r['policy']==policy]
            ids=sorted({r['customer'] for r in pp})
            for customer in ids:
                rr=[r for r in pp if r['customer']==customer];d=dict(customer=customer,tariff=tariff,policy=policy,n_windows=len(rr))
                d.update({m:float(np.mean([r[m] for r in rr])) for m in (*METRICS,'cost_difference_from_pfg','epsilon')});household_points.append(d)
    csvsave(out/'primary_household_points.csv',household_points)
    # Primary adverse cases are selected mechanically, including all cases in full points file.
    adverse=sorted([r for r in primary_points if r['policy']=='MPC'],key=lambda x:-x['cost_difference_from_pfg'])
    csvsave(out/'largest_observed_mpc_cost_increases.csv',adverse[:30])
    diag=base/'diagnostics'
    if (diag/'completion.json').exists():
        refs=json.loads((diag/'references.json').read_text());timings=json.loads((diag/'runtime_samples.json').read_text())
        for tariff in TARIFFS:
            for kind in ('perfect_forecast_mpc','hindsight_milp'):
                rr=[r for r in refs if r['tariff']==tariff and r['kind']==kind];d=dict(tariff=tariff,kind=kind,n_windows=len(rr),statuses=dict(Counter(str(r['status']) for r in rr)))
                if kind=='perfect_forecast_mpc':
                    hh=household_mean([(int(r['case_id'][1:4]),lookup[(r['case_id'],tariff,'primary__MPC')]['metrics']['cost']-r['metrics']['cost']) for r in rr]);d['causal_mpc_minus_reference']=stats(hh,False)
                else:
                    available=[r for r in rr if r.get('dual_bound') is not None];d['valid_dual_bound_windows']=len(available)
                    d['causal_mpc_minus_lower_bound']=stats(household_mean([(int(r['case_id'][1:4]),lookup[(r['case_id'],tariff,'primary__MPC')]['metrics']['cost']-r['dual_bound']) for r in available]),False)
                    d['largest_reported_mip_gap']=max((r['mip_gap'] for r in rr if r.get('mip_gap') is not None),default=None)
                summary['references'].append(d)
        for policy in POLICIES:
            rr=[r for r in timings if r['policy']==policy]
            for metric in ('forecast_seconds','supervisor_seconds','local_and_guard_seconds','total_decision_seconds'):
                a=np.array([r[metric] for r in rr]);summary['runtime'].append(dict(policy=policy,metric=metric,n_samples=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())))
    summary['analysis_scope']='Frozen estimands implemented after execution began; six primary screens separated from exploratory diagnostics. Household bootstrap is approximate cohort evidence.'
    save(out/'summary.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();s=analyze(a.package);print(json.dumps({k:s[k] for k in ('households','windows','logical_rows','expected_parameter_failures')},indent=2))
