"""Pre-outcome, one-factor-at-a-time household matrix; no household outcomes read."""
import csv,hashlib,json
from pathlib import Path

POLICIES=['PFG','THB','OMG','MPC','SCR-repair']


def configurations():
    out=[]
    def add(group,**changes):
        for policy in POLICIES:
            d=dict(group=group,policy=policy,capacity=15.,charge_kw=5.,discharge_kw=5.,initial_fraction=0.,
                   terminal_value=0.,states=501,forecast='weekday',supervisory_steps=1,
                   certificate='repair' if policy=='SCR-repair' else None,budget=1. if policy=='SCR-repair' else 0.)
            d.update(changes);d['config_id']=group+'__'+policy;out.append(d)
    for states in (501,251):
        for v in (0.,.076,.1425):
            add('primary' if states==501 and v==0 else f'endpoint_grid_{states}_{v:g}',states=states,terminal_value=v)
    for method in ('previous_day','persistence'):add('forecast_'+method,forecast=method)
    for capacity in (5.,10.,30.):add(f'capacity_{capacity:g}',capacity=capacity)
    for power in (2.5,10.):add(f'power_{power:g}',charge_kw=power,discharge_kw=power)
    for initial in (.5,1.):add(f'initial_{initial:g}',initial_fraction=initial)
    add('hourly',supervisory_steps=2)
    for cert,budget in [('global',0.),('copy',0.),('repair',0.),('global',1.),('copy',1.)]:
        d=dict(next(r for r in out if r['config_id']=='primary__SCR-repair'))
        d.update(group='certificate',config_id=f'certificate_{cert}_{budget:g}',certificate=cert,budget=budget,policy='SCR-'+cert)
        out.append(d)
    assert len(out)==85
    return out


def physical_key(config):
    d={k:config[k] for k in ('policy','capacity','charge_kw','discharge_kw','initial_fraction','terminal_value','states','forecast','supervisory_steps','certificate','budget')}
    if d['policy'] in ('PFG','THB','OMG'):
        d['states']=None;d['terminal_value']=0.
        if d['policy']=='PFG':d['forecast']='weekday'
    return json.dumps(d,sort_keys=True,separators=(',',':'))


def write_matrix(root,base):
    cohort=json.loads((root/'research/applied-energy-v1/external_cohort.json').read_text())
    configs=configurations();canonical={}
    for c in configs:canonical.setdefault(physical_key(c),c['config_id'])
    (base/'configurations.json').write_text(json.dumps(configs,indent=2)+'\n')
    cases=[]
    for c in cohort['cases']:
        cases.append(dict(**c,case_id=f"A{c['customer']:03d}_{c['test_start'].replace('-','')}"))
    (base/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    with (base/'execution_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['row_id','case_id','customer','test_start','tariff','config_id','execution_id','reused_physical_trajectory']);w.writeheader()
        for case in cases:
            for tariff in ('flat','tou','signed'):
                for c in configs:
                    source=canonical[physical_key(c)];prefix=case['case_id']+'__'+tariff+'__'
                    w.writerow(dict(row_id=prefix+c['config_id'],case_id=case['case_id'],customer=case['customer'],test_start=case['test_start'],tariff=tariff,config_id=c['config_id'],execution_id=prefix+source,reused_physical_trajectory=source!=c['config_id']))
    ids=sorted(cohort['balanced_two_window_households'],key=lambda n:hashlib.sha256(str(n).encode()).hexdigest())[:10]
    selected=[c for c in cases if c['customer'] in ids]
    reference=dict(selection='ten smallest SHA256(decimal customer ID) among the 131 balanced households, before controller outcomes',households=ids,cases=selected,
                   perfect_forecast_mpc=dict(states=501,terminal_value=0.,capacity=15.,power=5.,initial_fraction=0.,tariffs=['flat','tou','signed'],nonimplementable=True),
                   hindsight_milp=dict(time_limit_seconds=10.,mip_relative_gap=1e-6,capacity=15.,power=5.,terminal_value=0.,initial_fraction=0.,tariffs=['flat','tou','signed'],nonimplementable=True,retain_dual_bound_and_primal_gap=True),
                   runtime=dict(config='primary',decision_indices=sorted(range(1440),key=lambda n:hashlib.sha256(('decision:'+str(n)).encode()).hexdigest())[:20],mode='serial, isolated uncached preparation; no concurrent study workers'))
    (base/'diagnostic_matrix.json').write_text(json.dumps(reference,indent=2)+'\n')
    return dict(households=len(cohort['households']),household_windows=len(cases),tariffs=3,configurations_per_tariff=len(configs),
                logical_rows=len(cases)*3*len(configs),unique_physical_trajectories=len(cases)*3*len(canonical),
                primary_rows=len(cases)*3*5,diagnostic_perfect_forecast_rows=len(selected)*3,diagnostic_hindsight_rows=len(selected)*3)
