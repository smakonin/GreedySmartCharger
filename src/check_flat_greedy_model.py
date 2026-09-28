"""Post-outcome structural sanity check; no controller tuning or exclusions."""
import argparse,json
from pathlib import Path
import numpy as np
from grid_controllers import pfg
from grid_model import Interval,transition
from physical_simulator import BatterySpec
from household_hindsight import solve_hindsight


def synthetic():
    rng=np.random.default_rng(2026092807);records=[]
    for i in range(64):
        b=BatterySpec(float(rng.choice([1,5,15,30])),float(rng.choice([0,1,5,10])),float(rng.choice([0,1,5,10])),float(rng.choice([.6,.95,1.])),float(rng.choice([.6,.95,1.])))
        load=rng.uniform(0,5,16);pv=rng.uniform(0,5,16);price=float(rng.choice([0.,.08,.15]));prices=np.full(16,price)
        initial=float(rng.choice([0.,b.capacity/2,b.capacity]));soc=initial;cost=0.
        for l,p in zip(load,pv):
            x=Interval(float(l),float(p),price,.5);flow=transition(x,soc,pfg(x,soc,b),b);soc=flow.end_soc;cost+=flow.cost
        ref,_=solve_hindsight(load,pv,prices,b,initial_soc=initial,time_limit=10.)
        assert ref['status']==0 and ref['dual_bound'] is not None
        gap=cost-ref['dual_bound'];assert abs(gap)<=1e-5,(i,gap)
        records.append(dict(case=i,pfg_cost=cost,milp_lower_bound=ref['dual_bound'],difference=gap))
    return dict(passed=True,cases=64,seed=2026092807,maximum_absolute_gap=max(abs(x['difference']) for x in records),records=records,
                scope='Synthetic flat-price interval-mode MILP checks; post-outcome explanatory validation, not new household outcomes or novelty evidence.')


def household(base):
    assert json.loads((base/'execution_completion.json').read_text())['all_cases_complete']
    checked=0;minimum=float('inf');worst=None
    for folder in sorted((base/'cases').iterdir()):
        rows=[x for x in json.loads((folder/'results.json').read_text())['rows'] if x['tariff']=='flat' and x['status']=='ok']
        lookup={x['config_id']:x for x in rows}
        for x in rows:
            group='primary' if x['group']=='certificate' else x['group']
            reference=lookup[group+'__PFG'];gap=x['metrics']['cost']-reference['metrics']['cost']
            if gap<minimum:minimum=gap;worst=x['row_id']
            assert gap>=-1e-7,(x['row_id'],gap)
            checked+=1
    return dict(passed=True,flat_rows_checked=checked,minimum_non_greedy_minus_matched_pfg_cost=minimum,minimum_row=worst,
        scope='Cash import cost against the equipment/initial-state-matched PFG. Does not assert optimality of a terminal-adjusted objective with positive terminal credit. Post-outcome model consistency check, not a new confirmatory test.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=household(a.package) if a.package else synthetic();a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='records'})
