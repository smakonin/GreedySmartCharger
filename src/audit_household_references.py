"""Check the reference's interval-mode scope against each compared MPC trace."""
import argparse,json
from pathlib import Path
import numpy as np


def audit(base):
    refs=json.loads((base/'diagnostics/references.json').read_text())
    assert len(refs)==120
    records=[];maximum=0.
    for r in refs:
        if r['kind']!='hindsight_milp':continue
        folder=base/'cases'/r['case_id']
        path=folder/(r['tariff']+'__primary__MPC.npz')
        with np.load(path) as f:
            cols=json.loads(str(f['layout']))['intervals']['numeric'];iv={k:f['intervals'][:,i] for i,k in enumerate(cols)}
        c=iv['charge_ac_kwh'];d=iv['discharge_ac_kwh'];g=iv['import_kwh']
        used=iv['pv_kwh']-iv['curtailment_kwh'];s=iv['end_soc']
        residual=max(float(np.max(np.abs(g+used+d-iv['load_kwh']-c))),
                     float(np.max(np.abs(s-iv['start_soc']-.95*c+d/.95))))
        assert residual<=1e-7
        assert np.all(c<=2.5+1e-8) and np.all(d<=2.5+1e-8)
        assert np.all(d<=iv['load_kwh']+1e-8)
        assert np.all(used>=-1e-8) and np.all(used<=iv['pv_kwh']+1e-8)
        assert np.all(s>=-1e-8) and np.all(s<=15.+1e-8)
        assert not np.any((c>1e-8)&(d>1e-8)),(r['case_id'],r['tariff'],'MPC changes mode within interval')
        cost=float(iv['cost'].sum());bound=r.get('dual_bound')
        if bound is not None:assert bound<=cost+1e-5,(r['case_id'],r['tariff'],cost,bound)
        maximum=max(maximum,residual)
        records.append(dict(case_id=r['case_id'],tariff=r['tariff'],mpc_feasible_in_interval_mode_model=True,
                            mpc_cost=cost,dual_bound=bound,dual_bound_no_greater_than_mpc=None if bound is None else True,
                            arithmetic_residual=residual))
    assert len(records)==60
    result=dict(passed=True,compared_mpc_windows=60,maximum_residual=maximum,records=records,
        scope='MILP permits one charging/discharging mode per half-hour. Every primary MPC trajectory actually compared with its dual bound is feasible in that restricted interval model. The dual bound is not asserted to lower-bound all continuously switching SCR trajectories or the broader continuous-time optimum.')
    from check_flat_greedy_model import household
    flat=household(base)
    (base/'flat_greedy_household_check.json').write_text(json.dumps(flat,indent=2)+'\n')
    (base/'diagnostics/reference_scope_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();r=audit(a.package)
    print(json.dumps({k:r[k] for k in ('passed','compared_mpc_windows','maximum_residual')}))
