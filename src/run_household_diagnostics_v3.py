"""Fixed-subset perfect-information references and serial uncached decision timing."""
import argparse,datetime,json,math,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from interval_meter import MeterReading,MeterContext,forecast,request_from_action,HISTORY,STEP
from household_meter import BatteryRequest,local_command,tariff_bands,replay_meter
from household_controller import HouseholdController
from native_household_controller import ForecastBook
from household_artifacts import metrics,pack_result,sha,verify_result
from household_hindsight import solve_hindsight
from native_interval_planner import nominal_action
from available_price_control import thb,omg
from causal_replay import plan_pair,select_plan,Sample
from household_liability_v2 import ExecutedLiability
from physical_simulator import BatterySpec
from run_household_comparison import write_json


def read_inputs(path):
    with np.load(path) as f:load=f['load'];pv=f['pv']
    return [MeterReading((i-HISTORY)*STEP,(i-HISTORY+1)*STEP,float(l),float(p),(i-HISTORY+1)*STEP) for i,(l,p) in enumerate(zip(load,pv))]


def uncached_decision(ctx,b,name):
    start=time.perf_counter()
    if name=='PFG':return BatteryRequest(local_greedy=True),0.,time.perf_counter()-start
    xs,_=forecast(ctx);forecast_s=time.perf_counter()-start;start=time.perf_counter();x=xs[0]
    prices=[p.low for p in ctx.price_bands]
    if name=='THB':
        load=math.fsum(r.load_kwh for r in ctx.history);rho=min(1.,math.fsum(r.pv_kwh for r in ctx.history)/load) if load else 0.
        positive=[p for p in prices if p>0];a=thb(x,ctx.soc,b,min(positive) if positive else 1e-4,max(max(prices),1e-4),rho)
    elif name=='OMG':a=omg(x,ctx.soc,b,min(prices),max(prices))
    else:a=nominal_action(x,ctx.soc,b,xs[1:],501,0.)
    q=request_from_action(a,x.dt,x.price)
    return q,forecast_s,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();base=a.package
    if not json.loads((base/'execution_completion.json').read_text())['all_cases_complete']:raise ValueError('study workers must finish first')
    design=json.loads((base/'diagnostic_matrix.json').read_text());out=base/'diagnostics';out.mkdir(exist_ok=False)
    b=BatterySpec();references=[];timings=[]
    for case in design['cases']:
        cbase=base/'cases'/case['case_id'];rr=read_inputs(cbase/'inputs.npz')
        for tariff in design['perfect_forecast_mpc']['tariffs']:
            tag=case['case_id']+'__'+tariff;m=ExecutedLiability(tariff_bands(30,tariff),b)
            r=replay_meter(rr,m,HouseholdController('MPC',501,0.,ForecastBook(rr,m.bands,'perfect')))
            checked=verify_result(r,b,0.,None,0.);digest=pack_result(out/(tag+'__perfect.npz'),r)
            perfect=dict(case_id=case['case_id'],tariff=tariff,kind='perfect_forecast_mpc',status='ok',nonimplementable=True,
                         metrics=metrics(r,b,{'terminal_value':0.}),audit=checked,trace_sha256=digest)
            references.append(perfect);del r
            actual=np.array([[x.load_kwh,x.pv_kwh] for x in rr[HISTORY:]])
            prices=np.array([next(p.low for p in m.bands if p.start<=i*STEP<p.end) for i in range(1440)])
            try:
                report,solution=solve_hindsight(actual[:,0],actual[:,1],prices,b,time_limit=design['hindsight_milp']['time_limit_seconds'])
                if solution is not None:np.savez_compressed(out/(tag+'__hindsight.npz'),solution=solution)
                references.append(dict(case_id=case['case_id'],tariff=tariff,kind='hindsight_milp',**report))
            except Exception as exc:references.append(dict(case_id=case['case_id'],tariff=tariff,kind='hindsight_milp',status='solver_failure',error=repr(exc)))
            # No study workers run during these uncached, serial timing samples.
            for policy in ('PFG','THB','OMG','MPC','SCR-repair'):
                path=cbase/(tariff+'__primary__'+policy+'.npz')
                with np.load(path) as f:
                    layout=json.loads(str(f['layout']));cols=layout['decisions']['numeric'];values=f['decisions'].copy()
                for k in design['runtime']['decision_indices']:
                    row=dict(zip(cols,values[k]));soc=float(row['soc']);z=float(row['shadow_soc']);regret=float(row['regret_before']);t=k*STEP
                    ctx=MeterContext(t,soc,tuple(rr[k:k+HISTORY]),m.end,m.bands)
                    q,ft,st=uncached_decision(ctx,b,'MPC' if policy.startswith('SCR') else policy)
                    if abs(q.kw-row['request.kw'])>1e-8:raise AssertionError('timed request differs from executed request')
                    start=time.perf_counter();sample=rr[HISTORY+k];load=sample.load_kwh*2;pv=sample.pv_kwh*2;price=prices[k]
                    cmd=local_command(q,load,pv,price,soc,b);shadow=local_command(BatteryRequest(local_greedy=True),load,pv,price,z,b)
                    plan=plan_pair(cmd,shadow,load,pv,soc,z,b,STEP)
                    if policy.startswith('SCR'):select_plan(cmd,shadow,Sample(t,load,pv,price,t),soc,z,m,'repair',t,plan[0].duration,regret,1.)
                    local=time.perf_counter()-start
                    timings.append(dict(case_id=case['case_id'],tariff=tariff,policy=policy,decision_index=k,
                                        forecast_seconds=ft,supervisor_seconds=st,local_and_guard_seconds=local,total_decision_seconds=ft+st+local))
            write_json(out/'progress.json',dict(completed_case_tariffs=len(references)//2,total_case_tariffs=60,runtime_samples=len(timings)))
            print(json.dumps(dict(tag=tag,completed_references=len(references),runtime_samples=len(timings))),flush=True)
    write_json(out/'references.json',references);write_json(out/'runtime_samples.json',timings)
    write_json(out/'completion.json',dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),references=len(references),runtime_samples=len(timings),
        runtime_scope='20 predetermined decision indices per policy per selected case/tariff; fresh causal forecast, uncached planner, projection and initial guard. Later local events and full episode runtime are not timed by this benchmark.'))


if __name__=='__main__':main()
