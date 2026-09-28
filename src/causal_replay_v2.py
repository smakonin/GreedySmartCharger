"""Common executor for matched unshielded baselines and shielded nominal MPC.

The frozen v1 physics and shield are imported without modification. This version
adds an explicitly unshielded path; it does not certify comparator performance.
Instantaneous telemetry and event interrupts remain required.
"""
from dataclasses import asdict
import math
from causal_replay import (PowerCommand, Context, past_means, pfg_command,
    plan_pair, select_plan, rate, _snap)
from grid_model import Action, Interval, transition


def replay(samples,model,actual_policy,shadow_policy=pfg_command,*,kind='repair',
           budget=0.,initial_soc=0.,supervisory_seconds=300.,feedback_seconds=30.):
    """Synthetic event-driven replay; kind=None executes unshielded comparators. Callbacks receive immutable causal Contexts.

    Fixed telemetry/power model: instantaneous exact samples reveal every change.
    New samples interrupt the local guard but not the supervisory clock. No
    observed *future* sample time is used to choose or certify a command.
    """
    if not all(math.isfinite(x) for x in (budget,supervisory_seconds,feedback_seconds)) or budget<0 or min(supervisory_seconds,feedback_seconds)<=0:
        raise ValueError('invalid replay configuration')
    if kind not in (None,'global','copy','repair'):raise ValueError('unknown certificate')
    if not 0<=initial_soc<=model.b.capacity:raise ValueError('invalid initial SOC')
    if any(a.time>=b.time for a,b in zip(samples,samples[1:])):raise ValueError('samples must be strictly ordered')
    history=[];i=0;time=model.start;s=z=initial_soc;regret=0.;cost=basecost=0.
    decision_rows=[];trace=[];super_next=model.start;feedback_next=model.start
    while time<model.end-1e-9:
        while i<len(samples) and samples[i].ready_at<=time:
            history.append(samples[i]);i+=1
        if not history:raise ValueError('no available observation')
        sample=history[-1]
        if time-sample.time>60:raise ValueError('stale telemetry; no certification across gaps')
        if time>=super_next-1e-9:
            meanload,meanpv=past_means(history,time)
            context=lambda state:Context(time,state,sample.load_kw,sample.pv_kw,sample.price,
                                        meanload,meanpv,tuple(history),model.end,model.bands)
            actual_context=context(s);shadow_context=context(z)
            command=actual_policy(actual_context,model.b);shadow_command=shadow_policy(shadow_context,model.b)
            if not isinstance(command,PowerCommand) or not isinstance(shadow_command,PowerCommand):raise TypeError('policies must return PowerCommand')
            decision_rows.append(dict(time=time,actual_context=asdict(actual_context),shadow_context=asdict(shadow_context),command=asdict(command),shadow_command=asdict(shadow_command)))
            super_next+=supervisory_seconds
        while feedback_next<=time+1e-9:feedback_next+=feedback_seconds
        deadline=min(model.end,super_next,feedback_next,model.next_boundary(time),sample.time+60)
        if deadline<=time+1e-10:raise ValueError('stale telemetry before next feedback')
        if not model.price_in_range(time,sample.price):
            raise ValueError('observed price outside deterministic envelope')
        if kind is None:
            plan=plan_pair(command,shadow_command,sample.load_kw,sample.pv_kw,s,z,model.b,deadline-time)
            info=dict(fallback=False,branch=None,proposed_peak=None,accepted_peak=None)
        else:
            plan,info=select_plan(command,shadow_command,sample,s,z,model,kind,time,deadline-time,regret,budget)
        # Only the environmental simulator reads the next arrival to interrupt.
        stop=min(deadline,samples[i].ready_at if i<len(samples) else model.end)
        start=time
        for segment in plan:
            dt=min(segment.duration,stop-start-segment.offset)
            if dt<=1e-10:break
            ap,bp=segment.power,segment.shadow_power
            x=Interval(sample.load_kw*dt/3600,sample.pv_kw*dt/3600,sample.price,dt/3600)
            action=lambda p:Action(p.direct*dt/3600,p.pv_charge*dt/3600,p.discharge*dt/3600,p.grid_charge*dt/3600)
            flow=transition(x,s,action(ap),model.b);baseline=transition(x,z,action(bp),model.b)
            peak=(max(regret,regret+flow.cost-baseline.cost) if kind is None else
                  model.max_along(kind,start+segment.offset,dt,s,rate(ap,model.b),z,rate(bp,model.b),regret,(flow.cost-baseline.cost)/dt))
            old_s,old_z=s,z;s=_snap(flow.end_soc,model.b);z=_snap(baseline.end_soc,model.b)
            cost+=flow.cost;basecost+=baseline.cost;regret=cost-basecost
            end=start+segment.offset+dt;phi=0. if kind is None else model.value(kind,end,s,z)
            trace.append(dict(start=start+segment.offset,end=end,load_kw=sample.load_kw,pv_kw=sample.pv_kw,price=sample.price,
                              before_soc=old_s,before_shadow_soc=old_z,soc=s,shadow_soc=z,power=asdict(ap),shadow_power=asdict(bp),
                              cost=flow.cost,shadow_cost=baseline.cost,grid_kwh=flow.grid,curtailed_kwh=flow.curtailed,
                              regret=regret,liability=phi,slack=budget-regret-phi,segment_peak=peak,**info))
            if kind is not None and (peak>budget+1e-8 or regret+phi>budget+1e-8):raise AssertionError('executed prefix guarantee failed')
        time=stop
    terminal_difference=regret-model.v*(s-z)
    if kind is not None and terminal_difference>budget+1e-8:raise AssertionError('terminal-adjusted guarantee failed')
    return dict(shielded=kind is not None,kind=kind,budget=budget,initial_soc=initial_soc,final_soc=s,shadow_final_soc=z,
                cost=cost,shadow_cost=basecost,regret=regret,terminal_adjusted_difference=terminal_difference,
                max_prefix_exposure=max((r['segment_peak'] for r in trace),default=0.),
                decisions=decision_rows,trace=trace)
