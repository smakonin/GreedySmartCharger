"""Causal supervisory commands and local executed-action shielding.

Synthetic instantaneous telemetry contract only. No Cyprus loader or activation.
The policy never receives future samples, next sample time, realized future energy,
or environmental segment duration. Local feedback runs on a 30-second clock AND
on newly observed input changes; this explicit event-interrupt requirement matters.
"""
from dataclasses import asdict,dataclass
import math
from grid_model import Action,Interval,transition


@dataclass(frozen=True)
class PowerCommand:
    direct: float=0.
    pv_charge: float=0.
    discharge: float=0.
    grid_charge: float=0.

    def __post_init__(self):
        values=(self.direct,self.pv_charge,self.discharge,self.grid_charge)
        if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('invalid power command')
        if self.discharge>0 and self.pv_charge+self.grid_charge>0:
            raise ValueError('simultaneous charge/discharge command')


@dataclass(frozen=True)
class Sample:
    time: float
    load_kw: float
    pv_kw: float
    price: float
    ready_at: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.time,self.load_kw,self.pv_kw,self.price,self.ready_at)) or min(self.load_kw,self.pv_kw)<0:
            raise ValueError('invalid sample')
        if self.ready_at!=self.time:
            raise ValueError('instantaneous local telemetry required; delayed/averaged data unsupported')


@dataclass(frozen=True)
class Context:
    time: float
    soc: float
    load_kw: float
    pv_kw: float
    price: float
    mean_load_kw: float
    mean_pv_kw: float
    history: tuple
    episode_end: float
    price_bands: tuple


@dataclass(frozen=True)
class Segment:
    offset: float
    duration: float
    soc: float
    shadow_soc: float
    power: PowerCommand
    shadow_power: PowerCommand


def bounded_power(cmd,load,pv,soc,b):
    """Direct PV first, then PV charge, then grid charge; no export/dumping."""
    direct=min(cmd.direct,load,pv)
    discharge=min(cmd.discharge,b.discharge_kw,max(0.,load-direct)) if soc>0 else 0.
    pvc=min(cmd.pv_charge,max(0.,pv-direct),b.charge_kw) if soc<b.capacity else 0.
    gc=min(cmd.grid_charge,max(0.,b.charge_kw-pvc)) if soc<b.capacity else 0.
    return PowerCommand(direct,pvc,discharge,gc)


def rate(p,b):return (b.eta_charge*(p.pv_charge+p.grid_charge)-p.discharge/b.eta_discharge)/3600
def grid_rate(p,load):return (load-p.direct-p.discharge+p.grid_charge)/3600


def coupled_power(shadow,s,z,b,branch):
    if branch=='repair' and s<z:
        return PowerCommand(shadow.direct,shadow.pv_charge,0.,max(0.,b.charge_kw-shadow.pv_charge))
    if branch not in ('copy','repair'):raise ValueError('invalid fallback branch')
    pvc=shadow.pv_charge if s<b.capacity else 0.
    gc=shadow.grid_charge if s<b.capacity else 0.
    discharge=shadow.discharge if s>0 else 0.
    return PowerCommand(shadow.direct,pvc,discharge,gc)


def _snap(s,b):
    if not -1e-9<=s<=b.capacity+1e-9:raise AssertionError('SOC violation')
    if abs(s)<1e-12:return 0.
    if abs(s-b.capacity)<1e-12:return b.capacity
    return s


def plan_pair(command,shadow_command,load,pv,s,z,b,duration,branch=None):
    """Hypothetical constant-current-input execution for a *known* max duration.

    Includes exact SOC saturation and repair synchronization events. The simulator
    may interrupt this plan at any earlier, unannounced measurement time.
    """
    if duration<=0:raise ValueError('positive duration required')
    if not (0<=s<=b.capacity and 0<=z<=b.capacity):raise ValueError('invalid initial SOC')
    out=[];elapsed=0.
    for _ in range(32):
        remaining=duration-elapsed
        if remaining<=1e-10:break
        bp=bounded_power(shadow_command,load,pv,z,b)
        ap=coupled_power(bp,s,z,b,branch) if branch else bounded_power(command,load,pv,s,b)
        ds,dz=rate(ap,b),rate(bp,b);limits=[remaining]
        for state,deriv in [(s,ds),(z,dz)]:
            if deriv>0:limits.append((b.capacity-state)/deriv)
            elif deriv<0:limits.append(-state/deriv)
        if branch=='repair' and s<z and ds>dz:limits.append((z-s)/(ds-dz))
        step=min(limits)
        if step<=0:raise AssertionError('zero-time saturation event')
        out.append(Segment(elapsed,step,s,z,ap,bp))
        s=_snap(s+ds*step,b);z=_snap(z+dz*step,b)
        if branch=='repair' and abs(s-z)<1e-11:s=z
        elapsed+=step
    else:raise AssertionError('too many physical events')
    return out


def exposure(plan,model,kind,time,regret,load,price):
    peak=regret+model.value(kind,time,plan[0].soc,plan[0].shadow_soc)
    for segment in plan:
        delta=price*(grid_rate(segment.power,load)-grid_rate(segment.shadow_power,load))
        peak=max(peak,model.max_along(kind,time+segment.offset,segment.duration,
                 segment.soc,rate(segment.power,model.b),segment.shadow_soc,
                 rate(segment.shadow_power,model.b),regret,delta))
        regret+=delta*segment.duration
    return peak


def select_plan(command,shadow_command,sample,s,z,model,kind,time,duration,regret,budget):
    """Certify the physically saturated plan at EVERY possible interruption time."""
    if not model.price_in_range(time,sample.price):raise ValueError('observed price outside deterministic envelope')
    if time+duration>model.next_boundary(time)+1e-10:raise ValueError('guard crosses price-envelope boundary')
    b=model.b
    proposed=plan_pair(command,shadow_command,sample.load_kw,sample.pv_kw,s,z,b,duration)
    upper=exposure(proposed,model,kind,time,regret,sample.load_kw,sample.price)
    # Strict admission: no per-step allowance and no artificial margin that
    # would forbid mathematically exact zero-cost actions at a zero budget.
    # Floating-point residuals are audited separately, not added to the budget.
    if upper<=budget:
        return proposed,dict(fallback=False,branch=None,proposed_peak=upper,accepted_peak=upper)
    branch=model.fallback(kind,time,s,z)
    fallback=plan_pair(command,shadow_command,sample.load_kw,sample.pv_kw,s,z,b,duration,branch)
    peak=exposure(fallback,model,kind,time,regret,sample.load_kw,sample.price)
    if peak>budget+1e-8:raise AssertionError(('fallback not admissible',peak,budget,kind,branch))
    return fallback,dict(fallback=True,branch=branch,proposed_peak=upper,accepted_peak=peak)


def past_means(history,time,window=300.):
    cutoff=time-window
    if not history or history[0].time>cutoff:raise ValueError('insufficient causal warmup')
    loads=pvs=covered=0.
    for i,a in enumerate(history):
        end=history[i+1].time if i+1<len(history) else time
        dt=max(0.,min(time,end)-max(cutoff,a.time))
        if dt>0 and end-a.time>60:raise ValueError('gap in causal warmup')
        loads+=a.load_kw*dt;pvs+=a.pv_kw*dt;covered+=dt
    if abs(covered-window)>1e-7:raise ValueError('incomplete warmup')
    return loads/window,pvs/window


def pfg_command(ctx,b):
    """Supervisory power-command adaptation, not an unchanged hourly PFG row."""
    direct=min(ctx.load_kw,ctx.pv_kw);pvc=min(max(0.,ctx.pv_kw-direct),b.charge_kw)
    if ctx.price<0:return PowerCommand(direct,pvc,0.,b.charge_kw-pvc)
    if pvc>0:return PowerCommand(direct,pvc)
    return PowerCommand(direct,0.,min(max(0.,ctx.load_kw-direct),b.discharge_kw))


def replay(samples,model,actual_policy,shadow_policy=pfg_command,*,kind='repair',
           budget=0.,initial_soc=0.,supervisory_seconds=300.,feedback_seconds=30.):
    """Synthetic event-driven replay. Callbacks receive immutable causal Contexts.

    Fixed telemetry/power model: instantaneous exact samples reveal every change.
    New samples interrupt the local guard but not the supervisory clock. No
    observed *future* sample time is used to choose or certify a command.
    """
    if not all(math.isfinite(x) for x in (budget,supervisory_seconds,feedback_seconds)) or budget<0 or min(supervisory_seconds,feedback_seconds)<=0:
        raise ValueError('invalid replay configuration')
    if kind not in ('global','copy','repair'):raise ValueError('unknown certificate')
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
            peak=model.max_along(kind,start+segment.offset,dt,s,rate(ap,model.b),z,rate(bp,model.b),regret,(flow.cost-baseline.cost)/dt)
            old_s,old_z=s,z;s=_snap(flow.end_soc,model.b);z=_snap(baseline.end_soc,model.b)
            cost+=flow.cost;basecost+=baseline.cost;regret=cost-basecost
            end=start+segment.offset+dt;phi=model.value(kind,end,s,z)
            trace.append(dict(start=start+segment.offset,end=end,load_kw=sample.load_kw,pv_kw=sample.pv_kw,price=sample.price,
                              before_soc=old_s,before_shadow_soc=old_z,soc=s,shadow_soc=z,power=asdict(ap),shadow_power=asdict(bp),
                              cost=flow.cost,shadow_cost=baseline.cost,grid_kwh=flow.grid,curtailed_kwh=flow.curtailed,
                              regret=regret,liability=phi,slack=budget-regret-phi,segment_peak=peak,**info))
            if peak>budget+1e-8 or regret+phi>budget+1e-8:raise AssertionError('executed prefix guarantee failed')
        time=stop
    terminal_difference=regret-model.v*(s-z)
    if terminal_difference>budget+1e-8:raise AssertionError('terminal-adjusted guarantee failed')
    return dict(kind=kind,budget=budget,initial_soc=initial_soc,final_soc=s,shadow_final_soc=z,
                cost=cost,shadow_cost=basecost,regret=regret,terminal_adjusted_difference=terminal_difference,
                max_prefix_exposure=max((r['segment_peak'] for r in trace),default=0.),
                decisions=decision_rows,trace=trace)
