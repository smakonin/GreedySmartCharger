"""Household execution extension: held hourly requests and compact decision logs.

Frozen interval_meter.py physics and admission logic copied without alteration.

Only completed meter intervals enter supervisors. Upcoming energies are private
simulator inputs reconstructed as constant power. No measured fast telemetry,
field communication transport or deployment validation is claimed.
"""
from dataclasses import asdict, dataclass
import math
import time
from available_price_control import thb, omg
from causal_controllers import nominal_action
from causal_replay import PowerCommand, Sample, plan_pair, select_plan, rate, _snap
from executed_liability import ExecutedLiability, PriceBand
from grid_model import Action, Interval, transition

STEP = 1800
HISTORY = 28 * 48


@dataclass(frozen=True)
class MeterReading:
    start: float
    end: float
    load_kwh: float
    pv_kwh: float
    ready_at: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.start, self.end, self.load_kwh, self.pv_kwh, self.ready_at)):
            raise ValueError('nonfinite meter record')
        if self.end - self.start != STEP or min(self.load_kwh, self.pv_kwh) < 0 or self.ready_at < self.end:
            raise ValueError('meter intervals require half-hour energies released no earlier than their end')


@dataclass(frozen=True)
class MeterContext:
    time: float
    soc: float
    history: tuple
    episode_end: float
    price_bands: tuple


@dataclass(frozen=True)
class BatteryRequest:
    """AC kW, positive to charge; source mode is an explicit control decision."""
    kw: float = 0.
    source: str = 'pv_first'
    local_greedy: bool = False

    def __post_init__(self):
        if not math.isfinite(self.kw) or self.source not in ('pv_first', 'grid_only'):
            raise ValueError('invalid battery request')
        if self.local_greedy and self.kw != 0:
            raise ValueError('local greedy mode has no fixed battery setpoint')


def known_price(bands, t):
    for band in bands:
        if band.start <= t < band.end:
            if band.low != band.high:
                raise ValueError('interval adapter requires a tariff known in advance')
            return band.low
    raise ValueError('missing tariff at requested time')


def validate_history(ctx):
    if len(ctx.history) != HISTORY:
        raise ValueError('exactly 28 complete days of causal history required')
    for i, r in enumerate(ctx.history):
        if r.start != ctx.time - (HISTORY-i)*STEP or r.end > ctx.time or r.ready_at > ctx.time:
            raise ValueError('missing, stale or unreleased interval in history')


def forecast(ctx, method='weekday'):
    """First and all later intervals are forecasts, including the current one."""
    validate_history(ctx)
    if method not in ('weekday', 'previous_day', 'persistence'):
        raise ValueError('unknown forecast')
    by_start = {r.start:r for r in ctx.history}
    end = min(ctx.episode_end, ctx.time + 86400)
    if end <= ctx.time or (end-ctx.time) % STEP:
        raise ValueError('invalid forecast endpoint')
    xs=[]; origins=[]
    for k in range(int((end-ctx.time)/STEP)):
        target = ctx.time+k*STEP
        starts = ([target-7*86400*j for j in range(1,5)] if method == 'weekday' else
                  [target-86400] if method == 'previous_day' else [ctx.time-STEP])
        records=[by_start[t] for t in starts]
        xs.append(Interval(math.fsum(r.load_kwh for r in records)/len(records),
                           math.fsum(r.pv_kwh for r in records)/len(records),
                           known_price(ctx.price_bands,target), .5))
        origins.append(dict(target=target,source_ends=[r.end for r in records],
                            latest_ready_at=max(r.ready_at for r in records)))
    return xs, origins


def request_from_action(action, dt, price):
    return BatteryRequest((action.charge+action.grid_charge-action.discharge)/dt,
                          'grid_only' if price < 0 else 'pv_first')


def local_command(request, load, pv, price, soc, b):
    """Common source allocation at each profile/SOC event; never a forecast input.

    A fixed setpoint gives battery power priority on discharge. Positive-price
    charging serves load with PV first; negative-price economic controllers can
    explicitly curtail PV. PFG is an event-local rule with PV priority throughout.
    """
    if request.local_greedy:
        direct=min(load,pv)
        pvc=min(max(0.,pv-direct),b.charge_kw) if soc < b.capacity else 0.
        if price < 0:
            return PowerCommand(direct,pvc,0.,b.charge_kw-pvc if soc < b.capacity else 0.)
        discharge=min(max(0.,load-direct),b.discharge_kw) if soc > 0 else 0.
        return PowerCommand(direct,pvc,discharge)
    charge=min(max(0.,request.kw),b.charge_kw) if soc < b.capacity else 0.
    discharge=min(max(0.,-request.kw),b.discharge_kw,load) if soc > 0 else 0.
    if request.source == 'grid_only':
        return PowerCommand(0.,0.,discharge,charge)
    direct=min(pv,max(0.,load-discharge))
    pvc=min(max(0.,pv-direct),charge)
    return PowerCommand(direct,pvc,discharge,max(0.,charge-pvc))


@dataclass(frozen=True)
class MeterController:
    name: str
    states: int = 501
    terminal_value: float = 0.
    forecast_method: str = 'weekday'

    def __post_init__(self):
        if self.name not in ('PFG','THB','OMG','MPC'):
            raise ValueError('unknown controller')
        if isinstance(self.states,bool) or not isinstance(self.states,int) or self.states < 2:
            raise ValueError('invalid SOC grid')
        if not math.isfinite(self.terminal_value) or self.terminal_value < 0:
            raise ValueError('invalid terminal value')
        if self.forecast_method not in ('weekday','previous_day','persistence'):
            raise ValueError('unknown forecast')

    def decide(self, ctx, b):
        decision_start=time.perf_counter()
        validate_history(ctx)
        if self.name == 'PFG':
            request=BatteryRequest(local_greedy=True)
            return request, {'forecast_seconds':0.,'decision_seconds':time.perf_counter()-decision_start,'forecast':[]}
        started=time.perf_counter();xs,origins=forecast(ctx,self.forecast_method)
        forecast_seconds=time.perf_counter()-started;started=time.perf_counter()
        x=xs[0];prices=[band.low for band in ctx.price_bands]
        if any(band.low != band.high for band in ctx.price_bands):
            raise ValueError('unknown tariff schedule')
        if self.name == 'THB':
            load=math.fsum(r.load_kwh for r in ctx.history)
            rho=min(1.,math.fsum(r.pv_kwh for r in ctx.history)/load) if load else 0.
            positive=[p for p in prices if p>0]
            a=thb(x,ctx.soc,b,min(positive) if positive else 1e-4,max(max(prices),1e-4),rho)
        elif self.name == 'OMG':
            # The frozen formula has a zero derivative range for an all-zero tariff.
            a=Action(min(x.load,x.pv),0.,0.) if min(prices)==max(prices)==0 else omg(x,ctx.soc,b,min(prices),max(prices))
        else:
            a=nominal_action(x,ctx.soc,b,xs[1:],self.states,self.terminal_value)
        request=request_from_action(a,x.dt,x.price)
        return request,dict(forecast_seconds=forecast_seconds,decision_seconds=time.perf_counter()-decision_start-forecast_seconds,
                            forecast=[dict(**o,load_kwh=y.load,pv_kwh=y.pv,price=y.price) for y,o in zip(xs,origins)],
                            nominal_action=asdict(a))


def tariff_bands(days, kind='tou'):
    if isinstance(days,bool) or not isinstance(days,int) or days <= 0 or kind not in ('flat','tou','signed'):
        raise ValueError('invalid synthetic tariff')
    bands=[]
    for t in range(0, days*86400, STEP):
        hour=(t%86400)/3600
        p=.15 if kind!='flat' and 16<=hour<21 else .08
        if kind=='signed' and 1<=hour<6:p=-.05
        if bands and bands[-1].low==p:
            last=bands.pop();bands.append(PriceBand(last.start,t+STEP,p,p))
        else:bands.append(PriceBand(t,t+STEP,p,p))
    return tuple(bands)


def replay_meter(readings, model, controller, *, certificate=None, budget=0., initial_soc=0.,
                 trace_decisions=True, supervisory_steps=1):
    """One continuous episode; rolling forecast horizon never resets accounting.

    Source readings may include unavailable future observations; supervisors only
    get an immutable completed 28-day slice. Local inputs use private interval
    reconstruction. The analytical guard checks every prefix to the next event,
    with a maximum 30-minute commitment and immediate profile/SOC interrupts.
    """
    if supervisory_steps not in (1,2):raise ValueError('supervision must be half-hourly or hourly')
    if certificate not in (None,'global','copy','repair') or (certificate and controller.name!='MPC'):
        raise ValueError('only the matched MPC proposal can be shielded')
    if not math.isfinite(budget) or budget<0 or not math.isfinite(initial_soc) or not 0<=initial_soc<=model.b.capacity:
        raise ValueError('invalid episode accounting')
    if model.v!=controller.terminal_value:
        raise ValueError('planner/certificate endpoint value mismatch')
    if any(b.low!=b.high for b in model.bands):raise ValueError('unknown tariff')
    if (model.end-model.start)%STEP or any((b.end-model.start)%STEP for b in model.bands):
        raise ValueError('unaligned tariff or episode endpoint')
    if any(a.end!=b.start for a,b in zip(readings,readings[1:])):
        raise ValueError('meter gaps, duplicates or out-of-order intervals')
    expected_start=model.start-HISTORY*STEP
    if not readings or readings[0].start!=expected_start or readings[-1].end!=model.end:
        raise ValueError('exact history and episode coverage required')
    b=model.b;s=z=initial_soc;cost=basecost=0.;regret=0.
    trace=[];decisions=[];intervals=[];times=[];forecast_times=[];local_times=[]
    total_start=time.perf_counter();max_balance=0.
    for index in range(HISTORY,len(readings)):
        meter=readings[index];t=meter.start
        # Delayed readings stay unavailable, causing an explicit input failure instead
        # of interpolation. The current meter itself is never in this tuple.
        history=tuple(readings[index-HISTORY:index])
        if any(r.ready_at>t for r in history):raise ValueError('unreleased meter history')
        if (index-HISTORY)%supervisory_steps==0:
            ctx=MeterContext(t,s,history,model.end,model.bands)
            request,meta=controller.decide(ctx,b)
            if not isinstance(request,BatteryRequest):raise TypeError('policy must return BatteryRequest')
            times.append(meta['decision_seconds']);forecast_times.append(meta['forecast_seconds'])
            decision=dict(time=t,soc=s,shadow_soc=z,regret_before=regret,
                          history_start=history[0].start,latest_meter_end=history[-1].end,
                          latest_meter_ready_at=history[-1].ready_at,
                          planning_end=min(model.end,t+86400),true_episode_end=model.end,
                          request=asdict(request),**meta)
            if trace_decisions:decisions.append(decision)
        load=meter.load_kwh/.5;pv=meter.pv_kwh/.5;p=known_price(model.bands,t)
        row=dict(start=t,end=meter.end,load_kwh=meter.load_kwh,pv_kwh=meter.pv_kwh,price=p,
                 cost=0.,shadow_cost=0.,import_kwh=0.,shadow_import_kwh=0.,charge_ac_kwh=0.,
                 discharge_ac_kwh=0.,curtailment_kwh=0.,loss_kwh=0.,start_soc=s,start_shadow_soc=z)
        local_start=time.perf_counter()
        for _ in range(32):
            if t>=meter.end-1e-9:break
            cmd=local_command(request,load,pv,p,s,b)
            shadow=local_command(BatteryRequest(local_greedy=True),load,pv,p,z,b)
            proposed=plan_pair(cmd,shadow,load,pv,s,z,b,meter.end-t)
            # Refresh source allocation at the earliest actual/shadow SOC event.
            commitment=proposed[0].duration
            if certificate is None:
                plan=proposed;info=dict(fallback=False,branch=None,proposed_peak=None,accepted_peak=None)
            else:
                sample=Sample(t,load,pv,p,t)  # synthesized local signal, not a source receipt
                plan,info=select_plan(cmd,shadow,sample,s,z,model,certificate,t,commitment,regret,budget)
            seg=plan[0];dt=seg.duration
            ap,bp=seg.power,seg.shadow_power
            x=Interval(load*dt/3600,pv*dt/3600,p,dt/3600)
            action=lambda q:Action(q.direct*dt/3600,q.pv_charge*dt/3600,q.discharge*dt/3600,q.grid_charge*dt/3600)
            a=action(ap);sf=transition(x,z,action(bp),b);af=transition(x,s,a,b)
            peak=(None if certificate is None else model.max_along(certificate,t,dt,s,rate(ap,b),z,rate(bp,b),regret,(af.cost-sf.cost)/dt))
            before_s,before_z=s,z;s=_snap(af.end_soc,b);z=_snap(sf.end_soc,b)
            if info['branch']=='repair' and abs(s-z)<1e-11:s=z
            cost+=af.cost;basecost+=sf.cost;regret=cost-basecost
            phi=None if certificate is None else model.value(certificate,t+dt,s,z)
            if certificate is not None and (peak>budget+1e-8 or regret+phi>budget+1e-8):
                raise AssertionError('executed interval prefix guarantee failed')
            trace.append(dict(start=t,end=t+dt,soc_before=before_s,shadow_soc_before=before_z,soc=s,shadow_soc=z,
                              power=asdict(ap),shadow_power=asdict(bp),grid_kwh=af.grid,shadow_grid_kwh=sf.grid,
                              cost=af.cost,shadow_cost=sf.cost,regret=regret,liability=phi,segment_peak=peak,**info))
            for key,value in [('cost',af.cost),('shadow_cost',sf.cost),('import_kwh',af.grid),('shadow_import_kwh',sf.grid),
                               ('charge_ac_kwh',a.charge+a.grid_charge),('discharge_ac_kwh',a.discharge),
                               ('curtailment_kwh',af.curtailed),('loss_kwh',af.loss)]:row[key]+=value
            max_balance=max(max_balance,abs(af.balance_residual),abs(sf.balance_residual));t+=dt
        else:raise AssertionError('too many local physical events')
        row.update(end_soc=s,end_shadow_soc=z,regret_after=regret,peak_interval_average_kw=row['import_kwh']/.5)
        intervals.append(row);local_times.append(time.perf_counter()-local_start)
    terminal=regret-model.v*(s-z)
    if certificate is not None and terminal>budget+1e-8:raise AssertionError('true-endpoint guarantee failed')
    import numpy as np
    timing=lambda a:dict(total=math.fsum(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)))
    charge=math.fsum(r['charge_ac_kwh'] for r in intervals);discharge=math.fsum(r['discharge_ac_kwh'] for r in intervals)
    return dict(controller=controller.name if certificate is None else 'SCR-'+certificate,certificate=certificate,
        budget=budget,initial_soc=initial_soc,final_soc=s,shadow_final_soc=z,cost=cost,shadow_cost=basecost,
        objective=cost-model.v*(s-initial_soc),shadow_objective=basecost-model.v*(z-initial_soc),
        regret=regret,terminal_adjusted_difference=terminal,max_balance_residual=max_balance,
        efc=(b.eta_charge*charge+discharge/b.eta_discharge)/(2*b.capacity),
        peak_half_hour_import_kw=max(r['peak_interval_average_kw'] for r in intervals),
        max_prefix_exposure=max((r['segment_peak'] for r in trace if r['segment_peak'] is not None),default=None),
        decisions=decisions,intervals=intervals,trace=trace,
        timing_seconds=dict(supervisor=timing(times),forecast=timing(forecast_times),local_with_independent_shadow=timing(local_times),
                            total=time.perf_counter()-total_start),
        provenance='caller supplied; local power and instantaneous feedback are simulated',
        endpoint=model.end,states=controller.states,forecast_method=controller.forecast_method)
