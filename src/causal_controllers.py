"""Matched supervisory adapters, known deterministic tariffs only.

No dataset loading occurs here. All policies see the immutable causal Context.
THB/OMG are declared model/cadence adaptations, not reproduced theorems.
MPC and SCR share this exact nominal proposal. The executor alone adds SCR.
"""
from dataclasses import dataclass
import math
import numpy as np
from available_price_control import thb, omg
from causal_replay import PowerCommand, pfg_command
from grid_model import Interval, transition
from grid_dp import action_for_transition
from shadow_reserve_v2 import choose_index, SCORE_TOL


def observed_ratio(ctx):
    """Integrate only completed observed left-hold segments, including warmup."""
    load=pv=0.
    for i,sample in enumerate(ctx.history):
        end=ctx.history[i+1].time if i+1<len(ctx.history) else ctx.time
        if sample.time>ctx.time or end>ctx.time:raise ValueError('future observation')
        hours=(end-sample.time)/3600
        load+=sample.load_kw*hours;pv+=sample.pv_kw*hours
    return min(1.,max(0.,pv/load)) if load>0 else 0.


def known_price(ctx,time):
    for band in ctx.price_bands:
        if band.start<=time<band.end:
            if band.low!=band.high:
                raise ValueError('adapter requires a tariff known in advance; bounds are not forecasts')
            return band.low
    raise ValueError('missing known tariff')


def prediction(ctx,seconds):
    """First block: current point persistence. Later blocks: past-five-minute mean.

    Every block ends at a supervisor time, tariff boundary or episode end.
    This predicts held commands, never reads a realized interval energy total.
    """
    if ctx.episode_end-ctx.time>86400+1e-9:raise ValueError('episode horizon exceeds 24 hours')
    if not math.isclose(known_price(ctx,ctx.time),ctx.price,abs_tol=1e-12):
        raise ValueError('current price disagrees with known tariff')
    out=[];t=ctx.time;first=True
    while t<ctx.episode_end-1e-9:
        boundary=min(b.end for b in ctx.price_bands if b.end>t)
        end=min(ctx.episode_end,ctx.time+(math.floor((t-ctx.time)/seconds+1e-9)+1)*seconds,boundary)
        dt=(end-t)/3600
        load,pv=(ctx.load_kw,ctx.pv_kw) if first else (ctx.mean_load_kw,ctx.mean_pv_kw)
        out.append(Interval(load*dt,pv*dt,known_price(ctx,t),dt))
        t=end;first=False
    return out


def continuation_table(xs,b,states,terminal_value):
    """SOC-trellis continuation; interpolation only scores the current action."""
    levels=np.linspace(0.,b.capacity,states)
    delta=levels[None,:]-levels[:,None]
    charge=np.maximum(delta,0.)/b.eta_charge
    discharge=np.maximum(-delta,0.)*b.eta_discharge
    value=-terminal_value*levels
    for x in reversed(xs):
        valid=(charge<=b.charge_kw*x.dt+1e-10)&(discharge<=min(x.load,b.discharge_kw*x.dt)+1e-10)
        grid=np.maximum(0.,x.load-discharge+charge-x.pv) if x.price>=0 else x.load-discharge+charge
        costs=x.price*grid+value[None,:]
        costs[~valid]=np.inf
        value=np.min(costs,axis=1)
    return levels,value


def nominal_action(x,s,b,future,states,terminal_value):
    levels,value=continuation_table(future,b,states,terminal_value)
    lower=max(0.,s-min(x.load,b.discharge_kw*x.dt)/b.eta_discharge)
    upper=min(b.capacity,s+b.charge_kw*x.dt*b.eta_charge)
    kink=s+(x.pv-x.load)*(b.eta_charge if x.pv>=x.load else 1/b.eta_discharge)
    targets=np.unique(np.clip(np.r_[levels,lower,upper,s,kink],lower,upper))
    actions=[action_for_transition(x,s,float(e),b) for e in targets]
    flows=[transition(x,s,a,b) for a in actions]
    scores=[f.cost+float(np.interp(f.end_soc,levels,value)) for f in flows]
    return actions[choose_index(actions,scores,x,s,b)]


@dataclass(frozen=True)
class Controller:
    name: str
    seconds: float=300.
    states: int=501
    terminal_value: float=0.

    def __post_init__(self):
        if self.name not in ('PFG','THB','OMG','MPC'):raise ValueError('unknown controller')
        if not math.isfinite(self.seconds) or self.seconds<=0:raise ValueError('invalid cadence')
        if isinstance(self.states,bool) or not isinstance(self.states,int) or self.states<2:raise ValueError('invalid SOC grid')
        if not math.isfinite(self.terminal_value) or self.terminal_value<0:raise ValueError('invalid terminal value')

    def __call__(self,ctx,b):
        xs=prediction(ctx,self.seconds)
        if self.name=='PFG':return pfg_command(ctx,b)
        x=xs[0]
        # Global tariff extrema are design inputs, including already elapsed bands.
        lows=[a.low for a in ctx.price_bands];highs=[a.high for a in ctx.price_bands]
        if any(a.low!=a.high for a in ctx.price_bands):raise ValueError('unknown tariff')
        if self.name=='THB':
            positive=[p for p in lows if p>0]
            a=thb(x,ctx.soc,b,min(positive) if positive else 1e-4,max(max(highs),1e-4),observed_ratio(ctx))
        elif self.name=='OMG':
            # Frozen OMG formula divides by a derivative range. A zero-price
            # episode has no economic preference: choose physically idle SOC.
            a=(action_for_transition(x,ctx.soc,ctx.soc,b) if min(lows)==max(highs)==0
               else omg(x,ctx.soc,b,min(lows),max(highs)))
        else:a=nominal_action(x,ctx.soc,b,xs[1:],self.states,self.terminal_value)
        return PowerCommand(a.direct/x.dt,a.charge/x.dt,a.discharge/x.dt,a.grid_charge/x.dt)


def run_matched(samples,model,name,*,seconds=300.,states=501,initial_soc=0.,
                certificate=None,budget=0.,feedback_seconds=30.):
    """The four baselines are unshielded; only the MPC proposal can form SCR."""
    from causal_replay_v2 import replay
    if certificate is not None and name!='MPC':raise ValueError('SCR must share the nominal MPC proposal')
    for band in model.bands[:-1]:
        phase=(band.end-model.start)/seconds
        if abs(phase-round(phase))>1e-9:
            raise ValueError('tariff changes must align with supervisory decisions in this adapter')
    policy=Controller(name,seconds,states,model.v)
    baseline=Controller('PFG',seconds,states,model.v)
    result=replay(samples,model,policy,baseline,kind=certificate,budget=budget,
                  initial_soc=initial_soc,supervisory_seconds=seconds,feedback_seconds=feedback_seconds)
    result.update(controller=name if certificate is None else 'SCR-'+certificate,
                  states=states,supervisory_seconds=seconds,terminal_value=model.v,
                  score_tolerance=SCORE_TOL,data_provenance='caller supplied; not verified by adapter')
    return result
