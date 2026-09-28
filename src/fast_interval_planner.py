"""Range-minimum acceleration of the frozen uniform SOC-trellis recurrence.

No new objective or action family. Future values use exact index windows for each
piecewise-affine import-cost branch. First actions retain the 1e-7 score face.
"""
import math
import numpy as np
from scipy.ndimage import minimum_filter1d
from grid_dp import action_for_transition
from shadow_reserve_v2 import SCORE_TOL


def range_min(v, lo, hi):
    n=len(v);lo=max(-(n-1),lo);hi=min(n-1,hi)
    if lo>hi:return np.full(n,np.inf)
    width=hi-lo+1
    padded=np.pad(v,(n,n),constant_values=np.inf)
    filtered=minimum_filter1d(padded,size=width,mode='constant',cval=np.inf)
    return filtered[np.arange(n)+n+lo+width//2]


def continuation(xs,b,states,terminal_value):
    levels=np.linspace(0.,b.capacity,states);spacing=b.capacity/(states-1)
    value=-terminal_value*levels
    for x in reversed(xs):
        rcharge=min(states-1,math.floor(b.eta_charge*(b.charge_kw*x.dt+1e-10)/spacing))
        rdis=min(states-1,math.floor((min(x.load,b.discharge_kw*x.dt)+1e-10)/b.eta_discharge/spacing))
        net=x.load-x.pv
        out=np.full(states,np.inf)
        for lo,hi,slope in ((0,rcharge,1/b.eta_charge),(-rdis,0,b.eta_discharge)):
            if x.price<0:
                candidate=range_min(value+x.price*slope*levels,lo,hi)+x.price*(x.load-slope*levels)
            else:
                # Net import is max(0, net + slope * spacing * offset).
                zero_to=math.floor(-net/(slope*spacing))
                zero=range_min(value,lo,min(hi,zero_to))
                active=range_min(value+x.price*slope*levels,max(lo,zero_to+1),hi)+x.price*(net-slope*levels)
                candidate=np.minimum(zero,active)
            out=np.minimum(out,candidate)
        value=out
    return levels,value


def nominal_action(x,s,b,future,states,terminal_value):
    levels,value=continuation(future,b,states,terminal_value)
    return first_action(x,s,b,levels,value)


def first_action(x,s,b,levels,value):
    lower=max(0.,s-min(x.load,b.discharge_kw*x.dt)/b.eta_discharge)
    upper=min(b.capacity,s+b.charge_kw*x.dt*b.eta_charge)
    kink=s+(x.pv-x.load)*(b.eta_charge if x.pv>=x.load else 1/b.eta_discharge)
    targets=np.unique(np.clip(np.r_[levels,lower,upper,s,kink],lower,upper))
    delta=targets-s;charge=np.maximum(delta,0.)/b.eta_charge;discharge=np.maximum(-delta,0.)*b.eta_discharge
    grid=x.load-discharge+charge
    if x.price>=0:grid=np.maximum(0.,grid-x.pv)
    scores=x.price*grid+np.interp(targets,levels,value)
    index=np.flatnonzero(scores<=np.min(scores)+SCORE_TOL)[-1]
    return action_for_transition(x,s,float(targets[index]),b)
