"""Explicit grid-enabled revisions of all active physical controller families.

Grid is an allowed charging source, not a mandatory action at positive prices.
Archived PV-only implementations and guarantees retain their original meanings.
"""
import math,time,warnings,json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from grid_model import Action,transition
from physical_simulator import self_consumption,greedy_charge_first
from multistep_allocation import FRACTIONS,packet_value_lower as old_bound
from cumulative_allocation import packet_value_lower as cumulative_bound

NAMES=('GCF','PFG','PFG-S','TRG','ARG','C24','C24-TV','PF-C24','P24','P48','PF-OPT','OPT','PA','MGA-0','MGA-1','MGA-C')

def pfg(x,soc,b):
    a=self_consumption(x,soc,b)
    if x.price>=0:return Action(a.direct,a.charge,a.discharge)
    # Preserve direct-PV priority; avoid discharge at a negative price, then
    # fill unused charging power/headroom from the grid after PV charging.
    room=min(b.charge_kw*x.dt,max(0,b.capacity-soc)/b.eta_charge)
    return Action(a.direct,a.charge,0,max(0,room-a.charge))

def price_greedy(x,soc,b):
    if x.price>=0:return pfg(x,soc,b)
    # Buying negative-price energy is preferable to displacing it with PV.
    return Action(0,0,0,min(b.charge_kw*x.dt,max(0,b.capacity-soc)/b.eta_charge))

def gcf(x,soc,b):
    a=greedy_charge_first(x,soc,b)
    grid=0. if x.price>=0 else max(0,min(b.charge_kw*x.dt,max(0,b.capacity-soc)/b.eta_charge)-a.charge)
    return Action(a.direct,a.charge,a.discharge,grid)

def reserve_action(x,soc,b,hold_target,charge_target):
    if x.price<0:return price_greedy(x,soc,b)
    a=pfg(x,soc,b)
    discharge=min(a.discharge,max(0,soc-hold_target)*b.eta_discharge)
    pv_charge=a.charge;grid_charge=0.
    if charge_target>soc+b.eta_charge*pv_charge+1e-10:
        discharge=0.
        grid_charge=max(0,min(b.charge_kw*x.dt-pv_charge,(b.capacity-soc)/b.eta_charge-pv_charge,(charge_target-soc)/b.eta_charge-pv_charge))
    return Action(a.direct,pv_charge,discharge,grid_charge)

def tariff_reserve(x,soc,b,history,tariff,tou):
    prices=list(tou.values()) if tariff=='TimeOfDay' else [y.price for _,y in history[-168:]]
    if not prices:raise ValueError('Missing causal price history')
    threshold=(min(prices)+max(prices))/2 if tariff=='TimeOfDay' else sorted(prices)[math.ceil(.75*len(prices))-1]
    reserve=.5*b.capacity if x.price<threshold else 0.
    target=reserve if max(prices)*b.eta_charge*b.eta_discharge>x.price+1e-12 else 0.
    return reserve_action(x,soc,b,reserve,target)

def backward_target(predictions,b,price_threshold):
    reserve=0.
    for y in reversed(predictions):
        if y.pv>=y.load:reserve=max(0,reserve-min(y.pv-y.load,b.charge_kw*y.dt)*b.eta_charge)
        elif y.price>price_threshold+1e-12:reserve=min(b.capacity,reserve+min(y.load-y.pv,b.discharge_kw*y.dt)/b.eta_discharge)
    return reserve

def adaptive_reserve(x,soc,b,predictions):
    hold=backward_target(predictions,b,max(0,x.price))
    buy=backward_target(predictions,b,max(0,x.price)/(b.eta_charge*b.eta_discharge))
    return reserve_action(x,soc,b,hold,buy)

def action_to_soc(x,soc,target,b):
    """Feasible minimal-import action to a target SOC at nonnegative price.
    Direct PV first is an explicit source-allocation tie rule while charging.
    """
    if not -1e-9<=target<=b.capacity+1e-9:raise ValueError('Invalid target')
    target=max(0.,min(b.capacity,target))
    if target<soc:
        discharge=(soc-target)*b.eta_discharge
        a=Action(min(x.pv,max(0,x.load-discharge)),0,discharge)
    else:
        total=(target-soc)/b.eta_charge;direct=min(x.pv,x.load)
        pv=min(max(0,x.pv-direct),total)
        a=Action(direct,pv,0,max(0,total-pv))
    transition(x,soc,a,b)
    return a

def guarded_action(name,x,soc,b,envelopes,remaining_budget=1.):
    """Same packet enclosures; broader one-action SOC-increase candidate class.
    PA requires guaranteed emptying at the next step. MGA retains greedy future
    continuation. Guarantees below are only for nonnegative realized episodes.
    PA has no spending cap; MGA-0/1/C use the specified nonreplenishing cap.
    """
    if name not in ('PA','MGA-0','MGA-1','MGA-C'):raise ValueError(name)
    if name!='PA' and (not math.isfinite(remaining_budget) or remaining_budget<0):raise ValueError('Invalid budget')
    base=pfg(x,soc,b);bf=transition(x,soc,base,b)
    info=dict(packet=0.,extra_spend=0.,predicted_lower_gain=0.,certificate_applicable=False)
    if x.price<0 or not envelopes:return base,remaining_budget,info
    e0=max(0.,min(b.capacity,bf.end_soc));emax=min(b.capacity,soc+b.eta_charge*b.charge_kw*x.dt)
    if name=='PA':
        box=envelopes[0]
        emax=min(emax,min(max(0,box.net_low),b.discharge_kw*box.dt)/b.eta_discharge)
    if emax<=e0+1e-10:return base,remaining_budget,info
    def spend(e):return transition(x,soc,action_to_soc(x,soc,e,b),b).cost-bf.cost
    budget=math.inf if name=='PA' else remaining_budget
    if spend(emax)>budget+1e-12:
        low,high=e0,emax
        for _ in range(60):
            mid=(low+high)/2
            if spend(mid)<=budget:low=mid
            else:high=mid
        emax=low
    best=base;best_gain=0.;best_spend=best_packet=0.
    for frac in FRACTIONS:
        target=e0+frac*(emax-e0);packet=target-e0
        a=action_to_soc(x,soc,target,b);s=max(0,transition(x,soc,a,b).cost-bf.cost)
        if s>budget+1e-10:continue
        if name=='PA':value=envelopes[0].price_low*b.eta_discharge*packet
        else:value=(cumulative_bound if name=='MGA-C' else old_bound)(e0,packet,envelopes,b)
        gain=value-s
        if gain>best_gain+1e-10:best,best_gain,best_spend,best_packet=a,gain,s,packet
    if best_packet==0:return base,remaining_budget,info
    left=remaining_budget if name=='PA' else max(0,remaining_budget-best_spend)
    return best,left,dict(packet=best_packet,extra_spend=best_spend,predicted_lower_gain=best_gain,certificate_applicable=True)

def optimize(xs,b,initial_soc=0.,terminal_value=0.,pv_first=False,allow_grid_charging=True,time_limit=120):
    """Continuous grid/PV flows and binary battery mode; zero export.
    Lexicographic primary ties at the first action: maximize next SOC, maximize
    current PV use, maximize direct PV. These three quantities specify the first
    physical action (within numeric tolerances); residual future ties are unused
    in a receding-horizon policy. Hierarchy fixed before the new pilot results.
    """
    n=len(xs)
    if not n:raise ValueError('Empty horizon')
    if not math.isfinite(terminal_value) or terminal_value<0:raise ValueError('Invalid terminal credit')
    if not math.isfinite(initial_soc) or not -1e-8<=initial_soc<=b.capacity+1e-8:raise ValueError('Invalid initial state')
    initial_projection=abs(initial_soc-max(0.,min(b.capacity,initial_soc)))
    initial_soc=max(0.,min(b.capacity,initial_soc))
    # direct, PV charge, grid charge, discharge, SOC, binary mode.
    obj=np.zeros(6*n);upper=np.zeros(6*n);floor=np.zeros(6*n);integer=np.zeros(6*n)
    mat=lil_matrix((5*n,6*n));lo=np.full(5*n,-np.inf);hi=np.zeros(5*n)
    for t,x in enumerate(xs):
        obj[t]=obj[3*n+t]=-x.price;obj[2*n+t]=x.price
        upper[t]=min(x.pv,x.load);upper[n+t]=min(x.pv,b.charge_kw*x.dt)
        upper[2*n+t]=b.charge_kw*x.dt if allow_grid_charging else 0.
        upper[3*n+t]=min(x.load,b.discharge_kw*x.dt);upper[4*n+t]=b.capacity;upper[5*n+t]=1;integer[5*n+t]=1
        if pv_first:floor[t]=upper[t]
        mat[t,t]=mat[t,n+t]=1;hi[t]=x.pv
        mat[n+t,t]=mat[n+t,3*n+t]=1;hi[n+t]=x.load
        r=2*n+t;mat[r,4*n+t]=1;mat[r,n+t]=mat[r,2*n+t]=-b.eta_charge;mat[r,3*n+t]=1/b.eta_discharge
        if t:mat[r,4*n+t-1]=-1
        lo[r]=hi[r]=initial_soc if t==0 else 0.
        mat[3*n+t,n+t]=mat[3*n+t,2*n+t]=1;mat[3*n+t,5*n+t]=-b.charge_kw*x.dt
        mat[4*n+t,3*n+t]=1;mat[4*n+t,5*n+t]=b.discharge_kw*x.dt;hi[4*n+t]=b.discharge_kw*x.dt
    obj[5*n-1]=-terminal_value
    constraints=[LinearConstraint(mat.tocsc(),lo,hi)];start=time.perf_counter();stages=[]
    def solve(cost):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            r=milp(cost,integrality=integer,bounds=Bounds(floor,upper),constraints=constraints,options={'time_limit':time_limit,'mip_rel_gap':1e-9,'threads':1,'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
        if r.status!=0 or r.x is None:raise RuntimeError('Grid MILP failed: '+json.dumps(dict(message=r.message,stage=len(stages),initial_soc=initial_soc,terminal_value=terminal_value,pv_first=pv_first,battery=b.__dict__,intervals=[x.__dict__ for x in xs])))
        stages.append(dict(status=int(r.status),objective=float(r.fun),dual_bound=float(r.mip_dual_bound)))
        return r
    primary=solve(obj);result=primary
    secondary=[]
    for entries in [{4*n:-1.},{0:-1.,n:-1.},{0:-1.}]:
        v=np.zeros(6*n)
        for i,z in entries.items():v[i]=z
        secondary.append(v)
    previous=obj
    for v in secondary:
        constraints.append(LinearConstraint(previous[None,:],-np.inf,float(previous@result.x)+1e-7))
        result=solve(v);previous=v
    actions=[];soc=initial_soc;cost=projection=0.
    for t,x in enumerate(xs):
        raw=[float(result.x[k*n+t]) for k in (0,1,2,3)]
        direct=max(0,min(raw[0],x.pv,x.load))
        charge=max(0,min(raw[1],x.pv-direct,b.charge_kw*x.dt,(b.capacity-soc)/b.eta_charge))
        grid=max(0,min(raw[2],b.charge_kw*x.dt-charge,(b.capacity-soc)/b.eta_charge-charge))
        discharge=max(0,min(raw[3],x.load-direct,b.discharge_kw*x.dt,soc*b.eta_discharge))
        if result.x[5*n+t]>=.5:discharge=0.
        else:charge=grid=0.
        a=Action(direct,charge,discharge,grid);change=max(abs(v-w) for v,w in zip(raw,[direct,charge,grid,discharge]));projection=max(projection,change)
        if change>1e-6:raise RuntimeError('Material grid optimizer flow projection')
        f=transition(x,soc,a,b,allow_grid_charging);soc=f.end_soc;cost+=f.cost;actions.append(a)
    constant=sum(x.load*x.price for x in xs);adjusted=cost-terminal_value*soc;dual=float(primary.mip_dual_bound)+constant
    if abs(adjusted-(float(obj@result.x)+constant))>1e-6 or abs(adjusted-dual)>1e-6:raise RuntimeError('Grid optimizer objective certification failed')
    return actions,dict(status=0,cost=cost,objective_cost=adjusted,dual_bound=dual,primary_cost=float(primary.fun)+constant,max_projection=projection,initial_soc_projection=initial_projection,final_soc=soc,terminal_value=terminal_value,solver_seconds=time.perf_counter()-start,solves=len(stages),stages=stages,tie_rule='max_first_soc,max_first_pv_use,max_first_direct',grid_charging_allowed=allow_grid_charging)
