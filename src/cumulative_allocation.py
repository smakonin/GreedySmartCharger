"""MGA-C: cumulative packet accounting with unchanged MGA uncertainty envelopes.

The bound minimizes revenue over a relaxation of physically possible discharge
sequences. Prefix packet conservation retains credit for uncertain discharge time.
"""
import heapq
import math
from physical_simulator import Action, self_consumption
from multistep_allocation import soc_map, FRACTIONS


def prefix_minimum_revenue(lower, upper, prefix_lower, prices):
    """Solve min sum p[t]*d[t] with l<=d<=u and cumulative d>=D.

    Nonnegative prices; nested prefix constraints. Reserve per-step lower flows,
    then fill each outstanding deadline using the cheapest residual capacity
    available on or before that deadline. Exchange across nested deadlines proves
    this greedy allocation optimal for this relaxation. Units are battery kWh.
    """
    if not len(lower) == len(upper) == len(prefix_lower) == len(prices):
        raise ValueError('Mismatched constraint lengths')
    fixed_prefix = assigned = cost = 0.0
    heap = []
    for t, (lo, hi, target, price) in enumerate(zip(lower, upper, prefix_lower, prices)):
        if not all(math.isfinite(v) for v in (lo,hi,target,price)) or min(lo,hi,target,price)<0 or hi+1e-10<lo:
            raise ValueError('Invalid nonnegative flow constraints')
        fixed_prefix += lo
        cost += price*lo
        if hi > lo:
            heapq.heappush(heap,(price,t,hi-lo))
        needed = max(0.0,target-fixed_prefix-assigned)
        while needed > 0:
            if not heap:
                if needed > 1e-9:
                    raise ValueError('Inconsistent prefix enclosure')
                break  # float roundoff only; under-accounting keeps a lower bound
            p, index, available = heapq.heappop(heap)
            take = min(available,needed)
            cost += p*take
            assigned += take
            needed -= take
            if available > take:
                heapq.heappush(heap,(p,index,available-take))
    return cost


def packet_constraints(base_soc, packet, envelopes, b):
    """Same (e,z) enclosures as MGA-v1, plus per-step and prefix discharge bounds."""
    if not all(math.isfinite(v) for v in (base_soc,packet)) or base_soc < -1e-9 or packet < -1e-9 or base_soc+packet>b.capacity+1e-9:
        raise ValueError('Infeasible packet')
    el=eu=max(0.0,base_soc);zl=zu=max(0.0,packet)
    lower=[];upper=[];prefix=[];prices=[]
    waste_upper=discharge_lower=0.0
    for box in envelopes:
        lo,hi=box.net_low,box.net_high
        rmin=min(max(0.0,lo),b.discharge_kw*box.dt)/b.eta_discharge
        rmax=min(max(0.0,hi),b.discharge_kw*box.dt)/b.eta_discharge
        g=min(zl,max(0.0,rmin-eu)) if lo>=0 else 0.0
        cap=min(zu,max(0.0,rmax-el)) if hi>=0 else 0.0
        qmax=b.eta_charge*min(max(0.0,-lo),b.charge_kw*box.dt)
        # Packet loss to PV replenishment cannot exceed the replenishment itself.
        waste_cap=min(zu,qmax,max(0.0,eu+zu+qmax-b.capacity))
        lows=[];highs=[]
        if hi>=0:
            lows.append(min(zl,max(0.0,el+zl-rmax)))
            highs.append(min(zu,max(0.0,eu+zu-rmin)))
        if lo<=0:
            qmin=b.eta_charge*min(max(0.0,-hi),b.charge_kw*box.dt)
            lows.append(min(zl,max(0.0,b.capacity-eu-qmax)))
            highs.append(min(zu,max(0.0,b.capacity-el-qmin)))
        el,eu=soc_map(el,hi,b,box.dt),soc_map(eu,lo,b,box.dt)
        zl,zu=min(lows),max(highs)
        waste_upper=min(packet,waste_upper+waste_cap)
        discharge_lower=max(discharge_lower+g,packet-zu-waste_upper,0.0)
        lower.append(g);upper.append(cap);prefix.append(discharge_lower);prices.append(box.price_low)
    return dict(lower=lower,upper=upper,prefix_lower=prefix,prices=prices)


def packet_value_lower(base_soc, packet, envelopes, b):
    constraints=packet_constraints(base_soc,packet,envelopes,b)
    return b.eta_discharge*prefix_minimum_revenue(**constraints)


def guarded_action(x,soc,b,envelopes,remaining_budget):
    """Exactly MGA-v1 candidate grid, threshold and allowance; only value bound changes."""
    if not math.isfinite(remaining_budget) or remaining_budget<0:
        raise ValueError('Nonnegative finite budget required')
    base=self_consumption(x,soc,b)
    info=dict(delta=0.0,predicted_lower_gain=0.0,extra_spend=0.0)
    if x.pv<x.load or x.price<0 or not envelopes:
        return base,remaining_budget,info
    e=soc+b.eta_charge*base.charge
    limit=max(0.0,min(base.direct,b.charge_kw*x.dt-base.charge,(b.capacity-e)/b.eta_charge))
    if x.price>0:
        limit=min(limit,remaining_budget/x.price)
    best_delta=best_gain=0.0
    for fraction in FRACTIONS:
        delta=limit*fraction
        gain=packet_value_lower(e,b.eta_charge*delta,envelopes,b)-x.price*delta
        if gain>best_gain+1e-10:
            best_delta,best_gain=delta,gain
    if best_delta==0:
        return base,remaining_budget,info
    spend=x.price*best_delta
    return (Action(base.direct-best_delta,base.charge+best_delta,0.0),
            max(0.0,remaining_budget-spend),
            dict(delta=best_delta,predicted_lower_gain=best_gain,extra_spend=spend))
