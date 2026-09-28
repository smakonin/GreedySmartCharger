"""Finite-horizon clipping liability and a grid-powered deficit-repair option.

The actual settlement horizon and deterministic price intervals are inputs.
No point forecast, empirical interval or rolling terminal reset enters the bound.
"""
import math
import numpy as np
from grid_model import Action, transition
from shadow_reserve import couple, liability
from shadow_reserve_v2 import choose_index


def repair(x, soc, shadow, shadow_action, b):
    """Catch a higher shadow SOC as fast as charging power allows.

    Copy shadow direct PV and existing PV charging. Additional charge is grid
    supplied, including at negative prices. Never overshoot the next shadow SOC.
    """
    if soc >= shadow:
        return couple(x,soc,shadow_action,b)
    target=min(transition(x,shadow,shadow_action,b).end_soc,
               soc+b.eta_charge*b.charge_kw*x.dt)
    if target < soc:
        return Action(shadow_action.direct,0.,(soc-target)*b.eta_discharge,0.)
    total=max(0.,(target-soc)/b.eta_charge)
    pv=min(shadow_action.charge,total)
    return Action(shadow_action.direct,pv,0.,max(0.,total-pv))


class FiniteLiability:
    def __init__(self, lows, highs, durations, b):
        self.lo=np.asarray(lows,dtype=float)
        self.hi=np.asarray(highs,dtype=float)
        self.dt=np.asarray(durations,dtype=float)
        self.b=b
        self.n=len(self.lo)
        if (self.lo.shape!=self.hi.shape or self.lo.shape!=self.dt.shape
            or not np.all(np.isfinite([self.lo,self.hi,self.dt]))
            or np.any(self.lo>self.hi) or np.any(self.dt<=0)):
            raise ValueError('Invalid deterministic horizon bounds')
        self.dcap=b.discharge_kw*self.dt/b.eta_discharge
        self.ccap=b.charge_kw*self.dt*b.eta_charge
        self.deficit=self._prepare(b.eta_discharge*np.maximum(0,self.hi),self.dcap)
        self.surplus=self._prepare(np.maximum(0,-self.lo)/b.eta_charge,self.ccap)

    @staticmethod
    def _prepare(weights,caps):
        levels=np.unique(weights)[::-1]
        levels=levels[levels>0]
        suffix=np.zeros((len(caps)+1,len(levels)))
        for t in range(len(caps)-1,-1,-1):
            suffix[t]=suffix[t+1]
            suffix[t]+=caps[t]*(levels==weights[t])
        return weights,levels,suffix

    def _knapsack(self,t,common,gap,caps,prepared):
        weights,levels,suffix=prepared
        if not len(levels) or t==self.n:
            return np.zeros_like(gap)
        available=np.broadcast_to(suffix[t],(len(gap),len(levels))).copy()
        used=0.
        # Only the initial common energy/headroom must be cleared. This is at
        # most capacity, normally three or four slots rather than the horizon.
        for j in range(t,self.n):
            removed=np.clip(common-used,0.,caps[j])
            available-=removed[:,None]*(levels==weights[j])[None,:]
            used+=caps[j]
            if used>=float(np.max(common)):
                break
        remaining=gap.copy()
        value=np.zeros_like(gap)
        for k,weight in enumerate(levels):
            take=np.minimum(remaining,np.maximum(0.,available[:,k]))
            value+=weight*take
            remaining-=take
        return value

    def copy_values(self,t,socs,shadow):
        s=np.atleast_1d(np.asarray(socs,dtype=float))
        if not 0<=t<=self.n:
            raise ValueError('Outside actual evaluation horizon')
        out=np.zeros_like(s)
        low=s<shadow
        if np.any(low):
            out[low]=self._knapsack(t,s[low],shadow-s[low],self.dcap,self.deficit)
        high=s>shadow
        if np.any(high):
            out[high]=self._knapsack(t,self.b.capacity-s[high],s[high]-shadow,self.ccap,self.surplus)
        return out

    def repair_values(self,t,socs,shadow):
        s=np.atleast_1d(np.asarray(socs,dtype=float))
        out=np.full_like(s,np.inf)
        out[s==shadow]=0.
        if t==self.n:
            return np.zeros_like(s)
        low=s<shadow
        if np.any(low):
            room=self.b.capacity-s[low]
            used=0.
            maximum=np.zeros_like(room)
            for j in range(t,self.n):
                maximum=np.where(used<room,np.maximum(maximum,max(0.,self.hi[j])),maximum)
                used+=self.ccap[j]
                if used>=float(np.max(room)):
                    break
            out[low]=maximum*(shadow-s[low])/self.b.eta_charge
        return out

    def values(self,kind,t,socs,shadow):
        s=np.atleast_1d(np.asarray(socs,dtype=float))
        if kind=='global':
            b=self.b
            return (b.eta_discharge*max(0.,float(max(self.hi)))*np.maximum(0,shadow-s)
                +max(0.,-float(min(self.lo)))*np.maximum(0,s-shadow)/b.eta_charge)
        copied=self.copy_values(t,s,shadow)
        if kind=='copy':
            return copied
        if kind=='repair':
            return np.minimum(copied,self.repair_values(t,s,shadow))
        raise ValueError(kind)

    def fallback(self,kind,t,x,soc,shadow,shadow_action):
        if kind=='repair' and self.repair_values(t,[soc],shadow)[0] < self.copy_values(t,[soc],shadow)[0]:
            return repair(x,soc,shadow,shadow_action,self.b),'repair'
        return couple(x,soc,shadow_action,self.b),'copy'


class FiniteLedger:
    def __init__(self,model,kind,budget=1.):
        if kind not in ('global','copy','repair') or not math.isfinite(budget) or budget<0:
            raise ValueError('Invalid ledger')
        self.model=model
        self.kind=kind
        self.budget=budget
        self.regret=0.
        self.breach_debit=0.
        self.t=0

    def choose(self,x,soc,shadow,shadow_action,candidates,scores,b):
        t=self.t
        assert abs(x.dt-self.model.dt[t])<1e-12
        sf=transition(x,shadow,shadow_action,b)
        fallback,branch=self.model.fallback(self.kind,t,x,soc,shadow,shadow_action)
        ff=transition(x,soc,fallback,b)
        old_phi=self.model.values(self.kind,t,[soc],shadow)[0]
        fall_phi=self.model.values(self.kind,t+1,[ff.end_soc],sf.end_soc)[0]
        increment=ff.cost-sf.cost+fall_phi-old_phi
        breach=not self.model.lo[t]<=x.price<=self.model.hi[t]
        if not breach and increment>1e-8:
            raise AssertionError(('Fallback inequality',self.kind,t,increment,soc,shadow,branch))
        debit=max(0.,increment) if breach else 0.
        self.breach_debit+=debit
        flows=[transition(x,soc,a,b) for a in candidates]
        ends=[f.end_soc for f in flows]
        phis=self.model.values(self.kind,t+1,ends,sf.end_soc) if ends else []
        original=self.model.values('global',t+1,ends,sf.end_soc) if ends else []
        safe=[i for i,f in enumerate(flows) if not breach and
              self.regret+f.cost-sf.cost+phis[i]<=self.budget+self.breach_debit]
        additional=sum(self.regret+flows[i].cost-sf.cost+original[i]>self.budget+self.breach_debit for i in safe)
        if safe:
            k=choose_index(candidates,scores,x,soc,b,safe)
            action=candidates[k]
            f=flows[k]
            phi=float(phis[k])
            reason='candidate'
        else:
            k=None
            action=fallback
            f=ff
            phi=float(fall_phi)
            reason='price_breach' if breach else 'fallback'
        self.regret+=f.cost-sf.cost
        self.t+=1
        slack=self.budget+self.breach_debit-self.regret-phi
        if slack < -1e-7:
            raise AssertionError(('Prefix certificate',slack))
        global_phi=float(self.model.values('global',t+1,[f.end_soc],sf.end_soc)[0])
        return action,dict(reason=reason,chosen=k,fallback_branch=branch,admissible=len(safe),
            additional_admissible=additional,regret=self.regret,liability=phi,
            liability_reduction=global_phi-phi,slack=slack,price_breach=breach,debit=debit,
            cumulative_debit=self.breach_debit,fallback_increment=float(increment),
            certificate=self.budget+self.breach_debit)
