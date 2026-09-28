"""Information-gated storage controllers; no realized unknown price in decisions."""
from dataclasses import dataclass
import math
import numpy as np
from grid_model import Action, Interval, transition
from grid_controllers import pfg, price_greedy
from grid_dp import action_for_transition
from shadow_reserve import couple, propose
from shadow_reserve_v2 import choose_index
from finite_liability import FiniteLiability, repair


class Bounds(FiniteLiability):
    def values(self,kind,t,socs,shadow):
        if t==self.n:
            return np.zeros_like(np.atleast_1d(np.asarray(socs,dtype=float)))
        return super().values(kind,t,socs,shadow)


class PriceFeed:
    """Historical payload plus explicit release clock. Only read through known()."""
    def __init__(self, prices, releases, offset):
        self._prices=np.asarray(prices,float)
        self._releases=np.asarray(releases,int)
        self.offset=offset

    def known(self,index,now):
        k=index+self.offset
        return 0<=k<len(self._prices) and self._releases[k]<=now and math.isfinite(self._prices[k])

    def read(self,index,now):
        if not self.known(index,now):
            raise ValueError(('Unreleased price access',index,now))
        return float(self._prices[index+self.offset])

    def estimate(self,index,now):
        if self.known(index,now):
            return self.read(index,now),1,'announced'
        samples=[]
        for k in range(1,32):
            j=index-24*k
            if self.known(j,now):samples.append(self.read(j,now))
            if len(samples)==7:break
        if not samples:raise ValueError(('No past price history',index,now))
        return float(np.mean(samples)),len(samples),'forecast'

    def envelope(self,now,n,low,high,b):
        lo=np.full(n-now,low);hi=np.full(n-now,high)
        for j in range(now,n):
            if self.known(j,now):
                p=self.read(j,now)
                if not low-1e-12<=p<=high+1e-12:
                    raise ValueError(('Announced price outside design envelope',j,p,low,high))
                lo[j-now]=hi[j-now]=p
        return Bounds(lo,hi,np.ones(n-now),b)


def context(load_pv,offset,feed,t,n):
    """Current load/PV plus past-only same-hour forecasts, with sample counts."""
    current=load_pv[offset+t]
    if current is None:raise ValueError('Missing current measurement')
    p,pc,mode=feed.estimate(t,t)
    x=Interval(current[0],current[1],p,1.)
    predictions=[];counts=[]
    for j in range(t+1,min(n,t+24)):
        past=[]
        for k in range(1,32):
            q=j-24*k
            if q<t and 0<=offset+q<len(load_pv) and load_pv[offset+q] is not None:
                past.append(load_pv[offset+q])
            if len(past)==7:break
        if not past:raise ValueError(('No load/PV history',j,t))
        mean=np.mean(past,axis=0);fp,fc,fmode=feed.estimate(j,t)
        predictions.append(Interval(float(mean[0]),float(mean[1]),fp,1.))
        counts.append(dict(target=j,load_pv_samples=len(past),price_samples=fc,price_mode=fmode))
    return x,predictions,dict(current_price_mode=mode,current_price_samples=pc,forecast_counts=counts)


def thb(x,s,b,m,M,rho):
    """Chau Algorithm 1 on positive prices; explicit signed extension otherwise."""
    if x.price<=0:return price_greedy(x,s,b)
    rho=min(1.,max(0.,rho));m=max(1e-4,m);M=max(m,M)
    theta=.5*(math.sqrt(rho*rho*(M-m)**2+4*M*m)-rho*(M-m))*b.eta_charge*b.eta_discharge
    target=b.capacity*(1-rho);direct=min(x.load,x.pv)
    charge=min(max(0.,x.pv-direct),(b.capacity-s)/b.eta_charge,b.charge_kw*x.dt)
    if x.price<=theta:
        gc=min(max(0.,(target-s)/b.eta_charge-charge),max(0.,b.charge_kw*x.dt-charge))
        return Action(direct,charge,0.,gc)
    return Action(direct,charge,min(x.load-direct,b.discharge_kw*x.dt,s*b.eta_discharge),0.)


def omg(x,s,b,low,high):
    """Qin equation (17), maxW; exact endpoints/kinks, matched feasible set."""
    dlow=min(0.,low/b.eta_charge);dhigh=max(0.,high/b.eta_charge)
    umin=-b.discharge_kw*x.dt/b.eta_discharge;umax=b.charge_kw*x.dt*b.eta_charge
    W=(b.capacity-(umax-umin))/(dhigh-dlow)
    if W<=0:raise ValueError('OMG frequent-action condition failed')
    gamma=-W*dlow+umax-b.capacity
    lower=max(0.,s-min(x.load,b.discharge_kw*x.dt)/b.eta_discharge)
    upper=min(b.capacity,s+umax)
    kink=s+(x.pv-x.load)*b.eta_charge if x.pv>=x.load else s+(x.pv-x.load)/b.eta_discharge
    targets=np.unique(np.clip([lower,upper,s,kink],lower,upper))
    actions=[action_for_transition(x,s,float(e),b) for e in targets]
    scores=[(s+gamma)*(transition(x,s,a,b).end_soc-s)+W*transition(x,s,a,b).cost for a in actions]
    return actions[choose_index(actions,scores,x,s,b)]


def candidates(x,s,z,sa,b,levels,values,pred,q):
    sf=transition(x,z,sa,b);copy=couple(x,s,sa,b)
    actions,scores,info=propose(x,s,sf.end_soc,copy,b,levels,values,pred,q)
    ra=repair(x,s,z,sa,b);rf=transition(x,s,ra,b)
    actions.append(ra)
    scores.append(rf.cost+float(np.interp(rf.end_soc,levels,values))
                  +info['caution_weight']*info['forecast_width']*abs(rf.end_soc-sf.end_soc))
    return actions,scores


@dataclass
class PendingLedger:
    budget: float
    kind: str
    account: float=0.
    debit: float=0.

    def __post_init__(self):self.pending={}

    def settle(self,feed,t):
        settled=[]
        for i,(dg,upper) in list(self.pending.items()):
            if feed.known(i,t):
                actual=feed.read(i,t)*dg;excess=max(0.,actual-upper)
                self.account+=actual-upper;self.debit+=excess
                settled.append(dict(t=i,difference=actual,reservation=upper,debit=excess))
                del self.pending[i]
        return settled

    def choose(self,x,s,z,sa,actions,scores,b,model,t,known):
        sf=transition(x,z,sa,b)
        flows=[transition(x,s,a,b) for a in actions]
        ends=[f.end_soc for f in flows]
        h=np.array([max(model.lo[0]*(f.grid-sf.grid),model.hi[0]*(f.grid-sf.grid)) for f in flows])
        choices={};admissions={}
        for k in ('global','copy','repair'):
            phis=model.values(k,1,ends,sf.end_soc)
            safe=np.flatnonzero(self.account+h+phis<=self.budget+self.debit).tolist()
            admissions[k]=len(safe)
            if safe:
                i=choose_index(actions,scores,x,s,b,safe);a=actions[i];phi=float(phis[i])
            else:
                a,_=model.fallback(k,0,x,s,z,sa);f=transition(x,s,a,b)
                i=None;phi=float(model.values(k,1,[f.end_soc],sf.end_soc)[0])
            choices[k]=(i,a,phi)
        i,a,phi=choices[self.kind];f=transition(x,s,a,b)
        fallback,branch=model.fallback(self.kind,0,x,s,z,sa)
        ff=transition(x,s,fallback,b)
        fh=max(model.lo[0]*(ff.grid-sf.grid),model.hi[0]*(ff.grid-sf.grid))
        oldphi=float(model.values(self.kind,0,[s],z)[0])
        fphi=float(model.values(self.kind,1,[ff.end_soc],sf.end_soc)[0])
        increment=fh+fphi-oldphi
        if increment>1e-7:raise AssertionError(('Robust fallback',self.kind,t,increment))
        dg=f.grid-sf.grid;upper=max(model.lo[0]*dg,model.hi[0]*dg)
        before=self.account;self.account+=upper
        if not known:self.pending[t]=(dg,upper)
        slack=self.budget+self.debit-self.account-phi
        if slack < -1e-7:raise AssertionError(('Reservation invariant',self.kind,t,slack))
        def changed(k):
            other=choices[k][1]
            return max(abs(getattr(a,v)-getattr(other,v)) for v in ('direct','charge','discharge','grid_charge'))>1e-7
        preferred=choose_index(actions,scores,x,s,b)
        rejected_global=before+upper+float(model.values('global',1,[f.end_soc],sf.end_soc)[0])>self.budget+self.debit+1e-7
        rejected_copy=before+upper+float(model.values('copy',1,[f.end_soc],sf.end_soc)[0])>self.budget+self.debit+1e-7
        return a,dict(account_before=before,account=self.account,observed_debit=self.debit,
            budget=self.budget,liability=phi,slack=slack,reservation=upper,grid_difference=dg,
            current_known=known,pending_count=len(self.pending),fallback_branch=branch,
            fallback_increment=increment,chosen=i,preferred=preferred,filter_changed=i!=preferred,
            admitted=admissions[self.kind],additional_vs_global=admissions[self.kind]-admissions['global'],
            selected_rejected_by_global=rejected_global,selected_rejected_by_copy=rejected_copy,
            same_state_changed_vs_global=changed('global'),same_state_changed_vs_copy=changed('copy'))
