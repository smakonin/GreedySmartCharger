"""Causal shared forecast/DP preparation for a matched household case.

The cache is keyed by every model/forecast/price parameter. It reuses only identical
causal predictions and state-independent continuation values; actions depend on
actual SOC. Perfect forecasts are explicitly nonimplementable references only.
"""
import math,time
import numpy as np
from physical_simulator import BatterySpec
from grid_model import Interval,Action
from available_price_control import thb,omg
from household_meter import BatteryRequest,request_from_action,known_price,HISTORY,STEP
from fast_interval_planner import continuation,first_action


class ForecastBook:
    def __init__(self,readings,price_bands,method='weekday',steps=1):
        self.method=method;self.steps=steps;self.bands=price_bands;self.n=len(readings)-HISTORY
        if method not in ('weekday','previous_day','persistence','perfect') or steps not in (1,2):raise ValueError('forecast setup')
        if self.n<=0 or self.n%steps:raise ValueError('episode length')
        if any(r.start!=(i-HISTORY)*STEP or r.end!=r.start+STEP or r.ready_at!=r.end for i,r in enumerate(readings)):
            raise ValueError('prepared book requires complete zero-extra-delay interval history')
        start=time.perf_counter()
        values=np.array([[r.load_kwh,r.pv_kwh] for r in readings])
        t=np.arange(self.n)[:,None];j=np.arange(48)[None,:]
        target=HISTORY+t+j
        if method=='weekday':idx=target[:,:,None]-336*np.arange(1,5)[None,None,:]
        elif method=='previous_day':idx=(target-48)[:,:,None]
        elif method=='persistence':idx=np.broadcast_to(HISTORY+t-1,target.shape)[:,:,None]
        else:idx=np.minimum(target,len(values)-1)[:,:,None]
        if method!='perfect' and np.any(idx>=HISTORY+t[:,:,None]):raise AssertionError('forecast accesses incomplete interval')
        self.pred=values[idx].mean(axis=2)
        self.load=values[HISTORY:,0];self.pv=values[HISTORY:,1]
        self.prices=np.array([known_price(price_bands,k*STEP) for k in range(self.n)])
        cumulative=np.vstack([np.zeros(2),np.cumsum(values,axis=0)])
        totals=cumulative[HISTORY:HISTORY+self.n]-cumulative[:self.n]
        self.rho=np.minimum(1.,np.divide(totals[:,1],totals[:,0],out=np.zeros(self.n),where=totals[:,0]>0))
        self.origin_latest=(HISTORY+t-1)*STEP+STEP if method=='persistence' else None
        self.preparation_seconds=time.perf_counter()-start
        self.xs={};self.values={};self.dp_build_seconds=0.;self.dp_build_count=0

    def intervals(self,t):
        k=int(t//STEP)
        if t!=k*STEP or k%self.steps:raise ValueError('unaligned supervisor')
        if k not in self.xs:
            end=min(self.n,k+48);rows=[]
            for j in range(0,end-k,self.steps):
                h=min(self.steps,end-k-j);part=self.pred[k,j:j+h].sum(axis=0)
                ps=self.prices[k+j:k+j+h]
                if not np.all(ps==ps[0]):raise ValueError('hourly command crosses tariff boundary')
                rows.append(Interval(float(part[0]),float(part[1]),float(ps[0]),h*.5))
            self.xs[k]=tuple(rows)
        return self.xs[k]

    def table(self,t,b,states,v):
        key=(t,b.capacity,b.charge_kw,b.discharge_kw,b.eta_charge,b.eta_discharge,states,v)
        if key not in self.values:
            start=time.perf_counter();self.values[key]=continuation(self.intervals(t)[1:],b,states,v)
            self.dp_build_seconds+=time.perf_counter()-start;self.dp_build_count+=1
        return self.values[key]


class HouseholdController:
    def __init__(self,name,states,v,book):
        self.name=name;self.states=states;self.terminal_value=v;self.forecast_method=book.method;self.book=book

    def decide(self,ctx,b):
        start=time.perf_counter();book=self.book;k=int(ctx.time//STEP)
        if ctx.time!=k*STEP or ctx.history[-1].end!=ctx.time or ctx.history[0].start!=ctx.time-HISTORY*STEP:
            raise ValueError('causal context mismatch')
        if self.name=='PFG':return BatteryRequest(local_greedy=True),dict(forecast_seconds=0.,decision_seconds=time.perf_counter()-start,forecast=[])
        xs=book.intervals(ctx.time);x=xs[0];prices=book.prices
        if self.name=='THB':
            positive=prices[prices>0];a=thb(x,ctx.soc,b,float(positive.min()) if len(positive) else 1e-4,max(float(prices.max()),1e-4),float(book.rho[k]))
        elif self.name=='OMG':
            a=Action(min(x.load,x.pv),0.,0.) if np.all(prices==0) else omg(x,ctx.soc,b,float(prices.min()),float(prices.max()))
        elif self.name=='MPC':
            levels,values=book.table(ctx.time,b,self.states,self.terminal_value)
            a=first_action(x,ctx.soc,b,levels,values)
        else:raise ValueError('unknown controller')
        q=request_from_action(a,x.dt,x.price)
        return q,dict(forecast_seconds=0.,decision_seconds=time.perf_counter()-start,forecast=[],
                      first_forecast_load_kwh=x.load,first_forecast_pv_kwh=x.pv,forecast_source='shared causal book' if book.method!='perfect' else 'nonimplementable perfect forecast')
