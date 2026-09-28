"""AC-bus, PV-only charging, zero-export simulator; energies kWh, prices $/kWh.
Controllers propose flows. A separate transition validates rather than clips them.
No self-discharge, degradation or within-interval dynamics are modeled.
"""
from dataclasses import dataclass
import math

TOL=1e-8

@dataclass(frozen=True)
class BatterySpec:
    capacity: float = 15.0
    charge_kw: float = 5.0
    discharge_kw: float = 5.0
    eta_charge: float = .95
    eta_discharge: float = .95
    def __post_init__(self):
        vals=(self.capacity,self.charge_kw,self.discharge_kw,self.eta_charge,self.eta_discharge)
        if not all(math.isfinite(v) for v in vals) or self.capacity<=0 or min(self.charge_kw,self.discharge_kw)<0 or not (0<self.eta_charge<=1 and 0<self.eta_discharge<=1):
            raise ValueError('Invalid battery specification')

@dataclass(frozen=True)
class Interval:
    load: float
    pv: float
    price: float
    dt: float = 1.0
    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.load,self.pv,self.price,self.dt)) or min(self.load,self.pv)<0 or self.dt<=0:
            raise ValueError('Invalid interval; load and PV are interval energy, not power')

@dataclass(frozen=True)
class Action:
    direct: float
    charge: float
    discharge: float

@dataclass(frozen=True)
class Flow:
    end_soc: float
    grid: float
    curtailed: float
    loss: float
    cost: float
    balance_residual: float


def transition(x, soc, a, b):
    """Reject infeasible actions, including PV over-allocation and battery cycling."""
    if not math.isfinite(soc) or soc < -TOL or soc > b.capacity+TOL:
        raise ValueError('Invalid initial SOC')
    if not all(math.isfinite(v) for v in (a.direct,a.charge,a.discharge)) or min(a.direct,a.charge,a.discharge)<-TOL:
        raise ValueError('Invalid flow')
    if a.charge>TOL and a.discharge>TOL: raise ValueError('Simultaneous charge/discharge')
    if a.charge>b.charge_kw*x.dt+TOL or a.discharge>b.discharge_kw*x.dt+TOL:
        raise ValueError('Power limit')
    grid=x.load-a.direct-a.discharge
    curtailed=x.pv-a.direct-a.charge
    end=soc+b.eta_charge*a.charge-a.discharge/b.eta_discharge
    if min(grid,curtailed)<-TOL: raise ValueError('PV/load over-allocation or prohibited grid charging/export')
    if end < -TOL or end > b.capacity+TOL: raise ValueError('SOC limit')
    loss=(1-b.eta_charge)*a.charge+(1/b.eta_discharge-1)*a.discharge
    residual=x.pv+grid-x.load-curtailed-(end-soc)-loss
    if abs(residual)>TOL: raise ValueError('Energy conservation')
    return Flow(end,grid,curtailed,loss,grid*x.price,residual)


def self_consumption(x,soc,b):
    direct=min(x.pv,x.load)
    charge=min(x.pv-direct,b.charge_kw*x.dt,max(0,b.capacity-soc)/b.eta_charge)
    discharge=min(x.load-direct,b.discharge_kw*x.dt,max(0,soc)*b.eta_discharge)
    return Action(direct,charge,discharge)


def greedy_charge_first(x,soc,b):
    """Physical adaptation of legacy Greedy1a, including negative-price ranking."""
    charge=min(x.pv,b.charge_kw*x.dt,max(0,b.capacity-soc)/b.eta_charge)
    if x.price<0: return Action(0,charge,0)
    direct=min(x.load,x.pv-charge)
    discharge=0.0 if charge>0 else min(x.load-direct,b.discharge_kw*x.dt,max(0,soc)*b.eta_discharge)
    return Action(direct,charge,discharge)

CONTROLLERS={'GreedyChargeFirst':greedy_charge_first,'SelfConsumption':self_consumption}


def simulate(intervals, controller, spec=BatterySpec(), initial_soc=0):
    soc=initial_soc; cost=grid=pv=load=loss=charge=discharge=curtailed=0.0
    max_residual=peak=0.0; hours=0.0
    for x in intervals:
        a=controller(x,soc,spec); f=transition(x,soc,a,spec); soc=f.end_soc
        cost+=f.cost;grid+=f.grid;pv+=x.pv;load+=x.load;loss+=f.loss
        charge+=a.charge;discharge+=a.discharge;curtailed+=f.curtailed
        peak=max(peak,f.grid/x.dt);hours+=x.dt;max_residual=max(max_residual,abs(f.balance_residual))
    return dict(cost=cost,import_kwh=grid,load_kwh=load,pv_kwh=pv,loss_kwh=loss,charge_ac_kwh=charge,discharge_ac_kwh=discharge,curtailed_kwh=curtailed,initial_soc=initial_soc,final_soc=soc,peak_kw=peak,hours=hours,max_balance_residual=max_residual,efc=(spec.eta_charge*charge+discharge/spec.eta_discharge)/(2*spec.capacity))


def integer_oracle(intervals,capacity=2,power=1,initial_soc=0):
    """Exact DP for the DECLARED integer-kWh, unity-efficiency discrete model.
    Not a bound/optimality certificate for continuous physical experiments.
    Exhausts feasible direct, charge, discharge choices; free terminal SOC.
    """
    if any(x.dt!=1 or x.load!=int(x.load) or x.pv!=int(x.pv) for x in intervals):
        raise ValueError('Oracle requires integer interval energies and dt=1')
    if any(not isinstance(v,int) for v in (capacity,power,initial_soc)) or capacity<=0 or power<0 or not 0<=initial_soc<=capacity:
        raise ValueError('Oracle requires feasible integer parameters')
    states={initial_soc:0.0}
    for x in intervals:
        nxt={}
        for soc,cost in states.items():
            for direct in range(min(int(x.load),int(x.pv))+1):
                for charge in range(min(int(x.pv)-direct,power,capacity-soc)+1):
                    for discharge in range((min(int(x.load)-direct,power,soc) if charge==0 else 0)+1):
                        end=soc+charge-discharge
                        val=cost+(x.load-direct-discharge)*x.price
                        nxt[end]=min(nxt.get(end,math.inf),val)
        states=nxt
    return min(states.values())
