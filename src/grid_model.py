"""Grid-charging v1 AC-bus model. Frozen PV-only model remains separate."""
from dataclasses import dataclass
import math
from physical_simulator import BatterySpec, Interval
TOL=1e-8

@dataclass(frozen=True)
class Action:
    direct: float
    charge: float       # PV -> battery, AC-side kWh
    discharge: float    # battery -> load, AC-side kWh
    grid_charge: float = 0.0

@dataclass(frozen=True)
class Flow:
    end_soc: float
    grid: float
    grid_load: float
    grid_charge: float
    curtailed: float
    loss: float
    cost: float
    balance_residual: float

def transition(x,soc,a,b,allow_grid_charging=True):
    vals=(soc,a.direct,a.charge,a.discharge,a.grid_charge)
    if not all(math.isfinite(v) for v in vals) or min(vals)<-TOL or soc>b.capacity+TOL:
        raise ValueError('Invalid SOC or flow')
    if not allow_grid_charging and a.grid_charge>TOL:raise ValueError('Grid charging disabled')
    total_charge=a.charge+a.grid_charge
    if total_charge>TOL and a.discharge>TOL:raise ValueError('Simultaneous charge/discharge')
    if total_charge>b.charge_kw*x.dt+TOL or a.discharge>b.discharge_kw*x.dt+TOL:raise ValueError('Power limit')
    grid_load=x.load-a.direct-a.discharge;curtailed=x.pv-a.direct-a.charge
    if min(grid_load,curtailed)<-TOL:raise ValueError('PV/load over-allocation or export')
    end=soc+b.eta_charge*total_charge-a.discharge/b.eta_discharge
    if end < -TOL or end>b.capacity+TOL:raise ValueError('SOC limit')
    grid=grid_load+a.grid_charge
    loss=(1-b.eta_charge)*total_charge+(1/b.eta_discharge-1)*a.discharge
    residual=x.pv+grid-x.load-curtailed-(end-soc)-loss
    if abs(residual)>TOL:raise ValueError('Energy conservation')
    return Flow(end,grid,grid_load,a.grid_charge,curtailed,loss,grid*x.price,residual)

def simulate(xs,policy,b=BatterySpec(),initial_soc=0):
    soc=initial_soc;total={k:0. for k in ['cost','import_kwh','grid_charge_kwh','pv_charge_kwh','discharge_ac_kwh','loss_kwh','curtailment_kwh','peak_kw','max_balance_residual']}
    for x in xs:
        a=policy(x,soc,b);f=transition(x,soc,a,b);soc=f.end_soc
        for key,value in [('cost',f.cost),('import_kwh',f.grid),('grid_charge_kwh',a.grid_charge),('pv_charge_kwh',a.charge),('discharge_ac_kwh',a.discharge),('loss_kwh',f.loss),('curtailment_kwh',f.curtailed)]:total[key]+=value
        total['peak_kw']=max(total['peak_kw'],f.grid/x.dt);total['max_balance_residual']=max(total['max_balance_residual'],abs(f.balance_residual))
    total.update(initial_soc=initial_soc,final_soc=soc,charge_ac_kwh=total['grid_charge_kwh']+total['pv_charge_kwh'])
    total['efc']=(b.eta_charge*total['charge_ac_kwh']+total['discharge_ac_kwh']/b.eta_discharge)/(2*b.capacity)
    return total
