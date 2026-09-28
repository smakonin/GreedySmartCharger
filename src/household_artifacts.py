"""Lossless compact result serialization and independent executed-flow accounting."""
import hashlib,json,math
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_result(r,b,v,certificate,budget):
    s=z=r['initial_soc'];cost=base=0.;charge=discharge=0.;max_error=0.;igrid=np.zeros(len(r['intervals']))
    def close(a,c):
        nonlocal max_error
        max_error=max(max_error,abs(a-c))
        if abs(a-c)>1e-7:raise AssertionError(('independent result arithmetic',a,c))
    for q in r['trace']:
        h=(q['end']-q['start'])/3600;i=int(q['start']//1800);x=r['intervals'][i]
        load=x['load_kwh']*2;pv=x['pv_kwh']*2;p=x['price'];close(s,q['soc_before']);close(z,q['shadow_soc_before'])
        for power,before,after,gkey,ckey in [(q['power'],s,q['soc'],'grid_kwh','cost'),(q['shadow_power'],z,q['shadow_soc'],'shadow_grid_kwh','shadow_cost')]:
            direct,pvc,d,gc=[power[k] for k in ('direct','pv_charge','discharge','grid_charge')]
            if min(direct,pvc,d,gc)<-1e-8 or pvc+gc>b.charge_kw+1e-8 or d>b.discharge_kw+1e-8:raise AssertionError('power violation')
            if d>1e-8 and pvc+gc>1e-8:raise AssertionError('simultaneous operation')
            if direct+d>load+1e-8 or direct+pvc>pv+1e-8:raise AssertionError('allocation violation')
            if not -1e-8<=after<=b.capacity+1e-8:raise AssertionError('SOC violation')
            grid=(load-direct-d+gc)*h;close(grid,q[gkey]);close(grid*p,q[ckey])
            close(before+b.eta_charge*(pvc+gc)*h-d/b.eta_discharge*h,after)
            curtail=(pv-direct-pvc)*h;loss=((1-b.eta_charge)*(pvc+gc)+(1/b.eta_discharge-1)*d)*h
            close(pv*h+grid-load*h-curtail-loss,after-before)
        cost+=q['cost'];base+=q['shadow_cost'];s=q['soc'];z=q['shadow_soc'];close(cost-base,q['regret'])
        charge+=(q['power']['pv_charge']+q['power']['grid_charge'])*h;discharge+=q['power']['discharge']*h;igrid[i]+=q['grid_kwh']
        if certificate and (q['segment_peak']>budget+1e-8 or q['regret']+q['liability']>budget+1e-8):raise AssertionError('guarantee violation')
    close(cost,r['cost']);close(base,r['shadow_cost']);close(s,r['final_soc']);close(z,r['shadow_final_soc'])
    close(cost-base-v*(s-z),r['terminal_adjusted_difference']);close((b.eta_charge*charge+discharge/b.eta_discharge)/(2*b.capacity),r['efc'])
    close(max(igrid)*2,r['peak_half_hour_import_kw'])
    for i,x in enumerate(r['intervals']):close(igrid[i],x['import_kwh'])
    for d in r['decisions']:
        i=round(d['time']/1800);x=r['intervals'][i]
        close(x['start_soc'],d['soc']);close(x['start_shadow_soc'],d['shadow_soc'])
        if i:close(r['intervals'][i-1]['regret_after'],d['regret_before'])
        if d['latest_meter_end']>d['time'] or d['latest_meter_ready_at']>d['time']:raise AssertionError('future meter access')
    return dict(independent_max_residual=max_error,segments_checked=len(r['trace']),decisions_checked=len(r['decisions']))


def pack_result(path,result):
    # No float32 conversion: exact binary64 values and every local physical event.
    arrays={};layout={}
    for name in ('intervals','trace','decisions'):
        rows=result[name];flat=[]
        for row in rows:
            d={}
            for k,v in row.items():
                if isinstance(v,dict):
                    for sk,sv in v.items():d[k+'.'+sk]=sv
                elif not isinstance(v,list):d[k]=v
            flat.append(d)
        keys=sorted({k for row in flat for k in row})
        numeric=[];categorical={}
        for k in keys:
            vals=[row.get(k) for row in flat]
            if any(isinstance(v,str) for v in vals):
                cats=sorted({v for v in vals if v is not None});categorical[k]=cats
                arrays[name+'__'+k]=np.array([cats.index(v) if v in cats else -1 for v in vals],np.int16)
            else:
                numeric.append(k)
        arrays[name]=np.array([[np.nan if row.get(k) is None else row[k] for k in numeric] for row in flat],dtype=np.float64)
        layout[name]=dict(numeric=numeric,categorical=categorical)
    arrays['layout']=np.array(json.dumps(layout,sort_keys=True))
    np.savez_compressed(path,**arrays)
    return sha(path)


def metrics(result,b,config):
    rows=result['intervals']
    return dict(cost=result['cost'],final_soc=result['final_soc'],initial_soc=result['initial_soc'],
        objective=result['cost']-config['terminal_value']*(result['final_soc']-result['initial_soc']),
        shadow_cost=result['shadow_cost'],shadow_final_soc=result['shadow_final_soc'],regret=result['regret'],
        terminal_difference=result['terminal_adjusted_difference'],efc=result['efc'],peak_kw=result['peak_half_hour_import_kw'],
        import_kwh=math.fsum(r['import_kwh'] for r in rows),charge_ac_kwh=math.fsum(r['charge_ac_kwh'] for r in rows),
        discharge_ac_kwh=math.fsum(r['discharge_ac_kwh'] for r in rows),curtailment_kwh=math.fsum(r['curtailment_kwh'] for r in rows),
        loss_kwh=math.fsum(r['loss_kwh'] for r in rows),load_kwh=math.fsum(r['load_kwh'] for r in rows),pv_kwh=math.fsum(r['pv_kwh'] for r in rows),
        last_day_cost=math.fsum(r['cost'] for r in rows[-48:]),fallback_segments=sum(t['fallback'] for t in result['trace']),
        max_prefix_exposure=result['max_prefix_exposure'],timing_seconds=result['timing_seconds'])
