"""Execute the frozen complete household matrix with lossless traces and aliases."""
import argparse,concurrent.futures,datetime,hashlib,json,math,os,platform,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from ausgrid_interval_loader import load_case
from physical_simulator import BatterySpec
from executed_liability import ExecutedLiability
from household_meter import replay_meter,tariff_bands,HISTORY
from household_controller import ForecastBook,HouseholdController
from household_matrix import physical_key
from household_artifacts import metrics,pack_result,sha,verify_result


def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def work_case(root_str,base_str,case):
    root=Path(root_str);base=Path(base_str);out=base/'cases'/case['case_id'];out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();configs=json.loads((base/'configurations.json').read_text());rows=[]
    try:
        readings,source=load_case(root,case['customer'],case['test_start'])
        np.savez_compressed(out/'inputs.npz',load=np.array([r.load_kwh for r in readings]),pv=np.array([r.pv_kwh for r in readings]))
        history=readings[:HISTORY];hl=math.fsum(r.load_kwh for r in history);hp=math.fsum(r.pv_kwh for r in history)
        features=dict(history_load_kwh=hl,history_pv_kwh=hp,pv_load_ratio=hp/hl if hl else None,
                      pv_coincidence=math.fsum(min(r.load_kwh,r.pv_kwh) for r in history)/hp if hp else None,
                      mean_daily_load_kwh=hl/28,source=source)
        write_json(out/'history_features.json',features)
        for tariff in ('flat','tou','signed'):
            bands=tariff_bands(30,tariff);books={};executions={}
            for c in configs:
                if (base/'abort.json').exists():raise RuntimeError('run aborted by coordinator')
                row_id=case['case_id']+'__'+tariff+'__'+c['config_id'];key=physical_key(c)
                if key in executions:
                    original=executions[key];row={**original,'row_id':row_id,'config_id':c['config_id'],'group':c['group'],'reused_physical_trajectory':True}
                    if row['status']=='ok':
                        row['metrics']=dict(row['metrics']);m=row['metrics'];v=c['terminal_value']
                        m['objective']=m['cost']-v*(m['final_soc']-m['initial_soc'])
                        m['terminal_difference']=m['regret']-v*(m['final_soc']-m['shadow_final_soc'])
                else:
                    row=dict(row_id=row_id,execution_id=row_id,case_id=case['case_id'],customer=case['customer'],test_start=case['test_start'],
                             tariff=tariff,config_id=c['config_id'],group=c['group'],reused_physical_trajectory=False)
                    try:
                        b=BatterySpec(c['capacity'],c['charge_kw'],c['discharge_kw'],.95,.95)
                        bk=(c['forecast'],c['supervisory_steps'])
                        if bk not in books:
                            books[bk]=ForecastBook(readings,bands,*bk)
                            # Forecast values retained once per source/tariff/method/cadence.
                            np.savez_compressed(out/f'forecast_{tariff}_{bk[0]}_{bk[1]}.npz',prediction=books[bk].pred,rho=books[bk].rho,prices=books[bk].prices)
                        book=books[bk];ctrlname='MPC' if c['policy'].startswith('SCR-') else c['policy']
                        model=ExecutedLiability(bands,b,c['terminal_value']);ctrl=HouseholdController(ctrlname,c['states'],c['terminal_value'],book)
                        result=replay_meter(readings,model,ctrl,certificate=c['certificate'],budget=c['budget'],
                                            initial_soc=c['initial_fraction']*b.capacity,supervisory_steps=c['supervisory_steps'])
                        checked=verify_result(result,b,c['terminal_value'],c['certificate'],c['budget'])
                        filename=tariff+'__'+c['config_id']+'.npz';trace_hash=pack_result(out/filename,result)
                        row.update(status='ok',metrics=metrics(result,b,c),audit=checked,trace_file=filename,trace_sha256=trace_hash)
                        del result
                    except ValueError as exc:
                        if ctrlname=='OMG' and 'frequent-action condition' in str(exc):
                            row.update(status='expected_parameter_domain_failure',error=str(exc))
                        else:raise
                    executions[key]=row
                rows.append(row)
                write_json(out/'progress.json',dict(case_id=case['case_id'],logical_rows=len(rows),physical_runs=len(executions),status='running',elapsed_seconds=time.perf_counter()-started))
            write_json(out/f'planning_cache_{tariff}.json',{str(k):dict(forecast_preparation_seconds=v.preparation_seconds,dp_build_seconds=v.dp_build_seconds,dp_build_count=v.dp_build_count) for k,v in books.items()})
            del books
        write_json(out/'results.json',dict(case=case,rows=rows,elapsed_seconds=time.perf_counter()-started,
            process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_rss_units='bytes on macOS; KiB on Linux',
            note='Parallel study timings are diagnostic; serial uncached runtime benchmark is separate.'))
        write_json(out/'progress.json',dict(case_id=case['case_id'],logical_rows=len(rows),status='complete',elapsed_seconds=time.perf_counter()-started))
        return dict(case_id=case['case_id'],status='complete',logical_rows=len(rows),expected_failures=sum(r['status']!='ok' for r in rows),elapsed_seconds=time.perf_counter()-started)
    except Exception as exc:
        failure=dict(case_id=case['case_id'],status='unexpected_failure',completed_rows=rows,error=repr(exc),traceback=traceback.format_exc())
        write_json(out/'failure.json',failure)
        return dict(case_id=case['case_id'],status='unexpected_failure',error=repr(exc))


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=a.package.resolve();frozen=json.loads((base/'execution_freeze.json').read_text())
    if (base/'execution_started.json').exists():raise ValueError('new frozen execution only; no silent overwrite/resume')
    for name,digest in frozen['source_sha256'].items():
        if sha(root/'research'/name)!=digest:raise ValueError('changed frozen source: '+name)
    for name,digest in frozen['design_sha256'].items():
        if sha(base/name)!=digest:raise ValueError('changed frozen matrix: '+name)
    if not frozen['preflight_passed']:raise ValueError('preflight not passed')
    if shutil.disk_usage(base).free<20*1024**3:raise ValueError('less than 20 GiB free before full run')
    cases=json.loads((base/'cases.json').read_text());start=time.perf_counter();completed=[]
    write_json(base/'execution_started.json',dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=a.workers,
        matrix_sha256=sha(base/'execution_matrix.csv'),freeze_sha256=sha(base/'execution_freeze.json'),python=sys.version,platform=platform.platform(),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},source_scope=frozen['source_scope']))
    pending_iter=iter(cases)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        active={pool.submit(work_case,str(root),str(base),next(pending_iter)):None for _ in range(min(a.workers,len(cases)))}
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future);r=future.result();completed.append(r)
                if r['status']!='complete':write_json(base/'abort.json',r)
                progress=dict(completed_cases=sum(x['status']=='complete' for x in completed),total_cases=len(cases),
                    completed_logical_rows=sum(x.get('logical_rows',0) for x in completed),expected_parameter_failures=sum(x.get('expected_failures',0) for x in completed),
                    elapsed_seconds=time.perf_counter()-start,unexpected_failures=[x for x in completed if x['status']!='complete'])
                write_json(base/'progress.json',progress);print(json.dumps(progress),flush=True)
                if not (base/'abort.json').exists():
                    try:c=next(pending_iter)
                    except StopIteration:continue
                    if shutil.disk_usage(base).free<8*1024**3:
                        write_json(base/'abort.json',dict(error='free disk below 8 GiB; completed outputs preserved'));continue
                    active[pool.submit(work_case,str(root),str(base),c)]=None
    write_json(base/'execution_completion.json',dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        all_cases_complete=len(completed)==len(cases) and all(r['status']=='complete' for r in completed),cases=completed,elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':main()
