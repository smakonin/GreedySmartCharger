#!/usr/bin/env python3
"""Entry point for the paper's fixed household experiments and reference outputs."""
from pathlib import Path
import argparse,concurrent.futures,csv,datetime,gzip,hashlib,json,os,platform,shutil,subprocess,sys,time,urllib.request

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.build/matplotlib'))
DESIGN=ROOT/'experiments/household-v3'
SOURCE_URL='https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip'

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fresh(path):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError(f'Choose a new output directory; preserving {path}')
    path.mkdir(parents=True);return path

def build():
    compiler=shutil.which(os.environ.get('CC','cc'))
    if not compiler:raise RuntimeError('Install a C compiler, or use GSC_BACKEND=python. On Windows use WSL for the native build.')
    out=ROOT/'.build';out.mkdir(exist_ok=True)
    target=out/('libcontinuation.dylib' if sys.platform=='darwin' else 'libcontinuation.so')
    flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
    command=[compiler,'-O3','-fno-fast-math','-ffp-contract=off',*flags,str(ROOT/'src/continuation.c'),'-o',str(target)]
    subprocess.run(command,check=True)
    write(out/'native-build.json',dict(command=command,compiler=subprocess.check_output([compiler,'--version'],text=True),source_sha256=sha(ROOT/'src/continuation.c'),binary_sha256=sha(target),built_at_utc=now()))
    print('Built',target)

def fetch_data(source=None):
    dest=ROOT/'data/raw/ausgrid.zip';dest.parent.mkdir(exist_ok=True)
    expected=read(ROOT/'data/metadata/source_adjudication.json')['source_archive_sha256']
    if dest.exists():
        if sha(dest)!=expected:raise ValueError('Existing source archive has an unexpected hash; preserving it.')
        print('Source archive already verified.');return
    tmp=dest.with_suffix('.download')
    if source:shutil.copyfile(source,tmp)
    else:
        print('Downloading the qualified preservation copy (57 MB).',flush=True)
        with urllib.request.urlopen(SOURCE_URL,timeout=120) as response,tmp.open('wb') as f:shutil.copyfileobj(response,f)
    if sha(tmp)!=expected:raise ValueError(f'Download hash differs from the paper; retained for inspection at {tmp}')
    tmp.replace(dest);print('Verified source archive SHA-256:',expected)

def verify_reference():
    manifest=read(ROOT/'provenance/release-files.json')
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('Release file changed: '+name)
    import numpy as np
    points=list(csv.DictReader((ROOT/'reference/analysis/primary_case_points.csv').open()))
    homes=list(csv.DictReader((ROOT/'reference/analysis/primary_household_points.csv').open()))
    summary=read(ROOT/'reference/analysis/summary.json')
    assert len(points)==4035 and len(homes)==2070
    from collections import defaultdict
    for tariff in ('flat','tou','signed'):
        grouped={}
        for policy in ('PFG','THB','OMG','MPC','SCR-repair'):
            d=defaultdict(list)
            for row in points:
                if (row['tariff'],row['policy'])==(tariff,policy):d[int(row['customer'])].append(float(row['cost']))
            assert len(d)==138
            grouped[policy]={k:sum(v)/len(v) for k,v in d.items()}
            average=sum(grouped[policy].values())/138
            assert abs(average-summary['primary'][tariff][policy]['cost']['mean'])<1e-10
        for name,a,b in [('greedy','PFG','MPC'),('protected','SCR-repair','MPC')]:
            hh={int(r['customer']):r for r in homes if (r['tariff'],r['policy'])==(tariff,a)}
            values=np.array([grouped[a][k]-grouped[b][k]-float(hh[k]['epsilon']) for k in sorted(hh)])
            rng=np.random.default_rng(20260928)
            upper=float(np.quantile(values[rng.integers(0,138,(10000,138))].mean(axis=1),1-.05/6))
            expected=next(r for r in summary['screens'] if r['tariff']==tariff and r['comparison']==name and r['primary'])
            assert abs(upper-expected['bootstrap_upper_six_screen'])<1e-10
    result=dict(files_verified=len(manifest['files']),households=138,windows=269,primary_rows=4035,independently_recomputed_screens=6,passed=True)
    print(json.dumps(result,indent=2));return result

def materialize_reference(out):
    out=fresh(out)
    for name in ('cases.json','configurations.json','execution_matrix.csv','diagnostic_matrix.json'):shutil.copy2(DESIGN/name,out/name)
    shutil.copytree(ROOT/'reference/cases',out/'cases')
    for p in (out/'cases').glob('*/*.json.gz'):
        with gzip.open(p,'rb') as src,p.with_suffix('').open('wb') as dst:shutil.copyfileobj(src,dst)
        p.unlink()
    for kind in ('diagnostics','mechanism'):shutil.copytree(ROOT/'reference'/kind,out/kind)
    from run_household_diagnostics import read_inputs
    from household_controller import ForecastBook
    from household_meter import tariff_bands
    import numpy as np
    for folder in (out/'cases').iterdir():
        readings=read_inputs(folder/'inputs.npz')
        # The saved analysis uses flat-tariff primary-cadence forecasts only.
        for method in ('weekday','previous_day','persistence'):
            book=ForecastBook(readings,tariff_bands(30,'flat'),method,1)
            np.savez_compressed(folder/f'forecast_flat_{method}_1.npz',prediction=book.pred,rho=book.rho,prices=book.prices)
    write(out/'execution_completion.json',dict(all_cases_complete=True,scope='Reanalysis of the published saved summaries; no new controller execution'))
    return out

def compare_tree(actual,expected,path='summary',atol=1e-10):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise AssertionError('Keys differ: '+path)
        for key in expected:compare_tree(actual[key],expected[key],path+'.'+key,atol)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise AssertionError('Length differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare_tree(a,b,f'{path}[{i}]',atol)
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if abs(actual-expected)>atol:raise AssertionError(f'{path}: {actual} vs {expected}')
    elif actual!=expected:raise AssertionError(f'{path}: {actual} vs {expected}')

def review(out):
    verify_reference();out=materialize_reference(out)
    from analyze_household_comparison import analyze
    summary=analyze(out)
    compare_tree(summary,read(ROOT/'reference/analysis/summary.json'))
    figures(out)
    write(out/'reproduction-check.json',dict(passed=True,scope='All saved-summary analysis fields matched at absolute tolerance 1e-10; six figures regenerated; controllers not rerun',at_utc=now()))
    print('Complete saved-summary analysis reproduced:',out)

def figures(base):
    from plot_household_comparison import plot
    from matplotlib.figure import Figure
    original=Figure.savefig
    def save(self,path,*args,**kw):
        if Path(path).suffix=='.png':original(self,Path(path).with_suffix('.pdf'),bbox_inches='tight')
        return original(self,path,*args,**kw)
    Figure.savefig=save
    try:plot(Path(base))
    finally:Figure.savefig=original

def run_worker(base,case):
    from run_household_comparison_v3 import work_case
    return work_case(str(ROOT),str(base),case)

def execute(mode,out,workers,case_id=None):
    verify_reference()
    if workers<1:raise ValueError('workers must be positive')
    source=ROOT/'data/raw/ausgrid.zip'
    if not source.exists():raise RuntimeError('First run: python reproduce.py fetch-data')
    if sha(source)!=read(ROOT/'data/metadata/source_adjudication.json')['source_archive_sha256']:raise ValueError('Source hash mismatch')
    cases=read(DESIGN/'cases.json');configs=read(DESIGN/'configurations.json')
    if mode=='smoke':
        selected=case_id or cases[0]['case_id'];cases=[c for c in cases if c['case_id']==selected]
        if len(cases)!=1:raise ValueError('Unknown registered case ID')
    elif case_id:raise ValueError('--case is only for smoke')
    if mode=='primary':configs=[c for c in configs if c['group']=='primary']
    out=Path(out).resolve()
    required=20*1024**3 if mode=='full' else 2*1024**3
    anchor=out.parent
    while not anchor.exists():anchor=anchor.parent
    if shutil.disk_usage(anchor).free<required:raise ValueError(f'Need {required//1024**3} GiB free before starting {mode}')
    out=fresh(out)
    for name in ('execution_matrix.csv','diagnostic_matrix.json','matrix_counts.json'):shutil.copy2(DESIGN/name,out/name)
    write(out/'cases.json',cases);write(out/'configurations.json',configs)
    source_files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*')) if p.is_file()}
    for name in source_files:
        dest=out/'release_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(out/'execution_freeze.json',dict(source_sha256=source_files,design_sha256={name:sha(out/name) for name in ('cases.json','configurations.json','execution_matrix.csv','diagnostic_matrix.json')},scope='Portable reproduction of the completed v3 study; numerical method and registered matrix unchanged'))
    import numpy,scipy
    write(out/'execution_started.json',dict(at_utc=now(),mode=mode,workers=workers,python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,backend=os.environ.get('GSC_BACKEND','native'),source_sha256=sha(source)))
    start=time.perf_counter();completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_worker,str(out),c):c for c in cases}
        for future in concurrent.futures.as_completed(pending):
            try:r=future.result()
            except Exception as exc:r=dict(case_id=pending[future]['case_id'],status='unexpected_failure',error=repr(exc))
            completed.append(r)
            if r['status']!='complete':write(out/'abort.json',r)
            progress=dict(completed=len(completed),total=len(cases),logical_rows=sum(x.get('logical_rows',0) for x in completed),failures=[x for x in completed if x['status']!='complete'],elapsed_seconds=time.perf_counter()-start)
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    passed=len(completed)==len(cases) and all(r['status']=='complete' for r in completed)
    write(out/'execution_completion.json',dict(all_cases_complete=passed,cases=completed,mode=mode,elapsed_seconds=time.perf_counter()-start,at_utc=now()))
    if not passed:raise RuntimeError('Execution failed; see preserved abort/failure records')
    # Compare every reproduced metric except machine-dependent elapsed time.
    diffs=[];maxdiff=0.;compared=0
    for case in cases:
        cid=case['case_id']
        with gzip.open(ROOT/'reference/cases'/cid/'results.json.gz','rt') as f:old={r['row_id']:r for r in json.load(f)['rows']}
        for r in read(out/'cases'/cid/'results.json')['rows']:
            expected=old[r['row_id']]
            if r['status']!=expected['status']:raise AssertionError('Status differs: '+r['row_id'])
            if r['status']!='ok':continue
            for key,v in r['metrics'].items():
                if key=='timing_seconds':continue
                if v is None or expected['metrics'][key] is None:
                    if v is not expected['metrics'][key]:raise AssertionError('Missing metric mismatch: '+r['row_id']+' '+key)
                    continue
                delta=abs(v-expected['metrics'][key]);maxdiff=max(maxdiff,delta);compared+=1
                if delta>1e-8:diffs.append(dict(row=r['row_id'],metric=key,absolute_difference=delta))
    check=dict(passed=not diffs,compared_numeric_metrics=compared,max_absolute_difference=maxdiff,tolerance=1e-8,differences=diffs,scope=mode)
    write(out/'metric-reproduction-check.json',check);print(json.dumps(check,indent=2))
    if diffs:raise AssertionError('Numerical differences exceed the declared tolerance; preserved in metric-reproduction-check.json')
    print('Execution complete:',out)

def postprocess(base):
    base=Path(base).resolve()
    if not read(base/'execution_completion.json').get('all_cases_complete'):raise ValueError('Execution incomplete')
    if read(base/'execution_completion.json').get('mode')!='full':raise ValueError('Full analysis requires the full frozen matrix; smoke/primary are explicitly partial runs')
    steps=['audit_household_event_rounding.py','audit_household_comparison.py','run_household_diagnostics_v3.py','audit_household_references.py','analyze_household_mechanism.py','check_flat_greedy_model.py','analyze_household_comparison.py']
    ledger=[]
    for name in steps:
        print('Running',name,flush=True);started=time.perf_counter()
        command=[sys.executable,str(ROOT/'src'/name),'--package',str(base)]
        if name=='check_flat_greedy_model.py':command += ['--output',str(base/'flat_greedy_household_check.json')]
        with (base/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ledger.append(dict(stage=name,returncode=result.returncode,seconds=time.perf_counter()-started));write(base/'postprocessing.json',ledger)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {base/name}.log')
    figures(base);print('Full analysis and figures:',base/'analysis')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('build');sub.add_parser('verify');sub.add_parser('test')
    p=sub.add_parser('fetch-data');p.add_argument('--from-file',type=Path)
    p=sub.add_parser('review');p.add_argument('--output',type=Path,default=ROOT/'runs/review')
    p=sub.add_parser('run');p.add_argument('--mode',choices=['smoke','primary','full'],default='smoke');p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--case')
    p=sub.add_parser('postprocess');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':build()
    elif args.command=='fetch-data':fetch_data(args.from_file)
    elif args.command=='verify':verify_reference()
    elif args.command=='review':review(args.output)
    elif args.command=='run':execute(args.mode,args.output,args.workers,args.case)
    elif args.command=='postprocess':postprocess(args.output)
    elif args.command=='test':
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
