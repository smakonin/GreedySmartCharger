"""Publication figures from the saved household analysis tables."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_household_comparison import TARIFFS,POLICIES

COLORS={'PFG':'#52616B','THB':'#DF8C28','OMG':'#9473AB','MPC':'#167C9B','SCR-repair':'#2D9566'}
MARKERS={'PFG':'o','THB':'s','OMG':'^','MPC':'o','SCR-repair':'D'}
LABEL={'flat':'Flat positive tariff','tou':'Positive time-of-use','signed':'Signed-price stress'}

def plot(base,output_directory=None):
    out=base/'analysis';figs=Path(output_directory) if output_directory is not None else out/'figures';figs.mkdir(parents=True,exist_ok=True);s=json.loads((out/'summary.json').read_text());hp=list(csv.DictReader((out/'primary_household_points.csv').open()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'gsc-household-comparison-v3'})
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(figs/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Creator':'GreedySmartCharger household study'});plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),sharey=False,layout='constrained')
    for ax,tariff in zip(axs,TARIFFS):
        for i,policy in enumerate(POLICIES[1:]):
            a=np.array([float(r['cost_difference_from_pfg']) for r in hp if r['tariff']==tariff and r['policy']==policy]);offset=np.random.default_rng(801).uniform(-.14,.14,len(a))
            ax.scatter(i+offset,a,color=COLORS[policy],alpha=.4,s=10,linewidths=0)
            st=s['primary'][tariff][policy]['versus_pfg'];lo,hi=st['bootstrap_95_percentile'];ax.errorbar(i,st['mean'],yerr=[[st['mean']-lo],[hi-st['mean']]],color='black',marker='D',markersize=4,capsize=4,zorder=6)
        ax.axhline(0,color='#687278',lw=.8);ax.set_xticks(range(4),['THB','OMG','MPC','SCR']);ax.set_title(LABEL[tariff]);ax.set_ylabel('Controller − PFG cost (CU / 30 days)')
    fig.suptitle('Household variation remains visible in the matched comparison',fontsize=15)
    fig.supxlabel('Dots: 138 household means; diamonds: equal-household mean and exploratory 95% bootstrap interval. Negative favours the controller. Panel scales differ.',fontsize=9)
    save(fig,'household_cost_differences')
    fig,axs=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
    for j,tariff in enumerate(TARIFFS):
        for policy in POLICIES[1:]:
            a=s['primary'][tariff][policy];p=s['primary'][tariff]['PFG'];dy=a['cost']['mean']-p['cost']['mean'];axs[0,j].scatter(a['efc']['mean']-p['efc']['mean'],dy,color=COLORS[policy],marker=MARKERS[policy],s=65,label=policy.replace('-repair',''));axs[1,j].scatter(a['peak_kw']['mean']-p['peak_kw']['mean'],dy,color=COLORS[policy],marker=MARKERS[policy],s=65)
        for ax in axs[:,j]:ax.axhline(0,color='#aaa',lw=.7);ax.axvline(0,color='#aaa',lw=.7);ax.set_ylabel('Cost difference from PFG (CU)')
        axs[0,j].set_title(LABEL[tariff]);axs[0,j].set_xlabel('Battery-side EFC difference');axs[1,j].set_xlabel('Half-hour import peak difference (kW)')
    axs[0,0].legend(fontsize=8);fig.suptitle('Energy cost, cycling and import peaks are distinct outcomes',fontsize=15);save(fig,'cost_cycling_peak_tradeoffs')
    fig,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,tariff in zip(axs,TARIFFS):
        for policy in ('MPC','SCR-repair'):
            rows=[r for r in s['wear'] if r['tariff']==tariff and r['policy']==policy];ax.plot([r['coefficient_cu_per_ac_kwh'] for r in rows],[r['mean'] for r in rows],linestyle='--' if policy=='SCR-repair' else '-',marker=MARKERS[policy],color=COLORS[policy],label=policy.replace('-repair',''),ms=4)
        ax.axhline(0,color='#aaa',lw=.8);ax.set_title(LABEL[tariff]);ax.set_xlabel('Assumed penalty (CU / discharged AC-kWh)');ax.set_ylabel('Cost + throughput penalty − PFG (CU)')
    axs[0].legend();fig.suptitle('Fixed-trajectory throughput accounting',fontsize=15);fig.supxlabel('Accounting scenarios only: no battery aging model and no wear-aware reoptimization.',fontsize=9);save(fig,'throughput_penalty_sensitivity')
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,tariff in zip(axs,TARIFFS):
        for policy in ('MPC','SCR-repair'):
            vals=[]
            for v in (0.,.076,.1425):
                if v==0:vals.append(s['primary'][tariff][policy]['objective']['mean']-s['primary'][tariff]['PFG']['objective']['mean'])
                else:
                    table=list(csv.DictReader((out/'configuration_household_means.csv').open()));rr={r['policy']:r for r in table if r['tariff']==tariff and r['group']==f'endpoint_grid_501_{v:g}'};vals.append(float(rr[policy]['objective'])-float(rr['PFG']['objective']))
            ax.plot([0,.076,.1425],vals,linestyle='--' if policy=='SCR-repair' else '-',marker=MARKERS[policy],color=COLORS[policy],label=policy.replace('-repair',''))
        ax.axhline(0,color='#aaa',lw=.8);ax.set_title(LABEL[tariff]);ax.set_xlabel('Terminal value (CU / stored-kWh)');ax.set_ylabel('Terminal-adjusted objective − PFG (CU)')
    axs[0].legend();fig.suptitle('Endpoint convention can alter the usefulness comparison',fontsize=15);save(fig,'terminal_sensitivity')
    fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,tariff in zip(axs,TARIFFS):
        rows=[r for r in s['regimes'] if r['tariff']==tariff and r['feature']=='pv_load_ratio' and r['n_households']]
        for i,r in enumerate(rows):
            ci=r.get('bootstrap_95_percentile');ax.scatter(i,r['mean'],color=COLORS['MPC'],s=45)
            if ci:ax.errorbar(i,r['mean'],yerr=[[r['mean']-ci[0]],[ci[1]-r['mean']]],color=COLORS['MPC'],capsize=4)
        ax.axhline(0,color='#aaa',lw=.8);ax.set_xticks(range(len(rows)),[r['bin']+'\n(n='+str(r['n_households'])+')' for r in rows]);ax.set_title(LABEL[tariff]);ax.set_xlabel('Pre-test PV / load energy ratio');ax.set_ylabel('MPC − PFG cost (CU)')
    fig.suptitle('Predefined household regimes: exploratory contrasts',fontsize=15);fig.supxlabel('Windows enter their own history bin; a household may contribute to more than one bin. No interval for fewer than 20 homes.',fontsize=9);save(fig,'history_regimes')
    points=list(csv.DictReader((out/'primary_case_points.csv').open()))
    fig,axs=plt.subplots(1,3,figsize=(14,4.9),layout='constrained')
    selected=[]
    for ax,tariff in zip(axs,TARIFFS):
        case=sorted([x for x in points if x['tariff']==tariff and x['policy']=='MPC'],key=lambda x:(-float(x['cost_difference_from_pfg']),x['case_id']))[0]
        selected.append({'tariff':tariff,'case_id':case['case_id'],'mpc_minus_pfg':float(case['cost_difference_from_pfg'])})
        for policy in ('MPC','SCR-repair'):
            with np.load(base/'cases'/case['case_id']/(tariff+'__primary__'+policy+'.npz')) as f:
                cols=json.loads(str(f['layout']))['trace']['numeric'];tr={k:f['trace'][:,i] for i,k in enumerate(cols)}
            ax.plot(tr['end']/86400,tr['regret'],color=COLORS[policy],lw=1.1,label=policy.replace('-repair','')+' cost difference')
            if policy=='SCR-repair':ax.plot(tr['end']/86400,tr['segment_peak'],color='#30383B',linestyle='--',lw=.8,label='SCR segment maximum: cost + reserve')
        ax.axhline(1.,color='#9D7040',linestyle=':',lw=1.1,label='1 CU allowance');ax.axhline(0.,color='#aaa',lw=.6)
        ax.set_title(LABEL[tariff]+'\n'+case['case_id']);ax.set_xlabel('Elapsed days');ax.set_ylabel('Cost difference / exposure (CU)')
    axs[0].legend(fontsize=7,loc='best');fig.suptitle('Protection in windows with the largest MPC − PFG cost difference',fontsize=14)
    fig.supxlabel('One outcome-selected illustrative window per tariff; ties resolved by case ID. Not typical-household estimates. Panel scales differ.',fontsize=9)
    save(fig,'selected_protection_trajectories')
    (figs/'selected_trajectory_rule.json').write_text(json.dumps({'rule':'largest primary MPC minus PFG cost per tariff; ascending case ID breaks ties','scope':'exploratory outcome-selected illustrations, not additional statistical evidence','cases':selected},indent=2)+'\n')
    return [str(x) for x in sorted(figs.glob('*.png'))]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();print(json.dumps(plot(a.package),indent=2))
