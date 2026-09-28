"""Nonimplementable finite-horizon MILP reference with solver bounds retained."""
import time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix


def solve_hindsight(load,pv,prices,b,initial_soc=0.,terminal_value=0.,time_limit=10.):
    n=len(load);dt=.5;size=6*n
    # c,d,g,pv_used,s_after,charge_mode; AC energies, stored energy, binary mode.
    c=np.zeros(size);c[2*n:3*n]=prices;c[5*n-1]=-terminal_value
    lower=np.zeros(size);upper=np.full(size,np.inf)
    upper[:n]=b.charge_kw*dt;upper[n:2*n]=np.minimum(load,b.discharge_kw*dt)
    upper[3*n:4*n]=pv;upper[4*n:5*n]=b.capacity;upper[5*n:]=1
    integrality=np.zeros(size,int);integrality[5*n:]=1
    A=lil_matrix((4*n,size));lo=np.full(4*n,-np.inf);hi=np.full(4*n,np.inf)
    for t in range(n):
        # g + usedPV + d = load + c
        A[t,2*n+t]=1;A[t,3*n+t]=1;A[t,n+t]=1;A[t,t]=-1;lo[t]=hi[t]=load[t]
        # state change = eta_c*c - d/eta_d
        j=n+t;A[j,4*n+t]=1;A[j,t]=-b.eta_charge;A[j,n+t]=1/b.eta_discharge
        if t:A[j,4*n+t-1]=-1;lo[j]=hi[j]=0.
        else:lo[j]=hi[j]=initial_soc
        j=2*n+t;A[j,t]=1;A[j,5*n+t]=-b.charge_kw*dt;hi[j]=0.
        j=3*n+t;A[j,n+t]=1;A[j,5*n+t]=b.discharge_kw*dt;hi[j]=b.discharge_kw*dt
    start=time.perf_counter()
    result=milp(c,integrality=integrality,bounds=Bounds(lower,upper),constraints=LinearConstraint(A.tocsr(),lo,hi),
                options={'time_limit':time_limit,'mip_rel_gap':1e-6,'disp':False})
    report=dict(status=int(result.status),message=result.message,wall_seconds=time.perf_counter()-start,
                mip_gap=None if getattr(result,'mip_gap',None) is None else float(result.mip_gap),
                dual_bound=None if getattr(result,'mip_dual_bound',None) is None else float(result.mip_dual_bound)+terminal_value*initial_soc,
                primal_objective=None if result.fun is None else float(result.fun)+terminal_value*initial_soc,
                nonimplementable=True,scope='Full-window known load/PV; no export, grid charging, separate efficiencies, no simultaneous charge/discharge. Solver lower bound is not a causal policy cost.')
    solution=None
    if result.x is not None:
        x=result.x;Ax=A@x
        residual=max(float(np.max(np.maximum(lo-Ax,0))),float(np.max(np.maximum(Ax-hi,0))),
                     float(np.max(np.maximum(lower-x,0))),float(np.max(np.maximum(x-upper,0))),
                     float(np.max(np.abs(x[5*n:]-np.round(x[5*n:])))))
        if residual>1e-5:raise AssertionError(('MILP primal feasibility',residual))
        if np.any((x[:n]>1e-5)&(x[n:2*n]>1e-5)):raise AssertionError('simultaneous primal flows')
        report['max_primal_residual']=residual;solution=x.reshape(6,n)
    return report,solution
