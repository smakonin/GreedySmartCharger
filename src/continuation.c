/* Same binary64 range-minimum DP recurrence as fast_interval_planner.py.
   Compile without fast-math or floating-point contraction. No control changes. */
#include <math.h>
#include <stdlib.h>
static double smaller(double a,double b){return a<b?a:b;}
static int imin(int a,int b){return a<b?a:b;}
static int imax(int a,int b){return a>b?a:b;}
static void window_min(const double *v,int n,int lo,int hi,double *out,int *queue){
    lo=imax(-(n-1),lo);hi=imin(n-1,hi);
    if(lo>hi){for(int i=0;i<n;i++)out[i]=INFINITY;return;}
    int head=0,tail=0,added=-1;
    for(int i=0;i<n;i++){
        int end=imin(n-1,i+hi),start=imax(0,i+lo);
        while(added<end){
            ++added;
            while(tail>head && v[queue[tail-1]]>=v[added])--tail;
            queue[tail++]=added;
        }
        while(head<tail && queue[head]<start)++head;
        out[i]=(head<tail)?v[queue[head]]:INFINITY;
    }
}
int continuation(const double *xs,int steps,const double *levels,int n,double spacing,
                 double charge_kw,double discharge_kw,double eta_c,double eta_d,double terminal,
                 double *answer){
    if(n<2 || steps<0 || spacing<=0 || eta_c<=0 || eta_d<=0)return 1;
    double *mem=malloc((size_t)n*5*sizeof(double));int *queue=malloc((size_t)n*sizeof(int));
    if(!mem || !queue){free(mem);free(queue);return 2;}
    double *value=mem,*out=mem+n,*shifted=mem+2*n,*zero=mem+3*n,*active=mem+4*n;
    for(int j=0;j<n;j++)value[j]=-terminal*levels[j];
    for(int t=steps-1;t>=0;t--){
        const double load=xs[4*t],pv=xs[4*t+1],price=xs[4*t+2],dt=xs[4*t+3];
        int rcharge=imin(n-1,(int)floor(eta_c*(charge_kw*dt+1e-10)/spacing));
        int rdis=imin(n-1,(int)floor((smaller(load,discharge_kw*dt)+1e-10)/eta_d/spacing));
        double net=load-pv;
        for(int j=0;j<n;j++)out[j]=INFINITY;
        for(int mode=0;mode<2;mode++){
            int lo=mode?-rdis:0,hi=mode?0:rcharge;
            double slope=mode?eta_d:1./eta_c;
            for(int j=0;j<n;j++)shifted[j]=value[j]+(price*slope)*levels[j];
            if(price<0){
                window_min(shifted,n,lo,hi,active,queue);
                for(int j=0;j<n;j++)out[j]=smaller(out[j],active[j]+price*(load-slope*levels[j]));
            }else{
                int cut=(int)floor(-net/(slope*spacing));
                window_min(value,n,lo,imin(hi,cut),zero,queue);
                window_min(shifted,n,imax(lo,cut+1),hi,active,queue);
                for(int j=0;j<n;j++)out[j]=smaller(out[j],smaller(zero[j],active[j]+price*(net-slope*levels[j])));
            }
        }
        double *swap=value;value=out;out=swap;
    }
    for(int j=0;j<n;j++)answer[j]=value[j];
    free(mem);free(queue);return 0;
}
