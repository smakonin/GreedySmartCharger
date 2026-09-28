"""Continuous-time copying/repair liabilities, including nonnegative terminal value.

Prices have a fixed deterministic envelope, load/PV have no forecast restriction.
Seconds index time; states are stored kWh. This does not alter earlier releases.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PriceBand:
    start: float
    end: float
    low: float
    high: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.start,self.end,self.low,self.high)) or self.end<=self.start or self.low>self.high:
            raise ValueError('invalid price band')


class ExecutedLiability:
    def __init__(self, bands, battery, terminal_value=0.):
        self.bands=tuple(bands);self.b=battery;self.v=float(terminal_value)
        if not self.bands or not math.isfinite(self.v) or self.v<0:
            raise ValueError('finite nonnegative terminal value and nonempty schedule required')
        if any(a.end!=b.start for a,b in zip(self.bands,self.bands[1:])):
            raise ValueError('price schedule must be contiguous and ordered')
        self.start=self.bands[0].start;self.end=self.bands[-1].end
        self.d=battery.discharge_kw/battery.eta_discharge/3600
        self.q=battery.charge_kw*battery.eta_charge/3600
        self.a=max(self.v,battery.eta_discharge*max(0.,max(x.high for x in self.bands)))
        self.h=max(0.,-min(x.low for x in self.bands))/battery.eta_charge
        self.boundaries=sorted({x.start for x in self.bands}|{self.end})

    def price_in_range(self,t,price):
        if not math.isfinite(price):return False
        return any(x.start<=t<x.end and x.low<=price<=x.high for x in self.bands)

    def next_boundary(self,t):
        return min(x for x in self.boundaries if x>t)

    def _check(self,t,s,z):
        if not all(math.isfinite(x) for x in (t,s,z)) or not self.start<=t<=self.end or not (0<=s<=self.b.capacity and 0<=z<=self.b.capacity):
            raise ValueError('state/time outside liability domain')

    def _items(self,t,s,deficit):
        rate=self.d if deficit else self.q
        common=s if deficit else self.b.capacity-s
        clear=t+common/rate if rate>0 else math.inf
        out=[]
        for band in self.bands:
            cap=rate*max(0.,band.end-max(band.start,clear))
            weight=max(0.,self.b.eta_discharge*band.high-self.v) if deficit else max(0.,-band.low)/self.b.eta_charge
            if weight>0:out.append((weight,cap))
        return sorted(out,key=lambda item:-item[0])

    def copy(self,t,s,z):
        self._check(t,s,z)
        if s==z:return 0.
        gap=abs(z-s);value=self.v*gap if s<z else 0.
        for weight,cap in self._items(t,s,s<z):
            take=min(gap,cap);value+=weight*take;gap-=take
        return value

    def repair(self,t,s,z):
        self._check(t,s,z)
        if s>z:return math.inf
        if s==z:return 0.
        deadline=min(self.end,t+(self.b.capacity-s)/self.q) if self.q>0 else self.end
        coefficient=0.
        for band in self.bands:
            if band.end>t and band.start<deadline:
                coefficient=max(coefficient,max(0.,band.high)/self.b.eta_charge)
        gap=z-s
        unresolved=min(gap,max(0.,self.b.capacity-s-self.q*(self.end-t)))
        return coefficient*gap+max(0.,self.v-coefficient)*unresolved

    def value(self,kind,t,s,z):
        self._check(t,s,z)
        if kind=='global':return self.a*max(0.,z-s)+self.h*max(0.,s-z)
        if kind=='copy':return self.copy(t,s,z)
        if kind=='repair':return min(self.copy(t,s,z),self.repair(t,s,z))
        raise ValueError('unknown certificate')

    def fallback(self,kind,t,s,z):
        if kind not in ('global','copy','repair'):raise ValueError('unknown certificate')
        return 'repair' if kind=='repair' and self.repair(t,s,z)<self.copy(t,s,z) else 'copy'

    @staticmethod
    def _root(points,left,right,fleft,fright):
        if fright!=fleft:
            r=left-fleft*(right-left)/(fright-fleft)
            if left<r<right:points.add(r)

    def affine_breaks(self,kind,t,duration,s,ds,z,dz):
        """All formula-change times along affine SOCs, in seconds from t.

        Includes sign, price-band, common-energy clearing, sorted-knapsack
        capacity, repair deadline and copy/repair branch intersections.
        One-sided values at repair-deadline jumps are handled by max_along().
        """
        pts={0.,float(duration)}
        self._root(pts,0.,duration,z-s,z-s+(dz-ds)*duration)
        for boundary in self.boundaries:
            if 0<boundary-t<duration:pts.add(boundary-t)
        if kind=='global':return sorted(pts)
        for left,right in zip(sorted(pts),sorted(pts)[1:]):
            mid=(left+right)/2;deficit=s+ds*mid<z+dz*mid
            rate=self.d if deficit else self.q
            if rate>0:
                def clear(u):return t+u+((s+ds*u) if deficit else (self.b.capacity-s-ds*u))/rate
                for boundary in self.boundaries:
                    self._root(pts,left,right,clear(left)-boundary,clear(right)-boundary)
            if kind=='repair' and self.q>0:
                def deadline(u):return t+u+(self.b.capacity-s-ds*u)/self.q
                for boundary in self.boundaries:
                    self._root(pts,left,right,deadline(left)-boundary,deadline(right)-boundary)
        if kind=='repair':
            # Terminal deficit remaining after maximum-rate repair is bounded by
            # min(gap, max(0, C-s-q*(T-t))). Both clamps create affine kinks.
            def unresolved(u):return self.b.capacity-s-ds*u-self.q*(self.end-t-u)
            for left,right in zip(sorted(pts),sorted(pts)[1:]):
                self._root(pts,left,right,unresolved(left),unresolved(right))
                self._root(pts,left,right,unresolved(left)-(z-s+(dz-ds)*left),unresolved(right)-(z-s+(dz-ds)*right))
        for left,right in zip(sorted(pts),sorted(pts)[1:]):
            deficit=s+ds*(left+right)/2<z+dz*(left+right)/2
            items0=self._items(t+left,s+ds*left,deficit)
            items1=self._items(t+right,s+ds*right,deficit)
            c0=c1=0.
            for (_,a),(_,b) in zip(items0,items1):
                c0+=a;c1+=b
                self._root(pts,left,right,c0-abs(z-s+(dz-ds)*left),c1-abs(z-s+(dz-ds)*right))
        if kind=='repair':
            for left,right in zip(sorted(pts),sorted(pts)[1:]):
                x=left+(right-left)/3;y=left+2*(right-left)/3
                ux=self.repair(t+x,s+ds*x,z+dz*x);uy=self.repair(t+y,s+ds*y,z+dz*y)
                if math.isfinite(ux) and math.isfinite(uy):
                    fx=self.copy(t+x,s+ds*x,z+dz*x)-ux
                    fy=self.copy(t+y,s+ds*y,z+dz*y)-uy
                    if fx!=fy:
                        cross=x-fx*(y-x)/(fy-fx)
                        if left<cross<right:pts.add(cross)
        return sorted(pts)

    def max_along(self,kind,t,duration,s,ds,z,dz,regret,cost_rate):
        """Supremum of R + Phi over one constant-power segment, all prefixes.

        Exact piecewise-affine construction in real arithmetic; floating-point
        evaluation is tested against dense independent grids. Candidate filters
        use a strict budget test, never a per-step spending allowance.
        """
        if duration<=0:raise ValueError('positive segment duration required')
        def f(u):
            # Only roundoff at previously validated state boundaries is clamped.
            ss=s+ds*u;zz=z+dz*u
            if not (-1e-10<=ss<=self.b.capacity+1e-10 and -1e-10<=zz<=self.b.capacity+1e-10):
                raise ValueError('affine path violates SOC bounds')
            return regret+cost_rate*u+self.value(kind,t+u,min(self.b.capacity,max(0.,ss)),min(self.b.capacity,max(0.,zz)))
        points=self.affine_breaks(kind,t,duration,s,ds,z,dz)
        peak=max(f(u) for u in points)
        for left,right in zip(points,points[1:]):
            if right-left<1e-10:continue
            x=left+(right-left)/3;y=left+2*(right-left)/3
            fx,fy=f(x),f(y)
            # Affine one-sided limits also cover upward/downward repair jumps.
            peak=max(peak,2*fx-fy,2*fy-fx)
        return peak
