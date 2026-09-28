"""Versioned numerical correction for internal affine certificate probes.

The liability formulas, budget admission and real-arithmetic controller are
unchanged. See HOUSEHOLD_EXECUTION_AMENDMENT_V2.md for the observed failure.
"""
import math
from executed_liability import ExecutedLiability as OriginalLiability

class ExecutedLiability(OriginalLiability):
    def _bounded(self,x):
        if not math.isfinite(x) or not -1e-10<=x<=self.b.capacity+1e-10:
            raise ValueError('affine path violates SOC bounds')
        return min(self.b.capacity,max(0.,x))

    def affine_breaks(self,kind,t,duration,s,ds,z,dz):
        """All formula-change times along affine SOCs, in seconds from t.

        Includes sign, price-band, common-energy clearing, sorted-knapsack
        capacity, repair deadline and copy/repair branch intersections.
        One-sided values at repair-deadline jumps are handled by max_along().
        """
        # Validate both endpoints before any internal probe. Affine paths remain
        # inside their endpoint range; only the existing 1e-10 roundoff envelope
        # is permitted. This does not add a spending allowance to the guard.
        for u in (0.,duration):
            self._bounded(s+ds*u);self._bounded(z+dz*u)
        def probe(fn,u):
            return fn(t+u,self._bounded(s+ds*u),self._bounded(z+dz*u))
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
                ux=probe(self.repair,x);uy=probe(self.repair,y)
                if math.isfinite(ux) and math.isfinite(uy):
                    fx=probe(self.copy,x)-ux
                    fy=probe(self.copy,y)-uy
                    if fx!=fy:
                        cross=x-fx*(y-x)/(fy-fx)
                        if left<cross<right:pts.add(cross)
        return sorted(pts)

