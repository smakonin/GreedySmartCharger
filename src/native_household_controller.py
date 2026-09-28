"""Same causal forecast book with compiled DP continuation execution."""
import time
from household_controller import ForecastBook as ReferenceBook
from native_interval_planner import continuation

class ForecastBook(ReferenceBook):
    def table(self,t,b,states,v):
        key=(t,b.capacity,b.charge_kw,b.discharge_kw,b.eta_charge,b.eta_discharge,states,v)
        if key not in self.values:
            start=time.perf_counter();self.values[key]=continuation(self.intervals(t)[1:],b,states,v)
            self.dp_build_seconds+=time.perf_counter()-start;self.dp_build_count+=1
        return self.values[key]
