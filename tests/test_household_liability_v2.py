import json,math,unittest
from pathlib import Path
import numpy as np
from executed_liability import ExecutedLiability as Original,PriceBand
from household_liability_v2 import ExecutedLiability
from household_meter import tariff_bands,replay_meter
from household_controller import ForecastBook,HouseholdController
from physical_simulator import BatterySpec
from run_household_diagnostics import read_inputs
from household_artifacts import verify_result

class BoundaryTests(unittest.TestCase):
    def test_recorded_roundoff_and_invalid_paths(self):
        b=BatterySpec();bands=tariff_bands(30,'flat');m=ExecutedLiability(bands,b)
        args=('repair',2188829.216942748,11.468209309311707,0.,0.,0.003748964329769148,-0.00032690058479532163)
        with self.assertRaises(ValueError):Original(bands,b).affine_breaks(*args)
        value=m.max_along(*args,0.,0.)
        self.assertTrue(math.isfinite(value));self.assertGreaterEqual(value,0.)
        for state in (-1e-7,b.capacity+1e-7,float('nan')):
            with self.assertRaises(ValueError):m.max_along('repair',0,100,state,0,1,0,0,0)
        with self.assertRaises(ValueError):m.value('repair',0,-1e-12,1)

    def test_interior_equivalence_and_dense_bound(self):
        rng=np.random.default_rng(513);b=BatterySpec();bands=tariff_bands(2,'signed')
        for v in (0.,.076,.1425):
            m=ExecutedLiability(bands,b,v);old=Original(bands,b,v)
            for k in range(36):
                t=float(rng.uniform(0,m.end-1800));duration=1800.;s,z,s2,z2=rng.uniform(.01,14.99,4)
                ds=(s2-s)/duration;dz=(z2-z)/duration;regret=.01;dc=float(rng.uniform(-.0001,.0001))
                for kind in ('global','copy','repair'):
                    a=m.max_along(kind,t,duration,s,ds,z,dz,regret,dc)
                    self.assertEqual(a,old.max_along(kind,t,duration,s,ds,z,dz,regret,dc))
                    dense=max(regret+dc*u+m.value(kind,t+u,s+ds*u,z+dz*u) for u in np.linspace(0,duration,101))
                    self.assertGreaterEqual(a+1e-10,dense)

    def test_failed_household_configuration_replays_with_accounting(self):
        path=Path(__file__).resolve().parents[1]/'reference/cases/A003_20120701/inputs.npz'
        rr=read_inputs(path);m=ExecutedLiability(tariff_bands(30,'flat'),BatterySpec())
        r=replay_meter(rr,m,HouseholdController('MPC',501,0.,ForecastBook(rr,m.bands)),certificate='repair',budget=1.)
        verify_result(r,m.b,0.,'repair',1.)
        self.assertLessEqual(r['max_prefix_exposure'],1+1e-8)
        self.assertEqual(len(r['intervals']),1440)

if __name__=='__main__':unittest.main()
