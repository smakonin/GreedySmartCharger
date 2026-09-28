"""Execution equivalence, including forecast horizon and SOC-boundary cases."""
import unittest
from dataclasses import asdict
import numpy as np
import fast_interval_planner as reference
import native_interval_planner as native
from grid_model import Interval
from physical_simulator import BatterySpec


class NativePlannerTests(unittest.TestCase):
    def test_seeded_values_and_actions(self):
        rng = np.random.default_rng(2026092803)
        for i in range(400):
            b = BatterySpec(float(rng.choice([5,10,15,30])), float(rng.choice([2.5,5,10])),
                            float(rng.choice([2.5,5,10])), float(rng.choice([.8,.95,1.])),
                            float(rng.choice([.8,.95,1.])))
            states = int(rng.choice([2,11,101,251,501]))
            v = float(rng.choice([0.,.076,.1425]))
            xs = [Interval(float(rng.uniform(0,8)), float(rng.uniform(0,8)),
                           float(rng.choice([-.05,0.,.08,.15])), float(rng.choice([.5,1.])))
                  for _ in range(int(rng.choice([1,2,8,24,48])))]
            levels, old = reference.continuation(xs[1:],b,states,v)
            levels2, new = native.continuation(xs[1:],b,states,v)
            np.testing.assert_array_equal(levels,levels2)
            np.testing.assert_array_equal(old,new)
            # Bit-level differences must consist only of the sign of numeric zero.
            different = old.view(np.uint64) != new.view(np.uint64)
            self.assertTrue(np.all(old[different] == 0.) and np.all(new[different] == 0.))
            soc = float(rng.choice([0., b.capacity, rng.uniform(0,b.capacity)]))
            self.assertEqual(asdict(reference.first_action(xs[0],soc,b,levels,old)),
                             asdict(native.first_action(xs[0],soc,b,levels2,new)))

    def test_exact_kinks_and_empty_horizon(self):
        for states in (2,11,251,501):
            for eta in (.95,1.):
                b = BatterySpec(15.,5.,5.,eta,eta)
                for price in (-.05,0.,.08,.15):
                    for load,pv in ((0.,0.),(2.5,0.),(0.,2.5),(2.5,2.5)):
                        x = Interval(load,pv,price,.5)
                        for future in ([], [x]*47):
                            for soc in (0.,2.375,7.5,15.):
                                a = reference.nominal_action(x,soc,b,future,states,.076)
                                z = native.nominal_action(x,soc,b,future,states,.076)
                                self.assertEqual(asdict(a),asdict(z))


if __name__ == '__main__':
    unittest.main()
