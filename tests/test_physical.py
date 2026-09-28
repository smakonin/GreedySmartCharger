import itertools,math,random,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_simulator import *
from libModels import Battery,Solar,model_1a_greedy

class PhysicalTests(unittest.TestCase):
    def test_losses_and_half_hour_power(self):
        b=BatterySpec(10,2,2,.8,.9)
        x=Interval(0,2,.1,.5);a=self_consumption(x,0,b);f=transition(x,0,a,b)
        self.assertAlmostEqual(a.charge,1);self.assertAlmostEqual(f.end_soc,.8);self.assertAlmostEqual(f.loss,.2)
        y=Interval(2,0,.1,.5);a=self_consumption(y,f.end_soc,b);g=transition(y,f.end_soc,a,b)
        self.assertAlmostEqual(a.discharge,.72);self.assertAlmostEqual(g.loss,.08);self.assertAlmostEqual(g.end_soc,0)
    def test_reject_impossible_actions(self):
        b=BatterySpec();x=Interval(1,2,.1)
        for a in [Action(1,2,0),Action(0,1,1),Action(0,0,2),Action(-1,0,0),Action(0,float('nan'),0),Action(0,6,0)]:
            with self.assertRaises(ValueError): transition(x,0,a,b)
    def test_invalid_inputs(self):
        for kw in [dict(eta_charge=0),dict(eta_discharge=1.1),dict(capacity=-1),dict(charge_kw=float('inf'))]:
            with self.assertRaises(ValueError):BatterySpec(**kw)
        for vals in [(-1,0,0,1),(0,0,float('nan'),1),(0,0,0,0)]:
            with self.assertRaises(ValueError):Interval(*vals)
    def test_legacy_greedy_equivalence(self):
        rng=random.Random(171);b=BatterySpec(15,5,5,1,1); old=Battery();soc=old.state;s=Solar()
        for _ in range(10000):
            x=Interval(rng.random()*8,rng.random()*8,rng.choice([-.1,0,.08,.15]));s.state=x.pv
            cost=model_1a_greedy(x.load,x.price,s,old)
            f=transition(x,soc,greedy_charge_first(x,soc,b),b)
            self.assertAlmostEqual(cost,f.cost);self.assertAlmostEqual(old.state,f.end_soc);soc=f.end_soc
    def test_randomized_conservation(self):
        rng=random.Random(99)
        for ctrl in CONTROLLERS.values():
            b=BatterySpec(12,3,4,.91,.94);soc=0;net=0
            for _ in range(15000):
                x=Interval(rng.random()*10,rng.random()*10,rng.uniform(-.2,.5),rng.choice([.25,.5,1]))
                a=ctrl(x,soc,b);f=transition(x,soc,a,b)
                net+=x.pv+f.grid-x.load-f.curtailed-f.loss;soc=f.end_soc
            self.assertAlmostEqual(net,soc,places=8)
    def test_analytic_price_threshold(self):
        b=BatterySpec(1,1,1,.9,.9)
        for future in [.05,.1,.2]:
            xs=[Interval(1,1,.1),Interval(.81,0,future)]
            g=simulate(xs,greedy_charge_first,b);s=simulate(xs,self_consumption,b)
            self.assertAlmostEqual(s['cost']-g['cost'],.81*future-.1)
    def test_counterexamples_and_oracle(self):
        b=BatterySpec(1,1,1,1,1)
        xs=[Interval(1,1,.1),Interval(1,0,.3)]
        self.assertAlmostEqual(integer_oracle(xs,1,1),.1)
        self.assertAlmostEqual(simulate(xs,greedy_charge_first,b)['cost'],.1)
        self.assertAlmostEqual(simulate(xs,self_consumption,b)['cost'],.3)
        xs=[Interval(1,0,.1),Interval(1,0,.3)]
        self.assertAlmostEqual(integer_oracle(xs,1,1,1),.1)
        self.assertAlmostEqual(simulate(xs,self_consumption,b,1)['cost'],.3)
    def test_flat_price_pv_first_small_exhaustive(self):
        # Exhaustive evidence for this finite model, not a general proof.
        b=BatterySpec(2,1,1,1,1)
        for pairs in itertools.product(itertools.product(range(2),repeat=2),repeat=4):
            xs=[Interval(l,p,.1) for l,p in pairs]
            for initial in [0,1,2]:
                self.assertAlmostEqual(simulate(xs,self_consumption,b,initial)['cost'],integer_oracle(xs,2,1,initial))
    def test_three_period_ratio_counterexample(self):
        b=BatterySpec(1,1,1,1,1)
        for epsilon in [.1,.01,.001]:
            xs=[Interval(0,1,.1),Interval(1,0,epsilon),Interval(1,0,1)]
            self.assertAlmostEqual(integer_oracle(xs,1,1),epsilon)
            for fn in CONTROLLERS.values():
                m=simulate(xs,fn,b);self.assertAlmostEqual(m["cost"],1);self.assertAlmostEqual(m["final_soc"],0)

    def test_nonoverlap_equivalence(self):
        rng=random.Random(18);b=BatterySpec(3,1,2,.9,.95)
        xs=[]
        for i in range(500):
            value=rng.random()*5
            xs.append(Interval(value if i%2 else 0,0 if i%2 else value,rng.random()))
        self.assertEqual(simulate(xs,greedy_charge_first,b),simulate(xs,self_consumption,b))

    def test_negative_price_avoid_free_pv(self):
        x=Interval(1,1,-.1);b=BatterySpec(1,1,1,1,1)
        self.assertAlmostEqual(simulate([x],greedy_charge_first,b)['cost'],-.1)
        self.assertAlmostEqual(simulate([x],self_consumption,b)['cost'],0)

if __name__=='__main__':unittest.main()
