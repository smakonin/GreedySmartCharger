import math
import unittest
from dataclasses import replace
from interval_meter import (MeterReading, MeterContext, BatteryRequest, MeterController,
    STEP,HISTORY,forecast,local_command,replay_meter,tariff_bands)
from physical_simulator import BatterySpec
from executed_liability import ExecutedLiability,PriceBand
from grid_model import Interval,transition
from grid_dp import action_for_transition


def readings(days=1, mode='varied'):
    out=[]
    for i in range(-HISTORY,days*48):
        hour=(i%48)/2
        load=1.+.3*(16<=hour<21)
        pv=max(0.,2.*math.sin((hour-6)*math.pi/12))
        if mode=='constant':load,pv=1.,.4
        if mode=='idle':load,pv=0.,0.
        out.append(MeterReading(i*STEP,(i+1)*STEP,load,pv,(i+1)*STEP))
    return out


def model(days=1,kind='signed',v=0.,battery=None):
    return ExecutedLiability(tariff_bands(days,kind),battery or BatterySpec(),v)


class IntervalMeterTests(unittest.TestCase):
    def test_meter_units_end_and_release_validation(self):
        for args in [(0,1800,-1,0,1800),(0,1800,1,0,0),(0,3600,1,0,3600),(0,1800,float('nan'),0,1800)]:
            with self.assertRaises(ValueError):MeterReading(*args)

    def test_first_interval_is_forecast_not_current_energy(self):
        rr=readings();ctx=MeterContext(0,0,tuple(rr[:HISTORY]),86400,tariff_bands(1))
        xs,origins=forecast(ctx)
        self.assertEqual(len(xs),48)
        self.assertTrue(all(o['latest_ready_at']<=0 for o in origins))
        expected=sum(rr[HISTORY-7*48*j].load_kwh for j in range(1,5))/4
        self.assertAlmostEqual(xs[0].load,expected)
        self.assertNotIn('load_kw',ctx.__dataclass_fields__)
        self.assertNotIn('pv_kw',ctx.__dataclass_fields__)

    def test_forecast_modes_use_available_intervals(self):
        rr=readings();ctx=MeterContext(0,0,tuple(rr[:HISTORY]),2*STEP,tariff_bands(1))
        for method in ('weekday','previous_day','persistence'):
            xs,origins=forecast(ctx,method)
            self.assertEqual(len(xs),2)
            self.assertTrue(all(t<=0 for o in origins for t in o['source_ends']))
        self.assertEqual(forecast(ctx,'persistence')[0][0].load,rr[HISTORY-1].load_kwh)

    def test_delayed_and_gap_history_rejected(self):
        rr=readings();hist=list(rr[:HISTORY]);hist[-1]=replace(hist[-1],ready_at=1)
        ctx=MeterContext(0,0,tuple(hist),86400,tariff_bands(1))
        with self.assertRaises(ValueError):forecast(ctx)
        with self.assertRaises(ValueError):forecast(replace(ctx,history=ctx.history[1:]))
        with self.assertRaises(ValueError):replay_meter(rr[:10]+rr[11:],model(),MeterController('PFG'))

    def test_common_projection_recovers_nominal_economic_allocation(self):
        b=BatterySpec();s=7.
        from interval_meter import request_from_action
        for price in (-.05,0.,.15):
            for load,pv in [(2.,.3),(.3,3.),(0.,0.)]:
                x=Interval(load,pv,price,.5)
                for end in (s,s+.3,s-min(.2,load/b.eta_discharge)):
                    a=action_for_transition(x,s,end,b)
                    q=request_from_action(a,.5,price)
                    p=local_command(q,load/.5,pv/.5,price,s,b)
                    self.assertAlmostEqual(a.direct,p.direct*.5)
                    self.assertAlmostEqual(a.charge,p.pv_charge*.5)
                    self.assertAlmostEqual(a.grid_charge,p.grid_charge*.5)
                    self.assertAlmostEqual(a.discharge,p.discharge*.5)

    def test_source_allocation_refreshes_after_empty_and_full(self):
        b=BatterySpec(1,5,5,.95,.95)
        a=local_command(BatteryRequest(-2.),3.,2.,.15,.2,b)
        empty=local_command(BatteryRequest(-2.),3.,2.,.15,0.,b)
        self.assertEqual(a.direct,1.)
        self.assertEqual(empty.direct,2.)
        full=local_command(BatteryRequest(2.),1.,3.,.15,1.,b)
        self.assertEqual(full.pv_charge+full.grid_charge,0.)

    def test_future_energy_and_receipt_mutation_cannot_change_first_request(self):
        rr=readings();other=list(rr)
        for i in range(HISTORY,len(other)):
            other[i]=replace(other[i],load_kwh=9.,pv_kwh=8.,ready_at=other[i].end+123.)
        for name in ('PFG','THB','OMG','MPC'):
            c=MeterController(name,11)
            a=c.decide(MeterContext(0,2.,tuple(rr[:HISTORY]),86400,tariff_bands(1)),BatterySpec())[0]
            b=c.decide(MeterContext(0,2.,tuple(other[:HISTORY]),86400,tariff_bands(1)),BatterySpec())[0]
            self.assertEqual(a,b)
        # Full executor prefix poison test: later values cannot change past outcomes.
        a=replay_meter(rr,model(kind='tou'),MeterController('THB',11))
        other=list(rr)
        for i in range(HISTORY+4,len(other)):other[i]=replace(other[i],load_kwh=9.,pv_kwh=8.)
        b=replay_meter(other,model(kind='tou'),MeterController('THB',11))
        self.assertEqual(a['intervals'][:4],b['intervals'][:4])
        self.assertEqual(a['decisions'][4]['request'],b['decisions'][4]['request'])

    def test_all_baselines_unshielded_shadow_independent(self):
        rr=readings();pfg=replay_meter(rr,model(),MeterController('PFG',11),initial_soc=3.)
        for name in ('THB','OMG','MPC'):
            r=replay_meter(rr,model(),MeterController(name,11),initial_soc=3.)
            self.assertIsNone(r['certificate'])
            self.assertAlmostEqual(r['shadow_cost'],pfg['cost'])
            self.assertAlmostEqual(r['shadow_final_soc'],pfg['final_soc'])
            self.assertLess(r['max_balance_residual'],1e-8)
        self.assertEqual(pfg['regret'],0.)

    def test_guarantees_signed_endpoints_and_intermediate_prefixes(self):
        rr=readings()
        for v in (0.,.076,.1425):
            for cert in ('global','copy','repair'):
                r=replay_meter(rr,model(v=v),MeterController('MPC',11,v),certificate=cert,budget=.2,initial_soc=7.)
                self.assertLessEqual(r['max_prefix_exposure'],.2+1e-8)
                self.assertLessEqual(r['terminal_adjusted_difference'],.2+1e-8)
                self.assertTrue(all(t['regret']+t['liability']<=.2+1e-8 for t in r['trace']))
                # Independent midpoint/end recomputation of each continuous segment.
                m=model(v=v)
                for row in r['trace']:
                    dt=row['end']-row['start'];r0=row['regret']-(row['cost']-row['shadow_cost'])
                    for f in (.25,.5,.75,1.):
                        s=row['soc_before']+f*(row['soc']-row['soc_before'])
                        z=row['shadow_soc_before']+f*(row['shadow_soc']-row['shadow_soc_before'])
                        exposure=r0+f*(row['cost']-row['shadow_cost'])+m.value(cert,row['start']+dt*f,s,z)
                        self.assertLessEqual(exposure,.2+1e-8)

    def test_midnight_keeps_state_and_account(self):
        r=replay_meter(readings(2),model(2),MeterController('MPC',11),certificate='repair',budget=.2,initial_soc=3.)
        self.assertEqual(len(r['intervals']),96)
        for i in (48,):
            self.assertAlmostEqual(r['intervals'][i-1]['end_soc'],r['decisions'][i]['soc'])
            self.assertAlmostEqual(r['intervals'][i-1]['end_shadow_soc'],r['decisions'][i]['shadow_soc'])
            self.assertAlmostEqual(r['intervals'][i-1]['regret_after'],r['decisions'][i]['regret_before'])
        self.assertEqual(r['decisions'][0]['planning_end'],86400)
        self.assertEqual(r['decisions'][1]['planning_end'],86400+STEP)
        self.assertEqual(r['decisions'][-1]['planning_end'],2*86400)
        self.assertTrue(all(d['true_episode_end']==2*86400 for d in r['decisions']))

    def test_small_positive_controller_domain_failure_retained(self):
        with self.assertRaisesRegex(ValueError,'frequent-action'):
            replay_meter(readings(),model(battery=BatterySpec(.1,5,5,.95,.95)),MeterController('OMG',11))

    def test_unknown_tariff_and_endpoint_mismatch_rejected(self):
        m=ExecutedLiability([PriceBand(0,86400,-.05,.15)],BatterySpec())
        with self.assertRaises(ValueError):replay_meter(readings(),m,MeterController('MPC',11))
        with self.assertRaises(ValueError):replay_meter(readings(),model(v=.076),MeterController('MPC',11))
        with self.assertRaises(ValueError):replay_meter(readings(),model(),MeterController('THB',11),certificate='repair')

    def test_energy_and_interval_peak_reconstruction(self):
        r=replay_meter(readings(),model(),MeterController('MPC',11))
        self.assertAlmostEqual(r['cost'],math.fsum(x['import_kwh']*x['price'] for x in r['intervals']))
        self.assertAlmostEqual(r['peak_half_hour_import_kw'],max(x['import_kwh']/.5 for x in r['intervals']))
        for x in r['intervals']:
            self.assertAlmostEqual(x['pv_kwh']+x['import_kwh']-x['load_kwh']-x['curtailment_kwh']-x['loss_kwh'],x['end_soc']-x['start_soc'])

    def test_endpoint_grid_251_501_and_zero_price(self):
        rr=readings();rr=rr[:HISTORY+2]
        for n in (251,501):
            m=ExecutedLiability([PriceBand(0,2*STEP,.08,.08)],BatterySpec(),.076)
            r=replay_meter(rr,m,MeterController('MPC',n,.076),certificate='repair')
            self.assertLessEqual(r['terminal_adjusted_difference'],1e-8)
        m=ExecutedLiability([PriceBand(0,2*STEP,0.,0.)],BatterySpec())
        r=replay_meter(rr,m,MeterController('OMG',11))
        self.assertEqual(r['cost'],0.)


if __name__=='__main__':unittest.main()
