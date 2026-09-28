"""Shadow-Certified Reserve (SCR), research prototype v1.

Prediction chooses a useful action; a separate shadow coupling certifies cost.
The uncertainty penalty and interpolated DP scores are heuristics, not bounds.
All physical actions use exact continuous SOC and allow grid charging.
"""
from dataclasses import dataclass
import math
import numpy as np
from grid_model import Action, transition
from grid_controllers import pfg
from grid_dp import action_for_transition


def liability(soc, shadow, b, price_low, price_high):
    """Maximum remaining coupling cost for prices in the declared interval."""
    return (b.eta_discharge * max(0., price_high) * max(0., shadow-soc)
            + max(0., -price_low) * max(0., soc-shadow) / b.eta_charge)


def couple(x, soc, shadow_action, b):
    """Copy shadow flows, clipping only unavailable discharge or full-battery charge.

    Retain PV charge first when clipping. Sign of SOC difference is preserved;
    its magnitude never increases. No claim that this equals PFG at actual SOC.
    """
    a = shadow_action
    discharge = min(a.discharge, max(0., soc)*b.eta_discharge)
    room = min(b.charge_kw*x.dt, max(0., b.capacity-soc)/b.eta_charge)
    pv_charge = min(a.charge, room)
    grid_charge = min(a.grid_charge, max(0., room-pv_charge))
    return Action(a.direct, pv_charge, discharge, grid_charge)


@dataclass
class Ledger:
    budget: float = 1.
    price_low: float = -.10
    price_high: float = .50
    regret: float = 0.
    breach_debit: float = 0.

    def __post_init__(self):
        if not (math.isfinite(self.budget) and self.budget >= 0
                and math.isfinite(self.price_low) and math.isfinite(self.price_high)
                and self.price_low <= self.price_high):
            raise ValueError('Invalid safety ledger')

    def choose(self, x, soc, shadow, shadow_action, candidates, scores, b):
        """Reject unsafe actions; force coupling when the current price breaks range.

        At a range breach the realized coupling's positive potential increment
        is explicitly added to the certificate, never hidden in the base budget.
        """
        sf = transition(x, shadow, shadow_action, b)
        old_phi = liability(soc, shadow, b, self.price_low, self.price_high)
        fallback = couple(x, soc, shadow_action, b)
        ff = transition(x, soc, fallback, b)
        fallback_phi = liability(ff.end_soc, sf.end_soc, b, self.price_low, self.price_high)
        increment = ff.cost-sf.cost+fallback_phi-old_phi
        breach = not self.price_low <= x.price <= self.price_high
        if not breach and increment > 1e-8:
            raise AssertionError('Coupling potential inequality failed')
        debit = max(0., increment) if breach else 0.
        self.breach_debit += debit
        admissible = []
        if not breach:
            for i, a in enumerate(candidates):
                f = transition(x, soc, a, b)
                phi = liability(f.end_soc, sf.end_soc, b, self.price_low, self.price_high)
                # Strict comparison: no accumulating per-step feasibility allowance.
                if self.regret+f.cost-sf.cost+phi <= self.budget+self.breach_debit:
                    admissible.append(i)
        if admissible:
            chosen = min(admissible, key=lambda i: (scores[i], -transition(x,soc,candidates[i],b).end_soc))
            action = candidates[chosen]
            reason = 'candidate'
        else:
            chosen = None; action = fallback
            reason = 'price_breach' if breach else 'fallback'
        f = transition(x, soc, action, b)
        self.regret += f.cost-sf.cost
        phi = liability(f.end_soc, sf.end_soc, b, self.price_low, self.price_high)
        slack = self.budget+self.breach_debit-self.regret-phi
        if slack < -1e-7:
            raise AssertionError(('Economic invariant failed', slack))
        return action, dict(reason=reason, chosen=chosen, admissible=len(admissible),
                            rejected=len(candidates)-len(admissible), regret=self.regret,
                            liability=phi, slack=slack, price_breach=breach,
                            debit=debit, cumulative_debit=self.breach_debit,
                            fallback_increment=increment, certificate=self.budget+self.breach_debit)


def value_table(predictions, b, states=101):
    """Finite-SOC nominal DP, zero terminal value. Returns future cost scores.

    Interpolation is used only to score continuous current actions. It is not
    a physically executed transition or a continuous-optimum certificate.
    """
    levels = np.linspace(0., b.capacity, states)
    delta = levels[None,:]-levels[:,None]
    charge = np.maximum(delta,0.)/b.eta_charge
    discharge = np.maximum(-delta,0.)*b.eta_discharge
    value = np.zeros(states)
    for x in reversed(predictions):
        valid = (charge <= b.charge_kw*x.dt+1e-10) & (discharge <= min(x.load,b.discharge_kw*x.dt)+1e-10)
        grid = np.maximum(0.,x.load-discharge+charge-x.pv) if x.price>=0 else x.load-discharge+charge
        cost = x.price*grid+value[None,:]
        value = np.min(np.where(valid,cost,np.inf),axis=1)
    return levels,value


def propose(x, soc, shadow_next, fallback, b, levels, values, predictions, radius=0., caution=.25):
    """Score charging/reserving/discharging actions, with fixed caution penalty.

    lambda = caution*q/(1+q). Cost units come from forecast storage-value width.
    This is a declared regularizer, not a robust objective or calibrated risk.
    """
    if not math.isfinite(radius) or radius < 0:
        return [fallback],[0.],dict(caution_weight=None,forecast_width=None,unusable_set=True)
    low = max(0.,soc-min(x.load,b.discharge_kw*x.dt)/b.eta_discharge)
    high = min(b.capacity,soc+b.charge_kw*x.dt*b.eta_charge)
    greedy = pfg(x,soc,b)
    targets = np.unique(np.r_[levels[(levels>=low)&(levels<=high)], low, high, soc,
        np.clip(shadow_next,low,high),transition(x,soc,greedy,b).end_soc,
        transition(x,soc,fallback,b).end_soc])
    actions=[action_for_transition(x,soc,float(e),b) for e in targets]
    # Include the exact coupling source allocation, even at negative prices.
    actions.append(fallback)
    pplus=max([0.]+[y.price for y in predictions]);pminus=max([0.]+[-y.price for y in predictions])
    width=b.eta_discharge*pplus+pminus/b.eta_charge
    weight=caution*radius/(1.+radius)
    scores=[]
    for a in actions:
        f=transition(x,soc,a,b)
        scores.append(f.cost+float(np.interp(f.end_soc,levels,values))
                      +weight*width*abs(f.end_soc-shadow_next))
    return actions,scores,dict(caution_weight=weight,forecast_width=width,unusable_set=False)
