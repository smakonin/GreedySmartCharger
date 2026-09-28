"""MGA: multistep guarded PV allocation. Empirical envelopes are not guarantees."""
from dataclasses import dataclass
import math
from physical_simulator import Action, self_consumption

HORIZON = 23  # future intervals; current plus future matches the 24-hour MPC span
FRACTIONS = tuple(i / 8 for i in range(1, 9))
DEFAULT_BUDGET = 1.0  # dollars per independently initialized 720-hour episode

@dataclass(frozen=True)
class Envelope:
    net_low: float
    net_high: float
    price_low: float
    dt: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.net_low, self.net_high, self.price_low, self.dt)):
            raise ValueError('Finite envelope required')
        if self.net_low > self.net_high or self.price_low < 0 or self.dt <= 0:
            raise ValueError('Invalid nonnegative-price envelope')


def soc_map(e, net, b, dt):
    if net < 0:
        return min(b.capacity, e + b.eta_charge * min(-net, b.charge_kw * dt))
    return max(0.0, e - min(net, b.discharge_kw * dt) / b.eta_discharge)


def packet_value_lower(base_soc, packet, envelopes, b):
    """Lower bound on extra PFG discharge revenue from an extra battery-side packet.

    Propagate an outer interval enclosure for baseline SOC e and excess SOC z.
    Future PV can erase z at capacity; early demand can consume it at low prices.
    Leftover z receives zero terminal credit. No probabilistic coverage is implied.
    """
    if base_soc < -1e-9 or packet < -1e-9 or base_soc + packet > b.capacity + 1e-9:
        raise ValueError('Infeasible packet')
    el = eu = max(0.0, base_soc)
    zl = zu = max(0.0, packet)
    value = 0.0
    for box in envelopes:
        lo, hi = box.net_low, box.net_high
        if lo >= 0:
            rmin = min(lo, b.discharge_kw * box.dt) / b.eta_discharge
            extra_lower = min(zl, max(0.0, rmin - eu))
            value += box.price_low * b.eta_discharge * extra_lower
        lower, upper = [], []
        if hi >= 0:
            rmin = min(max(0.0, lo), b.discharge_kw * box.dt) / b.eta_discharge
            rmax = min(hi, b.discharge_kw * box.dt) / b.eta_discharge
            lower.append(min(zl, max(0.0, el + zl - rmax)))
            upper.append(min(zu, max(0.0, eu + zu - rmin)))
        if lo <= 0:
            qmin = b.eta_charge * min(max(0.0, -hi), b.charge_kw * box.dt)
            qmax = b.eta_charge * min(-lo, b.charge_kw * box.dt)
            lower.append(min(zl, max(0.0, b.capacity - eu - qmax)))
            upper.append(min(zu, max(0.0, b.capacity - el - qmin)))
        el, eu = soc_map(el, hi, b, box.dt), soc_map(eu, lo, b, box.dt)
        zl, zu = min(lower), max(upper)
    return value


def empirical_envelopes(history, current_hour, tariff, tou, horizon=HORIZON):
    """Past seven same-hour observations only. No calibrated coverage claim."""
    boxes = []
    for step in range(1, horizon + 1):
        hour = (current_hour + step) % 24
        samples = [x for h, x in history[-168:] if h == hour]
        if not samples:
            raise ValueError('Missing causal history')
        lo = min(x.load for x in samples) - max(x.pv for x in samples)
        hi = max(x.load for x in samples) - min(x.pv for x in samples)
        price = tou[hour] if tariff == 'TimeOfDay' else min(x.price for x in samples)
        # Signed-price historical envelopes do not support this controller's certificate.
        if price < 0:
            return []
        boxes.append(Envelope(lo, hi, price))
    return boxes


def guarded_action(x, soc, b, envelopes, remaining_budget):
    if not math.isfinite(remaining_budget) or remaining_budget < 0:
        raise ValueError('Nonnegative finite budget required')
    base = self_consumption(x, soc, b)
    info = dict(delta=0.0, predicted_lower_gain=0.0, extra_spend=0.0)
    if x.pv < x.load or x.price < 0 or not envelopes:
        return base, remaining_budget, info
    e = soc + b.eta_charge * base.charge
    limit = max(0.0, min(base.direct, b.charge_kw*x.dt-base.charge,
                         (b.capacity-e)/b.eta_charge))
    if x.price > 0:
        limit = min(limit, remaining_budget/x.price)
    best_delta = best_gain = 0.0
    for fraction in FRACTIONS:
        delta = limit*fraction
        gain = packet_value_lower(e, b.eta_charge*delta, envelopes, b) - x.price*delta
        # Strictly positive gain; ties retain the smaller diversion (ascending grid).
        if gain > best_gain + 1e-10:
            best_delta, best_gain = delta, gain
    if best_delta == 0:
        return base, remaining_budget, info
    spend = x.price*best_delta
    return (Action(base.direct-best_delta, base.charge+best_delta, 0.0),
            max(0.0, remaining_budget-spend),
            dict(delta=best_delta, predicted_lower_gain=best_gain, extra_spend=spend))
