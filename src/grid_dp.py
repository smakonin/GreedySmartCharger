"""Matched, hindsight SOC-trellis DP with grid charging and signed prices.

Independent transition cost construction. No interpolation, energy rounding,
or export. A finite SOC grid restricts feasible trajectories, so its bill is an
upper bound on the continuous optimum (apart from declared numeric tolerance).
Credit: the SOC-trellis planning idea is present in Bhotto's supplied precursor.
This is a new implementation using the current study's physical conventions.
"""
import time
import numpy as np
from grid_model import Action, transition


def action_for_transition(x, before, after, b, pv_first=False):
    """Cheapest source allocation for a fixed, feasible SOC transition."""
    change = after - before
    if change >= 0:
        charge = change / b.eta_charge
        direct = min(x.load, x.pv) if x.price >= 0 or pv_first else 0.
        pv_charge = min(charge, x.pv - direct) if x.price >= 0 else 0.
        action = Action(direct, pv_charge, 0., max(0., charge - pv_charge))
    else:
        discharge = -change * b.eta_discharge
        direct = min(x.pv, max(0., x.load - discharge)) if x.price >= 0 else 0.
        if pv_first:
            direct = min(x.pv, x.load)
        action = Action(direct, 0., discharge, 0.)
    transition(x, before, action, b)
    return action


def optimize_dp(xs, b, states=101, initial_soc=0., terminal_value=0., pv_first=False):
    if not xs or states < 2 or not 0 <= initial_soc <= b.capacity or terminal_value < 0:
        raise ValueError('Invalid DP horizon, grid, initial SOC or terminal value')
    start = time.perf_counter()
    # Including the exact initial SOC avoids assigning free rounding energy.
    levels = np.unique(np.r_[np.linspace(0., b.capacity, states), initial_soc])
    m = len(levels)
    change = levels[None, :] - levels[:, None]
    charge = np.maximum(change, 0.) / b.eta_charge
    discharge = np.maximum(-change, 0.) * b.eta_discharge
    values = np.empty((len(xs) + 1, m))
    values[-1] = -terminal_value * levels
    choices = np.empty((len(xs), m), dtype=np.int32)
    for t in range(len(xs) - 1, -1, -1):
        x = xs[t]
        max_discharge = min(b.discharge_kw * x.dt, max(0., x.load-x.pv) if pv_first else x.load)
        feasible = (charge <= b.charge_kw*x.dt + 1e-10) & (discharge <= max_discharge + 1e-10)
        if x.price >= 0:
            # On the AC bus, total PV use can serve load and/or the fixed charge.
            grid = np.maximum(0., x.load-discharge+charge-x.pv)
        else:
            direct = min(x.load, x.pv) if pv_first else 0.
            grid = x.load-direct-discharge+charge
        costs = x.price*grid + values[t+1][None, :]
        costs[~feasible] = np.inf
        # Exact numerical ties select larger next SOC, then the source rule above
        # maximizes current PV use and direct PV without changing transition cost.
        choices[t] = m-1-np.argmin(costs[:, ::-1], axis=1)
        values[t] = costs[np.arange(m), choices[t]]
    idx = int(np.flatnonzero(levels == initial_soc)[0])
    initial_value = float(values[0, idx])
    soc = initial_soc
    actions = []
    cost = max_error = 0.
    for t, x in enumerate(xs):
        j = int(choices[t, idx])
        a = action_for_transition(x, soc, float(levels[j]), b, pv_first)
        f = transition(x, soc, a, b)
        max_error = max(max_error, abs(f.end_soc-levels[j]))
        soc = f.end_soc
        cost += f.cost
        actions.append(a)
        idx = j
    residual = abs(cost-terminal_value*soc-initial_value)
    if residual > 1e-7 or max_error > 1e-8:
        raise RuntimeError('DP path/value reconstruction failed')
    return actions, dict(cost=cost, objective_cost=cost-terminal_value*soc,
                        states=m, nominal_states=states, grid_spacing=b.capacity/(states-1),
                        initial_soc=initial_soc, final_soc=soc, terminal_value=terminal_value,
                        value_residual=residual, max_state_error=max_error,
                        compute_seconds=time.perf_counter()-start, pv_first=pv_first,
                        information='full-window hindsight', grid_charging_allowed=True)
