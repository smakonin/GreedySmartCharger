#
# libModels.py -- Copyright (C) 2018 Stephen Makonin
#

from libUtil import *


class Solar:
    def __init__(self, capacity=4, units='kWh'):
        self.capacity = capacity
        self.wear_and_tear_cost = 0
        self.units = units
        self.state = 0

    def reset(self):
        self.excess = 0


class Battery:
    def __init__(self, capacity=15, charge_rate=5, discharge_rate=5, units='kWh'):
        self.capacity = capacity
        self.state = self.inital_state = capacity / 2
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.units = units
        self.wear_and_tear_cost = 0
        self.reset()

    def reset(self):
        self.activity = 0
        self.charging = False
        self.discharging = False

    def calc_costs(self):
        pass

    def drained(self):
        amount = self.capacity - self.state
        if amount < 0: errout('ERROR: battery drained() cannot return less than 0')
        return amount

    def charge(self, amount):
        if amount < 0: errout('ERROR: battery charge amount cannot be less than 0')
        if amount == 0: return 0
        if self.discharging: return amount
        if self.drained() == 0: return amount

        self.charging = True
        self.activity = limiter(self.drained(), amount, self.charge_rate)
        self.state += self.activity
        return lbound(amount - self.activity, 0)

    def discharge(self, amount, update=True):
        if amount < 0: errout('ERROR: battery discharge amount cannot be less than 0')
        if amount == 0: return 0
        if self.charging: return 0
        if self.state == 0: return 0

        activity = limiter(amount, self.state, self.discharge_rate)
        if update:
            self.discharging = True
            self.activity = -1 * activity
            self.state += self.activity
        return activity


def model_grid_only(consumption, grid_price, solar=None, battery=None):
    the_cost = consumption * grid_price
    return the_cost

def model_solar_only(consumption, grid_price, solar, battery=None):
    solar.reset()

    amount_solar_left = solar.state - consumption
    solar_cost = solar.state * solar.wear_and_tear_cost

    if amount_solar_left > 0:
        solar.excess = amount_solar_left
        amount_solar_left = 0

    grid_amount = abs(amount_solar_left)
    grid_cost = grid_amount * grid_price
    the_cost = solar_cost + grid_cost
    return the_cost

def model_grid_tied(consumption, grid_price, solar, battery=None):
    grid_cost = model_solar_only(consumption, grid_price, solar)
    gird_sold = solar.excess * grid_price
    the_cost = grid_cost - gird_sold
    return the_cost

def model_batt_last(consumption, grid_price, solar, battery):
    solar.reset()
    battery.reset()

    amount_solar_left = solar.state - consumption
    solar_cost = solar.state * solar.wear_and_tear_cost

    if amount_solar_left > 0:
        solar.excess = battery.charge(amount_solar_left)
        amount_solar_left -= solar.excess

    consumption_left = abs(amount_solar_left)
    battery_amount = battery.discharge(consumption_left)
    battery_cost = battery_amount * battery.wear_and_tear_cost
    grid_amount = consumption_left - battery_amount
    grid_cost = grid_amount * grid_price
    the_cost = solar_cost + battery_cost + grid_cost
    return the_cost


SOURCE_SOLAR   = 0
SOURCE_BATTERY = 1
SOURCE_GRID    = 2

priority = [[0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]]

def model_1a_greedy(consumption, grid_price, solar, battery):
    solar.reset()
    battery.reset()

    amounts = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]

    amount_solar_left = battery.charge(solar.state)

    for j in range(len(priority)):
        consumption_left = consumption
        for l in range(3):
            k = priority[j][l]

            if k == SOURCE_SOLAR:
                amounts[j][k] = limiter(consumption_left, amount_solar_left, amount_solar_left)
            elif k == SOURCE_BATTERY:
                amounts[j][k] = battery.discharge(consumption_left, update=False)
            elif k == SOURCE_GRID:
                amounts[j][k] = consumption_left

            consumption_left = consumption_left - amounts[j][k]

    amounts = rmdup(amounts)
    power_from_grid = 0
    solutions = {'optimal': -1, 'amounts': [], 'costs': []}
    costs = []

    for amount in amounts:
        cost = [0, 0, 0]
        for k in range(len(cost)):
            if k == SOURCE_SOLAR:
                cost[k] = amount[k] * solar.wear_and_tear_cost
            elif k == SOURCE_BATTERY:
                cost[k] = amount[k] * battery.wear_and_tear_cost
            elif k == SOURCE_GRID:
                cost[k] = amount[k] * grid_price
        costs.append(cost)

    solutions['amounts'] = amounts
    solutions['costs'] = costs
    tl_costs = [sum(c) for c in solutions['costs']]
    m = argmin(tl_costs)
    solutions['optimal'] = m
    battery.discharge(solutions['amounts'][m][SOURCE_BATTERY])
    the_cost = tl_costs[m]
    return the_cost


model_tests = [
    {
        'name': 'Grid Only',
        'tname': 'GridOnly', # max 8 characters
        'base-case': True,
        'func': model_grid_only
    },
    {
        'name': 'Solar Only',
        'tname': 'PVOnly', # max 8 characters
        'base-case': False,
        'func': model_solar_only
    },
    {
        'name': 'Grid-Tied Solar',
        'tname': 'GridTied', # max 8 characters
        'base-case': False,
        'func': model_grid_tied
    },
    {
        'name': 'Battery Last',
        'tname': 'BattLast', # max 8 characters
        'base-case': False,
        'func': model_batt_last
    },
    {
        'name': 'Model 1a Greedy',
        'tname': 'Greedy1a', # max 8 characters
        'base-case': False,
        'func': model_1a_greedy
    }
]
