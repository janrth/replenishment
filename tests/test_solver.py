from itertools import product
import math

import pytest

pytest.importorskip('scipy')

from replenishment.simulation import InventoryState
from replenishment.risk import RiskObjective
from replenishment.solver import (
    SolverItemConfig, SharedResourceLimits, SolverOptions, PlanningError,
    plan_replenishment,
)


def state(stock=0, period=0, arrivals=()):
    return InventoryState(period, stock, sum(q for _, q in arrivals), 0, arrivals)


def spec(lead=1, cap=4, **kw):
    return SolverItemConfig(lead_time=lead, max_order_quantity=cap, **kw)


def plan(paths, config=None, observation=None, **kw):
    return plan_replenishment(
        states={'A': observation or state()}, items={'A': config or spec()},
        demand_scenarios={'A': paths}, **kw,
    )


def replay(config, observation, demands, orders):
    stock = observation.on_hand
    receipts = dict(observation.arrival_schedule)
    cost = 0
    inventories, losses = [], []
    for k, q in enumerate((*orders, 0)):
        absolute = observation.period + k
        if k:
            stock += receipts.get(absolute, 0)
            lost = max(0, demands[k-1] - stock)
            stock = max(0, stock - demands[k-1])
            cost += lost * config.stockout_cost_per_unit
            losses.append(lost)
        if q:
            cost += config.order_cost_per_order + q * config.order_cost_per_unit
            if config.lead_time:
                due = absolute + config.lead_time
                receipts[due] = receipts.get(due, 0) + q
            else:
                stock += q
        if k:
            inventories.append(stock)
            cost += stock * config.holding_cost_per_unit
        elif config.lead_time == 0:
            cost += q * config.holding_cost_per_unit
    return cost, tuple(inventories), tuple(losses)


@pytest.mark.parametrize('lead', [0, 1, 2])
@pytest.mark.parametrize('fixed,pack,minimum', [(0, 1, 0), (3, 2, 2)])
def test_deterministic_matches_exhaustive_integer_enumeration(lead, fixed, pack, minimum):
    config = spec(lead, holding_cost_per_unit=.7, stockout_cost_per_unit=6,
                  order_cost_per_unit=.2, order_cost_per_order=fixed,
                  pack_size=pack, min_order_quantity=minimum)
    observation = state(1)
    demands = (2, 1, 3)
    choices = [q for q in range(5) if q == 0 or q >= minimum and q % pack == 0]
    candidates = []
    for orders in product(choices, repeat=3):
        if any(q and k + max(1, lead) > 3 for k, q in enumerate(orders)):
            continue
        candidates.append(replay(config, observation, demands, orders)[0])
    result = plan([demands], config, observation)
    assert result.objective_value == pytest.approx(min(candidates))
    orders = result.orders['A'][0]
    cost, inventories, losses = replay(config, observation, demands, orders)
    assert result.scenario_costs == pytest.approx((cost,))
    assert result.inventories['A'][0] == pytest.approx(inventories)
    assert result.lost_sales['A'][0] == pytest.approx(losses)
    assert result.first_orders['A'] == orders[0]
    assert result.status == 'optimal'
    assert result.mip_gap <= 1e-7


def test_receipt_timing_changes_decision_and_review_uses_absolute_time():
    config = spec(2, review_period=2, holding_cost_per_unit=1, stockout_cost_per_unit=10)
    early = plan([(2, 2, 2)], config, state(arrivals=((1, 2),)))
    late = plan([(2, 2, 2)], config, state(arrivals=((2, 2),)))
    assert early.first_orders['A'] == 4
    assert late.first_orders['A'] == 2
    off = plan([(2, 2, 2)], config, state(period=1))
    assert off.first_orders['A'] == 0


@pytest.mark.parametrize('objective', [RiskObjective(), RiskObjective('worst_case'),
    RiskObjective('cvar', alpha=.25), RiskObjective('mean_cvar', alpha=.25, risk_weight=.4)])
def test_scenario_tree_matches_exhaustive_nonanticipative_search(objective):
    # Demand is identical through k=1, so the order at k=1 must still be shared.
    paths = ((1, 0, 2), (1, 2, 0))
    config = spec(cap=2, holding_cost_per_unit=1, stockout_cost_per_unit=7,
                  order_cost_per_order=1)
    best = math.inf
    for q0, q1, q2a, q2b in product(range(3), repeat=4):
        costs = [replay(config, state(), p, (q0, q1, q2))[0]
                 for p, q2 in zip(paths, (q2a, q2b))]
        best = min(best, objective.score(costs))
    result = plan(paths, config, objective=objective)
    assert result.orders['A'][0][:2] == result.orders['A'][1][:2]
    assert result.objective_value == pytest.approx(best)
    assert objective.score(result.scenario_costs) == pytest.approx(best)


@pytest.mark.parametrize('limits', [SharedResourceLimits(max_order_units=2),
    SharedResourceLimits(procurement_budget=2), SharedResourceLimits(max_receiving_units=2)])
def test_shared_constraint_allocates_to_item_with_larger_shortage_penalty(limits):
    result = plan_replenishment(
        states={'A': state(), 'B': state()},
        items={'A': spec(cap=2, stockout_cost_per_unit=20, order_cost_per_unit=1),
               'B': spec(cap=2, stockout_cost_per_unit=5, order_cost_per_unit=1)},
        demand_scenarios={'A': [(2,)], 'B': [(2,)]}, limits=limits,
    )
    assert dict(result.first_orders) == {'A': 2, 'B': 0}
    assert result.scenario_costs == pytest.approx((12,))


def test_receiving_limit_counts_committed_pipeline_and_zero_lead_orders():
    result = plan_replenishment(
        states={'A': state(arrivals=((1, 2),)), 'B': state()},
        items={'A': spec(stockout_cost_per_unit=9), 'B': spec(stockout_cost_per_unit=8)},
        demand_scenarios={'A': [(2,)], 'B': [(2,)]},
        limits=SharedResourceLimits(max_receiving_units=3),
    )
    assert result.first_orders['B'] == 1
    immediate = plan([(5,)], spec(0, stockout_cost_per_unit=10),
                     limits=SharedResourceLimits(max_receiving_units=2))
    assert immediate.first_orders['A'] == 2


def test_infeasible_committed_receipts_raise_instead_of_returning_zero():
    with pytest.raises(PlanningError, match='infeasible'):
        plan([(1,)], observation=state(arrivals=((1, 3),)),
             limits=SharedResourceLimits(max_receiving_units=2))


def test_zero_costs_still_force_exact_fulfillment():
    result = plan([(3, 0)], spec(cap=0), state(2))
    assert result.inventories['A'][0] == (0, 0)
    assert result.lost_sales['A'][0] == (1, 0)


@pytest.mark.parametrize('kwargs', [dict(max_order_quantity=-1), dict(lead_time=True),
    dict(pack_size=0), dict(min_order_quantity=5), dict(holding_cost_per_unit=float('nan')),
    dict(stockout_cost_per_unit=-1), dict(review_period=0)])
def test_invalid_item_parameters(kwargs):
    values = dict(lead_time=1, max_order_quantity=4)
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        SolverItemConfig(**values)


@pytest.mark.parametrize('paths', [[], [()], [(1,), (1, 2)], [(1.5,)], [(-1,)]])
def test_invalid_scenarios(paths):
    with pytest.raises((ValueError, TypeError)):
        plan(paths)


def test_observation_and_horizon_validation():
    with pytest.raises(ValueError, match='arrival_schedule'):
        plan([(1,)], observation=InventoryState(0, 0, 2, 0))
    with pytest.raises(ValueError, match='horizon'):
        plan([(1,)], spec(2))
    with pytest.raises(ValueError, match='backlog'):
        plan([(1,)], observation=InventoryState(0, 0, 0, 1))


def test_solver_failure_never_executes_incumbent(monkeypatch):
    scipy = pytest.importorskip('scipy.optimize')
    from types import SimpleNamespace
    monkeypatch.setattr(scipy, 'milp', lambda **kw: SimpleNamespace(
        status=1, success=False, message='Time limit reached', x=[2], fun=0))
    with pytest.raises(PlanningError, match='limit'):
        plan([(1,)])


def test_inputs_and_results_are_immutable():
    paths = [[2, 3]]
    result = plan(paths, spec(stockout_cost_per_unit=10))
    with pytest.raises(TypeError):
        result.first_orders['A'] = 5
    assert paths == [[2, 3]]


def test_shared_packs_fixed_cost_budget_matches_exhaustive_first_orders():
    items = {'A': spec(cap=4, pack_size=2, min_order_quantity=2,
                      order_cost_per_order=2, order_cost_per_unit=1, stockout_cost_per_unit=9),
             'B': spec(cap=3, order_cost_per_order=1, order_cost_per_unit=2, stockout_cost_per_unit=5)}
    paths = {'A': [(3,), (1,)], 'B': [(1,), (3,)]}
    objective = RiskObjective('cvar', alpha=.25)
    limits = SharedResourceLimits(max_order_units=4, procurement_budget=6, max_receiving_units=4)
    best = math.inf
    for qa,qb in product([0,2,4], range(4)):
        expense = (2+qa if qa else 0)+(1+2*qb if qb else 0)
        if qa+qb > 4 or expense > 6:
            continue
        costs = [sum(replay(items[i],state(), paths[i][s], (q,))[0]
                     for i,q in [('A',qa),('B',qb)]) for s in range(2)]
        best = min(best,objective.score(costs))
    result = plan_replenishment(states={i:state() for i in items}, items=items,
        demand_scenarios=paths, limits=limits, objective=objective)
    assert result.objective_value == pytest.approx(best)


def test_joint_history_reveals_only_observed_cross_item_information():
    result = plan_replenishment(states={'A':state(), 'B':state()},
        items={'A':spec(cap=3, stockout_cost_per_unit=10, holding_cost_per_unit=1),
               'B':spec(cap=0)},
        demand_scenarios={'A': [(0,0,3),(0,0,0)], 'B':[(0,1,0),(0,0,0)]})
    assert result.orders['A'][0][:2] == result.orders['A'][1][:2]
    # The signal in B's second demand is observed before A's third-demand order.
    assert result.orders['A'][0][2] == 3
    assert result.orders['A'][1][2] == 0


def test_already_received_units_consume_current_capacity_for_immediate_orders():
    result = plan([(4,)], spec(0, stockout_cost_per_unit=10),
        limits=SharedResourceLimits(max_receiving_units=3), received_this_period=2)
    assert result.first_orders['A'] == 1
    assert result.received_this_period == 2


@pytest.mark.parametrize('status', [1,2,3,4])
def test_all_unsuccessful_solver_statuses_raise(monkeypatch, status):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'milp', lambda **kw: SimpleNamespace(
        status=status, success=False, x=None, message='test failure'))
    with pytest.raises(PlanningError):
        plan([(2,)])


@pytest.mark.parametrize('mutation', ['fractional', 'nan', 'infeasible'])
def test_invalid_solver_vectors_are_not_accepted(monkeypatch, mutation):
    import scipy.optimize
    real = scipy.optimize.milp
    def corrupt(**kwargs):
        result = real(**kwargs)
        if mutation == 'fractional':
            result.x[0] = .5
        elif mutation == 'nan':
            result.x[0] = float('nan')
        else:
            result.x[0] = 400
        return result
    monkeypatch.setattr(scipy.optimize,'milp',corrupt)
    with pytest.raises(PlanningError):
        plan([(2,)], spec(stockout_cost_per_unit=10))


@pytest.mark.parametrize('objective', [RiskObjective('mean',alpha=.25),
    RiskObjective('cvar',alpha=.25), RiskObjective('mean_cvar',alpha=.25,risk_weight=.4),
    RiskObjective('worst_case',alpha=.25)])
def test_plan_reports_objective_and_alpha_aware_risk_diagnostics(objective):
    from replenishment.risk import empirical_cvar
    result = plan([(0,), (2,), (3,)], spec(cap=2, stockout_cost_per_unit=8,
        holding_cost_per_unit=1), objective=objective)
    assert result.objective == objective
    assert result.mean_cost == pytest.approx(sum(result.scenario_costs)/3)
    assert result.cvar_cost == pytest.approx(empirical_cvar(result.scenario_costs,objective.alpha))
    assert result.worst_cost == max(result.scenario_costs)


@pytest.mark.parametrize('limits', [SharedResourceLimits(procurement_budget=0),
    SharedResourceLimits(max_order_units=.9999999),
    SharedResourceLimits(max_receiving_units=.9999999)])
def test_hard_resource_limits_do_not_allow_absolute_tolerance_violations(limits):
    result = plan([(1,)], spec(cap=1,stockout_cost_per_unit=100,order_cost_per_unit=1e-8),
                  limits=limits)
    assert result.first_orders['A'] == 0


def test_small_currency_cvar_is_not_treated_as_zero_cost():
    result = plan([(2,)], spec(cap=2,stockout_cost_per_unit=1e-7,order_cost_per_unit=2e-8),
                  objective=RiskObjective('cvar'))
    assert result.first_orders['A'] == 2
    assert result.objective_value == pytest.approx(4e-8,abs=1e-15)
    assert result.dual_bound == pytest.approx(result.objective_value,abs=1e-15)


def test_actual_reported_gap_respects_requested_zero_gap_at_small_currency_scale():
    items = {'A':spec(cap=4,holding_cost_per_unit=1e-5,stockout_cost_per_unit=1e-5,
                     order_cost_per_order=2e-5,order_cost_per_unit=2e-5),
             'B':spec(cap=4,holding_cost_per_unit=3e-5,stockout_cost_per_unit=4e-5,
                     order_cost_per_order=1e-5,order_cost_per_unit=2e-5)}
    result = plan_replenishment(states={i:state() for i in items},items=items,
        demand_scenarios={'A':[(0,1,1,0),(4,0,4,1),(3,3,0,1)],
                          'B':[(3,0,2,2),(3,1,3,4),(2,3,2,1)]},
        limits=SharedResourceLimits(max_order_units=4))
    assert result.mip_gap <= 1e-7
    assert result.dual_bound == pytest.approx(result.objective_value,abs=1e-12)


@pytest.mark.parametrize('field,value', [('mip_gap',.5),('mip_gap',float('nan')),
    ('mip_dual_bound',float('nan')),('mip_dual_bound',-100)])
def test_invalid_solver_gap_or_bound_is_not_accepted(monkeypatch,field,value):
    import scipy.optimize
    real = scipy.optimize.milp
    def corrupt(**kwargs):
        result = real(**kwargs)
        result[field] = value
        return result
    monkeypatch.setattr(scipy.optimize,'milp',corrupt)
    with pytest.raises(PlanningError):
        plan([(2,)],spec(stockout_cost_per_unit=10,order_cost_per_unit=1))
