import math

import pytest

from replenishment import (
    BaseStockPolicy, ConstantOrderPolicy, RiskObjective, empirical_cvar,
    optimize_risk_sensitive_policy_candidates,
)


@pytest.mark.parametrize('alpha, expected', [(0, 4.5), (.5, 8), (.625, 28/3), (.9, 12)])
def test_cvar_includes_fractional_tail(alpha, expected):
    # Sorted losses 12,4,2,0; alpha=.625 leaves 1.5 observations in the tail.
    assert empirical_cvar([0, 2, 4, 12], alpha) == pytest.approx(expected)


def test_risk_objectives_distinguish_average_tail_and_worst():
    losses = [0, 2, 4, 12]
    assert RiskObjective('mean').score(losses) == 4.5
    assert RiskObjective('cvar', alpha=.5).score(losses) == 8
    assert RiskObjective('mean_cvar', alpha=.5, risk_weight=.25).score(losses) == 5.375
    assert RiskObjective('worst_case').score(losses) == 12
    assert empirical_cvar([1e12]*10, .9) == 1e12


@pytest.mark.parametrize('losses', [[], [float('nan')], [float('inf')], [True]])
def test_invalid_scenario_losses_rejected(losses):
    with pytest.raises((ValueError, TypeError)):
        empirical_cvar(losses, .9)


@pytest.mark.parametrize('kwargs', [{'alpha': 1}, {'alpha': -1}, {'alpha': float('nan')},
    {'risk_weight': 2}, {'mode': 'unknown'}, {'alpha': True}])
def test_invalid_risk_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RiskObjective(**kwargs)


def test_risk_search_can_trade_mean_cost_for_tail_protection():
    # S=4 costs 8 or 11; S=0 costs 0 or 15. Mean favors S=0,
    # while CVaR/worst prefer S=4. Zero-lead holding includes immediate orders.
    candidates = {'low': BaseStockPolicy(0), 'protected': BaseStockPolicy(4)}
    args = dict(candidates=candidates, demand_scenarios=[[0, 0], [0, 5]],
                periods=2, initial_on_hand=0, lead_time=0,
                holding_cost_per_unit=1, stockout_cost_per_unit=3)
    mean = optimize_risk_sensitive_policy_candidates(**args, objective=RiskObjective('mean'))
    tail = optimize_risk_sensitive_policy_candidates(**args, objective=RiskObjective('cvar', alpha=.5))
    worst = optimize_risk_sensitive_policy_candidates(**args, objective=RiskObjective('worst_case'))
    assert mean.label == 'low'
    assert tail.label == worst.label == 'protected'
    assert mean.mean_cost == 7.5
    assert tail.scenario_costs == (8, 11)


def test_risk_search_replays_bounded_generators_and_stably_breaks_ties():
    demand = iter([1, 2, 99])
    result = optimize_risk_sensitive_policy_candidates(
        candidates={'first': BaseStockPolicy(3), 'same': BaseStockPolicy(3)},
        demand_scenarios=[demand], periods=2, initial_on_hand=0, lead_time=1,
        objective=RiskObjective('mean'),
    )
    assert result.label == 'first'
    assert next(demand) == 99
    assert result.candidate_scores['first'] == result.candidate_scores['same']


@pytest.mark.parametrize('scenarios', [[], [[]], [[1]], [[-1, 2]], [[1.5, 2]]])
def test_risk_search_validates_scenario_shapes_and_demand(scenarios):
    with pytest.raises((TypeError, ValueError)):
        optimize_risk_sensitive_policy_candidates(
            candidates={'base': BaseStockPolicy(3)}, demand_scenarios=scenarios,
            periods=2, initial_on_hand=0, lead_time=1,
        )
