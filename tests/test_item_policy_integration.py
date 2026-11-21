from dataclasses import replace

import pytest

from replenishment import (
    BaseStockPolicy, ConstantOrderPolicy, CappedBaseStockPolicy, sSPolicy,
    ProjectedInventoryLevelPolicy, ProtectionPeriodQuantilePolicy,
    StandardSimulationRow, build_article_configs_from_standard_rows,
    build_replenishment_decisions_from_simulations, optimize_policy_candidates,
    simulate_replenishment, simulate_replenishment_for_articles,
    simulate_replenishment_with_aggregation,
)
from replenishment.aggregation import aggregate_policy
from replenishment.optimization import _attach_metadata


def search(demand, **kwargs):
    params = dict(candidates={'zero': ConstantOrderPolicy(0), 'three': ConstantOrderPolicy(3)},
                  periods=3, demand=demand, initial_on_hand=0, lead_time=1,
                  holding_cost_per_unit=1, stockout_cost_per_unit=5)
    params.update(kwargs)
    return optimize_policy_candidates(**params)


def test_candidate_search_replays_generator_and_finds_lowest_cost():
    result = search(iter([2, 2, 2]))
    assert result.label == 'three'
    assert result.policy == ConstantOrderPolicy(3)
    assert result.candidate_costs == {'zero': 30, 'three': 13}
    assert result.simulation.summary.total_demand == 6
    assert result.simulation.metadata.policy_name == 'ConstantOrderPolicy'
    assert dict(result.simulation.metadata.policy_parameters)['quantity'] == 3


def test_callable_demand_evaluated_once_per_period_for_all_candidates():
    calls = []
    def demand(t):
        calls.append(t)
        return len(calls)
    result = search(demand)
    assert calls == [0, 1, 2]
    assert result.simulation.summary.total_demand == 6


def test_candidate_ties_are_stable_and_empty_set_is_rejected():
    result = search([0, 0, 0], holding_cost_per_unit=0, stockout_cost_per_unit=0)
    assert result.label == 'zero'
    with pytest.raises(ValueError, match='candidates'):
        search([1, 1, 1], candidates={})


@pytest.mark.parametrize('cost', [float('nan'), float('inf'), -1])
def test_candidate_search_rejects_invalid_costs(cost):
    with pytest.raises(ValueError):
        search([1, 1, 1], holding_cost_per_unit=cost)


def rows(lead_time=0, forecast=False):
    return [StandardSimulationRow(
        unique_id='A', ds=f'2026-01-0{i+1}', demand=d, forecast=2, actuals=999,
        holding_cost_per_unit=.5, stockout_cost_per_unit=5,
        order_cost_per_order=2, lead_time=lead_time, initial_on_hand=1,
        current_stock=4, forecast_percentiles={}, is_forecast=forecast,
    ) for i, d in enumerate([2, 3, 2])]


def test_standard_row_builder_uses_demand_and_explicit_item_policy():
    configs = build_article_configs_from_standard_rows(rows(), policies={'A': BaseStockPolicy(5)})
    assert configs['A'].initial_on_hand == 1
    assert configs['A'].demand == [2, 3, 2]
    assert configs['A'].order_cost_per_order == 2
    future = build_article_configs_from_standard_rows(rows(forecast=True), policies={'A': BaseStockPolicy(5)})
    assert future['A'].initial_on_hand == 4
    explicit = build_article_configs_from_standard_rows(rows(), policies={'A': BaseStockPolicy(5)}, use_current_stock=True)
    assert explicit['A'].initial_on_hand == 4
    with pytest.raises(ValueError, match='policy'):
        build_article_configs_from_standard_rows(rows(), policies={})


@pytest.mark.parametrize('lead_time', [0, 1, 2])
def test_decision_rows_match_actual_simulation_stock_and_lost_sales(lead_time):
    data = rows(lead_time)
    configs = build_article_configs_from_standard_rows(data, policies={'A': BaseStockPolicy(5)})
    simulations = simulate_replenishment_for_articles(configs)
    decisions = build_replenishment_decisions_from_simulations(data, simulations)
    for row, snapshot in zip(decisions, simulations['A'].snapshots):
        assert row.starting_stock == snapshot.starting_on_hand
        assert row.ending_stock == snapshot.ending_on_hand
        assert row.missed_sales == snapshot.lost_sales
        assert row.reorder_point is None
        assert row.order_up_to == 5
    assert decisions[0].missed_sales == 1


@pytest.mark.parametrize('policy, reorder_point, target', [
    (sSPolicy(2, 5), 2, 5), (ConstantOrderPolicy(2), None, None),
    (ProjectedInventoryLevelPolicy(5, {2: 1}, 1), None, None),
    (ProtectionPeriodQuantilePolicy(.9, 1, aggregate_samples=[4]), None, None),
])
def test_decision_rows_do_not_invent_forecast_thresholds_for_new_policies(policy, reorder_point, target):
    data = rows(lead_time=1)
    configs = build_article_configs_from_standard_rows(data, policies={'A': policy})
    decisions = build_replenishment_decisions_from_simulations(data, simulate_replenishment_for_articles(configs))
    assert all(row.reorder_point == reorder_point for row in decisions)
    assert all(row.order_up_to == target for row in decisions)


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(5, 2), ConstantOrderPolicy(2, 2), CappedBaseStockPolicy(5, 2, 2),
    sSPolicy(2, 5, 2), ProjectedInventoryLevelPolicy(5, {2: 1}, 1, 2),
    ProtectionPeriodQuantilePolicy(.9, 1, 2, aggregate_samples=[4]),
])
def test_metadata_survives_attachment_and_identity_aggregation(policy):
    sim = simulate_replenishment(periods=3, demand=[2, 3, 2], initial_on_hand=0, lead_time=1, policy=policy)
    assert sim.metadata.policy_name == type(policy).__name__
    assert sim.metadata.review_period == 2
    attached = _attach_metadata(sim, percentile_target='test')
    assert attached.metadata.policy_parameters == sim.metadata.policy_parameters
    assert attached.metadata.policy_name == sim.metadata.policy_name
    wrapped = simulate_replenishment_with_aggregation(
        periods=3, demand=[2, 3, 2], initial_on_hand=0, lead_time=1, policy=policy,
        aggregation_window=1,
    )
    assert wrapped.metadata.policy_parameters == sim.metadata.policy_parameters
    assert wrapped.metadata.review_period == 2
    assert aggregate_policy(policy, periods=3, window=1, lead_time=1) is policy


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(5), ConstantOrderPolicy(2), CappedBaseStockPolicy(5, 2),
    sSPolicy(2, 5), ProjectedInventoryLevelPolicy(5, {2: 1}, 1),
    ProtectionPeriodQuantilePolicy(.9, 1, aggregate_samples=[4]),
])
def test_new_policies_reject_implicit_time_unit_transformations(policy):
    with pytest.raises(ValueError, match='review_period'):
        aggregate_policy(policy, periods=4, window=2, lead_time=1)
    with pytest.raises(ValueError, match='review_period'):
        simulate_replenishment_with_aggregation(
            periods=4, demand=[2] * 4, initial_on_hand=0, lead_time=1,
            policy=policy, aggregation_window=2,
        )
