from dataclasses import replace

import numpy as np
import pytest

from replenishment import (
    ArticleSimulationConfig, InventoryState, BaseStockPolicy, ConstantOrderPolicy,
    CappedBaseStockPolicy, sSPolicy, ProjectedInventoryLevelPolicy,
    ProtectionPeriodQuantilePolicy, StandardSimulationRow,
    PointForecastOptimizationPolicy,
    build_replenishment_decisions_from_simulations,
    build_replenishment_decisions_from_optimization_results,
    optimize_aggregation_windows, optimize_policy_candidates,
    simulate_replenishment, simulate_replenishment_with_aggregation,
)


@pytest.mark.parametrize('quantile, target', [(.14, 13), (np.nextafter(.14, 1), 14), (.28, 27), (.56, 55)])
def test_empirical_quantile_selects_exact_cdf_boundary(quantile, target):
    policy = ProtectionPeriodQuantilePolicy(quantile, 1, aggregate_samples=range(100))
    assert policy.target_for(InventoryState(0, 0, 0, 0)) == target


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(np.uint64(2)), CappedBaseStockPolicy(np.uint64(2), np.uint64(4)),
    sSPolicy(np.uint64(5), np.uint64(5)),
])
def test_unsigned_policy_and_state_integers_do_not_wrap_inventory_position(policy):
    assert policy.order_quantity_for(InventoryState(0, 3, 0, 0)) <= 2
    # A fixed-width integer addition would wrap this stock position to zero.
    assert policy.order_quantity_for(InventoryState(0, np.uint64(1), np.uint64(2**64-1), 0)) == 0


def test_quantile_path_totals_use_unbounded_native_integer_arithmetic():
    policy = ProtectionPeriodQuantilePolicy(1, 2, demand_paths=[[np.uint64(2**63), np.uint64(2**63)]])
    assert policy.target_for(InventoryState(0, 0, 0, 0)) == 2**64


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(np.int64(5), np.int64(3)),
    ConstantOrderPolicy(np.int64(2), np.int64(3)),
    CappedBaseStockPolicy(np.int64(5), np.int64(2), np.int64(3)),
    sSPolicy(np.int64(2), np.int64(5), np.int64(3)),
    ProjectedInventoryLevelPolicy(np.int64(5), {2: 1}, np.int64(1), np.int64(3)),
    ProtectionPeriodQuantilePolicy(.9, np.int64(1), np.int64(3), aggregate_samples=[4]),
])
def test_numpy_integral_policy_cadence_is_compatible_with_decision_output(policy):
    result = simulate_replenishment(periods=3, demand=[1, 2, 1], initial_on_hand=0, lead_time=1, policy=policy)
    assert type(result.metadata.review_period) is int
    decisions = build_replenishment_decisions_from_simulations(make_rows(), {'A': result})
    assert all(d.review_period == 3 for d in decisions)


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(5), ConstantOrderPolicy(2), CappedBaseStockPolicy(5, 2),
    sSPolicy(2, 5), ProjectedInventoryLevelPolicy(5, {2: 1}, 1),
    ProtectionPeriodQuantilePolicy(.9, 1, aggregate_samples=[4]),
])
def test_aggregation_optimizer_rejects_unsupported_new_policy_windows(policy):
    config = ArticleSimulationConfig(3, [1, 2, 1], 0, 1, policy)
    with pytest.raises(ValueError, match='review_period'):
        optimize_aggregation_windows({'A': config}, [2])


def test_identity_aggregation_optimizer_returns_usable_item_policy():
    policy = BaseStockPolicy(5, 3)
    config = ArticleSimulationConfig(3, [1, 2, 1], 0, 1, policy)
    result = optimize_aggregation_windows({'A': config}, [1])['A']
    assert result.policy is policy
    assert result.simulation.metadata.review_period == 3


def test_identity_wrapper_preserves_quantile_protection_horizon_metadata():
    policy = ProtectionPeriodQuantilePolicy(.9, 2, 3, aggregate_samples=[4])
    result = simulate_replenishment_with_aggregation(
        periods=3, demand=[1, 2, 1], initial_on_hand=0, lead_time=2,
        policy=policy, aggregation_window=1,
    )
    assert result.metadata.forecast_horizon == 4


def make_rows():
    return [StandardSimulationRow('A', f'2026-01-0{i+1}', d, 2, d, .5, 5, 2, 1, 0, 0, {})
            for i, d in enumerate([1, 2, 1])]


def test_review_cadence_stays_separate_from_aggregation_in_both_decision_converters():
    fit = optimize_policy_candidates(candidates={'base': BaseStockPolicy(5, 3)}, periods=3,
        demand=[1, 2, 1], initial_on_hand=0, lead_time=1)
    direct = build_replenishment_decisions_from_simulations(make_rows(), {'A': fit.simulation})
    optimized = build_replenishment_decisions_from_optimization_results(make_rows(), {'A': fit})
    assert [(d.aggregation_window, d.review_period) for d in direct] == [(1, 3)] * 3
    assert [(d.aggregation_window, d.review_period) for d in optimized] == [(1, 3)] * 3


def test_pil_avoids_large_inventory_cancellation_when_rounding_expected_gap():
    policy = ProjectedInventoryLevelPolicy(10_000, {0: .05, 1: .9, 2: .05}, 4)
    state = InventoryState(0, 10_000, 0, 0)
    # No stockout is possible: exactly 3 periods of mean-one demand intervene.
    assert policy.order_quantity_for(state) == 3


def test_candidate_search_consumes_only_its_requested_demand_horizon():
    # Raise if the tuner reads the fourth value, rather than actually hanging
    # the test on an infinite stream. Also verify the tail remains available.
    def trace():
        yield from [1, 2, 3]
        raise AssertionError('Demand beyond requested horizon was consumed')
    result = optimize_policy_candidates(candidates={'base': BaseStockPolicy(5)},
        periods=3, demand=trace(), initial_on_hand=0, lead_time=1)
    assert result.simulation.summary.total_demand == 6
    demand = iter([1, 2, 3, 99])
    optimize_policy_candidates(candidates={'base': BaseStockPolicy(5)},
        periods=3, demand=demand, initial_on_hand=0, lead_time=1)
    assert next(demand) == 99


def test_generic_tuning_does_not_reinterpret_legacy_forecast_decision_rows():
    policy = PointForecastOptimizationPolicy(forecast=[2]*3, actuals=[], fixed_rmse=0, lead_time=1)
    direct = simulate_replenishment(periods=3, demand=[1, 2, 1], initial_on_hand=0, lead_time=1, policy=policy)
    fitted = optimize_policy_candidates(candidates={'legacy': policy}, periods=3,
        demand=[1, 2, 1], initial_on_hand=0, lead_time=1)
    data = [replace(row, current_stock=40) for row in make_rows()]
    plain_rows = build_replenishment_decisions_from_simulations(data, {'A': direct})
    fitted_rows = build_replenishment_decisions_from_simulations(data, {'A': fitted.simulation})
    assert plain_rows == fitted_rows
    assert fitted_rows[0].starting_stock == 40
    assert fitted_rows[0].reorder_point is not None


@pytest.mark.parametrize('bad', [-1, 1.5, float('nan'), float('inf'), True])
@pytest.mark.parametrize('callable_demand', [False, True])
def test_new_policies_reject_invalid_demand_at_simulator_entry_points(bad, callable_demand):
    demand = (lambda t: bad) if callable_demand else [bad]
    with pytest.raises((TypeError, ValueError)):
        simulate_replenishment(periods=1, demand=demand, initial_on_hand=0,
            lead_time=1, policy=BaseStockPolicy(5))


@pytest.mark.parametrize('overrides', [
    {'initial_on_hand': -1}, {'initial_on_hand': 1.5}, {'initial_on_hand': True},
    {'lead_time': True}, {'lead_time': 1.5}, {'periods': True},
    {'holding_cost_per_unit': float('nan')}, {'stockout_cost_per_unit': float('inf')},
])
def test_new_policy_simulations_validate_initial_state_and_costs(overrides):
    parameters = dict(periods=1, demand=[1], initial_on_hand=0, lead_time=1, policy=BaseStockPolicy(5))
    parameters.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        simulate_replenishment(**parameters)


def test_external_arrival_schedule_cannot_mutate_frozen_observation():
    schedule = [[1, 4]]
    state = InventoryState(0, 0, 4, 0, schedule)
    schedule[0][1] = 9
    assert state.arrival_schedule == ((1, 4),)
