from replenishment import (
    ArticleSimulationConfig,
    ForecastBasedPolicy,
    InventoryState,
    QuantileForecastPolicy,
    ReorderPointPolicy,
    optimize_quantile_levels,
    optimize_service_level_factors,
    simulate_replenishment,
)


def test_simulation_summary_consistency():
    policy = ReorderPointPolicy(reorder_point=10, order_quantity=20)
    result = simulate_replenishment(
        periods=4,
        demand=[5, 15, 8, 12],
        initial_on_hand=10,
        lead_time=1,
        policy=policy,
        holding_cost_per_unit=0.5,
        stockout_cost_per_unit=2.0,
    )

    assert result.summary.total_demand == 40
    assert result.summary.total_fulfilled <= result.summary.total_demand
    assert 0 <= result.summary.fill_rate <= 1
    assert result.summary.holding_cost >= 0
    assert result.summary.stockout_cost >= 0
    assert result.summary.total_cost == result.summary.holding_cost + result.summary.stockout_cost


def test_forecast_based_policy_orders_forecast_plus_safety_stock():
    policy = ForecastBasedPolicy(
        forecast=[10, 12, 11],
        actuals=[9, 14, 10],
        lead_time=1,
        service_level_factor=1.0,
    )
    state_period0 = InventoryState(period=0, on_hand=0, on_order=0, backorders=0)
    state_period1 = InventoryState(period=1, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period0) == 12
    assert policy.order_quantity_for(state_period1) == 12


def test_optimize_service_level_factors_selects_lowest_cost():
    forecast = [10, 10, 10, 10, 10]
    actuals = [8, 12, 9, 11, 10]
    candidates = [0.0, 1.5, 3.0]
    base_policy = ForecastBasedPolicy(
        forecast=forecast,
        actuals=actuals,
        lead_time=1,
        service_level_factor=0.0,
    )
    config = ArticleSimulationConfig(
        periods=4,
        demand=actuals[:4],
        initial_on_hand=5,
        lead_time=1,
        policy=base_policy,
        holding_cost_per_unit=2.0,
        stockout_cost_per_unit=2.0,
    )

    results = optimize_service_level_factors({"A": config}, candidates)
    result = results["A"]

    candidate_costs = []
    for factor in candidates:
        policy = ForecastBasedPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=1,
            service_level_factor=factor,
        )
        simulation = simulate_replenishment(
            periods=config.periods,
            demand=config.demand,
            initial_on_hand=config.initial_on_hand,
            lead_time=config.lead_time,
            policy=policy,
            holding_cost_per_unit=config.holding_cost_per_unit,
            stockout_cost_per_unit=config.stockout_cost_per_unit,
        )
        candidate_costs.append(simulation.summary.total_cost)

    expected_factor = candidates[candidate_costs.index(min(candidate_costs))]
    assert result.service_level_factor == expected_factor


def test_quantile_forecast_policy_orders_target_quantile():
    policy = QuantileForecastPolicy(
        mean_forecast=[10, 10, 10],
        quantile_forecasts={
            0.5: [9, 10, 11],
            0.9: [12, 13, 14],
        },
        lead_time=1,
        target_quantile=0.9,
    )
    state_period0 = InventoryState(period=0, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period0) == 13


def test_optimize_quantile_levels_selects_lowest_cost():
    mean_forecast = [10, 10, 10, 10, 10]
    quantiles = {
        0.5: [9, 9, 9, 9, 9],
        0.9: [12, 12, 12, 12, 12],
    }
    candidates = [0.5, 0.9]
    base_policy = QuantileForecastPolicy(
        mean_forecast=mean_forecast,
        quantile_forecasts=quantiles,
        lead_time=1,
        target_quantile=0.5,
    )
    config = ArticleSimulationConfig(
        periods=4,
        demand=[8, 11, 9, 10],
        initial_on_hand=5,
        lead_time=1,
        policy=base_policy,
        holding_cost_per_unit=2.0,
        stockout_cost_per_unit=2.0,
    )

    results = optimize_quantile_levels({"A": config}, candidates)
    result = results["A"]

    candidate_costs = []
    for quantile in candidates:
        policy = QuantileForecastPolicy(
            mean_forecast=mean_forecast,
            quantile_forecasts=quantiles,
            lead_time=1,
            target_quantile=quantile,
        )
        simulation = simulate_replenishment(
            periods=config.periods,
            demand=config.demand,
            initial_on_hand=config.initial_on_hand,
            lead_time=config.lead_time,
            policy=policy,
            holding_cost_per_unit=config.holding_cost_per_unit,
            stockout_cost_per_unit=config.stockout_cost_per_unit,
        )
        candidate_costs.append(simulation.summary.total_cost)

    expected_quantile = candidates[candidate_costs.index(min(candidate_costs))]
    assert result.quantile == expected_quantile
