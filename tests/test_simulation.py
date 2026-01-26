from replenishment import (
    ArticleSimulationConfig,
    ForecastCandidatesConfig,
    InventoryState,
    PercentileForecastOptimizationPolicy,
    PointForecastOptimizationPolicy,
    ReorderPointPolicy,
    optimize_aggregation_and_forecast_targets,
    optimize_aggregation_and_service_level_factors,
    simulate_replenishment_with_aggregation,
    optimize_aggregation_windows,
    optimize_forecast_targets,
    optimize_service_level_factors,
    percentile_forecast_optimisation,
    point_forecast_optimisation,
    simulate_replenishment,
)
from replenishment.aggregation import (
    aggregate_lead_time,
    aggregate_periods,
    aggregate_series,
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


def test_point_forecast_policy_orders_forecast_plus_safety_stock():
    policy = PointForecastOptimizationPolicy(
        forecast=[10, 12, 11],
        actuals=[9, 14, 10],
        lead_time=1,
        service_level_factor=1.0,
    )
    state_period0 = InventoryState(period=0, on_hand=0, on_order=0, backorders=0)
    state_period1 = InventoryState(period=1, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period0) == 12
    assert policy.order_quantity_for(state_period1) == 12


def test_point_forecast_policy_uses_last_forecast_value_for_horizon():
    policy = PointForecastOptimizationPolicy(
        forecast=[18, 22, 20, 19, 21, 23],
        actuals=[20, 21, 19, 18, 22, 24],
        lead_time=1,
        service_level_factor=0.0,
    )
    state_period5 = InventoryState(period=5, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period5) == 23


def test_optimize_point_forecast_selects_lowest_cost():
    forecast = [10, 10, 10, 10, 10]
    actuals = [8, 12, 9, 11, 10]
    candidates = [0.0, 1.5, 3.0]
    base_policy = PointForecastOptimizationPolicy(
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

    results = point_forecast_optimisation({"A": config}, candidates)
    result = results["A"]

    candidate_costs = []
    for factor in candidates:
        policy = PointForecastOptimizationPolicy(
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
    legacy_results = optimize_service_level_factors({"A": config}, candidates)
    assert legacy_results["A"].service_level_factor == expected_factor


def test_percentile_forecast_policy_orders_forecast():
    policy = PercentileForecastOptimizationPolicy(forecast=[10, 12], lead_time=1)
    state_period0 = InventoryState(period=0, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period0) == 12


def test_percentile_forecast_policy_uses_last_value_for_horizon():
    policy = PercentileForecastOptimizationPolicy(forecast=[10, 12], lead_time=1)
    state_period1 = InventoryState(period=1, on_hand=0, on_order=0, backorders=0)

    assert policy.order_quantity_for(state_period1) == 12


def test_optimize_percentile_forecast_selects_lowest_cost():
    config = ForecastCandidatesConfig(
        periods=2,
        demand=[10, 10],
        initial_on_hand=0,
        lead_time=0,
        forecast_candidates={
            "mean": [10, 10],
            "45-high": [20, 20],
        },
        holding_cost_per_unit=1.0,
        stockout_cost_per_unit=0.0,
    )

    results = percentile_forecast_optimisation({"A": config})

    assert results["A"].target == "mean"
    legacy_results = optimize_forecast_targets({"A": config})
    assert legacy_results["A"].target == "mean"


def test_simulation_with_aggregation_groups_demand():
    policy = ReorderPointPolicy(reorder_point=-1, order_quantity=0)
    result = simulate_replenishment_with_aggregation(
        periods=6,
        demand=[1, 2, 3, 4, 5, 6],
        initial_on_hand=100,
        lead_time=0,
        policy=policy,
        aggregation_window=3,
    )

    assert len(result.snapshots) == 2
    assert [snapshot.demand for snapshot in result.snapshots] == [6, 15]
    assert result.summary.total_demand == 21


def test_simulation_with_aggregation_truncates_demand_to_periods():
    policy = ReorderPointPolicy(reorder_point=-1, order_quantity=0)
    result = simulate_replenishment_with_aggregation(
        periods=5,
        demand=[1, 2, 3, 4, 5, 6],
        initial_on_hand=100,
        lead_time=0,
        policy=policy,
        aggregation_window=3,
    )

    assert [snapshot.demand for snapshot in result.snapshots] == [6, 9]
    assert result.summary.total_demand == 15


def test_optimize_aggregation_windows_picks_first_best():
    policy = ReorderPointPolicy(reorder_point=-1, order_quantity=0)
    config = ArticleSimulationConfig(
        periods=4,
        demand=[1, 1, 1, 1],
        initial_on_hand=0,
        lead_time=0,
        policy=policy,
        holding_cost_per_unit=0.0,
        stockout_cost_per_unit=0.0,
    )

    results = optimize_aggregation_windows({"A": config}, [1, 2])

    assert results["A"].window == 1
    assert len(results["A"].simulation.snapshots) == 4


def test_optimize_aggregation_and_service_level_factors_picks_lowest_cost():
    forecast = [10, 10, 10, 10]
    actuals = [8, 12, 9, 11]
    base_policy = PointForecastOptimizationPolicy(
        forecast=forecast,
        actuals=actuals,
        lead_time=1,
        service_level_factor=0.0,
    )
    config = ArticleSimulationConfig(
        periods=4,
        demand=actuals,
        initial_on_hand=0,
        lead_time=1,
        policy=base_policy,
        holding_cost_per_unit=1.0,
        stockout_cost_per_unit=2.0,
    )
    windows = [1, 2]
    factors = [0.0, 2.0]

    results = optimize_aggregation_and_service_level_factors(
        {"A": config},
        candidate_windows=windows,
        candidate_factors=factors,
    )
    result = results["A"]

    candidate_costs = {}
    for window in windows:
        aggregated_forecast = aggregate_series(
            forecast,
            periods=config.periods,
            window=window,
            extend_last=True,
        )
        aggregated_actuals = aggregate_series(
            actuals,
            periods=config.periods,
            window=window,
            extend_last=False,
        )
        aggregated_demand = aggregate_series(
            config.demand,
            periods=config.periods,
            window=window,
            extend_last=False,
        )
        aggregated_periods = aggregate_periods(config.periods, window)
        aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
        for factor in factors:
            policy = PointForecastOptimizationPolicy(
                forecast=aggregated_forecast,
                actuals=aggregated_actuals,
                lead_time=aggregated_lead_time,
                service_level_factor=factor,
            )
            simulation = simulate_replenishment(
                periods=aggregated_periods,
                demand=aggregated_demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=aggregated_lead_time,
                policy=policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
            )
            candidate_costs[(window, factor)] = simulation.summary.total_cost

    expected_window, expected_factor = min(
        candidate_costs, key=candidate_costs.get
    )
    assert result.window == expected_window
    assert result.service_level_factor == expected_factor


def test_optimize_aggregation_and_forecast_targets_picks_lowest_cost():
    config = ForecastCandidatesConfig(
        periods=4,
        demand=[5, 5, 5, 5],
        initial_on_hand=0,
        lead_time=0,
        forecast_candidates={
            "p50": [5, 5, 5, 5],
            "p90": [8, 8, 8, 8],
        },
        holding_cost_per_unit=1.0,
        stockout_cost_per_unit=0.5,
    )
    windows = [1, 2]

    results = optimize_aggregation_and_forecast_targets(
        {"A": config},
        candidate_windows=windows,
    )
    result = results["A"]

    candidate_costs = {}
    for window in windows:
        aggregated_demand = aggregate_series(
            config.demand,
            periods=config.periods,
            window=window,
            extend_last=False,
        )
        aggregated_periods = aggregate_periods(config.periods, window)
        aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
        for target, forecast in config.forecast_candidates.items():
            aggregated_forecast = aggregate_series(
                forecast,
                periods=config.periods,
                window=window,
                extend_last=True,
            )
            policy = PercentileForecastOptimizationPolicy(
                forecast=aggregated_forecast,
                lead_time=aggregated_lead_time,
            )
            simulation = simulate_replenishment(
                periods=aggregated_periods,
                demand=aggregated_demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=aggregated_lead_time,
                policy=policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
            )
            candidate_costs[(window, target)] = simulation.summary.total_cost

    expected_window, expected_target = min(
        candidate_costs, key=candidate_costs.get
    )
    assert result.window == expected_window
    assert result.target == expected_target


def test_optimize_aggregation_and_forecast_targets_accepts_generator_targets():
    candidate_configs = {
        "A": ForecastCandidatesConfig(
            periods=2,
            demand=[5, 5],
            initial_on_hand=0,
            lead_time=0,
            forecast_candidates={"p50": [5, 5], "p90": [7, 7]},
        ),
        "B": ForecastCandidatesConfig(
            periods=2,
            demand=[6, 6],
            initial_on_hand=0,
            lead_time=0,
            forecast_candidates={"p50": [6, 6], "p90": [8, 8]},
        ),
    }
    target_generator = (target for target in ["p50", "p90"])

    results = optimize_aggregation_and_forecast_targets(
        candidate_configs,
        candidate_windows=[1],
        candidate_targets=target_generator,
    )

    assert set(results.keys()) == {"A", "B"}


def test_optimize_aggregation_and_service_level_handles_short_actuals():
    policy = PointForecastOptimizationPolicy(
        forecast=[10, 11, 12, 13],
        actuals=[9, 12],
        lead_time=1,
        service_level_factor=0.0,
    )
    config = ArticleSimulationConfig(
        periods=4,
        demand=[10, 11, 12, 13],
        initial_on_hand=0,
        lead_time=1,
        policy=policy,
    )

    results = optimize_aggregation_and_service_level_factors(
        {"A": config},
        candidate_windows=[1, 2],
        candidate_factors=[0.0, 1.0],
    )

    assert results["A"].service_level_factor in {0.0, 1.0}
