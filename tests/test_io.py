import csv
import math

import pytest

from replenishment import (
    PointForecastRow,
    ReplenishmentDecisionRow,
    StandardSimulationRow,
    build_percentile_forecast_candidates,
    build_percentile_forecast_candidates_from_standard_rows,
    build_point_forecast_article_configs,
    build_point_forecast_article_configs_from_standard_rows,
    build_replenishment_decisions_from_optimization_results,
    build_replenishment_decisions_from_simulations,
    generate_standard_simulation_rows,
    iter_percentile_forecast_rows_from_csv,
    iter_point_forecast_rows_from_csv,
    iter_standard_simulation_rows_from_csv,
    optimize_point_forecast_policy_and_simulate_actuals,
    optimize_aggregation_windows,
    split_standard_simulation_rows,
    standard_simulation_rows_from_dataframe,
    simulate_replenishment_for_articles,
    simulate_replenishment_with_aggregation,
)
from replenishment.aggregation import aggregate_series
from replenishment.io import ReplenishmentDecisionMetadata


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_build_point_forecast_article_configs_from_csv(tmp_path):
    path = tmp_path / "point.csv"
    _write_csv(
        path,
        ["unique_id", "period", "demand", "forecast", "actual"],
        [
            ["A", 0, 10, 12, 11],
            ["A", 1, 9, 11, 10],
            ["B", 0, 3, 4, 5],
            ["B", 1, 4, 5, 4],
        ],
    )

    rows = iter_point_forecast_rows_from_csv(str(path))
    configs = build_point_forecast_article_configs(
        rows,
        lead_time={"A": 1, "B": 2},
        initial_on_hand=5,
        service_level_factor=0.8,
    )

    assert set(configs.keys()) == {"A", "B"}
    assert configs["A"].periods == 2
    assert configs["A"].lead_time == 1
    assert configs["B"].lead_time == 2


def test_build_percentile_forecast_candidates_from_csv(tmp_path):
    path = tmp_path / "percentile.csv"
    _write_csv(
        path,
        ["unique_id", "period", "demand", "target", "forecast"],
        [
            ["A", 0, 10, "p50", 11],
            ["A", 0, 10, "p90", 14],
            ["A", 1, 8, "p50", 9],
            ["A", 1, 8, "p90", 12],
        ],
    )

    rows = iter_percentile_forecast_rows_from_csv(str(path))
    configs = build_percentile_forecast_candidates(
        rows,
        lead_time=1,
        initial_on_hand=5,
    )

    assert set(configs.keys()) == {"A"}
    config = configs["A"]
    assert config.periods == 2
    assert set(config.forecast_candidates.keys()) == {"p50", "p90"}
    assert config.forecast_candidates["p50"] == [11, 9]


def test_build_point_forecast_article_configs_requires_contiguous_periods():
    rows = [PointForecastRow("A", 1, 10, 11, 12)]

    with pytest.raises(ValueError, match="Missing periods"):
        build_point_forecast_article_configs(
            rows,
            lead_time=1,
            initial_on_hand=5,
            service_level_factor=0.9,
        )


def test_iter_standard_simulation_rows_from_csv(tmp_path):
    path = tmp_path / "standard.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
            "current_stock",
            "is_forecast",
            "forecast_p50",
            "forecast_p90",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5, 8, "true", 11, 14],
        ],
    )

    rows = list(iter_standard_simulation_rows_from_csv(str(path)))

    assert rows == [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=11,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 11, "p90": 14},
            is_forecast=True,
        )
    ]


def test_build_configs_from_standard_rows():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=11,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 11, "p90": 14},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 10, "p90": 13},
        ),
    ]

    point_configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    percentile_configs = build_percentile_forecast_candidates_from_standard_rows(rows)

    assert set(point_configs.keys()) == {"A"}
    assert point_configs["A"].periods == 2
    assert point_configs["A"].initial_on_hand == 5
    assert set(percentile_configs["A"].forecast_candidates.keys()) == {"p50", "p90"}


def test_build_percentile_candidates_include_mean():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=11,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 11},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 10},
        ),
    ]

    configs = build_percentile_forecast_candidates_from_standard_rows(
        rows, include_mean=True
    )

    assert set(configs["A"].forecast_candidates.keys()) == {"p50", "mean"}


def test_build_point_forecast_article_configs_uses_actuals_override():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={},
            is_forecast=True,
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=11,
            forecast=13,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={},
            is_forecast=True,
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.5,
        actuals_override={"A": [9, 10]},
    )

    policy = configs["A"].policy
    assert policy._actual_values == [9, 10]


def test_optimize_point_forecast_policy_and_simulate_actuals():
    backtest_rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=11,
            actuals=10,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
    ]
    evaluation_rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-03",
            demand=8,
            forecast=9,
            actuals=8,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=6,
            forecast_percentiles={},
            is_forecast=True,
        )
    ]

    optimized, _, decisions = optimize_point_forecast_policy_and_simulate_actuals(
        backtest_rows,
        evaluation_rows,
        candidate_factors=[0.0],
    )

    assert optimized["A"].service_level_factor == 0.0
    assert decisions[0].sigma == 0.0
    assert decisions[0].demand == 8


def test_build_configs_use_current_stock_for_forecast_rows():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=12,
            forecast_percentiles={"p50": 11},
            is_forecast=True,
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=12,
            forecast_percentiles={"p50": 10},
            is_forecast=True,
        ),
    ]

    point_configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    percentile_configs = build_percentile_forecast_candidates_from_standard_rows(
        rows,
        include_mean=True,
    )

    assert point_configs["A"].initial_on_hand == 12
    assert percentile_configs["A"].initial_on_hand == 12


def test_generate_standard_simulation_rows_masks_actuals_after_forecast():
    rows = generate_standard_simulation_rows(
        n_unique_ids=1,
        periods=2,
        forecast_start_period=1,
        seed=1,
    )

    assert rows[0].actuals is not None
    assert rows[0].demand == rows[0].actuals
    assert rows[1].actuals is None
    assert rows[1].demand == rows[1].forecast


def test_build_replenishment_decisions_from_simulations():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=11,
            actuals=10,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-03",
            demand=8,
            forecast=9,
            actuals=8,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-04",
            demand=7,
            forecast=8,
            actuals=7,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    config = configs["A"]
    simulation = simulate_replenishment_with_aggregation(
        periods=config.periods,
        demand=config.demand,
        initial_on_hand=config.initial_on_hand,
        lead_time=config.lead_time,
        policy=config.policy,
        aggregation_window=2,
        holding_cost_per_unit=config.holding_cost_per_unit,
        stockout_cost_per_unit=config.stockout_cost_per_unit,
        order_cost_per_order=config.order_cost_per_order,
        order_cost_per_unit=config.order_cost_per_unit,
    )
    simulations = {"A": simulation}

    decisions = build_replenishment_decisions_from_simulations(
        rows,
        simulations,
        aggregation_window=2,
    )

    snapshots = simulation.snapshots
    forecast_values = [row.forecast for row in rows]
    actuals_values = [row.actuals for row in rows]
    review_period = 2
    forecast_horizon = 2
    rmse_window = 2
    horizon = rows[0].lead_time + forecast_horizon
    lead_time_factor = math.sqrt(horizon)
    safety_stock_values = []
    for index in range(len(snapshots)):
        max_index = min(index, len(actuals_values), len(forecast_values))
        if max_index <= 0:
            rmse = 0.0
        else:
            forecast_slice = forecast_values[:max_index]
            actuals_slice = actuals_values[:max_index]
            if rmse_window > 1:
                forecast_slice = aggregate_series(
                    forecast_slice,
                    periods=len(forecast_slice),
                    window=rmse_window,
                    extend_last=True,
                )
                actuals_slice = aggregate_series(
                    actuals_slice,
                    periods=len(actuals_slice),
                    window=rmse_window,
                    extend_last=False,
                )
            errors = [
                actuals_slice[i] - forecast_slice[i]
                for i in range(min(len(actuals_slice), len(forecast_slice)))
            ]
            if not errors:
                rmse = 0.0
            elif len(errors) == 1:
                rmse = abs(errors[0])
            else:
                rmse = math.sqrt(
                    sum(error * error for error in errors) / len(errors)
                )
        safety_stock_values.append(0.9 * rmse * lead_time_factor)
    expected = []
    running_stock = rows[0].current_stock
    lead_offset = max(1, rows[0].lead_time)
    for index, snapshot in enumerate(snapshots):
        start = index
        end = min(start + 2, len(forecast_values))
        stock_before = float(running_stock) + float(snapshot.received)
        starting_stock = int(round(stock_before))
        forecast_quantity = sum(forecast_values[start:end]) / 2
        lead_start = start + lead_offset
        lead_end = lead_start + forecast_horizon
        if lead_start >= len(forecast_values):
            forecast_quantity_lead_time = (
                forecast_values[-1] * forecast_horizon
                if forecast_values
                else 0
            )
        elif lead_end <= len(forecast_values):
            forecast_quantity_lead_time = sum(
                forecast_values[lead_start:lead_end]
            )
        else:
            forecast_quantity_lead_time = sum(
                forecast_values[lead_start:len(forecast_values)]
            )
            extra = lead_end - len(forecast_values)
            if extra > 0 and forecast_values:
                forecast_quantity_lead_time += forecast_values[-1] * extra
        forecast_quantity_lead_time = (
            forecast_quantity_lead_time / review_period
        )
        stock_after = stock_before - float(snapshot.demand)
        if stock_after < 0:
            missed_sales = int(round(-stock_after))
            stock_after = 0.0
        else:
            missed_sales = 0
        ending_stock = int(round(stock_after))
        current_stock = ending_stock
        running_stock = stock_after
        cycle_stock = sum(forecast_values[start:end])
        extra = start + 2 - len(forecast_values)
        if extra > 0 and forecast_values:
            cycle_stock += forecast_values[-1] * extra
        expected.append(
            ReplenishmentDecisionRow(
                unique_id="A",
                ds=rows[start].ds,
                quantity=snapshot.order_placed,
                demand=snapshot.demand,
                forecast_quantity=forecast_quantity,
                forecast_quantity_lead_time=forecast_quantity_lead_time,
                reorder_point=forecast_values[start] + safety_stock_values[index],
                order_up_to=(
                    forecast_values[start]
                    + safety_stock_values[index]
                    + cycle_stock
                ),
                incoming_stock=snapshot.received,
                starting_stock=starting_stock,
                ending_stock=ending_stock,
                safety_stock=safety_stock_values[index],
                starting_on_hand=snapshot.starting_on_hand,
                ending_on_hand=snapshot.ending_on_hand,
                current_stock=current_stock,
                on_order=snapshot.on_order,
                backorders=snapshot.backorders,
                missed_sales=missed_sales,
                sigma=0.9,
                service_level_mode="factor",
                aggregation_window=review_period,
                review_period=review_period,
                forecast_horizon=forecast_horizon,
                rmse_window=rmse_window,
            )
        )

    assert decisions == expected


def test_build_replenishment_decisions_from_optimization_results():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=11,
            actuals=10,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-03",
            demand=8,
            forecast=9,
            actuals=8,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    optimization_results = optimize_aggregation_windows(
        configs,
        candidate_windows=[1, 2],
    )

    decisions = build_replenishment_decisions_from_optimization_results(
        rows,
        optimization_results,
    )

    simulation = optimization_results["A"].simulation
    window = optimization_results["A"].window
    forecast_values = [row.forecast for row in rows]
    expected = []
    running_stock = rows[0].current_stock
    lead_offset = max(1, rows[0].lead_time)
    for index, snapshot in enumerate(simulation.snapshots):
        start = index
        end = min(start + window, len(forecast_values))
        forecast_quantity = sum(forecast_values[start:end]) / window
        total_horizon = window
        if window > 1:
            lead_start = start + lead_offset
            lead_end = lead_start + total_horizon
            if lead_start >= len(forecast_values):
                forecast_quantity_lead_time = (
                    forecast_values[-1] * total_horizon
                    if forecast_values
                    else 0
                )
            elif lead_end <= len(forecast_values):
                forecast_quantity_lead_time = sum(
                    forecast_values[lead_start:lead_end]
                )
            else:
                forecast_quantity_lead_time = sum(
                    forecast_values[lead_start:len(forecast_values)]
                )
                extra = lead_end - len(forecast_values)
                if extra > 0 and forecast_values:
                    forecast_quantity_lead_time += forecast_values[-1] * extra
            forecast_quantity_lead_time = forecast_quantity_lead_time / window
        else:
            start_period = start + lead_offset
            end_period = start_period + max(1, total_horizon)
            if end_period <= len(forecast_values):
                forecast_quantity_lead_time = sum(
                    forecast_values[start_period:end_period]
                )
            else:
                forecast_quantity_lead_time = sum(
                    forecast_values[start_period:len(forecast_values)]
                )
                extra = end_period - len(forecast_values)
                if extra > 0 and forecast_values:
                    forecast_quantity_lead_time += forecast_values[-1] * extra
        stock_before = float(running_stock) + float(snapshot.received)
        starting_stock = int(round(stock_before))
        stock_after = stock_before - float(snapshot.demand)
        if stock_after < 0:
            missed_sales = int(round(-stock_after))
            stock_after = 0.0
        else:
            missed_sales = 0
        ending_stock = int(round(stock_after))
        current_stock = ending_stock
        running_stock = stock_after
        cycle_stock = sum(forecast_values[start:end])
        extra = start + window - len(forecast_values)
        if extra > 0 and forecast_values:
            cycle_stock += forecast_values[-1] * extra
        expected.append(
            ReplenishmentDecisionRow(
                unique_id="A",
                ds=rows[start].ds,
                quantity=snapshot.order_placed,
                demand=snapshot.demand,
                forecast_quantity=forecast_quantity,
                forecast_quantity_lead_time=forecast_quantity_lead_time,
                reorder_point=forecast_values[start],
                order_up_to=forecast_values[start] + cycle_stock,
                incoming_stock=snapshot.received,
                starting_stock=starting_stock,
                ending_stock=ending_stock,
                starting_on_hand=snapshot.starting_on_hand,
                ending_on_hand=snapshot.ending_on_hand,
                current_stock=current_stock,
                on_order=snapshot.on_order,
                backorders=snapshot.backorders,
                    missed_sales=missed_sales,
                    aggregation_window=window,
                    review_period=window,
                    forecast_horizon=window,
                    rmse_window=window,
                )
            )

    assert decisions == expected


def test_build_replenishment_decisions_from_simulations_with_metadata_inputs():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=11,
            actuals=10,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    simulations = simulate_replenishment_for_articles(configs)

    decisions = build_replenishment_decisions_from_simulations(
        rows,
        simulations,
        sigma=1.25,
        percentile_target="p90",
    )

    running_stock = rows[0].current_stock
    lead_time_factor = math.sqrt(rows[0].lead_time + 1)
    safety_stock_values = [
        0.0,
        1.25 * abs(rows[0].actuals - rows[0].forecast) * lead_time_factor,
    ]
    expected = []
    for index, snapshot in enumerate(simulations["A"].snapshots):
        stock_before = float(running_stock) + float(snapshot.received)
        starting_stock = int(round(stock_before))
        stock_after = stock_before - float(snapshot.demand)
        if stock_after < 0:
            missed_sales = int(round(-stock_after))
            stock_after = 0.0
        else:
            missed_sales = 0
        ending_stock = int(round(stock_after))
        current_stock = ending_stock
        running_stock = stock_after
        expected.append(
            ReplenishmentDecisionRow(
                unique_id="A",
                ds=rows[index].ds,
                quantity=snapshot.order_placed,
                demand=snapshot.demand,
                forecast_quantity=None,
                reorder_point=None,
                incoming_stock=snapshot.received,
                starting_stock=starting_stock,
                ending_stock=ending_stock,
                safety_stock=safety_stock_values[index],
                starting_on_hand=snapshot.starting_on_hand,
                ending_on_hand=snapshot.ending_on_hand,
                current_stock=current_stock,
                on_order=snapshot.on_order,
                backorders=snapshot.backorders,
                missed_sales=missed_sales,
                sigma=1.25,
                aggregation_window=1,
                review_period=1,
                forecast_horizon=1,
                rmse_window=1,
                percentile_target="p90",
            )
        )

    assert decisions == expected


def test_build_replenishment_decisions_from_simulations_merges_metadata():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=11,
            actuals=10,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=8,
            current_stock=8,
            forecast_percentiles={},
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    simulations = simulate_replenishment_for_articles(configs)

    decisions = build_replenishment_decisions_from_simulations(
        rows,
        simulations,
        sigma=1.5,
        decision_metadata={
            "A": ReplenishmentDecisionMetadata(percentile_target="p95")
        },
    )

    running_stock = rows[0].current_stock
    forecast_values = [row.forecast for row in rows]
    lead_offset = max(1, rows[0].lead_time)
    lead_time_factor = math.sqrt(rows[0].lead_time + 1)
    safety_stock_values = [
        0.0,
        1.5 * abs(rows[0].actuals - rows[0].forecast) * lead_time_factor,
    ]
    expected = []
    for index, snapshot in enumerate(simulations["A"].snapshots):
        stock_before = float(running_stock) + float(snapshot.received)
        starting_stock = int(round(stock_before))
        forecast_quantity = forecast_values[index]
        total_horizon = 1
        start_period = index + lead_offset
        end_period = start_period + max(1, total_horizon)
        if end_period <= len(forecast_values):
            forecast_quantity_lead_time = sum(
                forecast_values[start_period:end_period]
            )
        else:
            forecast_quantity_lead_time = sum(
                forecast_values[start_period:len(forecast_values)]
            )
            extra = end_period - len(forecast_values)
            if extra > 0 and forecast_values:
                forecast_quantity_lead_time += forecast_values[-1] * extra
        stock_after = stock_before - float(snapshot.demand)
        if stock_after < 0:
            missed_sales = int(round(-stock_after))
            stock_after = 0.0
        else:
            missed_sales = 0
        ending_stock = int(round(stock_after))
        current_stock = ending_stock
        running_stock = stock_after
        expected.append(
            ReplenishmentDecisionRow(
                unique_id="A",
                ds=rows[index].ds,
                quantity=snapshot.order_placed,
                demand=snapshot.demand,
                forecast_quantity=forecast_quantity,
                forecast_quantity_lead_time=forecast_quantity_lead_time,
                reorder_point=forecast_values[index] + safety_stock_values[index],
                order_up_to=(
                    forecast_values[index]
                    + safety_stock_values[index]
                    + forecast_values[index]
                ),
                incoming_stock=snapshot.received,
                starting_stock=starting_stock,
                ending_stock=ending_stock,
                safety_stock=safety_stock_values[index],
                starting_on_hand=snapshot.starting_on_hand,
                ending_on_hand=snapshot.ending_on_hand,
                current_stock=current_stock,
                on_order=snapshot.on_order,
                backorders=snapshot.backorders,
                missed_sales=missed_sales,
                sigma=1.5,
                aggregation_window=1,
                review_period=1,
                forecast_horizon=1,
                rmse_window=1,
                percentile_target="p95",
            )
        )

    assert decisions == expected


def test_decision_current_stock_rolls_forward_from_current_stock():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=4,
            forecast=4,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=20,
            forecast_percentiles={"p50": 4},
            is_forecast=True,
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=4,
            forecast=4,
            actuals=None,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=20,
            forecast_percentiles={"p50": 4},
            is_forecast=True,
        ),
    ]

    configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.0,
    )
    simulations = simulate_replenishment_for_articles(configs)

    decisions = build_replenishment_decisions_from_simulations(
        rows,
        simulations,
    )

    assert decisions[0].current_stock == 16
    assert decisions[1].current_stock == 12


def test_standard_simulation_rows_from_dataframe_history_and_cutoff():
    class FakeDataFrame:
        def __init__(self, records):
            self._records = records

        def to_dict(self, orient="records"):
            assert orient == "records"
            return self._records

    records = [
        {
            "unique_id": "A",
            "ds": "2024-01-01",
            "history": 10,
            "forecast": 12,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
            "forecast_p50": 11,
        },
        {
            "unique_id": "A",
            "ds": "2024-02-01",
            "history": float("nan"),
            "forecast": 13,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
            "forecast_p50": 12,
        },
    ]

    rows = standard_simulation_rows_from_dataframe(
        FakeDataFrame(records),
        cutoff="2024-01-01",
    )

    assert rows[0].actuals == 10
    assert rows[0].is_forecast is False
    assert rows[1].actuals is None
    assert rows[1].is_forecast is True


def test_standard_simulation_rows_from_dataframe_with_actuals_only():
    class FakeDataFrame:
        def __init__(self, records):
            self._records = records

        def to_dict(self, orient="records"):
            assert orient == "records"
            return self._records

    records = [
        {
            "unique_id": "A",
            "ds": "2024-01-01",
            "forecast": 12,
            "actuals": 10,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
        },
        {
            "unique_id": "A",
            "ds": "2024-02-01",
            "forecast": 13,
            "actuals": float("nan"),
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
        },
    ]

    rows = standard_simulation_rows_from_dataframe(FakeDataFrame(records))

    assert rows[0].demand == 10
    assert rows[0].actuals == 10
    assert rows[0].is_forecast is False
    assert rows[1].demand == 13
    assert rows[1].actuals is None
    assert rows[1].is_forecast is True


def test_split_standard_simulation_rows_accepts_dataframe():
    class FakeDataFrame:
        def __init__(self, records):
            self._records = records

        def to_dict(self, orient="records"):
            assert orient == "records"
            return self._records

    records = [
        {
            "unique_id": "A",
            "ds": "2024-01-01",
            "forecast": 12,
            "actuals": 10,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
        },
        {
            "unique_id": "A",
            "ds": "2024-02-01",
            "forecast": 13,
            "actuals": float("nan"),
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
        },
    ]

    backtest_rows, forecast_rows = split_standard_simulation_rows(
        FakeDataFrame(records)
    )

    assert [row.ds for row in backtest_rows] == ["2024-01-01"]
    assert [row.ds for row in forecast_rows] == ["2024-02-01"]


def test_iter_standard_simulation_rows_warns_on_missing_columns(tmp_path):
    path = tmp_path / "missing.csv"
    _write_csv(
        path,
        ["unique_id", "ds", "demand"],
        [
            ["A", "2024-01-01", 10],
        ],
    )

    with pytest.warns(UserWarning, match="Missing required columns"):
        with pytest.raises(ValueError, match="missing required columns"):
            list(iter_standard_simulation_rows_from_csv(str(path)))


def test_iter_standard_simulation_rows_requires_current_stock(tmp_path):
    path = tmp_path / "missing_stock.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5],
        ],
    )

    with pytest.warns(UserWarning, match="current_stock"):
        with pytest.raises(ValueError, match="current_stock"):
            list(iter_standard_simulation_rows_from_csv(str(path)))


def test_iter_standard_simulation_rows_requires_matching_initial_inventory(tmp_path):
    path = tmp_path / "mismatch.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
            "initial_demand",
            "current_stock",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5, 7, 8],
        ],
    )

    with pytest.warns(UserWarning, match="Initial inventory mismatch"):
        with pytest.raises(ValueError, match="initial_on_hand and initial_demand"):
            list(iter_standard_simulation_rows_from_csv(str(path)))
