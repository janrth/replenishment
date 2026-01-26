"""Helpers for aggregating demand and forecast series into coarser windows."""

from __future__ import annotations

from collections.abc import Iterable
import math

from .policies import ForecastBasedPolicy, ForecastSeriesPolicy
from .simulation import DemandModel, OrderingPolicy


def aggregate_periods(periods: int, window: int) -> int:
    if periods <= 0:
        raise ValueError("Periods must be positive.")
    if window <= 0:
        raise ValueError("Aggregation window must be positive.")
    return math.ceil(periods / window)


def aggregate_series(
    series: Iterable[int] | DemandModel,
    *,
    periods: int,
    window: int,
    extend_last: bool = False,
) -> list[int]:
    if periods <= 0:
        raise ValueError("Periods must be positive.")
    if window <= 0:
        raise ValueError("Aggregation window must be positive.")

    if callable(series):
        values = [series(period) for period in range(periods)]
    else:
        values = list(series)
        if len(values) < periods:
            if not values or not extend_last:
                raise ValueError("Series shorter than periods.")
            values = values + [values[-1]] * (periods - len(values))

    return [
        sum(values[index : index + window]) for index in range(0, periods, window)
    ]


def aggregate_lead_time(lead_time: int, window: int) -> int:
    if lead_time < 0:
        raise ValueError("Lead time cannot be negative.")
    if window <= 0:
        raise ValueError("Aggregation window must be positive.")
    return math.ceil(lead_time / window)


def aggregate_policy(
    policy: OrderingPolicy,
    *,
    periods: int,
    window: int,
    lead_time: int,
) -> OrderingPolicy:
    if isinstance(policy, ForecastBasedPolicy):
        aggregated_forecast = aggregate_series(
            policy.forecast,
            periods=periods,
            window=window,
            extend_last=True,
        )
        aggregated_actuals = aggregate_series(
            policy.actuals,
            periods=periods,
            window=window,
            extend_last=False,
        )
        return ForecastBasedPolicy(
            forecast=aggregated_forecast,
            actuals=aggregated_actuals,
            lead_time=aggregate_lead_time(lead_time, window),
            service_level_factor=policy.service_level_factor,
        )
    if isinstance(policy, ForecastSeriesPolicy):
        aggregated_forecast = aggregate_series(
            policy.forecast,
            periods=periods,
            window=window,
            extend_last=True,
        )
        return ForecastSeriesPolicy(
            forecast=aggregated_forecast,
            lead_time=aggregate_lead_time(lead_time, window),
        )
    return policy


def simulate_replenishment_with_aggregation(
    *,
    periods: int,
    demand: Iterable[int] | DemandModel,
    initial_on_hand: int,
    lead_time: int,
    policy: OrderingPolicy,
    aggregation_window: int,
    holding_cost_per_unit: float = 0.0,
    stockout_cost_per_unit: float = 0.0,
):
    from .simulation import simulate_replenishment

    aggregated_demand = aggregate_series(
        demand,
        periods=periods,
        window=aggregation_window,
        extend_last=False,
    )
    aggregated_policy = aggregate_policy(
        policy,
        periods=periods,
        window=aggregation_window,
        lead_time=lead_time,
    )
    aggregated_periods = aggregate_periods(periods, aggregation_window)
    aggregated_lead_time = aggregate_lead_time(lead_time, aggregation_window)

    return simulate_replenishment(
        periods=aggregated_periods,
        demand=aggregated_demand,
        initial_on_hand=initial_on_hand,
        lead_time=aggregated_lead_time,
        policy=aggregated_policy,
        holding_cost_per_unit=holding_cost_per_unit,
        stockout_cost_per_unit=stockout_cost_per_unit,
    )
