"""Helpers for loading bulk optimization inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
import csv
from dataclasses import dataclass

from .optimization import ForecastCandidatesConfig
from .policies import PointForecastOptimizationPolicy
from .simulation import ArticleSimulationConfig


@dataclass(frozen=True)
class PointForecastRow:
    unique_id: str
    period: int
    demand: int
    forecast: int
    actual: int


@dataclass(frozen=True)
class PercentileForecastRow:
    unique_id: str
    period: int
    demand: int
    target: float | str
    forecast: int


def iter_point_forecast_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    period_field: str = "period",
    demand_field: str = "demand",
    forecast_field: str = "forecast",
    actual_field: str = "actual",
) -> Iterator[PointForecastRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield PointForecastRow(
                unique_id=row[unique_id_field],
                period=int(row[period_field]),
                demand=int(row[demand_field]),
                forecast=int(row[forecast_field]),
                actual=int(row[actual_field]),
            )


def iter_percentile_forecast_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    period_field: str = "period",
    demand_field: str = "demand",
    target_field: str = "target",
    forecast_field: str = "forecast",
) -> Iterator[PercentileForecastRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield PercentileForecastRow(
                unique_id=row[unique_id_field],
                period=int(row[period_field]),
                demand=int(row[demand_field]),
                target=row[target_field],
                forecast=int(row[forecast_field]),
            )


def build_point_forecast_article_configs(
    rows: Iterable[PointForecastRow],
    *,
    lead_time: Mapping[str, int] | int,
    initial_on_hand: Mapping[str, int] | int,
    service_level_factor: Mapping[str, float] | float,
    holding_cost_per_unit: Mapping[str, float] | float = 0.0,
    stockout_cost_per_unit: Mapping[str, float] | float = 0.0,
    order_cost_per_order: Mapping[str, float] | float = 0.0,
    order_cost_per_unit: Mapping[str, float] | float = 0.0,
) -> dict[str, ArticleSimulationConfig]:
    grouped: dict[str, dict[int, PointForecastRow]] = defaultdict(dict)
    for row in rows:
        if row.period < 0:
            raise ValueError("Period cannot be negative.")
        article_rows = grouped[row.unique_id]
        if row.period in article_rows:
            raise ValueError(
                f"Duplicate period {row.period} for unique_id '{row.unique_id}'."
            )
        article_rows[row.period] = row

    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, period_rows in grouped.items():
        periods = _validate_periods(unique_id, period_rows)
        demand = [period_rows[index].demand for index in range(periods)]
        forecast = [period_rows[index].forecast for index in range(periods)]
        actuals = [period_rows[index].actual for index in range(periods)]
        policy = PointForecastOptimizationPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=_resolve_value(lead_time, unique_id, "lead_time"),
            service_level_factor=_resolve_value(
                service_level_factor, unique_id, "service_level_factor"
            ),
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=periods,
            demand=demand,
            initial_on_hand=_resolve_value(
                initial_on_hand, unique_id, "initial_on_hand"
            ),
            lead_time=_resolve_value(lead_time, unique_id, "lead_time"),
            policy=policy,
            holding_cost_per_unit=_resolve_value(
                holding_cost_per_unit, unique_id, "holding_cost_per_unit"
            ),
            stockout_cost_per_unit=_resolve_value(
                stockout_cost_per_unit, unique_id, "stockout_cost_per_unit"
            ),
            order_cost_per_order=_resolve_value(
                order_cost_per_order, unique_id, "order_cost_per_order"
            ),
            order_cost_per_unit=_resolve_value(
                order_cost_per_unit, unique_id, "order_cost_per_unit"
            ),
        )
    return configs


def build_percentile_forecast_candidates(
    rows: Iterable[PercentileForecastRow],
    *,
    lead_time: Mapping[str, int] | int,
    initial_on_hand: Mapping[str, int] | int,
    holding_cost_per_unit: Mapping[str, float] | float = 0.0,
    stockout_cost_per_unit: Mapping[str, float] | float = 0.0,
    order_cost_per_order: Mapping[str, float] | float = 0.0,
    order_cost_per_unit: Mapping[str, float] | float = 0.0,
) -> dict[str, ForecastCandidatesConfig]:
    demand_by_article: dict[str, dict[int, int]] = defaultdict(dict)
    forecast_by_article: dict[str, dict[float | str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in rows:
        if row.period < 0:
            raise ValueError("Period cannot be negative.")
        demand_rows = demand_by_article[row.unique_id]
        if row.period in demand_rows and demand_rows[row.period] != row.demand:
            raise ValueError(
                "Demand must be consistent across targets for each period."
            )
        demand_rows[row.period] = row.demand
        target_rows = forecast_by_article[row.unique_id][row.target]
        if row.period in target_rows:
            raise ValueError(
                f"Duplicate period {row.period} for unique_id '{row.unique_id}' target '{row.target}'."
            )
        target_rows[row.period] = row.forecast

    configs: dict[str, ForecastCandidatesConfig] = {}
    for unique_id, period_rows in demand_by_article.items():
        periods = _validate_periods(unique_id, period_rows)
        demand = [period_rows[index] for index in range(periods)]
        candidate_series: dict[float | str, list[int]] = {}
        for target, target_rows in forecast_by_article[unique_id].items():
            _validate_periods(unique_id, target_rows)
            candidate_series[target] = [target_rows[index] for index in range(periods)]

        configs[unique_id] = ForecastCandidatesConfig(
            periods=periods,
            demand=demand,
            initial_on_hand=_resolve_value(
                initial_on_hand, unique_id, "initial_on_hand"
            ),
            lead_time=_resolve_value(lead_time, unique_id, "lead_time"),
            forecast_candidates=candidate_series,
            holding_cost_per_unit=_resolve_value(
                holding_cost_per_unit, unique_id, "holding_cost_per_unit"
            ),
            stockout_cost_per_unit=_resolve_value(
                stockout_cost_per_unit, unique_id, "stockout_cost_per_unit"
            ),
            order_cost_per_order=_resolve_value(
                order_cost_per_order, unique_id, "order_cost_per_order"
            ),
            order_cost_per_unit=_resolve_value(
                order_cost_per_unit, unique_id, "order_cost_per_unit"
            ),
        )

    return configs


def _resolve_value(
    value: Mapping[str, int | float] | int | float, unique_id: str, name: str
) -> int | float:
    if isinstance(value, Mapping):
        if unique_id not in value:
            raise ValueError(f"Missing {name} for unique_id '{unique_id}'.")
        return value[unique_id]
    return value


def _validate_periods(unique_id: str, period_rows: Mapping[int, object]) -> int:
    if not period_rows:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    max_period = max(period_rows)
    expected = set(range(max_period + 1))
    missing = expected.difference(period_rows.keys())
    if missing:
        missing_display = ", ".join(str(period) for period in sorted(missing))
        raise ValueError(
            f"Missing periods for unique_id '{unique_id}': {missing_display}."
        )
    return max_period + 1
