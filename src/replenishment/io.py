"""Helpers for loading bulk optimization inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import random
import string
import warnings

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


@dataclass(frozen=True)
class StandardSimulationRow:
    unique_id: str
    ds: str
    demand: int
    forecast: int
    actuals: int
    holding_cost_per_unit: float
    stockout_cost_per_unit: float
    order_cost_per_order: float
    lead_time: int
    initial_on_hand: int
    current_stock: int
    forecast_percentiles: Mapping[str, int]


def generate_standard_simulation_rows(
    *,
    n_unique_ids: int,
    periods: int,
    start_date: str | date | datetime = "2024-01-01",
    frequency_days: int = 30,
    history_mean: float = 20.0,
    history_std: float = 5.0,
    forecast_mean: float = 20.0,
    forecast_std: float = 4.0,
    percentile_multipliers: Mapping[str, float] | None = None,
    holding_cost_per_unit: float = 0.5,
    stockout_cost_per_unit: float = 3.0,
    order_cost_per_order: float = 12.5,
    lead_time: int = 1,
    initial_on_hand: int = 30,
    current_stock: int | None = None,
    seed: int | None = None,
) -> list[StandardSimulationRow]:
    """Generate synthetic rows that match the standard simulation schema."""
    if n_unique_ids <= 0:
        raise ValueError("n_unique_ids must be positive.")
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if frequency_days <= 0:
        raise ValueError("frequency_days must be positive.")

    if isinstance(start_date, str):
        base_date = date.fromisoformat(start_date)
    elif isinstance(start_date, datetime):
        base_date = start_date.date()
    else:
        base_date = start_date

    rng = random.Random(seed)

    def sample_int(mean: float, std: float) -> int:
        value = rng.gauss(mean, std)
        return max(0, int(round(value)))

    if percentile_multipliers is None:
        percentile_multipliers = {"p50": 1.0, "p90": 1.25}

    unique_ids: list[str] = []
    for index in range(n_unique_ids):
        if index < len(string.ascii_uppercase):
            unique_ids.append(string.ascii_uppercase[index])
        else:
            unique_ids.append(f"SKU-{index + 1:03d}")

    rows: list[StandardSimulationRow] = []
    for unique_id in unique_ids:
        for period in range(periods):
            ds = (base_date + timedelta(days=period * frequency_days)).isoformat()
            demand = sample_int(history_mean, history_std)
            actuals = sample_int(history_mean, history_std)
            forecast = sample_int(forecast_mean, forecast_std)
            forecast_percentiles = {
                label: max(0, int(round(forecast * multiplier)))
                for label, multiplier in percentile_multipliers.items()
            }
            rows.append(
                StandardSimulationRow(
                    unique_id=unique_id,
                    ds=ds,
                    demand=demand,
                    forecast=forecast,
                    actuals=actuals,
                    holding_cost_per_unit=holding_cost_per_unit,
                    stockout_cost_per_unit=stockout_cost_per_unit,
                    order_cost_per_order=order_cost_per_order,
                    lead_time=lead_time,
                    initial_on_hand=initial_on_hand,
                    current_stock=initial_on_hand if current_stock is None else current_stock,
                    forecast_percentiles=forecast_percentiles,
                )
            )
    return rows


def standard_simulation_rows_to_dicts(
    rows: Iterable[StandardSimulationRow],
    *,
    forecast_prefix: str = "forecast_",
) -> list[dict[str, str | int | float]]:
    """Convert standard rows into dictionaries suitable for DataFrame or CSV use."""
    serialized: list[dict[str, str | int | float]] = []
    for row in rows:
        entry: dict[str, str | int | float] = {
            "unique_id": row.unique_id,
            "ds": row.ds,
            "demand": row.demand,
            "forecast": row.forecast,
            "actuals": row.actuals,
            "holding_cost_per_unit": row.holding_cost_per_unit,
            "stockout_cost_per_unit": row.stockout_cost_per_unit,
            "order_cost_per_order": row.order_cost_per_order,
            "lead_time": row.lead_time,
            "initial_on_hand": row.initial_on_hand,
            "current_stock": row.current_stock,
        }
        for label, value in row.forecast_percentiles.items():
            entry[f"{forecast_prefix}{label}"] = value
        serialized.append(entry)
    return serialized


def write_standard_simulation_rows_to_csv(
    path: str,
    rows: Iterable[StandardSimulationRow],
    *,
    forecast_prefix: str = "forecast_",
) -> None:
    """Write standard simulation rows to a CSV that matches the README schema."""
    rows_list = list(rows)
    percentile_labels: list[str] = []
    for row in rows_list:
        for label in row.forecast_percentiles:
            if label not in percentile_labels:
                percentile_labels.append(label)
    fieldnames = [
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
    ] + [f"{forecast_prefix}{label}" for label in percentile_labels]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in standard_simulation_rows_to_dicts(
            rows_list, forecast_prefix=forecast_prefix
        ):
            writer.writerow(entry)


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
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                period_field,
                demand_field,
                forecast_field,
                actual_field,
            ],
            context="point-forecast CSV",
        )
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
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                period_field,
                demand_field,
                target_field,
                forecast_field,
            ],
            context="percentile-forecast CSV",
        )
        for row in reader:
            yield PercentileForecastRow(
                unique_id=row[unique_id_field],
                period=int(row[period_field]),
                demand=int(row[demand_field]),
                target=row[target_field],
                forecast=int(row[forecast_field]),
            )


def iter_standard_simulation_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    ds_field: str = "ds",
    demand_field: str = "demand",
    forecast_field: str = "forecast",
    actuals_field: str = "actuals",
    holding_cost_per_unit_field: str = "holding_cost_per_unit",
    stockout_cost_per_unit_field: str = "stockout_cost_per_unit",
    order_cost_per_order_field: str = "order_cost_per_order",
    lead_time_field: str = "lead_time",
    initial_on_hand_field: str = "initial_on_hand",
    initial_demand_field: str = "initial_demand",
    current_stock_field: str = "current_stock",
    forecast_prefix: str = "forecast_",
) -> Iterator[StandardSimulationRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                ds_field,
                demand_field,
                forecast_field,
                actuals_field,
                holding_cost_per_unit_field,
                stockout_cost_per_unit_field,
                order_cost_per_order_field,
                lead_time_field,
                current_stock_field,
            ],
            context="standard simulation CSV",
        )
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[],
            require_any_of=[initial_on_hand_field, initial_demand_field],
            context="standard simulation CSV",
        )
        for row in reader:
            forecast_percentiles = {
                key[len(forecast_prefix) :]: int(value)
                for key, value in row.items()
                if key.startswith(forecast_prefix)
                and key != forecast_field
                and value != ""
            }
            _validate_matching_initial_inventory(
                row,
                initial_on_hand_field=initial_on_hand_field,
                initial_demand_field=initial_demand_field,
            )
            initial_on_hand = _coalesce_row_value(
                row, initial_on_hand_field, initial_demand_field
            )
            if initial_on_hand is None:
                raise ValueError(
                    "Initial on-hand inventory is required (initial_on_hand or initial_demand)."
                )
            current_stock_value = _coalesce_row_value(row, current_stock_field)
            if current_stock_value is None:
                current_stock_value = initial_on_hand
            yield StandardSimulationRow(
                unique_id=row[unique_id_field],
                ds=row[ds_field],
                demand=int(row[demand_field]),
                forecast=int(row[forecast_field]),
                actuals=int(row[actuals_field]),
                holding_cost_per_unit=float(row[holding_cost_per_unit_field]),
                stockout_cost_per_unit=float(row[stockout_cost_per_unit_field]),
                order_cost_per_order=float(row[order_cost_per_order_field]),
                lead_time=int(row[lead_time_field]),
                initial_on_hand=initial_on_hand,
                current_stock=current_stock_value,
                forecast_percentiles=forecast_percentiles,
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


def build_point_forecast_article_configs_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    service_level_factor: Mapping[str, float] | float,
) -> dict[str, ArticleSimulationConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        forecast = [row.forecast for row in ordered]
        actuals = [row.actuals for row in ordered]
        policy = PointForecastOptimizationPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=lead_time,
            service_level_factor=_resolve_value(
                service_level_factor, unique_id, "service_level_factor"
            ),
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=initial_on_hand,
            lead_time=lead_time,
            policy=policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


def build_percentile_forecast_candidates_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, ForecastCandidatesConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ForecastCandidatesConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        targets = _validate_percentile_targets(unique_id, ordered)
        candidate_series = {
            target: [row.forecast_percentiles[target] for row in ordered]
            for target in targets
        }
        configs[unique_id] = ForecastCandidatesConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=initial_on_hand,
            lead_time=lead_time,
            forecast_candidates=candidate_series,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
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


def _coalesce_row_value(row: Mapping[str, str], *fields: str) -> int | None:
    for field in fields:
        if field in row and row[field] != "":
            return int(row[field])
    return None


def _group_standard_rows(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, dict[str, StandardSimulationRow]]:
    grouped: dict[str, dict[str, StandardSimulationRow]] = defaultdict(dict)
    for row in rows:
        article_rows = grouped[row.unique_id]
        if row.ds in article_rows:
            raise ValueError(f"Duplicate ds '{row.ds}' for unique_id '{row.unique_id}'.")
        article_rows[row.ds] = row
    return grouped


def _order_rows_by_ds(
    unique_id: str, ds_rows: Mapping[str, StandardSimulationRow]
) -> list[StandardSimulationRow]:
    if not ds_rows:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    return [ds_rows[ds] for ds in sorted(ds_rows)]


def _ensure_constant(
    unique_id: str, rows: Iterable[StandardSimulationRow], field: str
) -> int | float:
    values = {getattr(row, field) for row in rows}
    if len(values) != 1:
        value_list = ", ".join(str(value) for value in sorted(values))
        raise ValueError(
            f"{field} must be constant for unique_id '{unique_id}': {value_list}."
        )
    return values.pop()


def _validate_percentile_targets(
    unique_id: str, rows: Iterable[StandardSimulationRow]
) -> list[str]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    targets = set(rows_list[0].forecast_percentiles.keys())
    if not targets:
        raise ValueError(f"No percentile forecasts provided for unique_id '{unique_id}'.")
    for row in rows_list[1:]:
        if set(row.forecast_percentiles.keys()) != targets:
            raise ValueError(
                f"Percentile forecasts must be consistent across ds for unique_id '{unique_id}'."
            )
    return sorted(targets)


def _validate_required_columns(
    fieldnames: list[str] | None,
    *,
    required_fields: Iterable[str],
    require_any_of: Iterable[str] | None = None,
    context: str,
) -> None:
    if not fieldnames:
        warnings.warn(f"Missing header row for {context}.", stacklevel=2)
        raise ValueError(f"{context} is missing a header row.")
    field_set = set(fieldnames)
    missing_required = [field for field in required_fields if field not in field_set]
    if require_any_of:
        alternatives = [field for field in require_any_of if field in field_set]
        if not alternatives:
            missing_required.extend(require_any_of)
    if missing_required:
        missing_display = ", ".join(missing_required)
        warnings.warn(
            f"Missing required columns for {context}: {missing_display}.",
            stacklevel=2,
        )
        raise ValueError(
            f"{context} is missing required columns: {missing_display}."
        )


def _validate_matching_initial_inventory(
    row: Mapping[str, str],
    *,
    initial_on_hand_field: str,
    initial_demand_field: str,
) -> None:
    if (
        initial_on_hand_field in row
        and initial_demand_field in row
        and row[initial_on_hand_field] != ""
        and row[initial_demand_field] != ""
    ):
        initial_on_hand = int(row[initial_on_hand_field])
        initial_demand = int(row[initial_demand_field])
        if initial_on_hand != initial_demand:
            warnings.warn(
                "Initial inventory mismatch between "
                f"{initial_on_hand_field} ({initial_on_hand}) and "
                f"{initial_demand_field} ({initial_demand}).",
                stacklevel=2,
            )
            raise ValueError(
                f"{initial_on_hand_field} and {initial_demand_field} must match when both are provided."
            )
