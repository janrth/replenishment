"""Helpers for loading bulk optimization inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import random
import string
import warnings

from .optimization import ForecastCandidatesConfig
from .policies import PointForecastOptimizationPolicy
from .simulation import ArticleSimulationConfig, SimulationResult


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
    actuals: int | float | None
    holding_cost_per_unit: float
    stockout_cost_per_unit: float
    order_cost_per_order: float
    lead_time: int
    initial_on_hand: int
    current_stock: int
    forecast_percentiles: Mapping[str, int]
    is_forecast: bool = False


@dataclass(frozen=True)
class ReplenishmentDecisionRow:
    unique_id: str
    ds: str
    quantity: int


def generate_standard_simulation_rows(
    *,
    n_unique_ids: int,
    periods: int,
    start_date: str | date | datetime = "2024-01-01",
    frequency_days: int = 30,
    forecast_start_period: int | None = None,
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
    if forecast_start_period is not None and not (0 <= forecast_start_period <= periods):
        raise ValueError("forecast_start_period must be within the period range.")

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
            is_forecast = (
                forecast_start_period is not None and period >= forecast_start_period
            )
            ds = (base_date + timedelta(days=period * frequency_days)).isoformat()
            forecast = sample_int(forecast_mean, forecast_std)
            actuals = None if is_forecast else sample_int(history_mean, history_std)
            demand = forecast if is_forecast else int(actuals)
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
                    is_forecast=is_forecast,
                )
            )
    return rows


def standard_simulation_rows_to_dicts(
    rows: Iterable[StandardSimulationRow],
    *,
    forecast_prefix: str = "forecast_",
) -> list[dict[str, str | int | float | bool]]:
    """Convert standard rows into dictionaries suitable for DataFrame or CSV use."""
    serialized: list[dict[str, str | int | float]] = []
    for row in rows:
        entry: dict[str, str | int | float | bool | None] = {
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
            "is_forecast": row.is_forecast,
        }
        for label, value in row.forecast_percentiles.items():
            entry[f"{forecast_prefix}{label}"] = value
        serialized.append(entry)
    return serialized


def standard_simulation_rows_to_dataframe(
    rows: Iterable[StandardSimulationRow],
    *,
    library: str = "pandas",
    forecast_prefix: str = "forecast_",
    include_demand: bool = False,
):
    """Convert standard rows into a pandas or polars DataFrame."""
    data = standard_simulation_rows_to_dicts(rows, forecast_prefix=forecast_prefix)
    if not include_demand:
        for entry in data:
            entry.pop("demand", None)
    if library == "pandas":
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required for standard_simulation_rows_to_dataframe(library='pandas')."
            ) from exc
        return pd.DataFrame(data)
    if library == "polars":
        try:
            import polars as pl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "polars is required for standard_simulation_rows_to_dataframe(library='polars')."
            ) from exc
        return pl.DataFrame(data)
    raise ValueError("library must be 'pandas' or 'polars'.")


def replenishment_decision_rows_to_dicts(
    rows: Iterable[ReplenishmentDecisionRow],
) -> list[dict[str, str | int]]:
    return [
        {"unique_id": row.unique_id, "ds": row.ds, "quantity": row.quantity}
        for row in rows
    ]


def replenishment_decision_rows_to_dataframe(
    rows: Iterable[ReplenishmentDecisionRow],
    *,
    library: str = "pandas",
):
    data = replenishment_decision_rows_to_dicts(rows)
    if library == "pandas":
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required for replenishment_decision_rows_to_dataframe(library='pandas')."
            ) from exc
        return pd.DataFrame(data)
    if library == "polars":
        try:
            import polars as pl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "polars is required for replenishment_decision_rows_to_dataframe(library='polars')."
            ) from exc
        return pl.DataFrame(data)
    raise ValueError("library must be 'pandas' or 'polars'.")


def build_replenishment_decisions_from_simulations(
    rows,
    simulations: Mapping[str, SimulationResult],
    *,
    aggregation_window: Mapping[str, int] | int = 1,
) -> list[ReplenishmentDecisionRow]:
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows)
    grouped = _group_standard_rows(rows)
    decisions: list[ReplenishmentDecisionRow] = []
    for unique_id, simulation in simulations.items():
        if unique_id not in grouped:
            raise ValueError(f"Missing standard rows for unique_id '{unique_id}'.")
        ordered = _order_rows_by_ds(unique_id, grouped[unique_id])
        window = _resolve_value(aggregation_window, unique_id, "aggregation_window")
        if not isinstance(window, int):
            raise TypeError("aggregation_window must be an int or mapping of ints.")
        if window <= 0:
            raise ValueError("aggregation_window must be positive.")
        ds_values = [row.ds for row in ordered]
        max_index = (len(simulation.snapshots) - 1) * window
        if max_index >= len(ds_values):
            raise ValueError(
                f"Aggregation window {window} is incompatible with ds length for unique_id '{unique_id}'."
            )
        for index, snapshot in enumerate(simulation.snapshots):
            ds = ds_values[index * window]
            decisions.append(
                ReplenishmentDecisionRow(
                    unique_id=unique_id,
                    ds=ds,
                    quantity=snapshot.order_placed,
                )
            )
    return decisions


def build_replenishment_decisions_from_optimization_results(
    rows,
    optimization_results: Mapping[str, object],
    *,
    aggregation_window: Mapping[str, int] | int | None = None,
) -> list[ReplenishmentDecisionRow]:
    simulations: dict[str, SimulationResult] = {}
    windows: dict[str, int] = {}
    for unique_id, result in optimization_results.items():
        if not hasattr(result, "simulation"):
            raise TypeError("Optimization results must include a simulation attribute.")
        simulation = getattr(result, "simulation")
        if not isinstance(simulation, SimulationResult):
            raise TypeError("Optimization results must include a SimulationResult.")
        simulations[unique_id] = simulation
        if hasattr(result, "window"):
            window = getattr(result, "window")
            if not isinstance(window, int):
                raise TypeError("Optimization result window must be an int.")
            windows[unique_id] = window

    window_override: dict[str, int] = {}
    if aggregation_window is None:
        window_override = {}
    elif isinstance(aggregation_window, Mapping):
        window_override = dict(aggregation_window)
    elif isinstance(aggregation_window, int):
        window_override = {unique_id: aggregation_window for unique_id in simulations}
    else:
        raise TypeError("aggregation_window must be an int or mapping of ints.")

    for unique_id, window in window_override.items():
        if unique_id in windows and windows[unique_id] != window:
            raise ValueError(
                f"Aggregation window mismatch for unique_id '{unique_id}':"
                f" {windows[unique_id]} vs {window}."
            )
        windows.setdefault(unique_id, window)

    missing = [unique_id for unique_id in simulations if unique_id not in windows]
    if missing:
        if aggregation_window is None:
            for unique_id in missing:
                windows[unique_id] = 1
        else:
            missing_str = "', '".join(sorted(missing))
            raise ValueError(
                "Aggregation window missing for unique_id(s):"
                f" '{missing_str}'."
            )

    return build_replenishment_decisions_from_simulations(
        rows,
        simulations,
        aggregation_window=windows,
    )


def standard_simulation_rows_from_dataframe(
    df,
    *,
    unique_id_field: str = "unique_id",
    ds_field: str = "ds",
    demand_field: str = "demand",
    history_field: str = "history",
    forecast_field: str = "forecast",
    actuals_field: str = "actuals",
    holding_cost_per_unit_field: str = "holding_cost_per_unit",
    stockout_cost_per_unit_field: str = "stockout_cost_per_unit",
    order_cost_per_order_field: str = "order_cost_per_order",
    lead_time_field: str = "lead_time",
    initial_on_hand_field: str = "initial_on_hand",
    initial_demand_field: str = "initial_demand",
    current_stock_field: str = "current_stock",
    is_forecast_field: str = "is_forecast",
    period_field: str = "period",
    cutoff: int | str | date | datetime | None = None,
    forecast_prefix: str = "forecast_",
) -> list[StandardSimulationRow]:
    """Convert a pandas or polars DataFrame into standard simulation rows."""
    rows = _rows_from_dataframe(df)
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    has_demand = demand_field in fieldnames
    has_history = history_field in fieldnames
    has_actuals = actuals_field in fieldnames
    required = [
        unique_id_field,
        ds_field,
        forecast_field,
        holding_cost_per_unit_field,
        stockout_cost_per_unit_field,
        order_cost_per_order_field,
        lead_time_field,
        current_stock_field,
    ]
    _validate_required_columns(
        fieldnames,
        required_fields=required,
        require_any_of=[initial_on_hand_field, initial_demand_field],
        context="standard simulation DataFrame",
    )
    if not has_demand and not has_history and not has_actuals:
        raise ValueError(
            "DataFrame must include demand, history, or actuals columns to build simulation rows."
        )
    if not has_actuals and not has_history:
        raise ValueError(
            "DataFrame must include actuals or history columns to build simulation rows."
        )

    parsed_rows: list[StandardSimulationRow] = []
    for row in rows:
        forecast_percentiles = {
            key[len(forecast_prefix) :]: int(value)
            for key, value in row.items()
            if key.startswith(forecast_prefix)
            and key != forecast_field
            and not _is_missing(value)
        }
        initial_on_hand = _coalesce_value(
            row, initial_on_hand_field, initial_demand_field
        )
        if initial_on_hand is None:
            raise ValueError(
                "Initial on-hand inventory is required (initial_on_hand or initial_demand)."
            )
        current_stock_value = _coalesce_value(row, current_stock_field)
        if current_stock_value is None:
            current_stock_value = initial_on_hand
        is_forecast_value = _derive_is_forecast(
            row,
            ds_field=ds_field,
            is_forecast_field=is_forecast_field,
            period_field=period_field,
            actuals_field=actuals_field,
            history_field=history_field,
            cutoff=cutoff,
        )
        demand_value = _coalesce_value(row, demand_field, history_field)
        if demand_value is None:
            demand_value = _coalesce_value(row, actuals_field)
        if demand_value is None:
            demand_value = _coalesce_value(row, forecast_field)
        if demand_value is None:
            raise ValueError("Demand values are required for all periods.")
        actuals_value = _coalesce_value(row, actuals_field, history_field)
        if actuals_value is None and not is_forecast_value:
            raise ValueError("Actuals values are required for backtest periods.")
        parsed_rows.append(
            StandardSimulationRow(
                unique_id=str(row[unique_id_field]),
                ds=_normalize_ds(row[ds_field]),
                demand=int(demand_value),
                forecast=int(row[forecast_field]),
                actuals=actuals_value,
                holding_cost_per_unit=float(row[holding_cost_per_unit_field]),
                stockout_cost_per_unit=float(row[stockout_cost_per_unit_field]),
                order_cost_per_order=float(row[order_cost_per_order_field]),
                lead_time=int(row[lead_time_field]),
                initial_on_hand=int(initial_on_hand),
                current_stock=int(current_stock_value),
                forecast_percentiles=forecast_percentiles,
                is_forecast=is_forecast_value,
            )
        )
    return parsed_rows


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
        "is_forecast",
    ] + [f"{forecast_prefix}{label}" for label in percentile_labels]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in standard_simulation_rows_to_dicts(
            rows_list, forecast_prefix=forecast_prefix
        ):
            if _is_missing(entry.get("actuals")):
                entry["actuals"] = ""
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
    is_forecast_field: str = "is_forecast",
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
            is_forecast_value = False
            if is_forecast_field in row and row[is_forecast_field] != "":
                is_forecast_value = _parse_bool(
                    row[is_forecast_field], field=is_forecast_field
                )
            actuals_value = _parse_optional_int(row.get(actuals_field, ""))
            if actuals_value is None and not is_forecast_value:
                raise ValueError("Actuals values are required for backtest periods.")
            yield StandardSimulationRow(
                unique_id=row[unique_id_field],
                ds=row[ds_field],
                demand=int(row[demand_field]),
                forecast=int(row[forecast_field]),
                actuals=actuals_value,
                holding_cost_per_unit=float(row[holding_cost_per_unit_field]),
                stockout_cost_per_unit=float(row[stockout_cost_per_unit_field]),
                order_cost_per_order=float(row[order_cost_per_order_field]),
                lead_time=int(row[lead_time_field]),
                initial_on_hand=initial_on_hand,
                current_stock=current_stock_value,
                forecast_percentiles=forecast_percentiles,
                is_forecast=is_forecast_value,
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
        actuals = _trim_actuals_series(unique_id, ordered)
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


def split_standard_simulation_rows(
    rows,
    cutoff: int | str | date | datetime | None = None,
) -> tuple[list[StandardSimulationRow], list[StandardSimulationRow]]:
    """Split standard rows into backtest and forecast partitions."""
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows, cutoff=cutoff)
    backtest_rows: list[StandardSimulationRow] = []
    forecast_rows: list[StandardSimulationRow] = []
    for row in rows:
        if row.is_forecast:
            forecast_rows.append(row)
        else:
            backtest_rows.append(row)
    return backtest_rows, forecast_rows


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


def _coalesce_value(row: Mapping[str, object], *fields: str) -> int | float | None:
    for field in fields:
        if field in row and not _is_missing(row[field]):
            return row[field]  # type: ignore[return-value]
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


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value for {field}: {value!r}.")


def _parse_optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _rows_from_dataframe(df) -> list[dict[str, object]]:
    if hasattr(df, "to_dicts"):
        return df.to_dicts()  # type: ignore[no-any-return]
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")  # type: ignore[no-any-return]
    raise TypeError("df must be a pandas or polars DataFrame.")


def _normalize_ds(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _derive_is_forecast(
    row: Mapping[str, object],
    *,
    ds_field: str,
    is_forecast_field: str,
    period_field: str,
    actuals_field: str,
    history_field: str,
    cutoff: int | str | date | datetime | None,
) -> bool:
    if is_forecast_field in row and not _is_missing(row[is_forecast_field]):
        value = row[is_forecast_field]
        if isinstance(value, bool):
            return value
        return _parse_bool(str(value), field=is_forecast_field)
    if cutoff is None:
        return _is_missing(row.get(actuals_field)) and _is_missing(row.get(history_field))
    if isinstance(cutoff, int):
        if period_field not in row or _is_missing(row[period_field]):
            raise ValueError(
                "period field is required when using an integer cutoff for DataFrame inputs."
            )
        return int(row[period_field]) > cutoff
    ds_value = row.get(ds_field)
    if ds_value is None:
        return False
    if isinstance(ds_value, (date, datetime)):
        cutoff_value = cutoff
        if isinstance(cutoff, str):
            cutoff_value = _try_parse_date(cutoff)
        if isinstance(cutoff_value, datetime):
            cutoff_value = cutoff_value.date()
        if isinstance(cutoff_value, date):
            if isinstance(ds_value, datetime):
                ds_value = ds_value.date()
            return ds_value > cutoff_value
        return str(ds_value) > str(cutoff)
    return str(ds_value) > str(cutoff)


def _try_parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _trim_actuals_series(
    unique_id: str, rows: Iterable[StandardSimulationRow]
) -> list[int]:
    trimmed: list[int] = []
    seen_missing = False
    for row in rows:
        actuals_value = row.actuals
        if _is_missing(actuals_value):
            seen_missing = True
            continue
        if seen_missing:
            raise ValueError(
                f"Actuals must be present for all backtest periods for unique_id '{unique_id}'."
            )
        trimmed.append(int(actuals_value))
    return trimmed
