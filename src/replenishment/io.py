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

from .aggregation import aggregate_lead_time, aggregate_series
from .optimization import (
    AggregationForecastTargetOptimizationResult,
    AggregationServiceLevelOptimizationResult,
    AggregationWindowOptimizationResult,
    ForecastCandidatesConfig,
    PercentileForecastOptimizationResult,
    PointForecastOptimizationResult,
    optimize_service_level_factors,
)
from .policies import (
    LeadTimeForecastOptimizationPolicy,
    PointForecastOptimizationPolicy,
    RopPointForecastOptimizationPolicy,
)
from .service_levels import fill_rate_z, normalize_service_level_mode, service_level_multiplier
from .simulation import (
    ArticleSimulationConfig,
    SimulationResult,
    simulate_replenishment_for_articles,
)


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
    demand: int | None = None
    forecast_quantity: float | None = None
    forecast_quantity_lead_time: float | None = None
    reorder_point: float | None = None
    order_up_to: float | None = None
    incoming_stock: int | None = None
    starting_stock: int | None = None
    ending_stock: int | None = None
    safety_stock: float | None = None
    starting_on_hand: int | None = None
    ending_on_hand: int | None = None
    current_stock: int | None = None
    on_order: int | None = None
    backorders: int | None = None
    missed_sales: int | None = None
    sigma: float | None = None
    service_level_mode: str | None = None
    aggregation_window: int | None = None
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    percentile_target: float | str | None = None


@dataclass(frozen=True)
class ReplenishmentDecisionMetadata:
    sigma: float | None = None
    service_level_mode: str | None = None
    aggregation_window: int | None = None
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    percentile_target: float | str | None = None


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
) -> list[dict[str, str | int | float | None]]:
    return [
        {
            "unique_id": row.unique_id,
            "ds": row.ds,
            "quantity": row.quantity,
            "demand": row.demand,
            "forecast_quantity": row.forecast_quantity,
            "forecast_quantity_lead_time": row.forecast_quantity_lead_time,
            "reorder_point": row.reorder_point,
            "order_up_to": row.order_up_to,
            "incoming_stock": row.incoming_stock,
            "starting_stock": row.starting_stock,
            "ending_stock": row.ending_stock,
            "safety_stock": row.safety_stock,
            "starting_on_hand": row.starting_on_hand,
            "ending_on_hand": row.ending_on_hand,
            "current_stock": row.current_stock,
            "on_order": row.on_order,
            "backorders": row.backorders,
            "missed_sales": row.missed_sales,
            "sigma": row.sigma,
            "aggregation_window": row.aggregation_window,
            "review_period": row.review_period,
            "forecast_horizon": row.forecast_horizon,
            "rmse_window": row.rmse_window,
            "percentile_target": row.percentile_target,
        }
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
    aggregation_window: Mapping[str, int] | int | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    sigma: Mapping[str, float] | float | None = None,
    fixed_rmse: Mapping[str, float] | float | None = None,
    service_level_mode: Mapping[str, str] | str | None = None,
    percentile_target: Mapping[str, float | str] | float | str | None = None,
    decision_metadata: Mapping[str, ReplenishmentDecisionMetadata] | None = None,
) -> list[ReplenishmentDecisionRow]:
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows)
    grouped = _group_standard_rows(rows)
    decisions: list[ReplenishmentDecisionRow] = []
    metadata_lookup = decision_metadata or {}
    for unique_id, simulation in simulations.items():
        if unique_id not in grouped:
            raise ValueError(f"Missing standard rows for unique_id '{unique_id}'.")
        ordered = _order_rows_by_ds(unique_id, grouped[unique_id])
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        review_value: int | None = None
        if review_period is not None:
            if isinstance(review_period, Mapping):
                review_value = _resolve_value(
                    review_period, unique_id, "review_period"
                )
            elif isinstance(review_period, int):
                review_value = review_period
            else:
                raise TypeError("review_period must be an int or mapping of ints.")
        elif aggregation_window is not None:
            if isinstance(aggregation_window, Mapping):
                review_value = _resolve_value(
                    aggregation_window, unique_id, "aggregation_window"
                )
            elif isinstance(aggregation_window, int):
                review_value = aggregation_window
            else:
                raise TypeError("aggregation_window must be an int or mapping of ints.")
        else:
            if simulation.metadata is not None and simulation.metadata.review_period is not None:
                review_value = simulation.metadata.review_period
            elif simulation.metadata is not None and simulation.metadata.aggregation_window is not None:
                review_value = simulation.metadata.aggregation_window
            else:
                review_value = 1
        if review_value is None or not isinstance(review_value, int):
            raise TypeError("review_period must be an int or mapping of ints.")
        if review_value <= 0:
            raise ValueError("review_period must be positive.")
        window = review_value
        rmse_window_value: int | None = None
        if rmse_window is not None:
            if isinstance(rmse_window, Mapping):
                rmse_window_value = _resolve_value(
                    rmse_window, unique_id, "rmse_window"
                )
            elif isinstance(rmse_window, int):
                rmse_window_value = rmse_window
            else:
                raise TypeError("rmse_window must be an int or mapping of ints.")
        elif simulation.metadata is not None and simulation.metadata.rmse_window is not None:
            rmse_window_value = simulation.metadata.rmse_window
        else:
            rmse_window_value = window
        if rmse_window_value is None or rmse_window_value <= 0:
            raise ValueError("rmse_window must be positive.")
        forecast_horizon_value: int | None = None
        if forecast_horizon is not None:
            if isinstance(forecast_horizon, Mapping):
                forecast_horizon_value = _resolve_value(
                    forecast_horizon, unique_id, "forecast_horizon"
                )
            elif isinstance(forecast_horizon, int):
                forecast_horizon_value = forecast_horizon
            else:
                raise TypeError("forecast_horizon must be an int or mapping of ints.")
        elif simulation.metadata is not None and simulation.metadata.forecast_horizon is not None:
            forecast_horizon_value = simulation.metadata.forecast_horizon
        else:
            forecast_horizon_value = window
        if forecast_horizon_value is None or forecast_horizon_value <= 0:
            raise ValueError("forecast_horizon must be positive.")
        ds_values = [row.ds for row in ordered]
        snapshot_count = len(simulation.snapshots)
        daily_snapshots = snapshot_count == len(ds_values)
        snapshot_window = 1 if daily_snapshots else window
        max_index = (snapshot_count - 1) * snapshot_window
        if not daily_snapshots and max_index >= len(ds_values):
            raise ValueError(
                f"Aggregation window {window} is incompatible with ds length for unique_id '{unique_id}'."
            )
        sigma_value = _resolve_optional_value(sigma, unique_id, "sigma")
        if sigma_value is None and simulation.metadata is not None:
            sigma_value = simulation.metadata.service_level_factor
        fixed_rmse_value = _resolve_optional_value(
            fixed_rmse, unique_id, "fixed_rmse"
        )
        mode_value = _resolve_optional_value(
            service_level_mode, unique_id, "service_level_mode"
        )
        if mode_value is not None and not isinstance(mode_value, str):
            raise TypeError(
                "service_level_mode must be a string or mapping of strings."
            )
        if mode_value is None and simulation.metadata is not None:
            mode_value = simulation.metadata.service_level_mode
        target_value = _resolve_optional_value(
            percentile_target, unique_id, "percentile_target"
        )
        if target_value is None and simulation.metadata is not None:
            target_value = simulation.metadata.percentile_target
        forecast_values: list[int] | None = None
        if target_value is None or target_value == "mean":
            forecast_values = [row.forecast for row in ordered]
        else:
            if all(
                target_value in row.forecast_percentiles for row in ordered
            ):
                forecast_values = [
                    row.forecast_percentiles[target_value] for row in ordered
                ]
        forecast_length = len(forecast_values) if forecast_values is not None else 0
        def _sum_with_extension(
            values: list[int], start_index: int, horizon: int
        ) -> float:
            if horizon <= 0 or not values:
                return 0.0
            if start_index >= len(values):
                return float(values[-1]) * horizon
            end_index = start_index + horizon
            if end_index <= len(values):
                return float(sum(values[start_index:end_index]))
            total = float(sum(values[start_index:]))
            extra = end_index - len(values)
            if extra > 0:
                total += float(values[-1]) * extra
            return total

        rop_forecast_values: list[int] | None = None
        rop_lead_time = lead_time
        rop_cycle_horizon = forecast_horizon_value
        if forecast_values is not None:
            if not daily_snapshots and window > 1:
                rop_forecast_values = aggregate_series(
                    forecast_values,
                    periods=len(forecast_values),
                    window=window,
                    extend_last=True,
                )
                rop_lead_time = aggregate_lead_time(lead_time, window)
                rop_cycle_horizon = max(
                    1, math.ceil(forecast_horizon_value / window)
                )
            else:
                rop_forecast_values = forecast_values
        metadata = metadata_lookup.get(unique_id)
        if metadata is None:
            metadata = ReplenishmentDecisionMetadata(
                sigma=sigma_value,
                service_level_mode=mode_value,
                aggregation_window=window,
                review_period=window,
                forecast_horizon=forecast_horizon_value,
                rmse_window=rmse_window_value,
                percentile_target=target_value,
            )
        else:
            metadata = ReplenishmentDecisionMetadata(
                sigma=metadata.sigma if metadata.sigma is not None else sigma_value,
                service_level_mode=(
                    metadata.service_level_mode
                    if metadata.service_level_mode is not None
                    else mode_value
                ),
                aggregation_window=(
                    metadata.aggregation_window
                    if metadata.aggregation_window is not None
                    else window
                ),
                review_period=(
                    metadata.review_period
                    if metadata.review_period is not None
                    else window
                ),
                forecast_horizon=(
                    metadata.forecast_horizon
                    if metadata.forecast_horizon is not None
                    else forecast_horizon_value
                ),
                rmse_window=(
                    metadata.rmse_window
                    if metadata.rmse_window is not None
                    else rmse_window_value
                ),
                percentile_target=(
                    metadata.percentile_target
                    if metadata.percentile_target is not None
                    else target_value
                ),
            )
        safety_stock_values: list[float] | None = None
        if sigma_value is not None:
            forecast_series = [row.forecast for row in ordered]
            actuals_series = _trim_actuals_series(unique_id, ordered)
            safety_stock_values = _safety_stock_by_period(
                forecast_values=forecast_series,
                actuals_values=actuals_series,
                sigma=float(sigma_value),
                service_level_mode=mode_value,
                fixed_rmse=fixed_rmse_value,
                lead_time=lead_time,
                rmse_window=rmse_window_value,
                review_period=window,
                forecast_horizon=forecast_horizon_value,
                periods=len(simulation.snapshots),
            )
        running_stock: float | None = None
        for index, snapshot in enumerate(simulation.snapshots):
            ds = ds_values[index * snapshot_window]
            start = index * snapshot_window
            forecast_quantity = None
            forecast_quantity_lead_time = None
            total_horizon = lead_time + forecast_horizon_value
            if forecast_values is not None:
                if start >= forecast_length or forecast_length == 0:
                    forecast_quantity = None
                    forecast_quantity_lead_time = None
                else:
                    if daily_snapshots and window > 1:
                        cycle_end = min(start + window, forecast_length)
                        forecast_quantity = sum(
                            forecast_values[start:cycle_end]
                        )
                        horizon = total_horizon if total_horizon > 0 else 1
                        lead_end = min(start + horizon, forecast_length)
                        forecast_quantity_lead_time = sum(
                            forecast_values[start:lead_end]
                        )
                        if window > 0:
                            forecast_quantity_lead_time = (
                                forecast_quantity_lead_time / window
                            )
                            forecast_quantity = forecast_quantity / window
                    elif daily_snapshots:
                        forecast_quantity = forecast_values[start]
                        start_period = start + 1
                        horizon = max(1, total_horizon)
                        if start_period >= forecast_length:
                            forecast_quantity_lead_time = (
                                forecast_values[-1] * horizon
                                if forecast_values
                                else None
                            )
                        else:
                            end_period = start_period + horizon
                            if end_period <= forecast_length:
                                forecast_quantity_lead_time = sum(
                                    forecast_values[start_period:end_period]
                                )
                            else:
                                forecast_quantity_lead_time = sum(
                                    forecast_values[start_period:forecast_length]
                                )
                                extra = end_period - forecast_length
                                if extra > 0 and forecast_values:
                                    forecast_quantity_lead_time += (
                                        forecast_values[-1] * extra
                                    )
                    else:
                        end = min(start + snapshot_window, forecast_length)
                        forecast_quantity = sum(
                            forecast_values[start:end]
                        )
                        forecast_quantity_lead_time = forecast_quantity
                        if window > 0:
                            forecast_quantity_lead_time = (
                                forecast_quantity_lead_time / window
                            )
                            forecast_quantity = forecast_quantity / window
            if running_stock is None:
                running_stock = float(ordered[0].current_stock)
            stock_before = running_stock + float(snapshot.received)
            starting_stock = int(round(stock_before))
            current_stock = None
            missed_sales = None
            stock_after = stock_before - float(snapshot.demand)
            if stock_after < 0:
                missed_sales = int(round(-stock_after))
                stock_after = 0.0
            else:
                missed_sales = 0
            ending_stock = int(round(stock_after))
            current_stock = ending_stock
            running_stock = stock_after
            safety_stock = (
                safety_stock_values[index] if safety_stock_values else None
            )
            reorder_point = None
            order_up_to = None
            if rop_forecast_values is not None:
                rop_index = (
                    index if (not daily_snapshots and window > 1) else start
                )
                lead_horizon = max(0, rop_lead_time)
                lead_demand = _sum_with_extension(
                    rop_forecast_values, rop_index, lead_horizon
                )
                reorder_point = lead_demand
                if safety_stock is not None:
                    reorder_point += safety_stock
                if reorder_point is not None:
                    if not daily_snapshots and window > 1:
                        cycle_horizon = 1
                        cycle_index = index
                    else:
                        cycle_horizon = max(1, rop_cycle_horizon)
                        cycle_index = start
                    cycle_stock = _sum_with_extension(
                        rop_forecast_values, cycle_index, cycle_horizon
                    )
                    order_up_to = reorder_point + cycle_stock
            decisions.append(
                ReplenishmentDecisionRow(
                    unique_id=unique_id,
                    ds=ds,
                    quantity=snapshot.order_placed,
                    demand=snapshot.demand,
                    forecast_quantity=forecast_quantity,
                    forecast_quantity_lead_time=forecast_quantity_lead_time,
                    reorder_point=reorder_point,
                    order_up_to=order_up_to,
                    incoming_stock=snapshot.received,
                    starting_stock=starting_stock,
                    ending_stock=ending_stock,
                    safety_stock=safety_stock,
                    starting_on_hand=snapshot.starting_on_hand,
                    ending_on_hand=snapshot.ending_on_hand,
                    current_stock=current_stock,
                    on_order=snapshot.on_order,
                    backorders=snapshot.backorders,
                    missed_sales=missed_sales,
                    sigma=metadata.sigma,
                    service_level_mode=metadata.service_level_mode,
                    aggregation_window=metadata.aggregation_window,
                    review_period=metadata.review_period,
                    forecast_horizon=metadata.forecast_horizon,
                    rmse_window=metadata.rmse_window,
                    percentile_target=metadata.percentile_target,
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
    metadata: dict[str, ReplenishmentDecisionMetadata] = {}
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
        metadata[unique_id] = _decision_metadata_from_result(result)

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
        decision_metadata=metadata,
    )


def _decision_metadata_from_result(
    result: object,
) -> ReplenishmentDecisionMetadata:
    if isinstance(result, AggregationServiceLevelOptimizationResult):
        return ReplenishmentDecisionMetadata(
            sigma=result.service_level_factor,
            aggregation_window=result.window,
            review_period=result.window,
        )
    if isinstance(result, AggregationForecastTargetOptimizationResult):
        return ReplenishmentDecisionMetadata(
            aggregation_window=result.window,
            review_period=result.window,
            percentile_target=result.target,
        )
    if isinstance(result, AggregationWindowOptimizationResult):
        return ReplenishmentDecisionMetadata(
            aggregation_window=result.window,
            review_period=result.window,
        )
    if isinstance(result, PointForecastOptimizationResult):
        return ReplenishmentDecisionMetadata(
            sigma=result.service_level_factor,
        )
    if isinstance(result, PercentileForecastOptimizationResult):
        return ReplenishmentDecisionMetadata(
            percentile_target=result.target,
        )
    return ReplenishmentDecisionMetadata()


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
    service_level_mode: Mapping[str, str] | str | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    policy_mode: str = "base_stock",
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
        mode_value = _resolve_optional_value(
            service_level_mode, unique_id, "service_level_mode"
        )
        if mode_value is not None and not isinstance(mode_value, str):
            raise TypeError(
                "service_level_mode must be a string or mapping of strings."
            )
        if policy_mode == "rop":
            policy = RopPointForecastOptimizationPolicy(
                forecast=forecast,
                actuals=actuals,
                lead_time=_resolve_value(lead_time, unique_id, "lead_time"),
                review_period=_resolve_optional_value(
                    review_period, unique_id, "review_period"
                ),
                forecast_horizon=_resolve_optional_value(
                    forecast_horizon, unique_id, "forecast_horizon"
                ),
                rmse_window=_resolve_optional_value(
                    rmse_window, unique_id, "rmse_window"
                ),
                service_level_factor=_resolve_value(
                    service_level_factor, unique_id, "service_level_factor"
                ),
                service_level_mode=mode_value if mode_value is not None else "factor",
            )
        elif policy_mode == "base_stock":
            policy = PointForecastOptimizationPolicy(
                forecast=forecast,
                actuals=actuals,
                lead_time=_resolve_value(lead_time, unique_id, "lead_time"),
                review_period=_resolve_optional_value(
                    review_period, unique_id, "review_period"
                ),
                forecast_horizon=_resolve_optional_value(
                    forecast_horizon, unique_id, "forecast_horizon"
                ),
                rmse_window=_resolve_optional_value(
                    rmse_window, unique_id, "rmse_window"
                ),
                service_level_factor=_resolve_value(
                    service_level_factor, unique_id, "service_level_factor"
                ),
                service_level_mode=mode_value if mode_value is not None else "factor",
            )
        else:
            raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
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
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
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
            review_period=_resolve_optional_value(
                review_period, unique_id, "review_period"
            )
            or 1,
            forecast_horizon=_resolve_optional_value(
                forecast_horizon, unique_id, "forecast_horizon"
            ),
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
    service_level_mode: Mapping[str, str] | str | None = None,
    fixed_rmse: Mapping[str, float] | float | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    use_current_stock: bool | None = None,
    actuals_override: Mapping[str, Iterable[int]] | None = None,
    policy_mode: str = "base_stock",
) -> dict[str, ArticleSimulationConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        forecast = [row.forecast for row in ordered]
        if actuals_override is None:
            actuals = _trim_actuals_series(unique_id, ordered)
        else:
            if unique_id not in actuals_override:
                raise ValueError(
                    f"Missing actuals override for unique_id '{unique_id}'."
                )
            actuals = list(actuals_override[unique_id])
        mode_value = _resolve_optional_value(
            service_level_mode, unique_id, "service_level_mode"
        )
        if mode_value is not None and not isinstance(mode_value, str):
            raise TypeError(
                "service_level_mode must be a string or mapping of strings."
            )
        if policy_mode == "rop":
            policy = RopPointForecastOptimizationPolicy(
                forecast=forecast,
                actuals=actuals,
                lead_time=lead_time,
                review_period=_resolve_optional_value(
                    review_period, unique_id, "review_period"
                ),
                forecast_horizon=_resolve_optional_value(
                    forecast_horizon, unique_id, "forecast_horizon"
                ),
                rmse_window=_resolve_optional_value(
                    rmse_window, unique_id, "rmse_window"
                ),
                service_level_factor=_resolve_value(
                    service_level_factor, unique_id, "service_level_factor"
                ),
                service_level_mode=mode_value if mode_value is not None else "factor",
                fixed_rmse=_resolve_optional_value(
                    fixed_rmse, unique_id, "fixed_rmse"
                ),
            )
        elif policy_mode == "base_stock":
            policy = PointForecastOptimizationPolicy(
                forecast=forecast,
                actuals=actuals,
                lead_time=lead_time,
                review_period=_resolve_optional_value(
                    review_period, unique_id, "review_period"
                ),
                forecast_horizon=_resolve_optional_value(
                    forecast_horizon, unique_id, "forecast_horizon"
                ),
                rmse_window=_resolve_optional_value(
                    rmse_window, unique_id, "rmse_window"
                ),
                service_level_factor=_resolve_value(
                    service_level_factor, unique_id, "service_level_factor"
                ),
                service_level_mode=mode_value if mode_value is not None else "factor",
                fixed_rmse=_resolve_optional_value(
                    fixed_rmse, unique_id, "fixed_rmse"
                ),
            )
        else:
            raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
        configs[unique_id] = ArticleSimulationConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            policy=policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


def build_lead_time_forecast_article_configs_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    service_level_factor: Mapping[str, float] | float,
    service_level_mode: Mapping[str, str] | str | None = None,
    fixed_rmse: Mapping[str, float] | float | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    use_current_stock: bool | None = None,
    actuals_override: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, ArticleSimulationConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        forecast = [row.forecast for row in ordered]
        if actuals_override is None:
            actuals = _trim_actuals_series(unique_id, ordered)
        else:
            if unique_id not in actuals_override:
                raise ValueError(
                    f"Missing actuals override for unique_id '{unique_id}'."
                )
            actuals = list(actuals_override[unique_id])
        mode_value = _resolve_optional_value(
            service_level_mode, unique_id, "service_level_mode"
        )
        if mode_value is not None and not isinstance(mode_value, str):
            raise TypeError(
                "service_level_mode must be a string or mapping of strings."
            )
        policy = LeadTimeForecastOptimizationPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=lead_time,
            review_period=_resolve_optional_value(
                review_period, unique_id, "review_period"
            ),
            forecast_horizon=_resolve_optional_value(
                forecast_horizon, unique_id, "forecast_horizon"
            ),
            rmse_window=_resolve_optional_value(
                rmse_window, unique_id, "rmse_window"
            ),
            service_level_factor=_resolve_value(
                service_level_factor, unique_id, "service_level_factor"
            ),
            service_level_mode=mode_value if mode_value is not None else "factor",
            fixed_rmse=_resolve_optional_value(
                fixed_rmse, unique_id, "fixed_rmse"
            ),
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            policy=policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


def optimize_point_forecast_policy_and_simulate_actuals(
    backtest_rows: Iterable[StandardSimulationRow],
    evaluation_rows: Iterable[StandardSimulationRow],
    candidate_factors: Iterable[float],
    *,
    use_current_stock: bool | None = None,
    service_level_mode: str | None = None,
) -> tuple[
    dict[str, PointForecastOptimizationResult],
    dict[str, SimulationResult],
    list[ReplenishmentDecisionRow],
]:
    """Optimize safety-stock factors on backtest rows, then evaluate on actuals."""
    backtest_configs = build_point_forecast_article_configs_from_standard_rows(
        backtest_rows,
        service_level_factor=1.0,
    )
    optimized = optimize_service_level_factors(
        backtest_configs,
        candidate_factors=candidate_factors,
        service_level_mode=service_level_mode,
    )
    backtest_actuals = _actuals_by_article(backtest_rows)
    eval_factors = {
        unique_id: result.service_level_factor
        for unique_id, result in optimized.items()
    }
    eval_configs = build_point_forecast_article_configs_from_standard_rows(
        evaluation_rows,
        service_level_factor=eval_factors,
        service_level_mode=service_level_mode,
        use_current_stock=use_current_stock,
        actuals_override=backtest_actuals,
    )
    eval_simulations = simulate_replenishment_for_articles(eval_configs)
    eval_decisions = build_replenishment_decisions_from_simulations(
        evaluation_rows,
        eval_simulations,
        sigma=eval_factors,
        service_level_mode=service_level_mode,
    )
    return optimized, eval_simulations, eval_decisions


def build_percentile_forecast_candidates_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    include_mean: bool = False,
    use_current_stock: bool | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
) -> dict[str, ForecastCandidatesConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ForecastCandidatesConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        targets = _validate_percentile_targets(unique_id, ordered)
        candidate_series = {
            target: [row.forecast_percentiles[target] for row in ordered]
            for target in targets
        }
        if include_mean and "mean" not in candidate_series:
            candidate_series["mean"] = [row.forecast for row in ordered]
        configs[unique_id] = ForecastCandidatesConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            review_period=_resolve_optional_value(
                review_period, unique_id, "review_period"
            )
            or 1,
            forecast_horizon=_resolve_optional_value(
                forecast_horizon, unique_id, "forecast_horizon"
            ),
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


def _resolve_optional_value(
    value: Mapping[str, int | float | str] | int | float | str | None,
    unique_id: str,
    name: str,
) -> int | float | str | None:
    if value is None:
        return None
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


def _rmse_from_series(actuals: list[int], forecast: list[int]) -> float:
    max_index = min(len(actuals), len(forecast))
    if max_index <= 0:
        return 0.0
    errors = [actuals[index] - forecast[index] for index in range(max_index)]
    if not errors:
        return 0.0
    if len(errors) == 1:
        return abs(errors[0])
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def compute_backtest_rmse_by_article(
    rows: Iterable[StandardSimulationRow],
    *,
    aggregation_window: Mapping[str, int] | int = 1,
    rmse_window: Mapping[str, int] | int | None = None,
) -> dict[str, float]:
    """Compute fixed RMSE per article using backtest rows only."""
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows)
    grouped = _group_standard_rows(rows)
    rmse_by_article: dict[str, float] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        backtest_rows = [row for row in ordered if not row.is_forecast]
        if not backtest_rows:
            raise ValueError(
                f"No backtest rows available for unique_id '{unique_id}'."
            )
        forecast_values = [row.forecast for row in backtest_rows]
        actuals_values = [int(row.actuals) for row in backtest_rows]
        if rmse_window is not None:
            window = _resolve_value(rmse_window, unique_id, "rmse_window")
        else:
            window = _resolve_value(
                aggregation_window, unique_id, "aggregation_window"
            )
        if not isinstance(window, int) or window <= 0:
            raise ValueError("aggregation_window must be a positive int.")
        if window > 1:
            forecast_values = aggregate_series(
                forecast_values,
                periods=len(forecast_values),
                window=window,
                extend_last=True,
            )
            actuals_values = aggregate_series(
                actuals_values,
                periods=len(actuals_values),
                window=window,
                extend_last=False,
            )
        rmse_by_article[unique_id] = _rmse_from_series(
            actuals_values, forecast_values
        )
    return rmse_by_article


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


def _actuals_by_article(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, list[int]]:
    grouped = _group_standard_rows(rows)
    actuals_by_article: dict[str, list[int]] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        actuals_by_article[unique_id] = _trim_actuals_series(unique_id, ordered)
    return actuals_by_article


def _safety_stock_by_period(
    *,
    forecast_values: list[int],
    actuals_values: list[int],
    sigma: float,
    service_level_mode: str | None,
    fixed_rmse: float | None,
    lead_time: int,
    rmse_window: int,
    review_period: int,
    forecast_horizon: int,
    periods: int,
) -> list[float]:
    normalized_mode = normalize_service_level_mode(service_level_mode)
    def _sum_with_extension(
        values: list[int], start_index: int, horizon: int
    ) -> float:
        if horizon <= 0 or not values:
            return 0.0
        if start_index >= len(values):
            return float(values[-1]) * horizon
        end_index = start_index + horizon
        if end_index <= len(values):
            return float(sum(values[start_index:end_index]))
        total = float(sum(values[start_index:]))
        extra = end_index - len(values)
        if extra > 0:
            total += float(values[-1]) * extra
        return total
    use_aggregation = review_period > 1 and periods < len(forecast_values)
    if use_aggregation:
        forecast_values = aggregate_series(
            forecast_values,
            periods=len(forecast_values),
            window=review_period,
            extend_last=True,
        )
        if actuals_values:
            actuals_values = aggregate_series(
                actuals_values,
                periods=len(actuals_values),
                window=review_period,
                extend_last=False,
            )
        lead_time = aggregate_lead_time(lead_time, review_period)
        forecast_horizon = max(1, math.ceil(forecast_horizon / review_period))

    protection_horizon = lead_time + forecast_horizon
    lead_time_factor = math.sqrt(
        protection_horizon if protection_horizon > 0 else 1
    )
    sigma_multiplier = None
    if normalized_mode != "fill_rate":
        sigma_multiplier = service_level_multiplier(sigma, service_level_mode)
    safety_values: list[float] = []
    for period in range(periods):
        if fixed_rmse is not None:
            rmse = float(fixed_rmse)
        else:
            max_index = min(period, len(actuals_values), len(forecast_values))
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
                rmse = _rmse_from_series(actuals_slice, forecast_slice)
        if normalized_mode == "fill_rate":
            horizon = max(1, protection_horizon)
            forecast_qty = _sum_with_extension(
                forecast_values, period, horizon
            )
            std_dev = rmse * lead_time_factor
            if std_dev <= 0 or forecast_qty <= 0:
                safety_values.append(0.0)
            else:
                z_value = fill_rate_z(
                    float(sigma), mean_demand=float(forecast_qty), std_dev=std_dev
                )
                safety_values.append(z_value * std_dev)
        else:
            safety_values.append(sigma_multiplier * rmse * lead_time_factor)
    return safety_values
