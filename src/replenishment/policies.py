"""Inventory control policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import statistics

from .simulation import DemandModel, InventoryState
from .service_levels import (
    fill_rate_z,
    normalize_service_level_mode,
    service_level_multiplier,
)

SAFETY_STOCK_METHOD_SQRT_HORIZON = "sqrt_horizon"
SAFETY_STOCK_METHOD_K_RMSE = "k_rmse"
SAFETY_STOCK_METHOD_K_MAE = "k_mae"

_SAFETY_STOCK_METHOD_ALIASES = {
    SAFETY_STOCK_METHOD_SQRT_HORIZON: SAFETY_STOCK_METHOD_SQRT_HORIZON,
    "legacy": SAFETY_STOCK_METHOD_SQRT_HORIZON,
    "legacy_scaled": SAFETY_STOCK_METHOD_SQRT_HORIZON,
    "scaled_rmse": SAFETY_STOCK_METHOD_SQRT_HORIZON,
    "horizon_scaled": SAFETY_STOCK_METHOD_SQRT_HORIZON,
    SAFETY_STOCK_METHOD_K_RMSE: SAFETY_STOCK_METHOD_K_RMSE,
    "raw_rmse": SAFETY_STOCK_METHOD_K_RMSE,
    "raw": SAFETY_STOCK_METHOD_K_RMSE,
    "k*rmse": SAFETY_STOCK_METHOD_K_RMSE,
    SAFETY_STOCK_METHOD_K_MAE: SAFETY_STOCK_METHOD_K_MAE,
    "raw_mae": SAFETY_STOCK_METHOD_K_MAE,
    "k*mae": SAFETY_STOCK_METHOD_K_MAE,
}


def normalize_safety_stock_method(method: str | None) -> str:
    if method is None:
        return SAFETY_STOCK_METHOD_SQRT_HORIZON
    normalized = method.strip().lower()
    if normalized in _SAFETY_STOCK_METHOD_ALIASES:
        return _SAFETY_STOCK_METHOD_ALIASES[normalized]
    raise ValueError(
        "safety_stock_method must be one of: "
        f"{', '.join(sorted(_SAFETY_STOCK_METHOD_ALIASES))}."
    )


def _normalize_series(series: Iterable[int] | DemandModel) -> DemandModel:
    if callable(series):
        return series

    series_list = list(series)

    def model(period: int) -> int:
        if period < 0 or period >= len(series_list):
            raise IndexError("Series period out of range.")
        return series_list[period]

    return model


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _aggregate_series(
    values: list[int], window: int, *, extend_last: bool
) -> list[int]:
    if window <= 1:
        return list(values)
    if not values:
        return []
    series = list(values)
    remainder = len(series) % window
    if remainder:
        if extend_last:
            series.extend([series[-1]] * (window - remainder))
        else:
            series = series[:-remainder]
    return [sum(series[i : i + window]) for i in range(0, len(series), window)]


def _rmse_from_series(actuals: list[int], forecasts: list[int]) -> float:
    count = min(len(actuals), len(forecasts))
    if count <= 0:
        return 0.0
    errors = [actuals[index] - forecasts[index] for index in range(count)]
    if not errors:
        return 0.0
    if len(errors) == 1:
        return abs(errors[0])
    mean_squared_error = statistics.fmean(error**2 for error in errors)
    return math.sqrt(mean_squared_error)


def _mae_from_series(actuals: list[int], forecasts: list[int]) -> float:
    count = min(len(actuals), len(forecasts))
    if count <= 0:
        return 0.0
    errors = [abs(actuals[index] - forecasts[index]) for index in range(count)]
    if not errors:
        return 0.0
    return statistics.fmean(errors)


def _safety_stock_from_fill_rate(
    *,
    fill_rate: float,
    forecast_qty: float,
    rmse: float,
    horizon: int,
) -> float:
    std_dev = rmse * math.sqrt(horizon if horizon > 0 else 1)
    if std_dev <= 0 or forecast_qty <= 0:
        return 0.0
    z_value = fill_rate_z(fill_rate, mean_demand=forecast_qty, std_dev=std_dev)
    return z_value * std_dev


def _resolve_buffer_reference(
    reference: float | None, forecast_values: list[int] | None, horizon: int
) -> float | None:
    if reference is not None:
        return reference if reference > 0 else None
    if not forecast_values:
        return None
    mean_forecast = statistics.fmean(forecast_values)
    scaled_reference = mean_forecast * max(1, horizon)
    return scaled_reference if scaled_reference > 0 else None


def _demand_buffer_multiplier(
    *,
    forecast_qty: float,
    reference_qty: float | None,
    strength: float,
    max_multiplier: float | None,
) -> float:
    if strength <= 0 or forecast_qty <= 0 or reference_qty is None or reference_qty <= 0:
        return 1.0
    uplift = max(0.0, (forecast_qty / reference_qty) - 1.0)
    multiplier = 1.0 + (strength * uplift)
    if max_multiplier is not None:
        multiplier = min(multiplier, max_multiplier)
    return max(1.0, multiplier)


@dataclass(frozen=True)
class ReorderPointPolicy:
    """Order up to a fixed quantity when inventory position falls below a point."""

    reorder_point: int
    order_quantity: int

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.inventory_position <= self.reorder_point:
            return self.order_quantity
        return 0


@dataclass(frozen=True)
class ForecastBasedPolicy:
    """Order the forecasted demand plus safety stock using RMSE error."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    safety_stock_method: str = SAFETY_STOCK_METHOD_SQRT_HORIZON
    fixed_rmse: float | None = None
    demand_buffer_strength: float = 0.0
    demand_buffer_reference: float | None = None
    demand_buffer_max_multiplier: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)
    _service_level_mode_normalized: str = field(init=False, repr=False)
    _safety_stock_method_normalized: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        rmse_window = (
            self.rmse_window
            if self.rmse_window is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if rmse_window <= 0:
            raise ValueError("RMSE window must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "rmse_window", rmse_window)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        if self.demand_buffer_strength < 0:
            raise ValueError("Demand buffer strength must be non-negative.")
        if (
            self.demand_buffer_reference is not None
            and self.demand_buffer_reference <= 0
        ):
            raise ValueError("Demand buffer reference must be positive.")
        if (
            self.demand_buffer_max_multiplier is not None
            and self.demand_buffer_max_multiplier < 1
        ):
            raise ValueError("Demand buffer max multiplier must be at least 1.")
        normalized_mode = normalize_service_level_mode(self.service_level_mode)
        object.__setattr__(self, "_service_level_mode_normalized", normalized_mode)
        object.__setattr__(
            self,
            "_safety_stock_method_normalized",
            normalize_safety_stock_method(self.safety_stock_method),
        )
        if normalized_mode == "fill_rate":
            object.__setattr__(self, "_service_level_multiplier", 0.0)
        else:
            object.__setattr__(
                self,
                "_service_level_multiplier",
                service_level_multiplier(
                    self.service_level_factor, self.service_level_mode
                ),
            )
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

        if callable(self.actuals):
            object.__setattr__(self, "_actual_values", None)
            object.__setattr__(self, "_actual_model", self.actuals)
        else:
            actual_values = list(self.actuals)
            object.__setattr__(self, "_actual_values", actual_values)
            object.__setattr__(self, "_actual_model", _normalize_series(actual_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def _safety_stock(self, period: int) -> float:
        if period <= 0:
            return 0.0
        max_index = period
        if self._actual_values is not None:
            max_index = min(max_index, len(self._actual_values))
        if self._forecast_values is not None:
            max_index = min(max_index, len(self._forecast_values))
        if max_index <= 0:
            return 0.0
        if self.fixed_rmse is not None:
            error_value = self.fixed_rmse
        else:
            forecast_series: list[int] = []
            actuals_series: list[int] = []
            for index in range(max_index):
                if self._actual_values is not None:
                    actual = self._actual_values[index]
                    if _is_missing_value(actual):
                        continue
                else:
                    actual = self._actual_model(index)
                    if _is_missing_value(actual):
                        continue
                forecast_value = self._forecast_model(index)
                actuals_series.append(actual)
                forecast_series.append(forecast_value)
            if self.rmse_window is not None and self.rmse_window > 1:
                forecast_series = _aggregate_series(
                    forecast_series, self.rmse_window, extend_last=True
                )
                actuals_series = _aggregate_series(
                    actuals_series, self.rmse_window, extend_last=False
                )
            if self._safety_stock_method_normalized == SAFETY_STOCK_METHOD_K_MAE:
                error_value = _mae_from_series(actuals_series, forecast_series)
            else:
                error_value = _rmse_from_series(actuals_series, forecast_series)
        protection_horizon = self.lead_time + (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        lead_time_factor = math.sqrt(
            protection_horizon if protection_horizon > 0 else 1
        )
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = period + max(1, self.lead_time)
        if self._service_level_mode_normalized == "fill_rate":
            forecast_qty = self._forecast_sum_for(start_period, protection_horizon)
            safety_stock = _safety_stock_from_fill_rate(
                fill_rate=self.service_level_factor,
                forecast_qty=forecast_qty,
                rmse=error_value,
                horizon=protection_horizon,
            )
        else:
            forecast_qty = self._forecast_sum_for(start_period, total_horizon)
            if self._safety_stock_method_normalized in {
                SAFETY_STOCK_METHOD_K_RMSE,
                SAFETY_STOCK_METHOD_K_MAE,
            }:
                safety_stock = self._service_level_multiplier * error_value
            else:
                safety_stock = (
                    self._service_level_multiplier
                    * error_value
                    * lead_time_factor
                )
        demand_multiplier = 1.0
        if self._service_level_mode_normalized != "fill_rate":
            buffer_reference = _resolve_buffer_reference(
                self.demand_buffer_reference,
                self._forecast_values,
                total_horizon,
            )
            demand_multiplier = _demand_buffer_multiplier(
                forecast_qty=forecast_qty,
                reference_qty=buffer_reference,
                strength=self.demand_buffer_strength,
                max_multiplier=self.demand_buffer_max_multiplier,
            )
        return safety_stock * demand_multiplier

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        safety_stock = self._safety_stock(state.period)
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class ForecastSeriesPolicy:
    """Order to a deterministic forecast series for the forecast horizon."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            mean_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", mean_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(mean_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        return max(0, int(math.ceil(forecast_qty - state.inventory_position)))


MeanForecastPolicy = ForecastSeriesPolicy


@dataclass(frozen=True)
class PointForecastOptimizationPolicy:
    """Order mean forecast with safety stock tuned by historical RMSE."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    safety_stock_method: str = SAFETY_STOCK_METHOD_SQRT_HORIZON
    fixed_rmse: float | None = None
    demand_buffer_strength: float = 0.0
    demand_buffer_reference: float | None = None
    demand_buffer_max_multiplier: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)
    _service_level_mode_normalized: str = field(init=False, repr=False)
    _safety_stock_method_normalized: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        rmse_window = (
            self.rmse_window
            if self.rmse_window is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if rmse_window <= 0:
            raise ValueError("RMSE window must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "rmse_window", rmse_window)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        if self.demand_buffer_strength < 0:
            raise ValueError("Demand buffer strength must be non-negative.")
        if (
            self.demand_buffer_reference is not None
            and self.demand_buffer_reference <= 0
        ):
            raise ValueError("Demand buffer reference must be positive.")
        if (
            self.demand_buffer_max_multiplier is not None
            and self.demand_buffer_max_multiplier < 1
        ):
            raise ValueError("Demand buffer max multiplier must be at least 1.")
        normalized_mode = normalize_service_level_mode(self.service_level_mode)
        object.__setattr__(self, "_service_level_mode_normalized", normalized_mode)
        object.__setattr__(
            self,
            "_safety_stock_method_normalized",
            normalize_safety_stock_method(self.safety_stock_method),
        )
        if normalized_mode == "fill_rate":
            object.__setattr__(self, "_service_level_multiplier", 0.0)
        else:
            object.__setattr__(
                self,
                "_service_level_multiplier",
                service_level_multiplier(
                    self.service_level_factor, self.service_level_mode
                ),
            )
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

        if callable(self.actuals):
            object.__setattr__(self, "_actual_values", None)
            object.__setattr__(self, "_actual_model", self.actuals)
        else:
            actual_values = list(self.actuals)
            object.__setattr__(self, "_actual_values", actual_values)
            object.__setattr__(self, "_actual_model", _normalize_series(actual_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def _forecast_error(self, period: int) -> float:
        if self.fixed_rmse is not None:
            return self.fixed_rmse
        if period <= 0:
            return 0.0
        max_index = period
        if self._actual_values is not None:
            max_index = min(max_index, len(self._actual_values))
        if self._forecast_values is not None:
            max_index = min(max_index, len(self._forecast_values))
        if max_index <= 0:
            return 0.0
        forecast_series: list[int] = []
        actuals_series: list[int] = []
        for index in range(max_index):
            if self._actual_values is not None:
                actual = self._actual_values[index]
                if _is_missing_value(actual):
                    continue
            else:
                actual = self._actual_model(index)
                if _is_missing_value(actual):
                    continue
            forecast_value = self._forecast_model(index)
            actuals_series.append(actual)
            forecast_series.append(forecast_value)
        if self.rmse_window is not None and self.rmse_window > 1:
            forecast_series = _aggregate_series(
                forecast_series, self.rmse_window, extend_last=True
            )
            actuals_series = _aggregate_series(
                actuals_series, self.rmse_window, extend_last=False
            )
        if self._safety_stock_method_normalized == SAFETY_STOCK_METHOD_K_MAE:
            return _mae_from_series(actuals_series, forecast_series)
        return _rmse_from_series(actuals_series, forecast_series)

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        error_value = self._forecast_error(state.period)
        protection_horizon = self.lead_time + (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        lead_time_factor = math.sqrt(
            protection_horizon if protection_horizon > 0 else 1
        )
        if self._service_level_mode_normalized == "fill_rate":
            fill_rate_forecast_qty = self._forecast_sum_for(
                start_period, protection_horizon
            )
            safety_stock = _safety_stock_from_fill_rate(
                fill_rate=self.service_level_factor,
                forecast_qty=fill_rate_forecast_qty,
                rmse=error_value,
                horizon=protection_horizon,
            )
        else:
            if self._safety_stock_method_normalized in {
                SAFETY_STOCK_METHOD_K_RMSE,
                SAFETY_STOCK_METHOD_K_MAE,
            }:
                safety_stock = self._service_level_multiplier * error_value
            else:
                safety_stock = (
                    self._service_level_multiplier
                    * error_value
                    * lead_time_factor
                )
        if self._service_level_mode_normalized != "fill_rate":
            buffer_reference = _resolve_buffer_reference(
                self.demand_buffer_reference,
                self._forecast_values,
                total_horizon,
            )
            demand_multiplier = _demand_buffer_multiplier(
                forecast_qty=forecast_qty,
                reference_qty=buffer_reference,
                strength=self.demand_buffer_strength,
                max_multiplier=self.demand_buffer_max_multiplier,
            )
            safety_stock *= demand_multiplier
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class RopPointForecastOptimizationPolicy:
    """Reorder-point policy using mean forecast plus safety stock."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    safety_stock_method: str = SAFETY_STOCK_METHOD_SQRT_HORIZON
    fixed_rmse: float | None = None
    demand_buffer_strength: float = 0.0
    demand_buffer_reference: float | None = None
    demand_buffer_max_multiplier: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)
    _service_level_mode_normalized: str = field(init=False, repr=False)
    _safety_stock_method_normalized: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        rmse_window = (
            self.rmse_window
            if self.rmse_window is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if rmse_window <= 0:
            raise ValueError("RMSE window must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "rmse_window", rmse_window)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        if self.demand_buffer_strength < 0:
            raise ValueError("Demand buffer strength must be non-negative.")
        if (
            self.demand_buffer_reference is not None
            and self.demand_buffer_reference <= 0
        ):
            raise ValueError("Demand buffer reference must be positive.")
        if (
            self.demand_buffer_max_multiplier is not None
            and self.demand_buffer_max_multiplier < 1
        ):
            raise ValueError("Demand buffer max multiplier must be at least 1.")
        normalized_mode = normalize_service_level_mode(self.service_level_mode)
        object.__setattr__(self, "_service_level_mode_normalized", normalized_mode)
        object.__setattr__(
            self,
            "_safety_stock_method_normalized",
            normalize_safety_stock_method(self.safety_stock_method),
        )
        if normalized_mode == "fill_rate":
            object.__setattr__(self, "_service_level_multiplier", 0.0)
        else:
            object.__setattr__(
                self,
                "_service_level_multiplier",
                service_level_multiplier(
                    self.service_level_factor, self.service_level_mode
                ),
            )
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

        if callable(self.actuals):
            object.__setattr__(self, "_actual_values", None)
            object.__setattr__(self, "_actual_model", self.actuals)
        else:
            actual_values = list(self.actuals)
            object.__setattr__(self, "_actual_values", actual_values)
            object.__setattr__(self, "_actual_model", _normalize_series(actual_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def _forecast_error(self, period: int) -> float:
        if self.fixed_rmse is not None:
            return self.fixed_rmse
        if period <= 0:
            return 0.0
        max_index = period
        if self._actual_values is not None:
            max_index = min(max_index, len(self._actual_values))
        if self._forecast_values is not None:
            max_index = min(max_index, len(self._forecast_values))
        if max_index <= 0:
            return 0.0
        forecast_series: list[int] = []
        actuals_series: list[int] = []
        for index in range(max_index):
            if self._actual_values is not None:
                actual = self._actual_values[index]
                if _is_missing_value(actual):
                    continue
            else:
                actual = self._actual_model(index)
                if _is_missing_value(actual):
                    continue
            forecast_value = self._forecast_model(index)
            actuals_series.append(actual)
            forecast_series.append(forecast_value)
        if self.rmse_window is not None and self.rmse_window > 1:
            forecast_series = _aggregate_series(
                forecast_series, self.rmse_window, extend_last=True
            )
            actuals_series = _aggregate_series(
                actuals_series, self.rmse_window, extend_last=False
            )
        if self._safety_stock_method_normalized == SAFETY_STOCK_METHOD_K_MAE:
            return _mae_from_series(actuals_series, forecast_series)
        return _rmse_from_series(actuals_series, forecast_series)

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        lead_horizon = max(0, self.lead_time)
        lead_demand = (
            self._forecast_sum_for(state.period + 1, lead_horizon)
            if lead_horizon > 0
            else 0
        )
        cycle_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        cycle_stock = self._forecast_sum_for(
            state.period + 1 + lead_horizon, cycle_horizon
        )
        error_value = self._forecast_error(state.period)
        lead_time_factor = math.sqrt(lead_horizon if lead_horizon > 0 else 1)
        if self._service_level_mode_normalized == "fill_rate":
            safety_stock = _safety_stock_from_fill_rate(
                fill_rate=self.service_level_factor,
                forecast_qty=lead_demand,
                rmse=error_value,
                horizon=lead_horizon,
            )
        else:
            if self._safety_stock_method_normalized in {
                SAFETY_STOCK_METHOD_K_RMSE,
                SAFETY_STOCK_METHOD_K_MAE,
            }:
                safety_stock = self._service_level_multiplier * error_value
            else:
                safety_stock = (
                    self._service_level_multiplier
                    * error_value
                    * lead_time_factor
                )
        if self._service_level_mode_normalized != "fill_rate":
            buffer_reference = _resolve_buffer_reference(
                self.demand_buffer_reference,
                self._forecast_values,
                lead_horizon,
            )
            demand_multiplier = _demand_buffer_multiplier(
                forecast_qty=lead_demand,
                reference_qty=buffer_reference,
                strength=self.demand_buffer_strength,
                max_multiplier=self.demand_buffer_max_multiplier,
            )
            safety_stock *= demand_multiplier
        reorder_point = lead_demand + safety_stock
        order_up_to = reorder_point + cycle_stock
        if state.inventory_position <= reorder_point:
            return max(0, int(math.ceil(order_up_to - state.inventory_position)))
        return 0


@dataclass(frozen=True)
class LeadTimeForecastOptimizationPolicy:
    """Order up to the forecast-horizon sum plus safety stock using RMSE error."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    safety_stock_method: str = SAFETY_STOCK_METHOD_SQRT_HORIZON
    fixed_rmse: float | None = None
    demand_buffer_strength: float = 0.0
    demand_buffer_reference: float | None = None
    demand_buffer_max_multiplier: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)
    _service_level_mode_normalized: str = field(init=False, repr=False)
    _safety_stock_method_normalized: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        rmse_window = (
            self.rmse_window
            if self.rmse_window is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if rmse_window <= 0:
            raise ValueError("RMSE window must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "rmse_window", rmse_window)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        if self.demand_buffer_strength < 0:
            raise ValueError("Demand buffer strength must be non-negative.")
        if (
            self.demand_buffer_reference is not None
            and self.demand_buffer_reference <= 0
        ):
            raise ValueError("Demand buffer reference must be positive.")
        if (
            self.demand_buffer_max_multiplier is not None
            and self.demand_buffer_max_multiplier < 1
        ):
            raise ValueError("Demand buffer max multiplier must be at least 1.")
        normalized_mode = normalize_service_level_mode(self.service_level_mode)
        object.__setattr__(self, "_service_level_mode_normalized", normalized_mode)
        object.__setattr__(
            self,
            "_safety_stock_method_normalized",
            normalize_safety_stock_method(self.safety_stock_method),
        )
        if normalized_mode == "fill_rate":
            object.__setattr__(self, "_service_level_multiplier", 0.0)
        else:
            object.__setattr__(
                self,
                "_service_level_multiplier",
                service_level_multiplier(
                    self.service_level_factor, self.service_level_mode
                ),
            )
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

        if callable(self.actuals):
            object.__setattr__(self, "_actual_values", None)
            object.__setattr__(self, "_actual_model", self.actuals)
        else:
            actual_values = list(self.actuals)
            object.__setattr__(self, "_actual_values", actual_values)
            object.__setattr__(self, "_actual_model", _normalize_series(actual_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def _forecast_error(self, period: int) -> float:
        if self.fixed_rmse is not None:
            return self.fixed_rmse
        if period <= 0:
            return 0.0
        max_index = period
        if self._actual_values is not None:
            max_index = min(max_index, len(self._actual_values))
        if self._forecast_values is not None:
            max_index = min(max_index, len(self._forecast_values))
        if max_index <= 0:
            return 0.0
        forecast_series: list[int] = []
        actuals_series: list[int] = []
        for index in range(max_index):
            if self._actual_values is not None:
                actual = self._actual_values[index]
                if _is_missing_value(actual):
                    continue
            else:
                actual = self._actual_model(index)
                if _is_missing_value(actual):
                    continue
            forecast_value = self._forecast_model(index)
            actuals_series.append(actual)
            forecast_series.append(forecast_value)
        if self.rmse_window is not None and self.rmse_window > 1:
            forecast_series = _aggregate_series(
                forecast_series, self.rmse_window, extend_last=True
            )
            actuals_series = _aggregate_series(
                actuals_series, self.rmse_window, extend_last=False
            )
        if self._safety_stock_method_normalized == SAFETY_STOCK_METHOD_K_MAE:
            return _mae_from_series(actuals_series, forecast_series)
        return _rmse_from_series(actuals_series, forecast_series)

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        error_value = self._forecast_error(state.period)
        protection_horizon = self.lead_time + (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        lead_time_factor = math.sqrt(
            protection_horizon if protection_horizon > 0 else 1
        )
        if self._service_level_mode_normalized == "fill_rate":
            safety_stock = _safety_stock_from_fill_rate(
                fill_rate=self.service_level_factor,
                forecast_qty=forecast_qty,
                rmse=error_value,
                horizon=protection_horizon,
            )
        else:
            if self._safety_stock_method_normalized in {
                SAFETY_STOCK_METHOD_K_RMSE,
                SAFETY_STOCK_METHOD_K_MAE,
            }:
                safety_stock = self._service_level_multiplier * error_value
            else:
                safety_stock = (
                    self._service_level_multiplier
                    * error_value
                    * lead_time_factor
                )
        if self._service_level_mode_normalized != "fill_rate":
            buffer_reference = _resolve_buffer_reference(
                self.demand_buffer_reference,
                self._forecast_values,
                total_horizon,
            )
            demand_multiplier = _demand_buffer_multiplier(
                forecast_qty=forecast_qty,
                reference_qty=buffer_reference,
                strength=self.demand_buffer_strength,
                max_multiplier=self.demand_buffer_max_multiplier,
            )
            safety_stock *= demand_multiplier
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class PercentileForecastOptimizationPolicy:
    """Order directly from percentile forecasts without safety stock."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            mean_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", mean_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(mean_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        return max(0, int(math.ceil(forecast_qty - state.inventory_position)))


@dataclass(frozen=True)
class RopPercentileForecastOptimizationPolicy:
    """Reorder-point policy using percentile forecasts (no safety stock)."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            mean_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", mean_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(mean_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        lead_horizon = max(0, self.lead_time)
        lead_demand = (
            self._forecast_sum_for(state.period + 1, lead_horizon)
            if lead_horizon > 0
            else 0
        )
        cycle_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        cycle_stock = self._forecast_sum_for(
            state.period + 1 + lead_horizon, cycle_horizon
        )
        reorder_point = lead_demand
        order_up_to = reorder_point + cycle_stock
        if state.inventory_position <= reorder_point:
            return max(0, int(math.ceil(order_up_to - state.inventory_position)))
        return 0


@dataclass(frozen=True)
class EmpiricalMultiplierPolicy:
    """Order mean forecast multiplied by a calibrated factor.

    This policy uses a simple multiplier on the mean forecast to determine
    order quantities. The multiplier is typically calibrated empirically
    to achieve a target lost sales rate during backtesting.
    """

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    multiplier: float = 1.0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.multiplier < 0:
            raise ValueError("Multiplier must be non-negative.")
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        total_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        start_period = state.period + max(1, self.lead_time)
        forecast_qty = self._forecast_sum_for(start_period, total_horizon)
        target = forecast_qty * self.multiplier
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class RopEmpiricalMultiplierPolicy:
    """Reorder-point policy using mean forecast multiplied by a calibrated factor.

    This policy uses a simple multiplier on the mean forecast to determine
    reorder points and order-up-to levels. The multiplier is typically
    calibrated empirically to achieve a target lost sales rate.
    """

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    review_period: int | None = None
    forecast_horizon: int | None = None
    multiplier: float = 1.0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        review_period = (
            self.review_period
            if self.review_period is not None
            else self.aggregation_window
        )
        forecast_horizon = (
            self.forecast_horizon
            if self.forecast_horizon is not None
            else review_period
        )
        if review_period <= 0:
            raise ValueError("Review period must be positive.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "forecast_horizon", forecast_horizon)
        object.__setattr__(self, "aggregation_window", review_period)
        if self.multiplier < 0:
            raise ValueError("Multiplier must be non-negative.")
        if callable(self.forecast):
            object.__setattr__(self, "_forecast_values", None)
            object.__setattr__(self, "_forecast_model", self.forecast)
        else:
            forecast_values = list(self.forecast)
            object.__setattr__(self, "_forecast_values", forecast_values)
            object.__setattr__(self, "_forecast_model", _normalize_series(forecast_values))

    def _forecast_value_for(self, period: int) -> int:
        if period < 0:
            raise IndexError("Series period out of range.")
        if self._forecast_values is None:
            return self._forecast_model(period)
        if not self._forecast_values:
            raise IndexError("Series period out of range.")
        if period >= len(self._forecast_values):
            return self._forecast_values[-1]
        return self._forecast_values[period]

    def _forecast_sum_for(self, period: int, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return sum(
            self._forecast_value_for(period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.review_period is not None and self.review_period > 1:
            if state.period % self.review_period != 0:
                return 0
        lead_horizon = max(0, self.lead_time)
        lead_demand = (
            self._forecast_sum_for(state.period + 1, lead_horizon)
            if lead_horizon > 0
            else 0
        )
        cycle_horizon = (
            self.forecast_horizon if self.forecast_horizon is not None else 1
        )
        cycle_stock = self._forecast_sum_for(
            state.period + 1 + lead_horizon, cycle_horizon
        )
        # Apply multiplier to both lead demand and cycle stock
        reorder_point = lead_demand * self.multiplier
        order_up_to = (lead_demand + cycle_stock) * self.multiplier
        if state.inventory_position <= reorder_point:
            return max(0, int(math.ceil(order_up_to - state.inventory_position)))
        return 0
