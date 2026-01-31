"""Inventory control policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import statistics

from .simulation import DemandModel, InventoryState
from .service_levels import service_level_multiplier


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
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    fixed_rmse: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        object.__setattr__(
            self,
            "_service_level_multiplier",
            service_level_multiplier(self.service_level_factor, self.service_level_mode),
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
        errors = []
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
            errors.append(actual - forecast_value)
        if not errors:
            return 0.0
        if self.fixed_rmse is not None:
            rmse = self.fixed_rmse
        elif len(errors) == 1:
            rmse = abs(errors[0])
        else:
            mean_squared_error = statistics.fmean(error**2 for error in errors)
            rmse = math.sqrt(mean_squared_error)
        horizon = (
            self.lead_time + self.aggregation_window
            if self.aggregation_window > 1
            else self.lead_time
        )
        lead_time_factor = math.sqrt(horizon if horizon > 0 else 1)
        return self._service_level_multiplier * rmse * lead_time_factor

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.aggregation_window > 1 and state.period % self.aggregation_window != 0:
            return 0
        if self.aggregation_window > 1:
            horizon = self.lead_time + self.aggregation_window
            forecast_qty = self._forecast_sum_for(state.period, horizon)
        else:
            target_period = state.period + self.lead_time
            forecast_qty = self._forecast_value_for(target_period)
        safety_stock = self._safety_stock(state.period)
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class ForecastSeriesPolicy:
    """Order to a deterministic forecast series for the lead time horizon."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
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
        if self.aggregation_window > 1 and state.period % self.aggregation_window != 0:
            return 0
        if self.aggregation_window > 1:
            horizon = self.lead_time + self.aggregation_window
            forecast_qty = self._forecast_sum_for(state.period, horizon)
        else:
            target_period = state.period + self.lead_time
            forecast_qty = self._forecast_value_for(target_period)
        return max(0, int(math.ceil(forecast_qty - state.inventory_position)))


MeanForecastPolicy = ForecastSeriesPolicy


@dataclass(frozen=True)
class PointForecastOptimizationPolicy:
    """Order mean forecast with safety stock tuned by historical RMSE."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    fixed_rmse: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        object.__setattr__(
            self,
            "_service_level_multiplier",
            service_level_multiplier(self.service_level_factor, self.service_level_mode),
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

    def _rmse(self, period: int) -> float:
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
        errors = []
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
            errors.append(actual - forecast_value)
        if not errors:
            return 0.0
        if len(errors) == 1:
            return abs(errors[0])
        mean_squared_error = statistics.fmean(error**2 for error in errors)
        return math.sqrt(mean_squared_error)

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.aggregation_window > 1 and state.period % self.aggregation_window != 0:
            return 0
        if self.aggregation_window > 1:
            horizon = self.lead_time + self.aggregation_window
            forecast_qty = self._forecast_sum_for(state.period, horizon)
        else:
            target_period = state.period + self.lead_time
            forecast_qty = self._forecast_value_for(target_period)
        rmse = self._rmse(state.period)
        horizon = (
            self.lead_time + self.aggregation_window
            if self.aggregation_window > 1
            else self.lead_time
        )
        lead_time_factor = math.sqrt(horizon if horizon > 0 else 1)
        safety_stock = self._service_level_multiplier * rmse * lead_time_factor
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class LeadTimeForecastOptimizationPolicy:
    """Order up to lead-time forecast sum plus safety stock using RMSE error."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    service_level_factor: float = 1.0
    service_level_mode: str = "factor"
    fixed_rmse: float | None = None
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)
    _service_level_multiplier: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
        if self.fixed_rmse is not None and self.fixed_rmse < 0:
            raise ValueError("Fixed RMSE must be non-negative.")
        object.__setattr__(
            self,
            "_service_level_multiplier",
            service_level_multiplier(self.service_level_factor, self.service_level_mode),
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

    def _forecast_sum_for(self, period: int) -> int:
        if self.aggregation_window > 1:
            horizon = self.lead_time + self.aggregation_window
            start_period = period
        else:
            horizon = 1
            start_period = period + self.lead_time
        if horizon <= 0:
            return self._forecast_value_for(start_period)
        return sum(
            self._forecast_value_for(start_period + offset)
            for offset in range(horizon)
        )

    def _rmse(self, period: int) -> float:
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
        errors = []
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
            errors.append(actual - forecast_value)
        if not errors:
            return 0.0
        if len(errors) == 1:
            return abs(errors[0])
        mean_squared_error = statistics.fmean(error**2 for error in errors)
        return math.sqrt(mean_squared_error)

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.aggregation_window > 1 and state.period % self.aggregation_window != 0:
            return 0
        target_period = state.period
        forecast_qty = self._forecast_sum_for(target_period)
        rmse = self._rmse(state.period)
        horizon = (
            self.lead_time + self.aggregation_window
            if self.aggregation_window > 1
            else self.lead_time
        )
        lead_time_factor = math.sqrt(horizon if horizon > 0 else 1)
        safety_stock = self._service_level_multiplier * rmse * lead_time_factor
        target = forecast_qty + safety_stock
        return max(0, int(math.ceil(target - state.inventory_position)))


@dataclass(frozen=True)
class PercentileForecastOptimizationPolicy:
    """Order directly from percentile forecasts without safety stock."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    aggregation_window: int = 1
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.aggregation_window <= 0:
            raise ValueError("Aggregation window must be positive.")
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

    def _forecast_sum_for(self, period: int) -> int:
        if self.aggregation_window > 1:
            horizon = self.lead_time + self.aggregation_window
            start_period = period
        else:
            horizon = 1
            start_period = period + self.lead_time
        if horizon <= 0:
            return self._forecast_value_for(start_period)
        return sum(
            self._forecast_value_for(start_period + offset)
            for offset in range(horizon)
        )

    def order_quantity_for(self, state: InventoryState) -> int:
        if self.aggregation_window > 1 and state.period % self.aggregation_window != 0:
            return 0
        target_period = state.period
        forecast_qty = self._forecast_sum_for(target_period)
        return max(0, int(math.ceil(forecast_qty - state.inventory_position)))
