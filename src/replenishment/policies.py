"""Inventory control policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import statistics

from .simulation import DemandModel, InventoryState


def _normalize_series(series: Iterable[int] | DemandModel) -> DemandModel:
    if callable(series):
        return series

    series_list = list(series)

    def model(period: int) -> int:
        if period < 0 or period >= len(series_list):
            raise IndexError("Series period out of range.")
        return series_list[period]

    return model


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
    service_level_factor: float = 1.0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.service_level_factor < 0:
            raise ValueError("Service level factor must be non-negative.")
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
        errors = [
            self._actual_model(index) - self._forecast_model(index)
            for index in range(max_index)
        ]
        if len(errors) == 1:
            sigma = abs(errors[0])
        else:
            mean_squared_error = statistics.fmean(error**2 for error in errors)
            sigma = math.sqrt(mean_squared_error)
        lead_time_factor = math.sqrt(self.lead_time if self.lead_time > 0 else 1)
        return self.service_level_factor * sigma * lead_time_factor

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._forecast_value_for(target_period)
        safety_stock = self._safety_stock(state.period)
        return max(0, int(math.ceil(forecast_qty + safety_stock)))


@dataclass(frozen=True)
class ForecastSeriesPolicy:
    """Order to a deterministic forecast series for the lead time horizon."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
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

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._forecast_value_for(target_period)
        return max(0, int(math.ceil(forecast_qty)))


MeanForecastPolicy = ForecastSeriesPolicy


@dataclass(frozen=True)
class PointForecastOptimizationPolicy:
    """Order mean forecast with safety stock tuned by historical RMSE."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    service_level_factor: float = 1.0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)
    _actual_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.service_level_factor < 0:
            raise ValueError("Service level factor must be non-negative.")
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

    def _rmse(self, period: int) -> float:
        if period <= 0:
            return 0.0
        max_index = period
        if self._actual_values is not None:
            max_index = min(max_index, len(self._actual_values))
        if self._forecast_values is not None:
            max_index = min(max_index, len(self._forecast_values))
        if max_index <= 0:
            return 0.0
        errors = [
            self._actual_model(index) - self._forecast_model(index)
            for index in range(max_index)
        ]
        if len(errors) == 1:
            return abs(errors[0])
        mean_squared_error = statistics.fmean(error**2 for error in errors)
        return math.sqrt(mean_squared_error)

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._forecast_value_for(target_period)
        rmse = self._rmse(state.period)
        lead_time_factor = math.sqrt(self.lead_time if self.lead_time > 0 else 1)
        safety_stock = self.service_level_factor * rmse * lead_time_factor
        return max(0, int(math.ceil(forecast_qty + safety_stock)))


@dataclass(frozen=True)
class PercentileForecastOptimizationPolicy:
    """Order directly from percentile forecasts without safety stock."""

    forecast: Iterable[int] | DemandModel
    lead_time: int = 0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _forecast_values: list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
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

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._forecast_value_for(target_period)
        return max(0, int(math.ceil(forecast_qty)))
