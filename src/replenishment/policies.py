"""Inventory control policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
    """Order the forecasted demand plus safety stock from forecast error."""

    forecast: Iterable[int] | DemandModel
    actuals: Iterable[int] | DemandModel
    lead_time: int = 0
    service_level_factor: float = 1.0
    _forecast_model: DemandModel = field(init=False, repr=False)
    _actual_model: DemandModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if self.service_level_factor < 0:
            raise ValueError("Service level factor must be non-negative.")
        object.__setattr__(self, "_forecast_model", _normalize_series(self.forecast))
        object.__setattr__(self, "_actual_model", _normalize_series(self.actuals))

    def _safety_stock(self, period: int) -> float:
        if period <= 0:
            return 0.0
        errors = [
            self._actual_model(index) - self._forecast_model(index)
            for index in range(period)
        ]
        if len(errors) == 1:
            sigma = abs(errors[0])
        else:
            sigma = statistics.pstdev(errors)
        lead_time_factor = math.sqrt(self.lead_time if self.lead_time > 0 else 1)
        return self.service_level_factor * sigma * lead_time_factor

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._forecast_model(target_period)
        safety_stock = self._safety_stock(state.period)
        return max(0, int(math.ceil(forecast_qty + safety_stock)))


@dataclass(frozen=True)
class QuantileForecastPolicy:
    """Order to a target demand percentile from a distributional forecast."""

    mean_forecast: Iterable[int] | DemandModel
    quantile_forecasts: Mapping[float, Iterable[int] | DemandModel]
    lead_time: int = 0
    target_quantile: float = 0.5
    _mean_model: DemandModel = field(init=False, repr=False)
    _quantile_models: dict[float, DemandModel] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative.")
        if not 0 <= self.target_quantile <= 1:
            raise ValueError("Target quantile must be between 0 and 1.")
        if not self.quantile_forecasts:
            raise ValueError("Quantile forecasts must be provided.")
        quantile_models: dict[float, DemandModel] = {}
        for quantile, series in self.quantile_forecasts.items():
            if not 0 <= quantile <= 1:
                raise ValueError("Quantile levels must be between 0 and 1.")
            quantile_models[quantile] = _normalize_series(series)
        if self.target_quantile not in quantile_models:
            raise ValueError("Target quantile is not available in forecasts.")
        object.__setattr__(self, "_mean_model", _normalize_series(self.mean_forecast))
        object.__setattr__(self, "_quantile_models", quantile_models)

    def order_quantity_for(self, state: InventoryState) -> int:
        target_period = state.period + self.lead_time
        forecast_qty = self._quantile_models[self.target_quantile](target_period)
        return max(0, int(math.ceil(forecast_qty)))
