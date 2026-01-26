"""Optimization helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .aggregation import (
    aggregate_lead_time,
    aggregate_periods,
    aggregate_policy,
    aggregate_series,
)
from .policies import (
    ForecastBasedPolicy,
    ForecastSeriesPolicy,
    PercentileForecastOptimizationPolicy,
    PointForecastOptimizationPolicy,
)
from .simulation import ArticleSimulationConfig, DemandModel, SimulationResult, simulate_replenishment


@dataclass(frozen=True)
class PointForecastOptimizationResult:
    service_level_factor: float
    simulation: SimulationResult


@dataclass(frozen=True)
class ForecastCandidatesConfig:
    """Inputs for forecast-candidate optimization.

    Use string labels to disambiguate candidates (for example "45-low", "45-high").
    """

    periods: int
    demand: Iterable[int] | DemandModel
    initial_on_hand: int
    lead_time: int
    forecast_candidates: Mapping[float | str, Iterable[int] | DemandModel]
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0


@dataclass(frozen=True)
class PercentileForecastOptimizationResult:
    target: float | str
    policy: PercentileForecastOptimizationPolicy
    simulation: SimulationResult


ServiceLevelOptimizationResult = PointForecastOptimizationResult
ForecastTargetOptimizationResult = PercentileForecastOptimizationResult


@dataclass(frozen=True)
class AggregationWindowOptimizationResult:
    window: int
    simulation: SimulationResult
    policy: (
        ForecastSeriesPolicy
        | ForecastBasedPolicy
        | PointForecastOptimizationPolicy
        | PercentileForecastOptimizationPolicy
        | None
    )


@dataclass(frozen=True)
class AggregationServiceLevelOptimizationResult:
    window: int
    service_level_factor: float
    policy: PointForecastOptimizationPolicy
    simulation: SimulationResult


@dataclass(frozen=True)
class AggregationForecastTargetOptimizationResult:
    window: int
    target: float | str
    policy: PercentileForecastOptimizationPolicy
    simulation: SimulationResult


def optimize_service_level_factors(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_factors: Iterable[float],
) -> dict[str, PointForecastOptimizationResult]:
    """Pick the point-forecast safety stock factor that minimizes total cost."""
    factors = list(candidate_factors)
    if not factors:
        raise ValueError("Candidate factors must be provided.")
    if any(factor < 0 for factor in factors):
        raise ValueError("Service level factors must be non-negative.")

    results: dict[str, PointForecastOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy, (ForecastBasedPolicy, PointForecastOptimizationPolicy)
        ):
            raise TypeError(
                "Point forecast optimization requires point-forecast policies."
            )
        best_result: PointForecastOptimizationResult | None = None
        for factor in factors:
            candidate_policy = PointForecastOptimizationPolicy(
                forecast=policy.forecast,
                actuals=policy.actuals,
                lead_time=config.lead_time,
                service_level_factor=factor,
            )
            simulation = simulate_replenishment(
                periods=config.periods,
                demand=config.demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=config.lead_time,
                policy=candidate_policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
                order_cost_per_order=config.order_cost_per_order,
                order_cost_per_unit=config.order_cost_per_unit,
            )
            if best_result is None:
                best_result = PointForecastOptimizationResult(
                    service_level_factor=factor,
                    simulation=simulation,
                )
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = PointForecastOptimizationResult(
                    service_level_factor=factor,
                    simulation=simulation,
                )
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results


def point_forecast_optimisation(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_factors: Iterable[float],
) -> dict[str, PointForecastOptimizationResult]:
    """Alias for optimizing point-forecast safety stock factors."""
    return optimize_service_level_factors(articles, candidate_factors)


def optimize_forecast_targets(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_targets: Iterable[float | str] | None = None,
) -> dict[str, PercentileForecastOptimizationResult]:
    """Pick the percentile forecast candidate that minimizes total cost."""
    results: dict[str, PercentileForecastOptimizationResult] = {}
    for article_id, config in articles.items():
        if not config.forecast_candidates:
            raise ValueError("Forecast candidates must be provided.")
        targets = (
            list(candidate_targets)
            if candidate_targets is not None
            else list(config.forecast_candidates.keys())
        )
        if not targets:
            raise ValueError("Candidate targets must be provided.")

        best_result: PercentileForecastOptimizationResult | None = None
        for target in targets:
            if target not in config.forecast_candidates:
                raise ValueError(f"Unknown forecast target: {target}")
            candidate_policy = PercentileForecastOptimizationPolicy(
                forecast=config.forecast_candidates[target],
                lead_time=config.lead_time,
            )
            simulation = simulate_replenishment(
                periods=config.periods,
                demand=config.demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=config.lead_time,
                policy=candidate_policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
                order_cost_per_order=config.order_cost_per_order,
                order_cost_per_unit=config.order_cost_per_unit,
            )
            if best_result is None:
                best_result = PercentileForecastOptimizationResult(
                    target=target,
                    policy=candidate_policy,
                    simulation=simulation,
                )
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = PercentileForecastOptimizationResult(
                    target=target,
                    policy=candidate_policy,
                    simulation=simulation,
                )
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results


def percentile_forecast_optimisation(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_targets: Iterable[float | str] | None = None,
) -> dict[str, PercentileForecastOptimizationResult]:
    """Alias for optimizing percentile forecast targets without safety stock."""
    return optimize_forecast_targets(articles, candidate_targets)


def optimize_aggregation_windows(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_windows: Iterable[int],
) -> dict[str, AggregationWindowOptimizationResult]:
    """Pick the aggregation window that minimizes total cost."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    results: dict[str, AggregationWindowOptimizationResult] = {}
    for article_id, config in articles.items():
        best_result: AggregationWindowOptimizationResult | None = None
        for window in windows:
            aggregated_demand = aggregate_series(
                config.demand,
                periods=config.periods,
                window=window,
                extend_last=False,
            )
            aggregated_policy = aggregate_policy(
                config.policy,
                periods=config.periods,
                window=window,
                lead_time=config.lead_time,
            )
            aggregated_periods = aggregate_periods(config.periods, window)
            aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
            simulation = simulate_replenishment(
                periods=aggregated_periods,
                demand=aggregated_demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=aggregated_lead_time,
                policy=aggregated_policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
                order_cost_per_order=config.order_cost_per_order,
                order_cost_per_unit=config.order_cost_per_unit,
            )
            candidate = AggregationWindowOptimizationResult(
                window=window,
                simulation=simulation,
                policy=aggregated_policy
                if isinstance(
                    aggregated_policy,
                    (
                        ForecastSeriesPolicy,
                        ForecastBasedPolicy,
                        PointForecastOptimizationPolicy,
                        PercentileForecastOptimizationPolicy,
                    ),
                )
                else None,
            )
            if best_result is None:
                best_result = candidate
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = candidate
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results


def optimize_aggregation_and_service_level_factors(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_windows: Iterable[int],
    candidate_factors: Iterable[float],
) -> dict[str, AggregationServiceLevelOptimizationResult]:
    """Pick the aggregation window and service-level factor that minimize total cost."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    factors = list(candidate_factors)
    if not factors:
        raise ValueError("Candidate factors must be provided.")
    if any(factor < 0 for factor in factors):
        raise ValueError("Service level factors must be non-negative.")

    results: dict[str, AggregationServiceLevelOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy, (ForecastBasedPolicy, PointForecastOptimizationPolicy)
        ):
            raise TypeError(
                "Aggregation with service-level optimization requires point-forecast policies."
            )

        best_result: AggregationServiceLevelOptimizationResult | None = None
        for window in windows:
            aggregated_forecast = aggregate_series(
                policy.forecast,
                periods=config.periods,
                window=window,
                extend_last=True,
            )
            if callable(policy.actuals):
                aggregated_actuals = aggregate_series(
                    policy.actuals,
                    periods=config.periods,
                    window=window,
                    extend_last=False,
                )
            else:
                actual_values = list(policy.actuals)
                if not actual_values:
                    aggregated_actuals = []
                else:
                    actual_periods = min(config.periods, len(actual_values))
                    aggregated_actuals = aggregate_series(
                        actual_values,
                        periods=actual_periods,
                        window=window,
                        extend_last=False,
                    )
            aggregated_demand = aggregate_series(
                config.demand,
                periods=config.periods,
                window=window,
                extend_last=False,
            )
            aggregated_periods = aggregate_periods(config.periods, window)
            aggregated_lead_time = aggregate_lead_time(config.lead_time, window)

            for factor in factors:
                candidate_policy = PointForecastOptimizationPolicy(
                    forecast=aggregated_forecast,
                    actuals=aggregated_actuals,
                    lead_time=aggregated_lead_time,
                    service_level_factor=factor,
                )
                simulation = simulate_replenishment(
                    periods=aggregated_periods,
                    demand=aggregated_demand,
                    initial_on_hand=config.initial_on_hand,
                    lead_time=aggregated_lead_time,
                    policy=candidate_policy,
                    holding_cost_per_unit=config.holding_cost_per_unit,
                    stockout_cost_per_unit=config.stockout_cost_per_unit,
                    order_cost_per_order=config.order_cost_per_order,
                    order_cost_per_unit=config.order_cost_per_unit,
                )
                candidate = AggregationServiceLevelOptimizationResult(
                    window=window,
                    service_level_factor=factor,
                    policy=candidate_policy,
                    simulation=simulation,
                )
                if best_result is None:
                    best_result = candidate
                    continue
                if (
                    simulation.summary.total_cost
                    < best_result.simulation.summary.total_cost
                ):
                    best_result = candidate

        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result

    return results


def optimize_aggregation_and_forecast_targets(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_windows: Iterable[int],
    candidate_targets: Iterable[float | str] | None = None,
) -> dict[str, AggregationForecastTargetOptimizationResult]:
    """Pick the aggregation window and forecast target that minimize total cost."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    target_candidates = list(candidate_targets) if candidate_targets is not None else None
    results: dict[str, AggregationForecastTargetOptimizationResult] = {}
    for article_id, config in articles.items():
        if not config.forecast_candidates:
            raise ValueError("Forecast candidates must be provided.")
        candidate_series = {
            target: list(series) if not callable(series) else series
            for target, series in config.forecast_candidates.items()
        }
        targets = (
            list(target_candidates)
            if target_candidates is not None
            else list(candidate_series.keys())
        )
        if not targets:
            raise ValueError("Candidate targets must be provided.")

        best_result: AggregationForecastTargetOptimizationResult | None = None
        for window in windows:
            aggregated_periods = aggregate_periods(config.periods, window)
            aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
            aggregated_demand = aggregate_series(
                config.demand,
                periods=config.periods,
                window=window,
                extend_last=False,
            )
            aggregated_candidates = {
                target: aggregate_series(
                    forecast,
                    periods=config.periods,
                    window=window,
                    extend_last=True,
                )
                for target, forecast in candidate_series.items()
            }

            for target in targets:
                if target not in aggregated_candidates:
                    raise ValueError(f"Unknown forecast target: {target}")
                candidate_policy = PercentileForecastOptimizationPolicy(
                    forecast=aggregated_candidates[target],
                    lead_time=aggregated_lead_time,
                )
                simulation = simulate_replenishment(
                    periods=aggregated_periods,
                    demand=aggregated_demand,
                    initial_on_hand=config.initial_on_hand,
                    lead_time=aggregated_lead_time,
                    policy=candidate_policy,
                    holding_cost_per_unit=config.holding_cost_per_unit,
                    stockout_cost_per_unit=config.stockout_cost_per_unit,
                    order_cost_per_order=config.order_cost_per_order,
                    order_cost_per_unit=config.order_cost_per_unit,
                )
                candidate = AggregationForecastTargetOptimizationResult(
                    window=window,
                    target=target,
                    policy=candidate_policy,
                    simulation=simulation,
                )
                if best_result is None:
                    best_result = candidate
                    continue
                if (
                    simulation.summary.total_cost
                    < best_result.simulation.summary.total_cost
                ):
                    best_result = candidate

        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result

    return results
