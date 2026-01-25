"""Optimization helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .policies import ForecastBasedPolicy, QuantileForecastPolicy
from .simulation import ArticleSimulationConfig, SimulationResult, simulate_replenishment


@dataclass(frozen=True)
class ServiceLevelOptimizationResult:
    service_level_factor: float
    simulation: SimulationResult


@dataclass(frozen=True)
class QuantileOptimizationResult:
    quantile: float
    simulation: SimulationResult


def optimize_service_level_factors(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_factors: Iterable[float],
) -> dict[str, ServiceLevelOptimizationResult]:
    """Pick the service level factor per article that minimizes total cost."""
    factors = list(candidate_factors)
    if not factors:
        raise ValueError("Candidate factors must be provided.")
    if any(factor < 0 for factor in factors):
        raise ValueError("Service level factors must be non-negative.")

    results: dict[str, ServiceLevelOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(policy, ForecastBasedPolicy):
            raise TypeError(
                "Service level optimization requires ForecastBasedPolicy policies."
            )
        best_result: ServiceLevelOptimizationResult | None = None
        for factor in factors:
            candidate_policy = ForecastBasedPolicy(
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
            )
            if best_result is None:
                best_result = ServiceLevelOptimizationResult(
                    service_level_factor=factor,
                    simulation=simulation,
                )
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = ServiceLevelOptimizationResult(
                    service_level_factor=factor,
                    simulation=simulation,
                )
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results


def optimize_quantile_levels(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_quantiles: Iterable[float],
) -> dict[str, QuantileOptimizationResult]:
    """Pick the quantile level per article that minimizes total cost."""
    quantiles = list(candidate_quantiles)
    if not quantiles:
        raise ValueError("Candidate quantiles must be provided.")
    if any(quantile < 0 or quantile > 1 for quantile in quantiles):
        raise ValueError("Quantile levels must be between 0 and 1.")

    results: dict[str, QuantileOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(policy, QuantileForecastPolicy):
            raise TypeError(
                "Quantile optimization requires QuantileForecastPolicy policies."
            )
        best_result: QuantileOptimizationResult | None = None
        for quantile in quantiles:
            candidate_policy = QuantileForecastPolicy(
                mean_forecast=policy.mean_forecast,
                quantile_forecasts=policy.quantile_forecasts,
                lead_time=config.lead_time,
                target_quantile=quantile,
            )
            simulation = simulate_replenishment(
                periods=config.periods,
                demand=config.demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=config.lead_time,
                policy=candidate_policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
            )
            if best_result is None:
                best_result = QuantileOptimizationResult(
                    quantile=quantile,
                    simulation=simulation,
                )
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = QuantileOptimizationResult(
                    quantile=quantile,
                    simulation=simulation,
                )
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results
