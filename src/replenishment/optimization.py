"""Optimization helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .policies import ForecastBasedPolicy, ForecastSeriesPolicy
from .simulation import ArticleSimulationConfig, DemandModel, SimulationResult, simulate_replenishment


@dataclass(frozen=True)
class ServiceLevelOptimizationResult:
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


@dataclass(frozen=True)
class ForecastTargetOptimizationResult:
    target: float | str
    policy: ForecastSeriesPolicy
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


def optimize_forecast_targets(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_targets: Iterable[float | str] | None = None,
) -> dict[str, ForecastTargetOptimizationResult]:
    """Pick the forecast candidate (mean or quantile) that minimizes total cost."""
    results: dict[str, ForecastTargetOptimizationResult] = {}
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

        best_result: ForecastTargetOptimizationResult | None = None
        for target in targets:
            if target not in config.forecast_candidates:
                raise ValueError(f"Unknown forecast target: {target}")
            candidate_policy = ForecastSeriesPolicy(
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
            )
            if best_result is None:
                best_result = ForecastTargetOptimizationResult(
                    target=target,
                    policy=candidate_policy,
                    simulation=simulation,
                )
                continue
            if simulation.summary.total_cost < best_result.simulation.summary.total_cost:
                best_result = ForecastTargetOptimizationResult(
                    target=target,
                    policy=candidate_policy,
                    simulation=simulation,
                )
        if best_result is None:
            raise RuntimeError("No optimization result computed.")
        results[article_id] = best_result
    return results
