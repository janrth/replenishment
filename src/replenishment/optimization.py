"""Optimization helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace as dataclass_replace

from .aggregation import (
    aggregate_lead_time,
    aggregate_periods,
    aggregate_policy,
    aggregate_series,
)
from .policies import (
    EmpiricalMultiplierPolicy,
    ForecastBasedPolicy,
    ForecastSeriesPolicy,
    LeadTimeForecastOptimizationPolicy,
    PercentileForecastOptimizationPolicy,
    PointForecastOptimizationPolicy,
    RopEmpiricalMultiplierPolicy,
    RopPercentileForecastOptimizationPolicy,
    RopPointForecastOptimizationPolicy,
)
from .service_levels import normalize_service_level_mode
from .simulation import (
    ArticleSimulationConfig,
    DemandModel,
    SimulationMetadata,
    SimulationResult,
    simulate_replenishment,
)


def _attach_metadata(
    simulation: SimulationResult,
    *,
    service_level_factor: float | None = None,
    service_level_mode: str | None = None,
    aggregation_window: int | None = None,
    percentile_target: float | str | None = None,
) -> SimulationResult:
    base = simulation.metadata
    metadata = SimulationMetadata(
        service_level_factor=(
            service_level_factor
            if service_level_factor is not None
            else base.service_level_factor
            if base is not None
            else None
        ),
        service_level_mode=(
            service_level_mode
            if service_level_mode is not None
            else base.service_level_mode
            if base is not None
            else None
        ),
        aggregation_window=(
            aggregation_window
            if aggregation_window is not None
            else base.aggregation_window
            if base is not None
            else None
        ),
        percentile_target=(
            percentile_target
            if percentile_target is not None
            else base.percentile_target
            if base is not None
            else None
        ),
    )
    return SimulationResult(
        snapshots=simulation.snapshots,
        summary=simulation.summary,
        metadata=metadata,
    )


def _rmse_from_series(actuals: Iterable[int], forecasts: Iterable[int]) -> float:
    actual_values = list(actuals)
    forecast_values = list(forecasts)
    count = min(len(actual_values), len(forecast_values))
    if count <= 0:
        return 0.0
    errors = [
        actual_values[index] - forecast_values[index]
        for index in range(count)
    ]
    if not errors:
        return 0.0
    if len(errors) == 1:
        return abs(errors[0])
    mean_squared_error = sum(error * error for error in errors) / len(errors)
    return mean_squared_error**0.5


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


def _resolve_service_level_mode(
    policy: ForecastBasedPolicy | PointForecastOptimizationPolicy | LeadTimeForecastOptimizationPolicy,
    service_level_mode: str | None,
) -> str:
    if service_level_mode is not None:
        return normalize_service_level_mode(service_level_mode)
    return normalize_service_level_mode(getattr(policy, "service_level_mode", None))


def _validate_service_level_candidates(
    factors: list[float], service_level_mode: str
) -> None:
    if not factors:
        raise ValueError("Candidate factors must be provided.")
    if service_level_mode == "factor":
        if any(factor < 0 for factor in factors):
            raise ValueError("Service level factors must be non-negative.")
        return
    if any(factor <= 0 or factor >= 1 for factor in factors):
        raise ValueError(
            "Service level probabilities must be between 0 and 1 (exclusive)."
        )


def _candidate_policy_for(
    policy: (
        ForecastBasedPolicy
        | PointForecastOptimizationPolicy
        | LeadTimeForecastOptimizationPolicy
        | RopPointForecastOptimizationPolicy
    ),
    *,
    forecast: Iterable[int] | DemandModel,
    actuals: Iterable[int] | DemandModel,
    lead_time: int,
    factor: float,
    mode: str,
    fixed_rmse: float | None = None,
    aggregation_window: int | None = None,
) -> (
    PointForecastOptimizationPolicy
    | LeadTimeForecastOptimizationPolicy
    | RopPointForecastOptimizationPolicy
):
    if isinstance(policy, LeadTimeForecastOptimizationPolicy):
        return LeadTimeForecastOptimizationPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=lead_time,
            aggregation_window=(
                aggregation_window
                if aggregation_window is not None
                else policy.aggregation_window
            ),
            service_level_factor=factor,
            service_level_mode=mode,
            fixed_rmse=fixed_rmse,
        )
    if isinstance(policy, RopPointForecastOptimizationPolicy):
        return RopPointForecastOptimizationPolicy(
            forecast=forecast,
            actuals=actuals,
            lead_time=lead_time,
            aggregation_window=(
                aggregation_window
                if aggregation_window is not None
                else policy.aggregation_window
            ),
            service_level_factor=factor,
            service_level_mode=mode,
            fixed_rmse=fixed_rmse,
        )
    return PointForecastOptimizationPolicy(
        forecast=forecast,
        actuals=actuals,
        lead_time=lead_time,
        aggregation_window=(
            aggregation_window
            if aggregation_window is not None
            else policy.aggregation_window
        ),
        service_level_factor=factor,
        service_level_mode=mode,
        fixed_rmse=fixed_rmse,
    )


def optimize_service_level_factors(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_factors: Iterable[float],
    service_level_mode: str | None = None,
) -> dict[str, PointForecastOptimizationResult]:
    """Pick the point-forecast safety stock factor that minimizes total cost."""
    factors = list(candidate_factors)

    results: dict[str, PointForecastOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy,
            (
                ForecastBasedPolicy,
                PointForecastOptimizationPolicy,
                LeadTimeForecastOptimizationPolicy,
                RopPointForecastOptimizationPolicy,
            ),
        ):
            raise TypeError(
                "Point forecast optimization requires point-forecast policies."
            )
        mode = _resolve_service_level_mode(policy, service_level_mode)
        _validate_service_level_candidates(factors, mode)
        best_result: PointForecastOptimizationResult | None = None
        for factor in factors:
            candidate_policy = _candidate_policy_for(
                policy,
                forecast=policy.forecast,
                actuals=policy.actuals,
                lead_time=config.lead_time,
                factor=factor,
                mode=mode,
                fixed_rmse=getattr(policy, "fixed_rmse", None),
                aggregation_window=getattr(policy, "aggregation_window", None),
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
            simulation = _attach_metadata(
                simulation,
                service_level_factor=factor,
                service_level_mode=mode,
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
    *,
    policy_mode: str = "base_stock",
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
            if policy_mode == "rop":
                candidate_policy = RopPercentileForecastOptimizationPolicy(
                    forecast=config.forecast_candidates[target],
                    lead_time=config.lead_time,
                )
            elif policy_mode == "base_stock":
                candidate_policy = PercentileForecastOptimizationPolicy(
                    forecast=config.forecast_candidates[target],
                    lead_time=config.lead_time,
                )
            else:
                raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
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
            simulation = _attach_metadata(
                simulation, percentile_target=target
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


def evaluate_forecast_target_costs(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_targets: Iterable[float | str] | None = None,
    *,
    policy_mode: str = "base_stock",
) -> dict[str, dict[float | str, float]]:
    """Return total costs for each forecast candidate per article."""
    results: dict[str, dict[float | str, float]] = {}
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

        target_costs: dict[float | str, float] = {}
        for target in targets:
            if target not in config.forecast_candidates:
                raise ValueError(f"Unknown forecast target: {target}")
            if policy_mode == "rop":
                candidate_policy = RopPercentileForecastOptimizationPolicy(
                    forecast=config.forecast_candidates[target],
                    lead_time=config.lead_time,
                )
            elif policy_mode == "base_stock":
                candidate_policy = PercentileForecastOptimizationPolicy(
                    forecast=config.forecast_candidates[target],
                    lead_time=config.lead_time,
                )
            else:
                raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
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
            target_costs[target] = simulation.summary.total_cost
        results[article_id] = target_costs
    return results


def evaluate_service_level_factor_costs(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_factors: Iterable[float],
    service_level_mode: str | None = None,
) -> dict[str, dict[float, float]]:
    """Return total costs for each service-level factor per article."""
    factors = list(candidate_factors)

    results: dict[str, dict[float, float]] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy,
            (
                ForecastBasedPolicy,
                PointForecastOptimizationPolicy,
                LeadTimeForecastOptimizationPolicy,
                RopPointForecastOptimizationPolicy,
            ),
        ):
            raise TypeError(
                "Service-level cost evaluation requires point-forecast policies."
            )
        mode = _resolve_service_level_mode(policy, service_level_mode)
        _validate_service_level_candidates(factors, mode)
        factor_costs: dict[float, float] = {}
        for factor in factors:
            candidate_policy = _candidate_policy_for(
                policy,
                forecast=policy.forecast,
                actuals=policy.actuals,
                lead_time=config.lead_time,
                factor=factor,
                mode=mode,
                fixed_rmse=getattr(policy, "fixed_rmse", None),
                aggregation_window=getattr(policy, "aggregation_window", None),
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
            factor_costs[factor] = simulation.summary.total_cost
        results[article_id] = factor_costs
    return results


def evaluate_aggregation_and_service_level_factor_costs(
    articles: Mapping[str, ArticleSimulationConfig],
    candidate_windows: Iterable[int],
    candidate_factors: Iterable[float],
    service_level_mode: str | None = None,
) -> dict[str, dict[int, dict[float, float]]]:
    """Return total costs for each window and service-level factor per article."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    factors = list(candidate_factors)

    results: dict[str, dict[int, dict[float, float]]] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy,
            (
                ForecastBasedPolicy,
                PointForecastOptimizationPolicy,
                LeadTimeForecastOptimizationPolicy,
                RopPointForecastOptimizationPolicy,
            ),
        ):
            raise TypeError(
                "Aggregation with service-level costs requires point-forecast policies."
            )
        mode = _resolve_service_level_mode(policy, service_level_mode)
        _validate_service_level_candidates(factors, mode)
        window_costs: dict[int, dict[float, float]] = {}
        for window in windows:
            factor_costs: dict[float, float] = {}
            for factor in factors:
                candidate_policy = _candidate_policy_for(
                    policy,
                    forecast=policy.forecast,
                    actuals=policy.actuals,
                    lead_time=config.lead_time,
                    factor=factor,
                    mode=mode,
                    fixed_rmse=getattr(policy, "fixed_rmse", None),
                    aggregation_window=window,
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
                factor_costs[factor] = simulation.summary.total_cost
            window_costs[window] = factor_costs
        results[article_id] = window_costs
    return results


def evaluate_aggregation_and_forecast_target_costs(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_windows: Iterable[int],
    candidate_targets: Iterable[float | str] | None = None,
    *,
    policy_mode: str = "base_stock",
) -> dict[str, dict[int, dict[float | str, float]]]:
    """Return total costs for each window and forecast target per article."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    target_candidates = list(candidate_targets) if candidate_targets is not None else None
    results: dict[str, dict[int, dict[float | str, float]]] = {}
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

        window_costs: dict[int, dict[float | str, float]] = {}
        for window in windows:
            aggregated_demand = aggregate_series(
                config.demand,
                periods=config.periods,
                window=window,
                extend_last=False,
            )
            aggregated_periods = aggregate_periods(config.periods, window)
            aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
            target_costs: dict[float | str, float] = {}
            for target in targets:
                if target not in candidate_series:
                    raise ValueError(f"Unknown forecast target: {target}")
                aggregated_forecast = aggregate_series(
                    candidate_series[target],
                    periods=config.periods,
                    window=window,
                    extend_last=True,
                )
                if policy_mode == "rop":
                    candidate_policy = RopPercentileForecastOptimizationPolicy(
                        forecast=aggregated_forecast,
                        lead_time=aggregated_lead_time,
                        aggregation_window=1,
                    )
                elif policy_mode == "base_stock":
                    candidate_policy = PercentileForecastOptimizationPolicy(
                        forecast=aggregated_forecast,
                        lead_time=aggregated_lead_time,
                        aggregation_window=1,
                    )
                else:
                    raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
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
                target_costs[target] = simulation.summary.total_cost
            window_costs[window] = target_costs
        results[article_id] = window_costs
    return results


def percentile_forecast_optimisation(
    articles: Mapping[str, ForecastCandidatesConfig],
    candidate_targets: Iterable[float | str] | None = None,
    *,
    policy_mode: str = "base_stock",
) -> dict[str, PercentileForecastOptimizationResult]:
    """Alias for optimizing percentile forecast targets without safety stock."""
    return optimize_forecast_targets(
        articles, candidate_targets, policy_mode=policy_mode
    )


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
            if hasattr(config.policy, "aggregation_window"):
                aggregated_policy = dataclass_replace(
                    config.policy, aggregation_window=window
                )
            else:
                aggregated_policy = config.policy
            simulation = simulate_replenishment(
                periods=config.periods,
                demand=config.demand,
                initial_on_hand=config.initial_on_hand,
                lead_time=config.lead_time,
                policy=aggregated_policy,
                holding_cost_per_unit=config.holding_cost_per_unit,
                stockout_cost_per_unit=config.stockout_cost_per_unit,
                order_cost_per_order=config.order_cost_per_order,
                order_cost_per_unit=config.order_cost_per_unit,
            )
            simulation = _attach_metadata(
                simulation, aggregation_window=window
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
    service_level_mode: str | None = None,
) -> dict[str, AggregationServiceLevelOptimizationResult]:
    """Pick the aggregation window and service-level factor that minimize total cost."""
    windows = list(candidate_windows)
    if not windows:
        raise ValueError("Aggregation windows must be provided.")
    if any(window <= 0 for window in windows):
        raise ValueError("Aggregation windows must be positive.")

    factors = list(candidate_factors)

    results: dict[str, AggregationServiceLevelOptimizationResult] = {}
    for article_id, config in articles.items():
        policy = config.policy
        if not isinstance(
            policy,
            (ForecastBasedPolicy, PointForecastOptimizationPolicy, LeadTimeForecastOptimizationPolicy),
        ):
            raise TypeError(
                "Aggregation with service-level optimization requires point-forecast policies."
            )
        mode = _resolve_service_level_mode(policy, service_level_mode)
        _validate_service_level_candidates(factors, mode)

        best_result: AggregationServiceLevelOptimizationResult | None = None
        for window in windows:
            for factor in factors:
                candidate_policy = _candidate_policy_for(
                    policy,
                    forecast=policy.forecast,
                    actuals=policy.actuals,
                    lead_time=config.lead_time,
                    factor=factor,
                    mode=mode,
                    fixed_rmse=getattr(policy, "fixed_rmse", None),
                    aggregation_window=window,
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
                simulation = _attach_metadata(
                    simulation,
                    service_level_factor=factor,
                    service_level_mode=mode,
                    aggregation_window=window,
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
    *,
    policy_mode: str = "base_stock",
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
            aggregated_demand = aggregate_series(
                config.demand,
                periods=config.periods,
                window=window,
                extend_last=False,
            )
            aggregated_periods = aggregate_periods(config.periods, window)
            aggregated_lead_time = aggregate_lead_time(config.lead_time, window)
            for target in targets:
                if target not in candidate_series:
                    raise ValueError(f"Unknown forecast target: {target}")
                aggregated_forecast = aggregate_series(
                    candidate_series[target],
                    periods=config.periods,
                    window=window,
                    extend_last=True,
                )
                if policy_mode == "rop":
                    candidate_policy = RopPercentileForecastOptimizationPolicy(
                        forecast=aggregated_forecast,
                        lead_time=aggregated_lead_time,
                        aggregation_window=1,
                    )
                elif policy_mode == "base_stock":
                    candidate_policy = PercentileForecastOptimizationPolicy(
                        forecast=aggregated_forecast,
                        lead_time=aggregated_lead_time,
                        aggregation_window=1,
                    )
                else:
                    raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
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
                simulation = _attach_metadata(
                    simulation,
                    aggregation_window=window,
                    percentile_target=target,
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


@dataclass(frozen=True)
class EmpiricalCalibrationConfig:
    """Inputs for empirical multiplier calibration.

    The calibration will find the smallest multiplier that achieves
    the target lost sales rate during backtesting.
    """

    periods: int
    demand: Iterable[int] | DemandModel
    forecast: Iterable[int] | DemandModel
    initial_on_hand: int
    lead_time: int
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0


@dataclass(frozen=True)
class EmpiricalCalibrationResult:
    """Result of empirical multiplier calibration."""

    multiplier: float
    lost_sales_rate: float
    policy: EmpiricalMultiplierPolicy | RopEmpiricalMultiplierPolicy
    simulation: SimulationResult


def calibrate_empirical_multipliers(
    articles: Mapping[str, EmpiricalCalibrationConfig],
    candidate_multipliers: Iterable[float],
    target_lost_sales_rate: float = 0.05,
    *,
    policy_mode: str = "base_stock",
) -> dict[str, EmpiricalCalibrationResult]:
    """Calibrate forecast multipliers to achieve a target lost sales rate.

    For each article, this function simulates with different multipliers
    and selects the smallest multiplier that achieves <= target_lost_sales_rate.

    Parameters
    ----------
    articles : Mapping[str, EmpiricalCalibrationConfig]
        Configuration for each article to calibrate.
    candidate_multipliers : Iterable[float]
        Multipliers to try (e.g., [1.0, 1.1, 1.2, ..., 2.0]).
        Should be sorted in ascending order for efficiency.
    target_lost_sales_rate : float
        Target maximum lost sales rate (e.g., 0.05 for 5% lost sales).
    policy_mode : str
        Either "base_stock" or "rop" for reorder-point policy.

    Returns
    -------
    dict[str, EmpiricalCalibrationResult]
        Calibration results for each article.
    """
    multipliers = sorted(set(candidate_multipliers))
    if not multipliers:
        raise ValueError("Candidate multipliers must be provided.")
    if any(m < 0 for m in multipliers):
        raise ValueError("Multipliers must be non-negative.")
    if not 0.0 <= target_lost_sales_rate <= 1.0:
        raise ValueError("Target lost sales rate must be between 0 and 1.")

    results: dict[str, EmpiricalCalibrationResult] = {}

    for article_id, config in articles.items():
        best_result: EmpiricalCalibrationResult | None = None
        fallback_result: EmpiricalCalibrationResult | None = None

        for multiplier in multipliers:
            if policy_mode == "rop":
                candidate_policy = RopEmpiricalMultiplierPolicy(
                    forecast=config.forecast,
                    lead_time=config.lead_time,
                    multiplier=multiplier,
                )
            elif policy_mode == "base_stock":
                candidate_policy = EmpiricalMultiplierPolicy(
                    forecast=config.forecast,
                    lead_time=config.lead_time,
                    multiplier=multiplier,
                )
            else:
                raise ValueError("policy_mode must be 'base_stock' or 'rop'.")

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

            # Calculate lost sales rate
            total_demand = simulation.summary.total_demand
            total_fulfilled = simulation.summary.total_fulfilled
            if total_demand > 0:
                lost_sales_rate = (total_demand - total_fulfilled) / total_demand
            else:
                lost_sales_rate = 0.0

            simulation = _attach_metadata(
                simulation,
                service_level_factor=multiplier,
                service_level_mode="empirical_multiplier",
            )

            candidate = EmpiricalCalibrationResult(
                multiplier=multiplier,
                lost_sales_rate=lost_sales_rate,
                policy=candidate_policy,
                simulation=simulation,
            )

            # Track the best result that meets target
            if lost_sales_rate <= target_lost_sales_rate:
                if best_result is None or multiplier < best_result.multiplier:
                    best_result = candidate
                    # We found a valid result; since multipliers are sorted,
                    # we can stop here to get the smallest valid multiplier
                    break

            # Track fallback (lowest lost sales rate even if above target)
            if fallback_result is None or lost_sales_rate < fallback_result.lost_sales_rate:
                fallback_result = candidate

        # Use best result if found, otherwise use fallback with lowest lost sales
        if best_result is not None:
            results[article_id] = best_result
        elif fallback_result is not None:
            results[article_id] = fallback_result
        else:
            raise RuntimeError("No calibration result computed.")

    return results


def evaluate_empirical_multiplier_lost_sales(
    articles: Mapping[str, EmpiricalCalibrationConfig],
    candidate_multipliers: Iterable[float],
    *,
    policy_mode: str = "base_stock",
) -> dict[str, dict[float, float]]:
    """Return lost sales rates for each multiplier per article.

    This is useful for analyzing how different multipliers affect
    lost sales before selecting a target.
    """
    multipliers = list(candidate_multipliers)
    if not multipliers:
        raise ValueError("Candidate multipliers must be provided.")

    results: dict[str, dict[float, float]] = {}

    for article_id, config in articles.items():
        multiplier_rates: dict[float, float] = {}

        for multiplier in multipliers:
            if policy_mode == "rop":
                candidate_policy = RopEmpiricalMultiplierPolicy(
                    forecast=config.forecast,
                    lead_time=config.lead_time,
                    multiplier=multiplier,
                )
            elif policy_mode == "base_stock":
                candidate_policy = EmpiricalMultiplierPolicy(
                    forecast=config.forecast,
                    lead_time=config.lead_time,
                    multiplier=multiplier,
                )
            else:
                raise ValueError("policy_mode must be 'base_stock' or 'rop'.")

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

            total_demand = simulation.summary.total_demand
            total_fulfilled = simulation.summary.total_fulfilled
            if total_demand > 0:
                lost_sales_rate = (total_demand - total_fulfilled) / total_demand
            else:
                lost_sales_rate = 0.0

            multiplier_rates[multiplier] = lost_sales_rate

        results[article_id] = multiplier_rates

    return results
