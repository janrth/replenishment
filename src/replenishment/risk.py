"""Solver-free empirical risk objectives for independent-item scenarios.

Scenarios have equal probability. Worst-case scores protect only against the
supplied scenarios; this is not a distributionally robust guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
import math
from numbers import Real
from types import MappingProxyType

from .item_policies import _integer
from .simulation import DemandModel, OrderingPolicy, simulate_replenishment


def _real(name: str, value: float, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}." if minimum is not None else "."))
    return value


def _losses(costs: Iterable[float]) -> tuple[float, ...]:
    values = tuple(_real("scenario cost", value) for value in costs)
    if not values:
        raise ValueError("Scenario costs must not be empty.")
    return values


def _weighted_mean(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Scale before summing to prevent overflow; compensate signed cancellation."""
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    if weights is None:
        return (math.fsum(value / scale for value in values) / len(values)) * scale
    return (math.fsum((value / scale) * weight for value, weight in zip(values, weights))
            / math.fsum(weights)) * scale


def _tail_mass(count: int, alpha: float) -> tuple[int, float, float]:
    mass = (1 - alpha) * count
    whole = math.floor(mass)
    return whole, mass - whole, mass


def _confidence(alpha: float) -> float:
    alpha = _real("alpha", alpha, 0)
    if alpha >= 1:
        raise ValueError("alpha must be less than one.")
    return alpha


def empirical_cvar(costs: Iterable[float], alpha: float = .9) -> float:
    """Average the worst (1-alpha) mass of equal-weight costs.

    A fractional boundary observation contributes only its tail fraction.
    alpha=0 is the mean; alpha must be <1. Costs may be any finite real values.
    """
    alpha = _confidence(alpha)
    values = sorted(_losses(costs), reverse=True)
    whole, fraction, mass = _tail_mass(len(values), alpha)
    selected, weights = values[:whole], [1.0] * whole
    if fraction and whole < len(values):
        selected.append(values[whole])
        weights.append(fraction)
    return _weighted_mean(selected, weights)


@dataclass(frozen=True)
class RiskObjective:
    """Mean, upper-tail CVaR, mean/CVaR blend, or finite-scenario worst cost."""

    mode: str = "mean"
    alpha: float = .9
    risk_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"mean", "cvar", "mean_cvar", "worst_case"}:
            raise ValueError("mode must be mean, cvar, mean_cvar or worst_case.")
        object.__setattr__(self, "alpha", _confidence(self.alpha))
        weight = _real("risk_weight", self.risk_weight, 0)
        if weight > 1:
            raise ValueError("risk_weight must be <= 1.")
        object.__setattr__(self, "risk_weight", weight)

    def score(self, costs: Iterable[float]) -> float:
        values = _losses(costs)
        if self.mode == "mean":
            return _weighted_mean(values)
        if self.mode == "worst_case":
            return max(values)
        tail = empirical_cvar(values, self.alpha)
        if self.mode == "cvar":
            return tail
        return _weighted_mean((_weighted_mean(values), tail), (1 - self.risk_weight, self.risk_weight))


@dataclass(frozen=True)
class RiskPolicyOptimizationResult:
    label: str
    policy: OrderingPolicy
    objective: RiskObjective
    score: float
    candidate_scores: Mapping[str, float]
    scenario_costs: tuple[float, ...]
    mean_cost: float
    cvar_cost: float
    worst_cost: float


def _scenario_traces(
    scenarios: Iterable[Iterable[int] | DemandModel], periods: int,
) -> tuple[tuple[int, ...], ...]:
    periods = _integer("periods", periods, 1)
    traces = []
    for source in scenarios:
        values = [source(t) for t in range(periods)] if callable(source) else list(islice(source, periods))
        if len(values) != periods:
            raise ValueError("Each demand scenario must contain the requested periods.")
        traces.append(tuple(_integer("demand", value) for value in values))
    if not traces:
        raise ValueError("Demand scenarios must not be empty.")
    return tuple(traces)


def optimize_risk_sensitive_policy_candidates(
    *, candidates: Mapping[str, OrderingPolicy],
    demand_scenarios: Iterable[Iterable[int] | DemandModel], periods: int,
    initial_on_hand: int, lead_time: int,
    objective: RiskObjective = RiskObjective(),
    holding_cost_per_unit: float = 0.0, stockout_cost_per_unit: float = 0.0,
    order_cost_per_order: float = 0.0, order_cost_per_unit: float = 0.0,
) -> RiskPolicyOptimizationResult:
    """Select a reusable policy independently for one item across offline scenarios.

    Materializes only ``periods`` demand values per scenario and replays exactly
    those traces for every candidate. Scenarios must be a finite outer iterable.
    All distributions/forecasts/parameters must be prepared without held-out
    demand. Positive fixed costs are supported through the exact simulator.
    """
    if not candidates:
        raise ValueError("candidates must not be empty.")
    if not isinstance(objective, RiskObjective):
        raise TypeError("objective must be a RiskObjective.")
    periods = _integer("periods", periods, 1)
    initial_on_hand = _integer("initial_on_hand", initial_on_hand)
    lead_time = _integer("lead_time", lead_time)
    costs = {name: _real(name, value, 0) for name, value in dict(
        holding_cost_per_unit=holding_cost_per_unit, stockout_cost_per_unit=stockout_cost_per_unit,
        order_cost_per_order=order_cost_per_order, order_cost_per_unit=order_cost_per_unit,
    ).items()}
    traces = _scenario_traces(demand_scenarios, periods)
    best = None
    scores = {}
    for label, policy in candidates.items():
        scenario_costs = tuple(simulate_replenishment(
            periods=periods, demand=trace, initial_on_hand=initial_on_hand,
            lead_time=lead_time, policy=policy, **costs,
        ).summary.total_cost for trace in traces)
        score = objective.score(scenario_costs)
        scores[label] = score
        if best is None or score < best[2]:
            best = (label, policy, score, scenario_costs)
    label, policy, score, scenario_costs = best
    return RiskPolicyOptimizationResult(
        label, policy, objective, score, MappingProxyType(scores), scenario_costs,
        _weighted_mean(scenario_costs),
        empirical_cvar(scenario_costs, objective.alpha), max(scenario_costs),
    )
