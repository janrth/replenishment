"""Core simulation primitives for replenishment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol


DemandModel = Callable[[int], int]


class OrderingPolicy(Protocol):
    def order_quantity_for(self, state: "InventoryState") -> int:
        ...


@dataclass(frozen=True)
class InventoryState:
    """Post-demand, pre-order observation in a lost-sales system.

    ``arrival_schedule`` contains positive outstanding receipts as immutable
    ``(absolute_period, quantity)`` pairs. It excludes the order being decided.
    Simulator observations always sum to ``on_order``.
    """

    period: int
    on_hand: int
    on_order: int
    backorders: int
    arrival_schedule: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrival_schedule", tuple(
            (period, quantity) for period, quantity in self.arrival_schedule
        ))

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


@dataclass(frozen=True)
class InventorySnapshot:
    period: int
    starting_on_hand: int
    demand: int
    received: int
    ending_on_hand: int
    backorders: int
    order_placed: int
    on_order: int

    @property
    def lost_sales(self) -> int:
        """Demand lost this period; immediate orders arrive after demand."""
        return max(0, self.demand - self.starting_on_hand)


@dataclass(frozen=True)
class SimulationSummary:
    total_demand: int
    total_fulfilled: int
    total_backorders: int
    fill_rate: float
    average_on_hand: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    total_cost: float

    @property
    def total_lost_sales(self) -> int:
        """Unfulfilled demand, historically named ``total_backorders``."""
        return self.total_backorders


@dataclass(frozen=True)
class SimulationMetadata:
    service_level_factor: float | None = None
    service_level_mode: str | None = None
    safety_stock_method: str | None = None
    aggregation_window: int | None = None
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    percentile_target: float | str | None = None
    policy_name: str | None = None
    policy_parameters: tuple[tuple[str, object], ...] = ()
    policy_family: str | None = None


@dataclass(frozen=True)
class SimulationResult:
    snapshots: Sequence[InventorySnapshot]
    summary: SimulationSummary
    metadata: SimulationMetadata | None = None


@dataclass(frozen=True)
class ArticleSimulationConfig:
    periods: int
    demand: Iterable[int] | DemandModel
    initial_on_hand: int
    lead_time: int
    policy: OrderingPolicy
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0


def _normalize_demand(demand: Iterable[int] | DemandModel) -> DemandModel:
    if callable(demand):
        return demand

    demand_list = list(demand)

    def demand_model(period: int) -> int:
        if period < 0 or period >= len(demand_list):
            raise IndexError("Demand period out of range.")
        return demand_list[period]

    return demand_model


def simulate_replenishment(
    *,
    periods: int,
    demand: Iterable[int] | DemandModel,
    initial_on_hand: int,
    lead_time: int,
    policy: OrderingPolicy,
    holding_cost_per_unit: float = 0.0,
    stockout_cost_per_unit: float = 0.0,
    order_cost_per_order: float = 0.0,
    order_cost_per_unit: float = 0.0,
) -> SimulationResult:
    """Simulate receipts, demand, then ordering, with unmet demand lost.

    An order placed at t arrives at t+lead_time for positive lead times.
    At lead time zero it arrives after demand at t. Lost demand is never
    backlogged, and outstanding stock is excluded from period holding cost.
    There is no terminal salvage value or post-horizon fulfillment.
    """
    # Local import preserves the public custom-policy protocol and avoids a
    # module cycle: built-in policies consume InventoryState from this module.
    from .item_policies import (
        ProjectedInventoryLevelPolicy, ProtectionPeriodQuantilePolicy,
        _ITEM_POLICY_TYPES, _policy_parameters,
        _integer,
        StructuredNeuralPolicy,
    )

    from .solver_policies import SOLVER_POLICY_TYPES, solver_metadata

    solver_policy = isinstance(policy, SOLVER_POLICY_TYPES)
    item_policy = isinstance(policy, _ITEM_POLICY_TYPES) or solver_policy
    if item_policy:
        periods = _integer("periods", periods, 1)
        lead_time = _integer("lead_time", lead_time)
        initial_on_hand = _integer("initial_on_hand", initial_on_hand)
        for name, value in (
            ("holding_cost_per_unit", holding_cost_per_unit),
            ("stockout_cost_per_unit", stockout_cost_per_unit),
            ("order_cost_per_order", order_cost_per_order),
            ("order_cost_per_unit", order_cost_per_unit),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
    if solver_policy or isinstance(policy, (ProjectedInventoryLevelPolicy, ProtectionPeriodQuantilePolicy, StructuredNeuralPolicy)):
        if policy.lead_time != lead_time:
            raise ValueError("Policy lead_time must match simulation lead_time.")
    if periods <= 0:
        raise ValueError("Periods must be positive.")
    if lead_time < 0:
        raise ValueError("Lead time cannot be negative.")
    if (
        holding_cost_per_unit < 0
        or stockout_cost_per_unit < 0
        or order_cost_per_order < 0
        or order_cost_per_unit < 0
    ):
        raise ValueError("Cost inputs must be non-negative.")

    demand_model = _normalize_demand(demand)
    on_hand = initial_on_hand
    backorders = 0
    pipeline: list[int] = [0 for _ in range(lead_time)]
    snapshots: list[InventorySnapshot] = []

    total_demand = 0
    total_fulfilled = 0
    total_backorders = 0
    on_hand_total = 0
    ordering_cost_total = 0.0

    for period in range(periods):
        received = pipeline.pop(0) if lead_time > 0 else 0
        on_hand += received
        starting_on_hand = on_hand
        period_demand = demand_model(period)
        if item_policy:
            period_demand = _integer("demand", period_demand)
        total_demand += period_demand

        fulfilled = min(on_hand, period_demand)
        on_hand -= fulfilled
        unmet = period_demand - fulfilled
        if unmet:
            total_backorders += unmet

        on_order = sum(pipeline)
        state = InventoryState(
            period=period,
            on_hand=on_hand,
            on_order=on_order,
            backorders=0,
            arrival_schedule=tuple(
                (period + offset + 1, quantity)
                for offset, quantity in enumerate(pipeline)
                if quantity > 0
            ),
        )
        order_qty = max(0, policy.order_quantity_for(state))
        if order_qty > 0:
            ordering_cost_total += order_cost_per_order + (
                order_cost_per_unit * order_qty
            )
        if lead_time == 0:
            on_hand += order_qty
        else:
            pipeline.append(order_qty)

        on_order = sum(pipeline)
        total_fulfilled += fulfilled
        on_hand_total += on_hand

        snapshots.append(
            InventorySnapshot(
                period=period,
                starting_on_hand=starting_on_hand,
                demand=period_demand,
                received=received,
                ending_on_hand=on_hand,
                backorders=0,
                order_placed=order_qty,
                on_order=on_order,
            )
        )

    fill_rate = total_fulfilled / total_demand if total_demand else 1.0
    average_on_hand = on_hand_total / periods
    holding_cost = on_hand_total * holding_cost_per_unit
    stockout_cost = total_backorders * stockout_cost_per_unit
    total_cost = holding_cost + stockout_cost + ordering_cost_total

    summary = SimulationSummary(
        total_demand=total_demand,
        total_fulfilled=total_fulfilled,
        total_backorders=total_backorders,
        fill_rate=fill_rate,
        average_on_hand=average_on_hand,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        ordering_cost=ordering_cost_total,
        total_cost=total_cost,
    )

    metadata = None
    if item_policy:
        metadata = SimulationMetadata(
            policy_family="independent_item",
            policy_name=type(policy).__name__,
            policy_parameters=_policy_parameters(policy),
            aggregation_window=1,
            review_period=policy.review_period,
            forecast_horizon=(
                policy.protection_period
                if isinstance(policy, ProtectionPeriodQuantilePolicy)
                else policy.forecast_horizon if isinstance(policy, StructuredNeuralPolicy) else None
            ),
        )
    if solver_policy:
        metadata = solver_metadata(policy)
    return SimulationResult(snapshots=snapshots, summary=summary, metadata=metadata)


def simulate_replenishment_for_articles(
    articles: Mapping[str, ArticleSimulationConfig],
) -> dict[str, SimulationResult]:
    results: dict[str, SimulationResult] = {}
    for article_id, config in articles.items():
        results[article_id] = simulate_replenishment(
            periods=config.periods,
            demand=config.demand,
            initial_on_hand=config.initial_on_hand,
            lead_time=config.lead_time,
            policy=config.policy,
            holding_cost_per_unit=config.holding_cost_per_unit,
            stockout_cost_per_unit=config.stockout_cost_per_unit,
            order_cost_per_order=config.order_cost_per_order,
            order_cost_per_unit=config.order_cost_per_unit,
        )
    return results
