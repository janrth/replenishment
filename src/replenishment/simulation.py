"""Core simulation primitives for replenishment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol


DemandModel = Callable[[int], int]


class OrderingPolicy(Protocol):
    def order_quantity_for(self, state: "InventoryState") -> int:
        ...


@dataclass(frozen=True)
class InventoryState:
    period: int
    on_hand: int
    on_order: int
    backorders: int

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order - self.backorders


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


@dataclass(frozen=True)
class SimulationSummary:
    total_demand: int
    total_fulfilled: int
    total_backorders: int
    fill_rate: float
    average_on_hand: float


@dataclass(frozen=True)
class SimulationResult:
    snapshots: Sequence[InventorySnapshot]
    summary: SimulationSummary


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
) -> SimulationResult:
    if periods <= 0:
        raise ValueError("Periods must be positive.")
    if lead_time < 0:
        raise ValueError("Lead time cannot be negative.")

    demand_model = _normalize_demand(demand)
    on_hand = initial_on_hand
    backorders = 0
    pipeline: list[int] = [0 for _ in range(lead_time)]
    snapshots: list[InventorySnapshot] = []

    total_demand = 0
    total_fulfilled = 0
    total_backorders = 0
    on_hand_total = 0

    for period in range(periods):
        received = pipeline.pop(0) if lead_time > 0 else 0
        on_hand += received
        if backorders:
            fulfilled_backorders = min(on_hand, backorders)
            on_hand -= fulfilled_backorders
            backorders -= fulfilled_backorders

        period_demand = demand_model(period)
        total_demand += period_demand

        fulfilled = min(on_hand, period_demand)
        on_hand -= fulfilled
        unmet = period_demand - fulfilled
        if unmet:
            backorders += unmet
            total_backorders += unmet

        on_order = sum(pipeline)
        state = InventoryState(
            period=period,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
        )
        order_qty = max(0, policy.order_quantity_for(state))
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
                starting_on_hand=on_hand + fulfilled,
                demand=period_demand,
                received=received,
                ending_on_hand=on_hand,
                backorders=backorders,
                order_placed=order_qty,
                on_order=on_order,
            )
        )

    fill_rate = total_fulfilled / total_demand if total_demand else 1.0
    average_on_hand = on_hand_total / periods

    summary = SimulationSummary(
        total_demand=total_demand,
        total_fulfilled=total_fulfilled,
        total_backorders=total_backorders,
        fill_rate=fill_rate,
        average_on_hand=average_on_hand,
    )

    return SimulationResult(snapshots=snapshots, summary=summary)
