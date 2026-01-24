"""Inventory control policies."""

from __future__ import annotations

from dataclasses import dataclass

from .simulation import InventoryState


@dataclass(frozen=True)
class ReorderPointPolicy:
    """Order up to a fixed quantity when inventory position falls below a point."""

    reorder_point: int
    order_quantity: int

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.inventory_position <= self.reorder_point:
            return self.order_quantity
        return 0
