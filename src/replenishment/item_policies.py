"""Independent-item inventory policies using post-demand decisions.

Review dates are periods 0, R, 2R, ...; demand is never aggregated by a
policy. Targets and caps are decision parameters, not shared physical limits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType

from .simulation import InventoryState


def _integer(name: str, value: int, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _set_integer(policy, name: str, minimum: int | None = 0) -> None:
    object.__setattr__(policy, name, _integer(name, getattr(policy, name), minimum))


def _inventory_position(state: InventoryState) -> int:
    # Convert components before addition; fixed-width unsigned sums can wrap.
    return _integer("state.on_hand", state.on_hand) + _integer("state.on_order", state.on_order)


@dataclass(frozen=True)
class BaseStockPolicy:
    """Order up to ``target_stock``, counting the outstanding pipeline once."""

    target_stock: int
    review_period: int = 1

    def __post_init__(self) -> None:
        _set_integer(self, "target_stock")
        _set_integer(self, "review_period", 1)

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.period % self.review_period:
            return 0
        return max(0, self.target_stock - _inventory_position(state))


@dataclass(frozen=True)
class ConstantOrderPolicy:
    """Order a fixed quantity on review dates, regardless of stock."""

    quantity: int
    review_period: int = 1

    def __post_init__(self) -> None:
        _set_integer(self, "quantity")
        _set_integer(self, "review_period", 1)

    def order_quantity_for(self, state: InventoryState) -> int:
        return 0 if state.period % self.review_period else int(self.quantity)


@dataclass(frozen=True)
class CappedBaseStockPolicy:
    """Base-stock ordering limited by a tuned per-order policy ``cap``."""

    target_stock: int
    cap: int
    review_period: int = 1

    def __post_init__(self) -> None:
        _set_integer(self, "target_stock")
        _set_integer(self, "cap")
        _set_integer(self, "review_period", 1)

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.period % self.review_period:
            return 0
        return min(self.cap, max(0, self.target_stock - _inventory_position(state)))


@dataclass(frozen=True)
class sSPolicy:
    """Order up to S when inventory position <= s, including equality.

    ``reorder_point`` (s) and ``target_stock`` (S) are independently tunable;
    S must be nonnegative and >= s. Negative reorder points are allowed.
    Unlike an (r,Q) rule, the order size depends on inventory position.
    """

    reorder_point: int
    target_stock: int
    review_period: int = 1

    def __post_init__(self) -> None:
        _set_integer(self, "reorder_point", None)
        _set_integer(self, "target_stock")
        _set_integer(self, "review_period", 1)
        if self.target_stock < self.reorder_point:
            raise ValueError("target_stock must be >= reorder_point.")

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.period % self.review_period:
            return 0
        position = _inventory_position(state)
        return max(0, self.target_stock - position) if position <= self.reorder_point else 0


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one.")


@dataclass(frozen=True)
class ProjectedInventoryLevelPolicy:
    """Raise expected stock just before arrival demand to ``target_stock``.

    ``demand_pmf`` describes independent, identically distributed nonnegative
    integer demand per period. Exact probability propagation accounts for
    lost sales and dated receipts. Its state space can grow substantially for
    large inventory levels, lead times or demand supports; use modest models.

    The PMF is copied and made immutable; probabilities must sum to one within
    absolute tolerance 1e-12. Review dates are 0, R, 2R, ... . No realized future
    demand is read. This implements PIL under this simulator's timing, without
    claiming its cost is globally optimal or reproduces paper benchmarks.
    """

    target_stock: int
    demand_pmf: Mapping[int, float]
    lead_time: int
    review_period: int = 1

    def __post_init__(self) -> None:
        _set_integer(self, "target_stock")
        _set_integer(self, "lead_time")
        _set_integer(self, "review_period", 1)
        if not isinstance(self.demand_pmf, Mapping) or not self.demand_pmf:
            raise ValueError("demand_pmf must be a nonempty mapping.")
        pmf = {}
        for demand, probability in self.demand_pmf.items():
            _integer("demand support", demand)
            _probability("demand probability", probability)
            if probability > 0:
                pmf[int(demand)] = float(probability)
        total = math.fsum(pmf.values())
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("demand_pmf probabilities must sum to one.")
        object.__setattr__(self, "demand_pmf", MappingProxyType({
            demand: probability / total for demand, probability in sorted(pmf.items())
        }))

    def projected_inventory(self, state: InventoryState) -> float:
        """Expected inventory before demand at the new order's arrival.

        Receipts at that arrival are included; arrival-period demand is not.
        For L=0 or L=1 there is no intervening demand after this observation.
        Inconsistent externally constructed receipt schedules are rejected.
        """
        distribution = self._arrival_distribution(state)
        mass = math.fsum(distribution.values())
        reference = min(distribution)
        return reference + math.fsum((stock - reference) * p for stock, p in distribution.items()) / mass

    def _arrival_distribution(self, state: InventoryState) -> dict[int, float]:
        period_now = _integer("state.period", state.period)
        on_hand = _integer("state.on_hand", state.on_hand)
        on_order = _integer("state.on_order", state.on_order)
        if state.backorders != 0:
            raise ValueError("PIL requires lost-sales observations with no backlog.")
        arrival = period_now + self.lead_time
        schedule = {}
        previous = period_now
        for receipt_period, quantity in state.arrival_schedule:
            receipt_period = _integer("arrival period", receipt_period)
            quantity = _integer("arrival quantity", quantity, 1)
            if receipt_period <= previous or receipt_period > arrival:
                raise ValueError("arrival_schedule must be ordered after the observation and no later than the new arrival.")
            schedule[receipt_period] = quantity
            previous = receipt_period
        if sum(schedule.values()) != on_order:
            raise ValueError("arrival_schedule quantities must sum to on_order.")

        distribution = {on_hand: 1.0}
        for period in range(period_now + 1, arrival):
            receipt = schedule.get(period, 0)
            next_distribution: dict[int, float] = defaultdict(float)
            for stock, probability in distribution.items():
                for demand, demand_probability in self.demand_pmf.items():
                    next_stock = max(0, stock + receipt - demand)
                    next_distribution[next_stock] += probability * demand_probability
            distribution = next_distribution
        receipt = schedule.get(arrival, 0)
        return {stock + receipt: probability for stock, probability in distribution.items()}

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.period % self.review_period:
            return 0
        distribution = self._arrival_distribution(state)
        mass = math.fsum(distribution.values())
        # Center before taking expectations: S-E[I] loses precision when S
        # and E[I] are large and close, potentially rounding an extra unit.
        gap = max(0.0, math.fsum((self.target_stock - stock) * p for stock, p in distribution.items()) / mass)
        # Remove only absolute floating-point noise at integer boundaries.
        nearest = round(gap)
        if math.isclose(gap, nearest, rel_tol=0, abs_tol=1e-12):
            gap = nearest
        return math.ceil(gap)


AggregateSamples = Iterable[int] | Callable[[InventoryState], Iterable[int]]
DemandPaths = Iterable[Iterable[int]] | Callable[[InventoryState], Iterable[Iterable[int]]]


@dataclass(frozen=True)
class ProtectionPeriodQuantilePolicy:
    """Order up to an empirical aggregate-demand quantile (a lost-sales heuristic).

    Supply exactly one of ``aggregate_samples`` or ``demand_paths``. Static
    iterables are copied to immutable tuples; callbacks receive the post-demand
    state and supply as-of samples on each review. Providers must use only
    information available at the decision; the simulator does not expose future
    demand or provide online forecast training/history. Paths preserve dependence
    and must each contain H=max(1,L)+R-1 future periods, starting at t+1.

    Targets use the empirical inverse CDF: the first sample with CDF >= q.
    There is no interpolation or summing of marginal percentile forecasts.
    Quantile coverage is distinct from achieved inventory service/fill rate.
    """

    quantile: float
    lead_time: int
    review_period: int = 1
    aggregate_samples: AggregateSamples | None = None
    demand_paths: DemandPaths | None = None

    def __post_init__(self) -> None:
        _probability("quantile", self.quantile)
        object.__setattr__(self, "quantile", float(self.quantile))
        _set_integer(self, "lead_time")
        _set_integer(self, "review_period", 1)
        if (self.aggregate_samples is None) == (self.demand_paths is None):
            raise ValueError("Supply exactly one of aggregate_samples or demand_paths.")
        if self.aggregate_samples is not None and not callable(self.aggregate_samples):
            object.__setattr__(self, "aggregate_samples", self._samples(self.aggregate_samples))
        if self.demand_paths is not None and not callable(self.demand_paths):
            object.__setattr__(self, "demand_paths", self._paths(self.demand_paths))

    @property
    def protection_period(self) -> int:
        return max(1, self.lead_time) + self.review_period - 1

    @staticmethod
    def _samples(values: Iterable[int]) -> tuple[int, ...]:
        samples = tuple(values)
        if not samples:
            raise ValueError("Demand samples must be nonempty.")
        return tuple(_integer("demand sample", sample) for sample in samples)

    def _paths(self, values: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
        paths = tuple(self._samples(path) for path in values)
        if not paths:
            raise ValueError("Demand paths must be nonempty.")
        for path in paths:
            if len(path) != self.protection_period:
                raise ValueError("Each demand path must match the protection horizon.")
        return paths

    def target_for(self, state: InventoryState) -> int:
        """Compute the integer inverse-CDF target from current as-of samples."""
        if self.aggregate_samples is not None:
            source = self.aggregate_samples
            values = self._samples(source(state) if callable(source) else source)
        else:
            source = self.demand_paths
            paths = self._paths(source(state) if callable(source) else source)
            values = tuple(sum(path) for path in paths)
        ordered = sorted(values)
        # Compare empirical CDF values directly: q*n can round just above an
        # exact rank (e.g. .14*100), which would incorrectly select the next unit.
        low, high = 0, len(ordered) - 1
        while low < high:
            middle = (low + high) // 2
            if (middle + 1) / len(ordered) >= self.quantile:
                high = middle
            else:
                low = middle + 1
        return ordered[low]

    def order_quantity_for(self, state: InventoryState) -> int:
        if state.period % self.review_period:
            return 0
        return max(0, self.target_for(state) - _inventory_position(state))


from .neural import StructuredNeuralPolicy


_ITEM_POLICY_TYPES = (
    BaseStockPolicy, ConstantOrderPolicy, CappedBaseStockPolicy, sSPolicy,
    ProjectedInventoryLevelPolicy, ProtectionPeriodQuantilePolicy,
    StructuredNeuralPolicy,
)


def _policy_parameters(policy) -> tuple[tuple[str, object], ...]:
    """Compact immutable decision parameters for built-in simulation metadata."""
    parameters = [(name, getattr(policy, name)) for name in (
        "target_stock", "quantity", "cap", "reorder_point", "lead_time", "review_period", "quantile",
    ) if hasattr(policy, name)]
    if isinstance(policy, StructuredNeuralPolicy):
        parameters.extend((name, getattr(policy.config, name)) for name in (
            "target_stock", "residual_scale", "feature_scale", "hidden_size", "max_order_quantity", "use_forecast_target",
        ))
    if isinstance(policy, ProjectedInventoryLevelPolicy):
        parameters.append(("demand_pmf", tuple(policy.demand_pmf.items())))
    if isinstance(policy, ProtectionPeriodQuantilePolicy):
        parameters.append(("protection_period", policy.protection_period))
        source = policy.aggregate_samples if policy.aggregate_samples is not None else policy.demand_paths
        parameters.append(("sample_source", "provider" if callable(source) else "static"))
        if not callable(source):
            parameters.append(("sample_count", len(source)))
    return tuple(parameters)
