"""Synchronous multi-item simulation with jointly constrained solver decisions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .item_policies import _integer
from .risk import RiskObjective
from .simulation import (
    ArticleSimulationConfig, InventorySnapshot, InventoryState,
    SimulationResult, SimulationSummary,
)
from .solver import (
    COST_FIELDS, SharedResourceLimits, SolverOptions, RollingHorizonPlan,
    check_resource_use, plan_replenishment, _scenarios, _validate_cost_objective,
)
from .solver_policies import SOLVER_POLICY_TYPES, solver_metadata


@dataclass(frozen=True)
class ResourceUse:
    period: int
    order_units: int
    procurement_cost: float
    receiving_units: int


@dataclass(frozen=True)
class CoordinatedSimulationResult:
    simulations: Mapping[str, SimulationResult]
    plans: tuple[RollingHorizonPlan, ...]
    resource_use: tuple[ResourceUse, ...]


def simulate_coordinated_replenishment(
    *, articles: Mapping[str, ArticleSimulationConfig], demand_scenarios,
    limits: SharedResourceLimits = SharedResourceLimits(),
    objective: RiskObjective = RiskObjective(), options: SolverOptions = SolverOptions(),
    cost_objective: str = "total_cost",
) -> CoordinatedSimulationResult:
    """Advance all items, observe them together, then execute a joint first action.

    Configs must contain solver policies with matching simulation costs/lead times
    and a common horizon. Per-item forecast sources are unused here: the explicit
    joint scenario source defines the joint forecast. A callable source receives
    an immutable mapping of post-demand states, containing no future actuals.
    No initial pipeline is assumed, matching the ordinary simulation API.
    """
    _validate_cost_objective(cost_objective)
    articles = dict(articles)
    if not articles:
        raise ValueError('articles must not be empty.')
    lengths = set()
    horizons = set()
    items, demand_sources = {}, {}
    for i, article in articles.items():
        if not isinstance(article, ArticleSimulationConfig):
            raise TypeError('articles must contain ArticleSimulationConfig values.')
        periods = _integer('periods', article.periods, 1)
        lengths.add(periods)
        _integer('initial_on_hand', article.initial_on_hand)
        _integer('lead_time', article.lead_time)
        if not isinstance(article.policy, SOLVER_POLICY_TYPES):
            raise TypeError('Coordinated articles must contain solver policies.')
        item = article.policy.config
        if item.lead_time != article.lead_time:
            raise ValueError('Policy and simulation lead_time must match.')
        if any(getattr(item, name) != getattr(article, name) for name in COST_FIELDS):
            raise ValueError('Policy planning costs must match simulation costs in coordinated runs.')
        items[i] = item
        horizons.add(article.policy.horizon)
        demand_sources[i] = article.demand if callable(article.demand) else iter(article.demand)
    if len(lengths) != 1:
        raise ValueError('All articles must have equal simulation periods.')
    if len(horizons) != 1:
        raise ValueError('All coordinated policies must use a common horizon.')
    periods, horizon = next(iter(lengths)), next(iter(horizons))
    if not callable(demand_scenarios):
        demand_scenarios, _, provided_horizon = _scenarios(demand_scenarios, items)
        if provided_horizon != horizon:
            raise ValueError('Joint scenario horizon must match policy horizon.')
    stock = {i: a.initial_on_hand for i, a in articles.items()}
    pipeline = {i: {} for i in items}
    snapshots = {i: [] for i in items}
    plans, resources = [], []
    for period in range(periods):
        starts, received, states, demands = {}, {}, {}, {}
        for i in items:
            received[i] = pipeline[i].pop(period, 0)
            stock[i] += received[i]
            starts[i] = stock[i]
            source = demand_sources[i]
            try:
                value = source(period) if callable(source) else next(source)
            except StopIteration as exc:
                raise ValueError(f"Demand for {i!r} is shorter than simulation periods.") from exc
            demands[i] = _integer('demand', value)
            stock[i] = max(0, stock[i]-demands[i])
            states[i] = InventoryState(period, stock[i], sum(pipeline[i].values()), 0,
                                       tuple(sorted(pipeline[i].items())))
        immutable_states = MappingProxyType(states)
        source = demand_scenarios(immutable_states) if callable(demand_scenarios) else demand_scenarios
        source, _, provided_horizon = _scenarios(source, items)
        if provided_horizon != horizon:
            raise ValueError('Joint scenario horizon must match policy horizon.')
        plan = plan_replenishment(states=immutable_states, items=items,
            demand_scenarios=source, limits=limits, objective=objective, options=options,
            received_this_period=sum(received.values()), cost_objective=cost_objective)
        expense = sum((items[i].order_cost_per_order + q*items[i].order_cost_per_unit) if q else 0
                      for i, q in plan.first_orders.items())
        units = sum(plan.first_orders.values())
        incoming = sum(received.values()) + sum(q for i,q in plan.first_orders.items()
                                                if items[i].lead_time == 0)
        check_resource_use(units, expense, incoming, limits)
        for i, item in items.items():
            q = plan.first_orders[i]
            if item.lead_time == 0:
                stock[i] += q
            elif q:
                date = period+item.lead_time
                pipeline[i][date] = pipeline[i].get(date,0)+q
            snapshots[i].append(InventorySnapshot(
                period, starts[i], demands[i], received[i], stock[i], 0,
                q, sum(pipeline[i].values())))
        plans.append(plan)
        resources.append(ResourceUse(period, units, expense, incoming))
    results = {}
    for i, item in items.items():
        rows = tuple(snapshots[i])
        total_demand = sum(r.demand for r in rows)
        lost = sum(r.lost_sales for r in rows)
        on_hand = sum(r.ending_on_hand for r in rows)
        holding = on_hand*item.holding_cost_per_unit
        shortage = lost*item.stockout_cost_per_unit
        ordering = sum(item.order_cost_per_order + r.order_placed*item.order_cost_per_unit
                       for r in rows if r.order_placed)
        results[i] = SimulationResult(rows, SimulationSummary(
            total_demand, total_demand-lost, lost,
            (total_demand-lost)/total_demand if total_demand else 1., on_hand/periods,
            holding, shortage, ordering, holding+shortage+ordering),
            solver_metadata(articles[i].policy, coordinated=True, cost_objective=cost_objective))
    return CoordinatedSimulationResult(MappingProxyType(results), tuple(plans), tuple(resources))
