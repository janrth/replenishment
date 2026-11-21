"""Finite-horizon integer lost-sales planning, with an optional SciPy/HiGHS backend.

Scenarios start at the next demand event. Future actions are shared for identical
observed joint-demand prefixes; only the first action is executed before replanning.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from time import perf_counter
from types import MappingProxyType

from .item_policies import _integer
from .risk import RiskObjective, empirical_cvar, _real
from .simulation import InventoryState


@dataclass(frozen=True)
class SolverItemConfig:
    lead_time: int
    max_order_quantity: int
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0
    review_period: int = 1
    min_order_quantity: int = 0
    pack_size: int = 1

    def __post_init__(self):
        for name in ('lead_time', 'max_order_quantity', 'min_order_quantity'):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        for name in ('review_period', 'pack_size'):
            object.__setattr__(self, name, _integer(name, getattr(self, name), 1))
        for name in COST_FIELDS:
            object.__setattr__(self, name, _real(name, getattr(self, name), 0))
        if self.min_order_quantity > self.max_order_quantity:
            raise ValueError('min_order_quantity must not exceed max_order_quantity.')
        if (self.min_order_quantity and
            math.ceil(self.min_order_quantity / self.pack_size) * self.pack_size > self.max_order_quantity):
            raise ValueError('No positive pack multiple satisfies minimum and maximum order quantities.')


COST_FIELDS = ('holding_cost_per_unit', 'stockout_cost_per_unit',
               'order_cost_per_order', 'order_cost_per_unit')


@dataclass(frozen=True)
class SharedResourceLimits:
    """Per-period placed units, procurement expense, and received units."""
    max_order_units: float | None = None
    procurement_budget: float | None = None
    max_receiving_units: float | None = None

    def __post_init__(self):
        for name in ('max_order_units', 'procurement_budget', 'max_receiving_units'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _real(name, getattr(self, name), 0))


@dataclass(frozen=True)
class SolverOptions:
    time_limit: float = 30.0
    mip_rel_gap: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'time_limit', _real('time_limit', self.time_limit, 0))
        object.__setattr__(self, 'mip_rel_gap', _real('mip_rel_gap', self.mip_rel_gap, 0))
        if not self.time_limit:
            raise ValueError('time_limit must be positive.')


class PlanningError(RuntimeError):
    """No verified solution is available; no action should be executed."""


@dataclass(frozen=True)
class RollingHorizonPlan:
    first_orders: Mapping[str, int]
    orders: Mapping[str, tuple[tuple[int, ...], ...]]
    inventories: Mapping[str, tuple[tuple[int, ...], ...]]
    lost_sales: Mapping[str, tuple[tuple[int, ...], ...]]
    scenario_costs: tuple[float, ...]
    objective_value: float
    objective: RiskObjective
    mean_cost: float
    cvar_cost: float
    worst_cost: float
    status: str
    message: str
    mip_gap: float
    dual_bound: float
    solve_seconds: float
    horizon: int
    period: int
    terminal_value: float = 0.0
    received_this_period: int = 0
    cost_objective: str = "total_cost"


def _validate_cost_objective(mode):
    if mode not in {"total_cost", "lost_sales"}:
        raise ValueError('cost_objective must be "total_cost" or "lost_sales".')


def _validate_states(states, items):
    if not items or set(states) != set(items):
        raise ValueError('states and items must have identical nonempty keys.')
    if any(not isinstance(i, str) or not i for i in items):
        raise ValueError('Item identifiers must be nonempty strings.')
    periods = set()
    for name, config in items.items():
        if not isinstance(config, SolverItemConfig):
            raise TypeError('items must contain SolverItemConfig values.')
        state = states[name]
        if not isinstance(state, InventoryState):
            raise TypeError('states must contain InventoryState values.')
        periods.add(_integer('state.period', state.period))
        _integer('state.on_hand', state.on_hand)
        _integer('state.on_order', state.on_order)
        if state.backorders != 0:
            raise ValueError('Solver planning supports lost sales, not backlog.')
        total = 0
        seen = set()
        for due, quantity in state.arrival_schedule:
            _integer('arrival period', due)
            _integer('arrival quantity', quantity, 1)
            if due <= state.period or due > state.period + config.lead_time or due in seen:
                raise ValueError('arrival_schedule must contain distinct future dates within lead_time.')
            seen.add(due)
            total += quantity
        if total != state.on_order:
            raise ValueError('arrival_schedule quantities must sum to on_order.')
    if len(periods) != 1:
        raise ValueError('All states must refer to the same period.')
    return next(iter(periods))


def _scenarios(source, ids):
    if set(source) != set(ids):
        raise ValueError('demand_scenarios must match item keys.')
    paths = {i: tuple(tuple(_integer('scenario demand', d) for d in row)
                      for row in source[i]) for i in ids}
    shapes = {(len(rows), len(row)) for rows in paths.values() for row in rows}
    if any(not rows for rows in paths.values()) or len(shapes) != 1:
        raise ValueError('Scenarios must have equal nonempty scenario counts and horizons across items.')
    count, horizon = next(iter(shapes))
    if not horizon:
        raise ValueError('Scenario horizon must be positive.')
    return paths, count, horizon


class _Model:
    def __init__(self):
        self.lower, self.upper, self.integrality = [], [], []
        self.rows, self.row_lower, self.row_upper = [], [], []

    def variable(self, upper=math.inf, integer=False):
        index = len(self.lower)
        self.lower.append(0.)
        self.upper.append(float(upper))
        self.integrality.append(int(integer))
        return index

    def constraint(self, terms, lower=-math.inf, upper=math.inf):
        self.rows.append({j: v for j, v in terms.items() if v})
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def solve(self, objective, options):
        # Core imports, frozen policies and IO stay usable without SciPy.
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError('Solver planning requires pip install "replenishment[solver]".') from exc
        c = np.zeros(len(self.lower))
        for j, value in objective.items():
            c[j] = value
        row_indices, col_indices, data = [], [], []
        for k, row in enumerate(self.rows):
            for j, value in row.items():
                row_indices.append(k)
                col_indices.append(j)
                data.append(value)
        matrix = coo_matrix((data, (row_indices, col_indices)),
                            shape=(len(self.rows), len(c))).tocsc()
        if not np.isfinite(c).all() or not np.isfinite(matrix.data).all():
            raise ValueError('Model coefficients overflow; rescale input quantities/costs.')
        start = perf_counter()
        result = milp(c=c, integrality=np.array(self.integrality),
                      bounds=Bounds(self.lower, self.upper),
                      constraints=LinearConstraint(matrix, self.row_lower, self.row_upper),
                      options={'time_limit': options.time_limit, 'mip_rel_gap': options.mip_rel_gap})
        elapsed = perf_counter() - start
        if result.status != 0 or not result.success or result.x is None:
            status = {1: 'limit reached', 2: 'infeasible', 3: 'unbounded'}.get(result.status, 'solver error')
            raise PlanningError(f'{status}: {result.message}')
        gap = float(getattr(result, 'mip_gap', math.nan))
        bound = float(getattr(result, 'mip_dual_bound', math.nan))
        if (not math.isfinite(gap) or not math.isfinite(bound) or
            not math.isfinite(result.fun) or gap < 0 or
            gap > options.mip_rel_gap + 1e-7):
            raise PlanningError('Solver gap/bound failed requested optimality tolerance.')
        x = np.asarray(result.x, dtype=float)
        if x.shape != c.shape or not np.isfinite(x).all():
            raise PlanningError('Solver returned an invalid solution vector.')
        integral = np.asarray(self.integrality, dtype=bool)
        if np.any(np.abs(x[integral] - np.rint(x[integral])) > 1e-6):
            raise PlanningError('Solver returned noninteger order/setup decisions.')
        x[integral] = np.rint(x[integral])
        values = matrix @ x
        tolerance = 1e-5
        if (np.any(x < np.asarray(self.lower)-tolerance) or
            np.any(x > np.asarray(self.upper)+tolerance) or
            np.any(values < np.asarray(self.row_lower)-tolerance) or
            np.any(values > np.asarray(self.row_upper)+tolerance)):
            raise PlanningError('Solver solution failed feasibility validation.')
        return x, result, elapsed


def _add(expression, index, value):
    expression[index] = expression.get(index, 0.) + value


def _resource_constraints(model, items, states, q, setup, count, horizon, period, limits, received_this_period):
    for s in range(count):
        for k in range(horizon+1):
            units, expense, received = {}, {}, {}
            committed = sum(dict(states[i].arrival_schedule).get(period+k, 0) for i in items)
            if k == 0:
                committed += received_this_period
            for i, config in items.items():
                if k < horizon:
                    _add(units, q[i, s, k], config.pack_size)
                    _add(expense, q[i, s, k], config.pack_size * config.order_cost_per_unit)
                    _add(expense, setup[i, s, k], config.order_cost_per_order)
                placed = k - config.lead_time
                if 0 <= placed < horizon:
                    _add(received, q[i, s, placed], config.pack_size)
            for terms, limit, constant in (
                (units, limits.max_order_units, 0),
                (expense, limits.procurement_budget, 0),
                (received, limits.max_receiving_units, committed),
            ):
                if limit is not None:
                    if terms is expense:
                        scale = max([limit, *map(abs, terms.values())]) or 1.
                        model.constraint({j: value/scale for j,value in terms.items()},
                                         upper=limit/scale)
                    else:
                        model.constraint(terms, upper=math.floor(limit)-constant)


def plan_replenishment(*, states: Mapping[str, InventoryState],
                       items: Mapping[str, SolverItemConfig], demand_scenarios,
                       limits: SharedResourceLimits = SharedResourceLimits(),
                       objective: RiskObjective = RiskObjective(),
                       options: SolverOptions = SolverOptions(),
                       received_this_period: int = 0,
                       cost_objective: str = "total_cost") -> RollingHorizonPlan:
    """Solve a finite joint scenario tree and return verified first orders.

    Paths have equal weight and start at t+1. At decision t+k, only the first k
    demands are revealed. Terminal value is zero; unserviceable final orders are
    disabled. Constraints are per period, not cumulative resource budgets.
    cost_objective="lost_sales" excludes purchasing and holding from the score,
    while budget constraints still use actual fixed and variable purchasing costs.
    With shortage weights equal to selling prices, its mean maximizes revenue;
    total_cost with those weights instead maximizes expected profit.
    """
    if not isinstance(limits, SharedResourceLimits) or not isinstance(options, SolverOptions):
        raise TypeError('limits/options must be SharedResourceLimits/SolverOptions.')
    if not isinstance(objective, RiskObjective):
        raise TypeError('objective must be RiskObjective.')
    _validate_cost_objective(cost_objective)
    operating_weight = int(cost_objective == "total_cost")
    received_this_period = _integer("received_this_period", received_this_period)
    items, states = dict(items), dict(states)
    period = _validate_states(states, items)
    ids = tuple(items)
    paths, count, horizon = _scenarios(demand_scenarios, ids)
    for config in items.values():
        if horizon < max(1, config.lead_time) + config.review_period - 1:
            raise ValueError('Scenario horizon must cover each item protection horizon.')
    model = _Model()
    q, setup, inv, loss = {}, {}, {}, {}
    costs = [{} for _ in range(count)]
    # Share variable identities at every information node, not just stage zero.
    nodes = {}
    for s in range(count):
        for k in range(horizon):
            prefix = tuple(paths[i][s][:k] for i in ids)
            for i, config in items.items():
                key = (i, k, prefix)
                if key not in nodes:
                    eligible = ((period+k) % config.review_period == 0 and
                                k+max(1, config.lead_time) <= horizon)
                    cap = config.max_order_quantity // config.pack_size if eligible else 0
                    order = model.variable(cap, True)
                    active = model.variable(int(cap > 0), True)
                    model.constraint({order: 1, active: -cap}, upper=0)
                    minimum = max(1, math.ceil(config.min_order_quantity / config.pack_size))
                    model.constraint({order: 1, active: -minimum}, lower=0)
                    nodes[key] = (order, active)
                q[i, s, k], setup[i, s, k] = nodes[key]
                _add(costs[s], q[i, s, k], operating_weight * config.pack_size * config.order_cost_per_unit)
                _add(costs[s], setup[i, s, k], operating_weight * config.order_cost_per_order)
    for i, config in items.items():
        observation = states[i]
        stock_bound = observation.on_hand + observation.on_order + horizon * config.max_order_quantity
        if stock_bound > 2**45 or max(max(row) for row in paths[i]) > 2**45:
            raise ValueError('Quantities exceed the supported exact numerical range; rescale inputs.')
        for s in range(count):
            # I[k] is post-demand, post-order stock, including immediate orders.
            inv[i, s, 0] = model.variable(stock_bound)
            initial = {inv[i, s, 0]: 1}
            if config.lead_time == 0:
                _add(initial, q[i, s, 0], -config.pack_size)
                _add(costs[s], q[i, s, 0], operating_weight * config.pack_size * config.holding_cost_per_unit)
            model.constraint(initial, observation.on_hand, observation.on_hand)
            for k in range(1, horizon+1):
                d = paths[i][s][k-1]
                after_demand = model.variable(stock_bound)
                inv[i, s, k] = model.variable(stock_bound)
                loss[i, s, k] = model.variable(d)
                equation = {after_demand: 1, inv[i, s, k-1]: -1, loss[i, s, k]: -1}
                placed = k-config.lead_time
                if config.lead_time and 0 <= placed < horizon:
                    _add(equation, q[i, s, placed], -config.pack_size)
                receipt = dict(observation.arrival_schedule).get(period+k, 0)
                model.constraint(equation, receipt-d, receipt-d)
                if d:
                    shortage = model.variable(1, True)
                    model.constraint({after_demand: 1, shortage: stock_bound}, upper=stock_bound)
                    model.constraint({loss[i, s, k]: 1, shortage: -d}, upper=0)
                end = {inv[i, s, k]: 1, after_demand: -1}
                if config.lead_time == 0 and k < horizon:
                    _add(end, q[i, s, k], -config.pack_size)
                model.constraint(end, 0, 0)
                _add(costs[s], inv[i, s, k], operating_weight * config.holding_cost_per_unit)
                _add(costs[s], loss[i, s, k], config.stockout_cost_per_unit)
    _resource_constraints(model, items, states, q, setup, count, horizon, period, limits, received_this_period)
    # Normalize entire scenario costs before building risk epigraphs. Scaling
    # only the final objective leaves tiny monetary coefficients in CVaR rows.
    cost_scale = max((abs(v) for cost in costs for v in cost.values()), default=0.) or 1.
    normalized_costs = [{j:v/cost_scale for j,v in cost.items()} for cost in costs]
    target = {}
    mean_weight = 1. if objective.mode == 'mean' else (1-objective.risk_weight if objective.mode == 'mean_cvar' else 0.)
    tail_weight = 1-mean_weight
    if mean_weight:
        for expression in normalized_costs:
            for j, value in expression.items():
                _add(target, j, mean_weight * value / count)
    if tail_weight:
        threshold = model.variable()
        _add(target, threshold, tail_weight)
        for expression in normalized_costs:
            terms = dict(expression)
            _add(terms, threshold, -1)
            if objective.mode != 'worst_case':
                excess = model.variable()
                _add(terms, excess, -1)
                _add(target, excess, tail_weight / ((1-objective.alpha)*count))
            model.constraint(terms, upper=0)
    x, solved, elapsed = model.solve(target, options)
    orders = {i: tuple(tuple(int(round(x[q[i, s, k]]))*items[i].pack_size
                             for k in range(horizon)) for s in range(count)) for i in ids}
    inventory = {i: tuple(tuple(int(round(x[inv[i, s, k]])) for k in range(1,horizon+1))
                         for s in range(count)) for i in ids}
    losses = {i: tuple(tuple(int(round(x[loss[i, s, k]])) for k in range(1,horizon+1))
                      for s in range(count)) for i in ids}
    scenario_costs = tuple(math.fsum(value*x[j] for j, value in cost.items()) for cost in costs)
    score = objective.score(scenario_costs)
    scaled_objective = float(solved.fun)*cost_scale
    bound = float(solved.mip_dual_bound)*cost_scale
    precision = 1e-7*max(abs(score), cost_scale)
    if (not math.isfinite(score) or not math.isfinite(bound) or
        abs(score-scaled_objective) > precision):
        raise PlanningError('Solver objective failed risk-score reconciliation.')
    if (bound > score+precision or
        score-bound > options.mip_rel_gap*abs(score)+precision):
        raise PlanningError('Solver dual bound failed requested optimality tolerance.')
    _verify_replay(items, states, paths, orders, inventory, losses, scenario_costs, limits, received_this_period, operating_weight)
    return RollingHorizonPlan(
        first_orders=MappingProxyType({i: orders[i][0][0] for i in ids}),
        orders=MappingProxyType(orders), inventories=MappingProxyType(inventory),
        lost_sales=MappingProxyType(losses), scenario_costs=scenario_costs,
        objective_value=score, objective=objective,
        mean_cost=RiskObjective().score(scenario_costs),
        cvar_cost=empirical_cvar(scenario_costs,objective.alpha), worst_cost=max(scenario_costs),
        status='optimal', message=str(solved.message), mip_gap=float(solved.mip_gap),
        dual_bound=bound, solve_seconds=elapsed, horizon=horizon, period=period,
        received_this_period=received_this_period, cost_objective=cost_objective,
    )


def _verify_replay(items, states, paths, orders, inventories, losses, scenario_costs, limits, received_this_period, operating_weight):
    """Check executable integer trajectories independently of the MILP variables."""
    count, horizon = len(scenario_costs), len(next(iter(paths.values()))[0])
    for s in range(count):
        total = 0.
        receipts = {i: dict(states[i].arrival_schedule) for i in items}
        stock = {i: states[i].on_hand for i in items}
        for k in range(horizon+1):
            units, expense, received = 0, 0., (received_this_period if k == 0 else 0)
            for i, config in items.items():
                due = receipts[i].pop(states[i].period+k, 0)
                stock[i] += due
                received += due
                if k:
                    lost = max(0, paths[i][s][k-1]-stock[i])
                    stock[i] = max(0, stock[i]-paths[i][s][k-1])
                    total += lost*config.stockout_cost_per_unit
                    if lost != losses[i][s][k-1]:
                        raise PlanningError('Lost-sales trajectory failed independent replay.')
                quantity = orders[i][s][k] if k < horizon else 0
                if quantity:
                    if (quantity > config.max_order_quantity or quantity < config.min_order_quantity or
                        quantity % config.pack_size or (states[i].period+k) % config.review_period or
                        k+max(1,config.lead_time) > horizon):
                        raise PlanningError('Order failed feasibility replay.')
                    charge = config.order_cost_per_order+quantity*config.order_cost_per_unit
                    total += operating_weight * charge
                    expense += charge
                    units += quantity
                    if config.lead_time:
                        date = states[i].period+k+config.lead_time
                        receipts[i][date] = receipts[i].get(date,0)+quantity
                    else:
                        received += quantity
                        stock[i] += quantity
                if k:
                    total += operating_weight * stock[i]*config.holding_cost_per_unit
                    if stock[i] != inventories[i][s][k-1]:
                        raise PlanningError('Inventory trajectory failed independent replay.')
                elif config.lead_time == 0:
                    total += operating_weight * quantity*config.holding_cost_per_unit
            check_resource_use(units, expense, received, limits)
        if not math.isclose(total, scenario_costs[s], rel_tol=1e-7, abs_tol=1e-5):
            raise PlanningError('Scenario cost failed independent replay.')


def check_resource_use(units, expense, received, limits):
    for name, value, cap in (
        ('order units', units, limits.max_order_units),
        ('procurement budget', expense, limits.procurement_budget),
        ('receiving units', received, limits.max_receiving_units),
    ):
        if cap is not None:
            # Unit counts are integers: an absolute epsilon may admit a whole
            # extra unit at a fractional cap. Money permits only float roundoff,
            # never a fixed monetary epsilon, including when budget is zero.
            tolerance = (8*max(math.ulp(value),math.ulp(cap))
                         if name == 'procurement budget' and cap > 0 else 0.)
            if value > cap+tolerance:
                raise PlanningError(f'Executed {name} exceeds its shared limit: {value} > {cap}.')
