"""Frozen single-item adapters for the optional rolling-horizon solver."""
from __future__ import annotations

from dataclasses import dataclass

from .item_policies import _integer
from .risk import RiskObjective
from .solver import SolverItemConfig, SolverOptions, _scenarios, _validate_cost_objective, plan_replenishment


def _source(policy, source, deterministic):
    _validate_cost_objective(policy.cost_objective)
    if policy.horizon is not None:
        _integer("horizon", policy.horizon, 1)
    if not isinstance(policy.config, SolverItemConfig):
        raise TypeError('config must be SolverItemConfig.')
    if not isinstance(policy.options, SolverOptions):
        raise TypeError('options must be SolverOptions.')
    if callable(source):
        if policy.horizon is None:
            raise ValueError('An explicit horizon is required for a forecast/scenario callback.')
        horizon = _integer('horizon', policy.horizon, 1)
    else:
        rows = [source] if deterministic else source
        paths, _, horizon = _scenarios({'item': rows}, ['item'])
        if policy.horizon is not None and policy.horizon != horizon:
            raise ValueError('Explicit horizon must match the static forecast horizon.')
        object.__setattr__(policy, 'forecast' if deterministic else 'demand_scenarios',
                           paths['item'][0] if deterministic else paths['item'])
    if horizon < max(1,policy.config.lead_time)+policy.config.review_period-1:
        raise ValueError('Planning horizon must cover the protection horizon.')
    object.__setattr__(policy, 'horizon', horizon)


class _ItemAdapter:
    @property
    def lead_time(self):
        return self.config.lead_time

    @property
    def review_period(self):
        return self.config.review_period

    @property
    def forecast_horizon(self):
        return self.horizon

    def order_quantity_for(self, state):
        if state.period % self.review_period:
            return 0
        return self.plan(state).first_orders['item']

    def _plan(self, state, source, objective, deterministic):
        values = source(state) if callable(source) else source
        rows = [values] if deterministic else values
        paths, _, horizon = _scenarios({'item': rows}, ['item'])
        if horizon != self.horizon:
            raise ValueError('Provider horizon must match configured horizon.')
        return plan_replenishment(states={'item': state}, items={'item': self.config},
                                  demand_scenarios=paths, objective=objective, options=self.options,
                                  cost_objective=self.cost_objective)


@dataclass(frozen=True)
class DeterministicRollingHorizonPolicy(_ItemAdapter):
    """Replan using an integer forecast of demand at t+1 through t+H.

    Static forecasts repeat at each decision. A callback receives the post-demand
    state and must supply as-of future forecasts without reading realized demand.
    """
    config: SolverItemConfig
    forecast: object
    horizon: int | None = None
    options: SolverOptions = SolverOptions()
    cost_objective: str = "total_cost"

    def __post_init__(self):
        _source(self, self.forecast, True)

    def plan(self, state):
        return self._plan(state, self.forecast, RiskObjective(), True)


@dataclass(frozen=True)
class ScenarioRollingHorizonPolicy(_ItemAdapter):
    """Replan a finite equal-weight scenario tree with nonanticipative actions."""
    config: SolverItemConfig
    demand_scenarios: object
    horizon: int | None = None
    objective: RiskObjective = RiskObjective()
    options: SolverOptions = SolverOptions()
    cost_objective: str = "total_cost"

    def __post_init__(self):
        if not isinstance(self.objective, RiskObjective):
            raise TypeError('objective must be RiskObjective.')
        _source(self, self.demand_scenarios, False)

    def plan(self, state):
        return self._plan(state, self.demand_scenarios, self.objective, False)


SOLVER_POLICY_TYPES = (DeterministicRollingHorizonPolicy, ScenarioRollingHorizonPolicy)


def solver_metadata(policy, *, coordinated=False, cost_objective=None):
    from .simulation import SimulationMetadata
    config = policy.config
    return SimulationMetadata(
        policy_family='coordinated_solver' if coordinated else 'solver_item',
        policy_name=type(policy).__name__, aggregation_window=1,
        review_period=config.review_period, forecast_horizon=policy.horizon,
        policy_parameters=tuple((name, getattr(config, name)) for name in (
            'lead_time', 'review_period', 'max_order_quantity', 'min_order_quantity',
            'pack_size', 'holding_cost_per_unit', 'stockout_cost_per_unit',
            'order_cost_per_order', 'order_cost_per_unit')) +
            (('horizon', policy.horizon), ('terminal_value', 0.0),
             ('cost_objective', policy.cost_objective if cost_objective is None else cost_objective)),
    )
