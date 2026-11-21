"""Single-item hindsight differentiable policy optimisation (optional Torch).

Training differentiates a continuous-action lost-sales simulator. Deployment and
checkpoint selection use the exact integer Python simulator. Positive fixed
ordering costs are rejected: their indicator is not a pathwise differentiable
cost. This is not a claim of globally optimal or unbiased discrete optimisation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import time

from .item_policies import _integer
from .neural import StructuredNeuralPolicy, _model_from_policy, _policy_from_model, _require_torch, _bounded_tensor_order
from .risk import RiskObjective, _real, _tail_mass
from .simulation import InventoryState, simulate_replenishment


def _tensor_weighted_mean(values, weights=None):
    """Overflow-safe scaling plus differentiable Neumaier compensated summation."""
    torch = _require_torch()
    if weights is None:
        weights = values.new_ones(values.numel())
    scale = values.detach().abs().max()
    scaled = values if scale.item() == 0 else values / scale
    total, compensation = values.new_zeros(()), values.new_zeros(())
    for term in scaled * weights:
        updated = total + term
        compensation = compensation + torch.where(
            total.abs() >= term.abs(), (total - updated) + term, (term - updated) + total,
        )
        total = updated
    mean = (total + compensation) / weights.sum()
    return mean if scale.item() == 0 else mean * scale


def tensor_risk_score(costs, objective: RiskObjective = RiskObjective()):
    """Differentiable equal-weight mean, CVaR, blend or finite-scenario maximum."""
    torch = _require_torch()
    if not isinstance(objective, RiskObjective):
        raise TypeError("objective must be a RiskObjective.")
    if costs.ndim != 1 or costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError("costs must be a finite nonempty one-dimensional tensor.")
    if costs.is_complex() or costs.dtype == torch.bool:
        raise TypeError("costs must contain real numbers, not complex values or booleans.")
    if not costs.is_floating_point():
        costs = costs.double()
    if objective.mode == "mean":
        return _tensor_weighted_mean(costs)
    if objective.mode == "worst_case":
        return costs.max()
    ordered = costs.sort(descending=True).values
    whole, fraction, mass = _tail_mass(costs.numel(), objective.alpha)
    selected = ordered[:whole]
    weights = costs.new_ones(whole)
    if fraction and whole < costs.numel():
        selected = torch.cat((selected, ordered[whole:whole+1]))
        weights = torch.cat((weights, costs.new_tensor([fraction])))
    tail = _tensor_weighted_mean(selected, weights)
    if objective.mode == "cvar":
        return tail
    return _tensor_weighted_mean(torch.stack((_tensor_weighted_mean(costs), tail)),
        costs.new_tensor([1 - objective.risk_weight, objective.risk_weight]))


def _demand_rows(paths: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(_integer("demand", d) for d in path) for path in paths)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Demand must be nonempty equal-length finite scenarios.")
    return rows


def _forecast_rows(policy: StructuredNeuralPolicy, values, count: int, periods: int):
    if values is None:
        if callable(policy.forecast):
            raise ValueError("Supply explicit as-of forecast arrays for a dynamic forecast policy.")
        vector = policy.forecast_for(InventoryState(0, 0, 0, 0))
        return tuple((vector,) * periods for _ in range(count))
    rows = tuple(tuple(policy._training_forecast_values(vector) for vector in scenario) for scenario in values)
    if len(rows) != count or any(len(row) != periods for row in rows):
        raise ValueError("forecasts must have shape [scenario, period, forecast_horizon].")
    if policy.forecast is None and any(any(any(vector) for vector in row) for row in rows):
        raise ValueError("A deployment forecast provider is required when training with nonzero forecasts.")
    return rows


def _costs(holding_cost_per_unit, stockout_cost_per_unit, order_cost_per_unit, order_cost_per_order):
    fixed = _real("order_cost_per_order", order_cost_per_order, 0)
    if fixed > 0:
        raise ValueError("HDPO continuous training does not support positive fixed order_cost_per_order; use exact risk candidate search.")
    return dict(
        holding_cost_per_unit=_real("holding_cost_per_unit", holding_cost_per_unit, 0),
        stockout_cost_per_unit=_real("stockout_cost_per_unit", stockout_cost_per_unit, 0),
        order_cost_per_unit=_real("order_cost_per_unit", order_cost_per_unit, 0),
    )


@dataclass(frozen=True)
class TensorRollout:
    costs: object
    orders: object
    receipts: object
    on_hand: object
    lost_sales: object
    residual_penalty: object


def _rollout(policy, demand, forecasts, model, initial_on_hand, costs, round_orders=False):
    torch = _require_torch()
    count, periods = demand.shape
    stock = demand.new_full((count,), initial_on_hand)
    pipeline = demand.new_zeros((count, policy.lead_time))
    total_cost = demand.new_zeros(count)
    orders, receipts, inventories, lost_units, penalties = [], [], [], [], []
    c = policy.config
    for period in range(periods):
        received = pipeline[:, 0] if c.lead_time else demand.new_zeros(count)
        if c.lead_time:
            pipeline = torch.cat((pipeline[:, 1:], demand.new_zeros((count, 1))), dim=1)
        available = stock + received
        lost = torch.relu(demand[:, period] - available)
        post_demand = torch.relu(available - demand[:, period])
        if period % c.review_period == 0:
            vector = forecasts[:, period, :]
            features = torch.cat((post_demand[:, None], pipeline, vector), dim=1) / c.feature_scale
            raw = model(features)
            if raw.shape != (count, 1):
                raise ValueError("Neural model must return shape [scenario, 1].")
            correction = torch.tanh(raw[:, 0])
            target = c.target_stock + (vector[:, :c.protection_period].sum(dim=1) if c.use_forecast_target else 0)
            quantity = _bounded_tensor_order(target + c.residual_scale * correction - post_demand - pipeline.sum(dim=1),
                                            c.max_order_quantity)
            if round_orders:
                nearest = quantity.round()
                quantity = torch.where((quantity - nearest).abs() <= 1e-10, nearest, quantity.ceil())
            penalties.append((correction**2).mean())
        else:
            quantity = demand.new_zeros(count)
        stock = post_demand + quantity if c.lead_time == 0 else post_demand
        if c.lead_time:
            pipeline = torch.cat((pipeline[:, :-1], quantity[:, None]), dim=1)
        total_cost = total_cost + (stock * costs['holding_cost_per_unit']
            + lost * costs['stockout_cost_per_unit'] + quantity * costs['order_cost_per_unit'])
        orders.append(quantity)
        receipts.append(received)
        inventories.append(stock)
        lost_units.append(lost)
    return TensorRollout(total_cost, torch.stack(orders, dim=1), torch.stack(receipts, dim=1),
        torch.stack(inventories, dim=1), torch.stack(lost_units, dim=1), torch.stack(penalties).mean())


def differentiable_rollout(
    policy: StructuredNeuralPolicy, demand_scenarios: Iterable[Iterable[int]], *,
    forecasts=None, model=None, initial_on_hand: int = 0,
    holding_cost_per_unit: float = 0.0, stockout_cost_per_unit: float = 0.0,
    order_cost_per_unit: float = 0.0, order_cost_per_order: float = 0.0,
    round_orders: bool = False,
) -> TensorRollout:
    """CPU float64 receipt→demand→order rollout, with graph-connected costs.

    ``round_orders=False`` is the continuous training relaxation. True reproduces
    integer deployment but is unsuitable for pathwise learning through rounding.
    Forecasts are exogenous [scenario, period, H] as-of arrays; defaults are a
    static policy forecast or zero. Demand scenarios contain integer observations.
    There is no warmup, pipeline initialization, salvage or post-horizon cost.
    """
    initial_on_hand = _integer("initial_on_hand", initial_on_hand)
    rows = _demand_rows(demand_scenarios)
    future = _forecast_rows(policy, forecasts, len(rows), len(rows[0]))
    costs = _costs(holding_cost_per_unit, stockout_cost_per_unit, order_cost_per_unit, order_cost_per_order)
    torch = _require_torch()
    demand = torch.tensor(rows, dtype=torch.float64, device="cpu")
    future = torch.tensor(future, dtype=torch.float64, device="cpu")
    return _rollout(policy, demand, future, _model_from_policy(policy) if model is None else model,
                    initial_on_hand, costs, round_orders)


@dataclass(frozen=True)
class HDPOEpoch:
    epoch: int
    training_objective: float
    validation_score: float
    regularized_loss: float


@dataclass(frozen=True)
class HDPOTrainingResult:
    policy: StructuredNeuralPolicy
    objective: RiskObjective
    history: tuple[HDPOEpoch, ...]
    selected_epoch: int
    seed: int
    runtime_seconds: float
    training_continuous_score: float
    training_exact_score: float
    validation_score: float
    rounding_gap: float


def _exact_costs(policy, rows, forecasts, initial_on_hand, costs):
    values = []
    for trace, future in zip(rows, forecasts):
        deployed = policy.with_forecast(lambda state, f=future: f[state.period])
        result = simulate_replenishment(periods=len(trace), demand=trace,
            initial_on_hand=initial_on_hand, lead_time=policy.lead_time, policy=deployed, **costs)
        values.append(result.summary.total_cost)
    return tuple(values)


def train_hdpo(
    policy: StructuredNeuralPolicy, *, training_demand: Iterable[Iterable[int]],
    validation_demand: Iterable[Iterable[int]], training_forecasts=None,
    validation_forecasts=None, initial_on_hand: int = 0,
    epochs: int = 100, learning_rate: float = .01, batch_size: int | None = None,
    regularization: float = .001, max_gradient_norm: float = 10.0, seed: int = 0,
    objective: RiskObjective = RiskObjective(),
    holding_cost_per_unit: float = 0.0, stockout_cost_per_unit: float = 0.0,
    order_cost_per_unit: float = 0.0, order_cost_per_order: float = 0.0,
) -> HDPOTrainingResult:
    """Train via pathwise cost gradients; select by separate exact validation.

    The initial policy is eligible. Validation scenarios never enter gradients.
    A local CPU generator controls scenario mini-batches without changing caller
    RNG state. Input policies/forecasts are not mutated. Deploy on a third, unseen
    test window. Cross-version/device reproducibility and superiority to baseline
    policies are not guaranteed. Fixed ordering costs are explicitly unsupported.
    """
    started = time.perf_counter()
    epochs = _integer("epochs", epochs, 1)
    seed = _integer("seed", seed)
    initial_on_hand = _integer("initial_on_hand", initial_on_hand)
    rate, penalty = _real("learning_rate", learning_rate, 0), _real("regularization", regularization, 0)
    clip = _real("max_gradient_norm", max_gradient_norm, 0)
    if rate == 0 or clip == 0:
        raise ValueError("learning_rate and max_gradient_norm must be positive.")
    if not isinstance(objective, RiskObjective):
        raise TypeError("objective must be a RiskObjective.")
    costs = _costs(holding_cost_per_unit, stockout_cost_per_unit, order_cost_per_unit, order_cost_per_order)
    training = _demand_rows(training_demand)
    validation = _demand_rows(validation_demand)
    train_future = _forecast_rows(policy, training_forecasts, len(training), len(training[0]))
    val_future = _forecast_rows(policy, validation_forecasts, len(validation), len(validation[0]))
    size = len(training) if batch_size is None else _integer("batch_size", batch_size, 1)
    if size > len(training):
        raise ValueError("batch_size must not exceed the number of training scenarios.")
    active_tail = (objective.mode == "worst_case" or
        (objective.alpha > 0 and (objective.mode == "cvar" or
         (objective.mode == "mean_cvar" and objective.risk_weight > 0))))
    if size < len(training) and active_tail:
        raise ValueError("Tail-risk objectives require the full training scenario batch; partial-batch CVaR/worst cost changes the objective.")
    torch = _require_torch()
    demand, future = torch.tensor(training, dtype=torch.float64, device="cpu"), torch.tensor(train_future, dtype=torch.float64, device="cpu")
    model = _model_from_policy(policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best_policy, selected_epoch = policy, 0
    best_score = objective.score(_exact_costs(policy, validation, val_future, initial_on_hand, costs))
    with torch.no_grad():
        initial = _rollout(policy, demand, future, model, initial_on_hand, costs)
        score = tensor_risk_score(initial.costs, objective).item()
    history = [HDPOEpoch(0, score, best_score, score + penalty * initial.residual_penalty.item())]
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(len(training), generator=generator, device="cpu")[:size]
        optimizer.zero_grad()
        rollout = _rollout(policy, demand[indices], future[indices], model, initial_on_hand, costs)
        loss = tensor_risk_score(rollout.costs, objective) + penalty * rollout.residual_penalty
        if not torch.isfinite(loss):
            raise ValueError("HDPO loss became nonfinite; check scales, costs and learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        candidate = _policy_from_model(policy, model)
        validation_score = objective.score(_exact_costs(candidate, validation, val_future, initial_on_hand, costs))
        if validation_score < best_score:
            best_score, best_policy, selected_epoch = validation_score, candidate, epoch
        with torch.no_grad():
            current = _rollout(policy, demand, future, model, initial_on_hand, costs)
            score = tensor_risk_score(current.costs, objective).item()
        history.append(HDPOEpoch(epoch, score, validation_score, score + penalty * current.residual_penalty.item()))
    with torch.no_grad():
        selected_model = _model_from_policy(best_policy)
        continuous = tensor_risk_score(_rollout(best_policy, demand, future, selected_model,
            initial_on_hand, costs).costs, objective).item()
    exact = objective.score(_exact_costs(best_policy, training, train_future, initial_on_hand, costs))
    return HDPOTrainingResult(best_policy, objective, tuple(history), selected_epoch, seed,
        time.perf_counter() - started, continuous, exact, best_score, exact - continuous)
