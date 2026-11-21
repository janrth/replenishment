"""Structured neural ordering with optional Torch training and plain Python deployment.

Importing this module does not import Torch. Numeric JSON checkpoints contain no
executable model objects. Forecast providers must supply only as-of information.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
import importlib
import json
import math
from pathlib import Path

from .item_policies import _integer
from .risk import _real
from .simulation import InventoryState

ForecastSource = Iterable[float] | Callable[[InventoryState], Iterable[float]]


def _require_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError("Neural training requires the optional extra: pip install 'replenishment[advanced]'.") from exc


def _bounded_tensor_order(raw, cap: int | None = None):
    """Same forward bounds, explicit feasible-side derivatives at exact ties.

    Framework clamp derivatives at equality vary; zero would strand a zero
    initialized policy. At the lower/upper boundary select the interior branch.
    """
    torch = _require_torch()
    values = torch.where(raw >= 0, raw, raw.new_zeros(()))
    if cap is not None:
        values = torch.where(values <= cap, values, values.new_full((), cap))
    return values


@dataclass(frozen=True)
class NeuralPolicyConfig:
    lead_time: int
    target_stock: float = 0.0
    review_period: int = 1
    forecast_horizon: int | None = None
    hidden_size: int = 8
    residual_scale: float = 10.0
    feature_scale: float = 10.0
    max_order_quantity: int | None = None
    use_forecast_target: bool = False

    def __post_init__(self) -> None:
        for name, minimum in (("lead_time", 0), ("review_period", 1), ("hidden_size", 1)):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum))
        horizon = max(1, self.lead_time) + self.review_period - 1 if self.forecast_horizon is None else self.forecast_horizon
        object.__setattr__(self, "forecast_horizon", _integer("forecast_horizon", horizon, 1))
        for name in ("target_stock", "residual_scale", "feature_scale"):
            object.__setattr__(self, name, _real(name, getattr(self, name), 0))
        if self.feature_scale == 0:
            raise ValueError("feature_scale must be positive.")
        if self.max_order_quantity is not None:
            object.__setattr__(self, "max_order_quantity", _integer("max_order_quantity", self.max_order_quantity))
        if not isinstance(self.use_forecast_target, bool):
            raise TypeError("use_forecast_target must be a boolean.")
        if self.use_forecast_target and self.forecast_horizon < self.protection_period:
            raise ValueError("forecast_horizon must cover the protection period when forecast targeting is enabled.")

    @property
    def protection_period(self) -> int:
        return max(1, self.lead_time) + self.review_period - 1

    @property
    def input_size(self) -> int:
        return 1 + self.lead_time + self.forecast_horizon


@dataclass(frozen=True)
class NeuralWeights:
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_weights", tuple(tuple(_real("weight", v) for v in row) for row in self.input_weights))
        for name in ("hidden_bias", "output_weights"):
            object.__setattr__(self, name, tuple(_real("weight", v) for v in getattr(self, name)))
        object.__setattr__(self, "output_bias", _real("output_bias", self.output_bias))


@dataclass(frozen=True)
class StructuredNeuralPolicy:
    """A bounded learned target correction using inventory, receipts and forecasts.

    continuous q = clamp(S + optional forecast sum + scale*tanh(MLP(features))
                         - inventory position, 0, optional cap).
    Features are on-hand, L dated receipt slots and H future as-of forecasts,
    all divided by feature_scale. Orders are rounded upward at deployment; gaps
    within 1e-10 of an integer snap to it. No future realized demand is read.
    The weights and static forecasts are immutable defensive copies.
    """

    config: NeuralPolicyConfig
    weights: NeuralWeights
    forecast: ForecastSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, NeuralPolicyConfig) or not isinstance(self.weights, NeuralWeights):
            raise TypeError("config and weights must be NeuralPolicyConfig and NeuralWeights.")
        w, c = self.weights, self.config
        if (len(w.input_weights) != c.hidden_size or len(w.hidden_bias) != c.hidden_size
                or len(w.output_weights) != c.hidden_size
                or any(len(row) != c.input_size for row in w.input_weights)):
            raise ValueError("Neural weights do not match configured feature/hidden dimensions.")
        if self.forecast is not None and not callable(self.forecast):
            object.__setattr__(self, "forecast", self._forecast_values(self.forecast))

    @property
    def lead_time(self) -> int:
        return self.config.lead_time

    @property
    def review_period(self) -> int:
        return self.config.review_period

    @property
    def forecast_horizon(self) -> int:
        return self.config.forecast_horizon

    def _forecast_values(self, values: Iterable[float]) -> tuple[float, ...]:
        values = tuple(_real("forecast", v, 0) for v in values)
        if len(values) != self.forecast_horizon:
            raise ValueError("forecast must match the configured forecast_horizon.")
        return values

    def forecast_for(self, state: InventoryState) -> tuple[float, ...]:
        if self.forecast is None:
            return (0.0,) * self.forecast_horizon
        return self._forecast_values(self.forecast(state) if callable(self.forecast) else self.forecast)

    def _training_forecast_values(self, values: Iterable[float]) -> tuple[float, ...]:
        values = self._forecast_values(values)
        if not callable(self.forecast):
            expected = (0.0,) * self.forecast_horizon if self.forecast is None else self.forecast
            if values != expected:
                raise ValueError("Training forecast overrides must match static deployment forecasts; use an explicit as-of provider for varying forecasts.")
        return values

    def features_for(self, state: InventoryState, *, forecast: Iterable[float] | None = None) -> tuple[float, ...]:
        period = _integer("state.period", state.period)
        on_hand = _integer("state.on_hand", state.on_hand)
        on_order = _integer("state.on_order", state.on_order)
        if state.backorders != 0:
            raise ValueError("Structured neural policies require lost-sales states without backlog.")
        slots = [0] * self.lead_time
        previous = period
        for receipt_period, quantity in state.arrival_schedule:
            receipt_period = _integer("arrival period", receipt_period)
            quantity = _integer("arrival quantity", quantity, 1)
            if receipt_period <= previous or receipt_period > period + self.lead_time:
                raise ValueError("arrival_schedule must be ordered after the state within lead_time.")
            slots[receipt_period - period - 1] = quantity
            previous = receipt_period
        if sum(slots) != on_order:
            raise ValueError("arrival_schedule quantities must sum to on_order.")
        forecasts = self.forecast_for(state) if forecast is None else self._forecast_values(forecast)
        return tuple(value / self.config.feature_scale for value in (on_hand, *slots, *forecasts))

    def continuous_order_for(self, state: InventoryState) -> float:
        if state.period % self.review_period:
            return 0.0
        forecasts = self.forecast_for(state)
        features = self.features_for(state, forecast=forecasts)
        w = self.weights
        hidden = [math.tanh(math.fsum(a*b for a, b in zip(row, features)) + bias)
                  for row, bias in zip(w.input_weights, w.hidden_bias)]
        residual = self.config.residual_scale * math.tanh(math.fsum(a*b for a, b in zip(w.output_weights, hidden)) + w.output_bias)
        target = self.config.target_stock + (math.fsum(forecasts[:self.config.protection_period]) if self.config.use_forecast_target else 0)
        quantity = max(0.0, target + residual - int(state.on_hand) - int(state.on_order))
        return min(quantity, self.config.max_order_quantity) if self.config.max_order_quantity is not None else quantity

    def order_quantity_for(self, state: InventoryState) -> int:
        quantity = self.continuous_order_for(state)
        nearest = round(quantity)
        return nearest if abs(quantity - nearest) <= 1e-10 else math.ceil(quantity)

    def with_forecast(self, forecast: ForecastSource | None) -> StructuredNeuralPolicy:
        return replace(self, forecast=forecast)

    def save(self, path: str | Path) -> None:
        """Save numeric JSON; callable forecasts must be supplied again on load."""
        document = dict(version=1, config=asdict(self.config), weights=asdict(self.weights),
                        forecast_kind="external" if callable(self.forecast) else "static",
                        forecast=None if callable(self.forecast) else self.forecast)
        Path(path).write_text(json.dumps(document, allow_nan=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path, *, forecast: ForecastSource | None = None) -> StructuredNeuralPolicy:
        document = json.loads(Path(path).read_text())
        if document.get("version") != 1:
            raise ValueError("Unsupported neural checkpoint version.")
        if document.get("forecast_kind") == "external" and forecast is None:
            raise ValueError("Checkpoint needs an external as-of forecast provider.")
        return cls(NeuralPolicyConfig(**document["config"]), NeuralWeights(**document["weights"]),
                   forecast if forecast is not None else document.get("forecast"))


def _model_from_policy(policy: StructuredNeuralPolicy):
    torch = _require_torch()
    # Model construction initializes parameters; contain those RNG side effects.
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(
            torch.nn.Linear(policy.config.input_size, policy.config.hidden_size, device="cpu", dtype=torch.float64),
            torch.nn.Tanh(), torch.nn.Linear(policy.config.hidden_size, 1, device="cpu", dtype=torch.float64),
        ).double()
    w = policy.weights
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(w.input_weights, dtype=torch.float64, device="cpu"))
        model[0].bias.copy_(torch.tensor(w.hidden_bias, dtype=torch.float64, device="cpu"))
        model[2].weight.copy_(torch.tensor([w.output_weights], dtype=torch.float64, device="cpu"))
        model[2].bias.fill_(w.output_bias)
    return model


def _policy_from_model(policy: StructuredNeuralPolicy, model) -> StructuredNeuralPolicy:
    weights = NeuralWeights(model[0].weight.detach().cpu().tolist(), model[0].bias.detach().cpu().tolist(),
        model[2].weight.detach().cpu().tolist()[0], model[2].bias.detach().cpu().item())
    return replace(policy, weights=weights)


def create_structured_neural_policy(*, seed: int = 0, forecast: ForecastSource | None = None, **configuration) -> StructuredNeuralPolicy:
    """Initialize a seeded residual MLP with a zero output layer (baseline rule)."""
    config = NeuralPolicyConfig(**configuration)
    # Validate a static forecast before importing an optional runtime.
    empty = NeuralWeights(((0.0,) * config.input_size,) * config.hidden_size,
                         (0.0,) * config.hidden_size, (0.0,) * config.hidden_size, 0.0)
    template = StructuredNeuralPolicy(config, empty, forecast)
    seed = _integer("seed", seed)
    torch = _require_torch()
    model = _model_from_policy(template)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model[0].weight, generator=generator)
        torch.nn.init.zeros_(model[0].bias)
    return _policy_from_model(template, model)


@dataclass(frozen=True)
class NeuralTrainingResult:
    policy: StructuredNeuralPolicy
    loss_history: tuple[float, ...]
    seed: int


def train_structured_neural_policy(
    policy: StructuredNeuralPolicy, *, states: Sequence[InventoryState],
    target_orders: Iterable[float], forecasts: Sequence[Iterable[float]] | None = None,
    epochs: int = 100, learning_rate: float = .01, regularization: float = .001, seed: int = 0,
) -> NeuralTrainingResult:
    """Supervised residual learning from observed states and caller-supplied labels.

    This is distillation/regression, not HDPO. Prepare states/labels using only
    the training window. Full-batch CPU Adam is deterministic in a fixed runtime;
    the seed is recorded and caller Torch RNG state is preserved.
    """
    epochs = _integer("epochs", epochs, 1)
    seed = _integer("seed", seed)
    rate = _real("learning_rate", learning_rate, 0)
    penalty = _real("regularization", regularization, 0)
    if rate == 0:
        raise ValueError("learning_rate must be positive.")
    states = tuple(states)
    targets = tuple(_real("target order", q, 0) for q in target_orders)
    if not states or len(states) != len(targets) or (forecasts is not None and len(forecasts) != len(states)):
        raise ValueError("states, target_orders and forecasts must have the same nonzero length.")
    if any(s.period % policy.review_period and q != 0 for s, q in zip(states, targets)):
        raise ValueError("Target orders must be zero off review dates.")
    vectors = [policy.forecast_for(s) if forecasts is None else policy._training_forecast_values(forecasts[i]) for i, s in enumerate(states)]
    features = [policy.features_for(s, forecast=f) for s, f in zip(states, vectors)]
    torch = _require_torch()
    x, y = torch.tensor(features, dtype=torch.float64, device="cpu"), torch.tensor(targets, dtype=torch.float64, device="cpu")
    positions = torch.tensor([int(s.on_hand) + int(s.on_order) for s in states], dtype=torch.float64, device="cpu")
    baseline = torch.tensor([policy.config.target_stock + (sum(f[:policy.config.protection_period]) if policy.config.use_forecast_target else 0) for f in vectors], dtype=torch.float64, device="cpu")
    gate = torch.tensor([s.period % policy.review_period == 0 for s in states], dtype=torch.float64, device="cpu")
    model = _model_from_policy(policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        correction = torch.tanh(model(x).squeeze(-1))
        # A feasible right derivative at zero lets a zero-order baseline learn.
        orders = _bounded_tensor_order(baseline + policy.config.residual_scale * correction - positions,
                                      policy.config.max_order_quantity)
        loss = ((orders * gate - y)**2).mean() + penalty * (correction**2).mean()
        loss.backward()
        optimizer.step()
        history.append(loss.detach().item())
    return NeuralTrainingResult(_policy_from_model(policy, model), tuple(history), seed)
