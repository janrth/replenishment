# Advanced independent-item policies

This extension adds a structured neural policy, hindsight differentiable policy
optimisation (HDPO), and empirical risk-sensitive/finite-scenario robust
optimisation. They operate on one item at a time and require no Gurobi, LP/MIP
solver, or shared-resource allocator. The earlier six baselines remain available.

## Installation and deployment

```bash
pip install -e ".[advanced,dev]"
# Or: uv sync --extra advanced --extra dev
```

PyTorch >=2.6,<3 is optional. Normal package imports, risk search, and deployed
numeric neural checkpoints work without importing Torch. Creating initialized
neural models, supervised training, tensor rollouts and HDPO need the extra.
Examples/training use CPU float64. CPU-only Linux Torch wheels can be installed
from PyTorch's CPU index before installing the package extra; CI does this.
The resolver selects a Torch version compatible with the chosen Python version.

## Structured residual policy

```python
from replenishment import create_structured_neural_policy

policy = create_structured_neural_policy(
    lead_time=2, target_stock=10, hidden_size=8,
    feature_scale=10, residual_scale=5, seed=17,
)
```

The MLP computes a bounded target correction. On eligible dates:

`q_continuous = clamp(target + residual_scale*tanh(MLP(features)) - position, 0, cap)`.

The initial output layer is zero, so the policy starts as the baseline. Features
are post-demand on-hand stock, L dated receipt slots and H as-of forecast values,
divided by `feature_scale`. H defaults to `max(1,L)+R-1`. Full pipeline timing is
retained. The target is static `target_stock`, or that buffer plus the
protection-period forecast prefix when `use_forecast_target=True`. Extra forecast
features beyond P=`max(1,L)+R-1` do not inflate the baseline target; target-enabled
models require H>=P. Default forecasts are zeros.

Supply `forecast=[...]` for a static H-vector or a callback receiving
`InventoryState`. Use `policy.with_forecast(provider)` to adapt a frozen policy
to the evaluation window. Providers must not read future realized demand. They
are caller-managed as-of forecasts, not forecasts secretly fit by the trainer.

Review dates are 0,R,2R,... . Continuous actions round upward for deployment,
snapping values within absolute 1e-10 of an integer to that integer. The optional
integer `max_order_quantity` is a per-item decision bound. A bounded residual
does not establish monotonicity, optimality, or universal improvement.

`NeuralPolicyConfig` and `NeuralWeights` are frozen, defensively copied dataclasses.
Inference is ordinary Python math on numeric weights, not a shared mutable Torch
module. `policy.save(path)` writes validated numeric JSON; `StructuredNeuralPolicy.load(path)`
loads it without pickle/model execution. Callable forecasts are not serialized:
re-supply them with `load(path, forecast=provider)`. Static forecasts are stored.

`train_structured_neural_policy(policy, states=..., target_orders=..., forecasts=...)`
fits supervised order labels with full-batch Adam and a residual penalty. It
returns an immutable deployed policy and loss history without mutating its input.
This is supervised distillation/regression, not direct cost-gradient optimisation.
Prepare states/labels in the training window; teacher imitation need not improve
cost. Forecast override rows must correspond to those states. Static/absent
deployment forecasts must match all overrides; varying forecasts need an explicit
as-of provider. Dynamic provider alignment across deployment windows remains the
caller's responsibility.

## HDPO: train through inventory dynamics

```python
from replenishment import train_hdpo, RiskObjective

fit = train_hdpo(
    policy,
    training_demand=[[2, 3, 2, 4], [1, 4, 2, 3]],
    validation_demand=[[3, 2, 3, 4]],
    epochs=50, learning_rate=.02, seed=17,
    holding_cost_per_unit=.5, stockout_cost_per_unit=5,
    objective=RiskObjective("mean"),
)
# Evaluate fit.policy on separate, untouched test demand.
```

Training backpropagates through receipt → demand → observation → order lost-sales
dynamics, including L=0 receipts after current demand, dated pipeline, and review
cadence. It starts with empty pipeline, optional initial stock, zero warmup and
no terminal salvage/post-horizon fulfillment. It uses holding, shortage and
per-unit ordering costs. Positive `order_cost_per_order` is explicitly rejected
by HDPO because its indicator is not handled by the pathwise differentiable
formulation. Exact simulation and risk candidate search still support fixed costs.

Demand scenarios are finite, equal-length integer paths per split; training and
validation horizons may differ. Optional `training_forecasts` and
`validation_forecasts` have shape [scenario, period, H]. Prepare them externally
as-of each decision. Dynamic deployment providers require explicit arrays; the
trainer cannot execute a Python state callback against differentiable tensor
states. Nonzero forecast arrays require a deployment forecast source so the
returned policy does not silently deploy with zero forecasts. Explicit arrays
must match a static source exactly; varying forecasts require an as-of provider.
For dynamic providers, the caller must align the provider to each deployment
window/scenario and to its externally supplied training/validation forecast data.

Actions are continuous during training. Ordering clamps use explicit feasible-side
derivatives at exact bounds so zero-order and capped baselines can learn inward;
stockout dynamics use Torch's piecewise derivatives. No straight-through or unbiased integer-gradient claim
is made. `differentiable_rollout(..., round_orders=True)` enables forward parity
checks with integer deployment; rounding is unsuitable for pathwise training.

Adam updates use only training scenarios and gradient clipping. Scenario
mini-batches are allowed for mean objectives; active CVaR/worst objectives require
the full scenario set because sorting a smaller random tail changes the objective.
A private CPU generator plus scoped model construction preserves caller CPU and
accelerator RNG streams. Fixed inputs/seed reproduce within a given
runtime; [PyTorch does not promise cross-release/device reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Every epoch, including epoch zero, is scored on separate validation demand using
the **exact integer simulator**. Validation never enters gradient computation.
The cheapest validation checkpoint is returned, with stable earliest-checkpoint
ties. Treat this split as model-selection data, and reserve a third test split.
History records current full-training continuous objective, exact validation
score and regularized loss. The result includes selected epoch, seed, runtime,
continuous/exact training scores and `rounding_gap = exact - continuous` on the
same selected model/traces. Scores are total episode costs, not per-period means.

The implementation follows the pathwise-gradient principle in
[Alvo et al., HDPO](https://arxiv.org/html/2306.11246v3), translated to this
single-item simulator. The paper discusses continuous actions and variable costs;
this implementation measures integer deployment separately. It does not copy
external code or reproduce the paper's benchmark/dominance claims.

## Empirical risk and finite-scenario robustness

```python
from replenishment import BaseStockPolicy, RiskObjective
from replenishment import optimize_risk_sensitive_policy_candidates

fit = optimize_risk_sensitive_policy_candidates(
    candidates={f"S={s}": BaseStockPolicy(s) for s in [4, 8, 12]},
    demand_scenarios=[[2, 3, 2, 4], [1, 2, 1, 12]],
    periods=4, initial_on_hand=0, lead_time=2,
    objective=RiskObjective("mean_cvar", alpha=.9, risk_weight=.5),
    holding_cost_per_unit=.5, stockout_cost_per_unit=5,
)
```

`RiskObjective` supports:

| Mode | Objective |
| --- | --- |
| `mean` | Average episode cost |
| `cvar` | Average worst (1-alpha) probability mass |
| `mean_cvar` | `(1-risk_weight)*mean + risk_weight*CVaR` |
| `worst_case` | Maximum cost among supplied scenarios |

Scenarios have equal weight. Confidence is 0<=alpha<1 and risk weight is in [0,1].
CVaR includes a fractional boundary observation: at alpha=.625, costs [0,2,4,12]
have 1.5 tail observations and CVaR `(12 + .5*4)/1.5 = 28/3`. It is not rounded
top-k. `empirical_cvar` and `RiskObjective.score` are dependency-free;
`tensor_risk_score` implements the matching differentiable objective for HDPO.
Gradients flow through the selected empirical tail or worst-scenario costs.

Exact search materializes only the requested horizon per scenario once and
replays identical traces across reusable candidates. Iterator tails remain
available. It returns selected policy/label, candidate scores, selected scenario
costs and mean/CVaR/worst diagnostics. Ties favor input order. This is finite-grid
selection and equal-weight empirical risk, not global policy optimality.
Stateful custom forecast/policy objects remain the caller's responsibility.

Worst-case protection covers the supplied finite stress set only. It is not a
formal distributionally robust guarantee and can miss unrepresented events.
Tail-focused choices can trade higher ordinary inventory cost for lower extreme
shortage cost; evaluate both metrics on independent scenarios.

## Examples and validation

- [Structured neural policy](../notebooks/structured_neural_policy_example.ipynb):
  direct missed-sales training, demand-weighted fill-rate validation, inventory/cost
  comparisons, immutable deployment and JSON roundtrip. Unit shortage cost with
  zero other costs makes HDPO minimise missed units; for fixed traces this
  maximises pooled fill rate. This does not enforce a specified minimum service
  threshold. Initial inventory and action bounds are explicit in the example.
  Three independently trained generated items (seasonal, intermittent and rising
  demand) use standard rows/DataFrames, the public config/simulation/decision APIs,
  and the shared item/aggregate stock and missed-sales plotting helper. Aggregate
  stock is descriptive and cannot be transferred between items.
- [HDPO](../notebooks/hdpo_policy_example.ipynb): actual gradient training, exact
  receipt-timing parity and continuous/integer cost diagnostics.
- [Risk-sensitive policy](../notebooks/risk_sensitive_policy_example.ipynb):
  hand-computed CVaR, mean/tail/worst tradeoffs, and training-only block-bootstrap
  scenarios from the same standard dataset, followed by exact held-out simulation.

These CPU examples use explicit seeds, train/validation/evaluation windows, cold starts,
zero warmup, the same generated multi-item dataset and declared training budgets. They report
cost, fill rate, lost units, average/peak stock, order frequency and runtime.
They are useful verification examples, not evidence of superiority on all items.
Core tests additionally compare tensor/Python trajectories, finite-difference
gradients, caller RNG state, reproducibility and validation checkpoint selection.
Their workflow matches the existing mean-forecast/percentile notebooks:
`generate_standard_simulation_rows`, chronological splitting, per-item fitting,
policy inspection, `build_article_configs_from_standard_rows`, simulation,
standard decision tables, and `plot_replenishment_decisions` for items and totals.
Evaluation actuals and simulated demand are the same realization. Notebook
assertions reconcile decision rows and plotted cumulative missed-sales/share
against exact simulation outcomes.

Neural policies integrate with the standard-row builder, per-item candidate search,
metadata and decision output. Lead times must match; implicit time aggregation
beyond window one is rejected. Learned targets do not populate unrelated static
forecast reorder thresholds. Shared constraints, network models, Rust, perishables
and dual sourcing remain outside this PR's independent-item scope.
