# Advanced independent-item learning and risk optimisation

## Scope and implementation choice

The user requested the first three advanced roadmap directions in this PR:
structured neural policy/residual correction, HDPO, and risk-sensitive/robust
optimisation. They explicitly chose an optional PyTorch `advanced` extra and
asked the implementer to use judgment and implement. This extends, and preserves,
the six previously implemented policies. Every model and scenario batch concerns
one item; there are no shared constraints or LP/MIP solvers.

Approaches considered: independent RL/PPO and differentiable-learning stacks
(unnecessary duplication); a required Torch runtime (changes lightweight core
installation); a shared optional Torch stack plus dependency-free scenario-risk
evaluation (selected). Neural policy is an architecture, HDPO a training method,
and risk objectives can train that architecture or select existing candidate
policies. Do not present them as three interchangeable fixed ordering formulas.

## Architecture and interfaces

Add `risk.py` without Torch dependencies, `neural.py` for the optional policy and
serialization, and `hdpo.py` for differentiable rollouts and training. Root neural
exports are lazy so normal package import, star import, and all core/risk features
work without Torch. Missing optional dependencies raise a specific install hint
only when learning functionality is requested. Add CPU advanced tests to CI.

`StructuredNeuralPolicy` adapts an MLP to the existing integer policy protocol.
Its continuous order is `clamp(target + residual_scale*tanh(network(features))
- inventory_position, 0, max_order_quantity)`, on review dates only. Baseline
target is either a static nonnegative target or the protection-period forecast
sum plus that target (a buffer). The network's output layer is initially zero,
reproducing the baseline before learning. Features contain normalized post-demand
on-hand, dated pipeline by remaining delay, and as-of future forecast vector.
Pipeline timing is retained; no future realized demand enters policy features.
Lead time, review cadence, forecast horizon and scaling are explicit and fixed.
Inference rounds continuous quantities upward with documented numerical handling
at integer boundaries and returns native nonnegative integers. The optional order
cap is per item and is a decision bound, not shared capacity.

Forecasts are optional static or period-indexed/observation callback inputs.
Inference and training must use the same feature construction, baseline target,
review gating and receipt convention. Supplied training forecast arrays have
shape [scenario, period, horizon] and must be prepared as-of externally. Default
forecasts are zero: no demand-trace-derived forecasting is hidden in the trainer.
Freeze inference weight snapshots and scaling, support safe JSON checkpoints
with structural metadata and numeric weights (no pickle execution), and ensure
Torch model/weights passed to an inference adapter are copied rather than shared
with a mutable trainer.

Training entry `train_structured_neural_policy` performs supervised warm-start
from caller-supplied observed states, optional forecasts and target orders. This
makes the architecture independently trainable (e.g. distillation from a tuned
baseline) before HDPO fine-tuning. Enforce chronological state/label responsibility
in docs and demonstrate held-out evaluation. This is not called RL or HDPO.

`train_hdpo` optimizes actual rollout costs by backpropagating through the
single-item lost-sales recursion, using Torch Adam on fixed offline scenarios.
The tensor rollout reproduces receipt → demand → observation → order timing,
including L=0, review cadence, holding/shortage/per-unit costs and terminal
treatment. Actions are continuous during training. At kinks use Torch's standard
piecewise derivative; do not claim unbiased discrete gradients. Positive fixed
ordering costs are rejected by differentiable training (rather than silently
using a nondifferentiable indicator or inventing an unlabelled surrogate).
Exact Python evaluation/candidate risk search still supports all existing costs.

Separate training scenarios, validation scenarios for selecting checkpoints,
and untouched held-out evaluation. Select the initial policy or any epoch by
exact integer validation objective (initial is eligible); validation never enters
gradient computation. Report continuous training objective, exact validation
objective/history, selected epoch, update count/seed/runtime and final rounded
policy. Preserve a local RNG scope rather than reseeding the process globally;
CPU runs with fixed inputs/seed should reproduce within one environment. Do not
promise cross-platform/PyTorch-version reproducibility.

## Risk and empirical robustness

`RiskObjective` supports mean cost, empirical upper-tail CVaR at confidence
0 <= alpha < 1, a convex mean/CVaR blend with weight in [0,1], and worst-case
cost over supplied scenarios. CVaR includes a fractional boundary sample when
tail mass (1-alpha)*n is noninteger; it is not simply a rounded top-k mean.
Inputs are finite nonempty equal-weight scenario costs. Matching tensor
objectives propagate gradients through sorted tail/worst-scenario costs.

`optimize_risk_sensitive_policy_candidates` evaluates reusable existing/new
policies on identical materialized per-item scenario traces. Return the winning
policy/label, all candidate scores, winning per-scenario costs, and mean/CVaR/max
diagnostics. Ties choose input order. Materialize bounded horizons once, preserve
generator tails, validate equal scenario lengths, and disclose finite-grid scope.
Worst-scenario optimisation is robustness only over the supplied finite scenario
set, not a general distributionally robust guarantee. Scenario weights are equal;
unequal probabilities and global resource allocation are excluded.

## Integration, tests and examples

Neural adapters work with the existing simulator, metadata, standard-row builder,
decision output and candidate search. Register their validation/metadata contract
without importing Torch into normal simulations. Metadata identifies learning
architecture and target semantics and preserves review cadence independently of
aggregation. Reject nonidentity time aggregation. Do not invent forecast reorder
thresholds for the learned residual rule. Keep current baseline APIs unchanged.

Tests: dependency isolation without Torch, quantity/shape/finite validation,
zero residual equals base-stock, pipeline timing, input/weight defensive copies,
checkpoint roundtrip, supervised loss reduction, exact tensor/Python rollout
parity at integer actions (L=0/1/2, R>1), independent finite-difference gradient
checks away from kinks, reproducible updates, actual weight changes, validation
checkpoint selection, fixed-order-cost rejection, scenario replay/tail handling,
hand-computed CVaR/fractional tails, and mean-vs-CVaR/worst choices.

Add and execute `structured_neural_policy_example.ipynb`,
`hdpo_policy_example.ipynb`, and `risk_sensitive_policy_example.ipynb`.
Each is self-contained and trains/selects a working model, verifies a worked
case, and evaluates on chronological held-out demand against tuned simple
baselines with matched traces. Report cost, fill rate, lost units, mean/peak
inventory, order frequency, runtime, seed and training budget; learning examples
show continuous versus exact integer outcomes. Risk examples separately report
mean, CVaR and worst cost and make tradeoffs visible under shortage shocks.
State zero warmup/cold start/no salvage. Save model/checkpoint examples only to
temporary locations unless intentionally part of the example output.

Update README, advanced documentation, changelog and optional-dependency setup.
No promise is made that learned methods dominate tuned baselines on every case.

## Research references and limitations

HDPO follows the pathwise-gradient training principle, with a single-item timing
translation, not copied external source code or claimed published benchmarks:
https://arxiv.org/html/2306.11246v3 . That formulation explains continuous actions
and variable costs; this implementation explicitly measures integer deployment
and does not silently extend its differentiability claim to fixed ordering costs.
PyTorch autograd and reproducibility documentation guide runtime behavior.
Hyperparameter defaults, public helper signatures, feature dimensions, numerical
tolerances, exact scenario grids and notebook training budgets are builder calls.
