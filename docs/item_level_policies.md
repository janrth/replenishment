# Independent-item replenishment policies

The six policies below use the existing `order_quantity_for(InventoryState)`
protocol. They require no LP/MIP solver, Gurobi, or shared-item optimisation.
All quantities/targets/caps and lead times are nonnegative integers. Review
periods are positive integers; booleans, fractional quantities and nonfinite
inputs are rejected. The `(s,S)` reorder threshold may be a negative integer.
Accepted NumPy integer scalars are converted to native Python integers before
arithmetic. Simulations using these policies validate initial stock, lead time,
period count, costs and each demand observation, including callable demand.

| API | Rule on a review date | Tunable parameters |
| --- | --- | --- |
| `BaseStockPolicy(target_stock, review_period=1)` | `max(0, S - position)` | S |
| `ConstantOrderPolicy(quantity, review_period=1)` | Always quantity | quantity |
| `CappedBaseStockPolicy(target_stock, cap, review_period=1)` | `min(cap, max(0, S - position))` | S and cap jointly |
| `sSPolicy(reorder_point, target_stock, review_period=1)` | At position <= s, order `max(0, S - position)` | s and S jointly, S >= s |
| `ProjectedInventoryLevelPolicy(target_stock, demand_pmf, lead_time, review_period=1)` | Raise expected stock before arrival demand to target | Target; fit demand PMF on history |
| `ProtectionPeriodQuantilePolicy(quantile, lead_time, review_period=1, aggregate_samples=None, demand_paths=None)` | Target an aggregate-demand quantile, subtract position | Quantile; fit sample model on history |

Inventory position is on-hand plus total outstanding orders. Review dates are
0, R, 2R, ...; every other date returns zero. This cadence does not aggregate
demand or change stockout/holding-cost time resolution. A policy cap is a tuned
decision parameter, not a physical supplier-capacity limit. `(s,S)` uses a variable
order quantity and includes equality at s; existing `ReorderPointPolicy` is the
fixed-quantity `(r,Q)` policy.

## Lost sales and event timing

Each period receives due orders, fulfills current demand, exposes the post-demand
state, then places an order. An order placed at t arrives before demand at t+L
when L>0. A zero-lead-time order arrives after demand at t, and can first fulfill
demand at t+1. Unfulfilled demand disappears and is never fulfilled later.

The explicit metrics are `snapshot.lost_sales` and `summary.total_lost_sales`.
They are read-only derived properties, so dataclass field serialization retains
its existing schema. `summary.total_backorders` remains the compatibility name for
total lost units. Per-period `snapshot.backorders` and policy-state backorders
remain zero. No backlog mode has been added. `snapshot.starting_on_hand` is stock
available immediately before demand, after due receipts. At L=0 it excludes the
order placed after demand; `ending_on_hand` includes that order.

Here is a hand-calculated trace with initial stock 1, demand `[2,3,2,2]`, and
orders of four units at t=0 and t=2 (review period two):

| L | Period | Receipt | Stock before demand | Demand | Lost units | Order | Ending stock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | 2 | 1 | 4 | 4 |
| 0 | 1 | 0 | 4 | 3 | 0 | 0 | 1 |
| 0 | 2 | 0 | 1 | 2 | 1 | 4 | 4 |
| 0 | 3 | 0 | 4 | 2 | 0 | 0 | 2 |
| 1 | 0 | 0 | 1 | 2 | 1 | 4 | 0 |
| 1 | 1 | 4 | 4 | 3 | 0 | 0 | 1 |
| 1 | 2 | 0 | 1 | 2 | 1 | 4 | 0 |
| 1 | 3 | 4 | 4 | 2 | 0 | 0 | 2 |
| 2 | 0 | 0 | 1 | 2 | 1 | 4 | 0 |
| 2 | 1 | 0 | 0 | 3 | 3 | 0 | 0 |
| 2 | 2 | 4 | 4 | 2 | 0 | 4 | 2 |
| 2 | 3 | 0 | 2 | 2 | 0 | 0 | 0 |

At a decision, `state.arrival_schedule` is an immutable tuple of positive
`(absolute_arrival_period, quantity)` pairs, sorted by date. It excludes the
new order. Its quantities sum to `on_order`. Existing four-argument
`InventoryState` construction remains valid with an empty default schedule;
PIL additionally needs a consistent schedule whenever on-order stock is nonzero.

For a static inventory-position target, use future demand periods t+1 through
t+H, where **H=max(1,L)+R-1** under this event convention. For example, L=2/R=3
gives H=4; the current order can serve demand at t+2, while the next review's
order can first serve t+5. The target includes demand before the new receipt
because inventory position includes current stock and earlier pipeline too.
These targets remain heuristics in general lost-sales settings.

## Projected inventory-level

```python
from replenishment import InventoryState, ProjectedInventoryLevelPolicy

policy = ProjectedInventoryLevelPolicy(5, {3: 1.0}, lead_time=3)
early = InventoryState(0, 0, 4, 0, ((1, 4),))
late = InventoryState(0, 0, 4, 0, ((2, 4),))
assert policy.projected_inventory(early) == 0
assert policy.projected_inventory(late) == 1
assert policy.order_quantity_for(early) == 5
assert policy.order_quantity_for(late) == 4
```

The model uses finite discrete IID demand. At each intervening future period,
exact probability propagation applies `max(0, stock + receipt - demand)`.
The expectation is taken immediately before demand at the new arrival, including
old receipts at that time. There is no intervening demand at L=0 or L=1.
This avoids the incorrect approximation of applying lost-sales truncation only
after replacing demand with its mean.

PMFs are copied and made immutable. Probabilities must be finite, nonnegative
and sum to one within absolute tolerance 1e-12. Only summation noise is normalized.
The order is `ceil(max(0, target - expected_stock))`, snapping gaps within absolute
tolerance 1e-12 of an integer to that integer. The implementation averages centered
target-minus-stock differences and normalizes probability mass to avoid
cancellation when inventory and target are large and close. Inconsistent, unsorted, duplicate,
past-due or later-than-new-arrival receipt schedules are rejected. Policy and
simulation lead times must match, even with an empty pipeline.

This implementation needs no stochastic estimator or random seed. Exact state
space can grow with inventory, support size and lead time. It is a correctness
reference for modest models, not an accelerated large-scale research backend.
Its IID assumptions do not automatically describe seasonal or shifted demand.
The idea follows [van Jaarsveld and Arts (2024)](https://doi.org/10.1287/opre.2021.0032);
the notebook verifies our timing translation without asserting their performance
guarantees for arbitrary inputs.

## Aggregate-demand quantiles

Supply exactly one of `aggregate_samples` (nonnegative integer demand totals) or
`demand_paths` (equal-weight complete future paths of length H). Static inputs,
including generators, are copied to immutable tuples. A callback may instead
accept the current `InventoryState` and return as-of samples/paths on each review.
It is responsible for using only information available at that decision. The
simulator provides no future realized demand or online demand-history interface.

```python
from replenishment import InventoryState, ProtectionPeriodQuantilePolicy

policy = ProtectionPeriodQuantilePolicy(
    0.5, lead_time=2, demand_paths=[[0, 10], [10, 0]],
)
assert policy.target_for(InventoryState(0, 0, 0, 0)) == 10
```

Each marginal's 50th percentile is zero, but both path totals are ten. Summing
marginal quantiles would therefore give the wrong aggregate quantile. Full paths
retain dependence; individual horizons are never independently resampled. At
quantile .75, paths `[[0,0],[10,10]]` give target 20, whereas the four equally
likely independent combinations give target 10.

For n sorted samples, select the first value with empirical CDF >= quantile
(mathematically index `max(0, ceil(quantile*n)-1)`). The implementation compares CDF
values directly to avoid floating-point rank errors at exact boundaries. Quantile zero
selects the minimum and quantile one selects the maximum. There is no interpolation.
An input quantile is neither a fill-rate guarantee nor a general cost-optimal
newsvendor fractile for positive-lead-time lost sales. The existing percentile
forecast policies still sum supplied marginal forecasts as a distinct heuristic;
they have not been reinterpreted as exact aggregate coverage.

## Per-item candidate search and standard rows

```python
from replenishment import BaseStockPolicy, optimize_policy_candidates

fit = optimize_policy_candidates(
    candidates={f"S={s}": BaseStockPolicy(s) for s in [4, 8, 12]},
    periods=6, demand=[2, 3, 1, 5, 2, 4], initial_on_hand=0, lead_time=2,
    holding_cost_per_unit=.5, stockout_cost_per_unit=5,
)
print(fit.label, dict(fit.candidate_costs))
# Use fit.policy unchanged on a separate held-out demand trace.
```

`optimize_policy_candidates` materializes callable or iterable demand once and
compares candidates against the same realized trace. It consumes only the
requested period count, leaving an iterable's held-out tail untouched and
supporting unbounded demand streams. It minimizes simulated
total cost, breaking ties by input order, and returns the label, policy, winning
training simulation and immutable candidate-cost mapping. Supply reusable
policies (the built-ins are immutable); stateful custom policies/providers must
be managed by the caller. This is finite search, not a claim of global optimality.
Fit all distributions, samples and parameters on history and evaluate separately.

`build_article_configs_from_standard_rows(rows, policies={item_id: policy})`
uses `row.demand`, the existing costs/lead time, and the supplied item policies.
It does not silently substitute `actuals`. All-forecast rows use `current_stock`
by default; historical/mixed rows use `initial_on_hand`. `use_current_stock`
overrides that choice. Configs work with `simulate_replenishment_for_articles`
and the existing decision-row converter.

New-policy simulations include policy identity/family, review cadence and compact immutable
parameters in metadata. Quantile metadata records input kind/count rather than
embedding potentially large path sets. Decision rows for simulations carrying
the independent-item family report their actual simulated stock and lost sales, including
immediate replenishment. Static threshold/target columns reflect the new policy's
parameters; forecast-derived thresholds are suppressed for constant-order and
probabilistic policies rather than misrepresenting their decisions. Legacy forecast workflows retain their independently
reconstructed `current_stock` path for compatibility; immediate orders now enter
that path after demand. Do not mix those baseline conventions in comparisons.

Implicit time aggregation for these six policies is rejected when window != 1;
use `review_period` in the ordinary simulator. Identity aggregation retains their
metadata. This avoids silently changing demand distributions and cost resolution.
Existing forecast classes keep their established offsets, target/pipeline
heuristics and aggregation behavior; no blanket timing correction is claimed.

## Examples and extension boundaries

The six baseline `notebooks/*_policy_example.ipynb` walkthroughs use the same
seeded, generated standard-row dataset, with seasonal, intermittent and rising
articles. Each follows the established DataFrame workflow: split, fit each item,
inspect parameters and hypothetical decisions, build evaluation configs, simulate,
inspect standard decision rows, then use `plot_replenishment_decisions` for
individual and aggregate stock/missed-sales views. Candidate counts and tuning
runtime are displayed; rounded duplicate integer candidates are removed where
appropriate. Model parameters and sample distributions use training rows only.
The notebooks retain hand-calculated reference checks and reconcile plotted
cumulative loss/share to simulation totals. They report cost, fill rate, lost
units, inventory and order frequency. Cold starts, warmup and terminal treatment
are explicit. These are readable workflow examples, not comprehensive research
benchmarks or evidence of universal policy dominance.

| Roadmap direction | Separate from these six policies? | Reason |
| --- | --- | --- |
| Per-item MOQ, case packs, max orders, allowed dates | Yes, operational-composition follow-up | Can be solver-free, but needs explicit feasibility/rounding and tuning semantics |
| Shared budgets, warehouse space, receiving capacity | Yes | Couples item decisions; needs coordinated allocation (not necessarily LP in every implementation) |
| Scenario rolling-horizon optimisation | Yes | Solver-based planning is excluded even for a single item |
| Echelon/network allocation | Yes | Requires a network simulator and coupled decisions |
| Structured neural policies / HDPO | Implemented as optional advanced methods | See [advanced methods](advanced_item_policies.md) for residual learning, differentiable training and integer validation |
| Robust/risk-sensitive optimisation | Empirical mean/CVaR/worst-scenario methods implemented | Solver-free per-item search/training is supported; general distributionally robust formulations remain separate |
| Perishables / dual sourcing | Yes | Requires age buckets or multiple supplier pipelines and costs |
| Rust acceleration | Yes | Profile first; not needed for correctness or these APIs |
