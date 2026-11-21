# Solver-based rolling-horizon policies

Install the optional solver backend:

```bash
pip install -e ".[solver,dev]"
```

The `solver` extra installs SciPy >=1.11,<2, whose
[`scipy.optimize.milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
uses HiGHS. No Gurobi installation or commercial license is required. SciPy is
imported at the first solve; package imports, ordinary policies, and frozen solver
policy construction work without it. Solver planning is CPU based and does not
require the optional neural/PyTorch extra.

## Single-item methods

```python
from replenishment import (
    SolverItemConfig, SolverOptions, DeterministicRollingHorizonPolicy,
    ScenarioRollingHorizonPolicy, RiskObjective, simulate_replenishment,
)

item = SolverItemConfig(
    lead_time=2, max_order_quantity=20, review_period=1,
    holding_cost_per_unit=.4, stockout_cost_per_unit=8,
    order_cost_per_order=2, order_cost_per_unit=.3,
    min_order_quantity=4, pack_size=2,
)
policy = DeterministicRollingHorizonPolicy(item, forecast=[3, 5, 4, 3])
scenario_policy = ScenarioRollingHorizonPolicy(
    item, demand_scenarios=[[3, 5, 4, 3], [1, 2, 8, 4]],
    objective=RiskObjective("mean_cvar", alpha=.5, risk_weight=.5),
)
result = simulate_replenishment(
    periods=6, demand=[2, 3, 5, 4, 2, 3], initial_on_hand=8, lead_time=2,
    policy=scenario_policy, holding_cost_per_unit=.4, stockout_cost_per_unit=8,
    order_cost_per_order=2, order_cost_per_unit=.3,
)
```

The deterministic policy optimizes one integer demand forecast. The scenario
policy jointly optimizes equal-weight demand paths. Both solve again at eligible
review dates and execute only the first order. Review dates are absolute periods
0,R,2R,...; demand is never aggregated. Orders are zero or a pack multiple between
MOQ and the finite maximum, inclusive. A positive MOQ with no feasible pack multiple
is rejected. A cap of zero supports an item that cannot be replenished.

Supply static data or an as-of callback `forecast(state)` / `demand_scenarios(state)`.
Static data is copied into tuples and repeated relative to each decision. Callbacks
require an explicit `horizon=H`, must return that many **future** demands per path,
and must not access future realized demand. Use integer forecasts, rounding externally
if needed; fractional, negative and boolean demand values are rejected. Callbacks
receive the immutable post-demand `InventoryState`, including dated pipeline.

`policy.plan(state)` returns full diagnostics without mutating the policy:

- `first_orders`: immutable item-to-integer mapping (single-item adapters use key `"item"`).
- `orders[item][scenario][k]`: orders at t+k, k=0,...,H-1.
- `inventories[item][scenario][k]`, `lost_sales[...]`: end stock/lost demand at t+k+1.
- `scenario_costs`, `objective_value`, `objective` (including alpha and risk weight),
  `mean_cost`, `cvar_cost`, `worst_cost`, `horizon`, `period`, `terminal_value`.
- `status`, `message`, `mip_gap`, `dual_bound`, `solve_seconds`.

The ordinary simulator returns existing result/summary types and policy metadata;
standard-row config/decision helpers work with these policies. Requested time
aggregation beyond one is rejected. Use review cadence instead. Planning costs
live in `SolverItemConfig`; ordinary independent simulation costs are separate
inputs, permitting evaluation under different costs. Set them identically when
comparing the plan's cost model with execution.

## Coordinated multi-item planning

The ordinary `simulate_replenishment_for_articles` still runs articles independently.
To enforce shared limits, use the explicit joint planner/simulator:

```python
from replenishment import (
    ArticleSimulationConfig, SharedResourceLimits,
    simulate_coordinated_replenishment,
)

articles = {
    uid: ArticleSimulationConfig(
        periods=6, demand=trace, initial_on_hand=8, lead_time=item.lead_time,
        policy=policy, holding_cost_per_unit=.4, stockout_cost_per_unit=8,
        order_cost_per_order=2, order_cost_per_unit=.3,
    )
    for uid, trace in {"A": [2,3,5,4,2,3], "B": [4,1,3,2,5,2]}.items()
}
joint = simulate_coordinated_replenishment(
    articles=articles,
    demand_scenarios={"A": [[3,5,4,3], [1,2,8,4]],
                      "B": [[2,4,3,2], [5,3,2,6]]},
    limits=SharedResourceLimits(
        max_order_units=16, procurement_budget=8, max_receiving_units=16,
    ),
)
# Existing decision/plot helpers accept joint.simulations.
# joint.plans and joint.resource_use record each period's diagnostics.
```

All items advance through receipt and current demand before one coordinated call.
Callable demand and demand iterators are consumed one period at a time, so future
actuals are not sampled before the current decision.
The explicit joint scenario source replaces the per-item policy forecast sources.
Scenario index s must describe the same joint outcome across all items; independently
shuffling each item's paths destroys correlation. A dynamic source receives an
immutable `states` mapping and returns the same item/scenario/horizon shape.
All articles need equal simulation lengths and a common planning horizon. Lead
times, pack sizes and review cadences may differ. Simulation costs and lead times
must match each policy's planning config in coordinated runs.

Shared limits are optional, finite nonnegative scalars, applied in **each period**:

| Limit | Accounting |
| --- | --- |
| `max_order_units` | Sum of placed order quantities |
| `procurement_budget` | Variable purchase cost plus fixed setup charges for placed orders |
| `max_receiving_units` | Committed receipts plus new arrivals, including post-demand immediate orders |

Limits apply in every scenario and throughout the planning horizon. The simulator
also independently checks executed quantities, expense and receipts. It never
clips/rescales an action. These are not cumulative episode budgets or warehouse
stock-space constraints. Inventory remains item-specific; shared purchasing
resources do not allow transferring stock between items. Time-varying costs/limits,
weighted resource use, stock-space capacity, networks, expiry and dual sourcing
are outside this API.

For direct planning call `plan_replenishment(states=..., items=...,
demand_scenarios=..., limits=..., objective=..., options=...)`. Observations must
share a period and contain a consistent pipeline: positive quantities on distinct
future dates no later than t+L, summing to `on_order`; backlog must be zero.
Pass `received_this_period=<total>` when a receiving limit must include receipts
that already arrived before these observations. Their stock is already in the
states: this parameter consumes capacity without adding stock a second time.
The coordinated simulator supplies it automatically. Its own initial pipeline
is empty, like the existing simulator; the direct planner supports dated pipelines.

## Formulation and information available

An observation is **after demand at t and before ordering**. Forecast/scenario
entry zero is demand at t+1. The H-period horizon has demands t+1,...,t+H and
orders t,...,t+H-1. Require H >= max(1,L)+R-1 for each item.

For positive L, q at u arrives before demand u+L. At L=0, q arrives after current
demand, contributes to that period's holding cost, and first serves demand u+1.
Every scenario enforces inventory flow and `lost=max(0,demand-available_stock)`;
a stockout binary prevents the optimizer from discarding available inventory even
when costs are zero. Nonnegative post-demand stock and lost sales are mutually
exclusive. Integer pack/setup variables enforce quantity bounds and fixed costs.
Finite stock bounds are derived from current stock, committed receipts and order
caps rather than an arbitrary large constant. Inputs outside the supported
numerical quantity range (2^45 for each demand or aggregate stock bound) are
rejected; choose sensible physical units and avoid extreme cost scaling.

Costs comprise future end-of-period holding, lost units, variable procurement and
fixed order charges. Current holding is already incurred except for added L=0
stock, whose holding cost is included. No terminal salvage/target is assumed and
there is no fulfillment after the planning horizon. Decisions that cannot serve
any demand within the horizon are disabled. A finite horizon can encourage
terminal depletion; inspect horizon sensitivity instead of treating its solution
as infinite-horizon optimal. The evaluation simulator also has no post-episode
fulfillment and charges all placed orders, including its final-period decisions;
it does not reveal the evaluation episode's end to the policy.

**Nonanticipativity applies at every stage.** At decision t+k, scenarios sharing
all items' first k demands share the same action variables. Current orders are
therefore common; later orders may differ only after distinguishing demand has
been observed. This is a finite empirical scenario tree, not unrestricted
perfect-foresight recourse. A unique early observation can identify a whole sampled
path within this tree. More representative trees/scenarios are necessary before
using it as a model of operational uncertainty. True demand is observed even when
sales are censored by stockouts; learning from censored sales is not implemented.

The existing `RiskObjective` mean, CVaR, mean/CVaR blend and maximum supplied-
scenario cost are supported. The linear CVaR epigraph is
`eta + sum(max(cost_s-eta,0)) / ((1-alpha)*N)`; it includes fractional boundary
probability and matches `RiskObjective.score`. Risk applies to **total joint
scenario cost**, not independently summed item CVaRs. Worst case protects the
supplied finite set only; no distributionally robust guarantee is implied.

## Solver status and numerical checks

`SolverOptions(time_limit=30, mip_rel_gap=0)` defaults to a 30-second limit per
solve and zero relative gap. A configured nonzero gap permits an optimum within
that tolerance; inspect reported gap and bound. Infeasible, unbounded, limit or
error statuses raise `PlanningError`, even if the solver has an incumbent. There
is no implicit fallback policy. Missing SciPy raises an actionable install error.

Monetary costs are normalized before building the objective and risk epigraphs;
budget rows are normalized separately. Integer resource caps are floored to whole
units, and execution comparisons are exact for units. Budget replay allows only
floating-point roundoff (eight ULPs, no absolute monetary epsilon); zero budgets
allow no positive expense. Reported gaps and dual bounds must satisfy the requested
tolerance even when HiGHS reports successful termination.

Before returning actions the backend checks finite vectors, integrality within
1e-6, rounded-vector bounds/linear feasibility within 1e-5, risk-score agreement,
and independent integer replay of stocks, losses, costs and resources. Daily solver
decision rows require one date per snapshot and never average forecasts by review cadence.
Their point-forecast fields are unaggregated standard-row display diagnostics, not
a serialization of scenario paths; inspect the plan for scenario-specific forecasts/actions. Solver
optimality is for the supplied finite model and tolerances, not the true unknown
demand process. Different solver versions may choose different equal-cost orders.

## Maximize revenue or inventory operating profit

Both objectives use `stockout_cost_per_unit=selling_price` as the value of lost
sales, with the true purchase/holding/fixed costs retained in `SolverItemConfig`.
For exogenous demand and fixed prices, potential revenue is independent of actions:

- `cost_objective="lost_sales"`: minimize selling-price-weighted lost sales only,
  equivalent to maximizing **expected revenue** with `RiskObjective("mean")`.
- `cost_objective="total_cost"` (default): also include procurement and holding,
  equivalent to maximizing **expected operating profit** with the mean objective.

`cost_objective` is supported by `plan_replenishment`, both single-item adapters,
and `simulate_coordinated_replenishment`. In coordinated runs the explicit joint
argument selects the mode for all items, and the actual mode is stored in result
metadata and each plan. Defaults preserve the existing cost-minimization behavior.

```python
# articles contain solver policies; selling prices are the shortage weights.
revenue_run = simulate_coordinated_replenishment(
    articles=articles, demand_scenarios=joint_scenarios,
    limits=SharedResourceLimits(procurement_budget=35),
    objective=RiskObjective("mean"), cost_objective="lost_sales",
)
profit_run = simulate_coordinated_replenishment(
    articles=articles, demand_scenarios=joint_scenarios,
    limits=SharedResourceLimits(procurement_budget=35),
    objective=RiskObjective("mean"), cost_objective="total_cost",
)
```

Revenue mode excludes purchasing/holding **from the score only**; true unit and
fixed charges still consume shared procurement budget, appear in resource logs,
and are charged in simulation summaries. Do not zero purchase costs to maximize
revenue: that would also remove them from budget accounting. In this mode the
plan's scenario/mean/CVaR/worst costs describe weighted lost sales only. Simulation
summaries continue to report the full configured cost including forgone revenue.
The current `SimulationSummary.total_cost` is not actual financial expenditure.

Report revenue as `selling_price * summary.total_fulfilled` per item; profit is
that revenue minus `summary.ordering_cost` and `summary.holding_cost`. Do not
subtract `stockout_cost` again: it is forgone revenue rather than a cash expense.
Set the shortage weight to selling price, not unit margin, since purchases are
already charged. Positive initial stock is treated as previously paid/sunk; the
financial examples start with zero stock. All ordered stock is paid, including
remaining inventory/pipeline at the end; no terminal salvage is assumed.

This equivalence is for the mean objective. CVaR/worst cost are not generally
CVaR/worst negative profit when potential revenue varies between scenarios.
Policies optimize their supplied finite-horizon forecast model, not guaranteed
realized revenue/profit on held-out demand. Shared budgets reset each period;
this is not cash-flow management with sales proceeds, carryover, or financing.

Executed examples compare both objectives on identical demand with different
item prices/margins, inspect plans, reconcile financial ledgers, and plot service,
stock, revenue, profit and budget use:

- [Revenue maximization](../notebooks/revenue_maximization_policy_example.ipynb)
- [Profit maximization](../notebooks/profit_maximization_policy_example.ipynb)

## Executed examples and verification

- [Deterministic planner](../notebooks/deterministic_rolling_horizon_policy_example.ipynb)
- [Scenario planner](../notebooks/scenario_rolling_horizon_policy_example.ipynb)
- [Coordinated multi-item planner](../notebooks/coordinated_solver_policy_example.ipynb)

Each generates the same three-item dataset, fits only historical demand, inspects
plans, evaluates a chronological holdout and displays decision tables plus three
item plots and an aggregate stock/missed-sales plot. The joint example adds a
shared-resource utilization plot and a deterministic baseline under identical
limits. Independent examples compare training-tuned feasible constant orders.
Costs, demand-weighted fill rate, lost units, inventory, order frequency and runtime
are shown; no general superiority claim is made. Six joint residual paths and
alpha=.5 provide three empirical tail observations. Outputs are embedded in the
notebooks and saved under `docs/plots/`.

Tests compare small plans against independent exhaustive order enumeration,
including intermediate nonanticipativity, fractional CVaR, setups, packs, mixed
lead times and scarce shared resources. Replay, decision tables, resource
accounting, unsuccessful solver statuses, invalid vectors and no-SciPy imports
are also checked. CI has a separate optional-solver job.
