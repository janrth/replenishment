# Solver-based replenishment planning

## Intent and agreed scope

On `feat/solver_methods`, add both independent-item and coordinated multi-item
rolling-horizon optimization using optional open-source HiGHS. This is the user's
selected scope. Preserve the existing policies and independent simulation API.
Provide an executed example notebook for each new method, following the existing
generated-data, chronological-fit, inspect, simulate, decision-table and item/total
plot workflow. Leave the unrelated untracked VN2 notebook alone.

The roadmap is a proposal. Its scenario-planning family is the basis for this
work, not a request to implement networks, perishables, dual sourcing or Rust.

## Approaches

1. Independent-item solver planning only: simpler, but omits the shared constraints
   deferred from the previous PR.
2. Independent and coordinated planning with optional HiGHS: selected by the user.
   One formulation supports deterministic forecasts and joint demand scenarios.
3. Add Gurobi too: not selected; unnecessary commercial dependency and adapter work.

Implementation proposal: use `scipy.optimize.milp`, which bundles HiGHS, under a
`solver` extra (`scipy>=1.11,<2`). Keep all solver imports lazy. Sparse model
assembly and strict status checking avoid mandatory dependencies and silent
fallbacks. This backend choice is the implementer's recommendation.

## Public components

Names are proposals; the builder may adjust them consistently.

- Frozen item planning specification: lead time, review cadence, nonnegative
  holding/shortage/unit/fixed ordering costs, finite integer maximum order,
  optional minimum order and pack multiple. Bounds apply to executed orders.
- Immutable demand scenarios: equal-weight, rectangular nonnegative integer
  demand paths, with matching scenario index across items to preserve dependence.
  Values start at **the next demand event**, never current already-served demand.
- Planner returning first orders, scenario-contingent future orders, projected
  inventories/lost sales, scenario costs, objective, solver status/gap/bound and
  elapsed solve time. Provide inspectable results rather than a mutable cache.
- Deterministic rolling-horizon policy and scenario rolling-horizon policy use
  the existing `order_quantity_for(InventoryState)` protocol. Static forecasts or
  as-of callbacks are supported. Their explicit `plan(state)` returns diagnostics.
- Coordinated planner receives all items' post-demand observations in one call.
  Scenario providers see immutable states and period, not future actual demand.
- Joint simulator returns per-item `SimulationResult`s plus per-period planning
  diagnostics/resource use. These results feed the existing decision and plotting
  helpers, preserving the standard notebook workflow.

## Timing and exact model

Keep receipt -> demand -> order lost-sales semantics. At post-demand time t,
the horizon contains demand t+1 through t+H. Decisions are at t through t+H-1.
At positive lead time L, an order at u arrives before demand at u+L. At L=0
it arrives after demand at u, incurs that period's holding cost, and first serves
demand u+1. Current post-demand stock and dated committed receipts are fixed input.
Require consistent on_order/arrival_schedule and no backlog; validate all quantities,
finite costs, identifiers, horizon shapes, and matching lead times.

Use integer order/pack variables and optional binary setup variables for minimum
orders/fixed charges. Enforce review dates using absolute periods. Per-scenario
inventory balance and binary stockout disjunctions enforce actual lost sales
and immediate fulfillment even with zero holding or shortage costs. Derive finite
big-M bounds from initial stock, committed receipts, finite order caps and demand;
do not choose an arbitrary magic bound. Forecast demand is integer; no silent
rounding of fractional forecasts.

Include variable/fixed ordering costs for all horizon decisions, future end-of-
period holding and lost-sales costs, and the extra current holding from immediate
orders. Already incurred current costs are constants and excluded. Default terminal
value is zero, with no post-horizon fulfillment; disable decisions whose receipts
cannot serve any modeled demand (including immediate orders after final demand).
Require H >= max(1,L)+R-1 for each item so a review decision can cover its protection
period. Expose horizon and terminal convention in documentation and diagnostics;
longer horizons and horizon sensitivity are the mitigation for terminal depletion.
Do not claim infinite-horizon optimality.

## Scenarios and information

Deterministic planning is the one-scenario special case. For stochastic planning,
all current orders are shared across scenarios. At future decision t+k, orders
must match for scenarios whose demand histories through k are identical. In joint
planning use the joint history of all items. Thus decisions may depend on demand
already observed but never on unrevealed scenario identity. This is a finite
empirical scenario tree; unique early observations can reveal an empirical path,
and this limitation must be documented. Do not describe unrestricted perfect-
foresight recourse as implementable stochastic optimization.

Support the existing `RiskObjective` mean, CVaR, mean/CVaR blend and worst-case
objectives via linear epigraph variables over complete scenario costs. Fractional
CVaR tails must match the existing risk scorer. Report mean/tail/worst separately.
Only the first order is executed; re-solve after the next observed demand.

## Shared constraints and joint simulation

Optional scalar, per-period shared limits:

- Order units: sum of placed quantities, in every modeled period/scenario.
- Procurement budget: sum of variable purchases plus setup charges at placement.
- Receiving units: sum of dated committed receipts and new arrivals, irrespective
  of demand; immediate receipts count at placement time.

Limits cover the whole modeled horizon and all scenarios. Actual joint simulation
must independently check limits for executed actions/receipts, never silently
clip or rescale a jointly solved action. Validate incompatible initial pipelines
and report infeasibility explicitly. These are per-period limits, not a cumulative
budget. Receiving commitments beyond H cannot arise from enabled new decisions.
Warehouse stock-space constraints, weighted resource units, time-varying costs and
limits, cumulative budgets and supplier networks are outside this implementation.

The new joint loop advances all items synchronously, then asks the planner once
per period. Equal simulation lengths are required; lead times/cadences may differ.
Results must match independent simulation with the same decisions when constraints
are absent. Physical limits are distinct from per-item policy caps. Do not add a
global allocator to the existing per-item loop implicitly.

## Failures, compatibility and optional dependency

Require a successful optimal solve within configured tolerances before returning
an order. Default relative MIP gap is zero; allow a configured nonzero tolerance
and report the achieved gap/bound. Infeasible/unbounded/limit/error statuses raise
a specific planning exception including status and message. No silent zero-order
fallback or unchecked time-limit incumbent. Validate returned integer quantities
and model feasibility to numerical tolerance before execution.

Core package imports and old policies run without SciPy. The first solve without
the extra raises an actionable install message. Integrate single-item policies
with standard-row configs, lead-time validation, metadata and decision output.
Reject time aggregation beyond one, retaining review cadence. Joint metadata must
also use snapshot-based decision accounting rather than legacy forecast heuristics.

## Verification and deliverables

- Exhaustive tiny order enumerations independently verify deterministic and
  scenario-tree optimum costs, including shared constraints and CVaR.
- Hand-worked receipt cases cover L=0,1,2, pipeline timing, review cadence,
  fixed costs, MOQ/pack rounding, terminal cutoff and mandatory lost-sales dynamics.
- Test scenario nonanticipativity beyond the first stage; forecasts may differ
  without allowing early different decisions. Test joint demand correlation.
- Test constraint tradeoffs (scarce resources allocated to high-shortage-cost
  items), infeasibility, solver failure/limits and numerical acceptance.
- Test single/joint simulation parity, decision tables and resource reconciliation,
  as well as no-SciPy imports and preservation of existing tests.
- Add optional-solver CI coverage, README/changelog and a formulation/API guide.
- Three executed notebooks: deterministic rolling horizon, scenario rolling horizon,
  and coordinated multi-item planning. Each has generated three-item data, training-
  only forecast/scenario fitting, held-out simulation, policy/plan inspection,
  decision tables, per-item and aggregate stock/missed-sales plots. The coordinated
  example also plots binding shared limits and compares a feasible baseline.
  Save embedded outputs and plot assets; assert stock/loss/resource reconciliation.
  Label all comparisons illustrative, with forecast and terminal assumptions.

## Builder discretion

Module layout, exact API names, sparse matrix builder internals, example sizes,
training-only forecast models, and test fixture organization may be settled by
the builder. Reuse existing validation/metadata conventions where practical.
No commit, push, new PR or merge is requested yet for this branch.
