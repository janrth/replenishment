# Six independent item-level replenishment policies

## Intent and agreed scope

Implement the six missing Stage 1–2 policies in the supplied replenishment
roadmap. The user selected all six rather than a first change containing only
the four Stage 1 baselines. Each new policy gets a separate executable example
notebook that verifies its decisions and demonstrates chronological tuning and
evaluation. All decisions and tuning operate on one item at a time, without
LP/MIP solvers or shared resource constraints.

The roadmap is a proposal, not a source of instructions overriding the user's
request. Its reviewed baseline matches this checkout (`ff78fe3`). The existing
suite passes: 58 tests. An existing untracked VN2 notebook is unrelated and must
be preserved.

## Approaches

1. Four static baselines first: smallest initial change, but postpones the two
   probabilistic policies the user also wants.
2. All six policies plus the observation/timing foundations they need: selected
   by the user. This provides the complete independent-item Stage 1–2 scope.
3. Implement the full roadmap: rejected because shared constraints, solver-based
   planning, learned policies, and new inventory environments exceed this work.

The API details below are implementation proposals, not decisions already
approved by the user. Reuse the existing Python dataclass/protocol design and
standard library numerical tools; add no mandatory dependencies.

## Simulator contract and compatibility

Keep existing event order: receive previously placed orders, fulfill current
demand, observe the state, then decide/place an order. A lead-time-zero order
arrives after current demand and cannot fulfill lost demand retroactively.
Unfulfilled demand disappears; this work does not add backlogging.

Add an immutable receipt schedule to `InventoryState`, using a defaulted field
so existing custom policies and four-argument state constructors keep working.
Represent it as `(absolute_arrival_period, quantity)` pairs, sorted by arrival,
containing positive outstanding receipts after current demand and before the new
order is placed. Its quantities sum to `on_order` for simulator-generated states.
PIL rejects externally constructed inconsistent schedules rather than silently
guessing receipt timing. Other policies continue using inventory position.

For the two new lead-time-dependent policies, simulator entry validates that
policy and simulation lead times match, even when the initial pipeline is empty.
This check applies to these built-ins without imposing a new requirement on
custom policies or changing existing forecast classes.

Add unambiguous lost-sales aliases to snapshots and summaries while retaining
the existing `backorders`/`total_backorders` fields and their compatibility
behavior. Document that per-period `backorders` remains zero in lost-sales mode.
Verify snapshot stock accounting, including zero lead time; correct misleading
snapshot bookkeeping if necessary, without changing demand/order trajectories.

Review cadence is independent of aggregation: a policy with review period `R`
orders when `period % R == 0`, starting at period zero. New APIs must not silently
interpret review cadence as aggregating demand. Existing forecast-policy behavior
is outside the refactoring scope. Document any timing inconsistencies found in
existing heuristics instead of silently changing their public behavior.

For end-of-demand decisions, the first demand served by a new order is at
`t + max(1, L)`. The next review's order can first serve demand at
`t + R + max(1, L)`. A static inventory-position protection horizon therefore
spans future periods `t+1` through `t + H`, where
`H = max(1, L) + R - 1`. This includes demand served by existing pipeline and
on-hand stock before the new receipt. It is a targeting heuristic for general
lost sales, not a claim of optimality. Hand-calculated traces cover lead times
0, 1, and 2, varying demand, and review periods above one.

## Policy APIs and rules

Use frozen dataclasses and `order_quantity_for(state) -> int` throughout.
Validate integer, finite, nonnegative quantities and positive integer cadence;
reject booleans as integer parameters. Threshold `s` may be any integer, target
`S` must be nonnegative, and `S >= s`. Reject invalid parameters at construction.

| Public class | Parameters beyond `review_period=1` | Decision on review dates |
| --- | --- | --- |
| `BaseStockPolicy` | `target_stock` | `max(0, S - inventory_position)` |
| `ConstantOrderPolicy` | `quantity` | Always `quantity` |
| `CappedBaseStockPolicy` | `target_stock`, `cap` | `min(cap, max(0, S - inventory_position))` |
| `sSPolicy` | `reorder_point`, `target_stock` | If position <= s, `max(0, S - position)`, else zero |
| `ProjectedInventoryLevelPolicy` | `target_stock`, `lead_time`, discrete demand PMF | Raise expected pre-demand inventory at new receipt to target |
| `ProtectionPeriodQuantilePolicy` | `quantile`, `lead_time`, samples/provider | Order up to the aggregate-demand quantile, subtracting inventory position |

All policies return zero off review dates. A policy cap is a decision parameter,
not physical supplier capacity. Correct `ReorderPointPolicy`'s description to
say fixed-quantity `(r,Q)`, retaining its behavior.

### Projected inventory-level (PIL)

Start with a finite discrete, stationary, independent demand PMF. Validate
nonnegative integer support, finite nonnegative probabilities, nonempty support,
and total probability one within a documented numerical tolerance. Normalize
small floating-point summation error only.

Project the inventory distribution from the current post-demand on-hand state
through outstanding receipts and future demands strictly before the new order's
arrival. Apply `max(0, inventory + receipt - demand)` each period; this handles
lost sales rather than subtracting expected demand from expected inventory.
Include old receipts at the new arrival time before computing expected stock,
but do not consume demand at that arrival time yet. Choose
`ceil(max(0, target_stock - expected_stock))`, with documented floating-point
handling at integer boundaries. For lead times zero and one, no intervening
demand occurs. Reject schedules incompatible with the configured lead time.

Use exact probability propagation as the reference implementation. No Monte
Carlo estimator is needed in this change, so estimation seeds/error are not an
API concern. Document state-space growth and restrict examples to modest cases.
Compare the recursion with independent exhaustive demand-path enumeration in
tests. Show equal total pipeline with different receipt dates producing different
decisions. This is a translation of the expected-stock-at-arrival idea into this
simulator's timing, not a reproduction of published performance guarantees.

### Protection-period quantile

Accept either aggregate-demand samples or equal-weight joint demand paths.
A provider may supply these for the current decision using only information
available at that time; static samples are also supported. Require exactly one
representation. Joint paths have exactly `H` nonnegative integer demand entries;
sum each complete path to preserve dependence. Do not resample individual
horizons independently. Validate nonempty finite samples and `0 <= quantile <= 1`.
Use the discrete empirical inverse CDF, with quantile zero selecting the minimum
and quantile one selecting the maximum; there is no interpolated fractional
target. Clearly document the provider's decision-time/no-leakage responsibility.

Tests and the notebook contrast sum-of-marginal-quantiles with the quantile of
the sum, and show dependence changing aggregate uncertainty. Forecast coverage
and inventory fill rate are separate reported metrics.

The provider receives only `InventoryState`. It can use `state.period` to look up
externally prepared as-of forecasts. This change does not add observed-demand
history plumbing or an online forecasting/training interface to the simulator.

## Per-item tuning and integration

Provide a generic candidate-policy evaluator/selector rather than six unrelated
optimizers. It accepts the existing single-item simulation inputs plus a finite
ordered mapping of candidate labels to policy instances; returns candidate costs,
the selected label/policy, and its simulation. Minimize total cost; ties choose
the first candidate. Materialize iterable demand and callable demand (once per
period) so every candidate receives the same realized trace, including with
stateful/stochastic demand callables. Reject empty candidate sets. No claim is made
that a finite candidate search finds a globally optimal policy.

Jointly enumerate target/cap and s/S pairs. Tune PMF/quantile/target choices on
training data only and freeze them for evaluation. Support standard table rows
through a reusable per-item config builder, without duplicating forecast-specific
IO logic. Export the new public policies and tuning/config helpers and record
policy identity, cadence, and chosen parameters in simulation metadata without
discarding existing metadata.

The config builder accepts caller-supplied policies keyed by item, uses
`row.demand` as simulated demand, and follows the existing initial/current-stock
selection rule (all forecast rows use current stock by default). It does not
implicitly substitute `actuals` or introduce another tuning-on-actuals workflow.
Standard decision-row output must work for these configs: verify its stock
accounting against simulator snapshots for zero and positive lead times and fix
discrepancies needed for accurate output. Its independently reconstructed stock
path must not report false shortages after immediate replenishment.

For aggregation helpers, explicitly reject unsupported transformations of the
new policies rather than silently changing their time units or reusing an
untransformed probabilistic model. The ordinary unaggregated simulator and review
cadence remain fully supported. Any exact supported transformation may be added
only when its semantics are tested and documented.

## Six example notebooks

Create `base_stock_policy_example.ipynb`, `constant_order_policy_example.ipynb`,
`capped_base_stock_policy_example.ipynb`, `s_s_policy_example.ipynb`,
`projected_inventory_level_policy_example.ipynb`, and
`protection_period_quantile_policy_example.ipynb` under `notebooks/`.

Each notebook is self-contained, uses deterministic synthetic data, contains
assertions for a transparent worked case, tunes candidates on a chronological
training window, and evaluates the frozen choice on a held-out window. Use the
same held-out demand for comparisons and disclose initial stock, warmup (zero
unless stated), and terminal treatment (no salvage or post-horizon fulfillment).
Report total cost, fill rate, lost units, average/peak stock, order frequency,
runtime, and candidate count. Include a stock/order plot and relevant comparator
policies. Show more than one lead time/shortage penalty where useful, and avoid
performance superiority claims from a single illustrative trace.

PIL additionally verifies the exact projection; the quantile notebook includes
correlated paths and separate aggregate coverage. The constant-order example
includes candidate mean order rates at/below mean demand per review cycle.
Keep runtimes small enough to execute all six during validation. Link the
notebooks and describe the timing/scope in README and update the changelog.

## Exclusions and validation

Exclude Stage 3 order-constraint composition, shared budgets/capacity/allocation,
solver-based rolling-horizon planning, learned/robust/network policies,
perishables, dual sourcing, Rust, and new backlog modes. Item-specific operational
constraints need no global solver in principle but are additional scope, not one
of these six policies.

Validate with meaningful policy/parameter edge tests, hand-calculated event
traces, exact projection references, quantile dependence examples, generic tuning
and IO integration tests, all existing tests, and execution of all six notebooks.
Inspect generated plots for labels and legibility. Do not overwrite the user's
existing untracked notebook. Builder may choose helper names, module organization,
specific example demand series, candidate grids, and numerical tolerances, with
those choices documented in public docstrings and tests as appropriate.
