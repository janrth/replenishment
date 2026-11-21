# replenishment

[![CI](https://github.com/janrth/replenishment/actions/workflows/ci.yml/badge.svg)](https://github.com/janrth/replenishment/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/replenishment.svg)](https://pypi.org/project/replenishment/)
[![Python](https://img.shields.io/pypi/pyversions/replenishment.svg)](https://pypi.org/project/replenishment/)

`replenishment` is a Python library for stock replenishment simulation, policy optimization, and decision visualization.

It is designed for teams that want to:

- simulate replenishment policies against historical or synthetic demand
- optimize mean-forecast safety stock, `k*RMSE`/`k*MAE`, and percentile-target policies
- compare policies visually on replenishment time series
- move cleanly between CSV/DataFrame inputs and simulation configs

## Supported Python Versions

The project is tested on Python `3.10`, `3.11`, `3.12`, and `3.13`.

## Installation

### Install from PyPI

```bash
pip install replenishment
```

### Create a Development Environment with `uv`

Choose any supported Python version. `3.13` is the default below.

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
```

To work on another supported version, replace `3.13` with `3.10`, `3.11`, or `3.12`.

Run the test suite with:

```bash
uv run pytest
```

## Quickstart

This example optimizes a mean-forecast safety stock factor on a backtest window
and applies the learned policy on the forecast horizon.

```python
from replenishment import (
    generate_standard_simulation_rows,
    optimize_point_forecast_policy_and_simulate_actuals,
    plot_replenishment_decisions,
    replenishment_decision_rows_to_dataframe,
    split_standard_simulation_rows,
    standard_simulation_rows_to_dataframe,
)

rows = generate_standard_simulation_rows(
    n_unique_ids=1,
    periods=18,
    start_date="2031-01-01",
    frequency_days=30,
    forecast_start_period=10,
    history_mean=52,
    history_std=6,
    forecast_mean=48,
    forecast_std=5,
    lead_time=2,
    initial_on_hand=30,
    current_stock=30,
    seed=7,
)
backtest_rows, eval_rows = split_standard_simulation_rows(rows)

optimized, _, decision_rows = optimize_point_forecast_policy_and_simulate_actuals(
    backtest_rows,
    eval_rows,
    candidate_factors=[0.8, 0.9, 1.0],
)

rows_df = standard_simulation_rows_to_dataframe(rows, library="pandas")
decision_df = replenishment_decision_rows_to_dataframe(decision_rows, library="pandas")

example_id = decision_df["unique_id"].iloc[0]
plot_replenishment_decisions(
    rows_df,
    decision_df,
    unique_id=example_id,
    title="Mean forecast + safety stock (optimized)",
    decision_style="line",
)
```

## Example Workflows

### Mean Forecast + Safety Stock

Optimize a point-forecast safety stock factor on historical rows, then inspect
the resulting replenishment decisions on the forecast horizon.

![Mean forecast + safety stock example](docs/plots/mean_forecast_safety_stock.png)

### `k*RMSE`/`k*MAE` + Forecast-Level Buffering

Use `k*RMSE` (or `k*MAE`) for the base safety stock and optionally increase it when lead-time
forecast quantities rise above a baseline. The README plot below uses three
articles with progressively steeper ramps so the extra safety stock is visible
directly on the replenishment timeline.

![k*RMSE forecast buffering example](docs/plots/k_rmse_forecast_buffering.png)

### Percentile Target Optimization

Optimize the percentile target per article on backtest rows, then simulate and
visualize the chosen forecast target on the evaluation horizon.

![Percentile target optimization example](docs/plots/percentile_optimization.png)

## Notebooks

Six solver-free policies support independent item-level decisions and tuning:
`BaseStockPolicy`, `ConstantOrderPolicy`, `CappedBaseStockPolicy`, `sSPolicy`,
`ProjectedInventoryLevelPolicy`, and `ProtectionPeriodQuantilePolicy`.
Use `optimize_policy_candidates` for per-item cost-based candidate search and
`build_article_configs_from_standard_rows` for existing table inputs. Shared
budgets/capacities and solver-based planning are outside these APIs.
See [policy semantics and examples](docs/item_level_policies.md) for receipt
timing, lost-sales metrics, probabilistic assumptions and future PR boundaries.

Runnable walkthroughs live in [`notebooks/`](notebooks):

The nine solver-free examples follow the existing standard-data workflow:
generate rows/DataFrame, split chronologically, fit each article's policy, inspect
parameters, simulate, inspect decision rows, and call `plot_replenishment_decisions`
for example items and aggregate stock/missed-sales views.

- [`structured_neural_policy_example.ipynb`](notebooks/structured_neural_policy_example.ipynb): neural learning targeting missed sales/fill rate, exact service validation and safe JSON deployment
- [`hdpo_policy_example.ipynb`](notebooks/hdpo_policy_example.ipynb): differentiable inventory-cost training with separate exact integer validation
- [`risk_sensitive_policy_example.ipynb`](notebooks/risk_sensitive_policy_example.ipynb): mean/CVaR/worst-scenario candidate selection, standard-row evaluation and missed-sales plots

- [`base_stock_policy_example.ipynb`](notebooks/base_stock_policy_example.ipynb): static target, tuning and zero-lead-time verification
- [`constant_order_policy_example.ipynb`](notebooks/constant_order_policy_example.ipynb): fixed orders, long lead times and integer order rates
- [`capped_base_stock_policy_example.ipynb`](notebooks/capped_base_stock_policy_example.ipynb): joint target/cap tuning
- [`s_s_policy_example.ipynb`](notebooks/s_s_policy_example.ipynb): independent `(s,S)` thresholds and fixed ordering costs
- [`projected_inventory_level_policy_example.ipynb`](notebooks/projected_inventory_level_policy_example.ipynb): exact discrete projection and receipt-timing verification
- [`protection_period_quantile_policy_example.ipynb`](notebooks/protection_period_quantile_policy_example.ipynb): joint demand paths, dependence and aggregate coverage versus fill rate

- [`notebooks/mean_forecast_safety_stock_example.ipynb`](notebooks/mean_forecast_safety_stock_example.ipynb): mean forecast safety stock optimization
- [`notebooks/k_rmse_safety_stock_optimization_example.ipynb`](notebooks/k_rmse_safety_stock_optimization_example.ipynb): `k*RMSE`/`k*MAE` optimization and forecast-level buffering
- [`notebooks/percentile_optimization_example.ipynb`](notebooks/percentile_optimization_example.ipynb): percentile-target optimization
- [`notebooks/mean_forecast_policy_variants_example.ipynb`](notebooks/mean_forecast_policy_variants_example.ipynb): compare policy variants
- [`notebooks/generated_data_example.ipynb`](notebooks/generated_data_example.ipynb): synthetic data generation
- [`notebooks/stock_replenishment_example.ipynb`](notebooks/stock_replenishment_example.ipynb): table-oriented data loading workflow

Advanced learning is available through `pip install -e ".[advanced]"` (optional
PyTorch). Risk-based candidate search and saved neural-policy inference work
without Torch. See [advanced independent-item methods](docs/advanced_item_policies.md)
for APIs, training/deployment rounding, fixed-cost restrictions and risk assumptions.

## Solver-based planning

Install `pip install -e ".[solver]"` for the optional SciPy/HiGHS backend.
`DeterministicRollingHorizonPolicy` and `ScenarioRollingHorizonPolicy` plan integer
orders with dated receipts, fixed charges, MOQ and pack sizes. For shared per-period
budgets, order limits and receiving capacity, use the explicit
`simulate_coordinated_replenishment` loop. Scenario decisions share the same actions
until their observed demand histories diverge. See the
[formulation and API guide](docs/solver_policies.md) for timing and finite-horizon assumptions.

Executed examples include saved item/aggregate demand, stock and missed-sales plots:

- [Deterministic rolling horizon](notebooks/deterministic_rolling_horizon_policy_example.ipynb)
- [Scenario rolling horizon](notebooks/scenario_rolling_horizon_policy_example.ipynb)
- [Coordinated multi-item planning](notebooks/coordinated_solver_policy_example.ipynb), including shared-resource utilization
- [Maximize expected revenue](notebooks/revenue_maximization_policy_example.ipynb), preserving actual purchasing-budget constraints
- [Maximize expected profit](notebooks/profit_maximization_policy_example.ipynb), with matched revenue/profit comparisons and financial ledgers

## Development and Contribution

Contributions are welcome. If you open a PR, keep it easy to review:

- add or update tests for behavior changes
- update notebooks, README examples, and plot assets when public behavior changes
- include before/after plots when you change plotting behavior
- keep PRs focused rather than bundling unrelated refactors

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, test, and pull request guidance.

## Good Areas for Collaboration

If you want to make the project more useful and more credible to other teams,
these are high-leverage contribution areas:

- richer policy variants, such as minimum order quantities, case-pack constraints, and promotion-aware demand handling
- benchmark datasets and scenario packs for retail seasonality, long lead times, and stockout-heavy stress tests
- more notebook walkthroughs that compare service-level objectives and business tradeoffs
- packaging and docs polish, including API docs, changelogs, and release notes
- plotting improvements for aggregate dashboards and multi-SKU diagnostics

## Project Links

- Homepage: <https://github.com/janrth/replenishment>
- Issues: <https://github.com/janrth/replenishment/issues>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- License: [LICENSE](LICENSE)
