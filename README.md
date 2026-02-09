# replenishment

Stock replenishment simulation utilities.

## Environment (uv + Python 3.12)

This repo is set up to use Python 3.12 via `uv`.

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run checks with:

```bash
uv run pytest
```

## Usage
## Plot-first usage

This README focuses on two visual workflows:

- Mean forecast + safety stock optimization.
- Percentile target optimization.

## Example 1: mean forecast + safety stock

This example optimizes the safety stock factor on a backtest window, then
applies the learned policy to the evaluation horizon.

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

![Mean forecast + safety stock example](docs/plots/mean_forecast_safety_stock.png)

## Example 2: optimize percentile targets

This example learns the best percentile target on backtest rows and then plots
replenishment decisions on forecast rows.

```python
from replenishment import (
    PercentileForecastOptimizationPolicy,
    build_percentile_forecast_candidates_from_standard_rows,
    build_replenishment_decisions_from_simulations,
    generate_standard_simulation_rows,
    optimize_forecast_targets,
    plot_replenishment_decisions,
    replenishment_decision_rows_to_dataframe,
    simulate_replenishment_with_aggregation,
    split_standard_simulation_rows,
    standard_simulation_rows_to_dataframe,
)

rows = generate_standard_simulation_rows(
    n_unique_ids=1,
    periods=24,
    start_date="2031-01-01",
    frequency_days=30,
    forecast_start_period=14,
    history_mean=42,
    history_std=7,
    forecast_mean=40,
    forecast_std=5,
    lead_time=2,
    initial_on_hand=25,
    current_stock=25,
    seed=11,
    percentile_multipliers={"p50": 1.0, "p80": 1.2, "p95": 1.35},
)
backtest_rows, forecast_rows = split_standard_simulation_rows(rows)

percentile_configs = build_percentile_forecast_candidates_from_standard_rows(
    backtest_rows,
    include_mean=True,
    review_period=1,
    forecast_horizon=1,
)
optimized = optimize_forecast_targets(percentile_configs)

forecast_configs = build_percentile_forecast_candidates_from_standard_rows(
    forecast_rows,
    include_mean=True,
    review_period=1,
    forecast_horizon=1,
)
simulations = {}
for unique_id, config in forecast_configs.items():
    target = optimized[unique_id].target
    policy = PercentileForecastOptimizationPolicy(
        forecast=config.forecast_candidates[target],
        lead_time=config.lead_time,
    )
    simulations[unique_id] = simulate_replenishment_with_aggregation(
        periods=config.periods,
        demand=config.demand,
        initial_on_hand=config.initial_on_hand,
        lead_time=config.lead_time,
        policy=policy,
        aggregation_window=1,
        holding_cost_per_unit=config.holding_cost_per_unit,
        stockout_cost_per_unit=config.stockout_cost_per_unit,
        order_cost_per_order=config.order_cost_per_order,
        order_cost_per_unit=config.order_cost_per_unit,
    )

decision_rows = build_replenishment_decisions_from_simulations(
    forecast_rows,
    simulations,
    percentile_target={uid: optimized[uid].target for uid in simulations},
    review_period=1,
    forecast_horizon=1,
    rmse_window=1,
)
rows_df = standard_simulation_rows_to_dataframe(rows, library="pandas")
decision_df = replenishment_decision_rows_to_dataframe(decision_rows, library="pandas")

example_id = decision_df["unique_id"].iloc[0]
plot_replenishment_decisions(
    rows_df,
    decision_df,
    unique_id=example_id,
    title="Percentile forecast target (optimized)",
    decision_style="line",
)
```

![Percentile target optimization example](docs/plots/percentile_optimization.png)

## Notebooks

For full runnable walkthroughs (including additional variants):

- `notebooks/mean_forecast_safety_stock_example.ipynb`
- `notebooks/percentile_optimization_example.ipynb`
- `notebooks/generated_data_example.ipynb`

For data-loading and table-oriented flows (without plots), see:

- `notebooks/stock_replenishment_example.ipynb`

## Standard schema helpers

If you need CSV/DataFrame conversion helpers:

- `iter_standard_simulation_rows_from_csv`
- `standard_simulation_rows_from_dataframe`
- `build_point_forecast_article_configs_from_standard_rows`
