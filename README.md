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

```python
from replenishment import ReorderPointPolicy, simulate_replenishment

policy = ReorderPointPolicy(reorder_point=20, order_quantity=50)
result = simulate_replenishment(
    periods=12,
    demand=[15, 10, 25, 30, 5, 12, 18, 22, 17, 19, 11, 9],
    initial_on_hand=40,
    lead_time=2,
    policy=policy,
)

print(result.summary)
```

```python
from replenishment import PointForecastOptimizationPolicy, simulate_replenishment

# Point-forecast optimization uses a safety stock buffer. The
# service_level_factor is used to optimize the safety stock, so the policy
# orders the point forecast plus the safety stock amount.

policy = PointForecastOptimizationPolicy(
    forecast=[18, 20, 22, 21, 19, 17],
    actuals=[16, 19, 24, 20, 18, 15],
    service_level_factor=0.95,
)
result = simulate_replenishment(
    periods=6,
    demand=[16, 19, 24, 20, 18, 15],
    initial_on_hand=30,
    lead_time=1,
    policy=policy,
)

print(result.summary)
```

```python
from replenishment import PercentileForecastOptimizationPolicy, simulate_replenishment

# Percentile-forecast optimization orders directly from the percentile target.
# No safety stock is used; the order quantity is the chosen percentile forecast
# for each period.

policy = PercentileForecastOptimizationPolicy(
    forecast=[18, 20, 22, 21, 19, 17],
    lead_time=1,
)
result = simulate_replenishment(
    periods=6,
    demand=[16, 19, 24, 20, 18, 15],
    initial_on_hand=30,
    lead_time=1,
    policy=policy,
)

print(result.summary)
```

```python
from replenishment import (
    ArticleSimulationConfig,
    ForecastCandidatesConfig,
    PointForecastOptimizationPolicy,
    optimize_aggregation_windows,
    optimize_forecast_targets,
    optimize_service_level_factors,
    simulate_replenishment_with_aggregation,
)

# Optimize point-forecast service levels (safety stock factors).
service_level_config = {
    "A": ArticleSimulationConfig(
        periods=6,
        demand=[16, 19, 24, 20, 18, 15],
        initial_on_hand=30,
        lead_time=1,
        policy=PointForecastOptimizationPolicy(
            forecast=[18, 20, 22, 21, 19, 17],
            actuals=[16, 19, 24, 20, 18, 15],
            service_level_factor=0.9,
        ),
    )
}
service_level_result = optimize_service_level_factors(
    service_level_config,
    candidate_factors=[0.8, 0.9, 0.95],
)

# Optimize percentile forecast targets.
percentile_config = ForecastCandidatesConfig(
    periods=6,
    demand=[16, 19, 24, 20, 18, 15],
    initial_on_hand=30,
    lead_time=1,
    forecast_candidates={
        "p50": [16, 18, 20, 19, 18, 16],
        "p90": [22, 24, 26, 25, 23, 21],
    },
)
percentile_result = optimize_forecast_targets({"A": percentile_config})

# Optimize order-cycle (time aggregation) windows. Use this when you want to
# decide how often to order while still aggregating demand and lead time.
aggregation_result = optimize_aggregation_windows(
    service_level_config,
    candidate_windows=[1, 2, 3],
)

# Or hard-code an aggregation window if the ordering cadence is fixed.
hard_coded_window = 2
aggregated = simulate_replenishment_with_aggregation(
    periods=6,
    demand=[16, 19, 24, 20, 18, 15],
    initial_on_hand=30,
    lead_time=1,
    policy=service_level_config["A"].policy,
    aggregation_window=hard_coded_window,
)
```

## Notebook

See `notebooks/stock_replenishment_example.ipynb` for a runnable example.
