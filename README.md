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

## Notebook

See `notebooks/stock_replenishment_example.ipynb` for a runnable example.
