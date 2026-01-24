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

## Notebook

See `notebooks/stock_replenishment_example.ipynb` for a runnable example.
