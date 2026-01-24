# replenishment

Stock replenishment simulation utilities.

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
