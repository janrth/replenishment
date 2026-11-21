# Revenue and Profit Notebook Implementation Plan

The user requested two created and executed notebooks after discussing revenue/profit
formulations. Reuse the approved solver architecture and standard notebook workflow.

- Add `cost_objective="total_cost"|"lost_sales"` to the planner, adapters and joint
  simulator. Default behavior is unchanged. Weighted lost sales excludes procurement
  and holding from the objective while retaining actual procurement-budget accounting.
- Verify by failure-first tests: a scarce budget chooses a high-revenue low-margin item
  for revenue and a lower-revenue higher-margin item for profit; fixed charges still
  consume budget; mixed lead times/risk and default behavior remain valid.
- Create `revenue_maximization_policy_example.ipynb` and
  `profit_maximization_policy_example.ipynb` using identical generated three-item data,
  training-only joint scenarios, mean objective, shared monetary/order/receipt limits,
  inspected plans, reconciled financial ledgers, and item/aggregate/resource/financial
  plots. Include a tiny exact case that demonstrates different optimal allocations.
- Revenue = selling price * fulfilled units. Profit = revenue - all placed procurement
  costs - holding. Initial stock is zero; no sunk initial stock ambiguity. No salvage,
  cash recycling or sales-funded cash balance. Demand/prices are exogenous; the exact
  equivalence is for mean objectives only. Plot and report real profit for both runs.
- Execute notebooks and embed outputs, update guide/README/changelog, run full tests
  and independent review. Keep changes uncommitted on the selected branch.
